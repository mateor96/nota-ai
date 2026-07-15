import asyncio
import json
import logging
import mimetypes
import os
import tempfile
import time

import httpx

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .db import (
    AUDIO_DIR, delete_archive_entry, get_archive_entry, get_settings, init_db,
    list_archive, save_settings, save_summary, search_archive, update_filename,
    update_speaker_names,
)
from .export import to_json, to_markdown, to_srt, to_txt
from .pipeline import run_pipeline
from .services.exceptions import ProviderAuthError, ProviderError, ProviderModelError, ProviderUnavailableError
from .services.factory import CLOUD_MODELS, COMBINE_PROMPT, build_summary_prompt, format_transcript, get_summarizer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Uvicorn only configures its own loggers; give the app's loggers a
    # handler so pipeline/provider errors reach the terminal.
    logging.basicConfig(level=logging.INFO)
    await init_db()
    yield


app = FastAPI(title="Transcription App", lifespan=lifespan)

# In-memory job store: job_id → {status, queue, result, finished_at}.
# Finished jobs are kept briefly so the client can fetch /result, then
# evicted — the transcript is already persisted in the archive.
jobs: dict = {}
JOB_TTL_S = 3600

# Server-side cap; the UI advertises a 90-minute maximum, which fits well
# under this even at high bitrates.
MAX_UPLOAD_BYTES = 2 * 1024**3


def _purge_stale_jobs() -> None:
    now = time.monotonic()
    for job_id, job in list(jobs.items()):
        finished_at = job.get("finished_at")
        if finished_at is not None and now - finished_at > JOB_TTL_S:
            del jobs[job_id]


@app.post("/transcribe")
async def start_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Query(default=None),
    max_speakers: Optional[int] = Query(default=None),
):
    _purge_stale_jobs()
    job_id = str(uuid.uuid4())

    suffix = Path(file.filename).suffix or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    received = 0
    try:
        # Stream to disk in chunks — a 90-minute recording shouldn't be
        # buffered whole in RAM.
        while chunk := await file.read(1024 * 1024):
            received += len(chunk)
            if received > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp.name)
                return JSONResponse({"error": "file too large"}, status_code=413)
            tmp.write(chunk)
    finally:
        if not tmp.closed:
            tmp.close()

    jobs[job_id] = {"status": "queued", "queue": asyncio.Queue(), "result": None, "finished_at": None}
    background_tasks.add_task(
        run_pipeline, job_id, tmp.name, jobs, min_speakers, max_speakers, file.filename
    )

    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def stream():
        q = job["queue"]
        while True:
            event = await q.get()
            if event is None:
                break
            yield {"data": json.dumps(event)}

    return EventSourceResponse(stream())


@app.get("/result/{job_id}/{fmt}")
async def get_result(job_id: str, fmt: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "result not ready"}, status_code=404)

    segments = job["result"]
    return _format_response(segments, fmt)


# ── Archive ──────────────────────────────────────────────────────────────────

@app.get("/archive")
async def archive_list():
    return await list_archive()


@app.get("/archive/search")
async def archive_search(q: str = Query(default="")):
    return await search_archive(q)


@app.get("/archive/{entry_id}")
async def archive_get(entry_id: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return entry


# Browser-friendly MIME types — Python's defaults (e.g. audio/mp4a-latm for .m4a)
# aren't recognized by HTML5 audio.
_AUDIO_MIME = {
    ".m4a":  "audio/mp4",
    ".mp4":  "video/mp4",
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".webm": "audio/webm",
    ".ogg":  "audio/ogg",
    ".aac":  "audio/aac",
    ".flac": "audio/flac",
}


@app.get("/archive/{entry_id}/audio")
async def archive_audio(entry_id: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    ext = entry.get("audio_ext")
    if not ext:
        return JSONResponse({"error": "audio not stored"}, status_code=404)
    path = AUDIO_DIR / f"{entry_id}{ext}"
    if not path.exists():
        return JSONResponse({"error": "audio file missing"}, status_code=404)
    media_type = _AUDIO_MIME.get(ext.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/archive/{entry_id}/download/{fmt}")
async def archive_download(entry_id: str, fmt: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _format_response(entry["segments"], fmt)


@app.patch("/archive/{entry_id}/rename")
async def archive_rename(entry_id: str, data: dict):
    ok = await update_filename(entry_id, data.get("filename", ""))
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


@app.patch("/archive/{entry_id}/names")
async def archive_save_names(entry_id: str, names: dict):
    ok = await update_speaker_names(entry_id, names)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


@app.delete("/archive/{entry_id}")
async def archive_delete(entry_id: str):
    ok = await delete_archive_entry(entry_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"ok": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_response(segments: list, fmt: str):
    if fmt == "json":
        return JSONResponse(to_json(segments))
    if fmt == "txt":
        return PlainTextResponse(to_txt(segments))
    if fmt == "srt":
        return PlainTextResponse(to_srt(segments))
    if fmt == "md":
        return PlainTextResponse(to_markdown(segments))
    return JSONResponse({"error": f"unknown format: {fmt}"}, status_code=400)


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/settings")
async def api_get_settings():
    cfg = await get_settings()
    return {
        "provider":      cfg["provider"],
        "base_url":      cfg["base_url"],
        "model":         cfg["model"],
        "api_key_set":   bool(cfg["api_key"]),
        "prompt_style":  cfg.get("prompt_style", "meeting"),
        "custom_prompt": cfg.get("custom_prompt", ""),
    }


@app.post("/settings")
async def api_save_settings(body: dict):
    allowed = {"provider", "base_url", "model", "api_key", "prompt_style", "custom_prompt"}
    b = {k: str(v) for k, v in body.items() if k in allowed}
    # Preserve fields the client omitted so partial updates don't wipe other settings.
    saved = await get_settings()
    await save_settings(
        provider=b.get("provider",      saved["provider"]),
        base_url=b.get("base_url",      saved["base_url"]),
        model=b.get("model",            saved["model"]),
        api_key=b.get("api_key",        saved["api_key"]),
        prompt_style=b.get("prompt_style", saved.get("prompt_style", "meeting")),
        custom_prompt=b.get("custom_prompt", saved.get("custom_prompt", "")),
    )
    return {"ok": True}


@app.post("/settings/test")
async def api_test_provider(body: dict):
    saved = await get_settings()
    cfg = {
        "provider": body.get("provider") or saved["provider"],
        "base_url": body.get("base_url") or saved["base_url"],
        "model":    body.get("model") or saved["model"],
        # Use typed key if provided, otherwise fall back to saved key
        "api_key":  body.get("api_key") or saved["api_key"],
    }
    try:
        svc = get_summarizer(cfg)
        ok  = await svc.check_health()
        return {"ok": ok, "message": "Connected" if ok else "Not reachable — check URL or model name"}
    except ProviderAuthError as e:
        return {"ok": False, "message": str(e)}
    except ProviderUnavailableError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def _is_embedding_model(model_id: str) -> bool:
    """Embedding-only models can't do chat completions; filter them out of the picker."""
    lower = model_id.lower()
    return "embed" in lower or "embedding" in lower


@app.post("/models")
async def list_models(body: dict):
    provider = body.get("provider", "")
    base_url = (body.get("base_url") or "").rstrip("/")

    if provider in CLOUD_MODELS:
        return {"models": CLOUD_MODELS[provider]}

    if provider == "lmstudio":
        url = base_url or "http://localhost:1234"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                ids = [m["id"] for m in data.get("data", []) if "id" in m]
                return {"models": [m for m in ids if not _is_embedding_model(m)]}
        except Exception as e:
            return {"models": [], "error": f"Cannot reach LM Studio at {url}: {e}"}

    if provider == "ollama":
        url = base_url or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                names = [m["name"] for m in data.get("models", []) if "name" in m]
                return {"models": [n for n in names if not _is_embedding_model(n)]}
        except Exception as e:
            return {"models": [], "error": f"Cannot reach Ollama at {url}: {e}"}

    return {"models": []}


# ── Summarize ─────────────────────────────────────────────────────────────────

# Roughly: most local models comfortably handle ~30k characters per pass; cloud
# models handle much more, but chunking keeps quality consistent and avoids
# silent context-window failures on the small local side.
MAX_CHARS_PER_CHUNK = 30_000


def _chunk_transcript(text: str, max_chars: int) -> list[str]:
    """Split on line boundaries so we never bisect a single speaker turn."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        added = len(line) + 1
        if current and current_len + added > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = added
        else:
            current.append(line)
            current_len += added
    if current:
        chunks.append("\n".join(current))
    return chunks


@app.get("/summarize/{entry_id}")
async def summarize_entry(entry_id: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)

    cfg = await get_settings()
    if not cfg.get("provider"):
        return JSONResponse({"error": "no_provider", "message": "No AI provider configured."}, status_code=400)

    try:
        svc = get_summarizer(cfg)
    except (ProviderAuthError, ProviderUnavailableError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    text          = format_transcript(entry["segments"], entry.get("speaker_names") or {})
    style         = cfg.get("prompt_style") or "meeting"
    custom_prompt = cfg.get("custom_prompt") or ""
    chunks        = _chunk_transcript(text, MAX_CHARS_PER_CHUNK)

    async def stream():
        full: list[str] = []
        try:
            if len(chunks) == 1:
                # Single-pass — stream the model output straight to the client.
                prompt = build_summary_prompt(style, chunks[0], custom_prompt)
                async for token in svc.stream_chat(prompt):
                    full.append(token)
                    yield {"data": json.dumps({"type": "token", "text": token})}
            else:
                # Map-reduce: summarize each chunk, then ask the model to merge
                # the section summaries into one cohesive answer. Section work
                # is reported as stage events so the user sees progress; only
                # the final combine pass streams as tokens.
                yield {"data": json.dumps({
                    "type": "stage",
                    "message": f"Long transcript — summarizing in {len(chunks)} sections…",
                })}
                partials: list[str] = []
                for i, chunk in enumerate(chunks, 1):
                    yield {"data": json.dumps({
                        "type": "stage",
                        "message": f"Section {i} of {len(chunks)}…",
                    })}
                    chunk_prompt = build_summary_prompt(style, chunk, custom_prompt)
                    partial: list[str] = []
                    async for token in svc.stream_chat(chunk_prompt):
                        partial.append(token)
                    partials.append("".join(partial))

                yield {"data": json.dumps({"type": "stage", "message": "Combining sections…"})}
                sections_block = "\n\n".join(
                    f"Section {i+1}:\n{p}" for i, p in enumerate(partials)
                )
                combine_prompt = COMBINE_PROMPT.format(sections=sections_block)
                async for token in svc.stream_chat(combine_prompt):
                    full.append(token)
                    yield {"data": json.dumps({"type": "token", "text": token})}

            summary = "".join(full)
            await save_summary(entry_id, summary)
            yield {"data": json.dumps({"type": "done", "summary": summary})}
        except ProviderError as e:
            logger.warning("Summarization failed for %s via %s: %s", entry_id, e.provider, e)
            yield {"data": json.dumps({"type": "error", "message": str(e)})}
        except Exception as e:
            logger.exception("Unexpected error summarizing %s", entry_id)
            yield {"data": json.dumps({"type": "error", "message": f"Unexpected error: {e}"})}

    return EventSourceResponse(stream())


@app.get("/summary/{entry_id}")
async def get_entry_summary(entry_id: str):
    entry = await get_archive_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"summary": entry.get("summary")}


# Static files last so they don't shadow the API routes.
# Resolve relative to this file so uvicorn works no matter where it's launched from.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
