"""Tests for the live transcription flow: POST /transcribe → SSE /progress →
GET /result, plus upload limits and job-store eviction.

The heavy pipeline is stubbed at the module level (main.run_pipeline is the
name the endpoint schedules), so these tests exercise the HTTP surface, the
job store, and the SSE stream — not Whisper/pyannote themselves.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from app import main as main_module

from .conftest import SAMPLE_SEGMENTS


@pytest.fixture(autouse=True)
def _clean_jobs():
    """The job store is module-global — isolate it per test."""
    main_module.jobs.clear()
    yield
    main_module.jobs.clear()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace run_pipeline with a stub that completes the job immediately."""
    calls = []

    async def fake_pipeline(job_id, audio_path, jobs, min_speakers, max_speakers, filename="audio"):
        calls.append({
            "job_id": job_id,
            "audio_path": audio_path,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "filename": filename,
        })
        job = jobs[job_id]
        job["status"] = "done"
        job["result"] = SAMPLE_SEGMENTS
        job["finished_at"] = time.monotonic()
        await job["queue"].put({"stage": "done", "pct": 100, "message": "Done!"})
        await job["queue"].put(None)  # sentinel — closes the SSE stream

    monkeypatch.setattr(main_module, "run_pipeline", fake_pipeline)
    return calls


def _upload(client, content: bytes = b"fake-audio", filename: str = "meeting.mp3", query: str = ""):
    return client.post(
        f"/transcribe{query}",
        files={"file": (filename, content, "audio/mpeg")},
    )


# ── POST /transcribe ──────────────────────────────────────────────────────────

class TestStartTranscribe:
    async def test_returns_job_id_and_registers_job(self, client, stub_pipeline):
        r = _upload(client)
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert job_id in main_module.jobs

    async def test_pipeline_receives_upload_and_params(self, client, stub_pipeline):
        r = _upload(client, content=b"bytes-on-disk", query="?min_speakers=2&max_speakers=4")
        assert r.status_code == 200
        # TestClient runs background tasks before returning the response.
        assert len(stub_pipeline) == 1
        call = stub_pipeline[0]
        assert call["filename"] == "meeting.mp3"
        assert call["min_speakers"] == 2
        assert call["max_speakers"] == 4
        assert Path(call["audio_path"]).suffix == ".mp3"
        assert Path(call["audio_path"]).read_bytes() == b"bytes-on-disk"
        Path(call["audio_path"]).unlink()

    async def test_oversized_upload_rejected_with_413(self, client, stub_pipeline, monkeypatch):
        monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 8)
        r = _upload(client, content=b"way more than eight bytes")
        assert r.status_code == 413
        assert main_module.jobs == {}
        assert stub_pipeline == []


# ── GET /progress/{id} (SSE) ─────────────────────────────────────────────────

class TestProgress:
    async def test_unknown_job_returns_404(self, client):
        assert client.get("/progress/nope").status_code == 404

    async def test_streams_queued_events_until_sentinel(self, client):
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait({"stage": "processing", "pct": 10, "message": "Transcribing..."})
        q.put_nowait({"stage": "done", "pct": 100, "message": "Done!"})
        q.put_nowait(None)
        main_module.jobs["j1"] = {"status": "processing", "queue": q, "result": None, "finished_at": None}

        r = client.get("/progress/j1")
        assert r.status_code == 200
        payloads = [
            json.loads(line[len("data: "):])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [p["stage"] for p in payloads] == ["processing", "done"]
        assert payloads[-1]["pct"] == 100


# ── GET /result/{id}/{fmt} ───────────────────────────────────────────────────

class TestResult:
    def _done_job(self, job_id: str = "j1") -> None:
        main_module.jobs[job_id] = {
            "status": "done", "queue": asyncio.Queue(),
            "result": SAMPLE_SEGMENTS, "finished_at": time.monotonic(),
        }

    async def test_unknown_job_returns_404(self, client):
        assert client.get("/result/nope/json").status_code == 404

    async def test_unfinished_job_returns_404(self, client):
        main_module.jobs["j1"] = {"status": "processing", "queue": asyncio.Queue(), "result": None, "finished_at": None}
        assert client.get("/result/j1/json").status_code == 404

    async def test_json_result(self, client):
        self._done_job()
        body = client.get("/result/j1/json").json()
        assert "speakers" in body and "segments" in body

    async def test_txt_result(self, client):
        self._done_job()
        r = client.get("/result/j1/txt")
        assert r.status_code == 200
        assert "SPEAKER_00" in r.text

    async def test_unknown_format_returns_400(self, client):
        self._done_job()
        assert client.get("/result/j1/docx").status_code == 400


# ── Job eviction ─────────────────────────────────────────────────────────────

class TestJobEviction:
    async def test_stale_finished_jobs_purged_on_new_upload(self, client, stub_pipeline):
        stale = time.monotonic() - main_module.JOB_TTL_S - 1
        main_module.jobs["old"] = {
            "status": "done", "queue": asyncio.Queue(),
            "result": SAMPLE_SEGMENTS, "finished_at": stale,
        }
        _upload(client)
        assert "old" not in main_module.jobs

    async def test_running_and_fresh_jobs_survive_purge(self, client, stub_pipeline):
        main_module.jobs["running"] = {
            "status": "processing", "queue": asyncio.Queue(),
            "result": None, "finished_at": None,
        }
        main_module.jobs["fresh"] = {
            "status": "done", "queue": asyncio.Queue(),
            "result": SAMPLE_SEGMENTS, "finished_at": time.monotonic(),
        }
        _upload(client)
        assert "running" in main_module.jobs
        assert "fresh" in main_module.jobs
