"""Tests for app.pipeline — audio persistence, cleanup, and DB record on success/failure.

The heavy collaborators (ffmpeg, whisper, pyannote) are stubbed so the test
runs in milliseconds. The test focuses on the file-management contract:
- Source audio is MOVED to AUDIO_DIR on success (not deleted)
- WAV intermediate is always deleted
- save_transcription is called with the resolved audio_ext
- On pipeline error, the source audio is deleted and no DB row is written

It also pins the VAD contract added for the silence-hallucination fix: VAD
regions reach transcribe(), VAD is skipped on short audio, and a VAD failure
is never load-bearing (see app/transcribe.py's module docstring).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app import db as db_module
from app import pipeline as pipeline_module
from app.pipeline import run_pipeline


@pytest.fixture
def make_audio(tmp_path):
    """Factory creating a temp 'audio' file at a given path, returns the path."""
    def _factory(name: str = "input.mp3") -> str:
        path = tmp_path / name
        path.write_bytes(b"fake audio bytes")
        return str(path)
    return _factory


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub ffmpeg, transcribe, diarize, merge with deterministic fakes."""

    # ffmpeg.input → .output → .run(...). We don't care about ffmpeg behaviour,
    # just that the WAV file exists when the rest of the pipeline runs.
    class _FfmpegChain:
        def __init__(self, dst_path):
            self._dst = dst_path
        def output(self, dst, **kw):
            return _FfmpegChain(dst)
        def run(self, quiet=True, overwrite_output=True):
            Path(self._dst).write_bytes(b"wav bytes")

    monkeypatch.setattr(pipeline_module.ffmpeg, "input", lambda src: _FfmpegChain(src))

    monkeypatch.setattr(pipeline_module, "detect_speech", lambda wav: [(0.0, 1.0)])
    monkeypatch.setattr(pipeline_module, "transcribe",
                        lambda wav, speech_regions=None: {"segments": []})
    monkeypatch.setattr(pipeline_module, "diarize",    lambda wav, mn, mx: [])
    monkeypatch.setattr(pipeline_module, "merge",      lambda w, d: [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi", "words": []},
    ])


async def _collect_events(job: dict) -> list:
    events = []
    while True:
        e = await job["queue"].get()
        if e is None:
            break
        events.append(e)
    return events


# ── Happy path ───────────────────────────────────────────────────────────────

async def test_pipeline_persists_audio_to_audio_dir(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="meeting.mp3")
    await _collect_events(jobs["job1"])

    persisted = db_module.AUDIO_DIR / "job1.mp3"
    assert persisted.exists()
    assert persisted.read_bytes() == b"fake audio bytes"


async def test_pipeline_deletes_wav_intermediate(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    wav_path = src + ".wav"
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    await _collect_events(jobs["job1"])

    assert not os.path.exists(wav_path), "WAV intermediate should be cleaned up"


async def test_pipeline_writes_db_record_with_audio_ext(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="meeting.mp3")
    await _collect_events(jobs["job1"])

    entry = await db_module.get_archive_entry("job1")
    assert entry is not None
    assert entry["audio_ext"] == ".mp3"
    assert entry["filename"]  == "meeting.mp3"


async def test_pipeline_marks_job_as_done(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    await _collect_events(jobs["job1"])

    assert jobs["job1"]["status"] == "done"
    assert jobs["job1"]["result"] is not None


async def test_pipeline_emits_done_event_with_pct_100(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    events = await _collect_events(jobs["job1"])
    done = [e for e in events if e.get("stage") == "done"]
    assert done and done[-1]["pct"] == 100


async def test_pipeline_audio_ext_from_recorded_filename(initialized_db, stub_pipeline, make_audio):
    """Live-recorded files end with .webm; the extension should be preserved."""
    src = make_audio("input.webm")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="Recording_2026.webm")
    await _collect_events(jobs["job1"])

    entry = await db_module.get_archive_entry("job1")
    assert entry["audio_ext"] == ".webm"
    assert (db_module.AUDIO_DIR / "job1.webm").exists()


# ── Error path ───────────────────────────────────────────────────────────────

async def test_pipeline_failure_deletes_source_and_skips_db_write(initialized_db, stub_pipeline, make_audio, monkeypatch):
    src = make_audio("input.mp3")
    monkeypatch.setattr(pipeline_module, "transcribe",
                        lambda wav, speech_regions=None: (_ for _ in ()).throw(RuntimeError("MLX exploded")))

    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}
    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    events = await _collect_events(jobs["job1"])

    # No persisted audio
    assert not (db_module.AUDIO_DIR / "job1.mp3").exists()
    # Source is gone too
    assert not os.path.exists(src)
    # No DB row
    assert await db_module.get_archive_entry("job1") is None
    # Error event emitted
    assert any(e.get("stage") == "error" for e in events)
    assert jobs["job1"]["status"] == "error"


# ── VAD wiring (see app/transcribe.py — Layer 1 / Layer 2) ───────────────────

@pytest.fixture
def capture_transcribe(monkeypatch):
    """Record the kwargs transcribe() is called with."""
    calls = []

    def _transcribe(wav, speech_regions=None):
        calls.append({"wav": wav, "speech_regions": speech_regions})
        return {"segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe", _transcribe)
    return calls


async def test_pipeline_passes_speech_regions_to_transcribe(
    initialized_db, stub_pipeline, make_audio, capture_transcribe, monkeypatch
):
    monkeypatch.setattr(pipeline_module, "_speech_regions", lambda wav: [(1.0, 2.0)])
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    await _collect_events(jobs["job1"])

    assert capture_transcribe[0]["speech_regions"] == [(1.0, 2.0)]


async def test_pipeline_survives_vad_failure(
    initialized_db, stub_pipeline, make_audio, capture_transcribe, monkeypatch
):
    """VAD must never be load-bearing: a broken VAD degrades to a full decode."""
    def _raiser(wav):
        raise RuntimeError("pyannote exploded")

    monkeypatch.setattr(pipeline_module, "_MIN_VAD_SECONDS", 0.0)
    monkeypatch.setattr(pipeline_module, "detect_speech", _raiser)
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    events = await _collect_events(jobs["job1"])

    assert jobs["job1"]["status"] == "done"
    done = [e for e in events if e.get("stage") == "done"]
    assert done and done[-1]["pct"] == 100
    assert capture_transcribe[0]["speech_regions"] is None


async def test_pipeline_skips_vad_for_short_audio(
    initialized_db, stub_pipeline, make_audio, capture_transcribe, monkeypatch
):
    """A 20 s voice memo must not wait on a pyannote model load."""
    called = []
    monkeypatch.setattr(pipeline_module, "detect_speech",
                        lambda wav: called.append(wav) or [])
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    await _collect_events(jobs["job1"])

    assert called == []
    assert capture_transcribe[0]["speech_regions"] is None


async def test_pipeline_emits_detecting_speech_stage(initialized_db, stub_pipeline, make_audio):
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    events = await _collect_events(jobs["job1"])

    messages = [e.get("message") for e in events]
    assert "Detecting speech..." in messages
    assert messages.index("Detecting speech...") < messages.index("Transcribing...")
    vad_event = next(e for e in events if e.get("message") == "Detecting speech...")
    assert vad_event["pct"] == 8


async def test_pipeline_still_feeds_diarize_the_unnormalized_wav(
    initialized_db, stub_pipeline, make_audio, capture_transcribe, monkeypatch
):
    """Two WAVs, one absolute time axis — the clip must not disturb this."""
    diarized = []
    monkeypatch.setattr(pipeline_module, "diarize",
                        lambda wav, mn, mx: diarized.append(wav) or [])
    src = make_audio("input.mp3")
    jobs = {"job1": {"status": "queued", "queue": asyncio.Queue(), "result": None}}

    await run_pipeline("job1", src, jobs, None, None, filename="x.mp3")
    await _collect_events(jobs["job1"])

    assert diarized == [src + ".diar.wav"]
    assert capture_transcribe[0]["wav"] == src + ".wav"
