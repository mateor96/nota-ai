import asyncio
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import ffmpeg

from .transcribe import transcribe
from .diarize import diarize
from .vad import detect_speech
from .merge import merge
from .db import AUDIO_DIR, save_transcription

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

# 16 kHz mono s16 == 32000 bytes/s. On a sub-minute clip a pyannote model load
# can cost more than the whole decode, and the damage VAD guards against is
# bounded by the file: a loop cannot overwrite minutes of content there. Both
# layers WOULD work below this threshold (a 59 s file can carry 30 s of dead
# air and a 20 s run), so this is a wall-clock trade, not a capability limit —
# raise or lower it on that basis.
_MIN_VAD_SECONDS = 60.0
_VAD_BYTES_PER_SECOND = 16000 * 2


def _speech_regions(wav_path: str) -> list[tuple[float, float]] | None:
    """VAD regions for the Whisper input, or None. Never fails a job."""
    try:
        if os.path.getsize(wav_path) < _MIN_VAD_SECONDS * _VAD_BYTES_PER_SECOND:
            return None
        return detect_speech(wav_path)
    except Exception:
        logger.warning("VAD failed for %s - decoding the whole file",
                       wav_path, exc_info=True)
        return None


async def run_pipeline(
    job_id: str,
    audio_path: str,
    jobs: dict,
    min_speakers: int | None,
    max_speakers: int | None,
    filename: str = "audio",
) -> None:
    job = jobs[job_id]
    loop = asyncio.get_running_loop()

    async def emit(event: dict) -> None:
        await job["queue"].put(event)

    wav_path = audio_path + ".wav"            # loudnorm'd — for Whisper
    diar_wav_path = audio_path + ".diar.wav"  # un-normalized — for diarization

    try:
        job["status"] = "processing"

        # Normalize to 16 kHz mono WAV, with loudness normalization. Quiet
        # recordings (e.g. a call captured at a low input level) sit near the
        # noise floor, where Whisper hears "silence" and hallucinates filler
        # like "Thank you" / "Yeah" on a loop. loudnorm (EBU R128) lifts quiet
        # speech to a consistent level and leaves already-normal audio intact.
        #
        # Diarization gets a SEPARATE, un-normalized 16 kHz mono WAV. loudnorm's
        # time-varying gain compresses the level differences between speakers and
        # pumps up the noise floor / cross-talk between turns, which degrades
        # pyannote's speaker embeddings and can swap or merge speaker labels.
        # Both files share the same sample count (loudnorm changes amplitude, not
        # timing), so word and turn timestamps stay on one absolute axis for merge.
        await emit({"stage": "normalizing", "pct": 5, "message": "Normalizing audio..."})

        def _prepare_audio() -> None:
            ffmpeg.input(audio_path).output(
                wav_path, ar=16000, ac=1, af="loudnorm=I=-16:TP=-1.5:LRA=11"
            ).run(quiet=True, overwrite_output=True)
            ffmpeg.input(audio_path).output(
                diar_wav_path, ar=16000, ac=1
            ).run(quiet=True, overwrite_output=True)

        await loop.run_in_executor(_executor, _prepare_audio)

        # VAD runs on the loudnorm'd WAV, not the diarization WAV, deliberately:
        # it is exactly the audio Whisper decodes, so "no speech before t" means
        # "Whisper has nothing to transcribe before t"; and loudnorm has already
        # boosted genuinely quiet speech before pyannote judges it, which biases
        # the one dangerous direction (trimming real words) correctly. The
        # boosted-noise case is still detected: pyannote finds zero speech in the
        # silent intro even after a +25 dB lift. Both WAVs share a sample count
        # and one absolute time axis, so this choice is free w.r.t. merge.
        await emit({"stage": "processing", "pct": 8, "message": "Detecting speech..."})

        # Diarization is scheduled FIRST so it overlaps the VAD pass and the
        # decode. Peak concurrent executor tasks per job stays at 2.
        diarize_future = loop.run_in_executor(
            _executor, diarize, diar_wav_path, min_speakers, max_speakers)
        regions = await loop.run_in_executor(_executor, _speech_regions, wav_path)

        await emit({"stage": "processing", "pct": 10, "message": "Transcribing..."})

        # The single-pass Whisper decode exposes no fine-grained progress, so
        # the bar holds at 10% during transcription and jumps to 80% once only
        # diarization (running concurrently) remains.
        transcribe_future = loop.run_in_executor(
            _executor, partial(transcribe, wav_path, speech_regions=regions))

        whisper_result = await transcribe_future
        await emit({"stage": "processing", "pct": 80, "message": "Identifying speakers..."})
        diarization_turns = await diarize_future

        await emit({"stage": "merging", "pct": 90, "message": "Merging transcription and speakers..."})
        segments = await loop.run_in_executor(_executor, merge, whisper_result, diarization_turns)

        job["result"] = segments
        job["status"] = "done"

        # Persist the original audio alongside the transcript so click-to-seek
        # works on archived entries.
        audio_ext = Path(filename).suffix or Path(audio_path).suffix or ".audio"
        persisted_audio = AUDIO_DIR / f"{job_id}{audio_ext}"
        try:
            shutil.move(audio_path, persisted_audio)
        except OSError:
            audio_ext = None  # move failed — record entry without audio

        await save_transcription(job_id, filename, segments, audio_ext)
        await emit({"stage": "done", "pct": 100, "message": "Done!"})

    except Exception as exc:
        logger.exception("Pipeline failed for job %s (%s)", job_id, filename)
        job["status"] = "error"
        await emit({"stage": "error", "message": str(exc)})

    finally:
        job["finished_at"] = time.monotonic()
        # WAVs are always temporary; audio_path may have been moved to AUDIO_DIR.
        for tmp_wav in (wav_path, diar_wav_path):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        await job["queue"].put(None)  # sentinel — tells SSE stream to close
