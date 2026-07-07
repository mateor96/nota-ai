"""Whole-file mlx-whisper transcription.

An earlier version sliced the audio into VAD speech windows and transcribed each
window in isolation, to stop Whisper hallucinating filler over silence. It fixed
the silence case but broke normal speech: each window was decoded without the
surrounding context (`condition_on_previous_text=False`), so Whisper lost the
thread and mis-heard words a continuous pass gets right ("Reviews"->"Videos",
"Human"->"Jury"), and sentences were shredded at window boundaries.

Whisper's coherence comes from decoding the audio continuously and feeding each
segment's decoded text as the prompt for the next — so we hand it the whole file
in one pass, exactly like the original version that transcribed well. Silence is
handled without fragmenting context: `hallucination_silence_threshold` lets
Whisper skip long silent stretches during the continuous decode, and a light
per-segment confidence filter drops any residual noise-hallucination afterwards.
"""
from __future__ import annotations

from typing import Callable, Optional

import mlx_whisper
from mlx_whisper.audio import SAMPLE_RATE, load_audio

# large-v3-turbo. We tried the full large-v3 MLX build for higher accuracy, but
# that conversion is unreliable on real recordings — on some audio it collapses
# into lowercase, unpunctuated, repeated-word garbage ("die die die videos...")
# where turbo stays clean and correct. turbo is the consistent choice; the text
# quality comes from decoding the whole file continuously (see below), not the model.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Per-segment confidence gate — drops residual hallucination on noise/silence.
# Conservative so real (quiet) speech is kept.
_MAX_NO_SPEECH_PROB = 0.85
_MAX_COMPRESSION_RATIO = 2.4   # high ratio == repetitive gibberish

# Decode params. condition_on_previous_text=True (Whisper's default) is what
# carries context across the file and keeps transcription coherent — the earlier
# code disabled it and that was the main quality regression.
# hallucination_silence_threshold skips long silence during the continuous decode
# so we get the anti-hallucination benefit without shattering context into windows.
_DECODE = dict(
    word_timestamps=True,
    condition_on_previous_text=True,
    hallucination_silence_threshold=2.0,
)


# Language is detected from a window in the MIDDLE of the recording, not the
# first 30s that mlx-whisper would use by default. Openings are often greetings
# ("Hi", "hey", "good morning") that skew auto-detect to the wrong language — and
# with language unset for the whole-file pass, one wrong guess makes Whisper
# transcribe (really: translate) the entire German recording into English.
_LANG_PROBE_S = 30.0


def _detect_language(audio_path: str, model: str) -> Optional[str]:
    audio = load_audio(audio_path)
    half = int(_LANG_PROBE_S * SAMPLE_RATE / 2)
    mid = len(audio) // 2
    clip = audio[max(0, mid - half): mid + half]
    if len(clip) == 0:
        return None
    result = mlx_whisper.transcribe(
        clip, path_or_hf_repo=model, word_timestamps=False,
        condition_on_previous_text=False,
    )
    return result.get("language")


def _keep_segment(seg: dict) -> bool:
    if seg.get("no_speech_prob", 0.0) > _MAX_NO_SPEECH_PROB:
        return False
    if seg.get("compression_ratio", 0.0) > _MAX_COMPRESSION_RATIO:
        return False
    return bool(seg.get("text", "").strip())


def transcribe(
    audio_path: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> dict:
    """Transcribe the whole file in one continuous pass and return kept segments."""
    if language is None:
        language = _detect_language(audio_path, model)

    common = dict(_DECODE)
    if initial_prompt:
        common["initial_prompt"] = initial_prompt

    result = mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=model, language=language, **common
    )

    segments = [seg for seg in result.get("segments", []) if _keep_segment(seg)]
    if progress_cb:
        progress_cb(1.0)
    return {"segments": segments, "language": result.get("language", language)}
