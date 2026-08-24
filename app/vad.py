"""Voice Activity Detection — pyannote/segmentation-3.0.

Two consumers:

1. app/pipeline.py runs this over the normalized WAV and hands the regions to
   app/transcribe.py. Transcription uses them ONLY to (a) place the start and
   end of the single continuous decode and (b) tell a silence-seeded
   hallucination loop apart from audio that genuinely repeats. The interior of
   the audio is never sliced and every word still comes out of one continuous
   `condition_on_previous_text=True` pass — the reverted per-window design (see
   the module docstring in app/transcribe.py) is NOT coming back.

2. app/eval.py compares word timestamps against these regions to flag words
   transcribed over silence and speech that produced no words.

Uses the same model family already pulled in for diarization, so no extra
download.
"""
from __future__ import annotations

import threading

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

_vad: VoiceActivityDetection | None = None
# The pipeline calls detect_speech from a shared thread pool with no per-job
# concurrency limit, so two jobs can race the lazy init and load the model
# twice (wasted memory, duplicated MPS allocation).
_vad_lock = threading.Lock()

# Tuned for ASR gating, not maximal precision:
#  - keep short backchannels ("ja", "yeah") so we don't drop real speech
#  - bridge small gaps so a sentence isn't shredded into fragments (which would
#    hurt Whisper's per-window context and split words at boundaries)
_MIN_DURATION_ON = 0.25
_MIN_DURATION_OFF = 0.50


def _get_vad() -> VoiceActivityDetection:
    global _vad
    with _vad_lock:
        if _vad is not None:
            return _vad
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate(
            {"min_duration_on": _MIN_DURATION_ON, "min_duration_off": _MIN_DURATION_OFF}
        )
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        _vad = pipeline
        return _vad


def detect_speech(audio_path: str) -> list[tuple[float, float]]:
    """Return merged (start, end) speech regions in seconds, in order."""
    annotation = _get_vad()(audio_path)
    return [(seg.start, seg.end) for seg in annotation.get_timeline().support()]
