"""Voice Activity Detection — used only by the eval harness (`app.eval --vad`).

An earlier pipeline design gated transcription on these speech regions, but
per-window decoding degraded Whisper's accuracy and was reverted (see the
module docstring in app/transcribe.py). VAD remains useful as a *measurement*:
eval.py compares word timestamps against detected speech regions to flag words
transcribed over silence (hallucinations) and speech that produced no words
(dropped audio).

Uses pyannote/segmentation-3.0, the same model family already pulled in for
diarization, so no extra download.
"""
from __future__ import annotations

import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

_vad: VoiceActivityDetection | None = None

# Tuned for ASR gating, not maximal precision:
#  - keep short backchannels ("ja", "yeah") so we don't drop real speech
#  - bridge small gaps so a sentence isn't shredded into fragments (which would
#    hurt Whisper's per-window context and split words at boundaries)
_MIN_DURATION_ON = 0.25
_MIN_DURATION_OFF = 0.50


def _get_vad() -> VoiceActivityDetection:
    global _vad
    if _vad is None:
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
