"""Tests for app.transcribe pure helpers: the per-segment confidence filter.
The model call itself (whole-file continuous decode) is exercised end-to-end,
not here."""
from __future__ import annotations

from app.transcribe import _keep_segment


def test_keep_segment_drops_silence():
    assert not _keep_segment({"text": "hallo", "no_speech_prob": 0.9})


def test_keep_segment_drops_repetitive():
    assert not _keep_segment({"text": "x" * 80, "compression_ratio": 3.0})


def test_keep_segment_drops_empty_text():
    assert not _keep_segment({"text": "   ", "no_speech_prob": 0.0})


def test_keep_segment_keeps_normal_speech():
    assert _keep_segment({"text": "hallo welt", "no_speech_prob": 0.1, "compression_ratio": 1.4})
