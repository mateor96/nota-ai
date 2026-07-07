"""Tests for app.diarize._drop_phantom_speakers — the phantom-speaker filter
that fixes auto speaker over-counting without forcing a speaker count."""
from __future__ import annotations

from app.diarize import _drop_phantom_speakers


def test_folds_small_share_speaker_into_nearest():
    # The real failure: one dominant speaker, one real second speaker, and a
    # tiny phantom cluster (a few seconds). The phantom must be reassigned.
    turns = [
        {"start": 0.0,    "end": 900.0,  "speaker": "A"},        # 900s dominant
        {"start": 900.0,  "end": 905.0,  "speaker": "PHANTOM"},  # 5s (<5%)
        {"start": 905.0,  "end": 1120.0, "speaker": "B"},        # 215s (~19%)
    ]
    out = _drop_phantom_speakers(turns)
    speakers = {t["speaker"] for t in out}
    assert speakers == {"A", "B"}
    assert len(out) == len(turns)            # turns kept, only relabeled
    assert "PHANTOM" not in speakers


def test_noop_when_all_speakers_substantial():
    turns = [
        {"start": 0.0,   "end": 100.0, "speaker": "A"},
        {"start": 100.0, "end": 200.0, "speaker": "B"},
    ]
    assert _drop_phantom_speakers(turns) == turns


def test_noop_for_single_speaker():
    turns = [
        {"start": 0.0,  "end": 10.0, "speaker": "A"},
        {"start": 10.0, "end": 12.0, "speaker": "A"},
    ]
    assert _drop_phantom_speakers(turns) == turns


def test_two_real_speakers_kept_one_phantom_dropped():
    # Mirrors the observed sweep: 928 + 220 real, 30 + 5 phantom -> 2 speakers.
    turns = [
        {"start": 0.0,    "end": 928.0,  "speaker": "S1"},
        {"start": 928.0,  "end": 1148.0, "speaker": "S3"},
        {"start": 1148.0, "end": 1178.0, "speaker": "S0"},   # 30s, ~2.5%
        {"start": 1178.0, "end": 1183.0, "speaker": "S2"},   # 5s
    ]
    out = _drop_phantom_speakers(turns)
    assert {t["speaker"] for t in out} == {"S1", "S3"}


def test_quiet_but_real_speaker_with_many_turns_is_kept():
    # Regression: a quiet interviewer speaks only ~48s of a 20-min call (<5% of
    # talk time) but across many short, dispersed turns. Pruning by share alone
    # erased them wholesale onto the dominant speaker — "wrong person" for every
    # question. The turn-count gate must keep them.
    turns = []
    t = 0.0
    for _ in range(20):
        turns.append({"start": t, "end": t + 56.0, "speaker": "A"})   # dominant, long turns
        t += 56.0
        turns.append({"start": t, "end": t + 2.4, "speaker": "B"})    # brief interjection
        t += 2.4
    # B: 20 turns * 2.4s = 48s over a ~1168s call (~4.1%, below the 5% floor),
    # but 20 turns >> _MAX_PHANTOM_TURNS, so B is a real speaker and survives.
    out = _drop_phantom_speakers(turns)
    assert {t["speaker"] for t in out} == {"A", "B"}
    assert out == turns          # nothing relabeled


def test_sparse_low_share_cluster_still_pruned():
    # The turn-count gate must not neuter real phantom removal: a 1-2 turn blip
    # that is also below the time floor is still an artifact and gets folded.
    turns = [
        {"start": 0.0,   "end": 600.0, "speaker": "A"},
        {"start": 600.0, "end": 603.0, "speaker": "NOISE"},   # 3s, 1 turn
        {"start": 603.0, "end": 900.0, "speaker": "B"},
    ]
    out = _drop_phantom_speakers(turns)
    assert {t["speaker"] for t in out} == {"A", "B"}
