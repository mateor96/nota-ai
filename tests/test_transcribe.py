"""Tests for app.transcribe pure helpers: the per-segment confidence filter,
the VAD-derived decode span (Layer 1) and the silence-seeded repetition-loop
repair (Layer 2). The model call itself (whole-file continuous decode) is
exercised end-to-end, not here — every test below is pure or stubs
`mlx_whisper.transcribe`, so nothing loads a model."""
from __future__ import annotations

import numpy as np
import pytest

from app import transcribe as transcribe_module
from app.transcribe import (
    _MAX_REPAIRS,
    _clip_timestamps,
    _detect_language,
    _keep_segment,
    _key,
    _repair_repetition_loops,
    _repetition_run,
    _resume_after_silence,
)

# A sentence long enough to fail the _REPEAT_MAX_CHARS (60) gate, i.e. normal
# speech that can never be mistaken for a stock-phrase loop.
_LONG = "Guten Tag und herzlich willkommen zu diesem Gespraech heute Morgen hier."


def _seg(text, start, end, **kw):
    return {"text": text, "start": start, "end": end,
            "no_speech_prob": 0.0, "compression_ratio": 1.0, "words": [], **kw}


def _loop(text, n, start, step):
    """n consecutive segments of `text`, `step` seconds each, from `start`."""
    return [_seg(text, start + i * step, start + (i + 1) * step) for i in range(n)]


# ── _keep_segment (pre-existing) ─────────────────────────────────────────────

def test_keep_segment_drops_silence():
    assert not _keep_segment({"text": "hallo", "no_speech_prob": 0.9})


def test_keep_segment_drops_repetitive():
    assert not _keep_segment({"text": "x" * 80, "compression_ratio": 3.0})


def test_keep_segment_drops_empty_text():
    assert not _keep_segment({"text": "   ", "no_speech_prob": 0.0})


def test_keep_segment_keeps_normal_speech():
    assert _keep_segment({"text": "hallo welt", "no_speech_prob": 0.1, "compression_ratio": 1.4})


# ── A1. _clip_timestamps — Layer 1's judgement ───────────────────────────────

def test_clip_none_without_regions():
    assert _clip_timestamps(None, 1650.0) is None
    assert _clip_timestamps([], 1650.0) is None


def test_clip_trims_the_silent_intro():
    """The real case: 3 min of dead air at the head, speech to the end."""
    assert _clip_timestamps([(181.0, 400.0), (410.0, 1640.0)], 1650.0) == [180.0, 1650.0]


def test_clip_trims_both_ends():
    assert _clip_timestamps([(200.0, 500.0)], 1650.0) == [199.0, 501.0]


def test_clip_head_only():
    assert _clip_timestamps([(200.0, 1640.0)], 1650.0) == [199.0, 1650.0]


def test_clip_ignores_short_lead():
    """A 3 s lead is under _MIN_TRIM_S, so the head is left alone; the 50 s
    tail still is trimmed."""
    assert _clip_timestamps([(3.0, 1600.0)], 1650.0) == [0.0, 1601.0]


def test_clip_no_trim_at_either_end():
    assert _clip_timestamps([(5.0, 1645.0)], 1650.0) is None


def test_clip_rejects_tiny_span():
    """A door slam is not a recording — distrust VAD and decode everything."""
    assert _clip_timestamps([(800.0, 802.0)], 1650.0) is None


def test_clip_rejects_start_past_eof():
    """Pins the silent-empty-result trap shut: a clip start beyond
    content_frames makes `while seek < seek_clip_end` never run, and
    mlx_whisper returns text='' / segments=[] with no exception."""
    assert _clip_timestamps([(1700.0, 1800.0)], 1650.0) is None


def test_clip_rejects_zero_duration():
    assert _clip_timestamps([(1.0, 2.0)], 0.0) is None


def test_clip_rejects_an_implausibly_small_share_of_the_file():
    """VAD under-detection (music, a degraded phone leg) can return one
    plausible-looking short region. Clipping to it would silently throw away 27
    minutes of speech, so the clip is refused and the whole file is decoded."""
    assert _clip_timestamps([(100.0, 104.0)], 1650.0) is None
    assert _clip_timestamps([(600.0, 640.0)], 1650.0) is None
    # ...but a span that is a real share of the recording still clips.
    assert _clip_timestamps([(200.0, 500.0)], 1650.0) == [199.0, 501.0]


# ── A2. _key ─────────────────────────────────────────────────────────────────

def test_key_normalizes_punctuation_case_and_space():
    assert _key(_seg("Vielen Dank.", 0, 1)) == "vielen dank"
    assert _key(_seg("  vielen   DANK! ", 0, 1)) == "vielen dank"


def test_key_of_empty_text_is_empty():
    assert _key(_seg("   ", 0, 1)) == ""
    assert _key({}) == ""


# ── A3. _repetition_run — Layer 2's detector ─────────────────────────────────

def test_run_fires_on_the_real_loop():
    segs = _loop("Vielen Dank.", 8, 181.0, 30.0)
    assert _repetition_run(segs) == (0, 8)


def test_run_fires_on_an_ab_cycle():
    segs = [_seg("Vielen Dank." if i % 2 == 0 else "Danke.", i * 5.0, (i + 1) * 5.0)
            for i in range(8)]
    assert _repetition_run(segs) == (0, 8)


def test_run_ignores_three_repeats():
    assert _repetition_run(_loop("Vielen Dank.", 3, 181.0, 30.0)) is None


def test_run_ignores_a_short_span():
    assert _repetition_run(_loop("Vielen Dank.", 8, 0.0, 0.75)) is None


def test_run_ignores_normal_speech():
    segs = [_seg(f"Das ist Satz Nummer {i} in diesem Gespraech.", i * 5.0, (i + 1) * 5.0)
            for i in range(8)]
    assert _repetition_run(segs) is None


def test_run_ignores_long_repeated_sentences():
    """A repeated 90-char sentence is over _REPEAT_MAX_CHARS — loops are short
    stock phrases, and a long recurring sentence is far likelier to be real."""
    long_text = "Und das ist genau der Punkt, an dem wir heute gemeinsam ansetzen wollen, meine Damen."
    assert len(long_text) > 60
    assert _repetition_run(_loop(long_text, 8, 0.0, 30.0)) is None


def test_run_finds_a_loop_after_clean_speech():
    segs = [_seg(f"{_LONG} {i}", i * 10.0, (i + 1) * 10.0) for i in range(3)]
    segs += _loop("Vielen Dank.", 8, 30.0, 30.0)
    assert _repetition_run(segs) == (3, 11)


def test_run_skips_declined_starts():
    """Declining the run's first segment skips its same-text neighbours too —
    otherwise the identical loop comes straight back one segment later."""
    segs = _loop("Vielen Dank.", 8, 181.0, 30.0)
    assert _repetition_run(segs, frozenset({181.0})) is None


def test_run_returns_the_longest_of_two():
    segs = _loop("Ja.", 6, 0.0, 5.0)
    segs.append(_seg(_LONG, 30.0, 40.0))
    segs += _loop("Vielen Dank.", 10, 40.0, 5.0)
    assert _repetition_run(segs) == (7, 17)


# ── A4. _resume_after_silence — Layer 2's guard + locator ────────────────────

def test_resume_finds_speech_after_dead_air():
    assert _resume_after_silence([(181.0, 400.0)], 0.0, 350.0) == 181.0


def test_resume_none_when_run_sits_on_speech():
    """A chant or a repeated PA announcement: the audio genuinely repeats."""
    assert _resume_after_silence([(0.0, 400.0)], 100.0, 300.0) is None


def test_resume_none_for_a_short_gap():
    assert _resume_after_silence([(0.0, 100.0), (104.0, 400.0)], 100.0, 300.0) is None


def test_resume_none_when_a_region_covers_the_whole_run():
    assert _resume_after_silence([(170.0, 400.0)], 175.0, 300.0) is None


def test_resume_looks_past_a_blip_at_the_head_of_the_run():
    """One 0.5 s VAD false positive inside the boosted-noise intro must not be
    read as "the run sits on speech" — the widest silence is what locates the
    real resumption."""
    assert _resume_after_silence(
        [(100.0, 100.5), (181.0, 1640.0)], 99.0, 377.0) == 181.0


def test_resume_looks_past_a_real_utterance_at_the_head_of_the_run():
    """Record button, "Test, test.", three minutes of dead air, then the
    meeting. The loop is seeded by the dead air, not by the test words."""
    assert _resume_after_silence(
        [(5.0, 7.0), (190.0, 1600.0)], 5.0, 248.0) == 190.0


def test_resume_none_when_speech_resumes_after_the_run():
    assert _resume_after_silence([(500.0, 600.0)], 100.0, 300.0) is None


def test_resume_none_with_no_regions_after_the_start():
    assert _resume_after_silence([(0.0, 50.0)], 100.0, 300.0) is None


# ── A5. _repair_repetition_loops — stubbed model ─────────────────────────────

@pytest.fixture
def stub_model(monkeypatch):
    """Record calls to mlx_whisper.transcribe and return a scripted result."""
    calls: list[dict] = []
    state: dict = {"result": {"segments": []}, "audios": []}

    def stub(audio, **kwargs):
        calls.append(kwargs)
        state["audios"].append(audio)
        result = state["result"]
        return result(len(calls)) if callable(result) else result

    monkeypatch.setattr(transcribe_module.mlx_whisper, "transcribe", stub)
    return calls, state


# clean_a (long, never part of a run) + a "Vielen Dank." loop 30-270 s + clean_b
_LOOP_PASS1 = ([_seg(_LONG, 0.0, 6.0)]
               + _loop("Vielen Dank.", 8, 30.0, 30.0)
               + [_seg("Und damit kommen wir langsam zum Ende dieses ausfuehrlichen Gespraechs.", 430.0, 440.0)])
_RECOVERED = [_seg("Hallo, kannst du mich gut hoeren?", 181.0, 190.0),
              _seg("Ja, ich hoere dich sehr gut, vielen Dank fuers Treffen.", 260.0, 275.0),
              _seg("Dann stelle ich mich kurz vor und beschreibe die Rolle.", 340.0, 360.0)]
_REGIONS_SEEDED = [(181.0, 1640.0)]
_DURATION = 1650.0


def _repair(segments, regions, decode_kwargs=None):
    return _repair_repetition_loops(
        np.zeros(4, dtype=np.float32), _DURATION, "model", "de",
        list(segments), regions, decode_kwargs or dict(transcribe_module._DECODE),
    )


def test_repair_splices_recovered_speech(stub_model):
    calls, state = stub_model
    state["result"] = {"segments": list(_RECOVERED)}

    out = _repair(_LOOP_PASS1, _REGIONS_SEEDED)

    assert [s["text"] for s in out] == [
        _LOOP_PASS1[0]["text"], *[s["text"] for s in _RECOVERED], _LOOP_PASS1[-1]["text"]]
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)
    assert len(calls) == 1
    assert calls[0]["clip_timestamps"] == [180.0, 430.0]
    assert calls[0]["condition_on_previous_text"] is True
    assert calls[0]["word_timestamps"] is True


def test_repair_leaves_genuine_repetition_alone(stub_model):
    calls, _ = stub_model
    out = _repair(_LOOP_PASS1, [(0.0, 1640.0)])
    assert out == _LOOP_PASS1
    assert calls == []


def test_repair_never_deletes_when_redecode_is_empty(stub_model):
    calls, state = stub_model
    state["result"] = {"segments": []}
    out = _repair(_LOOP_PASS1, _REGIONS_SEEDED)
    assert out == _LOOP_PASS1
    assert len(calls) == 1


def test_repair_never_deletes_when_redecode_still_loops(stub_model):
    calls, state = stub_model
    state["result"] = {"segments": _loop("Vielen Dank.", 8, 181.0, 30.0)}
    out = _repair(_LOOP_PASS1, _REGIONS_SEEDED)
    assert out == _LOOP_PASS1
    assert len(calls) == 1


def test_repair_no_op_without_regions(stub_model):
    calls, _ = stub_model
    assert _repair(_LOOP_PASS1, None) == _LOOP_PASS1
    assert calls == []


def test_repair_no_op_without_a_loop(stub_model):
    """Clean files cost nothing: no detector hit, no model call."""
    calls, _ = stub_model
    clean = [_seg(f"{_LONG} {i}", i * 10.0, (i + 1) * 10.0) for i in range(12)]
    assert _repair(clean, _REGIONS_SEEDED) == clean
    assert calls == []


def test_repair_uses_duration_when_the_run_reaches_eof(stub_model):
    calls, state = stub_model
    state["result"] = {"segments": list(_RECOVERED)}
    _repair(_LOOP_PASS1[:-1], _REGIONS_SEEDED)
    assert len(calls) == 1
    assert calls[0]["clip_timestamps"] == [180.0, _DURATION]


def test_repair_forwards_initial_prompt(stub_model):
    calls, state = stub_model
    state["result"] = {"segments": list(_RECOVERED)}
    kwargs = dict(transcribe_module._DECODE, initial_prompt="Nota.ai, pyannote")
    _repair(_LOOP_PASS1, _REGIONS_SEEDED, kwargs)
    assert calls[0]["initial_prompt"] == "Nota.ai, pyannote"


def test_repair_is_bounded(stub_model):
    """A model that keeps looping must not spin: the repair is budgeted."""
    calls, state = stub_model
    state["result"] = lambda n: {"segments": _loop("Vielen Dank.", 8, 181.0, 30.0)}
    out = _repair(_LOOP_PASS1, _REGIONS_SEEDED)
    assert out == _LOOP_PASS1
    assert 0 < len(calls) <= _MAX_REPAIRS


def test_repair_keeps_a_real_segment_swallowed_by_the_run(stub_model):
    """_REPEAT_MAX_DISTINCT lets one short REAL utterance sit at the head of a
    run. It lies outside the re-decoded clip, so the splice must keep it rather
    than delete audio VAD itself calls speech."""
    calls, state = stub_model
    segs = ([_seg(_LONG, 80.0, 100.0), _seg("Genau.", 100.0, 105.0)]
            + _loop("Vielen Dank.", 30, 106.0, 10.0))
    regions = [(0.0, 105.0), (300.0, 1600.0)]
    state["result"] = {"segments": [_seg("Und weiter geht es mit dem eigentlichen Gespraech.", 301.0, 320.0)]}

    out = _repair(segs, regions)

    assert len(calls) == 1
    texts = [s["text"] for s in out]
    assert texts == [_LONG, "Genau.", "Und weiter geht es mit dem eigentlichen Gespraech."]
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)


def test_repair_drops_the_hallucinated_head_that_vad_calls_silence(stub_model):
    """The mirror of the test above: run-head segments over dead air are NOT
    kept, or the loop would survive its own repair."""
    calls, state = stub_model
    segs = _loop("Vielen Dank.", 30, 10.0, 10.0)
    regions = [(300.0, 1600.0)]
    state["result"] = {"segments": [_seg("Und weiter geht es mit dem eigentlichen Gespraech.", 301.0, 320.0)]}

    out = _repair(segs, regions)

    assert len(calls) == 1
    assert [s["text"] for s in out] == ["Und weiter geht es mit dem eigentlichen Gespraech."]


def test_repair_survives_a_failing_redecode(stub_model):
    """A repair that explodes must not lose the usable pass-1 transcript."""
    calls, state = stub_model

    def boom(n):
        raise RuntimeError("MLX exploded")

    state["result"] = boom
    assert _repair(_LOOP_PASS1, _REGIONS_SEEDED) == _LOOP_PASS1
    assert 0 < len(calls) <= _MAX_REPAIRS


def test_repair_trims_overrun_segments(stub_model):
    """A segment at/past the clip end would break ordering at the seam."""
    calls, state = stub_model
    state["result"] = {"segments": list(_RECOVERED)
                       + [_seg("Ueberlaeufer hinter der Naht.", 430.0, 445.0)]}
    out = _repair(_LOOP_PASS1, _REGIONS_SEEDED)
    assert [s["text"] for s in out] == [
        _LOOP_PASS1[0]["text"], *[s["text"] for s in _RECOVERED], _LOOP_PASS1[-1]["text"]]
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)


# ── A6. _detect_language ─────────────────────────────────────────────────────

@pytest.fixture
def probe_audio():
    """300 s of ramp audio: sample N holds the value N, so the clip a probe was
    handed reveals exactly where in the file it was taken from."""
    return np.arange(300 * 16000, dtype=np.float32)


def test_language_probe_centers_on_the_speech_span(stub_model, probe_audio):
    """On a recording that opens with dead air the file midpoint lands in
    silence; the speech-span midpoint cannot."""
    _, state = stub_model
    state["result"] = {"language": "de"}

    assert _detect_language(probe_audio, "model", [(280.0, 295.0)]) == "de"

    # speech midpoint 287.5 s, minus the 15 s half-window == 272.5 s
    assert state["audios"][0][0] == pytest.approx(272.5 * 16000)


def test_language_probe_falls_back_to_the_file_midpoint(stub_model, probe_audio):
    _, state = stub_model
    state["result"] = {"language": "en"}

    assert _detect_language(probe_audio, "model", None) == "en"

    # file midpoint 150 s, minus the 15 s half-window == 135 s
    assert state["audios"][0][0] == pytest.approx(135.0 * 16000)


# ── A7. transcribe() — the two integration points ────────────────────────────

@pytest.fixture
def stub_audio(monkeypatch):
    """load_audio returns a fixed-length silent buffer; `duration` follows it."""
    def _make(seconds: float):
        audio = np.zeros(int(seconds * 16000), dtype=np.float32)
        monkeypatch.setattr(transcribe_module, "load_audio", lambda path: audio)
        return audio
    return _make


def test_transcribe_installs_the_vad_clip_on_the_first_pass(stub_model, stub_audio):
    calls, state = stub_model
    stub_audio(300.0)
    state["result"] = {"segments": [_seg(_LONG, 120.0, 130.0)], "language": "de"}

    out = transcribe_module.transcribe(
        "x.wav", language="de", speech_regions=[(120.0, 290.0)])

    assert len(calls) == 1                       # no language probe, no repair
    assert calls[0]["clip_timestamps"] == [119.0, 300.0]
    assert calls[0]["condition_on_previous_text"] is True
    assert [s["text"] for s in out["segments"]] == [_LONG]


def test_transcribe_repairs_a_loop_in_the_pass_one_segments(stub_model, stub_audio):
    calls, state = stub_model
    stub_audio(300.0)
    pass1 = [_seg(_LONG, 0.0, 6.0)] + _loop("Vielen Dank.", 8, 30.0, 10.0)
    recovered = _seg("Hallo, kannst du mich gut hoeren, das ist wichtig hier.", 100.0, 110.0)
    state["result"] = lambda n: ({"segments": pass1, "language": "de"} if n == 1
                                 else {"segments": [recovered]})

    out = transcribe_module.transcribe(
        "x.wav", language="de", speech_regions=[(0.0, 6.0), (100.0, 290.0)])

    assert len(calls) == 2                       # pass 1, then the repair
    assert "clip_timestamps" not in calls[0]     # nothing worth trimming
    assert calls[1]["clip_timestamps"] == [99.0, 300.0]
    assert [s["text"] for s in out["segments"]] == [_LONG, recovered["text"]]
