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

Two VAD-derived guards sit around that single pass, both of which need the
speech regions the pipeline already computes (`speech_regions=`). Neither
fragments the decode. The first places the START and END of the one continuous
decode so it never walks through minutes of dead air — loudnorm lifts a silent
intro 20-35 dB, so `hallucination_silence_threshold` does not fire there,
Whisper captions the noise, and `condition_on_previous_text` turns that caption
into a self-reinforcing prompt that overwrites the REAL speech that follows.
The second detects such a loop after the fact — it is a property of the
SEQUENCE of segments, invisible to any per-segment gate — and re-decodes just
the affected span from the point VAD says speech actually resumes. Both are
no-ops without VAD regions, and the repair never deletes: if the re-decode does
not come back clean, the original segments are kept.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import mlx_whisper
from mlx_whisper.audio import SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)

# large-v3-turbo. We tried the full large-v3 MLX build for higher accuracy, but
# that conversion is unreliable on real recordings — on some audio it collapses
# into lowercase, unpunctuated, repeated-word garbage ("die die die videos...")
# where turbo stays clean and correct. turbo is the consistent choice; the text
# quality comes from decoding the whole file continuously (see below), not the model.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Per-segment confidence gate — drops residual hallucination on noise/silence.
# Conservative so real (quiet) speech is kept.
#
# NOTE: _MAX_NO_SPEECH_PROB is effectively inert on large-v3-turbo. Measured:
# 30 s of digital silence decodes to " Vielen Dank." with no_speech_prob
# 1.5e-10 and avg_logprob -0.278, so neither this gate nor mlx_whisper's own
# no_speech_threshold=0.6 (transcribe.py L302-315) ever fires. Do not try to
# fix hallucination by tuning it. Kept because it costs nothing and would work
# if the model is ever swapped.
_MAX_NO_SPEECH_PROB = 0.85
# High ratio == repetitive gibberish WITHIN one segment. It cannot see a
# repetition LOOP: "Vielen Dank." has compression_ratio 0.60 — each segment is
# individually innocent and only the sequence is pathological. That is what
# _repetition_run below is for.
_MAX_COMPRESSION_RATIO = 2.4

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

# ── Layer 1: VAD-gated decode span ───────────────────────────────────────────
_CLIP_PAD_S   = 1.0    # run-up kept before the first / after the last word
_MIN_TRIM_S   = 30.0   # only trim an end carrying at least this much dead air
_MIN_SPEECH_S = 5.0    # a shorter total span than this is a VAD artifact
# ...and neither is a span that is a negligible fraction of the recording. VAD
# under-detection (music/singing, a badly degraded phone leg) can return one
# plausible-looking short region on a 27-minute file; clipping to it would throw
# the whole recording away silently. The clip is an optimization, never
# load-bearing, so an implausibly small one is refused outright.
_MIN_SPEECH_FRACTION = 0.10

# ── Layer 2: silence-seeded repetition repair ────────────────────────────────
_REPEAT_MIN_SEGMENTS = 6      # consecutive segments drawn from <= 2 texts
_REPEAT_MIN_SECONDS  = 20.0   # ...and spanning at least this much audio
_REPEAT_MAX_CHARS    = 60     # loops are short stock phrases, not sentences
_REPEAT_MAX_DISTINCT = 2      # covers "A A A A" and "A B A B" cycles
_MIN_SEED_SILENCE_S  = 10.0   # dead air at the run's head to call it seeded
_RESUME_BACKOFF_S    = 1.0    # re-decode a beat before the recovered speech
_MAX_REPAIRS         = 2      # bounded: at most this many runs repaired

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _clip_timestamps(
    regions: Optional[list[tuple[float, float]]], duration: float
) -> Optional[list[float]]:
    """ONE clip spanning the detected speech, or None to decode the whole file.

    One clip only. mlx_whisper honours `seek_clips[0][0]` (transcribe.py L248)
    and the clip end (L286, L290), but the loop at L285 binds seek_clip_start
    without ever reading it and never advances clip_idx, so every LATER clip
    start is ignored and the gaps between clips are decoded anyway. Multi-clip
    gap-skipping would look like it worked while doing nothing.
    """
    if not regions or duration <= 0.0:
        return None
    start = max(0.0, min(regions[0][0], duration) - _CLIP_PAD_S)
    end = min(duration, max(regions[-1][1], 0.0) + _CLIP_PAD_S)
    if end - start < max(_MIN_SPEECH_S, _MIN_SPEECH_FRACTION * duration):
        return None                      # distrust VAD; decode everything
    if start < _MIN_TRIM_S:
        start = 0.0                      # not enough dead air to be worth it
    if duration - end < _MIN_TRIM_S:
        end = duration
    if start <= 0.0 and end >= duration:
        return None                      # nothing to trim — pass no clip at all
    return [start, end]


def _key(seg: dict) -> str:
    """Normalized comparison key: 'Vielen Dank.' == 'vielen dank!' == ' Vielen  Dank '."""
    return _NON_WORD.sub(" ", seg.get("text", "").lower()).strip()


def _repetition_run(
    segments: list[dict], declined: frozenset[float] = frozenset()
) -> Optional[tuple[int, int]]:
    """Longest maximal run of consecutive segments that is a repetition loop.

    Returns a half-open [i, j) index range, or None. A run qualifies when it is
    at least _REPEAT_MIN_SEGMENTS long, spans at least _REPEAT_MIN_SECONDS,
    draws on at most _REPEAT_MAX_DISTINCT distinct short keys, and actually
    repeats (fewer distinct keys than segments).

    `declined` holds the segment start times of runs the repair has already
    given up on. A declined segment is skipped together with its same-text
    neighbours: walking past only the first one would re-find the very same
    loop starting one segment later and burn the repair budget on it.
    """
    keys = [_key(s) for s in segments]
    best = None
    i, n = 0, len(keys)
    while i < n:
        if segments[i]["start"] in declined:
            declined_key = keys[i]
            i += 1
            while i < n and keys[i] == declined_key:
                i += 1
            continue
        if not keys[i] or len(keys[i]) > _REPEAT_MAX_CHARS:
            i += 1
            continue
        distinct, j = {keys[i]}, i + 1
        while (j < n and keys[j] and len(keys[j]) <= _REPEAT_MAX_CHARS
               and segments[j]["start"] not in declined
               and len(distinct | {keys[j]}) <= _REPEAT_MAX_DISTINCT):
            distinct.add(keys[j])
            j += 1
        span = segments[j - 1]["end"] - segments[i]["start"]
        if (j - i >= _REPEAT_MIN_SEGMENTS and len(distinct) < j - i
                and span >= _REPEAT_MIN_SECONDS
                and (best is None or j - i > best[1] - best[0])):
            best = (i, j)
        i = max(j, i + 1)
    return best


def _resume_after_silence(
    regions: list[tuple[float, float]], loop_start: float, loop_end: float
) -> Optional[float]:
    """When real speech resumes inside a repetition run, or None.

    The answer is the start of the speech region that follows the WIDEST stretch
    of dead air inside the run — NOT simply the first region at/after
    `loop_start`. The loop is seeded by dead air but does not stop when speech
    returns, so the run routinely opens with something VAD calls speech: a
    single spurious blip inside the boosted-noise intro, or a real "test, test"
    before the meeting starts. Anchoring on the first region would read those as
    "the run sits on speech" and abandon the repair on exactly the files it
    exists for.

    None means the run contains no long enough silence anywhere — the audio
    genuinely repeats (a chant, a repeated PA announcement, a stuck
    backchannel) and pass 1 was right. This one check replaces a whole
    speculative unconditioned probe pass: it is cheaper, and it is a stronger
    signal than re-decoding and hoping the repetition fails to reproduce.

    `regions` is sorted and merged (vad.detect_speech uses Timeline.support()).
    """
    best_start, best_gap = None, 0.0
    cursor = loop_start                  # end of the speech seen so far
    for s, e in regions:
        if e <= loop_start:
            continue
        if s >= loop_end:
            break                        # past the run — sorted, so we're done
        if s - cursor > best_gap:
            best_gap, best_start = s - cursor, s
        cursor = max(cursor, e)
    if best_start is None or best_gap < _MIN_SEED_SILENCE_S:
        return None                      # the run sits on continuous speech
    return best_start


def _overlaps_speech(seg: dict, regions: list[tuple[float, float]]) -> bool:
    """True if VAD calls any part of this segment's span speech."""
    return any(e > seg["start"] and s < seg["end"] for s, e in regions)


def _repair_repetition_loops(
    audio, decode_end: float, model: str, language: Optional[str],
    segments: list[dict], regions: Optional[list[tuple[float, float]]],
    decode_kwargs: dict,
) -> list[dict]:
    """Replace silence-seeded repetition loops with a clean re-decode.

    Never deletes real speech. Every failure path returns the pass-1 segments
    unchanged, so the worst case is the status quo for that file; and on the
    success path the only pass-1 segments dropped are the ones the re-decode
    covers plus, ahead of it, the ones VAD says sit on dead air. A run can open
    on a short REAL utterance (_REPEAT_MAX_DISTINCT allows one distinct short
    key beside the loop key), so the head of the run is filtered, not assumed
    to be hallucination. `decode_kwargs` carries
    condition_on_previous_text=True and word_timestamps=True — the repair span
    is decoded as ONE continuous conditioned clip, not per window. The only
    context it lacks is context from before `resume`, which is silence and
    hallucination: worthless context.
    """
    if not regions:
        return segments                   # no VAD -> no locator, no guard
    declined: set[float] = set()

    def _decline(run_segments: list[dict]) -> None:
        # Decline the WHOLE run, not just its first segment, so the next
        # _repetition_run call cannot hand back the same loop shifted by one.
        declined.update(seg["start"] for seg in run_segments)

    for _ in range(_MAX_REPAIRS):
        run = _repetition_run(segments, frozenset(declined))
        if run is None:
            return segments
        i, j = run
        loop_start, loop_end = segments[i]["start"], segments[j - 1]["end"]
        resume = _resume_after_silence(regions, loop_start, loop_end)
        if resume is None:
            logger.info(
                "Repetition run %.1f-%.1fs sits on detected speech - keeping it",
                loop_start, loop_end)
            _decline(segments[i:j])
            continue
        start = max(loop_start, resume - _RESUME_BACKOFF_S)
        # Snap the clip end to an existing pass-1 segment boundary (a pause), so
        # pad_or_trim's mel zero-padding on the clip's last window lands in a
        # gap rather than mid-sentence. `decode_end` is Layer 1's clip end, not
        # the file duration: a run reaching the last pass-1 segment must not
        # send the re-decode back through the tail dead air Layer 1 excluded.
        end = segments[j]["start"] if j < len(segments) else decode_end
        if end - start < _MIN_SPEECH_S:
            _decline(segments[i:j])
            continue
        logger.warning(
            "Repetition loop: %d segments, %.1f-%.1fs (%r) - re-decoding %.1f-%.1fs",
            j - i, loop_start, loop_end,
            segments[i].get("text", "").strip(), start, end)
        try:
            fixed = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model, language=language,
                clip_timestamps=[start, end], **decode_kwargs,
            )
        except Exception:
            # A usable pass-1 transcript already exists; a failed repair of it
            # must not fail the job.
            logger.warning("Re-decode of %.1f-%.1fs failed - keeping the original",
                           start, end, exc_info=True)
            _decline(segments[i:j])
            continue
        kept = [s for s in fixed.get("segments", [])
                if _keep_segment(s) and s["start"] < end]
        if not kept or _repetition_run(kept) is not None:
            logger.warning(
                "Re-decode of %.1f-%.1fs did not recover - keeping the original",
                start, end)
            _decline(segments[i:j])
            continue
        # The re-decode only covers [start, end). Anything in the run that ends
        # before `start` is either hallucination over dead air (dropped) or real
        # speech VAD confirms (kept) — never silently deleted.
        head = [s for s in segments[i:j]
                if s["end"] <= start and _overlaps_speech(s, regions)]
        logger.info("Recovered %.1f-%.1fs: %d segments replace %d repetitions"
                    " (%d kept ahead of the re-decode)",
                    start, kept[-1]["end"], len(kept), j - i, len(head))
        segments = segments[:i] + head + kept + segments[j:]
    return segments


def _detect_language(
    audio, model: str, regions: Optional[list[tuple[float, float]]] = None
) -> Optional[str]:
    """Probe the middle of the SPEECH, not the middle of the file.

    On a recording that opens (or ends) with minutes of dead air, the file
    midpoint is systematically pushed into that dead air; the midpoint of the
    first-speech-to-last-speech interval is not. It is only a better bet, not a
    guarantee — the interval has its own interior gaps and the probe can still
    land in one — and a probe over silence just returns whatever mlx_whisper
    guesses. Note that mlx_whisper's
    own auto-detect (transcribe.py L160-166) always uses pad_or_trim(mel,
    N_FRAMES) — the first 30 s — and ignores clip_timestamps entirely, so
    passing an explicit language here is what keeps a silent intro from
    deciding the language of the whole file. Do not remove this.
    """
    half = int(_LANG_PROBE_S * SAMPLE_RATE / 2)
    if regions:
        mid = int((regions[0][0] + regions[-1][1]) / 2 * SAMPLE_RATE)
    else:
        mid = len(audio) // 2
    mid = min(max(mid, 0), len(audio))
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
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    speech_regions: Optional[list[tuple[float, float]]] = None,
) -> dict:
    """Transcribe the whole file in one continuous pass and return kept segments.

    `speech_regions` is optional VAD output (see app/vad.py). With it, the decode
    skips long dead air at the ends and a silence-seeded repetition loop can be
    repaired; without it, behaviour is exactly as before.
    """
    audio = load_audio(audio_path)          # loaded ONCE and reused
    duration = len(audio) / SAMPLE_RATE

    if language is None:
        language = _detect_language(audio, model, speech_regions)

    decode_kwargs = dict(_DECODE)
    if initial_prompt:
        decode_kwargs["initial_prompt"] = initial_prompt

    first_pass = dict(decode_kwargs)
    clip = _clip_timestamps(speech_regions, duration)
    if clip is not None:
        logger.info("VAD: decoding %.1f-%.1fs of %.1fs", clip[0], clip[1], duration)
        first_pass["clip_timestamps"] = clip

    result = mlx_whisper.transcribe(
        audio, path_or_hf_repo=model, language=language, **first_pass
    )
    segments = [seg for seg in result.get("segments", []) if _keep_segment(seg)]
    segments = _repair_repetition_loops(
        audio, clip[1] if clip else duration, model, language,
        segments, speech_regions, decode_kwargs
    )
    return {"segments": segments, "language": result.get("language", language)}
