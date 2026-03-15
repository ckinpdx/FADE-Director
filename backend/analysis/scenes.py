"""
scenes.py
Scene-planning utilities: intonation stitching, frame snapping, scene validation.

Three responsibilities:

  1. stitch_intonation() — merge aligner word timestamps with Omni intonation
     sections by fuzzy-matching each section's lyrics_anchor phrase against the
     word list.  Result: time-grounded section descriptors ready for the 35B agent.

  2. snap_frames() — convert a scene duration to the nearest valid LTX-2 frame
     count (8k+1).  This is the binding constraint; HuMo (4k+1) is always
     satisfied by any 8k+1 value.

  3. validate_scenes() — check 35B-proposed scene boundaries against min/max
     duration constraints and audio length.  Returns a list of human-readable
     violation messages (empty list = all good).
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class WordTimestamp(TypedDict):
    word:    str
    start_s: float
    end_s:   float


class IntonationSection(TypedDict):
    section:       str
    lyrics_anchor: str
    intonation:    str
    energy:        str
    mood:          str


class StitchedSection(TypedDict):
    section:       str
    lyrics_anchor: str
    start_s:       float
    end_s:         float
    intonation:    str
    energy:        str
    mood:          str
    matched:       bool   # False = anchor not found; timestamps are approximate


class SceneEntry(TypedDict):
    start_s:    float
    end_s:      float
    frame_count: int
    label:      str


# ---------------------------------------------------------------------------
# snap_frames
# ---------------------------------------------------------------------------

def snap_frames(duration_s: float, fps: int) -> int:
    """
    Return the nearest frame count satisfying the LTX-2 constraint (8k+1).

    Feed this value directly to EmptyLTXVLatentVideo.length.
    Actual output duration = frame_count / fps (may differ from duration_s by <1 frame).

    Args:
        duration_s: Desired clip duration in seconds.
        fps:        Frames per second (typically 25).

    Returns:
        Frame count n where n = 8k+1 for some integer k >= 1.

    Examples:
        snap_frames(11.4, 25) → 281   (11.24s actual)
        snap_frames(6.2,  25) → 153   (6.12s actual)
        snap_frames(3.0,  25) →  25   (1.0s actual — enforces k>=1 so minimum is 9)
    """
    raw = round(duration_s * fps)
    k = max(1, round((raw - 1) / 8))
    return 8 * k + 1


# ---------------------------------------------------------------------------
# stitch_intonation — internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return list of tokens."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _anchor_match(
    anchor_tokens: list[str],
    word_tokens:   list[str],   # normalised word list parallel to words[]
    search_from:   int,
) -> tuple[int, float]:
    """
    Slide anchor_tokens over word_tokens[search_from:] and return
    (best_position_index, best_score) where score = matched_tokens / len(anchor).
    Returns (-1, 0.0) if anchor is empty.
    """
    n = len(anchor_tokens)
    if n == 0:
        return search_from, 0.0

    best_pos   = search_from
    best_score = 0.0

    for i in range(search_from, max(search_from + 1, len(word_tokens) - n + 1)):
        window = word_tokens[i : i + n]
        score  = sum(a == b for a, b in zip(anchor_tokens, window)) / n
        if score > best_score:
            best_score = score
            best_pos   = i

    return best_pos, best_score


_MATCH_THRESHOLD = 0.5   # accept if ≥50% of anchor tokens matched exactly


# ---------------------------------------------------------------------------
# stitch_intonation
# ---------------------------------------------------------------------------

def stitch_intonation(
    words:          list[WordTimestamp],
    sections:       list[IntonationSection],
    audio_duration: float,
) -> list[StitchedSection]:
    """
    Attach timestamps to Omni intonation sections by matching each section's
    lyrics_anchor phrase against the aligner word list.

    Sections are processed in order; the search cursor advances after each match
    so repeated sections (e.g. two identical choruses) resolve to distinct times.

    Unmatched sections (e.g. an instrumental Intro with no sung anchor) are
    assigned approximate timestamps by interpolation from adjacent matched sections.
    They are flagged with matched=False.

    Args:
        words:          Flat aligner output [{word, start_s, end_s}].
        sections:       Omni intonation output [{section, lyrics_anchor, ...}].
        audio_duration: Total audio duration in seconds (used as final end_s).

    Returns:
        List of StitchedSection dicts with start_s, end_s added.
        end_s of each section == start_s of the next (or audio_duration for last).
    """
    if not sections:
        return []

    # Build normalised parallel list for fast matching
    word_tokens = [_normalize(w["word"]) for w in words]
    word_tokens_flat = [t[0] if t else "" for t in word_tokens]

    # --- Pass 1: find match position (word index) for each section ----------
    cursor = 0
    match_indices: list[int | None] = []

    for sec in sections:
        anchor_tokens = _normalize(sec["lyrics_anchor"])
        pos, score = _anchor_match(anchor_tokens, word_tokens_flat, cursor)

        if score >= _MATCH_THRESHOLD:
            match_indices.append(pos)
            cursor = pos + max(1, len(anchor_tokens))   # advance past this match
            logger.debug(
                "  %-20s  anchor=%r  score=%.2f  pos=%d  t=%.2fs",
                sec["section"], sec["lyrics_anchor"][:30], score, pos,
                words[pos]["start_s"],
            )
        else:
            match_indices.append(None)
            logger.debug(
                "  %-20s  anchor=%r  NO MATCH (score=%.2f)",
                sec["section"], sec["lyrics_anchor"][:30], score,
            )

    # --- Pass 2: derive start_s for each section ----------------------------
    # Matched → use word start_s.  Unmatched → interpolate between neighbours.
    start_times: list[float | None] = []
    for i, idx in enumerate(match_indices):
        if idx is not None:
            start_times.append(words[idx]["start_s"])
        else:
            start_times.append(None)

    # Fill unmatched times by linear interpolation between adjacent known times
    # Edge: leading Nones get time 0.0; trailing Nones get audio_duration.
    known = [(i, t) for i, t in enumerate(start_times) if t is not None]

    if not known:
        # Nothing matched at all — distribute evenly
        n = len(sections)
        step = audio_duration / n
        start_times = [round(i * step, 3) for i in range(n)]
    else:
        # Fill left edge
        first_known_idx, first_known_t = known[0]
        for i in range(first_known_idx):
            start_times[i] = round(max(0.0, first_known_t - (first_known_idx - i) * 5.0), 3)

        # Fill right edge
        last_known_idx, last_known_t = known[-1]
        for i in range(last_known_idx + 1, len(start_times)):
            step = (audio_duration - last_known_t) / max(1, len(start_times) - last_known_idx)
            start_times[i] = round(last_known_t + (i - last_known_idx) * step, 3)

        # Fill interior gaps
        for j in range(len(known) - 1):
            lo_i, lo_t = known[j]
            hi_i, hi_t = known[j + 1]
            gap_count = hi_i - lo_i
            if gap_count > 1:
                step = (hi_t - lo_t) / gap_count
                for k in range(lo_i + 1, hi_i):
                    if start_times[k] is None:
                        start_times[k] = round(lo_t + (k - lo_i) * step, 3)

    # --- Pass 3: compute end_s and assemble output --------------------------
    stitched: list[StitchedSection] = []
    n = len(sections)
    for i, sec in enumerate(sections):
        start = start_times[i] or 0.0
        end   = start_times[i + 1] if i + 1 < n else audio_duration

        stitched.append({
            "section":       sec["section"],
            "lyrics_anchor": sec["lyrics_anchor"],
            "start_s":       round(float(start), 3),
            "end_s":         round(float(end),   3),
            "intonation":    sec["intonation"],
            "energy":        sec["energy"],
            "mood":          sec["mood"],
            "matched":       match_indices[i] is not None,
        })

    matched_count = sum(1 for s in stitched if s["matched"])
    logger.info(
        "Intonation stitched: %d/%d sections matched to aligner timestamps",
        matched_count, n,
    )
    return stitched


# ---------------------------------------------------------------------------
# validate_scenes
# ---------------------------------------------------------------------------

def validate_scenes(
    scenes:  list[SceneEntry],
    min_s:   float,
    max_s:   float,
    audio_duration: float,
) -> list[str]:
    """
    Validate 35B-proposed scene boundaries.

    Checks:
      - Each scene duration is within [min_s, max_s] (using snapped frame count)
      - Scenes are contiguous (no gaps, no overlaps)
      - Last scene does not exceed audio_duration

    Args:
        scenes:         List of scene dicts with start_s, end_s, frame_count.
        min_s, max_s:   Duration constraints in seconds.
        audio_duration: Total track length.

    Returns:
        List of human-readable violation strings.  Empty list means all good.
    """
    errors: list[str] = []

    for i, scene in enumerate(scenes):
        label    = scene.get("label", f"Scene {i + 1}")
        duration = scene["end_s"] - scene["start_s"]

        if duration < min_s:
            errors.append(f"{label}: {duration:.1f}s is below minimum {min_s}s")
        if duration > max_s:
            errors.append(f"{label}: {duration:.1f}s exceeds maximum {max_s}s")

    # Contiguity
    for i in range(1, len(scenes)):
        gap = scenes[i]["start_s"] - scenes[i - 1]["end_s"]
        if abs(gap) > 0.1:
            errors.append(
                f"Gap/overlap of {gap:.2f}s between scene {i} and scene {i + 1}"
            )

    # Audio boundary
    if scenes and scenes[-1]["end_s"] > audio_duration + 0.5:
        errors.append(
            f"Last scene ends at {scenes[-1]['end_s']:.1f}s "
            f"but audio is only {audio_duration:.1f}s"
        )

    return errors
