"""
segmentation.py
Algorithmic scene segmentation using librosa signal analysis.

Primary mode (when word timestamps are available):
  1. Detect instrumental breaks — gaps between sung words longer than
     instrumental_gap_s become mandatory cut points.
  2. Greedy fill-in — any segment still exceeding max_s gets additional cuts
     placed at signal peaks (mel onset + RMS delta + bar grid) within that
     segment only.
  3. Lyric snap — greedy fill-in cuts are snapped to the nearest inter-word gap.

Fallback mode (no word timestamps):
  Classic greedy peak picking on the blended salience curve using k as the
  target scene count.

Reroll varies the blend weights (Dirichlet-sampled) for different cut placement
at the same k — only meaningful in fallback mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .scenes import snap_frames

logger = logging.getLogger(__name__)

_SR  = 22050
_HOP = 512   # ~23ms per frame at 22050 Hz


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SegmentResult(TypedDict):
    start_s:     float
    end_s:       float
    frame_count: int


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def auto_k(duration_s: float, max_scene_s: float = 20.0) -> int:
    """Floor estimate of scene count. Minimum 2."""
    return max(2, int(duration_s / max_scene_s))


def default_weights() -> tuple[float, float, float]:
    """(w_structure, w_energy, w_beat) balanced default."""
    return (0.5, 0.3, 0.2)


def random_weights() -> tuple[float, float, float]:
    """
    Dirichlet-sampled weights for reroll. Biased toward structure signal.
    Always sums to 1.0.
    """
    w = np.random.dirichlet([3.0, 2.0, 1.0])
    return (round(float(w[0]), 3), round(float(w[1]), 3), round(float(w[2]), 3))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment(
    audio_path: str | Path,
    k: int,
    fps: int = 25,
    min_s: float = 3.0,
    max_s: float = 20.0,
    weights: tuple[float, float, float] | None = None,
    words: list[dict] | None = None,
    lyric_snap_s: float = 1.5,
    instrumental_gap_s: float = 4.0,
) -> list[SegmentResult]:
    """
    Segment audio into scenes.

    Primary mode (words provided):
      Instrumental breaks (word gaps >= instrumental_gap_s) become mandatory cut
      points. Any segment still over max_s gets additional cuts placed at signal
      peaks within that segment. Greedy fill-in cuts are lyric-snapped; mandatory
      cuts are not (they are already positioned inside the silence).

    Fallback mode (no words):
      Classic greedy peak picking on the blended salience curve, k target scenes.

    Args:
        audio_path:         Path to audio file.
        k:                  Target scene count (fallback mode only).
        fps:                Frames per second for snap_frames().
        min_s:              Minimum scene duration in seconds.
        max_s:              Soft maximum. Segments exceeding it get signal-driven
                            cuts placed inside them before _enforce_max runs.
        weights:            (w_structure, w_energy, w_beat). Fallback mode only.
        words:              Aligned word timestamps [{word, start_s, end_s}].
        lyric_snap_s:       Snap tolerance for greedy fill-in cuts (0.8s).
        instrumental_gap_s: Word gap threshold for mandatory cuts (default 4.0s).

    Returns:
        List of SegmentResult dicts: {start_s, end_s, frame_count}.
        frame_count snapped to nearest 8k+1 (LTX-2 / HuMo constraint).
    """
    w_struct, w_energy, w_beat = weights or default_weights()
    audio_path = str(Path(audio_path).resolve())

    y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    n_frames  = 1 + len(y) // _HOP

    # ── Always build salience curve (needed for fill-in even in primary mode) ─
    mel        = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=_HOP)
    mel_db     = librosa.power_to_db(mel, ref=np.max)
    struct_nov = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=_HOP)

    rms        = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    energy_nov = np.abs(np.diff(rms, prepend=rms[0]))

    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bar_nov = np.zeros(n_frames, dtype=np.float32)
    for i, bf in enumerate(beat_frames):
        if i % 4 == 0 and bf < n_frames:
            bar_nov[bf] = 1.0
    bar_nov = gaussian_filter1d(bar_nov, sigma=3.0)

    salience    = (
        w_struct * _norm(struct_nov, n_frames)
        + w_energy * _norm(energy_nov, n_frames)
        + w_beat   * _norm(bar_nov,    n_frames)
    )

    # ── Boost salience at lyric phrase boundaries ─────────────────────────────
    # Adds a fourth signal (word gap midpoints) so greedy fill-in prefers cuts
    # that land between phrases rather than mid-phrase.  Sentence endings (.!?)
    # get a stronger boost than mid-phrase gaps.
    if words:
        _pw = [w for w in words if w["end_s"] > w["start_s"] and not w["word"].startswith(('{', '('))]
        phrase_nov = np.zeros(n_frames, dtype=np.float32)
        for _pi in range(len(_pw) - 1):
            gap_s = _pw[_pi + 1]["start_s"] - _pw[_pi]["end_s"]
            if gap_s > 0.05:
                mid_s = (_pw[_pi]["end_s"] + _pw[_pi + 1]["start_s"]) / 2.0
                mid_f = min(n_frames - 1, int(mid_s * sr / _HOP))
                is_sentence = _pw[_pi]["word"].rstrip().endswith(('.', '!', '?'))
                phrase_nov[mid_f] = max(phrase_nov[mid_f], 1.0 if is_sentence else 0.5)
        phrase_nov = gaussian_filter1d(phrase_nov, sigma=2.0)
        salience = _norm(salience + 0.4 * _norm(phrase_nov, n_frames), n_frames)

    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=_HOP)
    min_gap_f   = max(1, int(min_s * sr / _HOP))

    if words:
        # ── Primary mode ──────────────────────────────────────────────────────
        # Step 1: mandatory cuts at instrumental breaks
        clean = [
            w for w in words
            if w["end_s"] > w["start_s"]
            and not w["word"].startswith(('{', '('))
        ]
        mandatory: list[float] = []
        for i in range(len(clean) - 1):
            gap_start = clean[i]["end_s"]
            gap_end   = clean[i + 1]["start_s"]
            if gap_end - gap_start >= instrumental_gap_s:
                mid = round((gap_start + gap_end) / 2.0, 3)
                if min_s <= mid <= duration - min_s:
                    mandatory.append(mid)

        logger.info(
            "Segmenting %s → primary mode: %d instrumental breaks",
            audio_path, len(mandatory),
        )

        # Step 2: greedy fill-in for segments still over max_s
        boundaries   = sorted(set([0.0] + mandatory + [duration]))
        greedy_cuts: list[float] = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end   = boundaries[i + 1]
            if seg_end - seg_start <= max_s:
                continue
            # mask salience to this segment only, respecting min_s from edges
            local = salience.copy()
            sf = int(seg_start * sr / _HOP)
            ef = min(n_frames, int(seg_end * sr / _HOP))
            local[:sf] = 0.0
            local[ef:] = 0.0
            local[sf:sf + min_gap_f] = 0.0
            local[max(0, ef - min_gap_f):ef] = 0.0
            n_cuts = max(1, int((seg_end - seg_start) / max_s))
            for f in _greedy_peaks(local, n_cuts, min_gap_f):
                greedy_cuts.append(round(float(frame_times[f]), 3))

        # Step 3: lyric-snap greedy cuts only (mandatory cuts stay put)
        if greedy_cuts and words:
            greedy_cuts = _snap_to_lyrics(greedy_cuts, words, lyric_snap_s, min_s)

        cut_times = sorted(mandatory + greedy_cuts)
        logger.info(
            "Segmenting %s → %d mandatory + %d fill-in = %d cuts",
            audio_path, len(mandatory), len(greedy_cuts), len(cut_times),
        )

    else:
        # ── Fallback mode: greedy peak picking ────────────────────────────────
        logger.info(
            "Segmenting %s → %d scenes fallback (w_struct=%.2f w_energy=%.2f w_beat=%.2f)",
            audio_path, k, w_struct, w_energy, w_beat,
        )
        edge_f = max(1, int(min_s * sr / _HOP))
        salience[:edge_f]  = 0.0
        salience[-edge_f:] = 0.0
        cut_frames = _greedy_peaks(salience, k - 1, min_gap_f)
        cut_times  = sorted(frame_times[f] for f in cut_frames)

    # ── Build segments ────────────────────────────────────────────────────────
    boundaries = [0.0] + list(cut_times) + [duration]
    segments: list[SegmentResult] = []
    for i in range(len(boundaries) - 1):
        start = round(float(boundaries[i]),     3)
        end   = round(float(boundaries[i + 1]), 3)
        segments.append({
            "start_s":     start,
            "end_s":       end,
            "frame_count": snap_frames(end - start, fps),
        })

    # ── Enforce max_s ─────────────────────────────────────────────────────────
    segments = _enforce_max(segments, max_s, fps, words or [])

    logger.info(
        "Done: %d scenes — %s",
        len(segments),
        ", ".join(f"{s['end_s'] - s['start_s']:.1f}s" for s in segments),
    )
    return segments


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _snap_to_lyrics(
    cut_times: list[float],
    words: list[dict],
    tolerance_s: float,
    min_s: float,
) -> list[float]:
    """
    Snap each cut time to the nearest inter-word gap midpoint within tolerance_s.

    Zero-duration words (start_s == end_s) and annotation tokens (text starting
    with '{' or '(') are excluded when building gap candidates.

    After snapping, any two cuts that land within min_s/2 of each other trigger
    a collision resolution: the cut that travelled further is reverted to its
    original position.
    """
    clean = [
        w for w in words
        if w["end_s"] > w["start_s"]
        and not w["word"].startswith(('{', '('))
    ]
    if not clean:
        return cut_times

    sentence_mids: list[float] = []
    all_gap_mids:  list[float] = []
    for i in range(len(clean) - 1):
        end_i   = clean[i]["end_s"]
        start_n = clean[i + 1]["start_s"]
        if start_n > end_i:
            mid = (end_i + start_n) / 2.0
            all_gap_mids.append(mid)
            if clean[i]["word"].rstrip().endswith(('.', '!', '?')):
                sentence_mids.append(mid)

    if not all_gap_mids:
        return cut_times

    all_arr  = np.array(all_gap_mids, dtype=np.float64)
    sent_arr = np.array(sentence_mids, dtype=np.float64) if sentence_mids else None
    snapped   = list(cut_times)
    distances = [0.0] * len(cut_times)

    for idx, t in enumerate(cut_times):
        # Prefer sentence-end gaps; fall back to any gap
        if sent_arr is not None:
            dists = np.abs(sent_arr - t)
            best  = int(np.argmin(dists))
            dist  = float(dists[best])
            if dist <= tolerance_s:
                snapped[idx]   = round(float(sent_arr[best]), 3)
                distances[idx] = dist
                continue
        dists = np.abs(all_arr - t)
        best  = int(np.argmin(dists))
        dist  = float(dists[best])
        if dist <= tolerance_s:
            snapped[idx]   = round(float(all_arr[best]), 3)
            distances[idx] = dist

    # Collision resolution: if two snapped cuts are within min_s/2 of each
    # other, revert the one that moved further (keep the better-placed one).
    collision_threshold = min_s * 0.5
    for i in range(len(snapped)):
        for j in range(i + 1, len(snapped)):
            if abs(snapped[i] - snapped[j]) < collision_threshold:
                if distances[i] >= distances[j]:
                    snapped[i] = cut_times[i]
                else:
                    snapped[j] = cut_times[j]

    for orig, new in zip(cut_times, snapped):
        if orig != new:
            logger.debug("Lyric snap: %.3fs → %.3fs", orig, new)

    return snapped


def _norm(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Resize to target_len via linear interpolation, then normalize to [0, 1]."""
    if len(sig) != target_len:
        sig = np.interp(
            np.linspace(0, len(sig) - 1, target_len),
            np.arange(len(sig)),
            sig,
        )
    mn, mx = float(sig.min()), float(sig.max())
    if mx > mn:
        sig = (sig - mn) / (mx - mn)
    return sig.astype(np.float32)


def _greedy_peaks(salience: np.ndarray, n: int, min_gap: int) -> list[int]:
    """
    Greedily select n frame indices from salience with at least min_gap frames
    between any two selected indices.
    """
    if n <= 0:
        return []

    available = salience.copy()
    selected: list[int] = []

    for _ in range(n):
        if available.max() == 0.0:
            break
        peak = int(np.argmax(available))
        selected.append(peak)
        lo = max(0, peak - min_gap)
        hi = min(len(available), peak + min_gap + 1)
        available[lo:hi] = 0.0

    return sorted(selected)


def _find_split_point(start_s: float, end_s: float, words: list[dict]) -> float | None:
    """
    Find the best inter-word gap midpoint to split [start_s, end_s] at.
    Prefers sentence-ending gaps (.!?) near the geometric midpoint.
    Falls back to any inter-word gap near the midpoint.
    Returns None if no usable gap exists within the segment.
    """
    clean = [w for w in words if w["end_s"] > w["start_s"] and not w["word"].startswith(('{', '('))]
    mid = (start_s + end_s) / 2.0
    sentence_cands: list[float] = []
    any_cands:      list[float] = []
    for i in range(len(clean) - 1):
        gap_start = clean[i]["end_s"]
        gap_end   = clean[i + 1]["start_s"]
        if gap_end > gap_start and start_s < gap_start and gap_end < end_s:
            gm = (gap_start + gap_end) / 2.0
            any_cands.append(gm)
            if clean[i]["word"].rstrip().endswith(('.', '!', '?')):
                sentence_cands.append(gm)
    if sentence_cands:
        return round(min(sentence_cands, key=lambda t: abs(t - mid)), 3)
    if any_cands:
        return round(min(any_cands, key=lambda t: abs(t - mid)), 3)
    return None


def _enforce_max(
    segments: list[SegmentResult],
    max_s: float,
    fps: int,
    words: list[dict] | None = None,
) -> list[SegmentResult]:
    """
    Split any segment exceeding max_s at a phrase boundary when possible,
    falling back to the geometric midpoint.
    Recursive — handles multiple violations.
    """
    out: list[SegmentResult] = []
    for seg in segments:
        dur = seg["end_s"] - seg["start_s"]
        if dur > max_s:
            mid = (_find_split_point(seg["start_s"], seg["end_s"], words) if words else None
                   ) or round(seg["start_s"] + dur / 2, 3)
            logger.warning(
                "Scene %.2f–%.2f (%.1fs) exceeds max_s=%.1f — splitting at %.3fs",
                seg["start_s"], seg["end_s"], dur, max_s, mid,
            )
            out.extend(_enforce_max([
                {"start_s": seg["start_s"], "end_s": mid,          "frame_count": snap_frames(mid - seg["start_s"], fps)},
                {"start_s": mid,            "end_s": seg["end_s"], "frame_count": snap_frames(seg["end_s"] - mid,   fps)},
            ], max_s, fps, words))
        else:
            out.append(seg)
    return out
