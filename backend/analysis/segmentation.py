"""
segmentation.py
Algorithmic scene segmentation using librosa signal analysis.

Three signals are blended into a per-frame "cut salience" curve, then k-1
peaks are greedily selected to produce k scenes.

  w_structure  — mel-spectral onset strength  (timbral / structural shifts)
  w_energy     — absolute change in RMS energy (loudness transitions)
  w_beat       — bar-grid alignment            (cuts that land on bar boundaries)

Calling segment() with different weights produces different cut placements at
the same k — this is the "reroll" mechanism. Use random_weights() to resample.
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
    lyric_snap_s: float = 0.8,
) -> list[SegmentResult]:
    """
    Segment audio into k scenes using a weighted blend of three signals.

    Args:
        audio_path:   Path to audio file.
        k:            Number of scenes. k-1 cuts will be placed.
        fps:          Frames per second for snap_frames() (typically 25).
        min_s:        Minimum scene duration in seconds. No cut will be placed
                      within min_s of another cut or either edge.
        max_s:        Soft maximum. Any segment exceeding max_s is split at its
                      midpoint after selection.
        weights:      (w_structure, w_energy, w_beat). Should sum to ~1.0.
                      Defaults to default_weights().
        words:        Aligned word timestamps [{word, start_s, end_s}]. When
                      provided, each cut is snapped to the nearest inter-word
                      gap within lyric_snap_s seconds.
        lyric_snap_s: Tolerance window for lyric gap snapping (default 0.8s).

    Returns:
        List of SegmentResult dicts: {start_s, end_s, frame_count}.
        frame_count is snapped to the nearest 8k+1 (LTX-2 / HuMo constraint).
    """
    w_struct, w_energy, w_beat = weights or default_weights()
    audio_path = str(Path(audio_path).resolve())

    logger.info(
        "Segmenting %s → %d scenes (w_struct=%.2f w_energy=%.2f w_beat=%.2f)",
        audio_path, k, w_struct, w_energy, w_beat,
    )

    y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    n_frames = 1 + len(y) // _HOP

    # ── Signal 1: structural — mel onset strength ────────────────────────────
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=_HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    struct_nov = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=_HOP)

    # ── Signal 2: dynamic — |Δ RMS| ─────────────────────────────────────────
    rms        = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    energy_nov = np.abs(np.diff(rms, prepend=rms[0]))

    # ── Signal 3: rhythmic — bar grid ───────────────────────────────────────
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bar_nov = np.zeros(n_frames, dtype=np.float32)
    for i, bf in enumerate(beat_frames):
        if i % 4 == 0 and bf < n_frames:
            bar_nov[bf] = 1.0
    bar_nov = gaussian_filter1d(bar_nov, sigma=3.0)

    # ── Blend ────────────────────────────────────────────────────────────────
    salience = (
        w_struct * _norm(struct_nov, n_frames)
        + w_energy * _norm(energy_nov, n_frames)
        + w_beat   * _norm(bar_nov,    n_frames)
    )

    # Suppress edges so no cut lands within min_s of start/end
    edge_frames = max(1, int(min_s * sr / _HOP))
    salience[:edge_frames]  = 0.0
    salience[-edge_frames:] = 0.0

    # ── Select k-1 peaks ─────────────────────────────────────────────────────
    min_gap     = max(1, int(min_s * sr / _HOP))
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=_HOP)
    cut_frames  = _greedy_peaks(salience, k - 1, min_gap)
    cut_times   = sorted(frame_times[f] for f in cut_frames)

    # ── Lyric gap snap ────────────────────────────────────────────────────────
    if words:
        cut_times = _snap_to_lyrics(cut_times, words, lyric_snap_s, min_s)

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
    segments = _enforce_max(segments, max_s, fps)

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

    gap_mids: list[float] = []
    for i in range(len(clean) - 1):
        end_i   = clean[i]["end_s"]
        start_n = clean[i + 1]["start_s"]
        if start_n > end_i:
            gap_mids.append((end_i + start_n) / 2.0)

    if not gap_mids:
        return cut_times

    gap_arr   = np.array(gap_mids, dtype=np.float64)
    snapped   = list(cut_times)
    distances = [0.0] * len(cut_times)

    for idx, t in enumerate(cut_times):
        dists = np.abs(gap_arr - t)
        best  = int(np.argmin(dists))
        dist  = float(dists[best])
        if dist <= tolerance_s:
            snapped[idx]   = round(float(gap_arr[best]), 3)
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


def _enforce_max(
    segments: list[SegmentResult],
    max_s: float,
    fps: int,
) -> list[SegmentResult]:
    """
    Split any segment exceeding max_s at its midpoint.
    Recursive — handles multiple violations. Logs a warning if a split half
    would fall below min_s (does not error; caller sees the result).
    """
    out: list[SegmentResult] = []
    for seg in segments:
        dur = seg["end_s"] - seg["start_s"]
        if dur > max_s:
            logger.warning(
                "Scene %.2f–%.2f (%.1fs) exceeds max_s=%.1f — splitting at midpoint",
                seg["start_s"], seg["end_s"], dur, max_s,
            )
            mid = round(seg["start_s"] + dur / 2, 3)
            out.extend(_enforce_max([
                {"start_s": seg["start_s"], "end_s": mid,          "frame_count": snap_frames(mid - seg["start_s"], fps)},
                {"start_s": mid,            "end_s": seg["end_s"], "frame_count": snap_frames(seg["end_s"] - mid,   fps)},
            ], max_s, fps))
        else:
            out.append(seg)
    return out
