"""
music.py
Librosa-based audio analysis: BPM, beats, RMS energy curve, agglomerative
structural segments, and key/mode estimation.

All functions share a module-level cache keyed by resolved audio path so that
multiple agent tool calls on the same file only trigger one load + compute.

Usage:
    from backend.analysis.music import analyze, get_beat_grid, get_energy_curve, get_segments

    data = analyze("sessions/abc/audio/original.mp3")
    # data["bpm"], data["key"], data["mode"], data["beats"], data["energy"], data["segments"]

    beats  = get_beat_grid("sessions/abc/audio/original.mp3")
    energy = get_energy_curve("sessions/abc/audio/original.mp3")
    segs   = get_segments("sessions/abc/audio/original.mp3", k=10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Analysis sample rate — 22050 is standard for librosa; faster than native 44100
_SR = 22050

# Cache: resolved path string → analysis dict
_cache: dict[str, dict] = {}


class BeatPoint(TypedDict):
    beat_s: float
    bar_number: int
    beat_in_bar: int


class EnergyPoint(TypedDict):
    t_s: float
    rms: float


class Segment(TypedDict):
    start_s: float
    end_s: float
    label: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _key_mode(chroma_mean: np.ndarray) -> tuple[str, str]:
    """Estimate musical key and mode from mean chroma vector."""
    best_score = -np.inf
    best_key = "C"
    best_mode = "major"
    for i in range(12):
        major_corr = np.corrcoef(np.roll(_MAJOR, i), chroma_mean)[0, 1]
        minor_corr = np.corrcoef(np.roll(_MINOR, i), chroma_mean)[0, 1]
        if major_corr > best_score:
            best_score, best_key, best_mode = major_corr, _NOTE_NAMES[i], "major"
        if minor_corr > best_score:
            best_score, best_key, best_mode = minor_corr, _NOTE_NAMES[i], "minor"
    return best_key, best_mode


def _beats_librosa(y: np.ndarray, sr: int) -> tuple[list[BeatPoint], float]:
    """Beat tracking via librosa (assumes 4/4 time)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = round(float(tempo[0] if hasattr(tempo, "__len__") else tempo), 1)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    beats: list[BeatPoint] = [
        {
            "beat_s":      round(t, 3),
            "bar_number":  i // 4,
            "beat_in_bar": (i % 4) + 1,
        }
        for i, t in enumerate(beat_times)
    ]
    return beats, bpm


def _compute(audio_path: str) -> dict:
    """Load audio and compute all analysis features. Results are cached."""
    logger.info("Loading audio for analysis: %s", audio_path)
    y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info("Audio loaded: %.1fs at %dHz", duration, sr)

    # --- BPM + beats --------------------------------------------------------
    beats, bpm = _beats_librosa(y, sr)
    logger.info("Beats: %.1f BPM, %d beats detected", bpm, len(beats))

    # --- RMS energy at 10 Hz ------------------------------------------------
    hop_rms = sr // 10                      # 2205 samples → exactly 10 frames/s
    rms_raw = librosa.feature.rms(y=y, hop_length=hop_rms)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_raw)), sr=sr, hop_length=hop_rms)
    energy: list[EnergyPoint] = [
        {"t_s": round(float(t), 2), "rms": round(float(r), 6)}
        for t, r in zip(rms_times, rms_raw)
    ]

    # --- Agglomerative structural segments (default k) ----------------------
    hop_seg = 512
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_seg)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    k_default = max(4, min(16, int(duration / 20)))   # ~1 boundary per 20s, clamped 4–16
    bound_frames = librosa.segment.agglomerative(mel_db, k_default)
    bound_frames = bound_frames[bound_frames > 0]     # drop any frame-0 boundary
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_seg)
    seg_starts = np.concatenate([[0.0], bound_times])
    seg_ends   = np.concatenate([bound_times, [duration]])
    segments: list[Segment] = [
        {
            "start_s": round(float(s), 2),
            "end_s":   round(float(e), 2),
            "label":   f"Section {i + 1}",
        }
        for i, (s, e) in enumerate(zip(seg_starts, seg_ends))
    ]
    logger.info("Segments: %d structural sections detected", len(segments))

    # --- Key / mode ---------------------------------------------------------
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, mode = _key_mode(chroma.mean(axis=1))
    logger.info("Key: %s %s", key, mode)

    return {
        "bpm":      round(bpm, 1),
        "key":      key,
        "mode":     mode,
        "duration": round(duration, 2),
        "beats":    beats,
        "energy":   energy,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(audio_path: str | Path) -> dict:
    """
    Run full librosa analysis on an audio file.

    Returns a dict with keys:
        bpm (float), key (str), mode (str), duration (float),
        beats (list[BeatPoint]), energy (list[EnergyPoint]), segments (list[Segment])

    Results are cached by resolved path — subsequent calls are instant.
    """
    key = str(Path(audio_path).resolve())
    if key not in _cache:
        _cache[key] = _compute(key)
    return _cache[key]


def get_beat_grid(audio_path: str | Path) -> list[BeatPoint]:
    """Return beat timestamps with bar/beat-in-bar position."""
    return analyze(audio_path)["beats"]


def get_energy_curve(audio_path: str | Path) -> list[EnergyPoint]:
    """Return RMS energy timeline at 10 Hz resolution."""
    return analyze(audio_path)["energy"]


def get_segments(
    audio_path: str | Path,
    k: int | None = None,
) -> list[Segment]:
    """
    Return agglomerative structural segment boundaries.

    Args:
        k: Number of boundaries to detect. If None, auto-selected based on duration
           (~1 boundary per 20s, clamped to 4–16). Pass the number of Suno sections
           for tighter matching to song structure.
    """
    if k is None:
        return analyze(audio_path)["segments"]

    # k differs from cached default — recompute segments only
    resolved = str(Path(audio_path).resolve())
    data = analyze(audio_path)          # ensures audio is loaded + cached
    sr = _SR
    duration = data["duration"]

    y, _ = librosa.load(resolved, sr=sr, mono=True)
    hop_seg = 512
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_seg)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    bound_frames = librosa.segment.agglomerative(mel_db, k)
    bound_frames = bound_frames[bound_frames > 0]     # drop any frame-0 boundary
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_seg)
    seg_starts = np.concatenate([[0.0], bound_times])
    seg_ends   = np.concatenate([bound_times, [duration]])
    return [
        {
            "start_s": round(float(s), 2),
            "end_s":   round(float(e), 2),
            "label":   f"Section {i + 1}",
        }
        for i, (s, e) in enumerate(zip(seg_starts, seg_ends))
    ]
