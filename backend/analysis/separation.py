"""
separation.py
Vocal separation via Demucs htdemucs_6s model.

Produces vocals.wav and instrumental.wav in the target directory.
Demucs uses FFmpeg internally to load non-WAV inputs (MP3, FLAC, etc.).
Output is saved with soundfile (no FFmpeg dependency for writing).

Usage:
    from backend.analysis.separation import separate

    paths = separate(
        audio_path="sessions/abc/audio/original.mp3",
        out_dir="sessions/abc/audio",
    )
    # paths = {"vocals": Path(...), "instrumental": Path(...)}
"""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf
import torch

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs_ft"


def separate(
    audio_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    """
    Separate vocals from a music file using Demucs.

    Args:
        audio_path: Input audio file. MP3, WAV, FLAC, etc. (Demucs uses FFmpeg to load).
        out_dir:    Directory where vocals.wav and instrumental.wav will be written.
                    Created if it does not exist.

    Returns:
        Dict with keys "vocals" and "instrumental", values are absolute Paths.
    """
    from demucs.api import Separator

    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = out_dir / "vocals.wav"
    instrumental_path = out_dir / "instrumental.wav"

    logger.info("Loading Demucs model '%s' ...", DEMUCS_MODEL)
    sep = Separator(DEMUCS_MODEL)

    logger.info("Separating '%s' ...", audio_path.name)
    _, stems = sep.separate_audio_file(str(audio_path))
    # stems: {"drums": Tensor, "bass": Tensor, "other": Tensor, "vocals": Tensor}
    # shape: (channels, samples), float32

    vocals = stems["vocals"].numpy().T          # (samples, channels)
    non_vocal = sum(                             # sum remaining stems
        v for k, v in stems.items() if k != "vocals"
    )
    if isinstance(non_vocal, torch.Tensor):
        non_vocal = non_vocal.numpy().T
    else:
        non_vocal = non_vocal.T

    sr = sep.samplerate

    sf.write(str(vocals_path), vocals, sr, subtype="PCM_16")
    logger.info("Saved vocals:       %s  (%.1fs)", vocals_path, len(vocals) / sr)

    sf.write(str(instrumental_path), non_vocal, sr, subtype="PCM_16")
    logger.info("Saved instrumental: %s  (%.1fs)", instrumental_path, len(non_vocal) / sr)

    # Explicitly release Demucs GPU memory before returning
    del sep, stems, vocals, non_vocal
    torch.cuda.empty_cache()
    logger.info("Demucs VRAM released.")

    return {
        "vocals": vocals_path.resolve(),
        "instrumental": instrumental_path.resolve(),
    }
