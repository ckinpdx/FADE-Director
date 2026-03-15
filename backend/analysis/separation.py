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
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs_ft"


def separate(
    audio_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    """
    Separate vocals from a music file using Demucs CLI.

    Args:
        audio_path: Input audio file. MP3, WAV, FLAC, etc.
        out_dir:    Directory where vocals.wav and instrumental.wav will be written.
                    Created if it does not exist.

    Returns:
        Dict with keys "vocals" and "instrumental", values are absolute Paths.
    """
    audio_path = Path(audio_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demucs CLI writes to: <tmp_dir>/<model>/<stem_name>/<filename>.wav
    tmp_dir = out_dir / "_demucs_tmp"

    logger.info("Running Demucs '%s' on '%s' ...", DEMUCS_MODEL, audio_path.name)
    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", DEMUCS_MODEL,
            "--device", "cuda",
            "-o", str(tmp_dir),
            str(audio_path),
        ],
        check=True,
    )

    stem_dir = tmp_dir / DEMUCS_MODEL / audio_path.stem
    raw_vocals = stem_dir / "vocals.wav"
    raw_no_vocals = stem_dir / "no_vocals.wav"

    vocals_path = out_dir / "vocals.wav"
    instrumental_path = out_dir / "instrumental.wav"

    shutil.move(str(raw_vocals), str(vocals_path))
    shutil.move(str(raw_no_vocals), str(instrumental_path))
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    logger.info("Saved vocals:       %s", vocals_path)
    logger.info("Saved instrumental: %s", instrumental_path)

    return {
        "vocals": vocals_path,
        "instrumental": instrumental_path,
    }
