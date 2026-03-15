"""
render.py
Final export: concatenate approved scene MP4s and mix with original audio.

Uses ffmpeg (must be on PATH). The concat demuxer is used for lossless stream
copy — no re-encoding of video. Audio is mixed fresh from the original source
so quality is not degraded through the video pipeline.

Usage:
    from backend.render import export

    output_path = export(
        clip_paths=["sessions/abc/videos/scene_1_00001_.mp4",
                    "sessions/abc/videos/scene_2_00001_.mp4"],
        audio_path="sessions/abc/audio/original.mp3",
        output_path="sessions/abc/final.mp4",
    )
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class RenderError(Exception):
    """Raised when ffmpeg exits with a non-zero code."""


def export(
    clip_paths:  list[str | Path],
    audio_path:  str | Path,
    output_path: str | Path,
    audio_offset_s: float = 0.0,
) -> Path:
    """
    Concatenate scene clips and mix with original audio track.

    Video streams are stream-copied (no re-encode). Audio is taken from
    audio_path, encoded to AAC 192k, and trimmed to match the total clip
    duration with -shortest.

    Args:
        clip_paths:     Ordered list of approved scene MP4 files.
        audio_path:     Original music file (MP3, WAV, FLAC, etc.).
        output_path:    Destination MP4 path (created; parent must exist).
        audio_offset_s: Start offset into the audio track in seconds.
                        Use when the first scene starts mid-song
                        (set to the start_s of scene 1).

    Returns:
        Resolved Path to the output file.

    Raises:
        ValueError:   If clip_paths is empty.
        RenderError:  If ffmpeg exits with a non-zero return code.
    """
    if not clip_paths:
        raise ValueError("clip_paths must not be empty.")

    clip_paths  = [Path(p).resolve() for p in clip_paths]
    audio_path  = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Rendering %d clip(s) + audio -> %s", len(clip_paths), output_path
    )

    # Write a concat list file that ffmpeg's concat demuxer reads
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        concat_file = f.name
        for p in clip_paths:
            # ffmpeg concat format requires forward slashes and escaped special chars
            safe = str(p).replace("\\", "/")
            f.write(f"file '{safe}'\n")

    logger.debug("Concat list:\n%s", Path(concat_file).read_text(encoding="utf-8"))

    try:
        cmd = [
            "ffmpeg", "-y",
            # Video: concat demuxer (stream copy, no re-encode)
            "-f", "concat", "-safe", "0", "-i", concat_file,
            # Audio: original track with optional start offset
            "-ss", str(audio_offset_s), "-i", str(audio_path),
            # Output: copy video, encode audio to AAC
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            # Trim output to the shorter of video vs audio
            "-shortest",
            str(output_path),
        ]

        logger.info("ffmpeg cmd: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", result.stderr[-3000:])
            raise RenderError(
                f"ffmpeg exited with code {result.returncode}.\n"
                f"stderr (last 3000 chars):\n{result.stderr[-3000:]}"
            )

        logger.info("Export complete: %s  (%.1f MB)",
                    output_path,
                    output_path.stat().st_size / 1_048_576)
        return output_path

    finally:
        os.unlink(concat_file)
