"""
aligner.py
Forced word-level alignment of lyrics against isolated vocals audio.

Uses stable-ts (Whisper large-v3) — forced alignment mode only, no transcription.
The Whisper model is loaded once and cached for the lifetime of the process.

Usage:
    from backend.analysis.aligner import align

    words = align(
        vocals_path="sessions/abc/audio/vocals.wav",
        lyrics=raw_lyrics_with_suno_tags,
    )
    # words = [{"word": "It", "start_s": 0.64, "end_s": 1.0}, ...]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# Module-level model cache — loaded once on first call, reused for all subsequent calls
_model = None


class WordTimestamp(TypedDict):
    word: str
    start_s: float
    end_s: float


def strip_suno_tags(text: str) -> str:
    """Remove lines that are purely Suno structural/direction tags, e.g. [Verse 1]."""
    return "\n".join(
        line for line in text.splitlines()
        if not re.fullmatch(r"\[.*?\]", line.strip())
    ).strip()


def _get_model(model_size: str):
    global _model
    if _model is None:
        import stable_whisper
        logger.info("Loading stable-ts model '%s' ...", model_size)
        _model = stable_whisper.load_model(model_size)
        logger.info("Aligner model ready.")
    return _model


def release_model() -> None:
    """Unload the Whisper model from VRAM. Call before handing off to llama-swap."""
    global _model
    if _model is not None:
        try:
            import torch
            _model.cpu()
            del _model
            _model = None
            torch.cuda.empty_cache()
            logger.info("Whisper model released from VRAM.")
        except Exception as e:
            logger.warning("Whisper release failed (non-fatal): %s", e)
            _model = None


def align(
    vocals_path: str | Path,
    lyrics: str,
    model_size: str = "large-v3",
    language: str = "en",
) -> list[WordTimestamp]:
    """
    Forced-align lyrics against isolated vocals audio.

    Args:
        vocals_path: Path to vocals WAV produced by Demucs separation.
        lyrics:      Raw lyrics text. Suno section tags ([Verse 1], etc.) are
                     stripped automatically before alignment.
        model_size:  Whisper model variant. 'large-v3' is recommended.
        language:    ISO 639-1 language code passed to Whisper.

    Returns:
        Flat list of WordTimestamp dicts, one per word, in chronological order.
        Times are in seconds, rounded to millisecond precision.

    Raises:
        ValueError: If no lyric text remains after stripping tags.
    """
    clean = strip_suno_tags(lyrics)
    if not clean:
        raise ValueError("No lyric text remaining after stripping Suno tags.")

    word_count = len(clean.split())
    logger.info("Aligning %d words against %s ...", word_count, vocals_path)

    model = _get_model(model_size)
    result = model.align(str(vocals_path), clean, language=language)
    raw_words = result.all_words()

    words: list[WordTimestamp] = [
        {
            "word": w.word.strip(),
            "start_s": round(w.start, 3),
            "end_s": round(w.end, 3),
        }
        for w in raw_words
    ]

    logger.info("Alignment complete: %d/%d words placed.", len(words), word_count)
    _log_gap_warnings(words)
    return words


def _log_gap_warnings(words: list[WordTimestamp], threshold_s: float = 10.0) -> None:
    """Warn about unusually large gaps that may indicate alignment failures."""
    for i in range(1, len(words)):
        gap = words[i]["start_s"] - words[i - 1]["end_s"]
        if gap > threshold_s:
            logger.warning(
                "Large gap %.1fs after '%s' (%.2fs) before '%s' (%.2fs) — "
                "may indicate a missed section.",
                gap,
                words[i - 1]["word"],
                words[i - 1]["end_s"],
                words[i]["word"],
                words[i]["start_s"],
            )
