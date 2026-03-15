"""
omni.py
Qwen2.5-Omni multimodal calls via llama-swap (OpenAI-compatible endpoint).

Two responsibilities:
  1. analyze_intonation() — listen to isolated vocals + read lyrics → top-level
     genre/subgenre classification + section-level qualitative descriptors
     (delivery, energy, mood) with a lyrics_anchor phrase for fuzzy-matching
     against aligner word timestamps.

  2. describe_image() — read a reference image → detailed text description of
     subject, clothing, lighting, setting, colour palette, mood. Stored as
     reference_description and injected into consistent_elements for all scenes.

Timestamps are NOT requested from Omni — they are hallucinated for audio.
All temporal grounding comes from the forced aligner (aligner.py).

Audio is downsampled to 16 kHz mono before sending (~9 MB base64 for 4 min)
to stay within practical request-body limits.

Environment:
    AGENT_URL   — llama-swap base URL (default: http://127.0.0.1:8000)
    OMNI_MODEL  — model name registered in llama-swap (default: Qwen2.5-Omni-7B)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict

import httpx
import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

AGENT_URL   = os.getenv("AGENT_URL",   "http://127.0.0.1:8000")
OMNI_MODEL  = os.getenv("OMNI_MODEL",  "Qwen2.5-Omni-7B")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5-35b")

_AUDIO_SR    = 16_000   # resample target for Omni audio input
_TIMEOUT     = 180.0   # seconds — model may need to swap in llama-swap
_MAX_CHUNK_S = 100     # max audio seconds per Omni call — llama-server audio encoder
                        # does not reset between requests; must reload model between chunks


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class IntonationSection(TypedDict):
    label:      str  # section type as heard: intro, verse, chorus, bridge, outro, etc.
    intonation: str  # 2-3 sentences: delivery style, technique, phrasing
    energy:     str  # low | medium | high | building | dropping
    mood:       str  # 2-3 word emotional descriptor


class IntonationResult(TypedDict):
    genre:    str                    # top-level genre e.g. "country", "hip-hop"
    subgenre: str                    # more specific e.g. "country pop", "trap"
    sections: list[IntonationSection]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _audio_to_b64(audio_path: str | Path, sr: int = _AUDIO_SR) -> str:
    """Load audio, resample to mono {sr} Hz, return base64-encoded WAV string."""
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return _array_to_b64(y, sr)


def _array_to_b64(y: "np.ndarray", sr: int) -> str:
    """Encode a float32 mono array as base64 PCM_16 WAV."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _image_to_b64(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_string, mime_type) for a PNG or JPG image."""
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = path.read_bytes()
    return base64.b64encode(data).decode(), mime


def _repair_json(text: str) -> str:
    """Apply lightweight repairs for common LLM JSON mistakes."""
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace Python True/False/None with JSON equivalents
    text = re.sub(r"\bTrue\b",  "true",  text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b",  "null",  text)
    return text


def _extract_json(text: str) -> list | dict:
    """
    Extract and parse JSON from Omni response text.
    Handles bare JSON, markdown code blocks, leading/trailing prose,
    think tags, trailing commas, and other common LLM output quirks.
    """
    # Strip <think>...</think> blocks (Qwen reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    def _try_parse(s: str) -> list | dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_repair_json(s))
        except json.JSONDecodeError:
            pass
        return None

    # Try the whole text first
    result = _try_parse(text)
    if result is not None:
        return result

    # Find the first { ... } or [ ... ] block (try larger containers first)
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end   = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            result = _try_parse(text[start : end + 1])
            if result is not None:
                return result

    # Last resort: response was truncated mid-JSON — try to close it and salvage
    # what we have. Handles the common case of Omni hitting max_tokens inside a
    # large sections array.
    start = text.find("{")
    if start != -1:
        fragment = text[start:]
        # Close any open array and object
        closes = ""
        depth_obj = fragment.count("{") - fragment.count("}")
        depth_arr = fragment.count("[") - fragment.count("]")
        # Strip trailing incomplete entry (ends mid-string or mid-field)
        # Find the last complete object in sections array
        last_complete = fragment.rfind("},")
        if last_complete == -1:
            last_complete = fragment.rfind("}")
        if last_complete != -1:
            fragment = fragment[: last_complete + 1]
            depth_obj = fragment.count("{") - fragment.count("}")
            depth_arr = fragment.count("[") - fragment.count("]")
        closes = "]" * max(0, depth_arr) + "}" * max(0, depth_obj)
        salvaged = fragment + closes
        result = _try_parse(salvaged)
        if result is not None:
            logger.warning("Salvaged truncated Omni JSON (%d sections recovered)",
                           len(result.get("sections", [])) if isinstance(result, dict) else 0)
            return result

    logger.error("No valid JSON found in Omni response (full):\n%s", text)
    raise ValueError(f"No valid JSON found in Omni response:\n{text[:400]}")


def _chat(messages: list[dict], timeout: float = _TIMEOUT) -> str:
    """POST to llama-swap chat completions endpoint, return assistant content."""
    url = f"{AGENT_URL}/v1/chat/completions"
    payload = {
        "model":       OMNI_MODEL,
        "messages":    messages,
        "temperature": 0.1,
        "max_tokens":  4096,
    }
    logger.debug("POST %s  model=%s", url, OMNI_MODEL)
    resp = httpx.post(url, json=payload, timeout=timeout)
    if not resp.is_success:
        body = resp.text[:2000]
        logger.error("llama-swap %d for model=%s: %s", resp.status_code, OMNI_MODEL, body)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SECTION_EXAMPLE = (
    '  {\n'
    '    "label": "<section as heard: intro, verse, pre-chorus, chorus, post-chorus, bridge, breakdown, drop, outro>",\n'
    '    "intonation": "<1 sentence: specific vocal technique — chest/falsetto, breathiness, vibrato, melisma, phrasing>",\n'
    '    "energy": "<one of: low, medium, high, building, dropping>",\n'
    '    "mood": "<3-5 evocative words, e.g. \'desperate, raw, defiant\' not just \'emotional\'>"\n'
    '  }'
)

_INTONATION_PROMPT = (
    "Listen carefully to this song. Identify every distinct structural section in the order it appears"
    " — including intro, verses, pre-choruses, choruses, post-choruses, bridges, breakdowns, drops, and outros.\n"
    "For each section, describe the vocal performance with specificity: actual technique (chest voice, falsetto,"
    " breathiness, vibrato, staccato phrasing, melisma, etc.), not generic adjectives.\n"
    "Do NOT guess timestamps. Do NOT pad with filler sections — only list what you actually hear.\n"
    "IMPORTANT: Output at most 12 sections total. If the song repeats a pattern (e.g. verse/chorus/verse/chorus),"
    " keep only the first occurrence of each distinct pattern and note the repetition in the intonation field.\n\n"
    "Return a single JSON object — no other text:\n"
    '{\n'
    '  "genre": "<primary genre, e.g. country, hip-hop, pop, R&B, rock, electronic>",\n'
    '  "subgenre": "<specific subgenre — not just \'pop\' but e.g. \'dark pop\', \'indie folk\', \'trap soul\', \'synthpop\', \'country rap\'>",\n'
    '  "sections": [\n'
    + _SECTION_EXAMPLE + '\n'
    '  ]\n'
    '}'
)


def _chunk_prompt(n: int, total: int, start_s: float, end_s: float) -> str:
    return (
        f"Listen to this audio clip (part {n} of {total}, covering roughly {start_s:.0f}s-{end_s:.0f}s of the song).\n"
        "Identify every distinct structural section you hear in order — verse, pre-chorus, chorus, bridge, breakdown, etc.\n"
        "Describe vocal technique specifically (chest voice, falsetto, breathiness, vibrato, phrasing style).\n"
        "Do NOT guess timestamps. Only list sections actually present in this clip.\n"
        "IMPORTANT: Output at most 8 sections. If a pattern repeats, note it in the intonation field — do not duplicate entries.\n\n"
        "Return a single JSON object — no other text:\n"
        '{\n'
        '  "genre": "<primary genre>",\n'
        '  "subgenre": "<more specific subgenre>",\n'
        '  "sections": [\n'
        + _SECTION_EXAMPLE + '\n'
        '  ]\n'
        '}'
    )


def analyze_intonation(audio_path: str | Path) -> IntonationResult:
    """
    Analyse vocal delivery and music structure by listening to the audio.
    Omni identifies sections (verse/chorus/etc.) from what it hears — no lyrics needed.
    Chunks long songs automatically to stay within Omni's ~150s limit.

    Args:
        audio_path: Full mix audio (original upload).

    Returns:
        IntonationResult with genre, subgenre, and audio-detected sections.
    """
    logger.info("Loading audio for Omni intonation analysis (%s) ...", audio_path)
    y, _ = librosa.load(str(audio_path), sr=_AUDIO_SR, mono=True)
    total_s = len(y) / _AUDIO_SR
    logger.info("Audio duration: %.1fs", total_s)

    chunk_samples = int(_MAX_CHUNK_S * _AUDIO_SR)

    if total_s <= _MAX_CHUNK_S:
        b64 = _array_to_b64(y, _AUDIO_SR)
        logger.info("Single chunk: %.1fMB base64", len(b64) * 3 / 4 / 1_048_576)
        raw = _chat([{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            {"type": "text", "text": _INTONATION_PROMPT},
        ]}])
        return _parse_intonation(raw)

    # Multi-chunk: split, analyze each, merge sections in order
    chunks = [y[i : i + chunk_samples] for i in range(0, len(y), chunk_samples)]
    total_chunks = len(chunks)
    logger.info("Splitting into %d chunks of ~%ds", total_chunks, _MAX_CHUNK_S)

    all_sections: list[IntonationSection] = []
    genre = subgenre = ""

    for i, chunk in enumerate(chunks):
        if i > 0:
            # Force llama-swap to unload Omni by calling the agent model.
            # llama-server's audio encoder doesn't reset between requests in the same
            # model session — reloading Omni is the only way to get a clean state.
            logger.info("Swapping out Omni via trivial %s call ...", AGENT_MODEL)
            try:
                httpx.post(
                    f"{AGENT_URL}/v1/chat/completions",
                    json={"model": AGENT_MODEL, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1},
                    timeout=60.0,
                )
            except Exception as e:
                logger.debug("Model-swap ping failed (non-fatal): %s", e)

        start_s = i * _MAX_CHUNK_S
        end_s   = min((i + 1) * _MAX_CHUNK_S, total_s)
        b64    = _array_to_b64(chunk, _AUDIO_SR)
        prompt = _chunk_prompt(i + 1, total_chunks, start_s, end_s)
        logger.info(
            "Chunk %d/%d (%.0fs-%.0fs, %.1fMB) ...",
            i + 1, total_chunks, start_s, end_s, len(b64) * 3 / 4 / 1_048_576,
        )
        raw    = _chat([{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            {"type": "text", "text": prompt},
        ]}])
        chunk_result = _parse_intonation(raw)
        if not genre:
            genre    = chunk_result["genre"]
            subgenre = chunk_result["subgenre"]
        all_sections.extend(chunk_result["sections"])

    merged = IntonationResult(genre=genre, subgenre=subgenre, sections=all_sections)
    logger.info(
        "Intonation analysis complete: genre=%r subgenre=%r sections=%d (from %d chunks)",
        merged["genre"], merged["subgenre"], len(merged["sections"]), total_chunks,
    )
    return merged


def _parse_intonation(raw: str, _retry: bool = True) -> IntonationResult:
    logger.debug("Omni intonation raw:\n%s", raw[:600])
    try:
        result = _extract_json(raw)
    except ValueError:
        if _retry:
            # Ask the agent model to repair and re-emit only the JSON
            logger.warning("Omni intonation JSON parse failed — asking %s to repair ...", AGENT_MODEL)
            fix_prompt = (
                "The following text should be a JSON object with keys 'genre', 'subgenre', and 'sections'."
                " It may be malformed. Output ONLY the corrected JSON object — no explanation, no markdown.\n\n"
                + raw[:3000]
            )
            try:
                fixed = httpx.post(
                    f"{AGENT_URL}/v1/chat/completions",
                    json={
                        "model":       AGENT_MODEL,
                        "messages":    [{"role": "user", "content": fix_prompt}],
                        "temperature": 0.0,
                        "max_tokens":  2048,
                    },
                    timeout=60.0,
                ).json()["choices"][0]["message"]["content"]
                return _parse_intonation(fixed, _retry=False)
            except Exception as e:
                logger.error("JSON repair attempt failed: %s", e)
        raise

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object from Omni, got: {type(result)}")
    sections = result.get("sections", [])
    if not isinstance(sections, list):
        sections = []
    return IntonationResult(
        genre    = result.get("genre",    ""),
        subgenre = result.get("subgenre", ""),
        sections = [
            IntonationSection(
                label      = s.get("label",      ""),
                intonation = s.get("intonation", ""),
                energy     = s.get("energy",     ""),
                mood       = s.get("mood",       ""),
            )
            for s in sections if isinstance(s, dict)
        ],
    )


def describe_image(image_path: str | Path) -> str:
    """
    Generate a detailed text description of a reference image.

    Used when the user uploads a character, style, or location reference.
    The description is stored as reference_description and injected into
    consistent_elements for all scene prompts.

    Args:
        image_path: Path to PNG or JPG reference image.

    Returns:
        Detailed text description covering subject, appearance, clothing,
        lighting, setting, colour palette, and mood.
    """
    logger.info("Sending reference image to Omni for description (%s) ...", image_path)
    b64, mime = _image_to_b64(image_path)

    prompt = """Describe the person in this reference image for use in AI image generation. Focus entirely on the subject's physical appearance — this will be used to keep the character consistent across many generated scenes.

Respond in exactly this format:

GENDER: [woman / man / non-binary / other — one word or short phrase]
CHARACTER: [physical appearance only — skin tone, face shape, hair colour/length/texture, eye colour, build, any distinctive facial features. 3-4 specific sentences. Do not mention clothing, setting, or mood.]"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    raw = _chat(messages).strip()
    logger.info("Image description raw: %d chars", len(raw))

    # Parse structured fields into a dict
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            if key in ("GENDER", "CHARACTER"):
                fields[key] = value.strip()

    # Fallback: if parsing failed (model didn't follow format), store raw prose
    if not fields:
        logger.warning("describe_image: structured parse failed — storing raw prose")
        return raw

    # Reassemble as labelled prose
    parts = []
    if fields.get("GENDER"):
        parts.append(f"Gender: {fields['GENDER']}")
    if fields.get("CHARACTER"):
        parts.append(f"Character: {fields['CHARACTER']}")
    description = "\n".join(parts)
    logger.info("Image description parsed: %d fields, %d chars", len(fields), len(description))
    return description
