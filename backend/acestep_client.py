"""
acestep_client.py
HTTP client for the ACE-Step 1.5 API server.

Model is always acestep-v15-sft (50 steps, guidance_scale 4.0).
lm_backend is always "pt" — vllm is not available on Windows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from backend import config

logger = logging.getLogger(__name__)

# SFT model settings — quality over speed, no tradeoff
_INFERENCE_STEPS  = 50
_GUIDANCE_SCALE   = 4.0
_LM_BACKEND       = "pt"   # vllm unavailable on Windows


async def generate(params: dict) -> str:
    """
    Submit a text2music generation job.
    params keys: caption, lyrics, bpm, key, time_signature, duration (seconds as str/float)
    Returns: task_id string
    """
    payload: dict = {
        "prompt":          params.get("caption", ""),
        "lyrics":          params.get("lyrics",  ""),
        "audio_duration":  float(params.get("duration", 90)),
        "inference_steps": _INFERENCE_STEPS,
        "guidance_scale":  _GUIDANCE_SCALE,
        "audio_format":    "wav",
        "batch_size":      1,
        "use_random_seed": True,
        "thinking":        True,
        "task_type":       "text2music",
        "lm_backend":      _LM_BACKEND,
    }

    if params.get("bpm"):
        try:
            payload["bpm"] = int(params["bpm"])
        except (ValueError, TypeError):
            pass

    if params.get("key"):
        payload["key_scale"] = params["key"].strip()

    if params.get("time_signature"):
        payload["time_signature"] = params["time_signature"].strip()

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{config.ACESTEP_URL}/release_task", json=payload)
        r.raise_for_status()
        data = r.json()

    task_id: str = data["data"]["task_id"]
    logger.info("ACEStep job submitted: %s", task_id)
    return task_id


async def poll(task_id: str) -> dict | None:
    """
    Check job status.
    Returns result dict if complete, None if still running.
    Raises RuntimeError on failure.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{config.ACESTEP_URL}/query_result",
            json={"task_id_list": [task_id]},
        )
        r.raise_for_status()
        items = r.json()["data"]

    item   = items[0]
    status = item["status"]  # 0=running, 1=done, 2=failed

    if status == 0:
        return None
    if status == 2:
        raise RuntimeError(f"ACEStep generation failed: {item.get('progress_text', 'unknown error')}")

    # result is JSON-encoded as a string — must json.loads() it
    results = json.loads(item["result"])
    return results[0]   # {"file": "/v1/audio?path=...", "metas": {...}}


async def download_audio(file_path: str, dest: Path) -> None:
    """
    Download audio from the ACEStep server to dest.
    file_path is the value of result["file"], e.g. "/v1/audio?path=%2Ftmp%2F...wav"
    """
    url = f"{config.ACESTEP_URL}{file_path}"
    logger.info("Downloading ACEStep audio: %s", url)
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)
    logger.info("Saved take to %s (%d bytes)", dest, dest.stat().st_size)
