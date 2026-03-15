"""
client.py
Low-level ComfyUI API client.

Responsibilities:
  - submit()       — POST /prompt, return prompt_id
  - poll()         — poll /history/{id} until complete, return outputs dict
  - get_image_path() — resolve PNG path from SaveImage outputs
  - get_video_path() — resolve MP4 path from VHS SaveVideo outputs
  - stage_audio()  — copy audio file to ComfyUI input dir for workflow use

All paths are resolved against COMFYUI_OUTPUT_DIR / COMFYUI_INPUT_DIR from env.

Environment:
    COMFYUI_URL        — ComfyUI server base URL (default: http://127.0.0.1:8188)
    COMFYUI_OUTPUT_DIR — ComfyUI output directory (default: C:/ComfyUI/output)
    COMFYUI_INPUT_DIR  — ComfyUI input directory  (default: C:/ComfyUI/input)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

COMFYUI_URL        = os.getenv("COMFYUI_URL",        "http://127.0.0.1:8188")
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", "C:/ComfyUI/output"))
COMFYUI_INPUT_DIR  = Path(os.getenv("COMFYUI_INPUT_DIR",  "C:/ComfyUI/input"))

# Stable client ID for this process lifetime — identifies requests in ComfyUI logs
_CLIENT_ID = str(uuid.uuid4())

_HTTP = httpx.Client(timeout=30.0)


class ComfyUIError(Exception):
    """Raised when ComfyUI reports a workflow execution error."""


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit(workflow: dict) -> str:
    """
    Submit a patched workflow to ComfyUI.

    Args:
        workflow: Fully-patched workflow dict (output of patch.apply()).

    Returns:
        prompt_id string — use with poll() to wait for completion.

    Raises:
        httpx.HTTPError: On network or HTTP errors.
    """
    payload = {"prompt": workflow, "client_id": _CLIENT_ID}
    resp = _HTTP.post(f"{COMFYUI_URL}/prompt", json=payload)
    resp.raise_for_status()
    data = resp.json()
    prompt_id = data["prompt_id"]
    logger.info("Submitted prompt_id=%s  queue_pos=%s", prompt_id, data.get("number", "?"))
    return prompt_id


# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def poll(
    prompt_id:     str,
    poll_interval: float = 2.0,
    timeout_s:     float = 3600.0,
) -> dict[str, Any]:
    """
    Poll /history/{prompt_id} until the workflow completes or errors.

    Args:
        prompt_id:     ID returned by submit().
        poll_interval: Seconds between history checks (default 2s).
        timeout_s:     Maximum total wait time in seconds (default 1 hour).
                       HuMo 17B passes can take 30–40 min on first run.

    Returns:
        The outputs dict from ComfyUI history:
        { node_id: { "images"|"gifs": [{filename, subfolder, type}] } }

    Raises:
        ComfyUIError:   If ComfyUI reports status "error".
        TimeoutError:   If timeout_s is exceeded before completion.
    """
    deadline = time.monotonic() + timeout_s
    elapsed  = 0.0

    while time.monotonic() < deadline:
        resp = _HTTP.get(f"{COMFYUI_URL}/history/{prompt_id}")
        resp.raise_for_status()
        history = resp.json()

        if prompt_id in history:
            entry  = history[prompt_id]
            status = entry.get("status", {})

            if status.get("completed"):
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise ComfyUIError(
                        f"ComfyUI execution failed for prompt {prompt_id}. "
                        f"Messages: {messages}"
                    )
                logger.info(
                    "prompt_id=%s  completed in %.0fs", prompt_id, elapsed
                )
                return entry.get("outputs", {})

        time.sleep(poll_interval)
        elapsed += poll_interval
        if int(elapsed) % 30 == 0:
            logger.info("Waiting for prompt_id=%s  (%.0fs elapsed)", prompt_id, elapsed)

    raise TimeoutError(
        f"ComfyUI prompt {prompt_id} did not complete within {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def get_image_path(outputs: dict, save_node_id: str) -> Path:
    """
    Extract the saved PNG path from a SaveImage node output.

    Args:
        outputs:      The outputs dict returned by poll().
        save_node_id: The node ID of the SaveImage node (e.g. "177").

    Returns:
        Absolute Path to the generated PNG on disk.

    Raises:
        KeyError:  If the node ID is not present in outputs.
        FileNotFoundError: If the resolved path does not exist.
    """
    node_out = outputs[save_node_id]
    img_info = node_out["images"][0]
    path = COMFYUI_OUTPUT_DIR / img_info["subfolder"] / img_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Expected ComfyUI image output not found: {path}")
    logger.info("Image output: %s", path)
    return path.resolve()


def get_video_path(outputs: dict, save_node_id: str) -> Path:
    """
    Extract the saved MP4 path from a VHS SaveVideo node output.

    VideoHelperSuite uses the key "gifs" for all video outputs regardless of
    format (historical naming). Falls back to "videos" if "gifs" is absent.

    Args:
        outputs:      The outputs dict returned by poll().
        save_node_id: The node ID of the SaveVideo node (e.g. "1381").

    Returns:
        Absolute Path to the generated MP4 on disk.
    """
    node_out = outputs[save_node_id]

    # VHS SaveVideo stores file info under "images" with "animated": [true] as a flag.
    # Older / alternative nodes may use "gifs" or "videos".
    if node_out.get("animated") and node_out.get("images"):
        vid_info = node_out["images"][0]
    else:
        vid_list = node_out.get("gifs") or node_out.get("videos") or []
        if not vid_list:
            raise KeyError(
                f"No video output found in node {save_node_id}. "
                f"Keys present: {list(node_out.keys())}"
            )
        vid_info = vid_list[0]

    path = COMFYUI_OUTPUT_DIR / vid_info["subfolder"] / vid_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Expected ComfyUI video output not found: {path}")
    logger.info("Video output: %s", path)
    return path.resolve()


# ---------------------------------------------------------------------------
# Audio staging
# ---------------------------------------------------------------------------

def stage_image(image_path: str | Path, filename: str) -> str:
    """
    Copy a generated PNG to the ComfyUI input directory so a LoadImage node
    can reference it in a subsequent workflow (e.g. T2I output → I2V start_frame).

    Args:
        image_path: Source PNG (typically from get_image_path()).
        filename:   Target filename in ComfyUI input dir, e.g. "session_abc_scene_1.png".

    Returns:
        The filename string — pass this directly to the start_frame workflow patch.
    """
    src  = Path(image_path)
    dest = COMFYUI_INPUT_DIR / filename
    shutil.copy2(str(src), str(dest))
    logger.info("Staged image: %s -> %s", src.name, dest)
    return filename


def stage_audio(audio_path: str | Path, filename: str) -> str:
    """
    Copy an audio file to the ComfyUI input directory.

    Called once per session at session start. The workflow's VHS_LoadAudioUpload
    node references this filename relative to the input dir.

    Args:
        audio_path: Source audio file (original MP3 or WAV from session dir).
        filename:   Target filename in ComfyUI input dir, e.g. "session_abc.mp3".

    Returns:
        The filename string — pass this directly to the audio_file workflow patch.
    """
    src  = Path(audio_path)
    dest = COMFYUI_INPUT_DIR / filename
    shutil.copy2(str(src), str(dest))
    logger.info("Staged audio: %s -> %s", src.name, dest)
    return filename
