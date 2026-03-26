"""
config.py
Load .env and expose typed settings.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv optional — env vars may be set by the OS


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# LLM
AGENT_URL   = _get("AGENT_URL",   "http://127.0.0.1:8000")
AGENT_MODEL = _get("AGENT_MODEL", "qwen3.5-35b")
OMNI_MODEL  = _get("OMNI_MODEL",  "Qwen2.5-Omni-7B")

# Aligner — stable-ts (Whisper large-v3), no path config needed (auto-downloads)
ALIGNER_MODEL_SIZE = _get("ALIGNER_MODEL_SIZE", "large-v3")

# ComfyUI
COMFYUI_URL  = _get("COMFYUI_URL", "http://127.0.0.1:8188")
# COMFYUI_DIR is the root ComfyUI folder; input/output/models are derived from it
# unless overridden individually.
COMFYUI_DIR        = Path(_get("COMFYUI_DIR", "C:/ComfyUI"))
COMFYUI_INPUT_DIR  = Path(_get("COMFYUI_INPUT_DIR",  str(COMFYUI_DIR / "input")))
COMFYUI_OUTPUT_DIR = Path(_get("COMFYUI_OUTPUT_DIR", str(COMFYUI_DIR / "output")))
COMFYUI_MODEL_DIR  = Path(_get("COMFYUI_MODEL_DIR",  str(COMFYUI_DIR / "models")))

# Service
SERVICE_PORT = int(_get("SERVICE_PORT", "8001"))
SESSION_DIR  = Path(_get("SESSION_DIR", "./sessions"))



# ACEStep local generator
ACESTEP_URL      = _get("ACESTEP_URL",      "http://127.0.0.1:8002")
# Dedicated venv for ACEStep — created by scripts/setup_acestep.bat|sh
# Never installed into FADE's own venv.
ACESTEP_VENV_DIR = _get("ACESTEP_VENV_DIR", "./acestep_venv")

# Defaults
DEFAULT_ORIENTATION = _get("DEFAULT_ORIENTATION", "landscape")  # portrait | landscape
DEFAULT_FPS         = int(_get("DEFAULT_FPS",  "25"))
SCENE_MIN_SECONDS   = float(_get("SCENE_MIN_SECONDS", "3"))
SCENE_MAX_SECONDS   = float(_get("SCENE_MAX_SECONDS", "20"))
