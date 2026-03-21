"""
acestep_process.py
Manage the ACE-Step 1.5 API server as a subprocess.

Lifecycle: start when user opens the Make a Song page, stop when they leave.
The server auto-downloads models on first run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx

from backend import config


def _acestep_python() -> Path:
    """
    Resolve the Python interpreter from the dedicated ACEStep venv.
    Raises RuntimeError with a clear install instruction if the venv doesn't exist.
    """
    venv = Path(config.ACESTEP_VENV_DIR).resolve()
    ext = ".exe" if sys.platform == "win32" else ""
    python = venv / ("Scripts" if sys.platform == "win32" else "bin") / f"python{ext}"
    if not python.exists():
        raise RuntimeError(
            f"ACEStep venv not found at {venv}. "
            "Run scripts/setup_acestep.bat (Windows) or scripts/setup_acestep.sh (Linux/Mac) "
            "to create the dedicated ACEStep environment."
        )
    return python


# Path to our standalone server script (sibling of this file)
_SERVER_SCRIPT = Path(__file__).parent / "acestep_server.py"

logger = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None
_start_lock = asyncio.Lock()


def _port_from_url(url: str) -> str:
    """Extract port string from a URL like http://127.0.0.1:8002."""
    try:
        return url.rstrip("/").split(":")[-1]
    except Exception:
        return "8002"


async def start() -> None:
    """
    Launch the ACE-Step API server if it isn't already running.
    Returns once the health endpoint responds (up to 120s).
    """
    global _proc

    async with _start_lock:
        if await health_check():
            logger.info("ACEStep already running")
            return

        port = _port_from_url(config.ACESTEP_URL)

        env = os.environ.copy()
        env["ACESTEP_API_PORT"] = port
        # ACESTEP_CHECKPOINT_DIR: "" means auto-download to ~/.cache/ace-step
        env["ACESTEP_CHECKPOINT_DIR"] = ""

        logger.info("Starting ACEStep server (port=%s)", port)

        python = _acestep_python()
        log_path = Path(__file__).parent.parent / "acestep_server.log"
        log_file = open(log_path, "w", buffering=1)
        _proc = subprocess.Popen(
            [str(python), str(_SERVER_SCRIPT)],
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        logger.info("ACEStep subprocess pid=%d — waiting for health…", _proc.pid)

        # Poll health up to 120s (model download may take a while on first run)
        for _ in range(60):
            await asyncio.sleep(2)
            if _proc.poll() is not None:
                raise RuntimeError(
                    f"ACEStep process exited early (rc={_proc.returncode}). "
                    "Run scripts/setup_acestep.bat to reinstall the ACEStep environment."
                )
            if await health_check():
                logger.info("ACEStep ready")
                return

        _proc.terminate()
        _proc = None
        raise RuntimeError(
            "ACEStep did not become healthy within 120s. "
            "Check ACESTEP_URL and that the server started correctly."
        )


async def stop() -> None:
    """Terminate the ACEStep subprocess if FADE owns it."""
    global _proc
    if _proc is None:
        return
    logger.info("Stopping ACEStep (pid=%d)", _proc.pid)
    _proc.terminate()
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: _proc.wait(timeout=10))
    except subprocess.TimeoutExpired:
        _proc.kill()
    _proc = None
    logger.info("ACEStep stopped")


async def health_check() -> bool:
    """Return True if the ACEStep server is responding."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{config.ACESTEP_URL}/health")
            return r.status_code == 200
    except Exception:
        return False
