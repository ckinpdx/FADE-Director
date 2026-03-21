"""
acestep_server.py — standalone FastAPI server for ACE-Step 1.5 generation.

Run with the ACEStep venv Python, NOT FADE's venv:
    acestep_venv/Scripts/python.exe backend/acestep_server.py

Env vars:
    ACESTEP_API_PORT        (default 8002)
    ACESTEP_CHECKPOINT_DIR  (default "" — auto-downloads to ~/.cache/acestep)
    ACESTEP_MODEL_CONFIG    (default "acestep-v15-sft")
    ACESTEP_OUTPUT_DIR      (default ./sessions/acestep/audio)

Exposes the task-queue API that acestep_client.py calls:
    POST /release_task    — queue a generation job, returns task_id immediately
    POST /query_result    — poll status of one or more tasks
    GET  /v1/audio        — serve a completed audio file by path
    GET  /health          — liveness check
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

_PORT         = int(os.environ.get("ACESTEP_API_PORT",      "8002"))
_CKPT_DIR     = os.environ.get("ACESTEP_CHECKPOINT_DIR",  "")  # "" → auto-download
_MODEL_CONFIG = os.environ.get("ACESTEP_MODEL_CONFIG",    "acestep-v15-sft")
_OUTPUT_DIR   = Path(os.environ.get("ACESTEP_OUTPUT_DIR", "./sessions/acestep/audio"))

# ---------------------------------------------------------------------------
# Pipeline singleton — loaded once at startup
# ---------------------------------------------------------------------------

_handler:    Any = None
_load_error: str | None = None


def _setup_ffmpeg_dlls() -> None:
    """
    PyAV ships FFmpeg DLLs with hashed names (avcodec-62-<hash>.dll).
    torchcodec links against standard names (avcodec-62.dll).
    Create standard-named copies so the DLL loader can find them.
    """
    import re
    import shutil
    try:
        import av as _av
        av_libs = Path(_av.__file__).parent.parent / "av.libs"
        if not av_libs.exists():
            logger.warning("av.libs not found at %s", av_libs)
            return
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(av_libs))
        os.environ["PATH"] = str(av_libs) + os.pathsep + os.environ.get("PATH", "")
        hashed = re.compile(r"^(.+)-([0-9a-f]{32})(\.dll)$", re.IGNORECASE)
        for dll in av_libs.glob("*.dll"):
            m = hashed.match(dll.name)
            if m:
                standard = av_libs / (m.group(1) + m.group(3))
                if not standard.exists():
                    shutil.copy2(dll, standard)
        logger.info("FFmpeg DLLs ready at %s", av_libs)
    except Exception as exc:
        logger.warning("FFmpeg DLL setup failed: %s", exc)


def _load_pipeline() -> None:
    """Load model in a background thread so uvicorn can start immediately."""
    global _handler, _load_error
    try:
        _setup_ffmpeg_dlls()
        logger.info("Loading ACEStep 1.5 handler (config=%r, checkpoint_dir=%r)…",
                    _MODEL_CONFIG, _CKPT_DIR or "auto")
        from acestep.handler import AceStepHandler
        h = AceStepHandler()
        status_msg, success = h.initialize_service(
            project_root=_CKPT_DIR if _CKPT_DIR else "",
            config_path=_MODEL_CONFIG,
            device="auto",
        )
        if not success:
            raise RuntimeError(f"initialize_service failed: {status_msg}")
        _handler = h
        logger.info("ACEStep 1.5 handler ready: %s", status_msg)
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("ACEStep handler failed to load: %s", exc)


# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------

# status: 0=running, 1=done, 2=failed
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _run_task(task_id: str, params: dict) -> None:
    """Run in a background thread. Updates _tasks on completion/failure."""
    try:
        import torchaudio

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"{task_id}.wav"

        caption         = params.get("prompt", "") or ""
        lyrics_text     = params.get("lyrics",  "") or ""
        audio_duration  = float(params.get("audio_duration", 90))
        inference_steps = int(params.get("inference_steps", 60))
        guidance_scale  = float(params.get("guidance_scale", 7.0))

        bpm_val = None
        if params.get("bpm"):
            try:
                bpm_val = int(params["bpm"])
            except (ValueError, TypeError):
                pass

        key_val  = params.get("key_scale")     or ""
        time_sig = params.get("time_signature") or ""

        result = _handler.generate_music(
            captions=caption,
            lyrics=lyrics_text,
            bpm=bpm_val,
            key_scale=key_val,
            time_signature=time_sig,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            audio_duration=audio_duration,
            use_random_seed=True,
            seed=-1,
            shift=1.0,
            infer_method="ode",
        )

        if not result.get("success"):
            raise RuntimeError(result.get("error") or result.get("status_message") or "generation failed")

        audios = result.get("audios", [])
        if not audios:
            raise RuntimeError("generate_music returned no audio tensors")

        audio_tensor = audios[0]["tensor"]   # shape [channels, samples]
        sample_rate  = audios[0]["sample_rate"]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        torchaudio.save(str(out_path), audio_tensor, sample_rate)
        logger.info("Saved take to %s", out_path)

        payload = json.dumps([{"file": f"/v1/audio?path={out_path}", "metas": {}}])
        with _tasks_lock:
            _tasks[task_id] = {"status": 1, "result": payload, "progress_text": "done"}

    except Exception as e:
        logger.exception("ACEStep task %s failed: %s", task_id, e)
        with _tasks_lock:
            _tasks[task_id] = {"status": 2, "result": None, "progress_text": str(e)}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FADE ACEStep Server")


@app.get("/health")
def health():
    if _handler is not None:
        return {"ok": True,  "status": "ready"}
    if _load_error:
        return {"ok": False, "status": "error", "detail": _load_error}
    return {"ok": False, "status": "loading"}


class ReleaseTaskRequest(BaseModel):
    prompt:          str   = ""
    lyrics:          str   = ""
    audio_duration:  float = 90.0
    inference_steps: int   = 60
    guidance_scale:  float = 7.0
    bpm:             int   | None = None
    key_scale:       str   | None = None
    time_signature:  str   | None = None
    # kept for forward compat — unused in v1.5
    audio_format:    str   = "wav"
    batch_size:      int   = 1
    use_random_seed: bool  = True
    task_type:       str   = "text2music"


@app.post("/release_task")
def release_task(req: ReleaseTaskRequest):
    if _handler is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {"status": 0, "result": None, "progress_text": "queued"}
    t = threading.Thread(target=_run_task, args=(task_id, req.model_dump()), daemon=True)
    t.start()
    return {"data": {"task_id": task_id}}


class QueryRequest(BaseModel):
    task_id_list: list[str]


@app.post("/query_result")
def query_result(req: QueryRequest):
    results = []
    with _tasks_lock:
        for tid in req.task_id_list:
            entry = _tasks.get(tid, {"status": 2, "result": None, "progress_text": "unknown task"})
            results.append({"task_id": tid, **entry})
    return {"data": results}


@app.get("/v1/audio")
def serve_audio(path: str):
    """Serve a generated audio file by its absolute path."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(p), media_type="audio/wav")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load model in background — uvicorn starts immediately so health polls succeed.
    # Health returns ok=false while loading, ok=true once ready.
    t = threading.Thread(target=_load_pipeline, daemon=True)
    t.start()
    uvicorn.run(app, host="127.0.0.1", port=_PORT, log_level="info")
