"""
main.py
FastAPI application — WebSocket hub, session management, REST endpoints.

WebSocket events (server → client):
  {"event": "status",        "message": str, "step": str}   — pipeline progress
  {"event": "analysis_done", "music_data": dict, "intonation": list}
  {"event": "segment_done",  "scenes": list, "k": int, "weights": list}
  {"event": "scene_update",  "scene_index": int, "scene": dict}
  {"event": "token",         "text": str}
  {"event": "assistant_done"}
  {"event": "error",         "message": str}
  {"event": "pong"}

WebSocket messages (client → server):
  {"type": "chat",  "message": str}
  {"type": "ping"}

Analysis pipeline (auto-started by POST /sessions on session creation):
  [step 0] Omni describe_image(reference) → reference_description  (if ref uploaded)
  [step 1] Omni analyze_intonation(audio, lyrics) → intonation map
  [step 2] htdemucs → vocals.wav + instrumental.wav
  [step 3] stable-ts align(vocals, lyrics) → word timestamps
  [step 4] librosa analyze(audio) → BPM, key, segments, energy
  All steps push status events. CPU-bound work runs in asyncio thread pool.
  Omni runs first (steps 0+1) so it loads into clean VRAM before Demucs and Whisper.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, UploadFile, File, Form, Body,
)
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.session import Session, SessionConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
# Quieten noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Music Director")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Chat trigger phrases that bypass the LLM and route directly to _run_prompts.
# Matched case-insensitively against the stripped message.
_GENERATE_PROMPTS_TRIGGERS = frozenset({
    "proceed",
    "generate prompts",
    "write prompts",
    "start prompts",
    "generate all prompts",
    "write all prompts",
})

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

class SessionEntry:
    def __init__(self, session: Session):
        self.session      = session
        self.orchestrator = None          # created after analysis completes
        self.ws: WebSocket | None = None
        self._status: str = "idle"        # idle | analyzing | segmenting | generating_* | exporting
        self._bg_task: asyncio.Task | None = None

    @property
    def busy(self) -> bool:
        return self._status != "idle"

    async def push(self, event: str, data: dict) -> None:
        if self.ws:
            try:
                await self.ws.send_json({"event": event, **data})
            except Exception as exc:
                logger.debug("WS push failed (event=%s): %s", event, exc)

    async def push_status(self, message: str, step: str = "") -> None:
        logger.info("[%s] %s%s", self.session.session_id, f"[{step}] " if step else "", message)
        await self.push("status", {"message": message, "step": step})

    async def push_error(self, message: str) -> None:
        logger.error("[%s] ERROR: %s", self.session.session_id, message)
        await self.push("error", {"message": message})


_sessions: dict[str, SessionEntry] = {}


def _get_entry(session_id: str) -> SessionEntry:
    entry = _sessions.get(session_id)
    if not entry:
        raise HTTPException(404, f"Session {session_id!r} not found")
    return entry


def _orientation_dims(orientation: str) -> tuple[int, int]:
    if orientation == "landscape":
        return 2048, 1152
    return 1152, 2048   # portrait default


def _slugify(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:64] or "project"


# ---------------------------------------------------------------------------
# Workflow discovery
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Proxy llama-swap's model list for the frontend model selector."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{config.AGENT_URL}/v1/models")
            r.raise_for_status()
            data = r.json()
            ids = [m["id"] for m in data.get("data", [])]
            return {"models": ids}
    except Exception:
        return {"models": []}


@app.get("/workflows/templates")
async def list_workflow_templates():
    """List downloadable reference workflow files from workflows_UI/."""
    ui_dir = Path("backend/comfyui/workflows_UI")
    if not ui_dir.exists():
        return {"templates": []}
    templates = [
        {"name": f.name, "url": f"/workflow-templates/{f.name}"}
        for f in sorted(ui_dir.glob("*.json"))
    ]
    return {"templates": templates}


@app.get("/workflows")
async def list_workflows():
    user_dir = Path("backend/comfyui/workflows/user")
    image_wfs = [
        {"id": "zit", "name": "ZIT with Reactor"},
        {"id": "qie", "name": "Qwen Image Edit"},
    ]
    video_wfs = [
        {"id": "humo",     "name": "InfiniteTalk with HuMo", "humo_resolution": True},
        {"id": "ltx_humo", "name": "LTX with HuMo",          "humo_resolution": True},
        {"id": "ltx",      "name": "LTX"},
    ]
    if user_dir.exists():
        for wf_file in sorted(user_dir.glob("*.json")):
            stem = wf_file.stem
            nodemap_path = user_dir / f"{stem}.nodemap.json"
            if not nodemap_path.exists():
                continue
            try:
                nodemap = json.loads(nodemap_path.read_text(encoding="utf-8"))
            except Exception:
                nodemap = {}
            entry = {
                "id":   f"user/{stem}",
                "name": nodemap.get("display_name", stem),
            }
            if nodemap.get("humo_resolution"):
                entry["humo_resolution"] = True
            if "_i2v" in stem:
                video_wfs.append(entry)
            elif "_t2i" in stem:
                image_wfs.append(entry)
    return {"image": image_wfs, "video": video_wfs}


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@app.post("/sessions")
async def create_session(
    audio:        UploadFile = File(...),
    lyrics:       str        = Form(...),
    project_name: str        = Form(default=None),
    orientation:   str        = Form(default=None),
    fps:           int        = Form(default=None),
    scene_min_s:   float      = Form(default=None),
    scene_max_s:   float      = Form(default=None),
    image_workflow:  str        = Form(default=None),
    video_workflow:  str        = Form(default=None),
    humo_resolution: int        = Form(default=None),
    reference:       UploadFile = File(default=None),
):
    """
    Create a session, save uploaded files, and immediately start the analysis pipeline.
    Returns session_id — the frontend should redirect to /video/{session_id} and
    connect via WebSocket to receive analysis progress events.
    """
    if not lyrics or not lyrics.strip():
        raise HTTPException(400, "Lyrics are required for alignment.")

    session_id = uuid.uuid4().hex[:12]
    if project_name and project_name.strip():
        slug     = _slugify(project_name.strip())
        dir_name = f"{slug}-{session_id[:6]}"
    else:
        dir_name = session_id

    session_dir = (config.SESSION_DIR / dir_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    orient = orientation or config.DEFAULT_ORIENTATION
    w, h   = _orientation_dims(orient)
    iwf  = image_workflow  if (image_workflow  in ("zit", "qie")              or (image_workflow  or "").startswith("user/")) else "zit"
    vwf  = video_workflow  if (video_workflow  in ("ltx_humo", "ltx", "humo") or (video_workflow  or "").startswith("user/")) else "ltx_humo"
    hres = humo_resolution if humo_resolution in (1280, 1536, 1920)          else 1280
    cfg = SessionConfig(
        width           = w,
        height          = h,
        fps             = fps         or config.DEFAULT_FPS,
        orientation     = orient,
        scene_min_s     = scene_min_s or config.SCENE_MIN_SECONDS,
        scene_max_s     = scene_max_s or config.SCENE_MAX_SECONDS,
        image_workflow  = iwf,
        video_workflow  = vwf,
        humo_resolution = hres,
    )

    session = Session(
        session_id   = session_id,
        session_dir  = session_dir,
        config       = cfg,
        project_name = (project_name.strip() if project_name and project_name.strip() else None),
    )

    # ── Save audio ──────────────────────────────────────────────────────────
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_suffix = Path(audio.filename).suffix or ".mp3"
    audio_dest   = audio_dir / f"original{audio_suffix}"
    with audio_dest.open("wb") as f:
        shutil.copyfileobj(audio.file, f)
    session.audio_path = audio_dest
    audio_mb = audio_dest.stat().st_size / 1024 / 1024

    # ── Save lyrics ──────────────────────────────────────────────────────────
    session.raw_lyrics = lyrics.strip()
    session.save_raw_lyrics()

    # ── Save reference image (optional) ─────────────────────────────────────
    if reference and reference.filename:
        ref_dir    = session_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_suffix = Path(reference.filename).suffix or ".png"
        ref_dest   = ref_dir / f"reference{ref_suffix}"
        with ref_dest.open("wb") as f:
            shutil.copyfileobj(reference.file, f)
        session.reference_image_path = ref_dest
        logger.info("Reference image saved: %s", ref_dest.name)

    session.save_meta()
    entry = SessionEntry(session)
    _sessions[session_id] = entry

    logger.info(
        "Session created: %s → %s  (%s %dx%d fps=%d min=%.0fs max=%.0fs) audio=%.1fMB lyrics=%d chars%s",
        session_id, dir_name, orient, w, h, cfg.fps, cfg.scene_min_s, cfg.scene_max_s,
        audio_mb, len(lyrics),
        " +ref" if session.reference_image_path else "",
    )

    # ── Auto-start analysis ──────────────────────────────────────────────────
    entry._status  = "analyzing"
    entry._bg_task = asyncio.create_task(_run_analysis(entry))

    return {
        "session_id": session_id,
        "save_path":  str(session_dir),
        "config": {
            "orientation": orient, "width": w, "height": h,
            "fps": cfg.fps, "scene_min_s": cfg.scene_min_s, "scene_max_s": cfg.scene_max_s,
        },
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    entry = _get_entry(session_id)
    s = entry.session
    return {
        "session_id":  session_id,
        "phase":       s.phase,
        "status":      entry._status,
        "audio_path":  str(s.audio_path) if s.audio_path else None,
        "vocals_path": str(s.vocals_path) if s.vocals_path else None,
        "has_words":   bool(s.words),
        "has_music":   bool(s.music_data),
        "scene_k":     s.scene_k,
        "config": {
            "orientation": s.config.orientation,
            "width":       s.config.width,
            "height":      s.config.height,
            "fps":         s.config.fps,
        },
    }


@app.get("/sessions/{session_id}/prompts")
async def get_prompts(session_id: str):
    return _get_entry(session_id).session.load_prompts()


# ---------------------------------------------------------------------------
# Project list & resume
# ---------------------------------------------------------------------------

_ANALYSIS_NAMES = {
    "vocals.wav", "instrumental.wav", "aligned.json",
    "intonation.json", "music.json", "raw_lyrics.txt",
}


@app.get("/projects")
async def list_projects():
    root = config.SESSION_DIR
    if not root.exists():
        return {"projects": []}

    projects = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "session.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        audio_dir  = d / "audio"
        audio_file = None
        has_audio  = has_vocals = has_words = has_music = False

        if audio_dir.exists():
            for p in sorted(audio_dir.iterdir()):
                if p.is_file() and p.name not in _ANALYSIS_NAMES:
                    has_audio  = True
                    audio_file = p
                    break
            has_vocals = (audio_dir / "vocals.wav").exists()
            has_words  = (audio_dir / "aligned.json").exists()
            has_music  = (audio_dir / "music.json").exists()

        n_scenes = 0
        phase    = meta.get("phase", "upload")
        try:
            pd       = json.loads((d / "prompts.json").read_text(encoding="utf-8"))
            n_scenes = len(pd.get("scenes", {}))
        except Exception:
            pass

        display_name = (
            meta.get("project_name")
            or (audio_file.stem if audio_file else None)
            or d.name
        )

        projects.append({
            "dir_name":     d.name,
            "project_name": display_name,
            "session_id":   meta.get("session_id"),
            "phase":        phase,
            "n_scenes":     n_scenes,
            "save_path":    str(d),
            "modified_at":  d.stat().st_mtime,
        })

    logger.debug("Listed %d projects", len(projects))
    return {"projects": projects}


@app.post("/projects/{dir_name}/resume")
async def resume_project(dir_name: str):
    session_dir = (config.SESSION_DIR / dir_name).resolve()
    try:
        session_dir.relative_to(config.SESSION_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not session_dir.exists():
        raise HTTPException(404, f"Project {dir_name!r} not found")

    try:
        session = Session.from_dir(session_dir)
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc))

    if session.session_id in _sessions:
        existing = _sessions[session.session_id]
        logger.info(
            "Resume: session %s already live (phase=%s status=%s)",
            session.session_id, existing.session.phase, existing._status,
        )
        return {
            "session_id":   session.session_id,
            "save_path":    str(session_dir),
            "project_name": session.project_name,
            "phase":        existing.session.phase,
            "resumed":      True,
        }

    entry = SessionEntry(session)
    _sessions[session.session_id] = entry

    logger.info(
        "Project resumed: dir=%s session=%s phase=%s words=%d scenes=%d",
        dir_name, session.session_id, session.phase,
        len(session.words),
        len(session.load_prompts().get("scenes", {})),
    )
    return {
        "session_id":   session.session_id,
        "save_path":    str(session_dir),
        "project_name": session.project_name,
        "phase":        session.phase,
        "scene_k":      session.scene_k,
        "resumed":      True,
    }


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------


async def _run_analysis(entry: SessionEntry) -> None:
    session = entry.session
    sid     = session.session_id

    try:
        session.advance_phase("analysis")

        # ── Step 0: reference image description (Omni) ──────────────────────
        if session.reference_image_path and session.reference_image_path.exists():
            await entry.push_status("Describing reference image (Omni) ...", step="reference")
            t0 = time.perf_counter()

            from backend.analysis.omni import describe_image
            description = await asyncio.to_thread(describe_image, session.reference_image_path)
            session.reference_description = description
            session.save_meta()

            logger.info(
                "[%s] Reference image described in %.1fs  chars=%d",
                sid, time.perf_counter() - t0, len(description),
            )
            await entry.push("analysis_result", {
                "title":   "Reference Image",
                "content": description,
            })

        # ── Step 1: Omni intonation analysis ────────────────────────────────
        # Runs BEFORE Demucs and Whisper so Omni loads into clean VRAM.
        # If step 0 ran (image call), swap Omni out and back in — going from an
        # image call to an audio call in the same Omni session leaves the audio
        # encoder in a dirty state (same issue as back-to-back audio chunks).
        if session.reference_image_path and session.reference_image_path.exists():
            from backend.analysis.omni import AGENT_URL, AGENT_MODEL
            import httpx as _httpx
            try:
                await asyncio.to_thread(
                    lambda: _httpx.post(
                        f"{AGENT_URL}/v1/chat/completions",
                        json={"model": AGENT_MODEL, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1},
                        timeout=60.0,
                    )
                )
            except Exception:
                pass

        await entry.push_status("Analysing intonation (Omni) ...", step="omni")
        t0 = time.perf_counter()

        from backend.analysis.omni import analyze_intonation
        result = await asyncio.to_thread(
            analyze_intonation, session.audio_path
        )
        session.intonation = result["sections"]
        session.genre      = result.get("genre", "")
        session.subgenre   = result.get("subgenre", "")
        session.save_intonation()

        logger.info(
            "[%s] Omni done in %.1fs  genre=%s subgenre=%s",
            sid, time.perf_counter() - t0, session.genre, session.subgenre,
        )
        genre_line = session.genre or "unknown"
        if session.subgenre:
            genre_line += f" / {session.subgenre}"
        section_lines = "\n".join(
            f"{s.get('label', '?'):12s}  {s.get('energy', '?'):8s}  {s.get('mood', '')}  —  {s.get('intonation', '')}"
            for s in session.intonation
        )
        await entry.push("analysis_result", {
            "title":   f"Intonation — {genre_line}",
            "content": section_lines,
        })

        # ── Step 2: htdemucs ────────────────────────────────────────────────
        await entry.push_status("Separating vocals (htdemucs) ...", step="htdemucs")
        t0 = time.perf_counter()

        from backend.analysis.separation import separate
        paths = await asyncio.to_thread(
            separate, session.audio_path, session.audio_dir
        )
        session.vocals_path       = paths["vocals"]
        session.instrumental_path = paths["instrumental"]

        logger.info(
            "[%s] htdemucs done in %.1fs  vocals=%s instrumental=%s",
            sid, time.perf_counter() - t0,
            session.vocals_path.name, session.instrumental_path.name,
        )
        await entry.push_status("Vocals separated.", step="htdemucs")

        # ── Step 3: stable-ts alignment ─────────────────────────────────────
        await entry.push_status("Aligning lyrics (stable-ts) ...", step="align")
        t0 = time.perf_counter()

        from backend.analysis.aligner import align
        words = await asyncio.to_thread(
            align, session.vocals_path, session.raw_lyrics
        )
        session.words = words
        session.save_words()

        logger.info(
            "[%s] Alignment done in %.1fs  words=%d  first=%r last=%r",
            sid, time.perf_counter() - t0, len(words),
            words[0]["word"] if words else None,
            words[-1]["word"] if words else None,
        )
        await entry.push_status(f"Lyrics aligned: {len(words)} words.", step="align")

        # ── Step 4: librosa music analysis ──────────────────────────────────
        await entry.push_status("Analysing music structure (librosa) ...", step="librosa")
        t0 = time.perf_counter()

        from backend.analysis.music import analyze as librosa_analyze
        music_data = await asyncio.to_thread(librosa_analyze, session.audio_path)
        session.music_data = music_data
        session.save_music()

        logger.info(
            "[%s] librosa done in %.1fs  bpm=%.1f key=%s %s duration=%.1fs",
            sid, time.perf_counter() - t0,
            music_data["bpm"], music_data["key"], music_data["mode"],
            music_data["duration"],
        )
        await entry.push_status(
            f"Music analysed: {music_data['bpm']} BPM, {music_data['key']} {music_data['mode']}.",
            step="librosa",
        )

        # ── Step 5: auto-segmentation ────────────────────────────────────────
        # Run immediately so the orchestrator has boundaries ready on first message.
        await entry.push_status("Computing scene boundaries ...", step="segment")
        t0 = time.perf_counter()

        from backend.analysis.segmentation import segment, default_weights, auto_k as _auto_k
        weights  = default_weights()
        k_auto   = _auto_k(music_data["duration"], session.config.scene_max_s)
        seg_scenes  = await asyncio.to_thread(
            segment,
            session.audio_path,
            k=k_auto,
            fps=session.config.fps,
            min_s=session.config.scene_min_s,
            max_s=session.config.scene_max_s,
            weights=weights,
            words=session.words or None,
        )
        actual_k = len(seg_scenes)
        session.proposed_scenes = [
            {"start_s": s["start_s"], "end_s": s["end_s"], "frame_count": s["frame_count"]}
            for s in seg_scenes
        ]
        session.scene_k     = actual_k
        session.seg_weights = weights
        session.save_meta()

        logger.info(
            "[%s] Auto-segmentation done in %.1fs  k_auto=%d  actual_k=%d",
            sid, time.perf_counter() - t0, k_auto, actual_k,
        )
        await entry.push_status(f"Scene boundaries computed: {actual_k} scenes.", step="segment")

        # ── Done ────────────────────────────────────────────────────────────
        await entry.push("analysis_done", {
            "music_data":  {
                "bpm":      music_data["bpm"],
                "key":      music_data["key"],
                "mode":     music_data["mode"],
                "duration": music_data["duration"],
            },
            "intonation": session.intonation,
            "word_count": len(words),
            "auto_k":     actual_k,
        })
        logger.info("[%s] Analysis pipeline complete.", sid)

    except Exception as exc:
        logger.exception("[%s] Analysis pipeline failed: %s", sid, exc)
        await entry.push_error(f"Analysis failed: {exc}")
    finally:
        entry._status = "idle"


def _auto_k(duration_s: float, max_scene_s: float) -> int:
    from backend.analysis.segmentation import auto_k
    return auto_k(duration_s, max_scene_s)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/segment")
async def run_segment(
    session_id: str,
    k:          int              = Body(...),
    weights:    list[float] | None = Body(default=None),
):
    """
    Run algorithmic segmentation.
    Body: {"k": 20, "weights": [0.5, 0.3, 0.2]}  (weights optional)
    Pushes segment_done + scene_update events via WebSocket.
    """
    entry = _get_entry(session_id)

    if not entry.session.music_data:
        raise HTTPException(400, "Run analysis first.")
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    entry._status  = "segmenting"
    entry._bg_task = asyncio.create_task(_run_segment(entry, k, weights))
    return {"ok": True}


@app.post("/sessions/{session_id}/segment/reroll")
async def reroll_segment(session_id: str):
    """Re-run segmentation at the same k with new random weights."""
    entry = _get_entry(session_id)
    if not entry.session.scene_k:
        raise HTTPException(400, "Run segment first.")
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    from backend.analysis.segmentation import random_weights
    weights = list(random_weights())
    logger.info("[%s] Reroll: k=%d new_weights=%s", session_id, entry.session.scene_k, weights)

    entry._status  = "segmenting"
    entry._bg_task = asyncio.create_task(
        _run_segment(entry, entry.session.scene_k, weights)
    )
    return {"ok": True, "weights": weights}


async def _run_segment(
    entry: SessionEntry,
    k: int,
    weights: list[float] | None,
) -> None:
    session = entry.session
    sid     = session.session_id

    try:
        await entry.push_status(f"Segmenting into {k} scenes ...", step="segment")
        t0 = time.perf_counter()

        from backend.analysis.segmentation import segment, default_weights
        w = tuple(weights) if weights and len(weights) == 3 else default_weights()

        scenes = await asyncio.to_thread(
            segment,
            session.audio_path,
            k,
            session.config.fps,
            session.config.scene_min_s,
            session.config.scene_max_s,
            w,
            session.words or None,
        )

        session.scene_k     = k
        session.seg_weights = w
        session.save_meta()

        logger.info(
            "[%s] Segmentation done in %.2fs  k=%d  weights=%s  durations=[%s]",
            sid, time.perf_counter() - t0, len(scenes), list(w),
            ", ".join(f"{s['end_s'] - s['start_s']:.1f}s" for s in scenes),
        )

        # Attach lyrics window to each scene for display
        for scene in scenes:
            scene["lyrics_window"] = session.extract_lyrics_window(
                scene["start_s"], scene["end_s"]
            )

        await entry.push("segment_done", {
            "scenes":  scenes,
            "k":       len(scenes),
            "weights": list(w),
        })
        await entry.push_status("Segmentation complete.", step="segment")

    except Exception as exc:
        logger.exception("[%s] Segmentation failed: %s", sid, exc)
        await entry.push_error(f"Segmentation failed: {exc}")
    finally:
        entry._status = "idle"


# ---------------------------------------------------------------------------
# Phase gates
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/approve/plan")
async def approve_plan(
    session_id: str,
    scenes: list[dict] = Body(...),
):
    """
    Gate 1: commit the scene plan. Timing is locked after this.
    Body: the scene list from the last segment_done event (with any user edits).
    Advances phase to style_bible.
    """
    entry = _get_entry(session_id)
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    entry.session.commit_scenes(scenes)
    entry.session.advance_phase("style_bible")

    logger.info(
        "[%s] Plan approved: %d scenes committed. Phase → style_bible.",
        session_id, len(scenes),
    )
    # Push all scenes so UI reflects committed state
    for i, scene in enumerate(entry.session.all_scenes(), start=1):
        await entry.push("scene_update", {"scene_index": i, "scene": scene})

    return {"ok": True, "phase": "style_bible", "n_scenes": len(scenes)}


@app.post("/sessions/{session_id}/approve/style-bible")
async def approve_style_bible(
    session_id: str,
    style_bible: dict = Body(...),
):
    """
    Gate 2: commit the approved style bible.
    Advances phase to prompts.
    """
    entry = _get_entry(session_id)
    entry.session.set_style_bible(style_bible)
    entry.session.advance_phase("prompts")

    logger.info("[%s] Style bible approved. Phase → prompts.", session_id)
    return {"ok": True, "phase": "prompts"}


@app.post("/sessions/{session_id}/approve/prompts")
async def approve_prompts(session_id: str):
    """
    Gate 3: all scene prompts reviewed. Advances phase to images.
    """
    entry = _get_entry(session_id)
    entry.session.advance_phase("images")
    logger.info("[%s] Prompts approved. Phase → images.", session_id)
    return {"ok": True, "phase": "images"}


# ---------------------------------------------------------------------------
# Generation triggers
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/generate/prompts")
async def generate_prompts(session_id: str):
    """
    Trigger per-scene prompt generation loop.
    Runs generate_prompts_loop() which calls the LLM once per scene with
    the full style bible + scene data as focused context.
    """
    entry = _get_entry(session_id)
    if entry.session.phase not in ("style_bible", "prompts"):
        raise HTTPException(400, "Approve the style bible before generating prompts.")
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")
    if entry.orchestrator is None:
        raise HTTPException(400, "Analysis must complete before generating prompts.")

    sb = entry.session.load_prompts().get("style_bible", {})
    if not sb.get("character", "").strip():
        raise HTTPException(
            400,
            "Style bible has no character description. "
            "Ask the agent to generate and commit the style bible first."
        )

    entry._status  = "generating_prompts"
    entry._bg_task = asyncio.create_task(_run_prompts(entry))
    return {"ok": True}


async def _run_prompts(entry: SessionEntry) -> None:
    sid = entry.session.session_id
    try:
        await entry.orchestrator.generate_prompts_loop()
        logger.info("[%s] Prompt generation complete.", sid)
    except Exception as exc:
        logger.exception("[%s] Prompt generation failed: %s", sid, exc)
        await entry.push_error(f"Prompt generation failed: {exc}")
    finally:
        entry._status = "idle"


@app.post("/sessions/{session_id}/generate/images")
async def generate_images(
    session_id:    str,
    scene_numbers: list[int] | None = Body(default=None),
):
    """
    Trigger image generation for all scenes (or specific scene_numbers).
    Runs as background task; progress via WebSocket scene_update events.
    """
    entry = _get_entry(session_id)
    if entry.session.phase not in ("images", "videos", "done"):
        raise HTTPException(400, "Approve all prompts before generating images.")
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    entry._status  = "generating_images"
    entry._bg_task = asyncio.create_task(_run_images(entry, scene_numbers))
    return {"ok": True}


async def _run_images(entry: SessionEntry, scene_numbers: list[int] | None) -> None:
    session = entry.session
    sid     = session.session_id

    try:
        from backend.agent.tools import generate_images_batch
        await generate_images_batch(session, scene_numbers, entry.push)
        logger.info("[%s] Image generation complete.", sid)
        await entry.push("gen_done", {"phase": "images"})

        if entry.orchestrator:
            scenes    = session.load_prompts().get("scenes", {})
            n_done    = sum(1 for v in scenes.values() if v.get("image_status") in ("done", "approved"))
            n_failed  = sum(1 for v in scenes.values() if v.get("image_status") == "failed")
            failed_ns = [k for k, v in scenes.items() if v.get("image_status") == "failed"]
            msg = f"Image generation complete: {n_done}/{len(scenes)} succeeded."
            if n_failed:
                msg += f" {n_failed} failed (scenes {', '.join(failed_ns)})."
            msg += " The user can now review and approve images, or request regens."
            entry.orchestrator.inject_tool_context(msg, ack="Got it.")
    except Exception as exc:
        logger.exception("[%s] Image generation failed: %s", sid, exc)
        await entry.push_error(f"Image generation failed: {exc}")
    finally:
        entry._status = "idle"


@app.post("/sessions/{session_id}/generate/videos")
async def generate_videos(
    session_id:    str,
    scene_numbers: list[int] | None = Body(default=None),
):
    """
    Trigger video generation for all approved scenes (or specific scene_numbers).
    """
    entry = _get_entry(session_id)
    if entry.session.phase not in ("videos", "done"):
        raise HTTPException(400, "Approve all images before generating videos.")
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    entry._status  = "generating_videos"
    entry._bg_task = asyncio.create_task(_run_videos(entry, scene_numbers))
    return {"ok": True}


async def _run_videos(entry: SessionEntry, scene_numbers: list[int] | None) -> None:
    session = entry.session
    sid     = session.session_id

    try:
        from backend.agent.tools import generate_videos_batch
        await generate_videos_batch(session, scene_numbers, entry.push)
        logger.info("[%s] Video generation complete.", sid)
        await entry.push("gen_done", {"phase": "videos"})

        if entry.orchestrator:
            scenes    = session.load_prompts().get("scenes", {})
            n_done    = sum(1 for v in scenes.values() if v.get("video_status") in ("done", "approved"))
            n_failed  = sum(1 for v in scenes.values() if v.get("video_status") == "failed")
            failed_ns = [k for k, v in scenes.items() if v.get("video_status") == "failed"]
            msg = f"Video generation complete: {n_done}/{len(scenes)} succeeded."
            if n_failed:
                msg += f" {n_failed} failed (scenes {', '.join(failed_ns)})."
            msg += " The user can now review and approve videos, or request regens."
            entry.orchestrator.inject_tool_context(msg, ack="Got it.")
    except Exception as exc:
        logger.exception("[%s] Video generation failed: %s", sid, exc)
        await entry.push_error(f"Video generation failed: {exc}")
    finally:
        entry._status = "idle"


@app.post("/sessions/{session_id}/export")
async def export_final(session_id: str, output_filename: str = "final.mp4"):
    entry = _get_entry(session_id)
    if entry.busy:
        raise HTTPException(409, f"Session is busy ({entry._status}).")

    entry._status  = "exporting"
    entry._bg_task = asyncio.create_task(_run_export(entry, output_filename))
    return {"ok": True}


async def _run_export(entry: SessionEntry, output_filename: str) -> None:
    sid = entry.session.session_id
    try:
        await entry.push_status("Exporting final video ...", step="export")
        from backend.render import export
        session = entry.session
        scenes = [s for s in session.all_scenes() if s.get("video_status") == "approved"]
        if not scenes:
            await entry.push_error("No approved scenes to export.")
            return
        clip_paths  = [s["video_path"] for s in scenes]
        output_path = session.session_dir / output_filename
        first_start = scenes[0].get("start_s", 0.0)
        out_path = await asyncio.to_thread(export, clip_paths, session.audio_path, output_path, first_start)
        logger.info("[%s] Export done: %s", sid, out_path)
        await entry.push("export_done", {"path": str(out_path)})
    except Exception as exc:
        logger.exception("[%s] Export failed: %s", sid, exc)
        await entry.push_error(f"Export failed: {exc}")
    finally:
        entry._status = "idle"


# ---------------------------------------------------------------------------
# Scene approvals + inline edits
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/scenes/{scene_index}/approve/image")
async def approve_image(session_id: str, scene_index: int):
    entry = _get_entry(session_id)
    entry.session.update_scene(scene_index, {"image_status": "approved"})
    scene = entry.session.get_scene(scene_index)
    await entry.push("scene_update", {"scene_index": scene_index, "scene": scene})
    logger.info("[%s] Image approved: scene %d", session_id, scene_index)

    # Advance phase to videos once all images are approved
    scenes = entry.session.all_scenes()
    all_approved = scenes and all(s.get("image_status") == "approved" for s in scenes)
    if all_approved:
        entry.session.advance_phase("videos")
        logger.info("[%s] All images approved. Phase → videos.", session_id)

    if entry.orchestrator:
        msg = f"User approved image for scene {scene_index}."
        if all_approved:
            msg += f" All {len(scenes)} images are now approved — ready for video generation."
        entry.orchestrator.inject_tool_context(msg, ack="Got it.")

    return {"ok": True}


@app.post("/sessions/{session_id}/scenes/{scene_index}/approve/video")
async def approve_video(session_id: str, scene_index: int):
    entry = _get_entry(session_id)
    entry.session.update_scene(scene_index, {"video_status": "approved"})

    # Advance phase to done if all videos approved
    scenes = entry.session.all_scenes()
    all_approved = scenes and all(s.get("video_status") == "approved" for s in scenes)
    if all_approved:
        entry.session.advance_phase("done")
        logger.info("[%s] All videos approved. Phase → done.", session_id)

    scene = entry.session.get_scene(scene_index)
    await entry.push("scene_update", {"scene_index": scene_index, "scene": scene})
    logger.info("[%s] Video approved: scene %d", session_id, scene_index)

    if entry.orchestrator:
        msg = f"User approved video for scene {scene_index}."
        if all_approved:
            msg += f" All {len(scenes)} videos are now approved — ready to export."
        entry.orchestrator.inject_tool_context(msg, ack="Got it.")

    return {"ok": True, "phase": entry.session.phase}


@app.delete("/sessions/{session_id}/scenes/{scene_index}/approve/image")
async def unapprove_image(session_id: str, scene_index: int):
    entry = _get_entry(session_id)
    entry.session.update_scene(scene_index, {"image_status": "done"})
    # Roll back phase if it had advanced to videos
    if entry.session.phase == "videos":
        entry.session.phase = "images"
    scene = entry.session.get_scene(scene_index)
    await entry.push("scene_update", {"scene_index": scene_index, "scene": scene})
    logger.info("[%s] Image unapproved: scene %d", session_id, scene_index)
    if entry.orchestrator:
        entry.orchestrator.inject_tool_context(
            f"User un-approved image for scene {scene_index} (reverted to done).", ack="Got it."
        )
    return {"ok": True}


@app.delete("/sessions/{session_id}/scenes/{scene_index}/approve/video")
async def unapprove_video(session_id: str, scene_index: int):
    entry = _get_entry(session_id)
    entry.session.update_scene(scene_index, {"video_status": "done"})
    # Roll back phase if it had advanced to done
    if entry.session.phase == "done":
        entry.session.phase = "videos"
    scene = entry.session.get_scene(scene_index)
    await entry.push("scene_update", {"scene_index": scene_index, "scene": scene})
    logger.info("[%s] Video unapproved: scene %d", session_id, scene_index)
    if entry.orchestrator:
        entry.orchestrator.inject_tool_context(
            f"User un-approved video for scene {scene_index} (reverted to done).", ack="Got it."
        )
    return {"ok": True}


@app.patch("/sessions/{session_id}/scenes/{scene_index}")
async def patch_scene(session_id: str, scene_index: int, fields: dict[str, Any]):
    """Directly update scene fields from the UI (inline prompt edits, etc.)."""
    entry = _get_entry(session_id)

    # Guard: cannot change timing fields after Gate 1
    locked = {"start_s", "end_s", "frame_count"}
    if entry.session.phase not in ("planning",) and locked.intersection(fields):
        raise HTTPException(400, "Timing fields are locked after plan approval.")

    # If a prompt field is being edited, promote status to prompts_ready
    # so the scene qualifies for image generation.
    prompt_fields = {"image_prompt", "video_prompt"}
    if prompt_fields.intersection(fields):
        scene = entry.session.get_scene(scene_index)
        if scene.get("image_status") == "planned":
            fields = {**fields, "image_status": "prompts_ready"}

    entry.session.update_scene(scene_index, fields)
    scene = entry.session.get_scene(scene_index)
    await entry.push("scene_update", {"scene_index": scene_index, "scene": scene})
    logger.debug("[%s] Scene %d patched: %s", session_id, scene_index, list(fields.keys()))
    return {"ok": True}


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/files/{subpath:path}")
async def session_file(session_id: str, subpath: str):
    entry = _get_entry(session_id)
    path  = (entry.session.session_dir / subpath).resolve()
    try:
        path.relative_to(entry.session.session_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Forbidden")
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), headers={"Cache-Control": "no-store"})


@app.get("/sessions/{session_id}/export/download")
async def download_export(session_id: str, filename: str = "final.mp4"):
    entry = _get_entry(session_id)
    path  = entry.session.session_dir / filename
    if not path.exists():
        raise HTTPException(404, "Export not found. Run export first.")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/sessions/{session_id}/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    entry = _sessions.get(session_id)
    if not entry:
        logger.warning("WS connect rejected: unknown session %s", session_id)
        await websocket.close(code=4004)
        return

    await websocket.accept()
    entry.ws = websocket
    logger.info(
        "WS connected: session=%s phase=%s status=%s words=%d",
        session_id, entry.session.phase, entry._status, len(entry.session.words),
    )

    # Re-hydrate frontend with all committed scenes
    try:
        scene_data = entry.session.load_prompts()
        n_rehydrated = 0
        for k, scene in scene_data.get("scenes", {}).items():
            await websocket.send_json({"event": "scene_update", "scene_index": int(k), "scene": scene})
            n_rehydrated += 1
        if n_rehydrated:
            logger.debug("WS re-hydrated %d scenes for session %s", n_rehydrated, session_id)
    except Exception as exc:
        logger.warning("WS re-hydration failed for %s: %s", session_id, exc)

    # Create orchestrator if analysis is complete and we haven't yet
    if entry.session.music_data and entry.orchestrator is None:
        _init_orchestrator(entry)

    try:
        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "chat":
                message = data.get("message", "").strip()
                if not message:
                    continue
                if entry.busy:
                    await websocket.send_json({
                        "event":   "error",
                        "message": f"Agent is busy ({entry._status}) — wait for it to finish.",
                    })
                    continue
                if entry.orchestrator is None:
                    await websocket.send_json({
                        "event":   "error",
                        "message": "Analysis must complete before chatting.",
                    })
                    continue
                logger.debug("[%s] Chat message: %r", session_id, message[:80])
                # Intercept "proceed" / "generate prompts" so they reliably
                # trigger generate_prompts_loop instead of the main chat loop.
                if (
                    message.lower().strip() in _GENERATE_PROMPTS_TRIGGERS
                    and entry.session.phase in ("style_bible", "prompts")
                    and not entry.busy
                    and entry.orchestrator is not None
                ):
                    _sb = entry.session.load_prompts().get("style_bible", {})
                    if not _sb.get("character", "").strip():
                        await websocket.send_json({
                            "event":   "error",
                            "message": "Style bible has no character description. Ask me to generate the style bible first.",
                        })
                        continue
                    entry._status  = "generating_prompts"
                    entry._bg_task = asyncio.create_task(_run_prompts(entry))
                    await websocket.send_json({
                        "event": "token",
                        "text":  "Writing prompts for all scenes now — storyboard cards will fill in as each one completes.",
                    })
                    await websocket.send_json({"event": "assistant_done"})
                else:
                    asyncio.create_task(_handle_chat(entry, message))

            elif msg_type == "auto_start":
                # Sent by frontend on connect (new sessions) and after analysis_done (resumed).
                # Init orchestrator if analysis is now complete; fire __auto_start__ if idle.
                if entry.orchestrator is None and entry.session.music_data:
                    _init_orchestrator(entry)
                if entry.orchestrator and not entry.busy:
                    asyncio.create_task(_handle_chat(entry, "__auto_start__"))

            elif msg_type == "ping":
                await websocket.send_json({"event": "pong"})

            else:
                logger.warning("[%s] Unknown WS message type: %r", session_id, msg_type)

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    finally:
        entry.ws = None
        # Rebind push on reconnect uses new entry.ws — nothing to clean up here


def _init_orchestrator(entry: SessionEntry) -> None:
    from backend.agent.orchestrator import Orchestrator
    entry.orchestrator = Orchestrator(
        session=entry.session,
        push=entry.push,
    )
    logger.info(
        "Orchestrator initialised: session=%s phase=%s scenes=%d",
        entry.session.session_id,
        entry.session.phase,
        len(entry.session.load_prompts().get("scenes", {})),
    )


async def _handle_chat(entry: SessionEntry, message: str) -> None:
    entry._status = "generating_response"
    try:
        await entry.orchestrator.chat(message)
    except Exception as exc:
        logger.exception("[%s] Chat error: %s", entry.session.session_id, exc)
        err = str(exc)
        if "connect" in err.lower():
            err = f"Cannot reach LLM — is llama-swap running on port 8000? ({err})"
        await entry.push_error(err)
    finally:
        entry._status = "idle"


# ---------------------------------------------------------------------------
# Suno prompt assistant
# ---------------------------------------------------------------------------

class _SunoEntry:
    def __init__(self, mode: str = "suno", model: str | None = None) -> None:
        from backend.agent.suno_orchestrator import SunoOrchestrator
        self.orch = SunoOrchestrator(mode=mode, model=model)
        self.ws: WebSocket | None = None
        self.busy = False

    async def push(self, event: str, data: dict) -> None:
        if self.ws:
            try:
                await self.ws.send_json({"event": event, **data})
            except Exception:
                pass


_suno_sessions: dict[str, _SunoEntry] = {}


@app.post("/suno/sessions")
async def create_suno_session(mode: str = "suno", model: str | None = None):
    sid = uuid.uuid4().hex[:12]
    _suno_sessions[sid] = _SunoEntry(mode=mode, model=model)
    logger.info("Suno session created: %s (mode=%s, model=%s)", sid, mode, model or "default")
    return {"session_id": sid}


@app.websocket("/suno/{session_id}/ws")
async def suno_websocket(websocket: WebSocket, session_id: str):
    entry = _suno_sessions.get(session_id)
    if not entry:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    entry.ws = websocket
    logger.info("Suno WS connected: %s", session_id)

    ping = None
    try:
        import asyncio as _asyncio

        async def _keepalive():
            while True:
                await _asyncio.sleep(25)
                try:
                    await websocket.send_json({"event": "pong"})
                except Exception:
                    break

        ping = asyncio.create_task(_keepalive())

        chat_task: asyncio.Task | None = None

        async def _run_chat(message: str) -> None:
            try:
                await entry.orch.chat(message, entry.push)
            except asyncio.CancelledError:
                await entry.push("assistant_done", {})
            except Exception as exc:
                logger.exception("Suno chat error: %s", exc)
                await entry.push("error", {"message": str(exc)})
            finally:
                entry.busy = False

        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "chat":
                message = data.get("message", "").strip()
                if not message or entry.busy:
                    continue
                entry.busy = True
                chat_task = asyncio.create_task(_run_chat(message))

            elif msg_type == "stop":
                if chat_task and not chat_task.done():
                    chat_task.cancel()

            elif msg_type == "ping":
                await websocket.send_json({"event": "pong"})

    except WebSocketDisconnect:
        logger.info("Suno WS disconnected: %s", session_id)
    finally:
        if ping:
            ping.cancel()
        entry.ws = None


# ---------------------------------------------------------------------------
# ACE-Step local generator
# ---------------------------------------------------------------------------

class _AceStepEntry:
    def __init__(self, session_id: str) -> None:
        self.session_id  = session_id
        self.ws: WebSocket | None = None
        self.busy_gen    = False
        self.params:      dict = {}
        self.takes:       list[dict] = []
        self.created_at   = datetime.datetime.utcnow().isoformat(timespec="seconds")
        self.session_dir  = config.SESSION_DIR / "acestep" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "params":     self.params,
            "takes": [
                {"take_n": t["take_n"], "filename": t.get("filename", f"take_{t['take_n']}.wav"), "metadata": t.get("metadata", {})}
                for t in self.takes
            ],
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_disk(cls, session_id: str) -> "_AceStepEntry":
        entry = cls(session_id)
        sf = entry.session_dir / "session.json"
        if sf.exists():
            data = json.loads(sf.read_text(encoding="utf-8"))
            entry.created_at = data.get("created_at", entry.created_at)
            entry.params     = data.get("params", {})
            entry.takes      = [
                {
                    "take_n":    t["take_n"],
                    "filename":  t.get("filename", f"take_{t['take_n']}.wav"),
                    "audio_url": f"/acestep/sessions/{session_id}/takes/{t['take_n']}/audio",
                    "metadata":  t.get("metadata", {}),
                    "status":    "done",
                }
                for t in data.get("takes", [])
                if (entry.session_dir / t.get("filename", f"take_{t['take_n']}.wav")).exists()
            ]
        return entry

    async def push(self, event: str, data: dict) -> None:
        if self.ws:
            try:
                await self.ws.send_json({"event": event, **data})
            except Exception:
                pass


_acestep_sessions: dict[str, _AceStepEntry] = {}


@app.post("/acestep/sessions")
async def create_acestep_session():
    sid = uuid.uuid4().hex[:12]
    _acestep_sessions[sid] = _AceStepEntry(sid)
    logger.info("ACEStep session created: %s", sid)
    return {"session_id": sid}


@app.post("/acestep/start")
async def start_acestep():
    """Launch the ACEStep subprocess (non-blocking — client should poll /acestep/health)."""
    from backend import acestep_process
    asyncio.create_task(acestep_process.start())
    return {"status": "starting"}


@app.post("/acestep/stop")
async def stop_acestep():
    """Terminate the ACEStep subprocess."""
    from backend import acestep_process
    await acestep_process.stop()
    return {"status": "stopped"}


@app.get("/acestep/health")
async def acestep_health():
    """Proxy the ACEStep server's health, forwarding its ok field.
    Returns ok=false while the model is still loading (not just while the process is starting)."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{config.ACESTEP_URL}/health")
            if r.status_code == 200:
                data = r.json()
                # v1.5 server returns {ok, status}; legacy returns {status: "ok"}
                ok = data.get("ok", data.get("status") == "ok")
                return {"ok": ok}
    except Exception:
        pass
    return {"ok": False}


@app.get("/acestep/log")
async def acestep_log(lines: int = 40):
    """Return the last N lines of the ACEStep server log."""
    log_path = Path(__file__).parent.parent / "acestep_server.log"
    if not log_path.exists():
        return {"lines": []}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = text.splitlines()[-lines:]
        return {"lines": tail}
    except Exception:
        return {"lines": []}


@app.get("/acestep/sessions")
async def list_acestep_sessions():
    base = config.SESSION_DIR / "acestep"
    sessions = []
    if base.exists():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            sf = d / "session.json"
            if sf.exists():
                data = json.loads(sf.read_text(encoding="utf-8"))
                caption = data.get("params", {}).get("caption", "")
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data.get("created_at", ""),
                    "caption":    caption[:80] if caption else "(no caption)",
                    "take_count": len(data.get("takes", [])),
                })
    return {"sessions": sessions}


@app.post("/acestep/sessions/{session_id}/resume")
async def resume_acestep_session(session_id: str):
    if session_id not in _acestep_sessions:
        _acestep_sessions[session_id] = _AceStepEntry.from_disk(session_id)
    entry = _acestep_sessions[session_id]
    return {
        "session_id": entry.session_id,
        "params":     entry.params,
        "takes":      entry.takes,
    }


class FromAceStepRequest(BaseModel):
    acestep_session_id: str
    take_n:             int
    project_name:       str | None = None
    orientation:        str | None = None


@app.post("/sessions/from_acestep")
async def create_session_from_acestep(req: FromAceStepRequest):
    """
    Create a video director session from an ACEStep take.
    Copies the WAV and lyrics into a new video session, then starts analysis.
    Returns {session_id, save_path} — same shape as POST /sessions.
    """
    # ── Resolve ACEStep source ───────────────────────────────────────────────
    entry = _acestep_sessions.get(req.acestep_session_id)
    if not entry:
        session_dir_as = config.SESSION_DIR / "acestep" / req.acestep_session_id
        sf = session_dir_as / "session.json"
        if sf.exists():
            entry = _AceStepEntry.from_disk(req.acestep_session_id)
        else:
            raise HTTPException(404, "ACEStep session not found")

    src_wav = entry.session_dir / f"take_{req.take_n}.wav"
    if not src_wav.exists():
        raise HTTPException(404, f"Take {req.take_n} not found")

    lyrics = (entry.params.get("lyrics") or "").strip()

    # ── Create video session ─────────────────────────────────────────────────
    session_id = uuid.uuid4().hex[:12]
    pname = req.project_name or entry.params.get("caption") or ""
    if pname.strip():
        slug     = _slugify(pname.strip())
        dir_name = f"{slug}-{session_id[:6]}"
    else:
        dir_name = session_id

    session_dir = (config.SESSION_DIR / dir_name).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    orient = req.orientation or config.DEFAULT_ORIENTATION
    w, h   = _orientation_dims(orient)
    cfg = SessionConfig(
        width           = w,
        height          = h,
        fps             = config.DEFAULT_FPS,
        orientation     = orient,
        scene_min_s     = config.SCENE_MIN_SECONDS,
        scene_max_s     = config.SCENE_MAX_SECONDS,
        image_workflow  = "zit",
        video_workflow  = "ltx_humo",
        humo_resolution = 1280,
    )

    session = Session(
        session_id   = session_id,
        session_dir  = session_dir,
        config       = cfg,
        project_name = pname.strip() or None,
    )

    # ── Copy WAV ─────────────────────────────────────────────────────────────
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_dest = audio_dir / "original.wav"
    shutil.copy2(src_wav, audio_dest)
    session.audio_path = audio_dest

    # ── Save lyrics ───────────────────────────────────────────────────────────
    session.raw_lyrics = lyrics
    if lyrics:
        session.save_raw_lyrics()

    session.save_meta()
    se = SessionEntry(session)
    _sessions[session_id] = se

    logger.info(
        "Session from ACEStep: %s (take %d) → %s",
        req.acestep_session_id, req.take_n, dir_name,
    )

    # ── Start analysis ────────────────────────────────────────────────────────
    if lyrics:
        se._status  = "analyzing"
        se._bg_task = asyncio.create_task(_run_analysis(se))

    return {
        "session_id": session_id,
        "save_path":  str(session_dir),
        "config": {
            "orientation": orient, "width": w, "height": h,
            "fps": cfg.fps, "scene_min_s": cfg.scene_min_s, "scene_max_s": cfg.scene_max_s,
        },
    }


@app.get("/acestep/sessions/{session_id}/takes/{take_n}/audio")
async def get_take_audio(session_id: str, take_n: int):
    entry = _acestep_sessions.get(session_id)
    if entry:
        take = next((t for t in entry.takes if t["take_n"] == take_n), None)
        if not take:
            raise HTTPException(status_code=404, detail="Take not found")
        audio_path = entry.session_dir / take.get("filename", f"take_{take_n}.wav")
    else:
        # Fall back to disk — load session.json to get filename
        session_dir = config.SESSION_DIR / "acestep" / session_id
        sf = session_dir / "session.json"
        filename = f"take_{take_n}.wav"
        if sf.exists():
            import json as _json
            data = _json.loads(sf.read_text(encoding="utf-8"))
            for t in data.get("takes", []):
                if t.get("take_n") == take_n:
                    filename = t.get("filename", filename)
                    break
        audio_path = session_dir / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Take not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@app.websocket("/acestep/{session_id}/ws")
async def acestep_websocket(websocket: WebSocket, session_id: str):
    entry = _acestep_sessions.get(session_id)
    if not entry:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    entry.ws = websocket
    logger.info("ACEStep WS connected: %s", session_id)

    ping = None
    try:
        async def _keepalive():
            while True:
                await asyncio.sleep(25)
                try:
                    await websocket.send_json({"event": "pong"})
                except Exception:
                    break

        ping = asyncio.create_task(_keepalive())

        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "generate":
                if entry.busy_gen:
                    continue
                params = data.get("params", {})
                entry.params = params
                entry.save()
                entry.busy_gen = True
                asyncio.create_task(_run_acestep_generation(entry, params))

            elif msg_type == "ping":
                await websocket.send_json({"event": "pong"})

    except WebSocketDisconnect:
        logger.info("ACEStep WS disconnected: %s", session_id)
    finally:
        if ping:
            ping.cancel()
        entry.ws = None


async def _run_acestep_generation(entry: _AceStepEntry, params: dict) -> None:
    import re
    from backend import acestep_client
    take_n = len(entry.takes) + 1
    raw_title = params.get("title", "").strip()
    safe_title = re.sub(r'[^\w\s-]', '', raw_title).strip().lower().replace(' ', '_')
    filename = f"{safe_title}_take{take_n}.wav" if safe_title else f"take_{take_n}.wav"
    try:
        await entry.push("gen_start", {"take_n": take_n})

        task_id = await acestep_client.generate(params)

        while True:
            await asyncio.sleep(3)
            result = await acestep_client.poll(task_id)
            if result is None:
                continue  # still running

            audio_path = entry.session_dir / filename
            await acestep_client.download_audio(result["file"], audio_path)

            take = {
                "take_n":    take_n,
                "filename":  filename,
                "audio_url": f"/acestep/sessions/{entry.session_id}/takes/{take_n}/audio",
                "metadata":  result.get("metas", {}),
            }
            entry.takes.append(take)
            entry.save()
            await entry.push("gen_done", {
                "take_n":    take_n,
                "audio_url": take["audio_url"],
                "metadata":  take["metadata"],
            })
            break

    except Exception as exc:
        logger.exception("ACEStep generation error: %s", exc)
        await entry.push("gen_error", {"take_n": take_n, "message": str(exc)})
    finally:
        entry.busy_gen = False


# ---------------------------------------------------------------------------
# Workflow templates (downloadable reference files)
# ---------------------------------------------------------------------------

_WF_TEMPLATES = Path(__file__).parent / "comfyui" / "workflows_UI"
if _WF_TEMPLATES.exists():
    app.mount("/workflow-templates", StaticFiles(directory=str(_WF_TEMPLATES)), name="workflow-templates")


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(_FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))
else:
    logger.warning("Frontend dist not found at %s — UI not served", _FRONTEND_DIST)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        reload=False,
        log_level="info",
    )
