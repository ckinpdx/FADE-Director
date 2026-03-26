"""
session.py
Session dataclass and prompts.json management.

Phases (in order):
  upload       — no audio yet
  analysis     — vocal separation, Omni, librosa running or complete
  planning     — analysis done; user reviews segmentation, adjusts k, approves
  style_bible  — plan approved; style bible generated and awaiting approval
  prompts      — style bible approved; per-scene prompts generated and awaiting approval
  images       — all prompts approved; image generation running or complete
  videos       — all images approved; video generation running or complete
  done         — all videos approved; ready to export

Phase is written to session.json on every transition so server restarts
never lose position. _infer_phase() is a fallback for legacy session dirs.
"""

from __future__ import annotations

import datetime
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PHASES = [
    "upload",
    "analysis",
    "planning",
    "style_bible",
    "prompts",
    "images",
    "videos",
    "done",
]


@dataclass
class SessionConfig:
    width:         int   = 2048
    height:        int   = 1152
    fps:           int   = 25
    orientation:   str   = "landscape"  # portrait | landscape
    scene_min_s:   float = 3.0
    scene_max_s:   float = 20.0
    image_workflow:  str        = "zit"        # zit | qie — set at session creation
    video_workflow:  str        = "ltx_humo"   # ltx_humo | ltx — set at session creation
    humo_resolution: int        = 1280         # HuMo long-edge for landscape (1280 | 1536 | 1920)
    lora_name:       str | None = None         # character LoRA filename (loras/ subdir)
    lora_strength:   float      = 0.6
    base_negative: str   = (
        "blurry, distorted, low quality, artifacts, watermark, text, "
        "duplicate, ugly, deformed, mutated, out of frame"
    )


@dataclass
class Session:
    session_id:  str
    session_dir: Path
    config:      SessionConfig = field(default_factory=SessionConfig)

    project_name: Optional[str] = None

    # ── Analysis outputs ─────────────────────────────────────────────────────
    audio_path:            Optional[Path] = None
    vocals_path:           Optional[Path] = None
    instrumental_path:     Optional[Path] = None
    raw_lyrics:            Optional[str]  = None
    words:                 list           = field(default_factory=list)  # [{word, start_s, end_s}]
    intonation:            list           = field(default_factory=list)  # Omni stitched sections
    music_data:            Optional[dict] = None                         # librosa results
    reference_description: Optional[str]  = None                        # Omni image description
    reference_image_path:  Optional[Path] = None                        # uploaded reference image
    genre:                 Optional[str]  = None                         # Omni genre tag
    subgenre:              Optional[str]  = None                         # Omni subgenre tag

    # ── Segmentation ─────────────────────────────────────────────────────────
    scene_k:        Optional[int]                        = None  # number of scenes
    seg_weights:    Optional[tuple[float, float, float]] = None  # (w_struct, w_energy, w_beat)
    proposed_scenes: list                                = field(default_factory=list)  # [{start_s, end_s, frame_count}] from last propose_scenes()

    # ── Phase ────────────────────────────────────────────────────────────────
    phase: str = "upload"

    # ────────────────────────────────────────────────────────────────────────
    # Paths
    # ────────────────────────────────────────────────────────────────────────

    @property
    def prompts_path(self) -> Path:
        return self.session_dir / "prompts.json"

    @property
    def audio_dir(self) -> Path:
        return self.session_dir / "audio"

    @property
    def images_dir(self) -> Path:
        return self.session_dir / "images"

    @property
    def videos_dir(self) -> Path:
        return self.session_dir / "videos"

    # ────────────────────────────────────────────────────────────────────────
    # Phase transitions
    # ────────────────────────────────────────────────────────────────────────

    def advance_phase(self, to: str) -> None:
        """Transition to a new phase and persist immediately."""
        if to not in PHASES:
            raise ValueError(f"Unknown phase: {to!r}")
        self.phase = to
        self.save_meta()

    # ────────────────────────────────────────────────────────────────────────
    # prompts.json
    # ────────────────────────────────────────────────────────────────────────

    def load_prompts(self) -> dict:
        if self.prompts_path.exists():
            return json.loads(self.prompts_path.read_text(encoding="utf-8"))
        return {
            "session_id":  self.session_id,
            "config":      {},
            "style_bible": {},
            "scenes":      {},
        }

    def save_prompts(self, data: dict) -> None:
        self.prompts_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompts_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Style bible
    # ────────────────────────────────────────────────────────────────────────

    def get_style_bible(self) -> dict:
        return self.load_prompts().get("style_bible", {})

    def set_style_bible(self, bible: dict) -> None:
        data = self.load_prompts()
        data["style_bible"] = bible
        self.save_prompts(data)

    # ────────────────────────────────────────────────────────────────────────
    # Scene helpers
    # ────────────────────────────────────────────────────────────────────────

    def get_scene(self, n: int) -> dict:
        """Return scene dict for 1-indexed scene number n."""
        return self.load_prompts()["scenes"].get(str(n), {})

    def update_scene(self, n: int, fields: dict) -> None:
        """Merge fields into scene n and persist."""
        data = self.load_prompts()
        data["scenes"].setdefault(str(n), {}).update(fields)
        self.save_prompts(data)

    def all_scenes(self) -> list[dict]:
        """Return list of scene dicts sorted by scene number (1-indexed)."""
        data = self.load_prompts()
        return [
            data["scenes"][k]
            for k in sorted(data["scenes"].keys(), key=int)
        ]

    def commit_scenes(self, scenes: list[dict]) -> None:
        """
        Write the full scene list to prompts.json, replacing any prior scene data.
        Each scene dict must include at minimum: start_s, end_s, frame_count.
        Called exactly once — after the user approves the segmentation plan.
        """
        data = self.load_prompts()
        data["scenes"] = {
            str(i + 1): {
                "start_s":        scene["start_s"],
                "end_s":          scene["end_s"],
                "frame_count":    scene["frame_count"],
                "label":          scene.get("label", f"Scene {i + 1}"),
                "lyrics_full":    scene.get("lyrics_full", ""),
                "lyric_theme":    scene.get("lyric_theme", ""),
                "intonation_note": scene.get("intonation_note", ""),
                "energy_level":   scene.get("energy_level", ""),
                "location":       scene.get("location", ""),
                "outfit":         scene.get("outfit", ""),
                "rationale":      scene.get("rationale", ""),
                "image_prompt":   scene.get("image_prompt", ""),
                "video_prompt":   scene.get("video_prompt", ""),
                "seed":           scene.get("seed") or self.next_seed(),
                "image_path":     None,
                "video_path":     None,
                "image_status":   "planned",
                "video_status":   "planned",
            }
            for i, scene in enumerate(scenes)
        }
        self.save_prompts(data)

    def next_seed(self) -> int:
        return random.randint(1, 2**31 - 1)

    # ────────────────────────────────────────────────────────────────────────
    # Lyrics helpers
    # ────────────────────────────────────────────────────────────────────────

    def extract_lyrics_window(self, start_s: float, end_s: float) -> str:
        """
        Extract verbatim aligned words that fall within [start_s, end_s].
        Gaps > 0.8s between words become ' / ' in the output.
        """
        window = [w for w in self.words if start_s <= w["start_s"] < end_s]
        if not window:
            return ""
        parts: list[str] = []
        for w in window:
            parts.append(w["word"])
        return " ".join(parts).strip()

    # ────────────────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────────────────

    def save_meta(self) -> None:
        """Write session.json — call after every phase transition."""
        meta: dict = {
            "session_id":   self.session_id,
            "project_name": self.project_name or "",
            "created_at":   (self.session_dir / "session.json").exists()
                            and json.loads((self.session_dir / "session.json")
                                .read_text(encoding="utf-8")).get("created_at")
                            or datetime.datetime.now().isoformat(),
            "phase": self.phase,
            "scene_k":        self.scene_k,
            "seg_weights":    list(self.seg_weights) if self.seg_weights else None,
            "proposed_scenes": self.proposed_scenes or None,
            "reference_image_path":  str(self.reference_image_path)  if self.reference_image_path  else None,
            "reference_description": self.reference_description or None,
            "config": {
                "orientation":    self.config.orientation,
                "width":          self.config.width,
                "height":         self.config.height,
                "fps":            self.config.fps,
                "scene_min_s":    self.config.scene_min_s,
                "scene_max_s":    self.config.scene_max_s,
                "image_workflow":  self.config.image_workflow,
                "video_workflow":  self.config.video_workflow,
                "humo_resolution": self.config.humo_resolution,
                "lora_name":       self.config.lora_name,
                "lora_strength":   self.config.lora_strength,
            },
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_raw_lyrics(self) -> None:
        if self.raw_lyrics:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            (self.audio_dir / "raw_lyrics.txt").write_text(
                self.raw_lyrics, encoding="utf-8"
            )

    def save_words(self) -> None:
        if self.words:
            (self.audio_dir / "aligned.json").write_text(
                json.dumps(self.words, ensure_ascii=False), encoding="utf-8"
            )

    def save_intonation(self) -> None:
        (self.audio_dir / "intonation.json").write_text(
            json.dumps({
                "genre":    self.genre    or "",
                "subgenre": self.subgenre or "",
                "sections": self.intonation,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_music(self) -> None:
        if self.music_data:
            (self.audio_dir / "music.json").write_text(
                json.dumps(self.music_data, ensure_ascii=False), encoding="utf-8"
            )

    # ────────────────────────────────────────────────────────────────────────
    # Reconstruction from disk
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dir(cls, session_dir: Path) -> "Session":
        """Reconstruct a Session from a saved project directory."""
        meta_path = session_dir / "session.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No session.json in {session_dir}")

        meta  = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg_d = meta.get("config", {})
        cfg   = SessionConfig(
            orientation    = cfg_d.get("orientation",    "landscape"),
            width          = cfg_d.get("width",          2048),
            height         = cfg_d.get("height",         1152),
            fps            = cfg_d.get("fps",            25),
            scene_min_s    = cfg_d.get("scene_min_s",    3.0),
            scene_max_s    = cfg_d.get("scene_max_s",    20.0),
            image_workflow  = cfg_d.get("image_workflow",  "zit"),
            video_workflow  = cfg_d.get("video_workflow",  "ltx_humo"),
            humo_resolution = cfg_d.get("humo_resolution", 1280),
            lora_name       = cfg_d.get("lora_name")  or None,
            lora_strength   = cfg_d.get("lora_strength", 0.6),
        )

        sw  = meta.get("seg_weights")
        rip = meta.get("reference_image_path")
        session = cls(
            session_id            = meta["session_id"],
            session_dir           = session_dir,
            config                = cfg,
            project_name          = meta.get("project_name") or None,
            scene_k               = meta.get("scene_k"),
            seg_weights           = tuple(sw) if sw else None,
            phase                 = meta.get("phase") or "upload",
            reference_image_path  = Path(rip) if rip else None,
            reference_description = meta.get("reference_description") or None,
            proposed_scenes       = meta.get("proposed_scenes") or [],
        )

        # ── Audio files ──────────────────────────────────────────────────────
        audio_dir = session_dir / "audio"
        if audio_dir.exists():
            _analysis_names = {
                "vocals.wav", "instrumental.wav", "aligned.json",
                "intonation.json", "music.json", "raw_lyrics.txt",
            }
            for p in sorted(audio_dir.iterdir()):
                if p.is_file() and p.name not in _analysis_names:
                    session.audio_path = p
                    break

            if (audio_dir / "vocals.wav").exists():
                session.vocals_path = audio_dir / "vocals.wav"
            if (audio_dir / "instrumental.wav").exists():
                session.instrumental_path = audio_dir / "instrumental.wav"

            lf = audio_dir / "raw_lyrics.txt"
            if lf.exists():
                session.raw_lyrics = lf.read_text(encoding="utf-8")

            af = audio_dir / "aligned.json"
            if af.exists():
                session.words = json.loads(af.read_text(encoding="utf-8"))

            inf = audio_dir / "intonation.json"
            if inf.exists():
                d = json.loads(inf.read_text(encoding="utf-8"))
                session.intonation = d.get("sections", [])
                session.genre      = d.get("genre")    or None
                session.subgenre   = d.get("subgenre") or None

            mf = audio_dir / "music.json"
            if mf.exists():
                session.music_data = json.loads(mf.read_text(encoding="utf-8"))

        # Phase stored explicitly — only infer if missing (legacy dirs)
        if "phase" not in meta:
            session.phase = session._infer_phase()

        # Reset any in-flight generation statuses left by a crash or restart.
        # "generating" means the server died mid-run — check the actual file on
        # disk to decide whether to promote to "done" or revert to "planned".
        session._reset_stale_generating()

        return session

    def _reset_stale_generating(self) -> None:
        """
        On session load, any scene with status="generating" was interrupted by a
        crash or server restart. Resolve by checking whether the output file
        actually landed on disk:
          - file exists  → promote to "done" (generation completed before crash)
          - no file      → revert to "planned" (generation never finished)
        Persists the corrected prompts.json only if at least one scene was changed.
        """
        if not self.prompts_path.exists():
            return
        data    = self.load_prompts()
        scenes  = data.get("scenes", {})
        changed = False

        for scene in scenes.values():
            if scene.get("image_status") == "generating":
                p = scene.get("image_path")
                scene["image_status"] = "done" if (p and Path(p).exists()) else "planned"
                changed = True

            if scene.get("video_status") == "generating":
                p = scene.get("video_path")
                scene["video_status"] = "done" if (p and Path(p).exists()) else "planned"
                changed = True

        if changed:
            self.save_prompts(data)

    def _infer_phase(self) -> str:
        """Fallback phase inference for legacy session dirs without a stored phase."""
        try:
            data   = self.load_prompts()
            scenes = data.get("scenes", {})
            if scenes:
                vals = list(scenes.values())
                if all(s.get("video_status") == "approved" for s in vals):
                    return "done"
                if any(s.get("video_status") in ("done", "approved", "generating") for s in vals):
                    return "videos"
                if any(s.get("image_status") in ("done", "approved", "generating") for s in vals):
                    return "images"
                if all(s.get("image_prompt") for s in vals):
                    return "prompts"
                if data.get("style_bible"):
                    return "style_bible"
                return "planning"
        except Exception:
            pass

        if self.music_data or self.words or self.vocals_path:
            return "analysis"
        return "upload"
