"""
tools.py
Tool definitions (JSON schema) and dispatch for the 35B agent.

TOOL_DEFINITIONS — passed to the LLM as the tools= parameter.
dispatch()       — routes tool_name + arguments to the correct implementation.

Each tool implementation receives (session, push, **kwargs) where:
  session  — Session dataclass (mutable, shared state)
  push     — async callable(event: str, data: dict) for WebSocket updates
  **kwargs — parsed tool arguments from the LLM

Tools that run ComfyUI generation return immediately with a status message;
the actual generation is awaited inside and pushes card-state updates via `push`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Awaitable

import httpx

from backend import config
from backend.session import Session
from backend.analysis.scenes import snap_frames

logger = logging.getLogger(__name__)

# Type alias for the WebSocket push callback
PushFn = Callable[[str, dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# Tool JSON schema definitions (passed to LLM)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "separate_vocals",
            "description": "Run Demucs HTDemucs to isolate vocals from the backing track. Returns paths to vocals.wav and instrumental.wav.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Path to the uploaded audio file (MP3, WAV, etc.)"}
                },
                "required": ["audio_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "align_lyrics",
            "description": "Force-align confirmed correct lyrics against the isolated vocals using stable-ts (Whisper large-v3). Call AFTER the user has reviewed and confirmed the lyrics text. Returns word-level timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vocals_path": {"type": "string"},
                    "correct_lyrics": {"type": "string", "description": "User-confirmed clean lyrics text. Suno section tags are stripped automatically."}
                },
                "required": ["vocals_path", "correct_lyrics"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_audio",
            "description": "Run Omni intonation analysis on the vocals + librosa signal analysis on the full mix. Returns intonation map, BPM, key, RMS energy, and structural segments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Full mix audio path"},
                    "vocals_path": {"type": "string", "description": "Isolated vocals path for Omni"},
                    "lyrics": {"type": "string", "description": "Full lyrics with Suno section tags (for Omni section labelling)"}
                },
                "required": ["audio_path", "vocals_path", "lyrics"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_reference_image",
            "description": "Send a user-uploaded reference image to Omni. Returns a detailed description of subject, clothing, lighting, setting, colour palette and mood. Stored as reference_description and injected into consistent_elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_scenes",
            "description": (
                "Run the algorithmic scene segmenter and return k scene boundaries "
                "(start_s, end_s, frame_count, duration_s) computed from the audio signal. "
                "Use these timestamps exactly as returned — do NOT manually adjust them. "
                "After receiving the boundaries, enrich each scene with label, lyrics_full, "
                "lyric_theme, intonation_note, energy_level, rationale from your analysis, "
                "present the table to the user, and call set_scenes() with the full enriched list. "
                "Pass reroll=true to resample with random weights for different cut placements "
                "at the same k."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "k":      {"type": "integer", "description": "Number of scenes to generate"},
                    "reroll": {"type": "boolean", "description": "Resample cut weights for variety (same k, different boundary placement)", "default": False},
                },
                "required": ["k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scenes",
            "description": "Commit the scene list to session state. Automatically extracts lyrics_window (verbatim aligned words for that time range) for each scene — used in video prompt lipsync. Call after user confirms the scene plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenes": {
                        "type": "array",
                        "description": "Array of scene objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_s":      {"type": "number"},
                                "end_s":        {"type": "number"},
                                "label":        {"type": "string"},
                                "lyrics_full":  {"type": "string", "description": "All lyrics sung in this scene"},
                                "lyric_theme":  {"type": "string", "description": "Your visual/thematic reading of these lyrics"},
                                "intonation_note": {"type": "string"},
                                "energy_level": {"type": "string", "enum": ["low", "medium", "high", "building", "dropping"]},
                                "location":          {"type": "string", "description": "Location name from the agreed location set"},
                                "establishing_shot": {"type": "boolean", "description": "True if this scene shows only the location — no character. Character description is suppressed from image generation."},
                                "rationale":         {"type": "string"}
                            },
                            "required": ["start_s", "end_s", "label", "lyrics_full", "lyric_theme", "energy_level"]
                        }
                    }
                },
                "required": ["scenes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_style_bible",
            "description": "Store the approved style bible. Called once after user approves it. Injected as shared context for all scene prompt generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "character":      {"type": "string", "description": "Physical appearance only — skin tone, face, hair colour/length/texture, eye colour, build. NO clothing — that varies per scene and belongs in each image_prompt."},
                    "locations": {
                        "type": "array",
                        "description": "2-4 recurring locations for the whole video. Each entry is a short name and concrete description.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":        {"type": "string", "description": "Short location name used to reference it in scene assignments"},
                                "description": {"type": "string", "description": "Concrete visual description — architecture, surfaces, light sources, atmosphere"}
                            },
                            "required": ["name", "description"]
                        }
                    },
                    "cinematography": {"type": "string", "description": "Lens choice, framing rules, camera movement style"},
                    "color_palette":  {"type": "string", "description": "3-5 dominant colours or film/painting references"},
                    "lighting":       {"type": "string", "description": "Quality, direction, colour temperature, shadow character"},
                    "negative":       {"type": "string", "description": "Shared negative prompt — what to avoid in all scenes"}
                },
                "required": ["character", "locations", "cinematography", "color_palette", "lighting", "negative"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_scene_prompts",
            "description": "Write image_prompt and video_prompt for one scene into prompts.json and update the UI card to prompts_ready state. Call once per scene sequentially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_index":    {"type": "integer", "description": "1-indexed scene number"},
                    "image_prompt":   {"type": "string", "description": "Static first-frame description (50-80 words). No character description — that is prepended. End on a specific detail the video prompt opens from."},
                    "video_prompt":   {"type": "string", "description": "Motion from that first frame (50-80 words). Describe how the character performs/delivers — vocal technique, physical expression, camera. Do NOT quote lyrics; they are appended automatically."}
                },
                "required": ["scene_index", "image_prompt", "video_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_images",
            "description": "Run the T2I image generation batch using ltx2_t2i.json. Generates all scenes with image_status=prompts_ready unless scene_numbers is specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific scene numbers to (re)generate. Omit to generate all pending."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_videos",
            "description": "Run the I2V+HuMo video generation batch using ltx2_i2v_humo.json. Only runs scenes with image_status=approved. Omit scene_numbers to generate all approved pending scenes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific scene numbers to (re)generate."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_locations",
            "description": "Assign locations (and optionally establishing_shot) to all scenes in one call. Use this instead of calling update_scene() per scene during location planning — it is a single tool call that handles all assignments at once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "description": "One entry per scene",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scene_index":       {"type": "integer", "description": "1-indexed scene number"},
                                "location":          {"type": "string",  "description": "Location name from the agreed location set"},
                                "establishing_shot": {"type": "boolean", "description": "True if character should be absent (environment only)"}
                            },
                            "required": ["scene_index", "location"]
                        }
                    }
                },
                "required": ["assignments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_outfits",
            "description": "Define 8 outfit options and assign one per scene. Call after proposing outfits alongside locations. Outfit descriptions are injected as clothing context in each scene's image_prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outfits": {
                        "type": "array",
                        "description": "Exactly 8 outfit options",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":        {"type": "string", "description": "Short outfit name used for scene assignment"},
                                "description": {"type": "string", "description": "Specific clothing — garments, colours, textures, fit. 1-2 sentences."}
                            },
                            "required": ["name", "description"]
                        }
                    },
                    "assignments": {
                        "type": "array",
                        "description": "One entry per scene",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scene_index": {"type": "integer", "description": "1-indexed scene number"},
                                "outfit":      {"type": "string",  "description": "Outfit name from the outfits list"}
                            },
                            "required": ["scene_index", "outfit"]
                        }
                    }
                },
                "required": ["outfits", "assignments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_scene",
            "description": "Update any scene field in prompts.json and push the change live to the UI. For bulk location assignment use set_locations() instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_index": {"type": "integer"},
                    "fields":      {"type": "object"}
                },
                "required": ["scene_index", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene",
            "description": "Read the current data for a specific scene by index. Call this FIRST when asked to rewrite or review a specific scene — do NOT rely on conversation history for scene data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_index": {"type": "integer", "description": "1-based scene index"}
                },
                "required": ["scene_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_scene",
            "description": "Reset a scene's generation state. Archives existing image/video, clears paths and statuses to 'planned'. Required before changing timing on a scene that already has generated assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_index": {"type": "integer"},
                    "reset":       {"type": "string", "enum": ["image", "video", "both"]}
                },
                "required": ["scene_index", "reset"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_archive",
            "description": "Delete images/archive/ and videos/archive/ for the current session. Called after all videos are approved.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_final",
            "description": "Concatenate all approved scene MP4s in order and mix with original audio. Produces the final music video file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_filename": {"type": "string", "description": "Optional output filename (default: final.mp4)"}
                }
            }
        }
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _separate_vocals(session: Session, push: PushFn, audio_path: str) -> str:
    from backend.analysis.separation import separate
    await push("status", {"message": "Separating vocals with Demucs..."})
    paths = await asyncio.to_thread(
        separate, audio_path, session.audio_dir
    )
    session.vocals_path       = paths["vocals"]
    session.instrumental_path = paths["instrumental"]
    session.audio_path        = Path(audio_path)

    # File existence check + confirmation
    vocals_ok = Path(paths["vocals"]).exists()
    instr_ok  = Path(paths["instrumental"]).exists()
    if vocals_ok and instr_ok:
        vocals_mb = round(Path(paths["vocals"]).stat().st_size / 1024 / 1024, 1)
        instr_mb  = round(Path(paths["instrumental"]).stat().st_size / 1024 / 1024, 1)
        await push("step_done", {"message": f"Vocals separated — vocals.wav ({vocals_mb} MB) · instrumental.wav ({instr_mb} MB) · saved to {session.audio_dir.name}/"})
    else:
        missing = ", ".join(p for p, ok in [("vocals.wav", vocals_ok), ("instrumental.wav", instr_ok)] if not ok)
        await push("status", {"message": f"Warning: separation completed but {missing} not found on disk"})

    return json.dumps({
        "vocals_path":       str(paths["vocals"]),
        "instrumental_path": str(paths["instrumental"]),
        "duration_s":        round(__import__("soundfile").info(str(paths["vocals"])).duration, 2),
    })


async def _align_lyrics(session: Session, push: PushFn, vocals_path: str, correct_lyrics: str) -> str:
    from backend.analysis.aligner import align
    await push("status", {"message": "Aligning lyrics with stable-ts..."})
    words = await asyncio.to_thread(align, vocals_path, correct_lyrics)
    session.words = words
    session.save_words()   # persist so session survives restart
    if words:
        span = f"{words[0]['start_s']:.1f}s → {words[-1]['end_s']:.1f}s"
        saved_ok = (session.audio_dir / "aligned.json").exists()
        save_note = " · saved" if saved_ok else " · save failed"
        await push("step_done", {"message": f"Lyrics aligned — {len(words)} words placed ({span}){save_note}"})
    else:
        await push("status", {"message": "Warning: alignment returned no words — check that lyrics match the audio"})
    return json.dumps({
        "words_placed": len(words),
        "first_word_s": words[0]["start_s"] if words else None,
        "last_word_s":  words[-1]["end_s"]   if words else None,
    })


async def _analyze_audio(session: Session, push: PushFn, audio_path: str, vocals_path: str, lyrics: str) -> str:
    from backend.analysis.omni import analyze_intonation
    from backend.analysis.music import analyze

    await push("status", {"message": "Running Omni intonation analysis..."})
    omni_result      = await asyncio.to_thread(analyze_intonation, audio_path)
    session.genre    = omni_result.get("genre", "")
    session.subgenre = omni_result.get("subgenre", "")
    session.intonation = omni_result.get("sections", [])

    await push("status", {"message": "Running librosa signal analysis..."})
    music_data = await asyncio.to_thread(analyze, audio_path)
    session.music_data = music_data

    session.save_intonation()   # persist genre + sections
    session.save_music()        # persist librosa results

    bpm      = music_data["bpm"]
    key_mode = f"{music_data['key']} {music_data['mode']}"
    dur_min  = int(music_data["duration"] // 60)
    dur_sec  = int(music_data["duration"] % 60)
    n_sects  = len(session.intonation)
    genre_str = f" · {session.genre}" + (f" / {session.subgenre}" if session.subgenre else "") if session.genre else ""
    intonation_saved = (session.audio_dir / "intonation.json").exists()
    music_saved      = (session.audio_dir / "music.json").exists()
    save_note = " · saved" if (intonation_saved and music_saved) else " · save failed"
    await push("step_done", {"message": f"Audio analysed — {bpm} BPM · {key_mode} · {dur_min}:{dur_sec:02d} · {n_sects} sections{genre_str}{save_note}"})
    await push("status", {"message": "Analysis complete — director is preparing your briefing…"})

    return json.dumps({
        "bpm":       bpm,
        "key":       key_mode,
        "duration":  music_data["duration"],
        "genre":     session.genre,
        "subgenre":  session.subgenre,
        "sections":  n_sects,
        "intonation_summary": [
            {"label": s.get("label", ""), "energy": s.get("energy", ""), "mood": s.get("mood", "")}
            for s in session.intonation
        ],
    })


async def _describe_reference_image(session: Session, push: PushFn, image_path: str) -> str:
    from backend.analysis.omni import describe_image
    await push("status", {"message": "Analysing reference image with Omni..."})
    description = await asyncio.to_thread(describe_image, image_path)
    session.reference_description = description
    return json.dumps({"reference_description": description})



async def _propose_scenes(session: Session, push: PushFn, k: int, reroll: bool = False) -> str:
    from backend.analysis.segmentation import segment, random_weights, default_weights
    weights = random_weights() if reroll else default_weights()
    session.scene_k     = k
    session.seg_weights = weights
    session.save_meta()
    await push("status", {"message": f"Computing {k} scene boundaries ({'reroll' if reroll else 'default weights'})..."})
    scenes = await asyncio.to_thread(
        segment,
        session.audio_path,
        k=k,
        fps=session.config.fps,
        min_s=session.config.scene_min_s,
        max_s=session.config.scene_max_s,
        weights=weights,
    )
    actual_k = len(scenes)
    # Store proposed boundaries on session so _set_scenes can enforce exact match
    session.proposed_scenes = [
        {"start_s": s["start_s"], "end_s": s["end_s"], "frame_count": s["frame_count"]}
        for s in scenes
    ]
    session.scene_k = actual_k  # update to actual after max enforcement
    session.save_meta()
    note = f"→ {actual_k} scenes" + (f" (requested {k}, {actual_k - k} added by max-duration enforcement)" if actual_k != k else "")
    await push("step_done", {"message": f"Segmentation complete: {note}"})
    return json.dumps({
        "scenes": [
            {
                "start_s":     s["start_s"],
                "end_s":       s["end_s"],
                "frame_count": s["frame_count"],
                "duration_s":  round(s["end_s"] - s["start_s"], 2),
                "lyrics":      session.extract_lyrics_window(s["start_s"], s["end_s"]),
            }
            for s in scenes
        ],
        "actual_k": actual_k,
        "weights_used": {"w_structure": weights[0], "w_energy": weights[1], "w_beat": weights[2]},
        "note": f"IMPORTANT: You MUST pass exactly {actual_k} scenes to set_scenes(). Copy start_s, end_s, and frame_count verbatim — do not add, split, merge, or modify any boundary.",
    })


async def _set_scenes(session: Session, push: PushFn, scenes: list[dict]) -> str:
    data    = session.load_prompts()
    fps     = session.config.fps
    min_s   = session.config.scene_min_s
    max_s   = session.config.scene_max_s

    # ── Server-side boundary enforcement ─────────────────────────────────────
    # Reject immediately if scene count doesn't match what propose_scenes() returned.
    if session.proposed_scenes:
        expected = len(session.proposed_scenes)
        if len(scenes) != expected:
            return json.dumps({
                "error": (
                    f"Scene count mismatch: propose_scenes() returned {expected} boundaries "
                    f"but you submitted {len(scenes)}. "
                    "You MUST NOT add, split, merge, or remove boundaries. "
                    f"Call set_scenes() with exactly {expected} scenes, "
                    "copying start_s/end_s/frame_count verbatim from the propose_scenes() response."
                )
            })
        # Also validate that start_s/end_s haven't been fabricated
        tol = 0.1  # 100ms tolerance for float rounding
        boundary_errors = []
        for i, (submitted, proposed) in enumerate(zip(scenes, session.proposed_scenes), 1):
            if abs(submitted.get("start_s", 0) - proposed["start_s"]) > tol:
                boundary_errors.append(
                    f"Scene {i}: submitted start_s={submitted.get('start_s')} "
                    f"but propose_scenes() returned {proposed['start_s']}"
                )
            if abs(submitted.get("end_s", 0) - proposed["end_s"]) > tol:
                boundary_errors.append(
                    f"Scene {i}: submitted end_s={submitted.get('end_s')} "
                    f"but propose_scenes() returned {proposed['end_s']}"
                )
        if boundary_errors:
            return json.dumps({
                "error": "Boundary values modified — copy them verbatim from propose_scenes():",
                "boundary_errors": boundary_errors,
            })

    # Hard constraint validation — reject the whole call so the model must fix it
    violations = []
    for i, sc in enumerate(scenes, 1):
        dur = round(sc["end_s"] - sc["start_s"], 3)
        if dur > max_s:
            violations.append(f"Scene {i} ({sc.get('label','?')}): {dur:.1f}s exceeds max {max_s}s — split it.")
        elif dur < min_s:
            violations.append(f"Scene {i} ({sc.get('label','?')}): {dur:.1f}s is below min {min_s}s — merge or extend it.")
    if violations:
        return json.dumps({
            "error": "Scene duration constraint violated — fix these scenes and call set_scenes() again:",
            "violations": violations,
        })

    new_scenes: dict[str, Any] = {}
    for i, sc in enumerate(scenes, 1):
        start_s = sc["start_s"]
        end_s   = sc["end_s"]
        fc      = snap_frames(end_s - start_s, fps)

        # Extract verbatim lyrics for this time window (lipsync anchor + display)
        lyrics_window = session.extract_lyrics_window(start_s, end_s)

        new_scenes[str(i)] = {
            "start_s":        round(start_s, 3),
            "end_s":          round(end_s,   3),
            "frame_count":    fc,
            "label":          sc.get("label", f"Scene {i}"),
            "lyrics_full":    lyrics_window,
            "lyrics_window":  lyrics_window,
            "lyric_theme":    sc.get("lyric_theme", ""),
            "intonation_note":sc.get("intonation_note", ""),
            "energy_level":   sc.get("energy_level", "medium"),
            "location":          sc.get("location", ""),
            "outfit":            sc.get("outfit", ""),
            "establishing_shot": sc.get("establishing_shot", False),
            "rationale":         sc.get("rationale", ""),
            "image_prompt":   "",
            "video_prompt":   "",
            "seed":           session.next_seed(),
            "image_path":     None,
            "video_path":     None,
            "image_status":   "planned",
            "video_status":   "planned",
        }
        await push("scene_update", {"scene_index": i, "scene": new_scenes[str(i)]})

    data["scenes"] = new_scenes
    session.save_prompts(data)
    session.phase = "prompts"

    return json.dumps({
        "scenes_set": len(new_scenes),
        "note":       "lyrics_window extracted for each scene — use it verbatim in double quotes in every video_prompt for lipsync.",
        "scenes":     [
            {"index": k, "label": v["label"], "start_s": v["start_s"],
             "end_s": v["end_s"], "lyrics_window": v["lyrics_window"]}
            for k, v in new_scenes.items()
        ]
    })


async def _set_style_bible(session: Session, push: PushFn, **kwargs) -> str:
    data = session.load_prompts()
    data["style_bible"] = kwargs
    session.save_prompts(data)
    await push("style_bible_update", {"style_bible": kwargs})
    return json.dumps({"status": "style_bible_saved"})


async def _set_scene_prompts(session: Session, push: PushFn,
                             scene_index: int, image_prompt: str, video_prompt: str) -> str:
    # Always append verbatim lyrics at the end for lipsync — server owns this, not the model
    scene = session.get_scene(scene_index)
    lw = scene.get("lyrics_window", "")
    if lw:
        lw_clean = lw.strip()
        video_prompt = video_prompt.rstrip().rstrip('.') + f' The character delivers the words "{lw_clean}"'

    # If the scene already has a generated image (status=done), revert to
    # prompts_ready so the Regen Unapproved button picks it up in the next pass.
    # Approved scenes are left as-is — the user explicitly kept that image.
    current_status = scene.get("image_status", "planned")
    new_status = "prompts_ready" if current_status != "approved" else "approved"

    session.update_scene(scene_index, {
        "image_prompt":   image_prompt,
        "video_prompt":   video_prompt,
        "image_status":   new_status,
    })
    await push("scene_update", {
        "scene_index": scene_index,
        "scene": session.get_scene(scene_index),
    })

    # Advance phase to images once all scenes have prompts
    all_scenes = session.all_scenes()
    if all_scenes and all(s.get("image_status") == "prompts_ready" for s in all_scenes):
        session.advance_phase("images")

    return json.dumps({"status": "ok", "scene_index": scene_index})


async def _evict_llm(push: PushFn) -> None:
    """Unload the LLM from VRAM before ComfyUI generation to avoid OOM."""
    agent_url = os.environ.get("AGENT_URL", "http://127.0.0.1:8000")
    await push("status", {"message": "Freeing VRAM — unloading LLM..."})
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{agent_url}/unload")
            logger.info("LLM evict: HTTP %d", resp.status_code)
    except Exception as e:
        logger.info("LLM evict skipped (%s) — proceeding", e)


async def _generate_images(session: Session, push: PushFn,
                           scene_numbers: list[int] | None = None) -> str:
    import json as _json
    from backend.comfyui.patch import apply
    from backend.comfyui import client as comfy

    data = session.load_prompts()
    cfg  = session.config
    sb   = data.get("style_bible", {})

    if cfg.image_workflow.startswith("user/"):
        stem     = cfg.image_workflow[5:]
        t2i_wf   = _json.load(open(f"backend/comfyui/workflows/user/{stem}.json",         encoding="utf-8"))
        node_map = _json.load(open(f"backend/comfyui/workflows/user/{stem}.nodemap.json"))
    elif cfg.image_workflow == "qie":
        t2i_wf   = _json.load(open("backend/comfyui/workflows/qie_t2i.json",  encoding="utf-8"))
        node_map = _json.load(open("backend/comfyui/node_map_qie.json"))
    else:
        t2i_wf   = _json.load(open("backend/comfyui/workflows/ltx2_t2i.json", encoding="utf-8"))
        node_map = _json.load(open("backend/comfyui/node_map_t2i.json"))

    results  = {}

    # Only physical character description is consistent across all scenes.
    # Lighting, palette, and setting vary per scene — they live in the image_prompt.
    with_char = sb.get("character", "").strip()   # stable: skin, hair, eyes, build
    # No character prepended for establishing shots — handled per-scene below

    targets = scene_numbers or [
        int(k) for k, v in data["scenes"].items()
        if v.get("image_status") == "prompts_ready"
    ]

    # Regen: assign a fresh seed to each explicitly requested scene so the
    # result is different from the previous run.
    if scene_numbers is not None:
        for n in targets:
            session.update_scene(n, {"seed": session.next_seed()})
        data = session.load_prompts()

    if not targets:
        # Give the agent a clear, actionable error rather than silently doing nothing
        blank = [k for k, v in data["scenes"].items()
                 if not v.get("image_prompt") or not v.get("video_prompt")]
        if blank:
            return json.dumps({
                "error": (
                    f"Cannot generate images — {len(blank)} scene(s) have no prompts: {blank}. "
                    "You MUST call set_scene_prompts(scene_index, image_prompt, video_prompt) "
                    "for EVERY scene before calling generate_images(). Do this now."
                )
            })
        return json.dumps({"results": {}, "note": "No scenes in prompts_ready state."})

    await _evict_llm(push)
    await push("status", {"message": f"Generating images for {len(targets)} scene{'s' if len(targets) != 1 else ''}..."})

    for i, n in enumerate(targets, 1):
        scene = data["scenes"].get(str(n))
        if not scene:
            continue

        await push("status", {"message": f"Scene {n} ({i}/{len(targets)}) — submitting to ComfyUI..."})

        # Establishing shots show environment only — no character prepended
        consistent = "" if scene.get("establishing_shot") else with_char
        positive   = f"{consistent}\n{scene['image_prompt']}".strip() if consistent else scene['image_prompt']
        save_key = f"sessions/{session.session_id}/images/scene_{n}"

        negative = cfg.base_negative + ("\n" + sb.get("negative", "")).strip()
        patches = {
            node_map["positive"]:        {"value": positive},
            node_map["negative_prompt"]: {"value": negative},
            node_map["seed"]:            {"value": scene["seed"]},
            node_map["width"]:           {"value": cfg.width},
            node_map["height"]:          {"value": cfg.height},
            node_map["save"]:            {"filename_prefix": save_key},
        }
        if "lora" in node_map and cfg.lora_name:
            patches[node_map["lora"]] = {
                "lora_name":      cfg.lora_name,
                "strength_model": cfg.lora_strength,
            }
        if cfg.image_workflow == "qie":
            if session.reference_image_path and session.reference_image_path.exists():
                ref_fname = comfy.stage_image(
                    session.reference_image_path,
                    f"session_{session.session_id}_ref.png",
                )
                if "load_image" in node_map:
                    patches[node_map["load_image"]] = {"image": ref_fname}

        session.update_scene(n, {"image_status": "generating"})
        await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})

        try:
            patched  = apply(t2i_wf, patches)
            pid      = await asyncio.to_thread(comfy.submit, patched)
            outputs  = await asyncio.to_thread(comfy.poll, pid, 2.0, 600.0)
            comfy_path = comfy.get_image_path(outputs, node_map["save"])

            # Copy from ComfyUI output dir into session_dir/images/ so the
            # file server (which is rooted at session_dir) can serve it.
            session.images_dir.mkdir(parents=True, exist_ok=True)
            local_path = session.images_dir / f"scene_{n}.png"
            existing = scene.get("image_path")
            if existing and Path(existing).exists():
                _archive(Path(existing), session.images_dir / "archive", n)
            await asyncio.to_thread(shutil.copy2, str(comfy_path), str(local_path))

            session.update_scene(n, {"image_path": str(local_path), "image_status": "done"})
            await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})
            await push("step_done", {"message": f"Scene {n} image done"})
            results[n] = "done"

        except Exception as e:
            logger.error("Image gen failed for scene %d: %s", n, e)
            # Revert to prompts_ready so the batch can be retried without intervention
            session.update_scene(n, {"image_status": "prompts_ready"})
            await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})
            results[n] = f"error: {e}"

    done_count = sum(1 for v in results.values() if v == "done")
    await push("step_done", {"message": f"Image batch complete — {done_count}/{len(targets)} succeeded"})

    if done_count == 0 and targets:
        first_err = next((v for v in results.values() if v != "done"), "")
        return json.dumps({
            "results": results,
            "error": (
                "ALL scenes failed — ComfyUI is likely not running or not reachable. "
                f"First error: {first_err}. "
                "All scenes have been reverted to prompts_ready. "
                "Tell the user to start ComfyUI and then call generate_images() again. "
                "Do NOT rewrite scenes or prompts."
            ),
        })
    return json.dumps({"results": results})


async def _generate_videos(session: Session, push: PushFn,
                           scene_numbers: list[int] | None = None,
                           workflow: str = "ltx_humo",
                           cancel_check=None) -> str:
    import json as _json
    from backend.comfyui.patch import apply
    from backend.comfyui import client as comfy

    data = session.load_prompts()
    if workflow.startswith("user/"):
        stem     = workflow[5:]
        i2v_wf   = _json.load(open(f"backend/comfyui/workflows/user/{stem}.json",         encoding="utf-8"))
        node_map = _json.load(open(f"backend/comfyui/workflows/user/{stem}.nodemap.json"))
    elif workflow == "ltx":
        i2v_wf   = _json.load(open("backend/comfyui/workflows/ltx2_i2v.json",         encoding="utf-8"))
        node_map = _json.load(open("backend/comfyui/node_map_i2v_ltx.json"))
    elif workflow == "ltx_humo_hq":
        i2v_wf   = _json.load(open("backend/comfyui/workflows/ltx2_i2v_humo_hq.json", encoding="utf-8"))
        node_map = _json.load(open("backend/comfyui/node_map_i2v_humo_hq.json"))
    else:  # ltx_humo
        i2v_wf   = _json.load(open("backend/comfyui/workflows/ltx2_i2v_humo.json",    encoding="utf-8"))
        node_map = _json.load(open("backend/comfyui/node_map_i2v.json"))
    sb       = data.get("style_bible", {})
    cfg      = session.config

    # LTX 2.3 dimension caps: portrait max 1080×1920, landscape max 2560×1440
    if cfg.orientation == "portrait":
        i2v_w, i2v_h = 1080, 1920
    else:
        i2v_w, i2v_h = 2560, 1440
    results  = {}

    targets = scene_numbers or [
        int(k) for k, v in data["scenes"].items()
        if v.get("image_status") == "approved" and v.get("video_status") in ("planned", None)
    ]

    # Regen: assign a fresh seed to each explicitly requested scene.
    if scene_numbers is not None:
        for n in targets:
            session.update_scene(n, {"seed": session.next_seed()})
        data = session.load_prompts()

    if not targets:
        return json.dumps({"results": {}, "note": "No approved scenes to generate."})

    await _evict_llm(push)
    await push("status", {"message": f"Generating videos for {len(targets)} scene{'s' if len(targets) != 1 else ''}..."})

    # Stage audio once
    audio_filename = f"session_{session.session_id}.wav"
    await asyncio.to_thread(comfy.stage_audio, session.audio_path, audio_filename)

    negative = "\n".join(filter(None, [cfg.base_negative, sb.get("negative", "")]))

    # Compute cumulative actual start times based on frame counts so audio
    # conditioning matches where each clip lands in the final concatenation.
    # scene.start_s is the beat-grid boundary; cumulative frame time may differ
    # slightly due to 8k+1 snapping. Using cumulative time keeps lipsync tight.
    first_scene_num   = min(int(k) for k in data["scenes"])
    first_scene_start = data["scenes"][str(first_scene_num)].get("start_s", 0.0)
    cumulative_s: dict[int, float] = {}
    acc = first_scene_start
    for k in sorted(data["scenes"].keys(), key=int):
        cumulative_s[int(k)] = acc
        acc += data["scenes"][k]["frame_count"] / cfg.fps

    for i, n in enumerate(targets, 1):
        if cancel_check and cancel_check():
            await push("status", {"message": "Video generation cancelled — stopping after current scene."})
            break

        scene = data["scenes"].get(str(n))
        if not scene or not scene.get("image_path"):
            continue

        await push("status", {"message": f"Scene {n} ({i}/{len(targets)}) — submitting to ComfyUI..."})

        img_filename = f"session_{session.session_id}_scene_{n}.png"
        await asyncio.to_thread(comfy.stage_image, scene["image_path"], img_filename)

        save_key = f"sessions/{session.session_id}/videos/scene_{n}"

        start_s = cumulative_s.get(n, scene["start_s"])

        # Audio patch: HuMo workflow uses separate primitive nodes for start/length;
        # LTX-only workflow puts start_time and duration directly on node 746.
        if "audio_start" in node_map:
            audio_patch = {"audio": audio_filename}
            patches_audio_start = {node_map["audio_start"]: {"value": start_s}}
        else:
            audio_patch = {"audio": audio_filename, "start_time": start_s}
            patches_audio_start = {}

        # If the workflow generates at a different fps than the session (e.g. HQ
        # workflow generates LTX at 50fps while session frame_counts are at cfg.fps),
        # rescale to the correct frame count for this workflow's generation fps.
        gen_fps = node_map.get("generation_fps", cfg.fps)
        if gen_fps != cfg.fps:
            fc = snap_frames(scene["frame_count"] / cfg.fps, gen_fps)
        else:
            fc = scene["frame_count"]
        if workflow == "ltx_humo":
            fc = max(fc, 81)

        patches = {
            node_map["video_prompt"]:  {"value": scene["video_prompt"]},
            node_map["start_frame"]:   {"image": img_filename},
            node_map["audio_file"]:    audio_patch,
            node_map["frame_count"]:   {"value": fc},
            node_map["save"]:          {"filename_prefix": save_key},
            **patches_audio_start,
        }
        # LTX dimension + seed patches (absent in HuMo-only)
        if "width" in node_map:
            patches[node_map["width"]]  = {"value": i2v_w}
            patches[node_map["height"]] = {"value": i2v_h}
        if "ltx_seed" in node_map:
            nid = node_map["ltx_seed"]
            if i2v_wf.get(nid, {}).get("class_type") == "PrimitiveInt":
                patches[nid] = {"value": scene["seed"]}
            else:
                patches[nid] = {"noise_seed": scene["seed"]}
        if "ltx_seed_2" in node_map:
            nid = node_map["ltx_seed_2"]
            if i2v_wf.get(nid, {}).get("class_type") == "PrimitiveInt":
                patches[nid] = {"value": scene["seed"] + 1}
            else:
                patches[nid] = {"noise_seed": scene["seed"] + 1}
        if "ltxv_conditioning" in node_map:
            patches[node_map["ltxv_conditioning"]] = {"frame_rate": cfg.fps}
        # Shared negative (PrimitiveStringMultiline)
        if "negative_prompt" in node_map:
            patches[node_map["negative_prompt"]] = {"value": negative}
        # HuMo-specific patches
        if "humo_negative" in node_map:
            patches[node_map["humo_negative"]] = {"text": negative}
        if "humo_negative_str" in node_map:
            patches[node_map["humo_negative_str"]] = {"String": negative}
        if "humo_seed" in node_map:
            patches[node_map["humo_seed"]] = {"seed": scene["seed"] + 2}
        if "humo_seed_value" in node_map:
            patches[node_map["humo_seed_value"]] = {"value": scene["seed"] + 2}
        if "humo_long_edge" in node_map:
            patches[node_map["humo_long_edge"]] = {"value": cfg.humo_resolution}
        if "it_width" in node_map:
            p_w  = node_map.get("it_portrait_w", 480)
            p_h  = node_map.get("it_portrait_h", 864)
            it_w = p_w if cfg.orientation == "portrait" else p_h
            it_h = p_h if cfg.orientation == "portrait" else p_w
            patches[node_map["it_width"]]  = {"value": it_w}
            patches[node_map["it_height"]] = {"value": it_h}
        if "create_video" in node_map:
            patches[node_map["create_video"]] = {"fps": cfg.fps}

        session.update_scene(n, {"video_status": "generating"})
        await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})

        try:
            patched    = apply(i2v_wf, patches)
            pid        = await asyncio.to_thread(comfy.submit, patched)
            outputs    = await asyncio.to_thread(comfy.poll, pid, 5.0, 3600.0)
            comfy_path = comfy.get_video_path(outputs, node_map["save"])

            # Copy from ComfyUI output dir into session_dir/videos/
            session.videos_dir.mkdir(parents=True, exist_ok=True)
            local_path = session.videos_dir / f"scene_{n}.mp4"
            existing = scene.get("video_path")
            if existing and Path(existing).exists():
                _archive(Path(existing), session.videos_dir / "archive", n)
            await asyncio.to_thread(shutil.copy2, str(comfy_path), str(local_path))

            session.update_scene(n, {"video_path": str(local_path), "video_status": "done"})
            await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})
            await push("step_done", {"message": f"Scene {n} video done"})
            results[n] = "done"

        except Exception as e:
            logger.error("Video gen failed for scene %d: %s", n, e)
            # Revert to planned so the batch can be retried without intervention
            session.update_scene(n, {"video_status": "planned"})
            await push("scene_update", {"scene_index": n, "scene": session.get_scene(n)})
            results[n] = f"error: {e}"

    done_count = sum(1 for v in results.values() if v == "done")
    await push("step_done", {"message": f"Video batch complete — {done_count}/{len(targets)} succeeded"})

    if done_count == 0 and targets:
        first_err = next((v for v in results.values() if v != "done"), "")
        return json.dumps({
            "results": results,
            "error": (
                "ALL scenes failed — ComfyUI is likely not running or not reachable. "
                f"First error: {first_err}. "
                "All scenes have been reverted to approved state. "
                "Tell the user to start ComfyUI and then call generate_videos() again. "
                "Do NOT rewrite scenes or prompts."
            ),
        })
    return json.dumps({"results": results})


async def _set_locations(session: Session, push: PushFn,
                         assignments: list[dict]) -> str:
    """
    Bulk location assignment — replaces N individual update_scene() calls.

    Each entry: {scene_index, location, establishing_shot (optional)}.
    Pushes the full updated scene object for each entry so the frontend
    always receives complete data (no partial-merge NaN risk).
    """
    for a in assignments:
        idx = int(a["scene_index"])
        fields: dict = {"location": a["location"]}
        if "establishing_shot" in a:
            fields["establishing_shot"] = bool(a["establishing_shot"])
        session.update_scene(idx, fields)
        await push("scene_update", {"scene_index": idx, "scene": session.get_scene(idx)})
    return json.dumps({"status": "ok", "updated": len(assignments)})


async def _set_outfits(session: Session, push: PushFn,
                       outfits: list[dict], assignments: list[dict]) -> str:
    """
    Store 8 outfit definitions in prompts.json and assign one per scene.
    Outfit descriptions are injected as clothing context per scene during prompt gen.
    """
    data = session.load_prompts()

    # Store outfit definitions at top level of prompts.json
    data["outfits"] = {o["name"]: o["description"] for o in outfits}

    # Apply per-scene assignments
    for a in assignments:
        idx = int(a["scene_index"])
        data["scenes"].setdefault(str(idx), {})["outfit"] = a["outfit"]

    session.save_prompts(data)

    # Push scene updates so storyboard cards reflect the assignment
    for a in assignments:
        idx = int(a["scene_index"])
        await push("scene_update", {"scene_index": idx, "scene": session.get_scene(idx)})

    return json.dumps({
        "status": "ok",
        "outfits_defined": len(outfits),
        "scenes_assigned": len(assignments),
    })


async def _update_scene(session: Session, push: PushFn,
                        scene_index: int, fields: dict) -> str:
    # Recalculate frame_count if timing changed + enforce hard duration constraints
    if "start_s" in fields or "end_s" in fields:
        scene = session.get_scene(scene_index)
        start = fields.get("start_s", scene.get("start_s", 0))
        end   = fields.get("end_s",   scene.get("end_s", 0))
        dur   = round(end - start, 3)
        min_s = session.config.scene_min_s
        max_s = session.config.scene_max_s
        if dur > max_s:
            return json.dumps({"error": f"Scene {scene_index} duration {dur:.1f}s exceeds hard max {max_s}s — adjust end_s."})
        if dur < min_s:
            return json.dumps({"error": f"Scene {scene_index} duration {dur:.1f}s is below hard min {min_s}s — adjust end_s."})
        fields["frame_count"] = snap_frames(end - start, session.config.fps)
        if session.words:
            fields["lyrics_window"] = session.extract_lyrics_window(start, end)

    session.update_scene(scene_index, fields)
    # Push the full scene (not just fields) so the frontend always has complete data
    await push("scene_update", {"scene_index": scene_index, "scene": session.get_scene(scene_index)})
    return json.dumps({"status": "ok"})


async def _get_scene(session: Session, push: PushFn, scene_index: int) -> str:
    scene = session.get_scene(scene_index)
    if not scene:
        return json.dumps({"error": f"Scene {scene_index} not found."})
    return json.dumps({"scene_index": scene_index, "scene": scene})


async def _reset_scene(session: Session, push: PushFn,
                       scene_index: int, reset: str) -> str:
    scene = session.get_scene(scene_index)
    updates: dict = {}
    comfy_out = Path(os.environ.get("COMFYUI_OUTPUT_DIR", ""))

    if reset in ("image", "both"):
        if scene.get("image_path") and Path(scene["image_path"]).exists():
            _archive(Path(scene["image_path"]), session.images_dir / "archive", scene_index)
        # Also archive from ComfyUI output dir so next regen saves as _00001_
        if comfy_out:
            comfy_img = comfy_out / f"sessions/{session.session_id}/images/scene_{scene_index}_00001_.png"
            if comfy_img.exists():
                _archive(comfy_img, comfy_out / f"sessions/{session.session_id}/images/archive", scene_index)
        # Restore to prompts_ready if prompts still exist, otherwise planned
        post_img_status = "prompts_ready" if (scene.get("image_prompt") and scene.get("video_prompt")) else "planned"
        updates.update({"image_path": None, "image_status": post_img_status})

    if reset in ("video", "both"):
        if scene.get("video_path") and Path(scene["video_path"]).exists():
            _archive(Path(scene["video_path"]), session.videos_dir / "archive", scene_index)
        # Also archive from ComfyUI output dir so next regen saves as _00001_
        if comfy_out:
            comfy_vid = comfy_out / f"sessions/{session.session_id}/videos/scene_{scene_index}_00001_.mp4"
            if comfy_vid.exists():
                _archive(comfy_vid, comfy_out / f"sessions/{session.session_id}/videos/archive", scene_index)
        updates.update({"video_path": None, "video_status": "planned"})

    session.update_scene(scene_index, updates)
    await push("scene_update", {"scene_index": scene_index, "scene": session.get_scene(scene_index)})
    return json.dumps({"status": "reset", "scene_index": scene_index, "reset": reset})


async def _clear_archive(session: Session, push: PushFn) -> str:
    cleared = []
    for d in [session.images_dir / "archive", session.videos_dir / "archive"]:
        if d.exists():
            shutil.rmtree(str(d))
            cleared.append(str(d))
    return json.dumps({"cleared": cleared})


async def _export_final(session: Session, push: PushFn,
                        output_filename: str = "final.mp4") -> str:
    from backend.render import export

    scenes = [s for s in session.all_scenes() if s.get("video_status") == "approved"]
    if not scenes:
        return json.dumps({"error": "No approved scenes to export."})

    clip_paths  = [s["video_path"] for s in scenes]
    output_path = session.session_dir / output_filename
    first_start = scenes[0]["start_s"]

    await push("status", {"message": f"Rendering final export ({len(scenes)} scenes)..."})
    result = await asyncio.to_thread(
        export, clip_paths, session.audio_path, output_path, first_start
    )
    return json.dumps({"output_path": str(result), "scenes": len(scenes)})


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------

def _archive(src: Path, archive_dir: Path, scene_index: int) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem    = src.stem
    suffix  = src.suffix
    version = 1
    while (archive_dir / f"{stem}_v{version}{suffix}").exists():
        version += 1
    shutil.move(str(src), str(archive_dir / f"{stem}_v{version}{suffix}"))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "separate_vocals":        _separate_vocals,
    "align_lyrics":           _align_lyrics,
    "analyze_audio":          _analyze_audio,
    "describe_reference_image": _describe_reference_image,
    "propose_scenes":         _propose_scenes,
    "set_scenes":             _set_scenes,
    "set_style_bible":        _set_style_bible,
    "set_scene_prompts":      _set_scene_prompts,
    "generate_images":        _generate_images,
    "generate_videos":        _generate_videos,
    "set_locations":          _set_locations,
    "set_outfits":            _set_outfits,
    "get_scene":              _get_scene,
    "update_scene":           _update_scene,
    "reset_scene":            _reset_scene,
    "clear_archive":          _clear_archive,
    "export_final":           _export_final,
}


# Public batch functions called from main.py REST endpoints
async def generate_images_batch(session: Session, scene_numbers: list[int] | None, push: PushFn) -> None:
    await _generate_images(session=session, push=push, scene_numbers=scene_numbers)

async def generate_videos_batch(session: Session, scene_numbers: list[int] | None, push: PushFn,
                                cancel_check=None) -> None:
    await _generate_videos(session=session, push=push, scene_numbers=scene_numbers,
                           workflow=session.config.video_workflow, cancel_check=cancel_check)


async def dispatch(
    tool_name:  str,
    arguments:  dict,
    session:    Session,
    push:       PushFn,
) -> str:
    """
    Route a tool call from the LLM to the correct implementation.

    Returns a JSON string suitable for the tool result message.
    Any exception is caught and returned as a JSON error string so the
    LLM can reason about failures rather than crashing the loop.
    """
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return await fn(session=session, push=push, **arguments)
    except Exception as e:
        logger.exception("Tool %s failed: %s", tool_name, e)
        return json.dumps({"error": str(e), "tool": tool_name})
