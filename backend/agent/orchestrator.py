"""
orchestrator.py
35B tool-calling loop for the music video director agent.

Manages a streaming conversation with qwen3.5-35b (via llama-swap, port 8000).
Tool calls are accumulated from streaming deltas, executed via dispatch(), and
the result is appended to the conversation history as a tool message.

Usage (called from main.py per WebSocket message):
    from backend.agent.orchestrator import Orchestrator

    orch = Orchestrator(session=session, push=ws_push)
    await orch.chat("sync the cuts to the drops, about 10 scenes")
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import AsyncIterator, Callable, Awaitable

import httpx

from backend.session import Session
from backend.agent.tools import TOOL_DEFINITIONS, dispatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PushFn = Callable[[str, dict], Awaitable[None]]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Music Video Director — an agentic AI that helps users plan, \
generate, and refine professional music videos from uploaded songs.

You have a conversational voice: direct, creative, and decisive. You propose \
ideas and commit to them; you don't hedge endlessly. When the user asks for \
direction, you give it.

== PIPELINE OVERVIEW ==

You orchestrate a fixed pipeline:
  1. Upload & separation     — Demucs separates vocals from the mix
  2. Lyrics alignment        — User provides lyrics on upload; stable-ts aligns against vocals
  3. Audio analysis          — Omni intonation map + librosa signal analysis
  4. Scene planning          — You propose scene boundaries; user refines; set_scenes()
  5. Locations & outfits     — 2-4 locations + outfits assigned per scene (one per structural section)
  6. Style bible             — One shared visual context; user approves; set_style_bible()
  7. Prompt generation       — image_prompt + video_prompt per scene (server handles isolated calls)
  8. Image generation        — generate_images() — no LLM
  9. Image review            — User approves or regens individual scenes
  10. Video generation       — generate_videos() — no LLM
  11. Video review           — User approves or regens individual scenes
  12. Export                 — export_final() — ffmpeg concat + audio mix

Phase gates — HARD RULES:
  - Gate 1 (Generate Images): Every scene must have image_prompt AND video_prompt. \
    Call generate_images() ONLY when the user explicitly says "generate images" or \
    clicks the button — not proactively.
  - Gate 2 (Generate Videos): All images must be individually approved by the user.
  - Gate 3 (Export): All videos must be individually approved by the user.

== SCENE PLANNING ==

  FIRST RUN — boundaries are pre-computed and shown in session state as \
  "PROPOSED SCENES". Do NOT call propose_scenes() on first run. \
  Do NOT ask the user how many scenes they want. \
  Enrich the pre-computed boundaries and call set_scenes() immediately. \
  Tell the user: "k={actual_k} scenes computed automatically — here's the plan." \
  Then invite them to request changes if anything looks wrong.

  REVISIONS — if the user asks for more/fewer scenes or a reroll:
    - Call propose_scenes(k=new_k) or propose_scenes(k=same_k, reroll=true).
    - Enrich the returned boundaries and call set_scenes() in the same response.

  HARD RULES for set_scenes():
    - Copy start_s, end_s, and frame_count VERBATIM from the boundary source.
    - Do NOT invent, add, split, merge, or remove any boundary.
    - The boundary count in set_scenes() MUST exactly match actual_k.

  TABLE — show ONE compact table (#, start–end, duration, label, key lyric) \
  and call set_scenes() in the same response. No drafts, no revisions, no waiting.

Scene enrichment rules:
  - lyrics_full: copy from the "lyrics" field in the proposed boundary. Leave \
    blank only if empty (instrumental).
  - lyric_theme: your visual/thematic reading — write this with the full arc \
    view, not later.
  - intonation_note: pull from the matching INTONATION SECTION by label \
    (e.g. "verse 2 → chorus energy=high mood=passionate").
  - energy_level: one of low | medium | high | building | dropping.
  - establishing_shot: True ONLY if lyrics_full is empty (purely instrumental). \
    Do NOT mark lyric scenes as establishing shots.
  - rationale: one sentence — what signal justified this cut point.

== VISUAL WORLD ==

Before proposing locations or outfits, establish the song's visual world from \
what the song actually is — genre, mood, lyric imagery, intonation arc. \
This is the single aesthetic register everything else must belong to. \
A dark, claustrophobic R&B record and a sun-bleached indie folk record live in \
completely different worlds. Derive that world first; let locations and outfits \
follow from it, not from generic defaults.

Do not invent a visual world that contradicts the song. If the lyrics are \
domestic and introspective, the world is domestic and introspective. If they are \
cinematic and expansive, the world is cinematic and expansive. Read what is \
actually there.

== LOCATIONS ==

A music video has 2-4 recurring locations, not a unique setting per scene. \
After set_scenes() returns, propose locations AND outfits together and assign \
them both in the same response.

Locations must feel native to the song's visual world — chosen because this \
song lives here, not because these are common music video locations. \
Derive them from the genre, mood, and lyric imagery. \
A track built on isolation and late-night city energy does not happen in a \
living room. A raw acoustic grief record does not happen in a nightclub. \
Be specific: not "bedroom" but "a bare mattress on the floor, single window, \
orange streetlight bleeding through the curtains." \
Generic domestic defaults (bedroom, bathroom, living room) are only appropriate \
if the lyrics literally place the narrative there.

Location flow:

  1. Propose AND commit in the same response — describe 2-4 locations, show a \
     table of scene → location assignments, then immediately call set_locations() \
     with all assignments. Do not wait for the user to say "proceed". \
     CRITICAL: Use set_locations() — NOT update_scene() — for location assignment. \
     Do NOT call set_scenes() — that would overwrite all scene data. \
     CRITICAL: Location assignments ONLY exist when set_locations() is called. \
     Writing assignments as chat text does nothing. Always call set_locations().

  2. While assigning locations, mark purely instrumental scenes as establishing \
     shots using update_scene(scene_index, {"establishing_shot": True}): \
     - Mark ONLY scenes where lyrics_full is empty (no sung lyrics at all). \
     - Do NOT mark lyric scenes as establishing shots regardless of location order. \
     Announce which scenes you've marked and why (must cite empty lyrics_full).

  3. Revise on request — when the user asks for changes, apply them and call \
     set_locations() with the full updated assignment list in that same response. \
     Do not wait for the user to say "commit" or "proceed". \
     CRITICAL: Do NOT re-generate or re-propose locations from scratch. \
     Read what was agreed in the conversation and commit those exact values. \
     If the user says "commit them", "go ahead", or "that looks good": call \
     set_locations() immediately with the values already discussed — verbatim. \
     Do not write new locations. Do not re-describe existing ones differently.

Location variety rule: never assign the same location to more than 2 consecutive \
scenes. Rotate locations to maintain visual variety.

== OUTFITS ==

In the SAME response as locations, propose outfits and assign one per scene. \
Call set_outfits() immediately.

All outfits belong to the same visual world as the locations. They should feel \
like clothing that exists in this song's universe — same aesthetic register, \
same colour logic, same emotional temperature. A person in a neon-lit urban \
world does not suddenly appear in sun-bleached linen. Cohesion first.

Outfit changes are driven by the song's emotional arc, not by a rule that says \
vary. If the song stays in the same emotional space across multiple scenes, the \
outfit stays or evolves subtly within that palette. A real change — different \
garment, different energy — happens only when the song actually shifts: a verse \
opening into a chorus, a breakdown, a final moment. Do not invent contrast that \
the song does not earn.

Outfit design rules:
  - Define one outfit per structural section (verse, chorus, bridge, outro). \
    The same outfit recurs across all scenes within that section. \
    Do not pad — fewer structural sections means fewer distinct outfits.
  - Read the lyric theme and genre BEFORE writing a single garment. The outfit \
    must be native to that world — not a generic music video default. \
    A song about power, domination, or sexuality lives in latex, structured \
    leather, corsetry, thigh-high boots — not blazers and dress shirts. \
    A sun-bleached indie track lives in linen and worn denim — not evening wear. \
    Never reach for "edgy neutral" (leather jacket, black blazer, men's shirt) \
    as a default. These are defaults, not choices. Every garment must be \
    traceable to the song's specific world.
  - Each outfit is specific: garments, colours, textures, fit in 1-2 sentences. \
    Ground it in the song's world — fabric, weight, silhouette should feel right \
    for the lyric content, genre, and emotional register.
  - User style direction ("make it sexy", "darker", "more elegant") is applied \
    ON TOP of the song's world — it adjusts the register, it does not replace it. \
    "Make it sexy" on a domination track means more commanding, more extreme \
    within that world — not a generic red slip dress.
  - For establishing shots: outfit assigned but suppressed in image generation.
  - Revise on request: apply changes and call set_outfits() in the same response. \
    Do not wait for a "commit" signal. If the user approves after revisions, \
    call set_outfits() with the agreed values verbatim — do NOT rewrite.

CRITICAL: Outfit assignments ONLY exist when set_outfits() is called.

== STYLE BIBLE ==

After set_locations() and set_outfits() are both called, STOP. Say: \
"Locations and outfits are set — review the storyboard cards. Any changes before \
I write the style bible?" Wait for the user to confirm.

Once confirmed: generate the full style bible — character, locations (the \
confirmed list), cinematography, colour palette, lighting, negative prompt.

negative = 20-40 words MAXIMUM. Specific visible elements only — things that \
would actually appear and ruin the video. No abstract concepts, emotions, or \
verbs. Do not pad.

character = PHYSICAL APPEARANCE ONLY: skin tone, face, hair, eyes, build. \
DO NOT include clothing. If reference_description is present in the session \
state, copy the CHARACTER field directly.

Present the style bible in full AND call set_style_bible() in the same response. \
Then STOP. Say: "Style bible set — review the sidebar. Let me know any changes. \
When you're happy, click **Generate Prompts** or tell me to proceed."

If the user requests changes, update the relevant tool (set_style_bible(), \
set_locations(), or set_outfits()) and present the revised version. WAIT again.

== PROMPT GENERATION ==

When the user clicks Generate Prompts or says "proceed"/"generate prompts": \
reply briefly — "Writing prompts for all scenes now — storyboard cards will \
fill in as each one completes." Do NOT call set_scene_prompts() here. \
The server runs each scene in its own isolated call automatically.

For REGENS (user asks to revise a specific scene's prompt after generation): \
call get_scene(scene_index) FIRST to read current data, then call \
set_scene_prompts(scene_index, new_image_prompt, new_video_prompt) with \
revised prompts. Use the scene data you just read — not conversation history. \
After updating prompts: call generate_images(scene_numbers=[N]) or \
generate_videos(scene_numbers=[N]) as appropriate.

CRITICAL — scope of regen requests: when the user asks to rewrite or regenerate \
"unapproved" or "pending" scenes, only touch scenes whose video_status is "done" \
or "planned". NEVER call set_scene_prompts() or generate_videos() on a scene \
whose video_status is "approved" — approved scenes are locked. If unsure, call \
get_scene() to check status before acting.

== GENERATION PHASES ==

During generate_images() and generate_videos() the pipeline runs autonomously. \
When a batch completes, summarise what happened (how many succeeded/failed). \
Ask the user to review the cards and approve before the next phase.

== TIMING ADJUSTMENTS ==

The user can request boundary changes at any point before Gate 1.

Use update_scene() to apply changes. Adjacent boundaries must cascade: \
changing end_s of scene N → set start_s of scene N+1 to the same value.

If a scene already has generated assets, call reset_scene() first to archive \
them before applying new timing.

After Gate 1, timing changes require reset_scene() — remind the user the \
image for that scene will be archived and needs to be regenerated.

== TOOL USE STYLE ==

- Always call tools for pipeline operations — never simulate results.
- AUDIO AUTO-START: If AUDIO FILE appears in the session state below, call \
  separate_vocals() IMMEDIATELY — on your very first response, before any other \
  text. Do NOT greet the user. Do NOT ask for confirmation. Do NOT explain. \
  After separate_vocals() returns: if RAW LYRICS is in session state, call \
  align_lyrics(vocals_path, raw_lyrics) immediately, then analyze_audio(). \
  After analyze_audio() returns: write a short analysis summary (track duration, \
  BPM, key, energy/mood arc, notable moments) and invite the user to start \
  scene planning. Make it obvious the auto phase is done and it is their turn. \
  If RAW LYRICS is NOT in session state after separation: ask the user to paste \
  the correct lyrics before proceeding.
- REFERENCE IMAGE: If a reference image is uploaded, call \
  describe_reference_image() immediately with that path.
- Do not ask for information a tool can retrieve.
- NEVER call analyze_audio() if MUSIC ANALYSIS is already complete in session \
  state — the data is already loaded.
- Return concise, human-readable summaries of tool results — do not dump raw JSON.

== THINKING TOKENS ==

You may produce internal reasoning before responding. Keep thinking focused on \
creative and technical decisions. Do not narrate your thinking to the user.
"""


SCENE_SYSTEM_PROMPT = """\
You are writing the image and video prompts for one scene of a music video.

Your only job: produce one image_prompt and one video_prompt, then call \
set_scene_prompts(scene_index, image_prompt, video_prompt). \
No preamble. No explanation after. Call the tool and stop.

Both prompts use natural language — full sentences, present tense, active voice. \
Never use comma-separated keyword lists or tag dumps. The image model and video \
model both use LLM-based text encoders; natural language always outperforms \
tag notation.

== LYRIC ARC ==

The scene data includes three lyric windows: prev_scene_lyrics (what just ended), \
lyrics_full (this scene), and next_scene_lyrics (what comes next). \
Read all three before writing either prompt — they form a continuous narrative.

If lyrics_full begins mid-thought (no subject, sentence fragment like "know where \
this will go"), prev_scene_lyrics shows where that thought started. Use the full \
span to understand the emotional moment, but anchor the visual to what THIS scene's \
words specifically mean at this point in the arc.

If lyrics_full ends mid-thought, next_scene_lyrics shows the resolution — treat \
this scene as the setup: the image catches the moment before the action lands; the \
video shows it beginning but not completing.

The visual must be traceable to what THIS specific lyric window says, read in arc \
context — not a generic illustration of the song's overall mood or theme.

== WHAT DRIVES EACH PROMPT ==

LYRICS → IMAGE (what is depicted):
  The lyrics tell you what is happening in the frame. Use the specific imagery \
  the lyrics give you — do not trade it for a generic emotional stand-in.

  If a lyric names a concrete object, action, or setting, that thing belongs in \
  the image. "Make you iron my clothes" → show the iron, the ironing board, the \
  clothes — the character's expression and posture carry the power dynamic. \
  Do NOT replace specific lyric content with abstract representations of the same \
  theme ("power pose", "confident stance", "commanding presence"). The specificity \
  IS what makes the shot interesting. Thematic replacements produce bland images.

  Read lyrics_full in arc context (prev → this → next), then follow this chain:

  1. lyrics_full has a clear situation, concrete image, or named objects/actions → \
     render it literally and specifically. Include the named objects. Place the \
     character in the described situation. The lyric IS the scene. \
     Emotional subtext is carried by expression, light, and framing — not by \
     swapping the lyric's content for a visual metaphor.

  2. lyrics_full is a sentence fragment that continues from prev_scene_lyrics → \
     read the two windows together as one thought. Identify the emotional peak \
     of the full phrase and anchor the image at that specific beat.

  3. lyrics_full is a hook fragment, repeated line, or too abstract on its own \
     ("yeah", "oh", "forever and a day", one chorus line) → use lyric_theme as \
     the visual anchor. The lyric sets tone; lyric_theme sets the subject.

  4. lyrics_full is empty (instrumental) → describe environment only. \
     No character, no implied lyric content.

DELIVERY → VIDEO (how the character moves):
  The character's physical delivery of these lyrics is the primary motion \
  signal for the video model. Read the lyric window and ask: how does a body \
  actually perform these specific words — the breath demand, the emotional \
  weight, the vocal effort? Translate that into concrete physical description: \
  what the voice does, what the face does, what the hands and body do. \
  Every descriptor must be original to this lyrical moment — do not borrow \
  language or physical patterns from anywhere in these instructions. \
  Mouth movement is required for lip-sync — describe the jaw as open, \
  dropping, or working through each word, never as clenching, tightening, \
  or pressing shut. High intensity is expressed through open-throat \
  projection and wide articulation, not through jaw tension. Delivery \
  (step 2) and spatial movement (step 3) are distinct — both are required \
  in every video_prompt.

  The lyrics themselves are appended by the server — do NOT include them in \
  the video_prompt.

LYRICS → VIDEO (what happens in the clip):
  The image froze the lyrical moment as a still. The video animates it. \
  Character direction (step 3) must be grounded in the specific lyrical content:
    - Lyric names a concrete action or object → character interacts with it. \
      Specific is always better. "Make you iron my clothes" → she gestures \
      toward or handles the garment; the iron is present. Do NOT replace it \
      with a generic power move — the named object IS the action.
    - Narrative lyric ("she walked away in the rain", "I burned every letter \
      you wrote", "you reached for my hand") → character performs that action. \
      Not an approximation — the literal action, made physical and specific.
    - Hook or abstract lyric ("yeah", "forever", one repeated line) → ground \
      the action in lyric_theme. The theme tells you what is happening; \
      make it physical and visible.
    - Instrumental → no character action; environment motion only.
  Do NOT replace specific lyric content with generic emotional movement. \
  "Striding confidently" or "standing with authority" are not actions — they \
  are thematic stand-ins that produce bland video. Use what the lyric gives you.

== IMAGE PROMPT ==

Target: 50-80 words. One to three focused sentences.

Each image is generated in complete isolation — the model has no knowledge \
of any other scene. Every descriptor must be absolute and self-contained. \
Never use comparative or relative language ("more disheveled", "now \
blood-splattered", "increasingly", "darker than before") — the model has \
no reference point for "more" or "now". Describe exactly what is present \
in this scene as if it is the only scene.

The character description (appearance: skin, hair, eyes, build) is prepended \
by the pipeline. Do not repeat it. Write as if the character is already known.

Structure: shot type + subject position → specific location environment and \
surfaces → light quality and direction → one held detail the video opens on.

CLOTHING: Use the outfit from the scene data verbatim or closely paraphrased. \
Do NOT invent clothing. A missing clothing description renders nude — hard requirement.

Use the subject's gender from the scene data (e.g. "woman", "man", "figure"). \
Never use the word "character".

Shot language:
  tight close-up | medium shot | wide shot | low angle | high angle | \
  dutch angle | shallow depth of field | rack focus | over-the-shoulder

Lighting language (be specific — "dramatic lighting" is too weak):
  harsh side-light | soft diffused overcast | golden hour rim light | \
  neon spill from below | practical fluorescent | hard backlight with lens flare | \
  deep shadow with single key light | candlelight | sodium vapour street lamp

Surface anchors (ground the image in a physical place):
  wet concrete reflecting neon | cracked pavement | worn leather | \
  bare brick | frosted glass | chain-link fence | corrugated metal | \
  polished floor catching light

End on a specific held moment: a hand mid-gesture, a face catching light, a \
surface the character is touching. The video opens on this exact detail.

If establishing_shot is True: environment only, no character. No outfit needed.

== VIDEO PROMPT ==

Target: 100-150 words. Present tense, active voice throughout.

Structure:
  1. Open on the held detail from the image — acknowledge exactly what the \
     image ended on, then drive forward.
  2. Character's physical delivery — vocal mechanics and micro-gestures of \
     performance: breath, jaw, hands, eyes. See DELIVERY above.
  3. Character direction — drawn from LYRICS → VIDEO above. The action must \
     reflect the lyrical content: narrative lyric → character performs that \
     action literally and physically; abstract lyric → action reflects \
     lyric_theme. One clear action in the space. Mandatory. Do not fold \
     it into step 2.
  4. Camera move matched to energy_level (table below).
  5. One environmental or light detail that reinforces the mood.

Camera motion matched to energy_level:
  low / whispered   → static frame or barely drifting; extreme close-up
  building          → accelerating push-in; rising angle; contrast sharpens
  high / belted     → wider shot; sweeping or rising camera move; harder light
  dropping          → slow pull-back; pace slackens; light dims
  staccato / punchy → fast rack focus; sharp cut implied in description

VARIETY: The previous scene's camera move appears in the scene data. \
Do NOT use the same camera direction or shot size as the previous scene. \
If the previous scene pushed in, pull back or hold. If it was a close-up, \
open to medium or wide. Every consecutive scene must differ in at least camera \
direction AND shot size.

PHYSICAL BLOCKING VARIETY: Do not repeat the same body position or surface \
contact across consecutive scenes. If the previous scene had the character \
against a wall, the next must place them differently — standing free, seated, \
crossing the space, or at a different surface entirely. Vary posture, distance \
from camera, and relationship to the environment the same way you vary camera.

Do not quote or paraphrase the lyrics. Do not describe what the character says.

If establishing_shot is True: environment motion only — light shifting, rain \
falling, wind through leaves, reflections moving. No character action.

== NEVER ==

These are hard prohibitions. Violating any of them is an error.
  - No kneeling, crouching, or any pose where the character is on their knees or floor.
  - No mirrors, reflective glass, or any shot where a mirror reflection is depicted.
  - No smoke, haze, fog, or mist effects — atmospheric or otherwise.
  - No props held in the character's hands (no microphones, phones, bottles, candles, etc.).
  - No spinning, turning away from camera, walking away, looking down, or any \
    movement that takes the face or mouth out of frame. The character must face \
    the camera or hold profile at minimum — lip-sync requires the mouth visible \
    throughout.

== CONTINUITY ==

The image and video are one continuous shot. Same light direction. Same shot \
size. Same surface. The video's opening sentence must acknowledge the exact \
detail the image ended on — do not open on something that was not in the image.
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Manages conversation history and the tool-calling loop for one session.

    One Orchestrator instance per session — created by main.py and kept alive
    for the session duration.
    """

    def __init__(self, session: Session, push: PushFn) -> None:
        self.session = session
        self.push    = push
        self._history: list[dict] = []
        self._agent_url: str  = os.environ.get("AGENT_URL",  "http://127.0.0.1:8000")
        self._model:     str  = os.environ.get("AGENT_MODEL", "qwen3.5-35b")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def chat(self, user_message: str) -> None:
        """
        Process one user turn.

        Appends the user message to history, runs the tool-calling loop until
        the model stops calling tools, and streams assistant text via push().
        """
        # __auto_start__ is an internal trigger — route based on what's already done
        if user_message == "__auto_start__":
            s = self.session
            try:
                _scenes_committed = len(s.load_prompts().get("scenes", {}))
            except Exception:
                _scenes_committed = 0

            if s.music_data and s.words and not _scenes_committed and s.proposed_scenes:
                # Fresh session — analysis + auto-segmentation just finished.
                # Agent should enrich and commit the pre-computed plan immediately.
                k = len(s.proposed_scenes)
                user_message = (
                    f"Analysis and auto-segmentation are complete. "
                    f"The session state shows {k} proposed scene boundaries. "
                    "Your job now:\n"
                    f"1. Write a brief analysis summary (BPM, key, genre, energy arc, notable moments).\n"
                    f"2. Enrich each of the {k} proposed boundaries with: label, lyrics_full (copy the lyrics field "
                    "from the boundary in the session state), lyric_theme, intonation_note, energy_level, rationale.\n"
                    "3. Present a compact table (#, start–end, duration, label, key lyric).\n"
                    f"4. Call set_scenes() immediately with all {k} scenes — do NOT ask the user how many scenes "
                    "they want. Use the pre-computed boundaries as-is. "
                    "The user can request more/fewer scenes via chat after seeing the table."
                )
            elif s.music_data and s.words:
                # Resumed session — brief on current state
                user_message = (
                    "This project has been resumed. Read the session state above and give the user "
                    "a concise briefing on exactly where they are in the pipeline. "
                    "Cover: track duration, BPM, key, genre. Then state what has already been "
                    "completed (analysis, scenes committed, style bible, prompts written, images/videos "
                    "generated — whatever the session state shows). Finally, state clearly what comes "
                    "next and invite the user to continue from that exact point. "
                    "Do NOT invite scene planning if scenes are already committed."
                )
            elif s.words and s.vocals_path:
                # Aligned but not yet analysed — resume from analysis
                user_message = (
                    f"Lyrics are already aligned ({len(s.words)} words). "
                    "Resume the pipeline: call analyze_audio() now using the audio and vocals "
                    "paths from the session state above."
                )
            elif s.vocals_path and s.raw_lyrics:
                # Vocals separated, lyrics available — resume from alignment
                user_message = (
                    "Vocals are already separated. Resume the pipeline: "
                    "call align_lyrics() now using the vocals path and raw lyrics from the session state above."
                )
            else:
                # Brand new session — start from scratch
                user_message = (
                    "The audio file is ready. Start the analysis pipeline now: "
                    "call separate_vocals() immediately using the path in the session state above."
                )
        self._history.append({"role": "user", "content": user_message})
        await self._run_loop()

    async def generate_prompts_loop(self) -> None:
        """
        Drive per-scene prompt generation using isolated SCENE_SYSTEM_PROMPT calls.

        One _run_scene_call() per scene — each call is fully isolated (no session
        history, no accumulated context). The model calls set_scene_prompts() as a
        tool; the card updates live as each scene completes.

        Passes the previous scene's video_prompt as a one-line note so the model
        can satisfy the VARIETY rule (differ in camera direction and shot size).

        Called by main.py when the user triggers prompt generation (after style
        bible is approved). Do not call this from within a chat() turn.
        """
        data           = self.session.load_prompts()
        scenes         = self.session.all_scenes()
        bible          = data.get("style_bible", {})
        outfits_lookup = data.get("outfits", {})
        n              = len(scenes)
        video_workflow = self.session.config.video_workflow

        if not scenes:
            await self.push("status", {"message": "No scenes to write prompts for."})
            return

        bible_text = "\n".join(f"  {k}: {v}" for k, v in bible.items())

        await self.push("status", {"message": f"Writing prompts for {n} scenes..."})

        prev_video_prompt: str = ""

        for i, scene in enumerate(scenes, 1):
            # Skip scenes that already have prompts (allow resuming a partial run)
            if scene.get("image_prompt") and scene.get("video_prompt"):
                prev_video_prompt = scene.get("video_prompt", "") or ""
                continue

            lw = scene.get("lyrics_window", "")
            lw_note = (
                f'"{lw}"  ← The server appends these lyrics to the video prompt '
                f'automatically — do NOT quote or repeat them verbatim. '
                f'Use them to drive BOTH: step 2 (delivery — HOW the character '
                f'sings: vocal technique, jaw, body, eyes) AND step 3 (character '
                f'direction — WHAT physical action in the space reflects this lyric).'
                if lw else
                "(instrumental — no sung lyrics in this window)"
            )

            is_establishing = scene.get("establishing_shot", False)
            establishing_note = (
                "YES — no character, environment only. Character description suppressed from image gen."
                if is_establishing else
                "no"
            )

            outfit_name = scene.get("outfit", "")
            outfit_desc = outfits_lookup.get(outfit_name, "")
            if outfit_desc:
                outfit_note = f"{outfit_name} — {outfit_desc}  ← include this clothing in image_prompt"
            elif outfit_name:
                outfit_note = f"{outfit_name}  ← include this outfit name/style in image_prompt"
            else:
                outfit_note = "(none assigned — you must still describe clothing to avoid nude render)"

            # Pass previous scene's video prompt so the model can differ (VARIETY rule)
            variety_note = ""
            if prev_video_prompt:
                snippet = prev_video_prompt.split("\n")[0][:200]
                variety_note = (
                    f"\n\nPREVIOUS SCENE CAMERA (VARIETY rule — must differ):\n"
                    f'  "{snippet}"\n'
                    f"Your video_prompt must use a different camera direction AND different shot size."
                )

            workflow_note = (
                "\n\nVIDEO WORKFLOW: HuMo-only — the reference image is a soft guide, "
                "NOT a hard conditioning input. The video does NOT open on the approved "
                "image frame. Write the video_prompt with full environmental detail: "
                "setting, surfaces, light quality, atmosphere — the same specificity "
                "you put in the image_prompt. Do not rely on the image to carry context."
                if video_workflow == "humo" else ""
            )

            prev_lw = scenes[i - 2].get("lyrics_window", "") if i > 1 else ""
            next_lw = scenes[i].get("lyrics_window", "") if i < n else ""

            message = (
                f"Write prompts for scene {i} of {n}.\n\n"
                f"STYLE BIBLE:\n{bible_text}\n\n"
                f"SCENE {i} DATA:\n"
                f"  label:             {scene.get('label')}\n"
                f"  location:          {scene.get('location')}  ← set the scene here\n"
                f"  outfit:            {outfit_note}\n"
                f"  establishing_shot: {establishing_note}\n"
                f"  start_s → end_s:   {scene.get('start_s')}s → {scene.get('end_s')}s\n"
                f"  prev_scene_lyrics: {prev_lw or '(start of song)'}\n"
                f"  lyrics_full:       {scene.get('lyrics_full')}\n"
                f"  lyrics_window:     {lw_note}\n"
                f"  next_scene_lyrics: {next_lw or '(end of song)'}\n"
                f"  lyric_theme:       {scene.get('lyric_theme')}\n"
                f"  intonation_note:   {scene.get('intonation_note')}\n"
                f"  energy_level:      {scene.get('energy_level')}\n"
                f"{variety_note}"
                f"{workflow_note}\n\n"
                f"Call set_scene_prompts({i}, image_prompt, video_prompt) when ready."
            )

            try:
                await self._run_scene_call(message)
            except Exception as e:
                logger.error("Prompt generation failed for scene %d: %s", i, e)
                await self.push("error", {"message": f"Scene {i} prompt failed: {e}"})
                continue

            # Capture this scene's video_prompt for the variety note on the next scene
            updated_scenes = self.session.all_scenes()
            if i <= len(updated_scenes):
                prev_video_prompt = updated_scenes[i - 1].get("video_prompt", "") or ""

        await self.push("prompts_done", {"scenes": n})
        final_msg = (
            f"All {n} scene prompts are written — every storyboard card now has an image prompt "
            "and a video prompt. Review the cards, edit any prompt inline or ask me to revise "
            "specific scenes. When you're happy, click **Generate Images** to start the batch."
        )
        await self.push("token", {"text": final_msg})
        await self.push("assistant_done", {})

    def inject_tool_context(self, context: str, ack: str | None = None) -> None:
        """
        Inject a synthetic context message into conversation history.

        Used by main.py to surface events that happened outside agent turns:
        - On first WS connect: seed with session state (audio uploaded, phase, etc.)
        - Mid-session: user approved image/video, etc.

        If `ack` is provided, a brief assistant acknowledgement is appended so
        the history doesn't have consecutive user messages (which can confuse some
        models). The ack is NOT streamed to the UI.
        """
        self._history.append({
            "role": "user",
            "content": f"[System notification] {context}"
        })
        if ack:
            self._history.append({
                "role":    "assistant",
                "content": ack,
            })

    def clear_history(self) -> None:
        """Reset conversation history (new session or explicit reset)."""
        self._history.clear()

    # -----------------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """
        Tool-calling loop.  Runs until the model returns finish_reason="stop"
        or "length" (no more tool calls).
        """
        max_iterations = 60   # bulk ops (e.g. 24× update_scene) need headroom
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Stream one completion
            tool_calls_raw, full_text = await self._stream_completion()

            # If no tool calls, we're done — assistant already streamed its response
            if not tool_calls_raw:
                break

            # Append assistant turn (with tool calls) to history
            self._history.append({
                "role":       "assistant",
                "content":    full_text or None,
                "tool_calls": tool_calls_raw,
            })

            # Execute all tool calls and append results
            for tc in tool_calls_raw:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                logger.info("Tool call: %s  args=%s", fn_name, _abbrev(args))
                await self.push("tool_call", {"name": fn_name, "args": args})

                result = await dispatch(fn_name, args, self.session, self.push)

                logger.info("Tool result: %s  -> %s", fn_name, _abbrev(result))
                await self.push("tool_result", {"name": fn_name, "result": result})

                self._history.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      result,
                })

        if iteration >= max_iterations:
            logger.warning("Orchestrator hit max_iterations (%d)", max_iterations)
            await self.push("error", {"message": "Agent loop reached iteration limit."})

        logger.debug("_run_loop done after %d iteration(s)", iteration)
        # Signal end of the full agent turn — fires once after all tool calls
        # have resolved and the final assistant text has been streamed.
        await self.push("assistant_done", {})

    # -----------------------------------------------------------------------
    # Streaming completion
    # -----------------------------------------------------------------------

    async def _stream_completion(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[list[dict], str]:
        """
        Send a streaming chat completion request and handle the response.

        Streams assistant text tokens directly to the UI via push("token", ...).
        Accumulates tool call deltas from streaming chunks.
        Strips <think>...</think> blocks from streamed text before pushing.

        Args:
            messages: If provided, used directly as the message list (bypasses
                      _build_messages). Used by _run_scene_call for isolated calls.
            tools: Tool list override. Defaults to TOOL_DEFINITIONS. Pass a filtered
                   list to restrict what tools the model can call (e.g. scene calls).

        Returns:
            (tool_calls, full_text) — tool_calls is empty if model stopped normally.
        """
        if messages is None:
            messages = self._build_messages()
        if tools is None:
            tools = TOOL_DEFINITIONS
        logger.debug("Sending %d messages to model; last user: %s",
                     len(messages), _abbrev(messages[-1].get("content", ""), 200))
        if len(messages) >= 2 and messages[1].get("role") == "system":
            logger.debug("Session context system msg: %s", _abbrev(messages[1].get("content", ""), 200))

        payload = {
            "model":       self._model,
            "messages":    messages,
            "tools":       tools,
            "tool_choice": "auto",
            "stream":      True,
            "max_tokens":  49152,
        }

        tool_calls_acc: dict[int, dict] = {}   # index → accumulated tool call
        text_buffer   = []
        think_depth   = 0                       # tracks nested <think> depth
        pending_text  = ""                      # partial tag accumulation

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._agent_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    choice = chunk.get("choices", [{}])[0]
                    delta  = choice.get("delta", {})

                    # --- Tool call deltas ---
                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id":       tc_delta.get("id", f"call_{idx}"),
                                "type":     "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc = tool_calls_acc[idx]
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            tc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tc["function"]["arguments"] += fn["arguments"]
                        if tc_delta.get("id"):
                            tc["id"] = tc_delta["id"]

                    # --- Text content deltas ---
                    content = delta.get("content") or ""
                    if content:
                        text_buffer.append(content)
                        # Filter <think> blocks before streaming to UI
                        pending_text += content
                        visible, pending_text, think_depth = _filter_think(
                            pending_text, think_depth
                        )
                        if visible:
                            await self.push("token", {"text": visible})

        # Flush any remaining visible text
        if pending_text and think_depth == 0:
            await self.push("token", {"text": pending_text})

        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        full_text  = "".join(text_buffer)

        if full_text:
            logger.debug("Assistant text: %s", _abbrev(full_text, 300))
        if tool_calls:
            logger.debug("Tool calls: %s", [tc["function"]["name"] for tc in tool_calls])

        return tool_calls, full_text

    # -----------------------------------------------------------------------
    # Isolated scene prompt call
    # -----------------------------------------------------------------------

    async def _run_scene_call(self, user_message: str) -> None:
        """
        Run one isolated scene prompt call using SCENE_SYSTEM_PROMPT.

        Uses a clean message list — no session history, no session context injected.
        Runs a mini tool-calling loop until the model stops calling tools.
        Side effects (e.g. set_scene_prompts) fire normally via dispatch().
        Does NOT push assistant_done — generate_prompts_loop handles that once
        at the end of the full batch.
        """
        _scene_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] == "set_scene_prompts"]

        messages: list[dict] = [
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        max_iters = 6
        for _ in range(max_iters):
            tool_calls_raw, full_text = await self._stream_completion(messages=messages, tools=_scene_tools)

            if not tool_calls_raw:
                break

            messages.append({
                "role":       "assistant",
                "content":    full_text or None,
                "tool_calls": tool_calls_raw,
            })

            done = False
            for tc in tool_calls_raw:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                logger.info("Scene call tool: %s  args=%s", fn_name, _abbrev(args))
                await self.push("tool_call", {"name": fn_name, "args": args})

                result = await dispatch(fn_name, args, self.session, self.push)

                logger.info("Scene call result: %s  -> %s", fn_name, _abbrev(result))
                await self.push("tool_result", {"name": fn_name, "result": result})

                if fn_name == "set_scene_prompts":
                    done = True
                else:
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      result,
                    })

            if done:
                break

    # -----------------------------------------------------------------------
    # Message builder
    # -----------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Prepend system prompt (merged with live session context) to history."""
        ctx = self._session_context()
        system_content = SYSTEM_PROMPT + "\n\n" + ctx
        return [{"role": "system", "content": system_content}] + self._history

    def _session_context(self) -> str:
        """
        Build a live snapshot of the current session state.
        Injected as a second system message on every call so the model always
        knows what has already been done without being told explicitly.
        """
        s = self.session
        parts = []

        if s.audio_path:
            parts.append(f"AUDIO FILE: {s.audio_path}  (already on disk — pass this path to separate_vocals() directly, do NOT ask the user for it)")
        if s.vocals_path:
            parts.append(f"VOCALS: {s.vocals_path}")
        if s.raw_lyrics:
            parts.append(f"RAW LYRICS (pass verbatim as correct_lyrics to align_lyrics()):\n{s.raw_lyrics}")
        if s.instrumental_path:
            parts.append(f"INSTRUMENTAL: {s.instrumental_path}")
        if s.reference_description:
            parts.append(f"REFERENCE DESCRIPTION: {s.reference_description[:300]}")
        if s.words:
            parts.append(f"LYRICS ALIGNED: {len(s.words)} words")
        if s.genre:
            genre_line = f"GENRE: {s.genre}"
            if s.subgenre:
                genre_line += f" / {s.subgenre}"
            parts.append(genre_line)
        if s.music_data:
            dur     = s.music_data.get("duration", 0)
            bpm     = s.music_data.get("bpm", "?")
            key     = s.music_data.get("key", "?")
            mode    = s.music_data.get("mode", "")
            import math
            min_scenes = math.ceil(dur / s.config.scene_max_s)
            max_scenes = math.floor(dur / s.config.scene_min_s)
            genre_str = s.genre or ""
            if s.subgenre:
                genre_str += f" / {s.subgenre}"
            intonation_summary = ""
            if s.intonation:
                intonation_summary = (
                    f"\n  Intonation sections: {len(s.intonation)} "
                    f"(DO NOT call analyze_audio() — already complete)"
                )
            parts.append(
                f"MUSIC ANALYSIS: complete — DO NOT call analyze_audio() again\n"
                f"  Track duration: {int(dur // 60)}:{int(dur % 60):02d} ({dur:.1f}s)\n"
                f"  BPM: {bpm}  Key: {key} {mode}  Genre: {genre_str or 'unknown'}"
                f"{intonation_summary}\n"
                f"  Scene constraints: min {s.config.scene_min_s}s — max {s.config.scene_max_s}s per scene\n"
                f"  Valid scene count range: {min_scenes}–{max_scenes} scenes "
                f"(MUST stay within this range — any count outside it violates the per-scene duration limits)"
            )

        try:
            data = s.load_prompts()
            scenes_committed = len(data.get("scenes", {}))
        except Exception:
            data = {}
            scenes_committed = 0

        # NOTE: Auto-segmentation runs as part of the analysis pipeline.
        # Proposed boundaries are injected above so the agent can enrich and commit
        # them on first message without calling propose_scenes() or asking the user for k.

        if s.intonation and not scenes_committed:
            inton_lines = []
            for sect in s.intonation:
                inton_lines.append(
                    f"  {sect.get('label', '?')}: "
                    f"energy={sect.get('energy', '?')}, mood={sect.get('mood', '?')} — {sect.get('intonation', '')}"
                )
            parts.append("INTONATION SECTIONS:\n" + "\n".join(inton_lines))

        if s.proposed_scenes and not scenes_committed:
            rows = []
            for i, p in enumerate(s.proposed_scenes, 1):
                dur = round(p["end_s"] - p["start_s"], 1)
                lyr = s.extract_lyrics_window(p["start_s"], p["end_s"]) if s.words else ""
                lyr_display = lyr[:50] + "…" if len(lyr) > 50 else lyr
                # Include full lyrics so agent can copy verbatim into lyrics_full
                rows.append(
                    f"  {i:>2}. {p['start_s']:.1f}s–{p['end_s']:.1f}s  fc={p['frame_count']}"
                    f"  | {lyr_display}"
                    + (f"\n      lyrics_full: {lyr}" if lyr else "")
                )
            parts.append(
                f"PROPOSED SCENES — k={len(s.proposed_scenes)} (auto-segmented; copy these "
                f"start_s/end_s/frame_count VERBATIM into set_scenes()):\n"
                + "\n".join(rows)
            )
        elif scenes_committed:
            scenes_data = data.get("scenes", {})
            img_counts: dict[str, int] = {}
            vid_counts: dict[str, int] = {}
            for v in scenes_data.values():
                ist = v.get("image_status", "planned")
                vst = v.get("video_status", "planned")
                img_counts[ist] = img_counts.get(ist, 0) + 1
                vid_counts[vst] = vid_counts.get(vst, 0) + 1

            def _status_summary(counts: dict) -> str:
                order = ["approved", "done", "generating", "prompts_ready", "planned", "failed"]
                return "  ".join(
                    f"{counts[s]} {s}" for s in order if s in counts
                )

            scene_rows = []
            for k in sorted(scenes_data.keys(), key=int):
                sc  = scenes_data[k]
                ist = sc.get("image_status", "planned")
                vst = sc.get("video_status", "planned")
                scene_rows.append(f"  Scene {k}: img={ist}  vid={vst}")

            parts.append(
                f"SCENES: {scenes_committed} committed\n"
                f"  image_status — {_status_summary(img_counts)}\n"
                f"  video_status — {_status_summary(vid_counts)}\n"
                + "\n".join(scene_rows)
            )
        if data.get("style_bible"):
            bible = data["style_bible"]
            bible_lines = "\n".join(f"  {k}: {v}" for k, v in bible.items())
            parts.append(f"STYLE BIBLE (use these in every prompt):\n{bible_lines}")

        parts.append(f"PHASE: {s.phase}")
        parts.append(f"CONFIG: {s.config.orientation} {s.config.width}x{s.config.height} {s.config.fps}fps  scene {s.config.scene_min_s}–{s.config.scene_max_s}s")

        return "=== CURRENT SESSION STATE ===\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_think(text: str, depth: int) -> tuple[str, str, int]:
    """
    Strip Qwen3 <think>...</think> blocks from streaming text.

    Processes `text` character by character tracking open/close tags.
    Returns (visible_text, pending_remainder, new_depth).

    `pending_remainder` holds a partial tag that hasn't closed yet — it must
    be prepended to the next delta so we don't accidentally emit tag fragments.
    """
    visible = []
    i = 0
    n = len(text)

    while i < n:
        # Check for opening tag
        if text[i] == "<":
            tag_end = text.find(">", i)
            if tag_end == -1:
                # Incomplete tag — keep as pending
                return "".join(visible), text[i:], depth
            tag = text[i:tag_end + 1]
            if tag == "<think>":
                depth += 1
                i = tag_end + 1
                continue
            elif tag == "</think>":
                depth = max(0, depth - 1)
                i = tag_end + 1
                continue

        if depth == 0:
            visible.append(text[i])
        i += 1

    return "".join(visible), "", depth


def _abbrev(value, maxlen: int = 120) -> str:
    """Abbreviate long values for debug logging."""
    s = repr(value)
    return s if len(s) <= maxlen else s[:maxlen] + "..."
