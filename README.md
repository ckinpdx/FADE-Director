# FADE — Music Video Director

A locally-hosted agentic service for directing music videos. Upload a song, converse with an LLM agent to shape the creative direction, and produce a complete music video using ComfyUI.

All inference is local. No cloud services required.

---

## How it works

1. Upload an audio file and paste the song lyrics. Optionally upload a character reference image.
2. FADE separates vocals, aligns lyrics word-by-word to the audio, and analyzes musical structure and mood.
3. Converse with the agent to plan scenes, define visual style, and generate per-scene prompts.
4. Approve the storyboard — FADE submits image generation to ComfyUI scene by scene.
5. Review images, trigger video generation, approve videos.
6. Export the final MP4.

---

## Prerequisites

### Hardware
- NVIDIA GPU with sufficient VRAM (tested on RTX 5090 32GB)
- The pipeline runs one model at a time via llama-swap — Omni-7B-Q4 ≈ 5GB, Qwen3.5-35B-Q4 ≈ 22GB, ComfyUI models load separately

### Software
- Python 3.11+
- Node.js 18+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running at `http://127.0.0.1:8188`
- [llama-swap](https://github.com/mostlygeek/llama-swap) running at `http://127.0.0.1:8000` (requires llama-server compiled from source for Omni `--mmproj` support) — see `llama-swap-config.example.yaml` for a reference config
- [Demucs](https://github.com/facebookresearch/demucs) — installed automatically via `requirements.txt`
- [stable-ts](https://github.com/jianfch/stable-ts) — installed automatically via `requirements.txt`; downloads Whisper large-v3 on first run

### LLM models (via llama-swap)
- `qwen3.5-35b` — main conversational agent (Qwen3.5-35B-A3B Q4_K_M recommended)
- `Qwen2.5-Omni-7B` — multimodal: intonation analysis, reference image description. Requires `--mmproj` flag in llama-swap config.

### ComfyUI models

**Image workflows — choose one per session:**

*ZIT with Reactor* (default):
- `z_image_turbo_bf16.safetensors` — main diffusion model (UNETLoader, lumina2/AuraFlow)
- `qwen_3_4b.safetensors` — CLIP text encoder (lumina2 type)
- `ae.safetensors` — VAE
- `gonzalomoXLFluxPony_v40UnityXLDMD.safetensors` — refinement checkpoint
- `NoeveV3.safetensors` — Reactor face model (user-specific)
- Ultralytics face/body detection models + SAM model (for FaceDetailer)

*Qwen Image Edit*:
- `Qwen-Rapid-AIO-NSFW-v23.safetensors` — checkpoint (CheckpointLoaderKJ)
- `anything2real_2601.safetensors` — LoRA

**Video workflows — choose one per session:**

*LTX with HuMo* (default) and *LTX only*:
- `ltx2-phr00tmerge-sfw-v5.safetensors` — LTX-2 checkpoint
- `LTX2_video_vae_bf16.safetensors` + `LTX2_audio_vae_bf16.safetensors`
- `ltx-2-spatial-upscaler-x2-1.0.safetensors`

*LTX with HuMo* and *HuMo only*:
- `humo_17B_fp16.safetensors` — HuMo model
- `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors` — HuMo LoRA
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` — Wan2.1 CLIP
- `Wan2_1_VAE_bf16.safetensors`
- `whisper_large_v3_encoder_fp16.safetensors`

### Required ComfyUI custom nodes

**ZIT with Reactor:**
- [ComfyUI-ReActor](https://github.com/Gourieff/comfyui-reactor-node)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)

**All video workflows:**
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ClownsharK samplers](https://github.com/ClownsharKent/ComfyUI-ClownsharKSamplers)
- Audio encoder nodes for HuMo whisper conditioning
- [WanExperiments](https://github.com/drozbay/WanExperiments) — **not in ComfyUI Manager, must be cloned manually** into `ComfyUI/custom_nodes/`

---

## Installation

**Prerequisites:** Python 3.11+ and Node.js must be installed. Everything else is handled automatically.

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your paths
3. Run `run_dev.bat`

`run_dev.bat` creates an isolated Python 3.11 virtual environment, installs PyTorch with CUDA 12.8 support (for Demucs GPU acceleration), installs all other Python dependencies, builds the frontend (running `npm install` automatically on first run), and starts the backend. No manual `pip install` or `npm install` needed.

> **CPU fallback:** If you don't have an NVIDIA GPU, Demucs will still run on CPU — just slower. Delete the `.venv` folder if it was created with GPU support and you need to switch, or vice versa.

All ComfyUI workflow files and node maps ship with the repo in `backend/comfyui/workflows/` — no additional workflow setup required.

---

## Configuration

Edit `.env`:

```env
# LLM — all models through llama-swap
AGENT_URL=http://127.0.0.1:8000
AGENT_MODEL=qwen3.5-35b
OMNI_MODEL=Qwen2.5-Omni-7B

# ComfyUI
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_INPUT_DIR=C:/ComfyUI/input
COMFYUI_OUTPUT_DIR=C:/ComfyUI/output

# Service
SERVICE_PORT=8001
SESSION_DIR=./sessions

# Defaults (overridable per session in the UI)
DEFAULT_ORIENTATION=landscape   # portrait | landscape
DEFAULT_FPS=25
SCENE_MIN_SECONDS=3
SCENE_MAX_SECONDS=20
```

**Output dimensions by workflow and orientation:**

| Workflow | Landscape | Portrait |
|---|---|---|
| ZIT / Qwen Image Edit (T2I) | 2048×1152 | 1152×2048 |
| LTX with HuMo / LTX only (I2V) | 2560×1440 | 1080×1920 |
| HuMo only | controlled by long edge (1536 landscape / 1152 portrait) | |

---

## Running

Make sure ComfyUI and llama-swap are running, then double-click `run_dev.bat`.

It will also start llama-swap automatically if it isn't already running (via Windows Scheduled Task).

Open `http://localhost:8001`.

---

## Using your own workflows

FADE patches specific nodes in each workflow at runtime. To use your own ComfyUI workflow, you tell FADE which nodes to patch by giving them special titles in the ComfyUI node editor, then re-exporting and running a script.

### Step 1 — Tag your nodes in ComfyUI

Right-click each node → **Title** and set the exact title string listed below. Only the title needs to change — the node type, connections, and all other settings stay as-is.

**Image workflow (`ltx2_t2i.json`) — 4 required:**

| Title | Node to tag |
|---|---|
| `FADE: Positive Prompt` | Main positive CLIPTextEncode |
| `FADE: Seed` | KSampler or equivalent sampler |
| `FADE: Dimensions` | Empty latent image node (must have `width` + `height` inputs) |
| `FADE: Save` | SaveImage node |

**Video workflow (`ltx2_i2v_humo.json`) — 9 required, 6 optional:**

| Title | Node to tag |
|---|---|
| `FADE: Video Prompt` | Positive text primitive or CLIPTextEncode |
| `FADE: Start Frame` | LoadImage (receives the approved PNG from the image phase) |
| `FADE: Audio File` | VHS_LoadAudioUpload (receives the full session audio file) |
| `FADE: Audio Start` | PrimitiveFloat — scene start time in seconds |
| `FADE: Frame Count` | PrimitiveInt — snapped 8k+1 frame count |
| `FADE: Width` | PrimitiveInt |
| `FADE: Height` | PrimitiveInt |
| `FADE: Seed` | RandomNoise or sampler seed node |
| `FADE: Save` | SaveVideo node |

Optional — only tag if present in your workflow:

| Title | Node to tag |
|---|---|
| `FADE: Negative Prompt` | Negative text primitive |
| `FADE: Seed 2` | Second RandomNoise (2-pass LTX workflows) |
| `FADE: HuMo Seed` | HuMo sampler seed node |
| `FADE: HuMo Long Edge` | JWImageResizeByLongerSide before HuMo |
| `FADE: FPS` | CreateVideo fps field |
| `FADE: FPS Conditioning` | LTXVConditioning frame_rate field |

### Step 2 — Export as API format

In ComfyUI: **Settings → Enable Dev Mode Options**, then use the **Save (API Format)** button (not the regular Save). This produces the JSON structure FADE reads.

### Step 3 — Drop files and rebuild the node map

```
backend/comfyui/workflows/
    ltx2_t2i.json         ← your exported image workflow
    ltx2_i2v_humo.json    ← your exported video workflow
```

Then run:

```
python scripts/build_node_map.py
```

This scans both files, validates all required titles are present, and writes `backend/comfyui/node_map_t2i.json` + `backend/comfyui/node_map_i2v.json`. It prints each matched node with its ID and class type so you can verify the right nodes were found. Re-run any time you update a workflow.

---

## Per-session options

Set at upload time in the UI — these are locked for the lifetime of the session:

| Option | Choices |
|---|---|
| Orientation | Landscape, Portrait |
| Image Workflow | ZIT with Reactor, Qwen Image Edit |
| Video Workflow | LTX with HuMo, LTX, HuMo |

---

## Usage notes

**Use buttons, not chat, to advance phases.** When a button is visible — Generate Prompts, Generate Images, Generate Videos — click it. Asking the agent to "proceed" or "generate" in chat may not trigger the same code path.

**Don't ask the agent to change scene boundaries.** The agent doesn't compute timestamps — it calls an algorithmic segmenter. Asking it to move a cut to a specific time tends to produce hallucinated or incorrect values. Instead: ask for a different scene count, say "reroll" to get a different cut placement at the same count, or adjust boundaries directly in the scene cards.

**HuMo-only: the reference image is a soft guide, not a hard lock.** In the LTX workflows, the approved PNG is conditioning input — the video opens on that exact frame. In HuMo-only, the reference image influences style and subject but doesn't anchor the first frame. Write the video prompt with the same environmental detail you'd put in an image prompt: setting, lighting, color, atmosphere. If you rely on the image to carry that context, the output will drift.

**HuMo-only: expect consistency loss at context window boundaries.** HuMo processes video in overlapping context windows. Across window seams, lighting, color grade, and fine character details can shift visibly. This is a model-level constraint, not a prompt issue. Shorter scenes reduce the number of window transitions and tend to produce more consistent output.

**The style bible must go through the agent.** The style bible is written by the agent in chat and committed via a tool call — it isn't a form you fill in. Let the agent write it, review it in the sidebar, request edits in chat if needed, and confirm. Skipping or manually editing `prompts.json` before the agent commits it will break the prompt generation phase.

---

## Session data

Each session is stored under `sessions/{project-name}-{id}/`:

```
prompts.json        — scene batch descriptor; source of truth for all generation
session.json        — phase, config, analysis outputs; persists across restarts
audio/              — original audio, vocals.wav, instrumental.wav, aligned lyrics
images/             — generated PNGs (current approved version)
images/archive/     — rejected regen history
videos/             — generated MP4s (current approved version)
videos/archive/     — rejected regen history
final.mp4           — export output
```

Sessions survive server restarts — the UI restores your last saved phase automatically.

---

## Suno assistant

FADE includes a Suno prompt engineering assistant at `/suno`. A dedicated chat agent interviews you about the song you want to create and produces a complete Suno prompt package: style tags, structured lyrics with section markers and metatags, and generation notes. No audio analysis or ComfyUI involved.
