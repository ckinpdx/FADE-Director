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
- [uv](https://github.com/astral-sh/uv) — Python environment manager (`winget install astral-sh.uv`)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running at `http://127.0.0.1:8188`
- An OpenAI-compatible LLM server on port 8000 — see **LLM server setup** below
- [Demucs](https://github.com/facebookresearch/demucs) — installed automatically via `requirements.txt`
- [stable-ts](https://github.com/jianfch/stable-ts) — installed automatically via `requirements.txt`; downloads Whisper large-v3 on first run

### LLM server setup

FADE requires an OpenAI-compatible LLM server on port 8000. The recommended stack is **llama-swap + llama-server**:

1. Run `scripts\build_llama.bat` — builds `llama-server.exe` and `llama-swap.exe` from source into `tools\`. Requires Git, CMake, CUDA Toolkit, and Go. Skips anything already on PATH.
2. Copy `llama-swap-config.example.yaml` to `llama-swap.yaml` and fill in your model paths.
3. `start_llm.bat` (called automatically by `run.bat` / `run_dev.bat`) starts llama-swap using `tools\llama-swap.exe` and `llama-swap.yaml`.

If you use a different server (Ollama, LM Studio, llama.cpp directly), replace `start_llm.bat` with whatever starts your server on port 8000.

### LLM models (via llama-swap)
- `qwen3.5-35b` — main conversational agent (Qwen3.5-35B-A3B Q4_K_M recommended)
- `Qwen2.5-Omni-7B` — multimodal: intonation analysis, reference image description. Requires `--mmproj` flag in llama-swap config.

### ComfyUI models

**Image workflows — choose one per session:**

*ZIT* (default) — clean z_image_turbo pipeline with a character LoRA slot:
- `unet/z_image_turbo_bf16.safetensors` — main diffusion model (lumina2/AuraFlow)
- `clip/qwen_3_4b.safetensors` — CLIP text encoder (lumina2 type)
- `vae/ae.safetensors` — VAE
- A character LoRA of your choice in `loras/` — selected per session at upload

*Qwen Image Edit*:
- `Qwen-Rapid-AIO-NSFW-v23.safetensors` — checkpoint (CheckpointLoaderKJ)
- `anything2real_2601.safetensors` — LoRA

**Video workflows — choose one per session:**

Shared by all three video workflows (LTX 2.3 22B stack):
- `diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors`
- `clip/gemma_3_12B_it_heretic_fp8_e4m3fn.safetensors`
- `diffusion_models/ltx-2.3-22b-dev_embeddings_connectors.safetensors`
- `loras/ltx-2.3-22b-distilled-lora-384.safetensors`
- `vae/LTX23_video_vae_bf16.safetensors` + `vae/LTX23_audio_vae_bf16.safetensors`
- `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

*LTX + HuMo* and *LTX 22B + HuMo HQ* (HuMo refinement pass):
- `diffusion_models/humo_17B_fp16.safetensors`
- `loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors`
- `loras/FaceDetailerV1.safetensors`
- `clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors` — Wan2.1 CLIP
- `vae/Wan2_1_VAE_bf16.safetensors`
- `clip/whisper_large_v3_encoder_fp16.safetensors`
- `MelBandRoformer_fp16.safetensors`

*LTX 22B + HuMo HQ* additionally uses RTX Video Super Resolution (NVIDIA driver feature, no model file required).

### Required ComfyUI custom nodes

**Video workflows only** — no custom nodes required for ZIT T2I:

- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [ClownsharK samplers](https://github.com/ClownsharKent/ComfyUI-ClownsharKSamplers)
- MelBandRoFormer nodes (for vocal separation inside the video workflow)
- [WanExperiments](https://github.com/drozbay/WanExperiments) — **not in ComfyUI Manager, must be cloned manually** into `ComfyUI/custom_nodes/`

---

## Installation

**Prerequisites:** Python 3.11+ and Node.js must be installed. Everything else is handled automatically.

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your paths
3. Run `run_dev.bat`

`run_dev.bat` creates an isolated Python 3.11 virtual environment, installs PyTorch with CUDA 12.8 support (for Demucs GPU acceleration), installs all other Python dependencies, builds the frontend (running `npm install` automatically on first run), and starts the backend. No manual `pip install` or `npm install` needed.

> **CPU fallback:** If you don't have an NVIDIA GPU, Demucs will still run on CPU — just slower. Delete the `.venv` folder if it was created with GPU support and you need to switch, or vice versa.

All built-in ComfyUI workflow files and node maps ship with the repo in `backend/comfyui/workflows/` — no additional workflow setup required.

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
COMFYUI_DIR=C:/ComfyUI            # root install folder; input/output/models derived from here
COMFYUI_INPUT_DIR=C:/ComfyUI/input    # override if your layout differs
COMFYUI_OUTPUT_DIR=C:/ComfyUI/output
COMFYUI_MODEL_DIR=C:/ComfyUI/models   # used by setup verification and LoRA picker

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

---

## Running

**First time / after dependency changes:** double-click `run_dev.bat`. It installs everything automatically (Python venv, CUDA torch, npm packages) and starts the server.

**Subsequent runs:** double-click `run.bat` for a faster start — skips all installs, just builds the frontend and starts the backend.

Both scripts call `start_llm.bat` automatically before starting the backend. By default this starts llama-swap from `tools\llama-swap.exe`. Replace `start_llm.bat` with your own server start command if needed.

Open `http://localhost:8001`.

---

## Setup verification

The home screen has a **Verify Setup** card. Click it to check that all model files exist on disk, all required custom nodes are installed in ComfyUI, and both LLM endpoints are reachable. Each item shows pass/fail. Missing items link to the relevant repo.

You can set the ComfyUI folder and models folder directly in the Verify Setup page — no need to edit `.env` by hand. Use **Save to .env** to persist the paths. This check requires `COMFYUI_MODEL_DIR` to point at a directory that contains your model files.

---

## Per-session options

Set at upload time in the UI — these are locked for the lifetime of the session:

| Option | Choices |
|---|---|
| Orientation | Landscape, Portrait |
| Image Workflow | ZIT, Qwen Image Edit, any installed user workflow |
| Character LoRA | Any `.safetensors` in `COMFYUI_MODEL_DIR/loras/` — optional, with configurable strength. Can also be set by dropping a file onto the LoRA field. |
| Video Workflow | LTX + HuMo, LTX 22B + HuMo HQ, LTX, any installed user workflow |
| Final Resolution | 1280 / 1536 / 1920 (long edge, LTX with HuMo only) |

---

## Usage notes

**Use buttons, not chat, to advance phases.** When a button is visible — Generate Prompts, Generate Images, Generate Videos — click it. Asking the agent to "proceed" or "generate" in chat may not trigger the same code path.

**Don't ask the agent to change scene boundaries.** The agent doesn't compute timestamps — it calls an algorithmic segmenter. Asking it to move a cut to a specific time tends to produce hallucinated or incorrect values. Instead: ask for a different scene count, say "reroll" to get a different cut placement at the same count, or adjust boundaries directly in the scene cards.

**The style bible must go through the agent.** The style bible is written by the agent in chat and committed via a tool call — it isn't a form you fill in. Let the agent write it, review it in the sidebar, request edits in chat if needed, and confirm. Skipping or manually editing `prompts.json` before the agent commits it will break the prompt generation phase.

---

## Using your own workflows

The home screen has a **Manage Workflows** card. Upload a ComfyUI API-format workflow JSON and FADE auto-generates the node map by scanning for nodes with `FADE:`-prefixed titles. The workflow then appears as a selectable option on the upload screen.

Workflows with missing required node titles are rejected at import with a specific message listing which titles need to be added.

### Step 1 — Tag your nodes in ComfyUI

Right-click each node → **Title** and set the exact title string listed below.

**Image workflow — required:**

| Title | Node to tag |
|---|---|
| `FADE: Positive Prompt` | PrimitiveStringMultiline — positive text |
| `FADE: Negative Prompt` | PrimitiveStringMultiline — negative text |
| `FADE: Seed` | PrimitiveInt — seed value |
| `FADE: Width` | PrimitiveInt — output width |
| `FADE: Height` | PrimitiveInt — output height |
| `FADE: Save` | SaveImage node |

Optional:

| Title | Node to tag | If omitted |
|---|---|---|
| `FADE: Load Image` | LoadImage — reference image input (QIE-style workflows) | Reference image is not passed to the workflow |

**Video workflow — required:**

| Title | Node to tag |
|---|---|
| `FADE: Positive Prompt` | PrimitiveStringMultiline — positive text |
| `FADE: Start Frame` | LoadImage (receives the approved PNG from the image phase) |
| `FADE: Audio File` | VHS_LoadAudioUpload (receives the full session audio file) |
| `FADE: Frame Count` | PrimitiveInt — snapped 8k+1 frame count |
| `FADE: Save` | SaveVideo node |

Optional:

| Title | Node to tag | If omitted |
|---|---|---|
| `FADE: Negative Prompt` | PrimitiveStringMultiline — negative text | No negative prompt is applied |
| `FADE: Width` | PrimitiveInt | Orientation setting has no effect — workflow runs at its baked-in dimensions |
| `FADE: Height` | PrimitiveInt | Same as above — tag both or neither |
| `FADE: Seed` | PrimitiveInt — seed value | Seed is not set per-scene; workflow uses whatever default is in the JSON |
| `FADE: Audio Start` | PrimitiveFloat — scene start time in seconds | Omit if your audio node accepts `start_time` directly as an input field; FADE patches it there instead |
| `FADE: Seed 2` | Second PrimitiveInt seed (2-pass workflows) | Second pass uses its baked-in seed |
| `FADE: HuMo Seed` | HuMo sampler seed node | HuMo pass uses its baked-in seed |
| `FADE: HuMo Long Edge` | PrimitiveInt — HuMo refinement long edge | Final resolution setting has no effect |
| `FADE: FPS` | CreateVideo fps field | Output FPS is not set; workflow default is used |
| `FADE: FPS Conditioning` | LTXVConditioning frame_rate field | Frame rate conditioning is not set; workflow default is used |

### Step 2 — Export as API format

In ComfyUI: **Settings → Enable Dev Mode Options**, then use the **Save (API Format)** button (not the regular Save).

### Step 3 — Install via Manage Workflows

Open the **Manage Workflows** page from the home screen. Enter a display name, select Image or Video type, choose the exported JSON file, and click **Install**. FADE validates the node map and rejects the upload with a specific error if any required titles are missing.

User-installed workflows are stored in `backend/comfyui/workflows/user/` and are excluded from git.

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

Sessions survive server restarts — the UI restores your last saved phase automatically. Any scene that was mid-generation when the server stopped is automatically reset to its last stable state on resume.

---

## Write a Song — Suno / ACE-Step prompt assistant

A conversational agent that interviews you about the song you want to make and produces a complete prompt package ready to paste into Suno or send directly to the local generator. No audio analysis or ComfyUI involved.

Select the target platform at the top of the page:

**Suno mode** — produces a Suno v5 prompt package:
- Style tags (genre, subgenre, instrumentation, tempo, era, vocal descriptor)
- Structured lyrics with section markers (`[Verse]`, `[Chorus]`, `[Bridge]`, etc.) and inline metatags (`[melodic guitar solo]`, `[build]`, `[whispered]`)
- Generation notes: which style tags are doing the heavy lifting, suggested tweaks if the first gen misses, extension prompt snippet for longer songs

**ACE-Step mode** — produces an ACE-Step 1.5 prompt package:
- Caption (concise text description: genre, mood, instrumentation, energy)
- Structured lyrics with section markers
- Generation settings: BPM, key, time signature, duration

In ACE-Step mode a **Send to Generator** button appears once the package is ready — it pre-fills the Make a Song page and opens it directly.

---

## Make a Song — ACE-Step 1.5 local generator

Generates music locally using ACE-Step 1.5 (`acestep-v15-sft`, 60 steps). FADE manages the ACE-Step server process — it starts when you open the page and stops on exit.

**ACE-Step must be installed separately** in its own virtual environment. Run `scripts/setup_acestep.bat` (or `.sh`) to create it. Set `ACESTEP_VENV_DIR` in `.env` if you install it outside the repo, and `ACESTEP_URL` if you need a port other than `8002`.

The page shows a startup log while the server initialises (model load takes ~30s on first run). Once ready:

1. Fill in the prompt panel — caption, optional lyrics, BPM, key, time signature, duration — or arrive pre-filled from the Write a Song page.
2. Click **Generate**. Takes appear in the take strip as they complete.
3. Play takes, adjust the prompt panel, generate more.
4. Click **Use in FADE** on the selected take to send the audio and lyrics straight to the video director upload screen.
