#!/usr/bin/env python3
"""
build_node_map.py
Scan both workflow files for FADE-titled nodes and write
node_map_t2i.json + node_map_i2v.json to backend/comfyui/.

Run from the music-director project root:
    python scripts/build_node_map.py

T2I workflow portability:
    Any T2I workflow works. In ComfyUI, rename the following 4 nodes
    to these exact titles before exporting as API format:
      "FADE: Positive Prompt"  — your main positive CLIPTextEncode node
      "FADE: Seed"             — your KSampler (or equivalent sampler) node
      "FADE: Dimensions"       — your empty latent image node (width + height)
      "FADE: Save"             — your SaveImage node

    The service patches:
      node["inputs"]["text"]   on FADE: Positive Prompt
      node["inputs"]["seed"]   on FADE: Seed
      node["inputs"]["width"], node["inputs"]["height"] on FADE: Dimensions
      node["inputs"]["filename_prefix"] on FADE: Save

    The negative prompt is intentionally NOT patched — z_image_turbo uses
    Decoupled DMD distillation (effective CFG=0); the negative has no effect.
    If you use a workflow where negatives matter, patch node 158 manually.

I2V workflow portability:
    Any I2V workflow works. In ComfyUI, set the title on each node listed
    in I2V_REQUIRED_TITLES (9 nodes) and any I2V_OPTIONAL_TITLES that apply
    to your specific pipeline before exporting as API format.

    Required (all workflows):
      "FADE: Video Prompt"    — positive text (PrimitiveStringMultiline or CLIPTextEncode)
      "FADE: Start Frame"     — LoadImage (approved PNG from T2I)
      "FADE: Audio File"      — VHS_LoadAudioUpload (full session audio)
      "FADE: Audio Start"     — PrimitiveFloat (scene start_s)
      "FADE: Frame Count"     — PrimitiveInt (8k+1 snapped frame count)
      "FADE: Width"           — PrimitiveInt
      "FADE: Height"          — PrimitiveInt
      "FADE: Seed"            — RandomNoise or sampler seed node
      "FADE: Save"            — SaveVideo (filename_prefix)

    Optional (title only the nodes present in your workflow):
      "FADE: Negative Prompt"   — PrimitiveStringMultiline negative text
      "FADE: Seed 2"            — second RandomNoise (2-pass LTX workflows)
      "FADE: HuMo Seed"         — ClownsharKSampler_Beta (HuMo refinement pass)
      "FADE: HuMo Long Edge"    — JWImageResizeByLongerSide (HuMo refinement pass)
      "FADE: FPS"               — CreateVideo fps field
      "FADE: FPS Conditioning"  — LTXVConditioning frame_rate field

    Optional titles that are absent are silently skipped at runtime.
    Re-run this script any time you update or rebuild a workflow file.
"""

import json
import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / "backend" / "comfyui" / "workflows"
OUTPUT_DIR   = ROOT / "backend" / "comfyui"

# ── T2I: required titles ────────────────────────────────────────────────────
# These titles must exist in ltx2_t2i.json._meta.title for the service to work.
T2I_REQUIRED_TITLES = {
    "positive":   "FADE: Positive Prompt",   # CLIPTextEncode — positive text
    "seed":       "FADE: Seed",              # KSampler or equivalent — seed field
    "dimensions": "FADE: Dimensions",        # EmptyLatentImage — width + height
    "save":       "FADE: Save",              # SaveImage — filename_prefix
}

# ── I2V: required titles ─────────────────────────────────────────────────────
# All 9 must be present in ltx2_i2v_humo.json — script errors if any is missing.
I2V_REQUIRED_TITLES = {
    "FADE: Video Prompt":   "video_prompt",   # PrimitiveStringMultiline — positive text
    "FADE: Start Frame":    "start_frame",    # LoadImage — approved PNG from T2I
    "FADE: Audio File":     "audio_file",     # VHS_LoadAudioUpload — full session audio
    "FADE: Audio Start":    "audio_start",    # PrimitiveFloat — scene start_s
    "FADE: Frame Count":    "frame_count",    # PrimitiveInt — 8k+1 snapped frame count
    "FADE: Width":          "width",          # PrimitiveInt
    "FADE: Height":         "height",         # PrimitiveInt
    "FADE: Seed":           "ltx_seed",       # RandomNoise or sampler seed node
    "FADE: Save":           "save",           # SaveVideo — filename_prefix
}

# ── I2V: optional titles ─────────────────────────────────────────────────────
# Title only the nodes present in your workflow. Absent titles are silently
# skipped at runtime — the corresponding patch is simply omitted.
I2V_OPTIONAL_TITLES = {
    "FADE: Negative Prompt":  "negative_prompt",   # PrimitiveStringMultiline
    "FADE: Seed 2":           "ltx_seed_2",        # second RandomNoise (2-pass LTX)
    "FADE: HuMo Seed":        "humo_seed",         # ClownsharKSampler_Beta
    "FADE: HuMo Long Edge":   "humo_long_edge",    # JWImageResizeByLongerSide
    "FADE: FPS":              "create_video",      # CreateVideo — fps field
    "FADE: FPS Conditioning": "ltxv_conditioning", # LTXVConditioning — frame_rate field
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_workflow(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_by_title(workflow: dict, title: str) -> str | None:
    """Return the node ID whose _meta.title matches exactly, or None."""
    for node_id, node in workflow.items():
        if node.get("_meta", {}).get("title") == title:
            return node_id
    return None


# ── T2I ─────────────────────────────────────────────────────────────────────

def build_t2i_map(workflow: dict) -> dict:
    node_map = {}
    errors   = []
    for logical, title in T2I_REQUIRED_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is None:
            errors.append(
                f"  [ERROR] ltx2_t2i.json: no node titled {title!r}\n"
                f"         In ComfyUI, rename your {logical} node to this exact title."
            )
        else:
            node_map[logical] = nid
            print(f"  T2I {logical:12s} ->node {nid} ({workflow[nid]['class_type']})")
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    return node_map


# ── I2V ─────────────────────────────────────────────────────────────────────

def build_i2v_map(workflow: dict) -> dict:
    node_map = {}
    errors   = []

    for title, logical in I2V_REQUIRED_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is None:
            errors.append(
                f"  [ERROR] ltx2_i2v_humo.json: no node titled {title!r}\n"
                f"         In ComfyUI, set the title on your {logical} node."
            )
        else:
            node_map[logical] = nid
            print(f"  I2V  {logical:22s} ->node {nid} ({workflow[nid]['class_type']})")

    for title, logical in I2V_OPTIONAL_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is not None:
            node_map[logical] = nid
            print(f"  I2V  {logical:22s} ->node {nid} ({workflow[nid]['class_type']})  [optional]")
        else:
            print(f"  I2V  {logical:22s} ->not present (optional, skipped at runtime)")

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    return node_map


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    t2i_path = WORKFLOWS_DIR / "ltx2_t2i.json"
    i2v_path = WORKFLOWS_DIR / "ltx2_i2v_humo.json"

    for p in (t2i_path, i2v_path):
        if not p.exists():
            print(f"ERROR: {p} not found — copy your workflow file here first.")
            sys.exit(1)

    t2i_wf = load_workflow(t2i_path)
    i2v_wf = load_workflow(i2v_path)

    print(f"Scanning {t2i_path.name} ...")
    t2i_map = build_t2i_map(t2i_wf)

    print(f"\nScanning {i2v_path.name} ...")
    i2v_map = build_i2v_map(i2v_wf)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t2i_out = OUTPUT_DIR / "node_map_t2i.json"
    t2i_out.write_text(json.dumps(t2i_map, indent=2), encoding="utf-8")
    print(f"\nWrote {t2i_out.relative_to(ROOT)}")

    i2v_out = OUTPUT_DIR / "node_map_i2v.json"
    i2v_out.write_text(json.dumps(i2v_map, indent=2), encoding="utf-8")
    print(f"Wrote {i2v_out.relative_to(ROOT)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
