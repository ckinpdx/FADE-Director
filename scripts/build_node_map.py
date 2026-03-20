#!/usr/bin/env python3
"""
build_node_map.py
Scan all four workflow files for FADE-titled nodes and write
node_map_t2i.json, node_map_qie.json, node_map_i2v.json, and
node_map_i2v_ltx.json to backend/comfyui/.

Run from the music-director project root:
    python scripts/build_node_map.py

T2I workflow portability (ZIT and QIE):
    In ComfyUI, rename the following nodes to these exact titles before
    exporting as API format:

    Required (all T2I workflows):
      "FADE: Positive Prompt"  — PrimitiveStringMultiline positive text
      "FADE: Negative Prompt"  — PrimitiveStringMultiline negative text
      "FADE: Seed"             — PrimitiveInt seed node
      "FADE: Width"            — PrimitiveInt width node
      "FADE: Height"           — PrimitiveInt height node
      "FADE: Save"             — SaveImage node

    Optional (title only if present in your workflow):
      "FADE: Load Image"       — LoadImage reference input (QIE only)

    The service patches:
      node["inputs"]["value"]            on FADE: Positive Prompt
      node["inputs"]["value"]            on FADE: Negative Prompt
      node["inputs"]["value"]            on FADE: Seed
      node["inputs"]["value"]            on FADE: Width / FADE: Height
      node["inputs"]["filename_prefix"]  on FADE: Save
      node["inputs"]["image"]            on FADE: Load Image (if present)

I2V workflow portability (LTX+HuMo and LTX I2V):
    In ComfyUI, set the title on each required node listed below before
    exporting as API format. Optional titles are silently skipped if absent.

    Required (all I2V workflows):
      "FADE: Positive Prompt"  — PrimitiveStringMultiline positive text
      "FADE: Start Frame"      — LoadImage (approved PNG from T2I)
      "FADE: Audio File"       — VHS_LoadAudioUpload (full session audio)
      "FADE: Frame Count"      — PrimitiveInt (8k+1 snapped frame count)
      "FADE: Width"            — PrimitiveInt
      "FADE: Height"           — PrimitiveInt
      "FADE: Seed"             — PrimitiveInt or RandomNoise seed node
      "FADE: Save"             — SaveVideo (filename_prefix)

    Optional (title only the nodes present in your workflow):
      "FADE: Negative Prompt"   — PrimitiveStringMultiline negative text
      "FADE: Audio Start"       — PrimitiveFloat (scene start_s) — omit if
                                  your audio node takes start_time directly
      "FADE: Seed 2"            — second PrimitiveInt/RandomNoise seed
      "FADE: HuMo Seed"         — ClownsharKSampler_Beta (HuMo pass)
      "FADE: HuMo Long Edge"    — PrimitiveInt (HuMo refinement long edge)
      "FADE: FPS"               — CreateVideo fps field
      "FADE: FPS Conditioning"  — LTXVConditioning frame_rate field

    Optional titles that are absent are silently skipped at runtime.
    Re-run this script any time you update or rebuild a workflow file.
"""

import json
import sys
from pathlib import Path

ROOT          = Path(__file__).parent.parent
WORKFLOWS_DIR = ROOT / "backend" / "comfyui" / "workflows"
OUTPUT_DIR    = ROOT / "backend" / "comfyui"

# ── T2I: required titles ─────────────────────────────────────────────────────
T2I_REQUIRED_TITLES = {
    "positive":        "FADE: Positive Prompt",
    "negative_prompt": "FADE: Negative Prompt",
    "seed":            "FADE: Seed",
    "width":           "FADE: Width",
    "height":          "FADE: Height",
    "save":            "FADE: Save",
}

# ── T2I: optional titles ─────────────────────────────────────────────────────
T2I_OPTIONAL_TITLES = {
    "load_image": "FADE: Load Image",   # LoadImage — reference input (QIE)
}

# ── I2V: required titles ─────────────────────────────────────────────────────
I2V_REQUIRED_TITLES = {
    "FADE: Positive Prompt": "video_prompt",   # PrimitiveStringMultiline
    "FADE: Start Frame":     "start_frame",    # LoadImage — approved PNG
    "FADE: Audio File":      "audio_file",     # VHS_LoadAudioUpload
    "FADE: Frame Count":     "frame_count",    # PrimitiveInt — 8k+1
    "FADE: Width":           "width",          # PrimitiveInt
    "FADE: Height":          "height",         # PrimitiveInt
    "FADE: Seed":            "ltx_seed",       # PrimitiveInt or RandomNoise
    "FADE: Save":            "save",           # SaveVideo
}

# ── I2V: optional titles ─────────────────────────────────────────────────────
I2V_OPTIONAL_TITLES = {
    "FADE: Negative Prompt":  "negative_prompt",   # PrimitiveStringMultiline
    "FADE: Audio Start":      "audio_start",       # PrimitiveFloat (scene start_s)
    "FADE: Seed 2":           "ltx_seed_2",        # second seed node (2-pass LTX)
    "FADE: HuMo Seed":        "humo_seed",         # ClownsharKSampler_Beta
    "FADE: HuMo Long Edge":   "humo_long_edge",    # PrimitiveInt
    "FADE: FPS":              "create_video",      # CreateVideo fps
    "FADE: FPS Conditioning": "ltxv_conditioning", # LTXVConditioning frame_rate
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_workflow(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_by_title(workflow: dict, title: str) -> str | None:
    """Return the node ID whose _meta.title matches exactly, or None."""
    for node_id, node in workflow.items():
        if node.get("_meta", {}).get("title") == title:
            return node_id
    return None


# ── T2I ──────────────────────────────────────────────────────────────────────

def build_t2i_map(workflow: dict, wf_name: str) -> dict:
    node_map = {}
    errors   = []
    for logical, title in T2I_REQUIRED_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is None:
            errors.append(
                f"  [ERROR] {wf_name}: no node titled {title!r}\n"
                f"         In ComfyUI, rename your {logical} node to this exact title."
            )
        else:
            node_map[logical] = nid
            print(f"  T2I {logical:16s} -> node {nid} ({workflow[nid]['class_type']})")
    for logical, title in T2I_OPTIONAL_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is not None:
            node_map[logical] = nid
            print(f"  T2I {logical:16s} -> node {nid} ({workflow[nid]['class_type']})  [optional]")
        else:
            print(f"  T2I {logical:16s} -> not present (optional, skipped at runtime)")
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    return node_map


# ── I2V ──────────────────────────────────────────────────────────────────────

def build_i2v_map(workflow: dict, wf_name: str) -> dict:
    node_map = {}
    errors   = []

    for title, logical in I2V_REQUIRED_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is None:
            errors.append(
                f"  [ERROR] {wf_name}: no node titled {title!r}\n"
                f"         In ComfyUI, set the title on your {logical} node."
            )
        else:
            node_map[logical] = nid
            print(f"  I2V  {logical:22s} -> node {nid} ({workflow[nid]['class_type']})")

    for title, logical in I2V_OPTIONAL_TITLES.items():
        nid = find_by_title(workflow, title)
        if nid is not None:
            node_map[logical] = nid
            print(f"  I2V  {logical:22s} -> node {nid} ({workflow[nid]['class_type']})  [optional]")
        else:
            print(f"  I2V  {logical:22s} -> not present (optional, skipped at runtime)")

    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    return node_map


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    tasks = [
        # (workflow_file, output_map_file, builder_fn)
        ("ltx2_t2i.json",      "node_map_t2i.json",      "t2i"),
        ("qie_t2i.json",       "node_map_qie.json",       "t2i"),
        ("ltx2_i2v_humo.json", "node_map_i2v.json",       "i2v"),
        ("ltx2_i2v.json",      "node_map_i2v_ltx.json",   "i2v"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for wf_name, out_name, kind in tasks:
        wf_path = WORKFLOWS_DIR / wf_name
        if not wf_path.exists():
            print(f"WARNING: {wf_path} not found — skipping {out_name}")
            continue

        print(f"\nScanning {wf_name} ...")
        wf = load_workflow(wf_path)

        if kind == "t2i":
            node_map = build_t2i_map(wf, wf_name)
        else:
            node_map = build_i2v_map(wf, wf_name)

        out_path = OUTPUT_DIR / out_name
        out_path.write_text(json.dumps(node_map, indent=2), encoding="utf-8")
        print(f"Wrote {out_path.relative_to(ROOT)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
