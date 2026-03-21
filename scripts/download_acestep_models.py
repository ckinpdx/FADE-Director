"""
download_acestep_models.py
Download all ACE-Step 1.5 models needed for the SFT config.

Usage:
    python download_acestep_models.py <checkpoint_dir>

Called from setup_acestep.bat with FADE-Director/checkpoints as the target.
Each model is skipped if weight files are already present.
GPU VRAM is detected at runtime to pick the right LM size.
"""

from __future__ import annotations

import os
import sys


def _has_weights(directory: str) -> bool:
    """Return True if directory contains at least one weight file."""
    if not os.path.isdir(directory):
        return False
    for _, _, files in os.walk(directory):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".ckpt")):
                return True
    return False


def _download(repo_id: str, local_dir: str, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: download_acestep_models.py <checkpoint_dir>")
        sys.exit(1)

    checkpoint_dir = os.path.abspath(sys.argv[1])
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()

    # ── 1. SFT transformer (separate repo) ───────────────────────────────────
    sft_dir = os.path.join(checkpoint_dir, "acestep-v15-sft")
    if _has_weights(sft_dir):
        print("[1/4] acestep-v15-sft already present — skipping.")
    else:
        print("[1/4] Downloading ACE-Step v1.5 SFT model (~4.5 GB)...")
        print("      Repo: ACE-Step/acestep-v15-sft")
        _download("ACE-Step/acestep-v15-sft", sft_dir)
        print("[1/4] Done.\n")

    # ── 2. VAE (unified repo, subfolder only) ────────────────────────────────
    vae_dir = os.path.join(checkpoint_dir, "vae")
    if _has_weights(vae_dir):
        print("[2/4] vae already present — skipping.")
    else:
        print("[2/4] Downloading VAE (~0.5 GB)...")
        print("      Repo: ACE-Step/Ace-Step1.5  folder: vae/")
        _download("ACE-Step/Ace-Step1.5", checkpoint_dir, allow_patterns=["vae/*"])
        print("[2/4] Done.\n")

    # ── 3. Text encoder (unified repo, subfolder only) ───────────────────────
    enc_dir = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    if _has_weights(enc_dir):
        print("[3/4] Qwen3-Embedding-0.6B already present — skipping.")
    else:
        print("[3/4] Downloading text encoder Qwen3-Embedding-0.6B (~1.2 GB)...")
        print("      Repo: ACE-Step/Ace-Step1.5  folder: Qwen3-Embedding-0.6B/")
        _download("ACE-Step/Ace-Step1.5", checkpoint_dir, allow_patterns=["Qwen3-Embedding-0.6B/*"])
        print("[3/4] Done.\n")

    # ── 4. LM model — size chosen by GPU VRAM ────────────────────────────────
    print("[4/4] Detecting GPU to select LM model size...")
    try:
        from acestep.gpu_config import get_gpu_config
        gpu_config = get_gpu_config()
        if gpu_config.available_lm_models:
            lm_model = gpu_config.available_lm_models[-1]   # largest that fits
            print(f"      GPU: {gpu_config.gpu_memory_gb:.1f} GB VRAM  →  selected: {lm_model}")
        else:
            lm_model = "acestep-5Hz-lm-0.6B"
            print(f"      GPU VRAM too low for LLM guidance; using smallest: {lm_model}")
    except Exception as e:
        lm_model = "acestep-5Hz-lm-1.7B"
        print(f"      GPU detection failed ({e}); defaulting to: {lm_model}")

    lm_dir = os.path.join(checkpoint_dir, lm_model)
    if _has_weights(lm_dir):
        print(f"[4/4] {lm_model} already present — skipping.")
    else:
        # Map model name to HuggingFace repo
        _UNIFIED_REPO = "ACE-Step/Ace-Step1.5"
        _LM_REPOS = {
            "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B": _UNIFIED_REPO,   # subfolder of unified repo
            "acestep-5Hz-lm-4B":   "ACE-Step/acestep-5Hz-lm-4B",
        }
        lm_repo = _LM_REPOS.get(lm_model, f"ACE-Step/{lm_model}")
        size_hint = {"0.6B": "~1.2 GB", "1.7B": "~3.4 GB", "4B": "~8 GB"}.get(
            lm_model.split("-")[-1], ""
        )
        print(f"[4/4] Downloading {lm_model} {size_hint}...")
        print(f"      Repo: {lm_repo}")
        if lm_repo == _UNIFIED_REPO:
            _download(lm_repo, checkpoint_dir, allow_patterns=[f"{lm_model}/*"])
        else:
            _download(lm_repo, lm_dir)
        print(f"[4/4] Done.\n")

    print("=" * 60)
    print("All ACE-Step models ready.")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
