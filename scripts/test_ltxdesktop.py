"""
Test the LTX-Desktop video generation pipeline.

Stages:
  1. Config / env check
  2. Audio slice (ffmpeg)
  3. Health check (non-blocking — skips gen if server not running)
  4. Full I2V generation  (requires --full flag + server running)

Usage:
    # Stages 1-3 only (safe, no generation):
    python scripts/test_ltxdesktop.py [audio] [image]

    # Full generation test (starts LTX-Desktop if not running):
    python scripts/test_ltxdesktop.py --full [audio] [image]

Defaults:
    audio  — sessions/test/audio/Empty.wav
    image  — sessions/test/images/scene_1.png
"""

from __future__ import annotations

import asyncio
import sys
import time
import os
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Args ──────────────────────────────────────────────────────────────────────

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = set(sys.argv[1:]) - set(args)
FULL = "--full" in flags

AUDIO_PATH = Path(args[0]) if len(args) > 0 else Path(r"C:\Users\chand\music-director\sessions\test\audio\Empty.wav")
IMAGE_PATH = Path(args[1]) if len(args) > 1 else Path(r"C:\Users\chand\music-director\sessions\test\images\scene_1.png")

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Stage 1: Config ───────────────────────────────────────────────────────────

async def test_config() -> bool:
    header("Stage 1 — Config")

    # Add project root to path so backend imports work
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))

    try:
        from backend import config
        print(f"{INFO} VIDEO_BACKEND   = {config.VIDEO_BACKEND!r}")
        print(f"{INFO} LTXDESKTOP_URL  = {config.LTXDESKTOP_URL!r}")
        print(f"{INFO} LTXDESKTOP_PATH = {config.LTXDESKTOP_PATH!r}")

        if config.VIDEO_BACKEND != "ltxdesktop":
            print(f"{FAIL} VIDEO_BACKEND is {config.VIDEO_BACKEND!r} — set VIDEO_BACKEND=ltxdesktop in .env")
            return False

        if not config.LTXDESKTOP_PATH:
            print(f"{FAIL} LTXDESKTOP_PATH is not set — set it to your LTX-Desktop fork root")
            return False

        if not config.LTXDESKTOP_DATA_DIR:
            print(f"{FAIL} LTXDESKTOP_DATA_DIR is not set — set it to your LTX_APP_DATA_DIR (models live there)")
            return False

        data_dir = Path(config.LTXDESKTOP_DATA_DIR)
        models_dir = data_dir / "models"
        print(f"{INFO} Models dir: {models_dir}")
        required = [
            "ltx-2.3-22b-distilled.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            "gemma-3-12b-it-qat-q4_0-unquantized",
        ]
        missing = [f for f in required if not (models_dir / f).exists()]
        if missing:
            print(f"{FAIL} Missing models in {models_dir}:")
            for m in missing:
                print(f"         - {m}")
            print(f"       Download via: huggingface-cli download Lightricks/LTX-2.3 ...")
            return False

        fork_root = Path(config.LTXDESKTOP_PATH)
        if not fork_root.exists():
            print(f"{FAIL} LTXDESKTOP_PATH does not exist: {fork_root}")
            return False

        server_file = fork_root / "ltx2_server.py"
        if not server_file.exists():
            print(f"{FAIL} Expected server at {server_file} — check LTXDESKTOP_PATH")
            return False

        print(f"{PASS} Config OK — fork found at {fork_root}")
        return True

    except Exception as e:
        print(f"{FAIL} Config load failed: {e}")
        return False


# ── Stage 2: Audio slice ──────────────────────────────────────────────────────

async def test_audio_slice() -> bool:
    header("Stage 2 — Audio slice (ffmpeg)")

    if not AUDIO_PATH.exists():
        print(f"{FAIL} Audio file not found: {AUDIO_PATH}")
        print(f"       Pass a valid audio path as the first argument.")
        return False

    print(f"{INFO} Source: {AUDIO_PATH}")

    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    from backend.ltxdesktop.client import slice_audio

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "slice_test.wav"
        t0   = time.perf_counter()
        try:
            await slice_audio(AUDIO_PATH, start_s=0.0, duration_s=5.0, dest=dest)
            elapsed = time.perf_counter() - t0

            if not dest.exists():
                print(f"{FAIL} slice_audio returned but file not found")
                return False

            size = dest.stat().st_size
            print(f"{PASS} Sliced 5s WAV in {elapsed:.2f}s  ({size // 1024} KB)")
            return True

        except Exception as e:
            print(f"{FAIL} slice_audio raised: {e}")
            return False


# ── Stage 3: Health check ─────────────────────────────────────────────────────

async def test_health() -> bool | None:
    """Returns True if healthy, False if error, None if not running (expected)."""
    header("Stage 3 — Health check")

    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    from backend.ltxdesktop.process import health_check
    from backend import config

    print(f"{INFO} Checking {config.LTXDESKTOP_URL}/health ...")
    healthy = await health_check()

    if healthy:
        print(f"{PASS} LTX-Desktop is running and healthy")
        return True
    else:
        print(f"{SKIP} LTX-Desktop is not running (expected if not started manually)")
        if not FULL:
            print(f"       Use --full to start it automatically and run generation test")
        return None


# ── Stage 4: Full I2V generation ──────────────────────────────────────────────

async def test_generation(server_already_running: bool) -> bool:
    header("Stage 4 — Full I2V generation")

    if not IMAGE_PATH.exists():
        print(f"{FAIL} Image file not found: {IMAGE_PATH}")
        print(f"       Pass a valid PNG path as the second argument.")
        return False

    if not AUDIO_PATH.exists():
        print(f"{FAIL} Audio file not found: {AUDIO_PATH}")
        return False

    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    from backend.ltxdesktop import process as ltx_proc
    from backend.ltxdesktop import client  as ltx_client
    from backend import config

    # Start server if not already running
    if not server_already_running:
        print(f"{INFO} Starting LTX-Desktop ...")
        try:
            await ltx_proc.start()
            print(f"{PASS} LTX-Desktop started")
        except Exception as e:
            print(f"{FAIL} Failed to start LTX-Desktop: {e}")
            return False

    # Use a small frame count (9 frames = minimum valid 8k+1) for a fast test
    TEST_FRAMES    = 9
    TEST_FPS       = 25
    TEST_DURATION  = TEST_FRAMES / TEST_FPS  # 0.36s — just enough to confirm the API works

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path      = Path(tmp)
        audio_clip    = tmp_path / "clip.wav"
        output_mp4    = tmp_path / "output.mp4"

        # Slice a minimal audio clip
        print(f"{INFO} Slicing {TEST_DURATION:.2f}s audio clip ...")
        await ltx_client.slice_audio(AUDIO_PATH, 0.0, TEST_DURATION, audio_clip)

        print(f"{INFO} Submitting I2V: {TEST_FRAMES} frames @ {TEST_FPS}fps ...")
        print(f"{INFO} Image:  {IMAGE_PATH}")
        print(f"{INFO} Output: {output_mp4}")

        t0 = time.perf_counter()
        try:
            out = await ltx_client.generate_video(
                video_prompt    = "A cinematic scene, smooth motion, high quality.",
                image_path      = IMAGE_PATH,
                audio_clip_path = audio_clip,
                frame_count     = TEST_FRAMES,
                output_path     = output_mp4,
                orientation     = "landscape",
                fps             = TEST_FPS,
                seed            = 42,
                negative_prompt = "blurry, low quality",
            )
            elapsed = time.perf_counter() - t0
            size    = out.stat().st_size
            print(f"{PASS} Generation done in {elapsed:.1f}s  output={out}  ({size // 1024} KB)")
            return True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"{FAIL} Generation failed after {elapsed:.1f}s: {e}")
            return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\nLTX-Desktop pipeline test")
    print(f"FULL mode: {FULL}")

    results: dict[str, str] = {}

    ok = await test_config()
    results["config"] = PASS if ok else FAIL
    if not ok:
        print(f"\n{FAIL} Config check failed — fix LTXDESKTOP_PATH / VIDEO_BACKEND first.")
        _print_summary(results)
        sys.exit(1)

    ok = await test_audio_slice()
    results["audio_slice"] = PASS if ok else FAIL

    health_result = await test_health()
    results["health"] = PASS if health_result else (SKIP if health_result is None else FAIL)

    if FULL:
        ok = await test_generation(server_already_running=(health_result is True))
        results["generation"] = PASS if ok else FAIL
    else:
        results["generation"] = SKIP

    _print_summary(results)

    failures = [k for k, v in results.items() if v == FAIL]
    if failures:
        sys.exit(1)


def _print_summary(results: dict[str, str]) -> None:
    header("Summary")
    for stage, result in results.items():
        print(f"  {result}  {stage}")
    print()


asyncio.run(main())
