"""
Quick backend smoke test — no frontend needed.
Creates a session, uploads audio, connects WebSocket, sends one chat message.

Usage:
    python scripts/test_backend.py [audio_path]
"""

import asyncio
import json
import sys
import httpx
import websockets

# Force UTF-8 output on Windows so model Unicode (arrows, em-dashes, etc.) prints cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8001"
WS   = "ws://localhost:8001"

AUDIO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\chand\music-director\sessions\test\audio\Empty.wav"


async def main():
    async with httpx.AsyncClient() as http:

        # 1. Create session
        r = await http.post(f"{BASE}/sessions", data={"orientation": "portrait"})
        r.raise_for_status()
        sid = r.json()["session_id"]
        print(f"Session: {sid}")

        # 2. Upload audio
        with open(AUDIO, "rb") as f:
            r = await http.post(
                f"{BASE}/sessions/{sid}/upload/audio",
                files={"file": ("Empty.wav", f, "audio/wav")},
            )
        r.raise_for_status()
        print(f"Audio: {r.json()['audio_path']}")

    # 3. WebSocket — send first message, stream response
    uri = f"{WS}/sessions/{sid}/ws"
    print(f"\nConnecting to {uri} ...\n")

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "chat",
            "message": "Hi — I've uploaded the song. Can you separate the vocals and give me a quick analysis summary?",
        }))

        import time
        t0 = time.time()
        print("--- Agent response ---")
        async for raw in ws:
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "token":
                print(msg["text"], end="", flush=True)
            elif event == "assistant_done":
                print(f"\n--- Done ({time.time()-t0:.1f}s) ---")
                break
            elif event == "tool_call":
                print(f"\n[tool +{time.time()-t0:.0f}s] {msg['name']}({json.dumps(msg.get('args', {}))[:80]})")
            elif event == "tool_result":
                result = msg.get("result", "")
                print(f"[result] {str(result)[:120]}")
            elif event == "status":
                print(f"[status] {msg.get('message')}")
            elif event == "error":
                print(f"[ERROR] {msg.get('message')}")
                break


asyncio.run(main())
