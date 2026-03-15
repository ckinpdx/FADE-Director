#!/usr/bin/env python3
"""
test_aligner.py
Quick test of Qwen3-ForcedAligner against Burn It All vocals.
Run from music-director root: python scripts/test_aligner.py
"""

import re
import sys
from pathlib import Path

MODEL_PATH = "C:/ComfyUI/models/Qwen3-ASR/Qwen3-ForcedAligner-0.6B"
AUDIO_PATH = "C:/Users/chand/music-director/tmp/vocals.wav"

# Suno-tagged lyrics — strip all [...] tag lines before passing to aligner
RAW_LYRICS = """
It started with your T-shirt, the one you left behind
Held my lighter to the sleeve and watched the cotton catch and climb
Something in my chest went quiet for the first time in days
So I grabbed your letters, your photos, threw 'em in the blaze

Called my sister crying, said some things I can't take back
Told her all the poison that's been building in the cracks
She said "calm down, take a breath, you don't mean the things you say"
But the fire in my belly needs to burn something today

It's not anger, it's not hate
It's just too much to hold inside
Something's gotta give, something's gotta break
Something's gotta die

So I'm gonna burn it, burn it, burn it all
Every bridge, every photo on the wall
Your memory, my history, everything we built
I'm gonna burn it, burn it, burn it 'til
There's nothing left but ashes and the smell
Of every good thing going up in smoke
I'm gonna burn it, watch it all go up
'Cause I can't hold this anymore
I'm gonna burn it

Texted twenty people things I shouldn't have said
Scorched earth on every friendship, left 'em all on read
Quit my job at 2 AM, told my boss to go to hell
Burned my whole life to the ground just to feel something else

Now I'm standing in the driveway, gasoline and a match
Looking at the whole damn house, there ain't no coming back
I can see my whole life in there, everything I am
But the pressure in my skull says "baby, burn it while you can"

It's not revenge, it's not a plan
It's just too big to stay inside
Something's gotta give, something's gotta end
Something's gotta die

So I'm gonna burn it, burn it, burn it all
Every word, every bridge, every wall
Your number, my future, everything that's real
I'm gonna burn it, burn it, burn it 'til
There's nothing left but embers and the sound
Of my whole world crashing to the ground
I'm gonna burn it, watch it all go up
'Cause the feeling's too much to ignore
I'm gonna burn it

Light it up, light it up, watch it go
Feel the heat, feel the heat, feel the glow
Everything I built, everything I saved
Everything I am going up in flames
Don't try to stop me, don't try to explain
The only thing that helps is the pain
Of watching something beautiful die
The only time I feel alive

So I burned it, burned it, burned it all
Every dream, every hope, watched 'em fall
My whole life, your whole life, nothing left to feel
I just burned it, burned it, burned it 'til
There's nothing left but ash and smoke and flame
Standing in the ruins, no one left to blame
I just burned it, watched it all go up

Burn it down
Burn it down
Burn it down
""".strip()


def strip_suno_tags(text: str) -> str:
    """Remove lines that are purely Suno structural/direction tags."""
    lines = text.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are only a [...] tag (section labels, direction cues)
        if re.fullmatch(r'\[.*?\]', stripped):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def main():
    clean_lyrics = strip_suno_tags(RAW_LYRICS)
    word_count = len(clean_lyrics.split())
    print(f"Clean lyrics: {word_count} words")
    print("-" * 60)
    print(clean_lyrics[:300], "..." if len(clean_lyrics) > 300 else "")
    print("-" * 60)

    print(f"\nLoading aligner from {MODEL_PATH} ...")
    from qwen_asr import Qwen3ForcedAligner
    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH)

    print(f"Aligning against {AUDIO_PATH} ...")
    segments = model.align(audio=AUDIO_PATH, text=clean_lyrics, language="English")

    # Flatten: result is list of ForcedAlignResult; each has .items (list of ForcedAlignItem)
    words = [item for chunk in segments for item in chunk.items]

    print(f"\nResults: {len(segments)} chunk(s), {len(words)} words total\n")

    # Print first 20 words
    print("-- First 20 words --")
    for w in words[:20]:
        dur = w.end_time - w.start_time
        print(f"  {w.start_time:7.3f}s [{dur:.3f}s]  {w.text}")

    # Print last 10 words
    print("\n-- Last 10 words --")
    for w in words[-10:]:
        dur = w.end_time - w.start_time
        print(f"  {w.start_time:7.3f}s [{dur:.3f}s]  {w.text}")

    # Check for suspiciously long gaps (potential alignment failures)
    print("\n-- Gaps > 3s between words --")
    found_gaps = False
    for i in range(1, len(words)):
        gap = words[i].start_time - words[i-1].end_time
        if gap > 3.0:
            print(f"  {gap:.2f}s gap after \"{words[i-1].text}\" ({words[i-1].end_time:.2f}s) before \"{words[i].text}\" ({words[i].start_time:.2f}s)")
            found_gaps = True
    if not found_gaps:
        print("  none")

    print(f"\nFinal word ends at: {words[-1].end_time:.2f}s")


if __name__ == "__main__":
    main()
