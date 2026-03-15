#!/usr/bin/env python3
"""
test_stable_ts.py
Test stable-ts (Whisper-based) forced alignment against Burn It All vocals.
Run from music-director root: python scripts/test_stable_ts.py

Whisper model is downloaded on first run (~3GB for large-v3).
Use MODEL_SIZE = "medium" for a faster first test (~1.5GB).
"""

import re
import sys
from pathlib import Path

MODEL_SIZE  = "large-v3"   # tiny / base / small / medium / large-v2 / large-v3 / turbo
AUDIO_PATH  = "C:/Users/chand/music-director/tmp/vocals.wav"

RAW_LYRICS = """
[Intro]
It started with your T-shirt, the one you left behind
Held my lighter to the sleeve and watched the cotton catch and climb
Something in my chest went quiet for the first time in days
So I grabbed your letters, your photos, threw 'em in the blaze

[Verse 2]
Called my sister crying, said some things I can't take back
Told her all the poison that's been building in the cracks
She said "calm down, take a breath, you don't mean the things you say"
But the fire in my belly needs to burn something today

[Pre-Chorus]
It's not anger, it's not hate
It's just too much to hold inside
Something's gotta give, something's gotta break
Something's gotta die

[Chorus]
So I'm gonna burn it, burn it, burn it all
Every bridge, every photo on the wall
Your memory, my history, everything we built
I'm gonna burn it, burn it, burn it 'til
There's nothing left but ashes and the smell
Of every good thing going up in smoke
I'm gonna burn it, watch it all go up
'Cause I can't hold this anymore
I'm gonna burn it

[Verse 3]
Texted twenty people things I shouldn't have said
Scorched earth on every friendship, left 'em all on read
Quit my job at 2 AM, told my boss to go to hell
Burned my whole life to the ground just to feel something else

[Verse 4]
Now I'm standing in the driveway, gasoline and a match
Looking at the whole damn house, there ain't no coming back
I can see my whole life in there, everything I am
But the pressure in my skull says "baby, burn it while you can"

[Pre-Chorus 2]
It's not revenge, it's not a plan
It's just too big to stay inside
Something's gotta give, something's gotta end
Something's gotta die

[Chorus 2]
So I'm gonna burn it, burn it, burn it all
Every word, every bridge, every wall
Your number, my future, everything that's real
I'm gonna burn it, burn it, burn it 'til
There's nothing left but embers and the sound
Of my whole world crashing to the ground
I'm gonna burn it, watch it all go up
'Cause the feeling's too much to ignore
I'm gonna burn it

[Bridge]
Light it up, light it up, watch it go
Feel the heat, feel the heat, feel the glow
Everything I built, everything I saved
Everything I am going up in flames
Don't try to stop me, don't try to explain
The only thing that helps is the pain
Of watching something beautiful die
The only time I feel alive

[Final Chorus]
So I burned it, burned it, burned it all
Every dream, every hope, watched 'em fall
My whole life, your whole life, nothing left to feel
I just burned it, burned it, burned it 'til
There's nothing left but ash and smoke and flame
Standing in the ruins, no one left to blame
I just burned it, watched it all go up

[Outro]
Burn it down
Burn it down
Burn it down
""".strip()


def strip_suno_tags(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not re.fullmatch(r'\[.*?\]', line.strip())
    ).strip()


def fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m}:{s:05.2f}"


def main():
    clean_lyrics = strip_suno_tags(RAW_LYRICS)
    word_count = len(clean_lyrics.split())
    print(f"Clean lyrics: {word_count} words")
    print("-" * 60)

    print(f"Loading stable-ts model '{MODEL_SIZE}' ...")
    import stable_whisper
    model = stable_whisper.load_model(MODEL_SIZE)

    print(f"Aligning against {AUDIO_PATH} ...")
    result = model.align(AUDIO_PATH, clean_lyrics, language="en")

    words = result.all_words()
    print(f"\nResults: {len(result.segments)} segment(s), {len(words)} words total\n")

    # First 20 words
    print("-- First 20 words --")
    for w in words[:20]:
        dur = w.end - w.start
        print(f"  {fmt(w.start)} [{dur:.3f}s]  {w.word.strip()!r}")

    # Last 10 words
    print("\n-- Last 10 words --")
    for w in words[-10:]:
        dur = w.end - w.start
        print(f"  {fmt(w.start)} [{dur:.3f}s]  {w.word.strip()!r}")

    # Gaps > 3s
    print("\n-- Gaps > 3s between words --")
    found_gaps = False
    for i in range(1, len(words)):
        gap = words[i].start - words[i-1].end
        if gap > 3.0:
            before = words[max(0, i-4):i]
            after  = words[i:i+4]
            print(f"\n  GAP {gap:.1f}s  |  ends {fmt(words[i-1].end)} -> resumes {fmt(words[i].start)}")
            print(f"  BEFORE: {' / '.join(w.word.strip() for w in before)}")
            print(f"  AFTER:  {' / '.join(w.word.strip() for w in after)}")
            found_gaps = True
    if not found_gaps:
        print("  none")

    print(f"\nFinal word ends at: {fmt(words[-1].end)}")

    # Compare word count
    placed = len(words)
    pct = 100 * placed / word_count
    print(f"Words placed: {placed}/{word_count} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
