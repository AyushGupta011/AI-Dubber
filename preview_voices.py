"""Voice Preview & Comparison Tool for Dubber.
Generates side-by-side audio samples for male and female voices across multiple accents,
and creates a combined dialogue sample where male and female voices converse.
"""

import asyncio
import subprocess
from pathlib import Path
import edge_tts
from dubber.utils.ffmpeg import FFmpegRunner

OUTPUT_DIR = Path("./voice_previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test phrases for individual voice samples
MALE_PHRASE = "Hello! I am testing the male voice for our AI video dubbing pipeline. The tone is clear and natural."
FEMALE_PHRASE = "Hi there! I am testing the female voice for our AI video dubbing pipeline. The pitch and clarity are balanced."

# Voice pairs to test (Male, Female) across regional accents
VOICE_PAIRS = {
    "1_Indian_English": {
        "male": ("en-IN-PrabhatNeural", "Prabhat (Indian Male)"),
        "female": ("en-IN-NeerjaNeural", "Neerja (Indian Female)"),
    },
    "2_US_Natural": {
        "male": ("en-US-GuyNeural", "Guy (US Male - Standard)"),
        "female": ("en-US-JennyNeural", "Jenny (US Female - Standard)"),
    },
    "3_US_Expressive": {
        "male": ("en-US-AndrewNeural", "Andrew (US Male - Energetic/Shorts)"),
        "female": ("en-US-AvaNeural", "Ava (US Female - Expressive)"),
    },
    "4_US_Deep_and_High": {
        "male": ("en-US-ChristopherNeural", "Christopher (US Male - Deep/Narrator)"),
        "female": ("en-US-AriaNeural", "Aria (US Female - Bright/Narrator)"),
    },
    "5_British_English": {
        "male": ("en-GB-RyanNeural", "Ryan (UK Male)"),
        "female": ("en-GB-SoniaNeural", "Sonia (UK Female)"),
    },
}

# Multi-speaker conversational dialogue script (Male & Female alternating)
DIALOGUE_SCRIPT = [
    ("en-IN-PrabhatNeural", "Prabhat (Male): Welcome to the voice comparison test. We are testing male and female voices at the same time."),
    ("en-IN-NeerjaNeural", "Neerja (Female): Exactly! Hearing both voices together helps you choose the perfect match for your video."),
    ("en-US-ChristopherNeural", "Christopher (Male): If your video features a deeper masculine tone, this voice is an excellent option."),
    ("en-US-JennyNeural", "Jenny (Female): And for natural, warm narrative explanations, this voice provides high clarity."),
    ("en-US-AndrewNeural", "Andrew (Male): For fast-paced YouTube Shorts or energetic content, this style keeps viewers engaged."),
    ("en-US-AvaNeural", "Ava (Female): And this expressive female voice keeps the delivery dynamic and engaging!"),
]


async def generate_single_sample(text: str, voice_id: str, out_path: Path):
    """Synthesizes a single audio clip via edge-tts."""
    communicate = edge_tts.Communicate(text=text, voice=voice_id)
    await communicate.save(str(out_path))
    print(f"  [+] Generated: {out_path.name}")


async def main():
    print("=" * 65)
    print(" DUBBER: MALE & FEMALE VOICE PREVIEW GENERATOR")
    print("=" * 65)
    print(f"Destination folder: {OUTPUT_DIR.resolve()}\n")

    tasks = []

    # 1. Generate individual male and female voice samples
    print("--- [1/2] Generating Individual Voice Previews ---")
    for group, pair in VOICE_PAIRS.items():
        male_voice, _ = pair["male"]
        female_voice, _ = pair["female"]

        male_file = OUTPUT_DIR / f"{group}_MALE_{male_voice}.mp3"
        female_file = OUTPUT_DIR / f"{group}_FEMALE_{female_voice}.mp3"

        tasks.append(generate_single_sample(MALE_PHRASE, male_voice, male_file))
        tasks.append(generate_single_sample(FEMALE_PHRASE, female_voice, female_file))

    # 2. Generate dialogue parts
    print("--- [2/2] Generating Alternating Male & Female Dialogue ---")
    dialogue_part_paths = []
    for idx, (voice, line) in enumerate(DIALOGUE_SCRIPT):
        part_file = OUTPUT_DIR / f"_temp_dialogue_part_{idx+1}.mp3"
        dialogue_part_paths.append(part_file)
        tasks.append(generate_single_sample(line, voice, part_file))

    await asyncio.gather(*tasks)

    # 3. Concatenate dialogue parts into one continuous audio track
    print("\nMerging dialogue parts into a single comparison audio file...")
    ffmpeg_bin = FFmpegRunner.get_ffmpeg_path()
    concat_list_file = OUTPUT_DIR / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in dialogue_part_paths:
            f.write(f"file '{p.name}'\n")

    combined_output = OUTPUT_DIR / "00_COMBINED_MALE_FEMALE_CONVERSATION.mp3"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(combined_output),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temp files
    concat_list_file.unlink(missing_ok=True)
    for p in dialogue_part_paths:
        p.unlink(missing_ok=True)

    print("\n" + "=" * 65)
    print("SUCCESS! All voice previews are ready.")
    print("=" * 65)
    print(f"Combined Conversation: {combined_output.resolve()}")
    print(f"All Audio Files Saved in: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
