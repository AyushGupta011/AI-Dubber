"""Helper script to generate realistic test video clips in Hindi and French for end-to-end pipeline validation."""

import asyncio
import os
import subprocess
from pathlib import Path
import edge_tts
import soundfile as sf
import numpy as np
from dubber.utils.ffmpeg import FFmpegRunner, get_media_duration


async def generate_speech_audio(text: str, voice: str, output_wav: Path, sr: int = 16000):
    temp_mp3 = output_wav.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(temp_mp3))

    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", str(temp_mp3),
        "-ar", str(sr),
        "-ac", "1",
        str(output_wav)
    ]
    FFmpegRunner.run_command(cmd, check=True)
    if temp_mp3.exists():
        temp_mp3.unlink()


def create_test_video(audio_path: Path, output_video_path: Path, title: str = "Test Video"):
    duration = get_media_duration(audio_path) + 1.0
    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate test video pattern with test audio
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration:.2f}:size=640x360:rate=25",
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_video_path)
    ]
    FFmpegRunner.run_command(cmd, check=True)
    print(f"Generated test video: {output_video_path} (Duration: {get_media_duration(output_video_path):.2f}s)")


async def build_hindi_clip():
    out_dir = Path("./test_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hindi script sentences
    hindi_text = (
        "नमस्ते दोस्तों! कृत्रिम बुद्धिमत्ता की दुनिया में आपका स्वागत है। "
        "आज हम वीडियो डबिंग तकनीक के बारे में बात करेंगे। "
        "यह प्रणाली आपकी आवाज़ को अंग्रेजी में अनुवाद और रूपांतरित कर सकती है।"
    )
    wav_path = out_dir / "hindi_speech.wav"
    await generate_speech_audio(hindi_text, "hi-IN-MadhurNeural", wav_path)
    video_path = out_dir / "hindi_test_clip.mp4"
    create_test_video(wav_path, video_path, title="Hindi Test Clip")
    return video_path


async def build_french_clip():
    out_dir = Path("./test_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # French script sentences
    french_text = (
        "Bonjour tout le monde! Bienvenue dans notre présentation sur l'intelligence artificielle. "
        "Aujourd'hui, nous explorons les méthodes modernes de traduction automatique et de synthèse vocale. "
        "Merci d'avoir regardé cette démonstration."
    )
    wav_path = out_dir / "french_speech.wav"
    await generate_speech_audio(french_text, "fr-FR-HenriNeural", wav_path)
    video_path = out_dir / "french_test_clip.mp4"
    create_test_video(wav_path, video_path, title="French Test Clip")
    return video_path


async def build_2min_hindi_video():
    out_dir = Path("./test_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Longer 2-minute Hindi lecture
    long_hindi_text = (
        "नमस्ते और स्वागत है। आज के इस व्याख्यान में हम मशीन लर्निंग और आर्टिफिशियल इंटेलिजेंस के मूलभूत सिद्धांतों का अध्ययन करेंगे। "
        "मशीन लर्निंग कंप्यूटर विज्ञान की वह शाखा है जो डेटा से सीखने वाले एल्गोरिदम विकसित करती है। "
        "पारंपरिक प्रोग्रामिंग में हम कंप्यूटर को स्पष्ट निर्देश देते हैं। "
        "परंतु मशीन लर्निंग में सिस्टम स्वयं उदाहरणों और पिछले अनुभवों से सीखता है। "
        "इसके तीन मुख्य प्रकार हैं: सुपरवाइज्ड लर्निंग, अनसुपरवाइज्ड लर्निंग, और रीइन्फोर्समेंट लर्निंग। "
        "सुपरवाइज्ड लर्निंग में मॉडल को लेबल किए गए डेटा पर प्रशिक्षित किया जाता है। "
        "उदाहरण के लिए, ईमेल स्पैम का पता लगाना या छवियों में वस्तुओं को पहचानना। "
        "अनसुपरवाइज्ड लर्निंग में डेटा बिना किसी लेबल के होता है, और मॉडल डेटा में छिपे पैटर्न ढूंढता है। "
        "रीइन्फोर्समेंट लर्निंग में एजेंट पुरस्कार और दंड के माध्यम से सर्वोत्तम निर्णय लेना सीखता है। "
        "आज प्राकृतिक भाषा प्रसंस्करण और न्यूरल नेटवर्क ने इस क्षेत्र में क्रांति ला दी है। "
        "आर्टिफिशियल न्यूरल नेटवर्क मानव मस्तिष्क के न्यूरॉन्स से प्रेरित हैं। "
        "भविष्य में एआई स्वास्थ्य सेवा, शिक्षा और परिवहन में महत्वपूर्ण भूमिका निभाएगा। धन्यवाद।"
    )
    wav_path = out_dir / "hindi_2min_speech.wav"
    await generate_speech_audio(long_hindi_text, "hi-IN-SwaraNeural", wav_path)
    video_path = out_dir / "hindi_2min_test.mp4"
    create_test_video(wav_path, video_path, title="2-Minute Hindi Test Video")
    return video_path


async def main():
    print("Generating test clips...")
    await build_hindi_clip()
    await build_french_clip()
    await build_2min_hindi_video()
    print("All test clips generated successfully.")


if __name__ == "__main__":
    asyncio.run(main())
