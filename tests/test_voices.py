"""Unit tests for speaker pitch estimation and dynamic voice matching."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from dubber.config import DubberConfig
from dubber.models import Segment
from dubber.stages.voices import PitchVoiceAssigner, estimate_segment_f0, select_matched_voice


def test_select_matched_voice():
    config = DubberConfig(
        url="test",
        voice_female="en-US-JennyNeural",
        voice_male="en-US-GuyNeural",
        voice_female_high="en-US-AriaNeural",
        voice_male_deep="en-US-ChristopherNeural",
        match_voice_pitch=True,
    )

    # Deep Male (e.g., 90 Hz)
    gender, voice, pitch_shift = select_matched_voice(90.0, config)
    assert gender == "male"
    assert voice == "en-US-ChristopherNeural"
    assert pitch_shift == "-15Hz"

    # Standard Male (e.g., 135 Hz)
    gender, voice, pitch_shift = select_matched_voice(135.0, config)
    assert gender == "male"
    assert voice == "en-US-GuyNeural"
    assert pitch_shift == "+7Hz"

    # Warm Female (e.g., 190 Hz)
    gender, voice, pitch_shift = select_matched_voice(190.0, config)
    assert gender == "female"
    assert voice == "en-US-JennyNeural"
    assert pitch_shift == "-5Hz"

    # Bright Female (e.g., 235 Hz)
    gender, voice, pitch_shift = select_matched_voice(235.0, config)
    assert gender == "female"
    assert voice == "en-US-AriaNeural"
    assert pitch_shift == "+10Hz"


def test_pitch_voice_assignment(tmp_path):
    sr = 16000
    duration = 2.0

    # 1. Create a 130 Hz tone (standard male pitch)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    male_wave = 0.5 * np.sin(2 * np.pi * 130 * t) + 0.25 * np.sin(2 * np.pi * 260 * t)
    male_wav_path = tmp_path / "male.wav"
    sf.write(str(male_wav_path), male_wave.astype(np.float32), sr)

    # 2. Create a 190 Hz tone (standard female pitch)
    female_wave = 0.5 * np.sin(2 * np.pi * 190 * t) + 0.25 * np.sin(2 * np.pi * 380 * t)
    female_wav_path = tmp_path / "female.wav"
    sf.write(str(female_wav_path), female_wave.astype(np.float32), sr)

    config = DubberConfig(
        url="test",
        voice_female="en-US-JennyNeural",
        voice_male="en-US-GuyNeural",
        voice_female_high="en-US-AriaNeural",
        voice_male_deep="en-US-ChristopherNeural",
        match_voice_pitch=True,
    )

    assigner = PitchVoiceAssigner()

    # Test male segment
    seg_male = [Segment(id=0, start=0.0, end=2.0, text="male sample")]
    assignments_male = assigner.assign_voices(config, seg_male, male_wav_path)
    assert assignments_male[0].gender == "male"
    assert assignments_male[0].voice_name == "en-US-GuyNeural"

    # Test female segment
    seg_female = [Segment(id=1, start=0.0, end=2.0, text="female sample")]
    assignments_female = assigner.assign_voices(config, seg_female, female_wav_path)
    assert assignments_female[0].gender == "female"
    assert assignments_female[0].voice_name == "en-US-JennyNeural"
