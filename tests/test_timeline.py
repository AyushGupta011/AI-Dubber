"""Unit tests for scalable timeline audio assembly."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from dubber.config import DubberConfig
from dubber.models import AlignedSegment
from dubber.stages.assemble import assemble_timeline_audio


def test_timeline_assembly_duration_and_alignment(tmp_path):
    sr = 24000
    config = DubberConfig(url="test", work_dir=tmp_path, tts_sample_rate=sr)

    # Create two 1-second audio clips
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone1 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone2 = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    clip1_path = tmp_path / "clip1.wav"
    clip2_path = tmp_path / "clip2.wav"
    sf.write(str(clip1_path), tone1, sr)
    sf.write(str(clip2_path), tone2, sr)

    # Segment 1: at 1.0s -> 2.0s
    # Segment 2: at 4.0s -> 5.0s
    # Total video length: 7.0s
    aligned_segs = [
        AlignedSegment(
            segment_id=0,
            target_start=1.0,
            target_end=2.0,
            original_tts_duration=1.0,
            final_audio_path=str(clip1_path),
            final_duration=1.0,
            speed_factor=1.0,
            borrowed_gap=0.0,
            retranslated=False,
            fit_imperfect=False,
            note="",
        ),
        AlignedSegment(
            segment_id=1,
            target_start=4.0,
            target_end=5.0,
            original_tts_duration=1.0,
            final_audio_path=str(clip2_path),
            final_duration=1.0,
            speed_factor=1.0,
            borrowed_gap=0.0,
            retranslated=False,
            fit_imperfect=False,
            note="",
        ),
    ]

    out_audio = assemble_timeline_audio(config, aligned_segs, total_video_duration=7.0)

    # Verify generated output audio
    assert out_audio.exists()
    data, file_sr = sf.read(str(out_audio))
    assert file_sr == sr

    total_expected_samples = int(np.ceil(7.0 * sr))
    assert abs(len(data) - total_expected_samples) <= sr * 0.05

    # Check that initial 0.0-0.9s is silence
    silence_check = data[: int(0.9 * sr)]
    assert np.all(silence_check == 0)

    # Check that audio exists between 1.0s and 2.0s
    audio_check = data[int(1.1 * sr) : int(1.9 * sr)]
    assert np.max(np.abs(audio_check)) > 0.1

    # Check that 2.5-3.5s is silence
    silence_check2 = data[int(2.5 * sr) : int(3.5 * sr)]
    assert np.all(silence_check2 == 0)
