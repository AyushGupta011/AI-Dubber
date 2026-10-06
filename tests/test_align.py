"""Unit tests for the 4-step audio alignment and time fitting engine."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from dubber.config import DubberConfig
from dubber.models import TranslatedSegment, TTSClip, AlignedSegment
from dubber.stages.align import align_single_segment


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Creates dummy WAV audio files for alignment tests."""
    sr = 24000

    def _create_wav(duration: float, name: str) -> Path:
        samples = int(duration * sr)
        # Create subtle tone
        t = np.linspace(0, duration, samples, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), audio, sr)
        return p

    return _create_wav


def test_align_natural_fit(temp_audio_dir, tmp_path):
    # Original slot: 3.0s (start 0.0 to 3.0). TTS duration: 2.0s. Fits naturally.
    wav_path = temp_audio_dir(2.0, "clip_natural")
    seg = TranslatedSegment(id=0, start=0.0, end=3.0, src_text="test", dst_text="test")
    clip = TTSClip(segment_id=0, audio_path=str(wav_path), duration=2.0, voice_name="en-US-JennyNeural")
    out_wav = tmp_path / "aligned_0.wav"

    config = DubberConfig(url="test")
    aligned = align_single_segment(
        seg=seg,
        clip=clip,
        next_seg_start=4.0,
        total_video_duration=10.0,
        output_wav=out_wav,
        config=config,
    )

    assert aligned.speed_factor == 1.0
    assert aligned.borrowed_gap == 0.0
    assert aligned.fit_imperfect is False
    assert abs(aligned.final_duration - 2.0) < 0.1


def test_align_speedup_step_a(temp_audio_dir, tmp_path):
    # Original slot: 2.0s (start 0.0 to 2.0). TTS duration: 2.3s.
    # Required speedup = 2.3 / 2.0 = 1.15x <= 1.25x max speed factor.
    wav_path = temp_audio_dir(2.3, "clip_speedup")
    seg = TranslatedSegment(id=1, start=0.0, end=2.0, src_text="test", dst_text="test")
    clip = TTSClip(segment_id=1, audio_path=str(wav_path), duration=2.3, voice_name="en-US-JennyNeural")
    out_wav = tmp_path / "aligned_1.wav"

    config = DubberConfig(url="test", max_speed_factor=1.25)
    aligned = align_single_segment(
        seg=seg,
        clip=clip,
        next_seg_start=5.0,
        total_video_duration=10.0,
        output_wav=out_wav,
        config=config,
    )

    assert 1.10 <= aligned.speed_factor <= 1.20
    assert aligned.borrowed_gap == 0.0
    assert aligned.fit_imperfect is False
    assert abs(aligned.final_duration - 2.0) < 0.15


def test_align_gap_borrow_step_b(temp_audio_dir, tmp_path):
    # Original slot: 2.0s. TTS duration: 3.2s.
    # At 1.25x, audio takes 3.2 / 1.25 = 2.56s (overrun = 0.56s).
    # Following gap: 2.0s (next seg at 4.0s). Max borrowable = 2.0 * 0.8 - 0.1 = 1.5s.
    # Overrun (0.56s) <= 1.5s -> Borrow 0.56s from gap.
    wav_path = temp_audio_dir(3.2, "clip_borrow")
    seg = TranslatedSegment(id=2, start=0.0, end=2.0, src_text="test", dst_text="test")
    clip = TTSClip(segment_id=2, audio_path=str(wav_path), duration=3.2, voice_name="en-US-JennyNeural")
    out_wav = tmp_path / "aligned_2.wav"

    config = DubberConfig(url="test", max_speed_factor=1.25, gap_borrow_ratio=0.8, min_gap_remaining=0.1)
    aligned = align_single_segment(
        seg=seg,
        clip=clip,
        next_seg_start=4.0,
        total_video_duration=10.0,
        output_wav=out_wav,
        config=config,
    )

    assert aligned.speed_factor == 1.25
    assert aligned.borrowed_gap > 0.4
    assert aligned.fit_imperfect is False
    assert abs(aligned.final_duration - 2.56) < 0.15


def test_align_hard_cap_imperfect_step_d(temp_audio_dir, tmp_path):
    # Original slot: 1.0s. TTS duration: 4.0s.
    # Next segment starts immediately at 1.1s (following gap = 0.1s, max borrowable = 0s).
    # Cannot fit within 1.25x without gap. Falls back to hard cap / imperfect fit.
    wav_path = temp_audio_dir(4.0, "clip_hard")
    seg = TranslatedSegment(id=3, start=0.0, end=1.0, src_text="test", dst_text="test")
    clip = TTSClip(segment_id=3, audio_path=str(wav_path), duration=4.0, voice_name="en-US-JennyNeural")
    out_wav = tmp_path / "aligned_3.wav"

    config = DubberConfig(url="test", max_speed_factor=1.25, hard_max_speed_factor=1.35)
    aligned = align_single_segment(
        seg=seg,
        clip=clip,
        next_seg_start=1.1,
        total_video_duration=10.0,
        output_wav=out_wav,
        config=config,
    )

    assert aligned.fit_imperfect is True
    assert aligned.speed_factor >= 1.25
