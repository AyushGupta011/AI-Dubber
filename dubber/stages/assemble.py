"""Stage 9: Scalable Low-Memory Timeline Audio Assembly."""

from pathlib import Path
from typing import List
import numpy as np
import soundfile as sf
from dubber.config import DubberConfig
from dubber.models import AlignedSegment
from dubber.utils.ffmpeg import get_media_duration
from dubber.utils.logger import logger


def assemble_timeline_audio(
    config: DubberConfig,
    aligned_segments: List[AlignedSegment],
    total_video_duration: float,
) -> Path:
    """
    Streams aligned audio clips into a unified dubbed.wav file on a silent timeline.
    Uses sequential streaming to ensure O(1) constant RAM footprint for 2+ hour videos.
    Output: config.dubbed_audio_path
    """
    dubbed_path = config.dubbed_audio_path
    sr = config.tts_sample_rate

    # Resumption check
    if config.resume and dubbed_path.exists():
        try:
            dur = get_media_duration(dubbed_path)
            if abs(dur - total_video_duration) <= 1.0 or dur > 0.5:
                logger.info(f"Resume: Using existing assembled audio at {dubbed_path} (Duration: {dur:.2f}s)")
                return dubbed_path
        except Exception:
            logger.warning("Existing dubbed audio invalid, re-assembling...")

    logger.info(f"Assembling dubbed audio timeline (Duration: {total_video_duration:.2f}s, Sample Rate: {sr}Hz)...")

    # Sort segments by start time
    sorted_segs = sorted(aligned_segments, key=lambda s: s.target_start)
    total_samples = int(np.ceil(total_video_duration * sr))

    current_sample = 0

    with sf.SoundFile(str(dubbed_path), mode="w", samplerate=sr, channels=1, subtype="PCM_16") as outfile:
        for seg in sorted_segs:
            target_start_sample = int(round(seg.target_start * sr))

            # 1. Write silence before segment if gap exists
            if target_start_sample > current_sample:
                gap_samples = target_start_sample - current_sample
                # Stream silence in blocks to avoid huge array allocations
                chunk_size = 48000
                zeros = np.zeros(min(gap_samples, chunk_size), dtype=np.int16)
                while gap_samples > 0:
                    write_len = min(gap_samples, chunk_size)
                    outfile.write(zeros[:write_len])
                    gap_samples -= write_len
                current_sample = target_start_sample

            # 2. Read and write segment audio
            clip_path = Path(seg.final_audio_path)
            if clip_path.exists():
                data, clip_sr = sf.read(str(clip_path), dtype="int16")
                if data.ndim > 1:
                    data = data[:, 0]  # Mono

                # Write audio chunk
                outfile.write(data)
                current_sample += len(data)

        # 3. Write trailing silence up to total_video_duration
        if current_sample < total_samples:
            trailing_samples = total_samples - current_sample
            chunk_size = 48000
            zeros = np.zeros(min(trailing_samples, chunk_size), dtype=np.int16)
            while trailing_samples > 0:
                write_len = min(trailing_samples, chunk_size)
                outfile.write(zeros[:write_len])
                trailing_samples -= write_len

    final_dur = get_media_duration(dubbed_path)
    logger.info(f"Timeline audio assembled successfully: {dubbed_path} (Duration: {final_dur:.2f}s)")
    return dubbed_path
