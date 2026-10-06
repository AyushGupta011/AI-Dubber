"""Stage 2: Audio Extraction for ASR (16kHz Mono 16-bit WAV)."""

from pathlib import Path
from dubber.config import DubberConfig
from dubber.utils.ffmpeg import extract_16k_mono_wav, get_media_duration
from dubber.utils.logger import logger


def extract_audio(config: DubberConfig) -> Path:
    """
    Extracts 16kHz mono 16-bit PCM WAV from source video for ASR.
    Output: config.source_audio_path
    """
    target_path = config.source_audio_path
    video_path = config.source_video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found at {video_path}")

    # Resumption check
    if config.resume and target_path.exists() and target_path.stat().st_size > 1024:
        try:
            dur = get_media_duration(target_path)
            if dur > 0:
                logger.info(f"Resume: Using existing extracted audio at {target_path} (Duration: {dur:.2f}s)")
                return target_path
        except Exception:
            logger.warning("Existing extracted audio invalid, re-extracting...")

    logger.info(f"Extracting 16kHz mono audio to {target_path}...")
    extract_16k_mono_wav(video_path, target_path)

    dur = get_media_duration(target_path)
    logger.info(f"Extracted audio ready. Duration: {dur:.2f}s")
    return target_path
