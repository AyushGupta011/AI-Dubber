"""Stage 10: Lossless Video/Audio Multiplexing via Stream Copy."""

from pathlib import Path
from dubber.config import DubberConfig
from dubber.utils.ffmpeg import mux_video_audio, get_media_duration
from dubber.utils.logger import logger


def mux_final_video(config: DubberConfig) -> Path:
    """
    Multiplexes source video with dubbed audio stream.
    Visuals are bit-identical (copied directly via -c:v copy).
    Output: config.final_video_path
    """
    video_path = config.source_video_path
    audio_path = config.dubbed_audio_path
    out_path = config.final_video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Source video missing at {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Dubbed audio missing at {audio_path}")

    logger.info(f"Multiplexing video and dubbed audio into {out_path} (Lossless video stream copy)...")
    mux_video_audio(video_path, audio_path, out_path, bitrate="192k")

    out_dur = get_media_duration(out_path)
    logger.info(f"Final dubbed video generated: {out_path} (Duration: {out_dur:.2f}s)")
    return out_path
