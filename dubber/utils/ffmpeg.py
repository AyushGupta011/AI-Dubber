"""FFmpeg and FFprobe binary management and audio/video transformation utilities."""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dubber.errors import DependencyMissingError, AudioProcessingError


class FFmpegRunner:
    """Manages discovery and execution of FFmpeg and FFprobe commands."""

    _ffmpeg_path: Optional[str] = None
    _ffprobe_path: Optional[str] = None

    @classmethod
    def get_ffmpeg_path(cls) -> str:
        """Finds a working ffmpeg executable on PATH, virtualenv, or imageio-ffmpeg."""
        if cls._ffmpeg_path and os.path.exists(cls._ffmpeg_path):
            return cls._ffmpeg_path

        # 1. Check system PATH
        path = shutil.which("ffmpeg")
        if path:
            cls._ffmpeg_path = path
            return path

        # 2. Check virtualenv Scripts
        venv_scripts = Path(__file__).resolve().parent.parent.parent / ".venv" / "Scripts" / "ffmpeg.exe"
        if venv_scripts.exists():
            cls._ffmpeg_path = str(venv_scripts)
            return cls._ffmpeg_path

        # 3. Check imageio-ffmpeg package
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                cls._ffmpeg_path = exe
                return exe
        except ImportError:
            pass

        raise DependencyMissingError(
            "FFmpeg executable not found! Please install FFmpeg and add it to your PATH, "
            "or install `imageio-ffmpeg` via pip."
        )

    @classmethod
    def get_ffprobe_path(cls) -> Optional[str]:
        """Finds ffprobe executable on PATH if available."""
        if cls._ffprobe_path and os.path.exists(cls._ffprobe_path):
            return cls._ffprobe_path

        path = shutil.which("ffprobe")
        if path:
            cls._ffprobe_path = path
            return path

        # Check in same directory as ffmpeg
        try:
            ffmpeg_dir = Path(cls.get_ffmpeg_path()).parent
            candidate = ffmpeg_dir / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if candidate.exists():
                cls._ffprobe_path = str(candidate)
                return cls._ffprobe_path
        except Exception:
            pass

        return None

    @classmethod
    def run_command(cls, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Runs an arbitrary ffmpeg/ffprobe command safely."""
        try:
            res = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if check and res.returncode != 0:
                raise AudioProcessingError(
                    f"Command failed with code {res.returncode}:\n{' '.join(args)}\n\nStderr:\n{res.stderr}"
                )
            return res
        except FileNotFoundError as e:
            raise DependencyMissingError(f"Executable not found: {args[0]}") from e


def get_media_duration(file_path: Path) -> float:
    """Gets exact duration in seconds of an audio or video file."""
    ffprobe = FFmpegRunner.get_ffprobe_path()
    if ffprobe:
        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        res = FFmpegRunner.run_command(cmd, check=False)
        if res.returncode == 0 and res.stdout.strip():
            try:
                return float(res.stdout.strip())
            except ValueError:
                pass

    # Fallback to ffmpeg stderr parsing or soundfile
    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        return float(info.duration)
    except Exception:
        pass

    # Fallback to ffmpeg -i info
    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    cmd = [ffmpeg, "-i", str(file_path)]
    res = FFmpegRunner.run_command(cmd, check=False)
    for line in res.stderr.splitlines():
        if "Duration:" in line:
            # Duration: 00:01:23.45, start: ...
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)

    raise AudioProcessingError(f"Could not determine media duration for {file_path}")


def extract_16k_mono_wav(video_path: Path, output_wav_path: Path) -> Path:
    """Extracts 16kHz mono 16-bit PCM WAV for ASR from a video or audio file."""
    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path)
    ]
    FFmpegRunner.run_command(cmd, check=True)
    return output_wav_path


def apply_audio_speed(input_audio: Path, output_audio: Path, speed_factor: float, sample_rate: int = 24000) -> Path:
    """
    Applies pitch-preserving time stretching using ffmpeg atempo filter.
    Handles chaining if speed_factor is outside [0.5, 2.0].
    """
    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    # Build atempo filter chain (atempo accepts 0.5 to 2.0)
    atempo_filters = []
    curr = speed_factor
    while curr > 2.0:
        atempo_filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        atempo_filters.append("atempo=0.5")
        curr /= 0.5
    atempo_filters.append(f"atempo={curr:.4f}")
    filter_str = ",".join(atempo_filters)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_audio),
        "-filter:a", filter_str,
        "-ar", str(sample_rate),
        str(output_audio)
    ]
    FFmpegRunner.run_command(cmd, check=True)
    return output_audio


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path, bitrate: str = "192k") -> Path:
    """
    Multiplexes video and dubbed audio into final MP4 using stream copy for video.
    Visuals are NOT re-encoded (-c:v copy).
    """
    ffmpeg = FFmpegRunner.get_ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", bitrate,
        "-shortest",
        str(output_path)
    ]
    FFmpegRunner.run_command(cmd, check=True)
    return output_path


