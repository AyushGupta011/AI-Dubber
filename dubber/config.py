"""Configuration dataclasses and validation for the Dubber pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import hashlib
import re


@dataclass
class DubberConfig:
    """Central configuration for a Dubber pipeline execution."""

    # Input / Output
    url: str = ""
    out_dir: Path = field(default_factory=lambda: Path("./output"))
    work_dir: Optional[Path] = None
    output_filename: Optional[str] = None

    # ASR (Whisper) Settings
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8_float16"
    whisper_batch_size: int = 16
    whisper_task: Literal["transcribe", "translate"] = "translate"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    vad_filter: bool = True
    vad_min_silence_ms: int = 500

    # Language & Translation
    src_lang: Optional[str] = None  # Auto-detected if None
    target_lang: str = "en"
    llm_refinement: bool = False
    translation_backend: Literal["auto", "marian", "nllb", "indictrans2"] = "auto"

    # Segmentation
    max_segment_duration: float = 12.0  # Max seconds per merged sentence
    min_pause_split: float = 0.4  # Split segment if pause exceeds this (seconds)
    max_words_per_segment: int = 28

    # Voice & Pitch Settings
    pitch_gender_threshold_hz: float = 165.0  # < 165Hz -> male, >= 165Hz -> female
    match_voice_pitch: bool = True  # Dynamically match TTS pitch to speaker's measured F0
    clone_voice: bool = False  # Enable zero-shot neural voice cloning
    force_gender: Optional[Literal["male", "female"]] = None
    voice_female: str = "en-US-JennyNeural"
    voice_male: str = "en-US-GuyNeural"
    voice_female_high: str = "en-US-AriaNeural"
    voice_male_deep: str = "en-US-ChristopherNeural"
    tts_concurrency: int = 24
    tts_sample_rate: int = 24000

    # Alignment & Fitting
    max_speed_factor: float = 1.25  # Max normal atempo speedup
    gap_borrow_ratio: float = 0.80  # Max percentage of trailing silence to borrow
    min_gap_remaining: float = 0.10  # Seconds of silence gap to always preserve
    hard_max_speed_factor: float = 1.35  # Hard safety cap

    # Stretch features
    keep_background: bool = False
    background_duck_db: float = -12.0
    diarization: bool = True
    diarization_threshold: float = 0.5
    use_xtts_cloning: bool = False

    # Pipeline controls
    resume: bool = True
    cookies_from_browser: Optional[str] = None
    cookies_file: Optional[str] = None
    ffmpeg_binary: Optional[str] = None

    def __post_init__(self):
        """Sanitize and compute dependent paths."""
        self.out_dir = Path(self.out_dir).resolve()
        if self.work_dir is None:
            # Generate deterministic work directory name based on URL/filename
            url_hash = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:10]
            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', self.url[-20:])
            self.work_dir = Path(f"./work_{sanitized}_{url_hash}").resolve()
        else:
            self.work_dir = Path(self.work_dir).resolve()

        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def source_video_path(self) -> Path:
        return self.work_dir / "source_video.mp4"

    @property
    def source_audio_path(self) -> Path:
        return self.work_dir / "source_audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.work_dir / "transcript.json"

    @property
    def segments_path(self) -> Path:
        return self.work_dir / "segments.json"

    @property
    def translations_path(self) -> Path:
        return self.work_dir / "translations.json"

    @property
    def voice_assignments_path(self) -> Path:
        return self.work_dir / "voice_assignments.json"

    @property
    def tts_dir(self) -> Path:
        p = self.work_dir / "tts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def aligned_dir(self) -> Path:
        p = self.work_dir / "aligned"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def aligned_metadata_path(self) -> Path:
        return self.work_dir / "aligned_metadata.json"

    @property
    def dubbed_audio_path(self) -> Path:
        return self.work_dir / "dubbed_audio.wav"

    @property
    def metrics_path(self) -> Path:
        return self.work_dir / "metrics.json"

    @property
    def final_video_path(self) -> Path:
        if self.output_filename:
            return self.out_dir / self.output_filename
        return self.out_dir / "dubbed_output.mp4"
