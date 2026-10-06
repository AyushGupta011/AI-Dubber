"""Data models and serialization structures for the Dubber pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTiming":
        return cls(**data)


@dataclass
class RawWhisperSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    words: List[WordTiming]
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["words"] = [w.to_dict() if isinstance(w, WordTiming) else w for w in self.words]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawWhisperSegment":
        words = [WordTiming.from_dict(w) if isinstance(w, dict) else w for w in data.get("words", [])]
        clean_data = dict(data)
        clean_data["words"] = words
        return cls(**clean_data)


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    raw_segments: List[RawWhisperSegment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "raw_segments": [s.to_dict() for s in self.raw_segments],
        }

    def save_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: Path) -> "TranscriptionResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = [RawWhisperSegment.from_dict(s) for s in data.get("raw_segments", [])]
        return cls(
            language=data["language"],
            language_probability=data.get("language_probability", 1.0),
            duration=data.get("duration", 0.0),
            raw_segments=segments,
        )


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)
    speaker_id: Optional[str] = None
    estimated_gender: str = "female"  # "male" | "female"
    f0_median_hz: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["words"] = [w.to_dict() if isinstance(w, WordTiming) else w for w in self.words]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        words = [WordTiming.from_dict(w) if isinstance(w, dict) else w for w in data.get("words", [])]
        clean_data = dict(data)
        clean_data["words"] = words
        return cls(**clean_data)


@dataclass
class TranslatedSegment:
    id: int
    start: float
    end: float
    src_text: str
    dst_text: str
    speaker_id: Optional[str] = None
    estimated_gender: str = "female"
    shortened_dst_text: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslatedSegment":
        return cls(**data)


@dataclass
class VoiceAssignment:
    segment_id: int
    speaker_id: str
    gender: str
    voice_name: str
    f0_median_hz: Optional[float] = None
    pitch_shift_str: Optional[str] = None  # e.g., "+10Hz", "-15Hz"
    ref_audio_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceAssignment":
        return cls(**data)


@dataclass
class TTSClip:
    segment_id: int
    audio_path: str
    duration: float
    voice_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSClip":
        return cls(**data)


@dataclass
class AlignedSegment:
    segment_id: int
    target_start: float
    target_end: float
    original_tts_duration: float
    final_audio_path: str
    final_duration: float
    speed_factor: float
    borrowed_gap: float
    retranslated: bool
    fit_imperfect: bool
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlignedSegment":
        return cls(**data)


@dataclass
class StageMetric:
    stage_name: str
    duration_seconds: float
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageMetric":
        return cls(**data)


@dataclass
class DubberMetrics:
    source_url: str
    detected_language: str
    total_duration_seconds: float
    stage_metrics: List[StageMetric]
    total_segments: int
    segments_speedup: int
    segments_gap_borrowed: int
    segments_retranslated: int
    segments_imperfect: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "detected_language": self.detected_language,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_metrics": [m.to_dict() for m in self.stage_metrics],
            "total_segments": self.total_segments,
            "segments_speedup": self.segments_speedup,
            "segments_gap_borrowed": self.segments_gap_borrowed,
            "segments_retranslated": self.segments_retranslated,
            "segments_imperfect": self.segments_imperfect,
        }

    def save_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
