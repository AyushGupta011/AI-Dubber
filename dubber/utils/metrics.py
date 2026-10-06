"""Metrics tracker for recording stage durations and alignment statistics."""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dubber.models import StageMetric, DubberMetrics


class MetricsTracker:
    """Collects pipeline metrics and exports to JSON."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self.detected_language: str = "unknown"
        self.start_time: float = time.time()
        self.stage_metrics: List[StageMetric] = []
        self.total_segments: int = 0
        self.segments_speedup: int = 0
        self.segments_gap_borrowed: int = 0
        self.segments_retranslated: int = 0
        self.segments_imperfect: int = 0

    def record_stage(self, stage_name: str, duration_seconds: float, status: str = "success", details: Optional[Dict[str, Any]] = None) -> None:
        """Appends a completed stage metric."""
        metric = StageMetric(
            stage_name=stage_name,
            duration_seconds=duration_seconds,
            status=status,
            details=details or {}
        )
        self.stage_metrics.append(metric)

    def to_metrics(self) -> DubberMetrics:
        """Constructs final DubberMetrics object."""
        total_duration = time.time() - self.start_time
        return DubberMetrics(
            source_url=self.source_url,
            detected_language=self.detected_language,
            total_duration_seconds=total_duration,
            stage_metrics=self.stage_metrics,
            total_segments=self.total_segments,
            segments_speedup=self.segments_speedup,
            segments_gap_borrowed=self.segments_gap_borrowed,
            segments_retranslated=self.segments_retranslated,
            segments_imperfect=self.segments_imperfect,
        )

    def save(self, output_path: Path) -> DubberMetrics:
        """Saves metrics to JSON file."""
        metrics = self.to_metrics()
        metrics.save_json(output_path)
        return metrics
