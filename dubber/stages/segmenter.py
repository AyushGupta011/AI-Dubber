"""Stage 4: Sentence-Level Segmentation and Boundary Clustering."""

import json
import re
from pathlib import Path
from typing import List, Optional
from dubber.config import DubberConfig
from dubber.models import TranscriptionResult, RawWhisperSegment, Segment, WordTiming
from dubber.utils.logger import logger

# Sentence end punctuation across English, French, German, and Indic scripts
SENTENCE_END_PUNCTUATION = {".", "!", "?", "।", "؟", "۔", "…"}
CLAUSE_PUNCTUATION = {",", ";", ":", "-", "—"}


def clean_text(text: str) -> str:
    """Cleans and standardizes text spacing and punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_sentence_end(word: str) -> bool:
    """Checks if a word ends with a full sentence terminator."""
    word = word.strip()
    if not word:
        return False
    return any(word.endswith(p) for p in SENTENCE_END_PUNCTUATION)


def build_segments_from_transcript(
    transcript: TranscriptionResult,
    max_duration: float = 12.0,
    min_pause_split: float = 0.4,
    max_words: int = 28,
) -> List[Segment]:
    """
    Pure function that merges raw Whisper word-level fragments into coherent,
    sentence-level segments respecting natural pauses and punctuation.
    """
    all_words: List[WordTiming] = []

    # Flatten all words with validated timestamps
    for raw_seg in transcript.raw_segments:
        if raw_seg.words:
            all_words.extend(raw_seg.words)
        else:
            # Fallback: create synthetic word timing from segment text
            seg_text = clean_text(raw_seg.text)
            if seg_text:
                all_words.append(
                    WordTiming(
                        word=seg_text,
                        start=raw_seg.start,
                        end=raw_seg.end,
                        probability=1.0,
                    )
                )

    if not all_words:
        # Fallback if no words present
        return [
            Segment(
                id=i,
                start=s.start,
                end=s.end,
                text=clean_text(s.text),
                words=s.words,
            )
            for i, s in enumerate(transcript.raw_segments)
            if clean_text(s.text)
        ]

    segments: List[Segment] = []
    curr_words: List[WordTiming] = []
    seg_counter = 0

    for i, w in enumerate(all_words):
        curr_words.append(w)
        seg_duration = curr_words[-1].end - curr_words[0].start
        has_next = i + 1 < len(all_words)

        should_split = False
        pause_gap = 0.0

        if has_next:
            next_w = all_words[i + 1]
            pause_gap = next_w.start - w.end

        # Condition 1: Word ends with sentence terminator
        if is_sentence_end(w.word):
            should_split = True

        # Condition 2: Significant pause between words
        elif pause_gap >= min_pause_split and len(curr_words) >= 3:
            should_split = True

        # Condition 3: Max duration or word limit reached
        elif seg_duration >= max_duration or len(curr_words) >= max_words:
            should_split = True

        # Condition 4: Last word in audio
        elif not has_next:
            should_split = True

        if should_split and curr_words:
            seg_text = clean_text(" ".join(item.word.strip() for item in curr_words))
            if seg_text:
                start_t = curr_words[0].start
                end_t = curr_words[-1].end
                # Ensure start < end
                if end_t <= start_t:
                    end_t = start_t + 0.1

                segments.append(
                    Segment(
                        id=seg_counter,
                        start=round(start_t, 3),
                        end=round(end_t, 3),
                        text=seg_text,
                        words=list(curr_words),
                    )
                )
                seg_counter += 1
            curr_words = []

    return segments


def segment_transcript(config: DubberConfig, transcript: TranscriptionResult) -> List[Segment]:
    """
    Executes sentence-level segmentation stage and saves segments.json artifact.
    """
    segments_path = config.segments_path

    # Resumption check
    if config.resume and segments_path.exists():
        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segments = [Segment.from_dict(s) for s in data]
            if segments:
                logger.info(f"Resume: Loaded {len(segments)} segments from {segments_path}")
                return segments
        except Exception:
            logger.warning("Existing segments file invalid, re-segmenting...")

    logger.info("Structuring transcript into sentence-level units...")
    segments = build_segments_from_transcript(
        transcript=transcript,
        max_duration=config.max_segment_duration,
        min_pause_split=config.min_pause_split,
        max_words=config.max_words_per_segment,
    )

    logger.info(f"Segmentation complete: created {len(segments)} timed sentence segments.")

    # Save artifact
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)

    return segments
