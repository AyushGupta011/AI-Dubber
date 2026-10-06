"""Unit tests for sentence segmentation and boundary clustering."""

import pytest
from dubber.models import TranscriptionResult, RawWhisperSegment, WordTiming
from dubber.stages.segmenter import build_segments_from_transcript, is_sentence_end, clean_text


def test_is_sentence_end():
    assert is_sentence_end("hello.") is True
    assert is_sentence_end("world?") is True
    assert is_sentence_end("yes!") is True
    assert is_sentence_end("नमस्ते।") is True
    assert is_sentence_end("running") is False
    assert is_sentence_end("comma,") is False


def test_clean_text():
    assert clean_text("  hello   world  \n") == "hello world"
    assert clean_text("") == ""


def test_segment_punctuation_split():
    words = [
        WordTiming(word="Hello", start=0.0, end=0.4),
        WordTiming(word="world.", start=0.4, end=0.8),
        WordTiming(word="How", start=0.9, end=1.2),
        WordTiming(word="are", start=1.2, end=1.5),
        WordTiming(word="you?", start=1.5, end=2.0),
    ]
    raw_seg = RawWhisperSegment(
        id=0,
        seek=0,
        start=0.0,
        end=2.0,
        text="Hello world. How are you?",
        tokens=[],
        words=words,
    )
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration=2.0,
        raw_segments=[raw_seg],
    )

    segments = build_segments_from_transcript(transcript, max_duration=10.0, min_pause_split=0.4)

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[0].start == 0.0
    assert segments[0].end == 0.8
    assert segments[1].text == "How are you?"
    assert segments[1].start == 0.9
    assert segments[1].end == 2.0


def test_segment_hindi_danda_split():
    words = [
        WordTiming(word="यह", start=0.0, end=0.3),
        WordTiming(word="एक", start=0.3, end=0.6),
        WordTiming(word="परीक्षण", start=0.6, end=1.1),
        WordTiming(word="है।", start=1.1, end=1.5),
        WordTiming(word="अगला", start=1.6, end=2.0),
        WordTiming(word="वाक्य।", start=2.0, end=2.5),
    ]
    raw_seg = RawWhisperSegment(
        id=0,
        seek=0,
        start=0.0,
        end=2.5,
        text="यह एक परीक्षण है। अगला वाक्य।",
        tokens=[],
        words=words,
    )
    transcript = TranscriptionResult(
        language="hi",
        language_probability=0.98,
        duration=2.5,
        raw_segments=[raw_seg],
    )

    segments = build_segments_from_transcript(transcript)

    assert len(segments) == 2
    assert segments[0].text == "यह एक परीक्षण है।"
    assert segments[0].start == 0.0
    assert segments[0].end == 1.5
    assert segments[1].text == "अगला वाक्य।"
    assert segments[1].start == 1.6
    assert segments[1].end == 2.5


def test_segment_pause_split():
    # Words without terminal punctuation, but with a 0.6s pause between them
    words = [
        WordTiming(word="First", start=0.0, end=0.5),
        WordTiming(word="phrase", start=0.5, end=1.0),
        WordTiming(word="here", start=1.0, end=1.5),
        # 0.6s pause here (1.5s to 2.1s)
        WordTiming(word="Second", start=2.1, end=2.5),
        WordTiming(word="phrase", start=2.5, end=3.0),
        WordTiming(word="there", start=3.0, end=3.5),
    ]
    raw_seg = RawWhisperSegment(
        id=0,
        seek=0,
        start=0.0,
        end=3.5,
        text="First phrase here Second phrase there",
        tokens=[],
        words=words,
    )
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration=3.5,
        raw_segments=[raw_seg],
    )

    segments = build_segments_from_transcript(transcript, min_pause_split=0.4)

    assert len(segments) == 2
    assert segments[0].text == "First phrase here"
    assert segments[0].end == 1.5
    assert segments[1].text == "Second phrase there"
    assert segments[1].start == 2.1


def test_segment_max_duration_cap():
    # Long continuous run of words without punctuation or pause
    words = [
        WordTiming(word=f"word{i}", start=i * 1.0, end=(i + 1) * 1.0)
        for i in range(15)
    ]
    raw_seg = RawWhisperSegment(
        id=0,
        seek=0,
        start=0.0,
        end=15.0,
        text=" ".join(f"word{i}" for i in range(15)),
        tokens=[],
        words=words,
    )
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration=15.0,
        raw_segments=[raw_seg],
    )

    # Max duration 5.0 seconds
    segments = build_segments_from_transcript(transcript, max_duration=5.0)

    assert len(segments) >= 3
    for s in segments:
        assert (s.end - s.start) <= 6.0
