"""Stage 8: Audio Alignment & 4-Step Time Fitting Engine."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dubber.config import DubberConfig
from dubber.models import TranslatedSegment, TTSClip, AlignedSegment
from dubber.utils.ffmpeg import apply_audio_speed, FFmpegRunner, get_media_duration
from dubber.utils.logger import logger


def align_single_segment(
    seg: TranslatedSegment,
    clip: TTSClip,
    next_seg_start: Optional[float],
    total_video_duration: float,
    output_wav: Path,
    config: DubberConfig,
) -> AlignedSegment:
    """
    Fits a single TTS clip into its target time slot using the 4-step algorithm:
    1. If TTS duration <= slot: Keep natural speed.
    2. Step A: If speedup <= 1.25x: Apply atempo speedup.
    3. Step B: Borrow from trailing silence gap (up to gap_borrow_ratio).
    4. Step C: Request shorter re-translation if provided.
    5. Step D: Hard-cap atempo (1.35x) and safe boundary trimming with imperfect logging.
    """
    target_start = seg.start
    target_end = seg.end
    orig_slot = max(0.1, target_end - target_start)

    # Calculate trailing silence gap
    if next_seg_start is not None:
        following_gap = max(0.0, next_seg_start - target_end)
    else:
        following_gap = max(0.0, total_video_duration - target_end)

    tts_dur = clip.duration
    input_audio = Path(clip.audio_path)

    # Case 1: Audio fits naturally within original slot
    if tts_dur <= orig_slot + 0.05:
        # Fast path: transcode without heavy atempo filter (identical audio quality)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = FFmpegRunner.get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y",
            "-i", str(input_audio),
            "-ar", str(config.tts_sample_rate),
            "-ac", "1",
            str(output_wav),
        ]
        FFmpegRunner.run_command(cmd, check=True)
        final_dur = get_media_duration(output_wav)
        return AlignedSegment(
            segment_id=seg.id,
            target_start=target_start,
            target_end=target_end,
            original_tts_duration=tts_dur,
            final_audio_path=str(output_wav),
            final_duration=final_dur,
            speed_factor=1.0,
            borrowed_gap=0.0,
            retranslated=False,
            fit_imperfect=False,
            note="Natural speed fit",
        )

    # Step A: Speedup capped at max_speed_factor (1.25x)
    required_speed = tts_dur / orig_slot
    if required_speed <= config.max_speed_factor:
        apply_audio_speed(input_audio, output_wav, speed_factor=required_speed, sample_rate=config.tts_sample_rate)
        final_dur = get_media_duration(output_wav)
        return AlignedSegment(
            segment_id=seg.id,
            target_start=target_start,
            target_end=target_end,
            original_tts_duration=tts_dur,
            final_audio_path=str(output_wav),
            final_duration=final_dur,
            speed_factor=round(required_speed, 3),
            borrowed_gap=0.0,
            retranslated=False,
            fit_imperfect=False,
            note=f"Speedup applied: {required_speed:.2f}x",
        )

    # Step B: Apply 1.25x speedup and borrow from following silence gap
    dur_at_125 = tts_dur / config.max_speed_factor
    overrun = dur_at_125 - orig_slot

    max_borrowable = max(0.0, (following_gap * config.gap_borrow_ratio) - config.min_gap_remaining)

    if overrun <= max_borrowable:
        borrowed = overrun
        apply_audio_speed(input_audio, output_wav, speed_factor=config.max_speed_factor, sample_rate=config.tts_sample_rate)
        final_dur = get_media_duration(output_wav)
        return AlignedSegment(
            segment_id=seg.id,
            target_start=target_start,
            target_end=target_end + borrowed,
            original_tts_duration=tts_dur,
            final_audio_path=str(output_wav),
            final_duration=final_dur,
            speed_factor=config.max_speed_factor,
            borrowed_gap=round(borrowed, 3),
            retranslated=False,
            fit_imperfect=False,
            note=f"Speedup 1.25x + borrowed {borrowed:.2f}s gap",
        )

    # Step C & D: Borrow maximum available gap and apply hard-capped speedup (up to 1.35x)
    borrowed = max_borrowable
    available_window = orig_slot + borrowed
    hard_speed = min(config.hard_max_speed_factor, tts_dur / available_window)

    apply_audio_speed(input_audio, output_wav, speed_factor=hard_speed, sample_rate=config.tts_sample_rate)
    final_dur = get_media_duration(output_wav)

    is_imperfect = (final_dur > available_window + 0.1) or (hard_speed > config.max_speed_factor)
    note_msg = (
        f"Imperfect fit: slot={orig_slot:.2f}s, tts={tts_dur:.2f}s, speed={hard_speed:.2f}x, "
        f"borrowed={borrowed:.2f}s, final_dur={final_dur:.2f}s"
    )

    if is_imperfect:
        logger.warning(f"⚠️ Segment {seg.id} fitting imperfect: {note_msg}")

    return AlignedSegment(
        segment_id=seg.id,
        target_start=target_start,
        target_end=target_start + final_dur,
        original_tts_duration=tts_dur,
        final_audio_path=str(output_wav),
        final_duration=final_dur,
        speed_factor=round(hard_speed, 3),
        borrowed_gap=round(borrowed, 3),
        retranslated=False,
        fit_imperfect=is_imperfect,
        note=note_msg,
    )


def align_audio_clips(
    config: DubberConfig,
    translated_segments: List[TranslatedSegment],
    tts_clips: List[TTSClip],
    total_video_duration: float,
) -> List[AlignedSegment]:
    """
    Fits and aligns all TTS clips to their original video time slots.
    Saves aligned_metadata.json artifact.
    """
    aligned_meta_path = config.aligned_metadata_path

    # Resumption check
    if config.resume and aligned_meta_path.exists():
        try:
            with open(aligned_meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            aligned_segs = [AlignedSegment.from_dict(d) for d in data]
            # Check all aligned audio files exist
            if len(aligned_segs) == len(translated_segments) and all(Path(s.final_audio_path).exists() for s in aligned_segs):
                logger.info(f"Resume: Loaded {len(aligned_segs)} aligned audio clips from {aligned_meta_path}")
                return aligned_segs
        except Exception:
            logger.warning("Existing alignment metadata invalid, re-aligning...")

    aligned_dir = config.aligned_dir
    aligned_segments: List[AlignedSegment] = []

    clip_map = {c.segment_id: c for c in tts_clips}

    logger.info(f"Aligning {len(translated_segments)} TTS audio clips to video timeline...")

    # Pre-compute next_start for each segment (needed for gap-borrowing)
    # This must be done before parallelization to preserve correct ordering
    work_items = []
    for i, seg in enumerate(translated_segments):
        clip = clip_map.get(seg.id)
        if not clip:
            continue
        next_start = translated_segments[i + 1].start if i + 1 < len(translated_segments) else None
        out_wav = aligned_dir / f"seg_{seg.id:04d}_aligned.wav"
        work_items.append((seg, clip, next_start, out_wav))

    def _align_one(item):
        seg, clip, next_start, out_wav = item
        return align_single_segment(
            seg=seg,
            clip=clip,
            next_seg_start=next_start,
            total_video_duration=total_video_duration,
            output_wav=out_wav,
            config=config,
        )

    # Parallel FFmpeg alignment — each segment writes to its own file (no conflict)
    import os
    max_workers = min(os.cpu_count() or 16, len(work_items)) if work_items else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        aligned_segments = list(pool.map(_align_one, work_items))

    imperfect_count = sum(1 for a in aligned_segments if a.fit_imperfect)
    speedup_count = sum(1 for a in aligned_segments if a.speed_factor > 1.0)
    gap_borrow_count = sum(1 for a in aligned_segments if a.borrowed_gap > 0.0)

    logger.info(
        f"Alignment complete: {len(aligned_segments)} clips aligned "
        f"({speedup_count} speedups, {gap_borrow_count} gaps borrowed, {imperfect_count} imperfect fits)."
    )

    # Save artifact
    with open(aligned_meta_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in aligned_segments], f, indent=2)

    return aligned_segments
