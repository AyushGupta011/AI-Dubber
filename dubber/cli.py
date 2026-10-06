"""CLI runner and pipeline orchestrator for the Dubber tool."""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Ensure utf-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dubber import __version__
from dubber.config import DubberConfig
from dubber.errors import DubberError, DependencyMissingError
from dubber.utils.gpu import print_gpu_status, cleanup_gpu_memory
from dubber.utils.logger import logger, print_banner, stage_timer
from dubber.utils.metrics import MetricsTracker
from dubber.utils.ffmpeg import FFmpegRunner, get_media_duration

# Import all stages
from dubber.stages.download import download_video
from dubber.stages.audio import extract_audio
from dubber.stages.transcribe import transcribe_audio
from dubber.stages.segmenter import segment_transcript
from dubber.stages.translate import translate_segments
from dubber.stages.voices import assign_voices
from dubber.stages.tts import generate_tts
from dubber.stages.align import align_audio_clips
from dubber.stages.assemble import assemble_timeline_audio
from dubber.stages.mux import mux_final_video


def run_pipeline(config: DubberConfig) -> Path:
    """Executes the complete end-to-end 10-stage dubbing pipeline."""
    total_stages = 10
    metrics = MetricsTracker(source_url=config.url)

    print("\n" + "=" * 60)
    print(f"DUBBER v{__version__} - AI Video Dubbing Pipeline")
    print("=" * 60)
    print(f"Target: {config.url}")
    print(f"Work Directory: {config.work_dir}")
    print(f"Output Directory: {config.out_dir}")
    print_gpu_status(stage_name="Startup")
    print("=" * 60 + "\n")

    # Verify ffmpeg availability early
    try:
        ffmpeg_bin = FFmpegRunner.get_ffmpeg_path()
        logger.info(f"Using FFmpeg binary: {ffmpeg_bin}")
    except DependencyMissingError as e:
        logger.error(str(e))
        sys.exit(1)

    # Stage 1: Download
    print_banner(1, total_stages, "Video Ingestion & Download")
    with stage_timer("Download") as t:
        video_path = download_video(config)
    metrics.record_stage("1_download", t["elapsed"], details={"video_path": str(video_path)})

    total_duration = get_media_duration(video_path)
    logger.info(f"Source video length: {total_duration:.2f}s")

    # Stage 2: Audio Extraction
    print_banner(2, total_stages, "16kHz Mono Audio Extraction")
    with stage_timer("Audio Extraction") as t:
        audio_path = extract_audio(config)
    metrics.record_stage("2_audio_extraction", t["elapsed"], details={"audio_path": str(audio_path)})

    # Stage 3: Transcription
    print_banner(3, total_stages, "Speech Recognition & Word Timestamps (faster-whisper)")
    with stage_timer("Transcription") as t:
        transcript = transcribe_audio(config)
    metrics.detected_language = transcript.language
    metrics.record_stage(
        "3_transcription",
        t["elapsed"],
        details={
            "detected_language": transcript.language,
            "language_prob": transcript.language_probability,
            "raw_segment_count": len(transcript.raw_segments),
        },
    )

    # Stage 4: Segmentation
    print_banner(4, total_stages, "Sentence Boundary Merging & Pause Clustering")
    with stage_timer("Segmentation") as t:
        segments = segment_transcript(config, transcript)
    metrics.total_segments = len(segments)
    metrics.record_stage("4_segmentation", t["elapsed"], details={"segment_count": len(segments)})

    if not segments:
        logger.warning("No speech segments detected in audio! Exiting early.")
        sys.exit(0)

    # Stages 5+6: Translation & Voice Assignment (run in parallel — they are independent)
    print_banner(5, total_stages, "Contextual Neural Machine Translation")
    print_banner(6, total_stages, "Speaker Pitch Estimation (F0 pyin) & Voice Selection")

    from concurrent.futures import ThreadPoolExecutor

    stage_timings = {}

    def _run_translate():
        t_start = time.time()
        result = translate_segments(config, segments, detected_lang=transcript.language)
        stage_timings["translate"] = time.time() - t_start
        return result

    def _run_voice_assignment():
        t_start = time.time()
        result = assign_voices(config, segments)
        stage_timings["voice"] = time.time() - t_start
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_translate = pool.submit(_run_translate)
        future_voices = pool.submit(_run_voice_assignment)
        translated_segments = future_translate.result()
        voice_assignments = future_voices.result()

    metrics.record_stage("5_translation", stage_timings.get("translate", 0), details={"translated_count": len(translated_segments)})
    metrics.record_stage("6_voice_assignment", stage_timings.get("voice", 0))

    # Stage 7: Speech Synthesis
    print_banner(7, total_stages, "Async Neural Speech Synthesis (edge-tts)")
    with stage_timer("TTS Synthesis") as t:
        tts_clips = generate_tts(config, translated_segments, voice_assignments)
    metrics.record_stage("7_tts_synthesis", t["elapsed"], details={"clip_count": len(tts_clips)})

    # Stage 8: Time Alignment & Fitting
    print_banner(8, total_stages, "4-Step Time Alignment & Slot Fitting")
    with stage_timer("Alignment") as t:
        aligned_segments = align_audio_clips(
            config=config,
            translated_segments=translated_segments,
            tts_clips=tts_clips,
            total_video_duration=total_duration,
        )
    metrics.segments_speedup = sum(1 for a in aligned_segments if a.speed_factor > 1.0)
    metrics.segments_gap_borrowed = sum(1 for a in aligned_segments if a.borrowed_gap > 0.0)
    metrics.segments_retranslated = sum(1 for a in aligned_segments if a.retranslated)
    metrics.segments_imperfect = sum(1 for a in aligned_segments if a.fit_imperfect)
    metrics.record_stage(
        "8_alignment",
        t["elapsed"],
        details={
            "speedup_count": metrics.segments_speedup,
            "gap_borrow_count": metrics.segments_gap_borrowed,
            "imperfect_count": metrics.segments_imperfect,
        },
    )

    # Stage 9: Timeline Assembly
    print_banner(9, total_stages, "Scalable Timeline Audio Assembly")
    with stage_timer("Timeline Assembly") as t:
        dubbed_audio = assemble_timeline_audio(config, aligned_segments, total_duration)

        # Stretch: Demucs background preservation if requested
        if config.keep_background:
            from dubber.stages.demucs_bg import separate_and_duck_background
            mixed_audio = config.work_dir / "dubbed_with_background.wav"
            dubbed_audio = separate_and_duck_background(
                config=config,
                source_audio=config.source_audio_path,
                dubbed_speech=dubbed_audio,
                output_audio=mixed_audio,
            )

    metrics.record_stage("9_assembly", t["elapsed"], details={"dubbed_audio": str(dubbed_audio)})

    # Stage 10: Lossless Stream-Copy Muxing
    print_banner(10, total_stages, "Lossless Video/Audio Multiplexing (-c:v copy)")
    with stage_timer("Muxing") as t:
        final_video = mux_final_video(config)
    metrics.record_stage("10_muxing", t["elapsed"], details={"output_video": str(final_video)})

    # Save metrics.json
    metrics_obj = metrics.save(config.metrics_path)

    # Final summary report
    print("\n" + "=" * 60)
    print("DUBBING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Output Video: {final_video}")
    print(f"Metrics Saved: {config.metrics_path}")
    print(f"Total Elapsed Time: {metrics_obj.total_duration_seconds:.2f}s")
    print(f"Detected Language: {metrics_obj.detected_language}")
    print(f"Total Segments: {metrics_obj.total_segments}")
    print(f"Segments Sped Up: {metrics_obj.segments_speedup}")
    print(f"Gaps Borrowed: {metrics_obj.segments_gap_borrowed}")
    print(f"Imperfect Fits: {metrics_obj.segments_imperfect}")
    print("=" * 60 + "\n")

    return final_video


def parse_args(args: Optional[list] = None) -> DubberConfig:
    """Parses command line arguments into a strongly-typed DubberConfig."""
    parser = argparse.ArgumentParser(
        prog="dubber",
        description="Dub a YouTube video (Hindi, French, German) into natural English speech with preserved video stream.",
    )
    parser.add_argument("url", type=str, help="YouTube video URL or path to local video file")
    parser.add_argument("--out", dest="out_dir", type=str, default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--work-dir", dest="work_dir", type=str, default=None, help="Working directory for intermediate files")
    parser.add_argument("--whisper-model", type=str, default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--whisper-task", choices=["transcribe", "translate"], default="translate", help="Whisper task (default: translate)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (default: auto)")
    parser.add_argument("--src-lang", type=str, default=None, help="Source language code (e.g., hi, de, fr; default: auto-detect)")
    parser.add_argument("--keep-background", action="store_true", help="Preserve background audio with Demucs separation & ducking")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable caching and force re-running all stages")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Browser to extract cookies from (chrome, brave, edge, firefox)")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to cookies.txt file")
    parser.add_argument("--voice-female", type=str, default="en-US-JennyNeural", help="Edge-TTS female voice")
    parser.add_argument("--voice-male", type=str, default="en-US-GuyNeural", help="Edge-TTS male voice")
    parser.add_argument("--force-gender", choices=["male", "female"], default=None, help="Force all voices to a specific gender")
    parser.add_argument("--clone-voice", action="store_true", help="Enable zero-shot speaker voice cloning")
    parser.add_argument("--no-pitch-match", dest="match_voice_pitch", action="store_false", help="Disable dynamic F0 pitch matching")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    parsed = parser.parse_args(args)

    return DubberConfig(
        url=parsed.url,
        out_dir=Path(parsed.out_dir),
        work_dir=Path(parsed.work_dir) if parsed.work_dir else None,
        whisper_model=parsed.whisper_model,
        whisper_task=parsed.whisper_task,
        device=parsed.device,
        src_lang=parsed.src_lang,
        keep_background=parsed.keep_background,
        resume=parsed.resume,
        cookies_from_browser=parsed.cookies_from_browser,
        cookies_file=parsed.cookies_file,
        voice_female=parsed.voice_female,
        voice_male=parsed.voice_male,
        force_gender=parsed.force_gender,
        clone_voice=parsed.clone_voice,
        match_voice_pitch=parsed.match_voice_pitch,
    )


def main():
    """Main CLI entrypoint."""
    try:
        config = parse_args()
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        cleanup_gpu_memory()
        sys.exit(130)
    except DubberError as e:
        logger.error(f"Pipeline error: {e}")
        cleanup_gpu_memory()
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        cleanup_gpu_memory()
        sys.exit(1)


if __name__ == "__main__":
    main()
