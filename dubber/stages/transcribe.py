"""Stage 3: ASR Transcription with faster-whisper and OOM Fallback Chain."""

from pathlib import Path
from typing import Optional, List, Tuple
from dubber.config import DubberConfig
from dubber.models import TranscriptionResult, RawWhisperSegment, WordTiming
from dubber.errors import TranscriptionError, UnsupportedLanguageError
from dubber.utils.gpu import print_gpu_status, cleanup_gpu_memory, get_gpu_info
from dubber.utils.ffmpeg import get_media_duration
from dubber.utils.logger import logger


def _run_whisper_inference(
    audio_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    vad_filter: bool,
    vad_min_silence_ms: int,
    language: Optional[str] = None,
    task: str = "translate",
    batch_size: int = 16,
) -> TranscriptionResult:
    """Runs faster-whisper inference for a specific model configuration."""
    from faster_whisper import WhisperModel

    logger.info(
        f"Loading WhisperModel(model='{model_size}', device='{device}', compute_type='{compute_type}')..."
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Use batched inference for 2-3x speedup (same model, identical output quality)
    batched_model = None
    try:
        from faster_whisper import BatchedInferencePipeline
        batched_model = BatchedInferencePipeline(model=model)
        logger.info("Using BatchedInferencePipeline for faster parallel decoding...")
        segments_gen, info = batched_model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            batch_size=batch_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=vad_min_silence_ms),
            word_timestamps=True,
        )
    except (ImportError, TypeError, Exception) as batch_err:
        logger.info(f"Batched inference unavailable ({batch_err}), using standard sequential mode.")
        batched_model = None
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=vad_min_silence_ms),
            word_timestamps=True,
        )

    try:
        raw_segments: List[RawWhisperSegment] = []
        for seg in segments_gen:
            words = [
                WordTiming(
                    word=w.word,
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(getattr(w, "probability", 1.0)),
                )
                for w in (seg.words or [])
            ]
            raw_segments.append(
                RawWhisperSegment(
                    id=seg.id,
                    seek=seg.seek,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    tokens=list(seg.tokens) if seg.tokens else [],
                    words=words,
                    avg_logprob=float(seg.avg_logprob),
                    no_speech_prob=float(seg.no_speech_prob),
                )
            )

        duration = get_media_duration(audio_path)
        return TranscriptionResult(
            language=info.language,
            language_probability=float(info.language_probability),
            duration=duration,
            raw_segments=raw_segments,
        )
    finally:
        if batched_model is not None:
            del batched_model
        del model
        cleanup_gpu_memory()


def transcribe_audio(config: DubberConfig) -> TranscriptionResult:
    """
    Transcribes audio to text with word-level timestamps using faster-whisper.
    Implements OOM fallback chain: large-v3 -> medium -> cpu int8.
    Output: config.transcript_path
    """
    transcript_path = config.transcript_path

    # Resumption check
    if config.resume and transcript_path.exists():
        try:
            res = TranscriptionResult.load_json(transcript_path)
            if res.raw_segments:
                logger.info(
                    f"Resume: Loaded existing transcript from {transcript_path} "
                    f"({len(res.raw_segments)} segments, detected language: '{res.language}')"
                )
                return res
        except Exception:
            logger.warning("Existing transcript invalid, re-transcribing...")

    # Hardware check & VRAM logging
    print_gpu_status(stage_name="Transcribe")
    is_cuda_avail, _, _, _ = get_gpu_info()

    target_device = config.device
    if target_device == "auto":
        target_device = "cuda" if is_cuda_avail else "cpu"

    # Define fallback attempt configurations
    fallback_attempts: List[Tuple[str, str, str, str]] = []

    if target_device == "cuda":
        # 1. Preferred GPU large-v3 int8_float16
        fallback_attempts.append((config.whisper_model, "cuda", "int8_float16", "GPU int8_float16"))
        # 2. GPU medium fallback
        if config.whisper_model == "large-v3":
            fallback_attempts.append(("medium", "cuda", "int8_float16", "GPU medium fallback"))
        # 3. CPU int8 fallback
        fallback_attempts.append((config.whisper_model, "cpu", "int8", "CPU int8 fallback"))
    else:
        fallback_attempts.append((config.whisper_model, "cpu", "int8", "CPU int8"))

    last_error: Optional[Exception] = None
    for model_size, dev, ctype, desc in fallback_attempts:
        try:
            logger.info(f"Attempting transcription via [{desc}]...")
            
            # Prevent silent C++ OOM crashes on CPU by lowering batch size
            safe_batch_size = min(4, config.whisper_batch_size) if dev == "cpu" else config.whisper_batch_size

            result = _run_whisper_inference(
                audio_path=config.source_audio_path,
                model_size=model_size,
                device=dev,
                compute_type=ctype,
                vad_filter=config.vad_filter,
                vad_min_silence_ms=config.vad_min_silence_ms,
                language=config.src_lang,
                task=config.whisper_task,
                batch_size=safe_batch_size,
            )

            # Validate language
            logger.info(
                f"Transcription complete: detected language '{result.language}' "
                f"(confidence: {result.language_probability:.2%}) with {len(result.raw_segments)} raw segments."
            )

            # Save artifact
            result.save_json(transcript_path)
            return result

        except Exception as e:
            last_error = e
            logger.warning(f"Transcription attempt [{desc}] failed: {e}")
            cleanup_gpu_memory()

    raise TranscriptionError(f"All transcription fallback attempts failed. Last error: {last_error}")
