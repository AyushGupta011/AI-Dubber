"""Stage 7: Asynchronous Speech Synthesis via edge-tts with Pitch Matching & Voice Cloning."""

import asyncio
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import edge_tts
from tqdm.asyncio import tqdm_asyncio
from dubber.config import DubberConfig
from dubber.models import TranslatedSegment, VoiceAssignment, TTSClip
from dubber.errors import TTSError
from dubber.utils.ffmpeg import get_media_duration
from dubber.utils.logger import logger
from tqdm import tqdm

def _synthesize_xtts(
    config: DubberConfig,
    translated_segments: List[TranslatedSegment],
    voice_assignments: List[VoiceAssignment],
) -> List[TTSClip]:
    """Synthesizes speech using Coqui XTTS v2 for zero-shot voice cloning."""
    import os
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS
    from dubber.utils.gpu import get_gpu_info, cleanup_gpu_memory
    
    is_cuda_avail, _, _, _ = get_gpu_info()
    device = "cuda" if (config.device in ["auto", "cuda"] and is_cuda_avail) else "cpu"
    
    logger.info(f"Loading Coqui XTTS v2 model on {device} for zero-shot voice cloning (this may take a moment)...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    clips = []
    tts_dir = config.tts_dir
    
    ref_audio_map = {a.segment_id: a.ref_audio_path for a in voice_assignments if a.ref_audio_path}
    default_ref = str(config.work_dir / "speaker_ref.wav") if (config.work_dir / "speaker_ref.wav").exists() else None

    for seg in tqdm(translated_segments, desc="Cloning voices with XTTS"):
        out_file = tts_dir / f"seg_{seg.id:04d}.wav"
        
        # Check cache
        if out_file.exists() and out_file.stat().st_size > 512:
            try:
                dur = get_media_duration(out_file)
                if dur > 0.05:
                    clips.append(TTSClip(segment_id=seg.id, audio_path=str(out_file), duration=dur, voice_name="xtts-clone"))
                    continue
            except Exception:
                pass
                
        text = seg.dst_text.strip()
        if not text:
            text = "..."
            
        ref_path = ref_audio_map.get(seg.id, default_ref)
        if not ref_path or not os.path.exists(ref_path):
            logger.warning(f"No reference audio found for segment {seg.id}. Skipping clone.")
            continue
            
        # XTTS generation
        try:
            tts_model.tts_to_file(
                text=text,
                speaker_wav=ref_path,
                language="en",
                file_path=str(out_file)
            )
            dur = get_media_duration(out_file)
            clips.append(TTSClip(segment_id=seg.id, audio_path=str(out_file), duration=dur, voice_name="xtts-clone"))
        except Exception as e:
            logger.error(f"XTTS cloning failed for segment {seg.id}: {e}")
        
    del tts_model
    cleanup_gpu_memory()
    return clips



async def _synthesize_single_segment(
    seg: TranslatedSegment,
    voice_name: str,
    pitch_shift: Optional[str],
    output_path: Path,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> TTSClip:
    """Synthesizes speech for a single segment with dynamic pitch adjustment and retry backoff."""
    # Check cache
    if output_path.exists() and output_path.stat().st_size > 512:
        try:
            dur = get_media_duration(output_path)
            if dur > 0.05:
                return TTSClip(
                    segment_id=seg.id,
                    audio_path=str(output_path),
                    duration=dur,
                    voice_name=voice_name,
                )
        except Exception:
            pass

    text = seg.dst_text.strip()
    if not text:
        text = "..."

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                # Pass pitch shift to edge_tts if available
                if pitch_shift:
                    communicate = edge_tts.Communicate(text=text, voice=voice_name, pitch=pitch_shift)
                else:
                    communicate = edge_tts.Communicate(text=text, voice=voice_name)

                await communicate.save(str(output_path))

                if output_path.exists() and output_path.stat().st_size > 256:
                    dur = get_media_duration(output_path)
                    return TTSClip(
                        segment_id=seg.id,
                        audio_path=str(output_path),
                        duration=dur,
                        voice_name=voice_name,
                    )
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                logger.debug(f"TTS retry {attempt + 1}/{max_retries} for seg {seg.id} after {wait_time:.2f}s: {e}")
                await asyncio.sleep(wait_time)

        raise TTSError(f"Failed to synthesize audio for segment {seg.id} ('{text}') after {max_retries} retries: {last_error}")


async def _synthesize_all_async(
    config: DubberConfig,
    translated_segments: List[TranslatedSegment],
    voice_map: Dict[int, str],
    pitch_map: Dict[int, Optional[str]],
) -> List[TTSClip]:
    """Runs concurrent async TTS synthesis across all segments with pitch matching."""
    semaphore = asyncio.Semaphore(config.tts_concurrency)
    tts_dir = config.tts_dir

    tasks = []
    for seg in translated_segments:
        voice = voice_map.get(seg.id, config.voice_female)
        pitch = pitch_map.get(seg.id)
        out_file = tts_dir / f"seg_{seg.id:04d}.mp3"
        tasks.append(
            _synthesize_single_segment(
                seg=seg,
                voice_name=voice,
                pitch_shift=pitch,
                output_path=out_file,
                semaphore=semaphore,
            )
        )

    logger.info(f"Synthesizing {len(tasks)} speech clips with pitch matching (concurrency: {config.tts_concurrency})...")
    clips = await tqdm_asyncio.gather(*tasks, desc="Generating TTS clips")
    clips.sort(key=lambda c: c.segment_id)
    return clips


def generate_tts(
    config: DubberConfig,
    translated_segments: List[TranslatedSegment],
    voice_assignments: List[VoiceAssignment],
) -> List[TTSClip]:
    """
    Synchronous entry point for Stage 7 TTS generation.
    Supports voice pitch matching and voice cloning fallback.
    """
    if config.clone_voice:
        try:
            clips = _synthesize_xtts(config, translated_segments, voice_assignments)
            if clips:
                logger.info(f"Speech synthesis (Voice Cloning) complete: generated {len(clips)} audio clips.")
                return clips
        except ImportError:
            logger.error("TTS package not installed! Falling back to edge-tts. Install it using: pip install TTS")
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}. Falling back to edge-tts.")

    voice_map = {a.segment_id: a.voice_name for a in voice_assignments}
    pitch_map = {a.segment_id: a.pitch_shift_str for a in voice_assignments}

    clips = asyncio.run(_synthesize_all_async(config, translated_segments, voice_map, pitch_map))
    logger.info(f"Speech synthesis complete: generated {len(clips)} audio clips.")
    return clips
