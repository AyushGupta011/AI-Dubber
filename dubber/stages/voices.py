"""Stage 6: Speaker Pitch Estimation (F0 pyin), Dynamic Voice Matching, and Reference Audio Extraction."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf
from dubber.config import DubberConfig
from dubber.models import Segment, VoiceAssignment
from dubber.utils.logger import logger

_gender_classifier = None

def get_gender_classifier():
    global _gender_classifier
    if _gender_classifier is None:
        try:
            from transformers import pipeline
            import torch
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                device = 0 if torch.cuda.is_available() else -1
                logger.info("Loading ML gender classifier (prithivMLmods/Common-Voice-Gender-Detection)...")
                _gender_classifier = pipeline(
                    "audio-classification",
                    model="prithivMLmods/Common-Voice-Gender-Detection",
                    device=device
                )
        except Exception as e:
            logger.warning(f"Failed to load ML gender classifier: {e}")
            _gender_classifier = "failed"
    return _gender_classifier if _gender_classifier != "failed" else None

def estimate_segment_gender(audio_path: Path, start_sec: float, end_sec: float, sr: int = 16000) -> Optional[str]:
    """Estimates gender using a HuggingFace audio classifier."""
    classifier = get_gender_classifier()
    if not classifier:
        return None
        
    start_sample = max(0, int(start_sec * sr))
    end_sample = max(start_sample + 1, int(end_sec * sr))
    frames_to_read = end_sample - start_sample

    try:
        y, file_sr = sf.read(str(audio_path), start=start_sample, frames=frames_to_read, dtype="float32")
        if len(y) == 0:
            return None

        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Skip silent/very low energy audio
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-4:
            return None

        result = classifier(y)
        if result and len(result) > 0:
            return result[0]['label'].lower()
    except Exception as e:
        logger.debug(f"ML gender estimation failed for segment [{start_sec:.2f}-{end_sec:.2f}]: {e}")
    return None

def extract_segment_mfcc(audio_path: Path, start_sec: float, end_sec: float, sr: int = 16000) -> Optional[np.ndarray]:
    """Extracts a 20-dim MFCC feature vector for a segment."""
    start_sample = max(0, int(start_sec * sr))
    end_sample = max(start_sample + 1, int(end_sec * sr))
    frames_to_read = end_sample - start_sample

    try:
        y, file_sr = sf.read(str(audio_path), start=start_sample, frames=frames_to_read, dtype="float32")
        if len(y) == 0:
            return None
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-4:
            return None

        import librosa
        mfcc = librosa.feature.mfcc(y=y, sr=file_sr, n_mfcc=20)
        # Drop the first coefficient (energy/volume) to cluster by timbre
        return np.mean(mfcc[1:], axis=1)
    except Exception as e:
        logger.debug(f"MFCC extraction failed for segment [{start_sec:.2f}-{end_sec:.2f}]: {e}")
        return None

# Base reference pitch (Hz) for default Microsoft Neural Voices
BASE_VOICE_PITCH_HZ = {
    "en-US-ChristopherNeural": 105.0,
    "en-US-GuyNeural": 128.0,
    "en-US-JennyNeural": 195.0,
    "en-US-AriaNeural": 225.0,
}


class VoiceAssigner(ABC):
    """Abstract interface for voice assignment strategies."""

    @abstractmethod
    def assign_voices(
        self,
        config: DubberConfig,
        segments: List[Segment],
        audio_path: Path,
    ) -> List[VoiceAssignment]:
        """Assigns TTS voice identifiers and pitch modifications to each segment."""
        pass


def estimate_segment_f0(audio_path: Path, start_sec: float, end_sec: float, sr: int = 16000) -> Optional[float]:
    """
    Estimates median fundamental frequency (F0) for an audio slice using librosa.pyin.
    """
    start_sample = max(0, int(start_sec * sr))
    end_sample = max(start_sample + 1, int(end_sec * sr))
    frames_to_read = end_sample - start_sample

    try:
        y, file_sr = sf.read(str(audio_path), start=start_sample, frames=frames_to_read, dtype="float32")
        if len(y) == 0:
            return None

        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Skip silent/very low energy audio
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-4:
            return None

        import librosa

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C6"),  # ~1046 Hz
            sr=file_sr,
        )

        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(valid_f0) > 0:
            return float(np.median(valid_f0))

    except Exception as e:
        logger.debug(f"F0 estimation failed for segment [{start_sec:.2f}-{end_sec:.2f}]: {e}")

    return None


def select_matched_voice(f0: Optional[float], ml_gender: Optional[str], config: DubberConfig) -> Tuple[str, str, Optional[str]]:
    """
    Selects the best matching voice profile and computes pitch shift (Hz):
    Returns: (gender, voice_name, pitch_shift_str)
    """
    if ml_gender in ["male", "female"]:
        gender = ml_gender
        if gender == "male":
            if f0 is not None and f0 < 115.0:
                voice = config.voice_male_deep
                base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 105.0)
            else:
                voice = config.voice_male
                base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 128.0)
        else:
            if f0 is not None and f0 >= 215.0:
                voice = config.voice_female_high
                base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 225.0)
            else:
                voice = config.voice_female
                base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 195.0)
    else:
        if f0 is None:
            # Extreme fallback if global median is also None
            return "male", config.voice_male, None

        if f0 < 115.0:
            # Deep Male voice
            gender = "male"
            voice = config.voice_male_deep
            base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 105.0)
        elif f0 < config.pitch_gender_threshold_hz:
            # Standard Male voice
            gender = "male"
            voice = config.voice_male
            base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 128.0)
        elif f0 < 215.0:
            # Warm / Natural Female voice
            gender = "female"
            voice = config.voice_female
            base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 195.0)
        else:
            # Higher Pitch Female voice
            gender = "female"
            voice = config.voice_female_high
            base_f0 = BASE_VOICE_PITCH_HZ.get(voice, 225.0)

    pitch_shift_str = None
    if config.match_voice_pitch and f0 is not None:
        # Compute difference from base voice and cap between -30Hz and +30Hz for naturalness
        delta_hz = int(np.clip(round(f0 - base_f0), -30, 30))
        if abs(delta_hz) >= 2:
            pitch_shift_str = f"{delta_hz:+d}Hz"

    return gender, voice, pitch_shift_str


def extract_cleanest_reference_audio(audio_path: Path, segments: List[Segment], output_ref: Path, sr: int = 16000) -> Optional[Path]:
    """
    Extracts a 6-10 second clean speech segment as speaker reference for voice cloning.
    """
    if not segments:
        return None

    # Pick the longest continuous segment with speech
    best_seg = max(segments, key=lambda s: s.duration)
    start_sec = best_seg.start
    end_sec = min(start_sec + 10.0, best_seg.end)

    start_sample = max(0, int(start_sec * sr))
    frames_to_read = int((end_sec - start_sec) * sr)

    try:
        data, file_sr = sf.read(str(audio_path), start=start_sample, frames=frames_to_read, dtype="float32")
        output_ref.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_ref), data, file_sr)
        return output_ref
    except Exception as e:
        logger.debug(f"Failed to extract reference audio: {e}")
        return None


def extract_segment_reference_audio(
    audio_path: Path,
    seg: Segment,
    output_path: Path,
    sr: int = 16000,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
) -> Optional[Path]:
    """
    Extracts a clean reference audio clip for a single segment from the source audio.
    The reference captures the original speaker's voice characteristics for that segment.

    For short segments (< min_duration), we extend the window slightly to give XTTS
    enough context for voice matching.
    """
    start_sec = seg.start
    end_sec = seg.end
    duration = end_sec - start_sec

    # Ensure minimum reference duration for XTTS (needs ~3s to capture voice)
    if duration < min_duration:
        # Extend symmetrically around the segment center
        center = (start_sec + end_sec) / 2.0
        half_dur = min_duration / 2.0
        start_sec = max(0.0, center - half_dur)
        end_sec = center + half_dur

    # Cap to max_duration for performance
    if (end_sec - start_sec) > max_duration:
        end_sec = start_sec + max_duration

    start_sample = max(0, int(start_sec * sr))
    frames_to_read = int((end_sec - start_sec) * sr)

    try:
        data, file_sr = sf.read(str(audio_path), start=start_sample, frames=frames_to_read, dtype="float32")
        if len(data) == 0:
            return None

        # Convert stereo to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Skip silent/near-silent segments
        rms = np.sqrt(np.mean(data ** 2))
        if rms < 1e-4:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), data, file_sr)
        return output_path
    except Exception as e:
        logger.debug(f"Failed to extract reference for segment {seg.id}: {e}")
        return None


class PitchVoiceAssigner(VoiceAssigner):
    """
    Dynamic voice assigner that categorizes speakers by pitch and matches voice timbre.
    When voice cloning is enabled, extracts per-segment reference audio from the
    original source so XTTS can clone the original speaker's voice for each segment.
    """

    def assign_voices(
        self,
        config: DubberConfig,
        segments: List[Segment],
        audio_path: Path,
    ) -> List[VoiceAssignment]:
        assignments: List[VoiceAssignment] = []
        logger.info(f"Analyzing pitch for {len(segments)} segments to dynamically match voice...")

        # Always extract a global reference (used as fallback for very short segments)
        global_ref_path = None
        if config.clone_voice:
            ref_path = config.work_dir / "speaker_ref.wav"
            result = extract_cleanest_reference_audio(audio_path, segments, ref_path)
            if result:
                global_ref_path = str(result)
                logger.info(f"Extracted global speaker reference audio: {ref_path}")

        # Extract per-segment reference audio clips for voice cloning
        refs_dir = config.work_dir / "refs"
        segment_refs: dict = {}
        if config.clone_voice:
            refs_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting per-segment speaker reference clips for voice cloning...")
            for seg in segments:
                ref_out = refs_dir / f"ref_{seg.id:04d}.wav"
                ref_result = extract_segment_reference_audio(audio_path, seg, ref_out)
                if ref_result:
                    segment_refs[seg.id] = str(ref_result)
            logger.info(f"Extracted {len(segment_refs)}/{len(segments)} segment reference clips.")

        # Pass 1: Estimate F0, ML Gender, and MFCCs for all segments in parallel (CPU-bound, thread-safe)
        from concurrent.futures import ThreadPoolExecutor

        def _estimate_params(seg):
            f0 = estimate_segment_f0(audio_path, seg.start, seg.end)
            ml_gen = estimate_segment_gender(audio_path, seg.start, seg.end)
            mfcc = extract_segment_mfcc(audio_path, seg.start, seg.end) if getattr(config, "diarization", True) else None
            return (seg.id, f0, ml_gen, mfcc)

        import os
        max_workers = min(os.cpu_count() or 16, len(segments)) if segments else 1
        f0_results = {}
        ml_gender_results = {}
        mfcc_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for seg_id, f0, ml_gen, mfcc in pool.map(_estimate_params, segments):
                f0_results[seg_id] = f0
                ml_gender_results[seg_id] = ml_gen
                if mfcc is not None:
                    mfcc_results[seg_id] = mfcc

        valid_f0s = []
        for seg in segments:
            f0 = f0_results.get(seg.id)
            seg._temp_f0 = f0
            if f0 is not None:
                valid_f0s.append((seg.start, f0))

        # Perform Diarization if enabled
        if getattr(config, "diarization", True) and len(mfcc_results) > 1:
            logger.info("Performing speaker diarization to ensure consistent voices...")
            try:
                from sklearn.cluster import AgglomerativeClustering
                
                valid_ids = sorted(list(mfcc_results.keys()))
                X = np.array([mfcc_results[sid] for sid in valid_ids])
                
                # Use cosine distance for timbre MFCCs (invariant to volume differences)
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.05,  # Appropriate for cosine distance
                    metric='cosine',
                    linkage='average'
                )
                labels = clusterer.fit_predict(X)
                
                cluster_f0s = {}
                cluster_genders = {}
                for sid, label in zip(valid_ids, labels):
                    if f0_results.get(sid) is not None:
                        cluster_f0s.setdefault(label, []).append(f0_results[sid])
                    if ml_gender_results.get(sid) is not None:
                        cluster_genders.setdefault(label, []).append(ml_gender_results[sid])
                
                cluster_agg_f0 = {}
                cluster_agg_gender = {}
                for label in set(labels):
                    if label in cluster_f0s:
                        cluster_agg_f0[label] = np.median(cluster_f0s[label])
                    if label in cluster_genders:
                        cluster_agg_gender[label] = max(set(cluster_genders[label]), key=cluster_genders[label].count)
                
                for sid, label in zip(valid_ids, labels):
                    if label in cluster_agg_f0:
                        f0_results[sid] = cluster_agg_f0[label]
                    if label in cluster_agg_gender:
                        ml_gender_results[sid] = cluster_agg_gender[label]
                        
            except ImportError:
                logger.warning("scikit-learn is missing. Skipping diarization...")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Falling back to per-segment assignment.")

        # Pass 2: Assign voices using temporal context for missing F0s
        for seg in segments:
            f0 = f0_results.get(seg.id)
            ml_gender = ml_gender_results.get(seg.id)
            
            # If pitch detection failed for this segment, fallback to the closest valid segment in time
            if f0 is None and valid_f0s:
                closest_f0 = min(valid_f0s, key=lambda x: abs(x[0] - seg.start))[1]
                f0 = closest_f0

            if getattr(config, "force_gender", None) == "male":
                gender = "male"
                voice = config.voice_male
                pitch_shift = None
            elif getattr(config, "force_gender", None) == "female":
                gender = "female"
                voice = config.voice_female
                pitch_shift = None
            else:
                gender, voice, pitch_shift = select_matched_voice(f0, ml_gender, config)

            seg.estimated_gender = gender
            seg.f0_median_hz = f0

            # Per-segment ref audio for cloning, fall back to global ref
            ref_audio = segment_refs.get(seg.id, global_ref_path)

            assignments.append(
                VoiceAssignment(
                    segment_id=seg.id,
                    speaker_id=f"speaker_{gender}",
                    gender=gender,
                    voice_name=voice,
                    f0_median_hz=f0,
                    pitch_shift_str=pitch_shift,
                    ref_audio_path=ref_audio,
                )
            )

        return assignments


def assign_voices(
    config: DubberConfig,
    segments: List[Segment],
) -> List[VoiceAssignment]:
    """
    Assigns dynamically matched voices to segments and saves voice_assignments.json.
    """
    assignments_path = config.voice_assignments_path

    # Resumption check
    if config.resume and assignments_path.exists():
        try:
            with open(assignments_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assignments = [VoiceAssignment.from_dict(v) for v in data]
            if len(assignments) == len(segments):
                for seg, ass in zip(segments, assignments):
                    seg.estimated_gender = ass.gender
                    seg.f0_median_hz = ass.f0_median_hz
                logger.info(f"Resume: Loaded voice assignments from {assignments_path}")
                return assignments
        except Exception:
            logger.warning("Existing voice assignments invalid, re-analyzing...")

    assigner = PitchVoiceAssigner()
    assignments = assigner.assign_voices(config, segments, config.source_audio_path)

    male_count = sum(1 for a in assignments if a.gender == 'male')
    female_count = sum(1 for a in assignments if a.gender == 'female')
    cloned_count = sum(1 for a in assignments if a.ref_audio_path)
    logger.info(f"Voice assignment complete: {male_count} Male, {female_count} Female, {cloned_count} with reference audio.")

    # Save artifact
    with open(assignments_path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in assignments], f, indent=2)

    return assignments
