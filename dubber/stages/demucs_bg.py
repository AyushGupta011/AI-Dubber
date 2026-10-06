"""Optional Stage: Background Music & Sound Preservation with Vocal Isolation & Ducking."""

from pathlib import Path
from typing import Optional
from dubber.config import DubberConfig
from dubber.utils.gpu import cleanup_gpu_memory, print_gpu_status
from dubber.utils.logger import logger


def separate_and_duck_background(
    config: DubberConfig,
    source_audio: Path,
    dubbed_speech: Path,
    output_audio: Path,
    duck_gain_db: float = -12.0,
) -> Path:
    """
    Separates background instruments/sounds from vocals using Demucs,
    applies ducking under speech, and mixes with the dubbed English speech.
    """
    logger.info("Separating vocal and background audio tracks with Demucs...")
    print_gpu_status(stage_name="Demucs Background Separation")

    try:
        import torch
        from demucs import pretrained
        from demucs.apply import apply_model
        import soundfile as sf
        import numpy as np

        # Load demucs model (e.g., htdemucs)
        model = pretrained.get_model("htdemucs")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Process source audio
        y, sr = sf.read(str(source_audio), dtype="float32")
        if y.ndim == 1:
            y = np.stack([y, y], axis=0)
        else:
            y = y.T

        tensor = torch.tensor(y[None, :, :], device=device, dtype=torch.float32)

        with torch.no_grad():
            sources = apply_model(model, tensor, device=device)[0]

        # Sources in htdemucs: [drums, bass, other, vocals]
        # Background = drums + bass + other
        bg_tensor = sources[0] + sources[1] + sources[2]
        bg_audio = bg_tensor.cpu().numpy().T  # (samples, 2)

        del model, tensor, sources, bg_tensor
        cleanup_gpu_memory()

        # Load dubbed speech
        speech_y, speech_sr = sf.read(str(dubbed_speech), dtype="float32")
        if speech_y.ndim == 1:
            speech_y = np.stack([speech_y, speech_y], axis=1)

        # Duck background by duck_gain_db (e.g. -12dB -> factor ~0.25)
        duck_factor = 10.0 ** (duck_gain_db / 20.0)
        ducked_bg = bg_audio[: len(speech_y)] * duck_factor

        # Mix speech + ducked background
        mixed = speech_y + ducked_bg
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val  # Normalize to prevent clipping

        sf.write(str(output_audio), mixed, sr, subtype="PCM_16")
        logger.info(f"Background audio mixed and saved to {output_audio}")
        return output_audio

    except ImportError:
        logger.warning("demucs package not installed. Skipping background preservation.")
        return dubbed_speech
    except Exception as e:
        logger.warning(f"Background separation failed: {e}. Falling back to clean dubbed speech.")
        return dubbed_speech
