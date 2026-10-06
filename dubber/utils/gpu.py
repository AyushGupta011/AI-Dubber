"""GPU and VRAM hardware diagnostics and memory lifecycle management."""

import gc
import logging
from typing import Tuple, Optional

logger = logging.getLogger("dubber.gpu")


def get_gpu_info() -> Tuple[bool, str, float, float]:
    """
    Queries CUDA availability, device name, free VRAM (GB), and total VRAM (GB).
    Returns: (is_available, name, free_gb, total_gb)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device_id)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            return True, name, free_gb, total_gb
    except Exception as e:
        logger.debug(f"GPU query failed: {e}")

    return False, "CPU / No GPU detected", 0.0, 0.0


def print_gpu_status(stage_name: Optional[str] = None) -> None:
    """Prints formatted GPU name and available free VRAM before stages."""
    is_avail, name, free_gb, total_gb = get_gpu_info()
    prefix = f"[{stage_name}] " if stage_name else ""
    if is_avail:
        print(f"[GPU] {prefix}Hardware: {name} | Free VRAM: {free_gb:.2f} GB / {total_gb:.2f} GB")
    else:
        print(f"[CPU] {prefix}Hardware: Running on CPU (No CUDA GPU available)")


def cleanup_gpu_memory() -> None:
    """
    Strict GPU model lifecycle cleanup:
    Forces Python garbage collection and clears PyTorch CUDA cache.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
