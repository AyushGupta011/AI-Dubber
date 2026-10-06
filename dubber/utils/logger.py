"""Console logging, stage banners, and timing utilities."""

import sys
import io
import time
import logging
from contextlib import contextmanager
from typing import Generator

# Ensure utf-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("dubber")


def print_banner(stage_num: int, total_stages: int, title: str) -> None:
    """Prints a styled stage banner to console."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f">> Stage {stage_num}/{total_stages}: {title}")
    print(f"{bar}")


@contextmanager
def stage_timer(stage_name: str) -> Generator[dict, None, None]:
    """Context manager to measure and report elapsed execution time of a stage."""
    state = {"start_time": time.time(), "elapsed": 0.0}
    print(f"[*] Starting {stage_name}...")
    try:
        yield state
    finally:
        state["elapsed"] = time.time() - state["start_time"]
        print(f"[+] Completed {stage_name} in {state['elapsed']:.2f}s")
