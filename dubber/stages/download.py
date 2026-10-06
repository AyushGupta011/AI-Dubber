"""Stage 1: Video Ingestion & Download via yt-dlp."""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from dubber.config import DubberConfig
from dubber.errors import DownloadError
from dubber.utils.ffmpeg import get_media_duration
from dubber.utils.logger import logger


def is_local_file(url_or_path: str) -> bool:
    """Checks if the input is a local file path."""
    p = Path(url_or_path)
    return p.exists() and p.is_file()


def download_video(config: DubberConfig) -> Path:
    """
    Downloads YouTube video using yt-dlp or copies local video to work directory.
    Output: config.source_video_path
    """
    target_path = config.source_video_path
    meta_path = config.work_dir / "video_meta.json"

    # Resumption check
    if config.resume and target_path.exists() and target_path.stat().st_size > 1024:
        try:
            dur = get_media_duration(target_path)
            if dur > 0:
                logger.info(f"Resume: Using existing video at {target_path} (Duration: {dur:.2f}s)")
                return target_path
        except Exception:
            logger.warning("Existing source video corrupted or invalid, re-downloading...")

    # Case 1: Local file input
    if is_local_file(config.url):
        logger.info(f"Using local video file: {config.url}")
        shutil.copy2(config.url, target_path)
        meta = {
            "title": Path(config.url).stem,
            "duration": get_media_duration(target_path),
            "is_local": True,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return target_path

    # Case 2: YouTube URL
    logger.info(f"Downloading YouTube video: {config.url}")
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError("yt-dlp is not installed. Run `pip install yt-dlp`.") from e

    ydl_opts: Dict[str, Any] = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(target_path.with_suffix("")),
        "merge_output_format": "mp4",
        "overwrites": True,
        "quiet": False,
        "no_warnings": False,
    }

    if config.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (config.cookies_from_browser,)
    elif config.cookies_file and os.path.exists(config.cookies_file):
        ydl_opts["cookiefile"] = config.cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(config.url, download=True)
            if info is None:
                raise DownloadError(f"Could not extract info for URL: {config.url}")

            meta = {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "view_count": info.get("view_count"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    except Exception as e:
        # Check if the file was downloaded despite minor hook warnings
        if target_path.exists() and target_path.stat().st_size > 1024:
            return target_path
            
        # Handle WinError 32 on Windows (persistent lock by Antivirus or delayed ffmpeg exit)
        import time
        temp_path = target_path.with_name(target_path.stem + ".temp.mp4")
        if temp_path.exists() and temp_path.stat().st_size > 1024:
            logger.info("File is locked by another process (likely Antivirus or ffmpeg). Waiting for lock to release...")
            
            # Retry loop: wait up to 30 seconds (15 retries * 2s) for the lock to release
            max_retries = 15
            for attempt in range(max_retries):
                time.sleep(2.0)
                try:
                    shutil.move(str(temp_path), str(target_path))
                    logger.info("Successfully recovered and renamed the locked file!")
                    return target_path
                except Exception as move_err:
                    if attempt == max_retries - 1:
                        logger.warning(f"Could not recover from temp file lock after {max_retries} attempts: {move_err}")
                    else:
                        logger.debug(f"File still locked, retrying ({attempt + 1}/{max_retries})...")

        raise DownloadError(f"Failed to download video from {config.url}: {e}") from e

    if not target_path.exists():
        # Handle case where yt-dlp appended extension or changed filename
        potential_files = list(config.work_dir.glob("source_video*"))
        for pf in potential_files:
            if pf.suffix in [".mp4", ".mkv", ".webm"] and pf.stat().st_size > 1024:
                shutil.move(str(pf), str(target_path))
                break

    if not target_path.exists():
        raise DownloadError(f"Download completed but output file not found at {target_path}")

    return target_path
