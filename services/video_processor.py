"""
Video Processor Service
All FFmpeg operations:
  - Audio extraction from video
  - Video duration measurement
  - Replacing video audio track with dubbed WAV
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from functools import partial
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)


def _run(cmd: list[str], step_name: str) -> None:
    """Run an FFmpeg/FFprobe command; raise on failure with clean error."""
    logger.debug("Running [%s]: %s", step_name, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("[%s] stderr: %s", step_name, result.stderr[-500:])
        raise RuntimeError(
            f"[{step_name}] FFmpeg exited with code {result.returncode}: "
            f"{result.stderr[-300:]}"
        )


# ---------------------------------------------------------------------------
# Step 1: Extract audio
# ---------------------------------------------------------------------------

def _extract_audio_sync(video_path: str, audio_path: str) -> str:
    """
    Extract audio from a video file as a 16-bit PCM WAV at 16 kHz mono.
    This format is optimal for Whisper and pyannote.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", "16000",                 # 16 kHz (Whisper-optimal)
        "-ac", "1",                     # mono
        audio_path,
    ]
    _run(cmd, "extract_audio")
    logger.info("Audio extracted: %s", audio_path)
    return audio_path


async def extract_audio(video_path: str, audio_path: str) -> str:
    """Async wrapper for audio extraction."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, partial(_extract_audio_sync, video_path, audio_path)
    )


# ---------------------------------------------------------------------------
# Step 2: Get video duration
# ---------------------------------------------------------------------------

def _get_duration_sync(video_path: str) -> float:
    """Return total video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
        logger.info("Video duration: %.3fs", duration)
        return duration
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not determine video duration: {result.stderr}") from exc


async def get_video_duration(video_path: str) -> float:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, partial(_get_duration_sync, video_path))


# ---------------------------------------------------------------------------
# Step 3: Replace audio track
# ---------------------------------------------------------------------------

def _replace_audio_sync(
    input_video: str,
    dubbed_audio: str,
    output_video: str,
) -> str:
    """
    Replace the audio track of input_video with dubbed_audio.
    Video stream is copied without re-encoding.

    ffmpeg -i video -i audio -map 0:v -map 1:a -c:v copy -shortest output.mp4
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", dubbed_audio,
        "-map", "0:v",           # video from source
        "-map", "1:a",           # audio from dubbed track
        "-c:v", "copy",          # no video re-encode
        "-c:a", "aac",           # encode audio to AAC for broad compatibility
        "-b:a", "192k",
        "-shortest",             # trim to shorter stream
        output_video,
    ]
    _run(cmd, "replace_audio")
    logger.info("Final video written: %s", output_video)
    return output_video


async def replace_audio(
    input_video: str,
    dubbed_audio: str,
    output_video: str,
) -> str:
    """Async wrapper for audio replacement."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        partial(_replace_audio_sync, input_video, dubbed_audio, output_video),
    )
