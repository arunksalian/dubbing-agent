"""
Text-to-Speech Service
Uses ElevenLabs multilingual v2 model for high-quality dubbing voices.
Features:
  - Async generation with semaphore-controlled concurrency
  - Exponential backoff retry on API errors
  - Per-segment audio file output
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
MODEL_ID = "eleven_multilingual_v2"

DEFAULT_VOICE_SETTINGS = {
    "stability": 0.50,
    "similarity_boost": 0.75,
    "style": 0.10,
    "use_speaker_boost": True,
}


async def _generate_tts(
    client: httpx.AsyncClient,
    text: str,
    voice_id: str,
    output_path: str,
) -> None:
    """
    Call ElevenLabs TTS endpoint with retry logic.
    Writes MP3 bytes to output_path.
    """
    url = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": DEFAULT_VOICE_SETTINGS,
    }

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=60.0,
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "10"))
                logger.warning("ElevenLabs rate limit hit. Waiting %ds …", retry_after)
                await asyncio.sleep(retry_after)
                raise httpx.HTTPStatusError(
                    "Rate limited", request=response.request, response=response
                )
            response.raise_for_status()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.debug("TTS saved: %s (%d bytes)", output_path, len(response.content))


async def generate_segment_audio(
    segment: dict,
    voice_map: Dict[str, str],
    output_dir: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Generate TTS audio for a single segment.

    Returns the segment dict enriched with:
      - tts_audio_path: str
      - tts_duration:   float  (seconds, measured from generated file)
    """
    speaker_id = segment["speaker_id"]
    voice_id = voice_map.get(speaker_id)
    if not voice_id:
        logger.error("No voice mapped for speaker %s — skipping TTS", speaker_id)
        return {**segment, "tts_audio_path": None, "tts_duration": None}

    text = segment.get("translated_text") or segment.get("text", "")
    if not text.strip():
        logger.warning("Empty text for segment [%.2f–%.2f]", segment["start"], segment["end"])
        return {**segment, "tts_audio_path": None, "tts_duration": 0.0}

    # Unique filename per segment
    seg_id = f"{speaker_id}_{segment['start']:.3f}_{segment['end']:.3f}"
    output_path = os.path.join(output_dir, f"tts_{seg_id}.mp3")

    async with semaphore:
        async with httpx.AsyncClient() as client:
            await _generate_tts(client, text, voice_id, output_path)

    # Measure actual TTS duration
    tts_duration = _measure_duration(output_path)
    logger.info(
        "TTS [%s %.2f–%.2f] target=%.2fs generated=%.2fs",
        speaker_id,
        segment["start"],
        segment["end"],
        segment["end"] - segment["start"],
        tts_duration,
    )

    return {**segment, "tts_audio_path": output_path, "tts_duration": tts_duration}


def _measure_duration(audio_path: str) -> float:
    """Return audio duration in seconds using pydub."""
    try:
        from pydub import AudioSegment as PyAudio

        audio = PyAudio.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception as exc:
        logger.warning("Could not measure duration of %s: %s", audio_path, exc)
        return 0.0


async def generate_all_tts(
    segments: List[dict],
    voice_map: Dict[str, str],
    output_dir: str,
    concurrency: int = 3,
) -> List[dict]:
    """
    Generate TTS audio for all translated segments concurrently.

    Args:
        segments:    translated segments with 'translated_text' field
        voice_map:   {"SPEAKER_0": "voice_id", ...}
        output_dir:  directory to write MP3 files
        concurrency: max parallel ElevenLabs requests

    Returns:
        Segments enriched with 'tts_audio_path' and 'tts_duration'
    """
    os.makedirs(output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        generate_segment_audio(seg, voice_map, output_dir, semaphore)
        for seg in segments
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)
    successful = sum(1 for r in results if r.get("tts_audio_path"))
    logger.info("TTS generation complete: %d/%d segments", successful, len(segments))
    return list(results)
