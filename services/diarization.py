"""
Speaker Diarization Service
Uses pyannote.audio 3.x to detect who spoke when.

Output format:
  [{"speaker_id": "SPEAKER_0", "start": 0.00, "end": 3.45}, ...]
"""
from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from config import settings

logger = logging.getLogger(__name__)

# Lazy-load heavy ML models
_pipeline = None
_executor = ThreadPoolExecutor(max_workers=1)


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            import torch
            from pyannote.audio import Pipeline as PyannotePipeline

            logger.info("Loading pyannote diarization model …")
            _pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.HF_TOKEN,
            )
            if torch.cuda.is_available():
                _pipeline = _pipeline.to(torch.device("cuda"))
                logger.info("Diarization model loaded on GPU")
            else:
                logger.info("Diarization model loaded on CPU")
        except Exception as exc:
            logger.error("Failed to load pyannote pipeline: %s", exc)
            raise
    return _pipeline


def _run_diarization(audio_path: str, num_speakers: int = None) -> list:
    """Synchronous diarization — called in executor thread."""
    pipeline = _load_pipeline()

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    logger.info("Running diarization on %s", audio_path)
    diarization = pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Filter very short segments (< 0.3 s) — likely noise
        if (turn.end - turn.start) >= 0.3:
            segments.append(
                {
                    "speaker_id": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

    logger.info("Diarization complete: %d segments, speakers=%s",
                len(segments),
                {s["speaker_id"] for s in segments})
    return segments


async def diarize(audio_path: str, num_speakers: int = None) -> List[dict]:
    """
    Async entry point.
    Returns list of dicts:
      {"speaker_id": str, "start": float, "end": float}
    """
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        _executor,
        functools.partial(_run_diarization, audio_path, num_speakers),
    )
    return segments
