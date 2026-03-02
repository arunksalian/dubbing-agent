"""
Speech-to-Text Service
Uses faster-whisper for high-speed, accurate transcription with word timestamps.
Then merges with speaker diarization output to produce speaker-labeled segments.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from config import settings

logger = logging.getLogger(__name__)

_model = None
_executor = ThreadPoolExecutor(max_workers=1)


def _load_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model: size=%s device=%s compute=%s",
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded")
    return _model


def _run_transcription(audio_path: str, language: Optional[str] = None) -> list:
    """Synchronous transcription — called in executor thread."""
    model = _load_model()
    kwargs = {"beam_size": 5, "word_timestamps": True, "vad_filter": True}
    if language:
        kwargs["language"] = language

    logger.info("Transcribing %s", audio_path)
    segments_iter, info = model.transcribe(audio_path, **kwargs)

    results = []
    for seg in segments_iter:
        results.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (seg.words or [])
                ],
            }
        )

    logger.info(
        "Transcription complete: %d segments, language=%s (prob=%.2f)",
        len(results),
        info.language,
        info.language_probability,
    )
    return results


def _merge_diarization_transcription(
    transcription: list,
    diarization: list,
) -> List[dict]:
    """
    Assign speaker labels to each transcription segment using diarization.
    Strategy: for each transcription segment, find which speaker occupied
    the majority of its time range in the diarization output.

    Returns:
      [{"speaker_id", "start", "end", "text"}, ...]
    """
    merged = []

    for t_seg in transcription:
        t_start, t_end = t_seg["start"], t_seg["end"]
        t_dur = t_end - t_start

        if t_dur <= 0:
            continue

        # Accumulate overlap per speaker
        speaker_overlap: dict[str, float] = {}
        for d_seg in diarization:
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                speaker = d_seg["speaker_id"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        # Assign dominant speaker (fallback to SPEAKER_0)
        if speaker_overlap:
            dominant = max(speaker_overlap, key=speaker_overlap.get)
        else:
            dominant = "SPEAKER_0"

        merged.append(
            {
                "speaker_id": dominant,
                "start": t_start,
                "end": t_end,
                "text": t_seg["text"],
            }
        )

    # Consolidate consecutive segments from same speaker (optional clean-up)
    consolidated = _consolidate(merged)
    logger.info(
        "Merge complete: %d raw → %d consolidated segments",
        len(merged),
        len(consolidated),
    )
    return consolidated


def _consolidate(segments: list, max_gap_s: float = 0.5) -> list:
    """Merge adjacent segments from same speaker if gap is small."""
    if not segments:
        return segments
    out = [segments[0].copy()]
    for seg in segments[1:]:
        prev = out[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker_id"] == prev["speaker_id"] and gap <= max_gap_s:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            out.append(seg.copy())
    return out


async def transcribe(
    audio_path: str,
    source_language: Optional[str] = None,
) -> list:
    """
    Async transcription.
    Returns raw Whisper segments (without speaker labels).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        functools.partial(_run_transcription, audio_path, source_language),
    )


async def merge_segments(
    transcription: list,
    diarization: list,
) -> List[dict]:
    """
    Merge transcription + diarization synchronously (CPU-light, no executor needed).
    Returns speaker-labeled segments:
      [{"speaker_id", "start", "end", "text"}, ...]
    """
    return _merge_diarization_transcription(transcription, diarization)
