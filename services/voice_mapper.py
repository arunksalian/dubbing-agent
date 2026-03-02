"""
Voice Mapper Service
Auto-assigns distinct ElevenLabs voices to each detected speaker.
Supports:
  1. User-provided voice mapping (speaker_id → voice name/id)
  2. Automatic gender detection from audio pitch analysis
  3. Round-robin fallback from curated voice pool
"""
from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import settings

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# Curated ElevenLabs voice pool (name → voice_id, gender)
# These are standard pre-made voices available on all ElevenLabs plans.
# ---------------------------------------------------------------------------
VOICE_POOL: List[Dict] = [
    {"name": "Rachel",   "voice_id": "21m00Tcm4TlvDq8ikWAM", "gender": "female"},
    {"name": "Adam",     "voice_id": "pNInz6obpgDQGcFmaJgB", "gender": "male"},
    {"name": "Antoni",   "voice_id": "ErXwobaYiN019PkySvjV", "gender": "male"},
    {"name": "Bella",    "voice_id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},
    {"name": "Callum",   "voice_id": "N2lVS1w4EtoT3dr4eOWO", "gender": "male"},
    {"name": "Charlotte","voice_id": "XB0fDUnXU5powFXDhCwa", "gender": "female"},
    {"name": "Clyde",    "voice_id": "2EiwWnXFnvU5JabPnv8n", "gender": "male"},
    {"name": "Domi",     "voice_id": "AZnzlk1XvdvUeBnXmlld", "gender": "female"},
    {"name": "Dorothy",  "voice_id": "ThT5KcBeYPX3keUQqHPh", "gender": "female"},
    {"name": "Ethan",    "voice_id": "g5CIjZEefAph4nQFvHAz", "gender": "male"},
    {"name": "Freya",    "voice_id": "jsCqWAovK2LkecY7zXl4", "gender": "female"},
    {"name": "Harry",    "voice_id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male"},
    {"name": "Josh",     "voice_id": "TxGEqnHWrfWFTfGW9XjX", "gender": "male"},
    {"name": "Lily",     "voice_id": "pFZP5JQG7iQjIQuC4Bku", "gender": "female"},
    {"name": "Matilda",  "voice_id": "XrExE9yKIg1WjnnlVkGX", "gender": "female"},
    {"name": "Nicole",   "voice_id": "piTKgcLEGmPE4e6mEKli", "gender": "female"},
    {"name": "Ryan",     "voice_id": "wViXBPUzp2ZZixB1xQuM", "gender": "male"},
    {"name": "Sam",      "voice_id": "yoZ06aMxZJJ28mfd3POQ", "gender": "male"},
    {"name": "Sarah",    "voice_id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},
    {"name": "Thomas",   "voice_id": "GBv7mTt0atIp3Br8iCZE", "gender": "male"},
]

# Build fast lookup maps
_NAME_TO_VOICE = {v["name"].lower(): v for v in VOICE_POOL}
_ID_TO_VOICE   = {v["voice_id"]: v  for v in VOICE_POOL}


def _resolve_voice_id(name_or_id: str) -> Optional[str]:
    """Resolve a voice name or raw ID to an ElevenLabs voice_id."""
    if name_or_id in _ID_TO_VOICE:
        return name_or_id
    voice = _NAME_TO_VOICE.get(name_or_id.lower())
    if voice:
        return voice["voice_id"]
    return None


# ---------------------------------------------------------------------------
# Gender detection via pitch analysis (librosa)
# ---------------------------------------------------------------------------

def _detect_gender_from_audio(
    audio_path: str,
    start: float,
    end: float,
) -> str:
    """
    Heuristic gender detection using fundamental frequency (F0).
    Average female F0 ≈ 165–255 Hz; male ≈ 85–155 Hz.
    Threshold: 165 Hz.
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=(end - start))
        if len(y) < sr * 0.3:
            return "unknown"

        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) == 0:
            return "unknown"

        mean_f0 = float(np.median(voiced_f0))
        gender = "female" if mean_f0 > 165 else "male"
        logger.debug("F0=%.1f Hz → %s", mean_f0, gender)
        return gender
    except Exception as exc:
        logger.warning("Gender detection failed: %s", exc)
        return "unknown"


async def _detect_genders(
    speakers: List[str],
    segments: List[dict],
    audio_path: str,
) -> Dict[str, str]:
    """Detect gender per speaker from a sample segment."""
    loop = asyncio.get_event_loop()
    speaker_gender: Dict[str, str] = {}

    # Use first segment per speaker for analysis
    sample_per_speaker: Dict[str, dict] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        if sid not in sample_per_speaker:
            sample_per_speaker[sid] = seg

    async def _detect_one(speaker_id: str, seg: dict) -> Tuple[str, str]:
        gender = await loop.run_in_executor(
            _executor,
            functools.partial(
                _detect_gender_from_audio, audio_path, seg["start"], seg["end"]
            ),
        )
        return speaker_id, gender

    tasks = [
        _detect_one(sid, sample_per_speaker[sid])
        for sid in speakers
        if sid in sample_per_speaker
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    speaker_gender = dict(results)
    logger.info("Speaker genders detected: %s", speaker_gender)
    return speaker_gender


def _assign_voices(
    speakers: List[str],
    speaker_genders: Dict[str, str],
    user_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Assign an ElevenLabs voice_id to every speaker.

    Priority:
      1. User-provided voice mapping (name or raw ID)
      2. Gender-matched pool selection (ensures unique voices)
      3. Round-robin fallback
    """
    assignment: Dict[str, str] = {}  # speaker_id → voice_id

    male_voices   = [v for v in VOICE_POOL if v["gender"] == "male"]
    female_voices = [v for v in VOICE_POOL if v["gender"] == "female"]
    male_idx = female_idx = 0

    for speaker in sorted(speakers):
        # 1. User override
        if user_mapping and speaker in user_mapping:
            resolved = _resolve_voice_id(user_mapping[speaker])
            if resolved:
                assignment[speaker] = resolved
                logger.info("Speaker %s → user voice %s", speaker, resolved)
                continue
            else:
                logger.warning(
                    "Unknown voice '%s' for %s; falling back to auto.",
                    user_mapping[speaker], speaker,
                )

        # 2. Gender-matched auto-assign
        gender = speaker_genders.get(speaker, "unknown")
        if gender == "female" and female_voices:
            voice = female_voices[female_idx % len(female_voices)]
            female_idx += 1
        elif gender == "male" and male_voices:
            voice = male_voices[male_idx % len(male_voices)]
            male_idx += 1
        else:
            # unknown gender — alternate male/female
            pool = VOICE_POOL
            idx = len(assignment)
            voice = pool[idx % len(pool)]

        assignment[speaker] = voice["voice_id"]
        logger.info(
            "Speaker %s (%s) → voice %s (%s)",
            speaker, gender, voice["name"], voice["voice_id"],
        )

    return assignment


async def build_voice_map(
    segments: List[dict],
    audio_path: str,
    user_mapping: Optional[Dict[str, str]] = None,
    detect_gender: bool = True,
) -> Dict[str, str]:
    """
    Public entry point.

    Args:
        segments:     merged speaker+transcription segments
        audio_path:   path to extracted WAV for gender analysis
        user_mapping: optional {"SPEAKER_0": "Rachel", ...}
        detect_gender: run pitch-based gender heuristic

    Returns:
        {"SPEAKER_0": "21m00Tcm4TlvDq8ikWAM", "SPEAKER_1": "pNInz6obpgDQGcFmaJgB", ...}
    """
    speakers = sorted({seg["speaker_id"] for seg in segments})
    logger.info("Building voice map for %d speakers: %s", len(speakers), speakers)

    if detect_gender and audio_path:
        speaker_genders = await _detect_genders(speakers, segments, audio_path)
    else:
        speaker_genders = {s: "unknown" for s in speakers}

    voice_map = _assign_voices(speakers, speaker_genders, user_mapping)
    return voice_map
