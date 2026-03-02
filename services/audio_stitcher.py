"""
Audio Stitcher Service
Reconstructs the full dubbed audio track by:
  1. Creating a silent base track matching the original video duration
  2. Time-stretching each TTS segment to fit its original time slot
  3. Overlaying each TTS segment at the correct timestamp
  4. Exporting the final WAV file
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

# Maximum tempo ratio for ffmpeg atempo (range: 0.5–2.0 per filter)
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

# Tolerance: if TTS duration differs < 5% from target, no stretching needed
STRETCH_TOLERANCE = 0.05


def _build_atempo_filter(ratio: float) -> str:
    """
    Build an ffmpeg atempo filter chain.
    atempo only accepts 0.5–2.0; chain filters for values outside this range.

    Examples:
      ratio=1.5  → "atempo=1.5"
      ratio=3.0  → "atempo=2.0,atempo=1.5"
      ratio=0.25 → "atempo=0.5,atempo=0.5"
    """
    filters = []
    r = ratio
    while r > ATEMPO_MAX:
        filters.append(f"atempo={ATEMPO_MAX}")
        r /= ATEMPO_MAX
    while r < ATEMPO_MIN:
        filters.append(f"atempo={ATEMPO_MIN}")
        r /= ATEMPO_MIN
    filters.append(f"atempo={r:.6f}")
    return ",".join(filters)


def _stretch_audio(
    input_path: str,
    output_path: str,
    target_duration_s: float,
    source_duration_s: float,
) -> str:
    """
    Time-stretch audio from source_duration_s to target_duration_s using FFmpeg atempo.
    Returns output_path.
    """
    ratio = source_duration_s / target_duration_s
    ratio = max(ATEMPO_MIN ** 2, min(ATEMPO_MAX ** 2, ratio))  # clamp to 2-stage limits

    rel_diff = abs(source_duration_s - target_duration_s) / max(target_duration_s, 0.001)
    if rel_diff < STRETCH_TOLERANCE:
        # No meaningful stretching needed — copy as-is
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    atempo_filter = _build_atempo_filter(ratio)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", atempo_filter,
        "-ar", "44100",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "atempo stretch failed (ratio=%.3f): %s — using original",
            ratio, result.stderr[-300:],
        )
        import shutil
        shutil.copy2(input_path, output_path)
    else:
        logger.debug(
            "Stretched %.2fs → %.2fs (ratio=%.3f)", source_duration_s, target_duration_s, ratio
        )
    return output_path


def _get_audio_duration(path: str) -> float:
    """Use ffprobe to measure audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def stitch_audio(
    segments: List[dict],
    total_duration_s: float,
    output_path: str,
    work_dir: str,
    sample_rate: int = 44100,
) -> str:
    """
    Stitch all TTS segments into a single WAV file.

    Strategy:
      - Create a silent base track of total_duration_s
      - For each segment: stretch TTS to fit the slot, then overlay at start_time
      - Export as 44100 Hz stereo WAV

    Args:
        segments:         list of translated segments with 'tts_audio_path', 'start', 'end'
        total_duration_s: duration of the original video in seconds
        output_path:      path for the output WAV file
        work_dir:         temp directory for intermediate files

    Returns:
        output_path
    """
    from pydub import AudioSegment as PyAudio

    # Create silent base track
    total_ms = int(total_duration_s * 1000)
    base = PyAudio.silent(duration=total_ms, frame_rate=sample_rate)
    base = base.set_channels(2)

    stretch_dir = os.path.join(work_dir, "stretched")
    os.makedirs(stretch_dir, exist_ok=True)

    placed = 0
    for seg in segments:
        tts_path: Optional[str] = seg.get("tts_audio_path")
        if not tts_path or not os.path.exists(tts_path):
            logger.warning(
                "Skipping segment [%.2f–%.2f] — no TTS audio", seg["start"], seg["end"]
            )
            continue

        start_ms = int(seg["start"] * 1000)
        target_dur_s = seg["end"] - seg["start"]
        source_dur_s = seg.get("tts_duration") or _get_audio_duration(tts_path)

        # Build stretched path
        basename = os.path.splitext(os.path.basename(tts_path))[0]
        stretched_path = os.path.join(stretch_dir, f"{basename}_stretched.wav")

        _stretch_audio(tts_path, stretched_path, target_dur_s, source_dur_s)

        # Load and normalise
        try:
            tts_audio = PyAudio.from_file(stretched_path)
            tts_audio = tts_audio.set_frame_rate(sample_rate).set_channels(2)

            # Safety: don't extend beyond base track length
            end_ms = start_ms + len(tts_audio)
            if end_ms > total_ms:
                overflow_ms = end_ms - total_ms
                logger.debug(
                    "Segment at %.2fs overflows by %dms — trimming", seg["start"], overflow_ms
                )
                tts_audio = tts_audio[: total_ms - start_ms]

            base = base.overlay(tts_audio, position=start_ms)
            placed += 1
        except Exception as exc:
            logger.error(
                "Failed to overlay segment [%.2f–%.2f]: %s", seg["start"], seg["end"], exc
            )

    logger.info("Overlaid %d/%d segments onto base track (%.1fs)", placed, len(segments), total_duration_s)

    # Export to WAV
    base.export(output_path, format="wav")
    logger.info("Dubbed audio written to %s", output_path)
    return output_path
