"""
Core Dubbing Pipeline
Orchestrates all 9 steps of the dubbing workflow with:
  - Per-step progress updates (stored in Redis)
  - Structured logging
  - Full error handling and cleanup
  - Async throughout
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Optional

import redis.asyncio as aioredis

from config import settings
from models.schemas import JobStatus, PipelineStep
from services import (
    audio_stitcher,
    diarization,
    transcription,
    translation,
    tts,
    video_processor,
    voice_mapper,
)
from storage.s3_client import s3_client

logger = logging.getLogger(__name__)

# Progress checkpoints per step (cumulative %)
STEP_PROGRESS: Dict[str, int] = {
    PipelineStep.AUDIO_EXTRACTION:    10,
    PipelineStep.DIARIZATION:         22,
    PipelineStep.TRANSCRIPTION:       36,
    PipelineStep.SEGMENT_MERGE:       44,
    PipelineStep.TRANSLATION:         56,
    PipelineStep.VOICE_MAPPING:       62,
    PipelineStep.TTS_GENERATION:      78,
    PipelineStep.AUDIO_STITCHING:     87,
    PipelineStep.VIDEO_RECONSTRUCTION:94,
    PipelineStep.UPLOAD:             100,
}


async def _update_job(
    redis: aioredis.Redis,
    job_id: str,
    step: str,
    progress: int,
    **extra,
) -> None:
    """Persist job state to Redis."""
    fields = {
        "status": JobStatus.PROCESSING,
        "current_step": step,
        "progress": progress,
        **extra,
    }
    await redis.hset(f"job:{job_id}", mapping={k: str(v) for k, v in fields.items()})
    logger.info("[%s] step=%s progress=%d%%", job_id, step, progress)


async def _fail_job(redis: aioredis.Redis, job_id: str, error: str) -> None:
    await redis.hset(
        f"job:{job_id}",
        mapping={
            "status": JobStatus.FAILED,
            "error": error,
            "completed_at": datetime.utcnow().isoformat(),
        },
    )
    logger.error("[%s] FAILED: %s", job_id, error)


async def process_video_pipeline(
    job_id: str,
    input_path: str,
    target_language: str,
    voice_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """
    Full dubbing pipeline. Called as a FastAPI BackgroundTask.

    Steps:
      1  Extract audio
      2  Speaker diarization
      3  Speech-to-text (Whisper)
      4  Merge diarization + transcription
      5  Translate segments
      6  Build voice map
      7  Generate TTS per segment
      8  Stitch audio timeline
      9  Reconstruct dubbed video
      10 Upload to S3 & return URL
    """
    redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    work_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # ------------------------------------------------------------------ #
        # STEP 1: Extract audio
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.AUDIO_EXTRACTION, 5)
        audio_path = os.path.join(work_dir, "audio.wav")
        await video_processor.extract_audio(input_path, audio_path)
        video_duration = await video_processor.get_video_duration(input_path)
        await _update_job(
            redis, job_id,
            PipelineStep.AUDIO_EXTRACTION,
            STEP_PROGRESS[PipelineStep.AUDIO_EXTRACTION],
        )

        # ------------------------------------------------------------------ #
        # STEP 2: Speaker diarization
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.DIARIZATION, 14)
        diarization_segments = await diarization.diarize(audio_path)
        speakers = {s["speaker_id"] for s in diarization_segments}
        await _update_job(
            redis, job_id,
            PipelineStep.DIARIZATION,
            STEP_PROGRESS[PipelineStep.DIARIZATION],
            speakers_detected=len(speakers),
        )
        logger.info("[%s] Detected %d speakers: %s", job_id, len(speakers), speakers)

        # ------------------------------------------------------------------ #
        # STEP 3: Speech-to-text
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.TRANSCRIPTION, 26)
        whisper_segments = await transcription.transcribe(audio_path)
        await _update_job(
            redis, job_id,
            PipelineStep.TRANSCRIPTION,
            STEP_PROGRESS[PipelineStep.TRANSCRIPTION],
        )

        # ------------------------------------------------------------------ #
        # STEP 4: Merge diarization + transcription
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.SEGMENT_MERGE, 40)
        merged_segments = await transcription.merge_segments(whisper_segments, diarization_segments)
        await _update_job(
            redis, job_id,
            PipelineStep.SEGMENT_MERGE,
            STEP_PROGRESS[PipelineStep.SEGMENT_MERGE],
            total_segments=len(merged_segments),
        )
        logger.info("[%s] Merged into %d speaker-labeled segments", job_id, len(merged_segments))

        # ------------------------------------------------------------------ #
        # STEP 5: Translate
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.TRANSLATION, 48)
        translated_segments = await translation.translate_segments(
            merged_segments, target_language
        )
        await _update_job(
            redis, job_id,
            PipelineStep.TRANSLATION,
            STEP_PROGRESS[PipelineStep.TRANSLATION],
        )

        # ------------------------------------------------------------------ #
        # STEP 6: Build voice map
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.VOICE_MAPPING, 59)
        voice_map = await voice_mapper.build_voice_map(
            segments=translated_segments,
            audio_path=audio_path,
            user_mapping=voice_mapping,
            detect_gender=True,
        )
        await _update_job(
            redis, job_id,
            PipelineStep.VOICE_MAPPING,
            STEP_PROGRESS[PipelineStep.VOICE_MAPPING],
        )
        logger.info("[%s] Voice map: %s", job_id, voice_map)

        # ------------------------------------------------------------------ #
        # STEP 7: Generate TTS per segment
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.TTS_GENERATION, 64)
        tts_dir = os.path.join(work_dir, "tts")
        tts_segments = await tts.generate_all_tts(
            segments=translated_segments,
            voice_map=voice_map,
            output_dir=tts_dir,
            concurrency=3,
        )
        await _update_job(
            redis, job_id,
            PipelineStep.TTS_GENERATION,
            STEP_PROGRESS[PipelineStep.TTS_GENERATION],
        )

        # ------------------------------------------------------------------ #
        # STEP 8: Stitch audio timeline
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.AUDIO_STITCHING, 83)
        dubbed_audio_path = os.path.join(work_dir, "dubbed_audio.wav")

        # Run in executor (pydub is synchronous & CPU-bound)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: audio_stitcher.stitch_audio(
                segments=tts_segments,
                total_duration_s=video_duration,
                output_path=dubbed_audio_path,
                work_dir=work_dir,
            ),
        )
        await _update_job(
            redis, job_id,
            PipelineStep.AUDIO_STITCHING,
            STEP_PROGRESS[PipelineStep.AUDIO_STITCHING],
        )

        # ------------------------------------------------------------------ #
        # STEP 9: Reconstruct video
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.VIDEO_RECONSTRUCTION, 91)
        input_ext = os.path.splitext(input_path)[1]
        output_video_path = os.path.join(work_dir, f"dubbed_output{input_ext}")
        await video_processor.replace_audio(input_path, dubbed_audio_path, output_video_path)
        await _update_job(
            redis, job_id,
            PipelineStep.VIDEO_RECONSTRUCTION,
            STEP_PROGRESS[PipelineStep.VIDEO_RECONSTRUCTION],
        )

        # ------------------------------------------------------------------ #
        # STEP 10: Upload to S3
        # ------------------------------------------------------------------ #
        await _update_job(redis, job_id, PipelineStep.UPLOAD, 96)
        await s3_client.ensure_bucket()
        object_key = f"jobs/{job_id}/dubbed_output{input_ext}"
        download_url = await s3_client.upload_and_sign(
            local_path=output_video_path,
            object_key=object_key,
            expiry=settings.JOB_TTL_SECONDS,
        )

        # ------------------------------------------------------------------ #
        # Mark complete
        # ------------------------------------------------------------------ #
        await redis.hset(
            f"job:{job_id}",
            mapping={
                "status": JobStatus.COMPLETED,
                "progress": "100",
                "current_step": PipelineStep.UPLOAD,
                "output_url": download_url,
                "completed_at": datetime.utcnow().isoformat(),
                "error": "",
            },
        )
        logger.info("[%s] Pipeline complete. URL: %s", job_id, download_url)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed at step %s", job_id, exc)
        await _fail_job(redis, job_id, str(exc))
    finally:
        await redis.aclose()
        # Optional: clean up work directory (comment out to keep files for debug)
        # shutil.rmtree(work_dir, ignore_errors=True)
