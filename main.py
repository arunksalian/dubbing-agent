"""
AI Video Dubbing API
FastAPI application with upload, status, and download endpoints.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import aiofiles
import redis.asyncio as aioredis
import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models.schemas import (
    DownloadResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    UploadResponse,
)
from worker.pipeline import process_video_pipeline

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    )
)
logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
_redis: Optional[aioredis.Redis] = None
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    settings.ensure_dirs()
    _redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("startup", redis_url=settings.REDIS_URL)
    yield
    await _redis.aclose()
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Video Dubbing API",
    description=(
        "Production-grade multi-speaker video dubbing workflow. "
        "Supports 2–10 speakers with automatic gender detection and voice assignment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency: Redis client
# ---------------------------------------------------------------------------
async def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    return _redis


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a video for dubbing",
    tags=["Dubbing"],
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (.mp4 .mov .mkv .avi .webm)"),
    target_language: str = Query(
        ...,
        description="BCP-47 language code for target dub language (e.g. 'hi', 'es', 'fr', 'ja')",
        min_length=2,
        max_length=10,
    ),
    voice_mapping: Optional[str] = Query(
        None,
        description='Optional JSON: {"SPEAKER_0":"Rachel","SPEAKER_1":"Adam"}',
    ),
    redis: aioredis.Redis = Depends(get_redis),
) -> UploadResponse:
    """
    Upload a video file and start the dubbing pipeline.

    Returns a `job_id` to track progress via **GET /status/{job_id}**.
    """
    # -- Validate extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # -- Read & size-check (stream to avoid double-buffering for large files)
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Limit: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # -- Parse optional voice mapping
    parsed_voice_mapping: Optional[dict] = None
    if voice_mapping:
        try:
            parsed_voice_mapping = json.loads(voice_mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid voice_mapping JSON: {exc}")

    # -- Save uploaded file
    job_id = str(uuid.uuid4())
    input_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}{ext.lower()}")
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(content)

    # -- Persist initial job state
    job_record = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "progress": "0",
        "current_step": "",
        "input_path": input_path,
        "target_language": target_language,
        "voice_mapping": json.dumps(parsed_voice_mapping) if parsed_voice_mapping else "",
        "speakers_detected": "",
        "total_segments": "",
        "error": "",
        "output_url": "",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": "",
    }
    await redis.hset(f"job:{job_id}", mapping=job_record)
    await redis.expire(f"job:{job_id}", settings.JOB_TTL_SECONDS)

    # -- Enqueue background pipeline
    background_tasks.add_task(
        process_video_pipeline,
        job_id=job_id,
        input_path=input_path,
        target_language=target_language,
        voice_mapping=parsed_voice_mapping,
    )

    logger.info(
        "job_created",
        job_id=job_id,
        filename=file.filename,
        size_mb=round(size_mb, 2),
        target_language=target_language,
    )
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Video uploaded. Dubbing pipeline started.",
    )


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------
@app.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status and progress",
    tags=["Dubbing"],
)
async def get_status(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
) -> JobStatusResponse:
    """
    Poll the status of a dubbing job.

    `progress` is 0–100. When `status == "completed"`, the `output_url` is available.
    """
    data = await redis.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    def _int_or_none(val: str) -> Optional[int]:
        try:
            return int(val) if val else None
        except ValueError:
            return None

    return JobStatusResponse(
        job_id=job_id,
        status=data.get("status", JobStatus.QUEUED),
        progress=int(data.get("progress", 0)),
        current_step=data.get("current_step") or None,
        speakers_detected=_int_or_none(data.get("speakers_detected", "")),
        total_segments=_int_or_none(data.get("total_segments", "")),
        error=data.get("error") or None,
        output_url=data.get("output_url") or None,
        created_at=data.get("created_at") or None,
        completed_at=data.get("completed_at") or None,
    )


# ---------------------------------------------------------------------------
# GET /download/{job_id}
# ---------------------------------------------------------------------------
@app.get(
    "/download/{job_id}",
    response_model=DownloadResponse,
    summary="Get download URL for completed job",
    tags=["Dubbing"],
)
async def download_video(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
) -> DownloadResponse:
    """
    Returns a pre-signed S3 download URL for the dubbed video.
    Only available when `status == "completed"`.
    """
    data = await redis.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    status = data.get("status")
    if status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed. Current status: {status}",
        )

    url = data.get("output_url")
    if not url:
        raise HTTPException(status_code=500, detail="Output URL missing — internal error")

    return DownloadResponse(
        job_id=job_id,
        download_url=url,
        expires_in=3600,
    )


# ---------------------------------------------------------------------------
# GET /health  (Kubernetes liveness + readiness probe)
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check(redis: aioredis.Redis = Depends(get_redis)) -> HealthResponse:
    """Kubernetes liveness / readiness probe. Checks Redis connectivity."""
    try:
        await redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    if not redis_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "redis": "unreachable", "version": "1.0.0"},
        )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /voices  — helper endpoint to list available ElevenLabs voices
# ---------------------------------------------------------------------------
@app.get(
    "/voices",
    summary="List available ElevenLabs voices",
    tags=["Operations"],
)
async def list_voices():
    """Returns the curated voice pool used for automatic speaker assignment."""
    from services.voice_mapper import VOICE_POOL
    return {"voices": VOICE_POOL}
