from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(str, Enum):
    AUDIO_EXTRACTION = "audio_extraction"
    DIARIZATION = "diarization"
    TRANSCRIPTION = "transcription"
    SEGMENT_MERGE = "segment_merge"
    TRANSLATION = "translation"
    VOICE_MAPPING = "voice_mapping"
    TTS_GENERATION = "tts_generation"
    AUDIO_STITCHING = "audio_stitching"
    VIDEO_RECONSTRUCTION = "video_reconstruction"
    UPLOAD = "upload"


# ---------------------------------------------------------------------------
# Speaker Segments
# ---------------------------------------------------------------------------

class SpeakerSegment(BaseModel):
    speaker_id: str
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class TranslatedSegment(SpeakerSegment):
    translated_text: str
    voice_id: str
    tts_audio_path: Optional[str] = None
    tts_duration: Optional[float] = None


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100, description="Progress 0–100%")
    current_step: Optional[str] = None
    speakers_detected: Optional[int] = None
    total_segments: Optional[int] = None
    error: Optional[str] = None
    output_url: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class DownloadResponse(BaseModel):
    job_id: str
    download_url: str
    expires_in: int = Field(3600, description="URL validity in seconds")


class VoiceMappingRequest(BaseModel):
    """Optional user-provided speaker→voice mapping."""
    mapping: Dict[str, str] = Field(
        ...,
        example={"SPEAKER_0": "Rachel", "SPEAKER_1": "Adam"},
        description="Map speaker ID to ElevenLabs voice name or voice ID",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Internal pipeline models (stored in Redis as JSON)
# ---------------------------------------------------------------------------

class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    current_step: Optional[str] = None
    input_path: str
    target_language: str
    voice_mapping: Optional[Dict[str, str]] = None
    speakers_detected: Optional[int] = None
    total_segments: Optional[int] = None
    error: Optional[str] = None
    output_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
