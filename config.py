from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # S3 / MinIO
    S3_ENDPOINT_URL: Optional[str] = None  # None → real AWS S3
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "dubbing-output"
    S3_REGION: str = "us-east-1"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # ElevenLabs
    ELEVENLABS_API_KEY: str = ""

    # Hugging Face (pyannote)
    HF_TOKEN: str = ""

    # Whisper
    WHISPER_MODEL_SIZE: str = "medium"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Application
    UPLOAD_DIR: str = "/tmp/dubbing/uploads"
    OUTPUT_DIR: str = "/tmp/dubbing/outputs"
    MAX_UPLOAD_SIZE_MB: int = 500
    JOB_TTL_SECONDS: int = 604800  # 7 days

    # Logging
    LOG_LEVEL: str = "INFO"

    def ensure_dirs(self) -> None:
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
