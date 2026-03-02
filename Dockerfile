# =============================================================================
# AI Video Dubbing Agent — Dockerfile
# Multi-stage build: system deps → Python deps → app
# =============================================================================

# ---- Stage 1: Base system with FFmpeg + audio libs -------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsoundfile1 \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Stage 2: Python dependencies (heavy layer, cached separately) ---------
FROM base AS builder

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model at build time (optional — saves cold-start time)
# ARG WHISPER_MODEL_SIZE=medium
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='int8')"

# ---- Stage 3: Final image --------------------------------------------------
FROM builder AS final

# Copy application source
COPY . /app

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /tmp/dubbing/uploads /tmp/dubbing/outputs && \
    chown -R appuser:appuser /app /tmp/dubbing

USER appuser

EXPOSE 8000

# Health check (Kubernetes probe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--log-level", "info"]
