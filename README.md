# AI Video Dubbing Agent

A production-grade, multi-speaker video dubbing workflow built with FastAPI. Upload a video, choose a target language, and get back a fully dubbed video with distinct voices per speaker — automatically.

---

## Features

- **Multi-speaker support** — detects 2–10 speakers via pyannote.audio diarization
- **Automatic voice assignment** — assigns unique ElevenLabs voices per speaker with gender detection
- **Custom voice mapping** — override any speaker with your chosen voice
- **Timing alignment** — TTS segments are time-stretched to fit original speech windows
- **Async pipeline** — fully non-blocking with per-step progress tracking (0–100%)
- **S3-compatible storage** — works with AWS S3 or MinIO; returns a presigned download URL
- **Retry logic** — exponential backoff on all external API calls (ElevenLabs, OpenAI)
- **Docker-ready** — single `docker compose up` gets the full stack running

---

## Architecture

```
CLIENT
  │
  │  POST /upload
  ▼
FastAPI  ──► Redis (job state)
  │
  ▼  BackgroundTask
Pipeline
  ①  Extract audio          (FFmpeg)           10%
  ②  Speaker diarization    (pyannote.audio)    22%
  ③  Speech-to-text         (faster-whisper)    36%
  ④  Merge segments                             44%
  ⑤  Translate              (GPT-4o)            56%
  ⑥  Voice mapping          (ElevenLabs)        62%
  ⑦  TTS generation         (ElevenLabs)        78%
  ⑧  Audio stitching        (pydub + FFmpeg)    87%
  ⑨  Rebuild video          (FFmpeg)            94%
  ⑩  Upload & sign URL      (S3 / MinIO)       100%
```

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Job state | Redis |
| Diarization | pyannote.audio 3.x |
| Speech-to-text | faster-whisper |
| Translation | OpenAI GPT-4o |
| TTS | ElevenLabs multilingual v2 |
| Audio processing | pydub + FFmpeg |
| Storage | AWS S3 / MinIO (aioboto3) |
| Retries | tenacity |
| Logging | structlog |

---

## Project Structure

```
dubbing-agent/
├── main.py                   # FastAPI app + endpoints
├── config.py                 # Environment configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        # API + Redis + MinIO
├── .env.example
├── models/
│   └── schemas.py            # Pydantic models
├── storage/
│   └── s3_client.py          # Async S3/MinIO client
├── services/
│   ├── diarization.py        # Speaker detection
│   ├── transcription.py      # Whisper STT + merge
│   ├── translation.py        # GPT-4o translation
│   ├── voice_mapper.py       # Voice assignment + gender detection
│   ├── tts.py                # ElevenLabs TTS generation
│   ├── audio_stitcher.py     # Timeline reconstruction
│   └── video_processor.py   # FFmpeg operations
└── worker/
    └── pipeline.py           # 10-step pipeline orchestration
```

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

```env
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
HF_TOKEN=hf_...          # Hugging Face token for pyannote model download
```

### 2. Run with Docker

```bash
docker compose up --build
```

This starts:
- **API** at `http://localhost:8000`
- **Redis** at `localhost:6379`
- **MinIO** at `http://localhost:9000` (console at `http://localhost:9001`)

### 3. Run locally (without Docker)

```bash
pip install -r requirements.txt
# Start Redis separately, then:
uvicorn main:app --reload
```

---

## API Reference

### `POST /upload`

Upload a video and start dubbing.

```bash
curl -X POST "http://localhost:8000/upload?target_language=hi" \
  -F "file=@interview.mp4"
```

With optional voice mapping:

```bash
curl -X POST "http://localhost:8000/upload?target_language=es" \
  -F "file=@interview.mp4" \
  -F 'voice_mapping={"SPEAKER_0":"Rachel","SPEAKER_1":"Adam"}'
```

**Response**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "message": "Video uploaded. Dubbing pipeline started."
}
```

---

### `GET /status/{job_id}`

Poll progress of a dubbing job.

```bash
curl http://localhost:8000/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "status": "processing",
  "progress": 56,
  "current_step": "translation",
  "speakers_detected": 3,
  "total_segments": 42,
  "error": null,
  "output_url": null,
  "created_at": "2026-03-02T10:00:00Z",
  "completed_at": null
}
```

`status` values: `queued` → `processing` → `completed` | `failed`

---

### `GET /download/{job_id}`

Get the presigned download URL (only available when `status == "completed"`).

```bash
curl http://localhost:8000/download/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response**
```json
{
  "job_id": "a1b2c3d4-...",
  "download_url": "https://...",
  "expires_in": 3600
}
```

---

### `GET /voices`

List all available ElevenLabs voices for manual mapping.

```bash
curl http://localhost:8000/voices
```

**Response**
```json
{
  "voices": [
    {"name": "Rachel", "voice_id": "21m00Tcm4TlvDq8ikWAM", "gender": "female"},
    {"name": "Adam",   "voice_id": "pNInz6obpgDQGcFmaJgB", "gender": "male"}
  ]
}
```

---

### `GET /health`

Kubernetes liveness / readiness probe.

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"1.0.0","timestamp":"..."}
```

---

## Supported Languages

Target language is specified as a BCP-47 code:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `hi` | Hindi | `ja` | Japanese |
| `es` | Spanish | `ko` | Korean |
| `fr` | French | `zh` | Chinese |
| `de` | German | `ar` | Arabic |
| `pt` | Portuguese | `ru` | Russian |
| `it` | Italian | `tr` | Turkish |

Full list of supported codes is in `services/translation.py`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `S3_ENDPOINT_URL` | _(blank = AWS S3)_ | Override for MinIO |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET_NAME` | `dubbing-output` | Target S3 bucket |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Model for translation |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key |
| `HF_TOKEN` | — | Hugging Face token (pyannote) |
| `WHISPER_MODEL_SIZE` | `medium` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` / `float16` / `float32` |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max video upload size |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Scaling

The pipeline runs as a FastAPI `BackgroundTask` by default. To scale horizontally, move `worker/pipeline.py` to a **Celery** task:

```python
# worker/celery_app.py
from celery import Celery
app = Celery("dubbing", broker=REDIS_URL, backend=REDIS_URL)

@app.task
def run_pipeline(job_id, input_path, target_language, voice_mapping):
    asyncio.run(process_video_pipeline(job_id, input_path, target_language, voice_mapping))
```

Then deploy API pods and worker pods independently and scale each as needed.

---

## License

MIT
