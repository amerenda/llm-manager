"""
Piper TTS Agent — runs on arm64 (rpi5) k8s nodes.
Wraps the piper TTS binary, exposes HTTP API + Prometheus metrics.
Port 8091.
"""
import asyncio
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

NODE = socket.gethostname()
PORT = int(os.environ.get("PORT", 8091))
AGENT_PSK = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "")
POD_IP = os.environ.get("POD_IP", "")
PIPER_VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "/voices")
PIPER_DEFAULT_VOICE = os.environ.get("PIPER_DEFAULT_VOICE", "en_US-lessac-medium")

AGENT_ADDRESS = os.environ.get("AGENT_ADDRESS") or f"http://{POD_IP}:{PORT}"

_RUNNER_ID: Optional[int] = None

# ── Prometheus metrics ─────────────────────────────────────────────────────────

requests_total = Counter(
    "piper_agent_requests_total",
    "Total requests",
    ["node", "endpoint"],
)
request_duration = Histogram(
    "piper_agent_request_duration_seconds",
    "Request duration seconds",
    ["node", "endpoint"],
)


# ── Voice helpers ──────────────────────────────────────────────────────────────

def _list_voices() -> list[str]:
    """Return sorted list of available voice names (filenames without .onnx)."""
    voices_dir = Path(PIPER_VOICES_DIR)
    if not voices_dir.exists():
        return []
    return sorted(p.stem for p in voices_dir.iterdir() if p.suffix == ".onnx")


def _resolve_voice(voice: Optional[str]) -> str:
    """Return the given voice if its .onnx exists, otherwise fall back to default."""
    if voice:
        onnx = Path(PIPER_VOICES_DIR) / f"{voice}.onnx"
        if onnx.exists():
            return voice
        logger.warning("Voice %s not found, falling back to default", voice)
    return PIPER_DEFAULT_VOICE


# ── Self-registration ──────────────────────────────────────────────────────────

def _build_capabilities() -> dict:
    return {
        "tts": True,
        "stt": False,
        "gpu": False,
        "voices": _list_voices(),
        "default_voice": PIPER_DEFAULT_VOICE,
    }


async def _register():
    global _RUNNER_ID
    if not BACKEND_URL or not AGENT_ADDRESS:
        return
    try:
        caps = _build_capabilities()
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"{BACKEND_URL}/api/runners/register",
                json={
                    "hostname": NODE,
                    "address": AGENT_ADDRESS,
                    "port": PORT,
                    "capabilities": caps,
                },
                headers={"X-Agent-PSK": AGENT_PSK} if AGENT_PSK else {},
            )
            r.raise_for_status()
            _RUNNER_ID = r.json()["runner_id"]
            logger.info("Registered with backend as runner_id=%d", _RUNNER_ID)
    except Exception as e:
        logger.warning("Could not register with backend at startup: %s", e)


async def _heartbeat_loop():
    while True:
        await asyncio.sleep(30)
        if not _RUNNER_ID or not BACKEND_URL:
            continue
        try:
            caps = _build_capabilities()
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(
                    f"{BACKEND_URL}/api/runners/heartbeat",
                    json={"runner_id": _RUNNER_ID, "capabilities": caps},
                    headers={"X-Agent-PSK": AGENT_PSK} if AGENT_PSK else {},
                )
        except Exception as e:
            logger.debug("Heartbeat failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _register()
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    yield
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Piper TTS Agent", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def psk_auth(request: Request, call_next):
    # Health and metrics are exempt from PSK (k8s probes, Prometheus)
    if request.url.path in ("/health", "/metrics"):
        return await call_next(request)
    if AGENT_PSK:
        psk = request.headers.get("X-Agent-PSK", "")
        if psk != AGENT_PSK:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing PSK"})
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return StreamingResponse(
        iter([generate_latest().decode()]),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/v1/status")
async def status():
    t0 = time.time()
    voices = _list_voices()
    requests_total.labels(node=NODE, endpoint="/v1/status").inc()
    request_duration.labels(node=NODE, endpoint="/v1/status").observe(time.time() - t0)
    return {
        "node": NODE,
        "tts": True,
        "voices": voices,
        "default_voice": PIPER_DEFAULT_VOICE,
    }


@app.get("/v1/voices")
async def list_voices():
    t0 = time.time()
    voices = _list_voices()
    requests_total.labels(node=NODE, endpoint="/v1/voices").inc()
    request_duration.labels(node=NODE, endpoint="/v1/voices").observe(time.time() - t0)
    return {"voices": voices}


class SpeechRequest(BaseModel):
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    t0 = time.time()
    requests_total.labels(node=NODE, endpoint="/v1/audio/speech").inc()

    voice = _resolve_voice(req.voice)
    model_path = Path(PIPER_VOICES_DIR) / f"{voice}.onnx"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice model not found: {voice}. Available: {_list_voices()}",
        )

    async def _stream_audio():
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", str(model_path),
                "--output_raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=req.input.encode("utf-8"))
            if proc.returncode != 0:
                logger.error("piper error: %s", stderr.decode(errors="replace"))
                raise HTTPException(status_code=500, detail="piper synthesis failed")

            # Prepend a minimal WAV header for 22050 Hz, mono, 16-bit PCM
            # (piper --output_raw produces raw 16-bit LE PCM at 22050 Hz mono by default)
            import struct
            num_channels = 1
            sample_rate = 22050
            bits_per_sample = 16
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8
            data_size = len(stdout)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + data_size,
                b"WAVE",
                b"fmt ",
                16,           # PCM chunk size
                1,            # PCM format
                num_channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                b"data",
                data_size,
            )
            yield header + stdout
        finally:
            request_duration.labels(node=NODE, endpoint="/v1/audio/speech").observe(
                time.time() - t0
            )

    return StreamingResponse(_stream_audio(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
