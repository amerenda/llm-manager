"""
LLM Agent — runs on GPU hosts (NVIDIA or AMD).
Wraps Ollama and ComfyUI, exposes HTTPS API + Prometheus metrics.
Port 8090.  Self-signed TLS cert is generated on startup.
"""
import asyncio
import datetime
import hashlib
import ipaddress
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import psutil
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# ── GPU backend detection (NVIDIA via pynvml, AMD via amdsmi) ────────────────
_GPU_BACKEND = "none"  # "nvidia", "amd", or "none"

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_BACKEND = "nvidia"
except Exception:
    pass

if _GPU_BACKEND == "none":
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        _amd_handles = amdsmi.amdsmi_get_processor_handles()
        if _amd_handles:
            _GPU_BACKEND = "amd"
    except Exception:
        pass

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
COMFYUI_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", "/outputs")
COMFYUI_IMAGE = os.environ.get("COMFYUI_IMAGE", "murderbot-image-comfyui:latest")
COMFYUI_CONTAINER = os.environ.get("COMFYUI_CONTAINER", "comfyui")
NODE = socket.gethostname()

# Docker client for managing ComfyUI container
try:
    import docker as docker_lib
    _docker = docker_lib.from_env()
    _DOCKER_OK = True
except Exception:
    _docker = None
    _DOCKER_OK = False

# PSK auth + self-registration
AGENT_PSK = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
BACKEND_URL = os.environ.get("BACKEND_URL", "")


def _detect_host_ip() -> str:
    """Return the host's outbound IP — the interface used to reach the internet.
    Works correctly when the agent runs with network_mode: host.
    Falls back to hostname if detection fails.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostname()
    finally:
        s.close()


TLS_CERT_DIR = os.environ.get("TLS_CERT_DIR", "/data/tls")
_TLS_CERT_PATH = os.path.join(TLS_CERT_DIR, "cert.pem")
_TLS_KEY_PATH = os.path.join(TLS_CERT_DIR, "key.pem")
_TLS_IP_PATH = os.path.join(TLS_CERT_DIR, "ip.txt")


def _read_cert_pem() -> str:
    """Return the PEM-encoded certificate as a string."""
    with open(_TLS_CERT_PATH, "r") as f:
        return f.read()


def _cert_fingerprint() -> str:
    """Return SHA-256 fingerprint of the current TLS certificate."""
    with open(_TLS_CERT_PATH, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    return hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()


def _generate_self_signed_cert(ip_addr: str, cert_dir: str) -> None:
    """Generate a self-signed TLS certificate with the given IP as SAN."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"llm-agent-{ip_addr}"),
    ])
    san = x509.SubjectAlternativeName([
        x509.IPAddress(ipaddress.ip_address(ip_addr)),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )
    os.makedirs(cert_dir, exist_ok=True)
    with open(os.path.join(cert_dir, "cert.pem"), "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(os.path.join(cert_dir, "key.pem"), "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(os.path.join(cert_dir, "ip.txt"), "w") as f:
        f.write(ip_addr)
    logger.info("Generated self-signed TLS certificate for IP %s in %s", ip_addr, cert_dir)


def _ensure_tls_cert(ip_addr: str) -> None:
    """Ensure a valid TLS cert exists for the given IP, regenerating if needed."""
    need_regen = False
    if not os.path.exists(_TLS_CERT_PATH) or not os.path.exists(_TLS_KEY_PATH):
        need_regen = True
    elif os.path.exists(_TLS_IP_PATH):
        with open(_TLS_IP_PATH, "r") as f:
            stored_ip = f.read().strip()
        if stored_ip != ip_addr:
            logger.info("IP changed from %s to %s — regenerating TLS cert", stored_ip, ip_addr)
            need_regen = True
    else:
        need_regen = True

    # Check if cert is expiring within 30 days
    if not need_regen:
        try:
            with open(_TLS_CERT_PATH, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            days_left = (cert.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)).days
            if days_left < 30:
                logger.info("TLS cert expires in %d days — regenerating", days_left)
                need_regen = True
        except Exception:
            need_regen = True

    if need_regen:
        _generate_self_signed_cert(ip_addr, TLS_CERT_DIR)


_DETECTED_IP = _detect_host_ip()
_ensure_tls_cert(_DETECTED_IP)

AGENT_ADDRESS = os.environ.get("AGENT_ADDRESS") or f"https://{_DETECTED_IP}:8090"

_RUNNER_ID: int | None = None

# ── Prometheus metrics ────────────────────────────────────────────────────────

gpu_vram_used = Gauge("llm_agent_gpu_vram_used_bytes", "GPU VRAM used bytes", ["node"])
gpu_vram_total = Gauge("llm_agent_gpu_vram_total_bytes", "GPU VRAM total bytes", ["node"])
gpu_vram_util = Gauge("llm_agent_gpu_vram_utilization_pct", "GPU VRAM utilization %", ["node"])
cpu_usage = Gauge("llm_agent_cpu_usage_pct", "CPU usage %", ["node"])
mem_used = Gauge("llm_agent_memory_used_bytes", "Memory used bytes", ["node"])
mem_total = Gauge("llm_agent_memory_total_bytes", "Memory total bytes", ["node"])
ollama_models_loaded = Gauge("llm_agent_ollama_models_loaded", "Ollama loaded models count", ["node"])
requests_total = Counter(
    "llm_agent_requests_total", "Total requests", ["node", "endpoint", "model"]
)
request_duration = Histogram(
    "llm_agent_request_duration_seconds",
    "Request duration seconds",
    ["node", "endpoint", "model"],
)
comfyui_running_gauge = Gauge("llm_agent_comfyui_running", "ComfyUI running (0/1)", ["node"])

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _RUNNER_ID
    if BACKEND_URL and AGENT_ADDRESS:
        try:
            caps = await _build_capabilities()
            caps["tls_cert"] = _read_cert_pem()
            caps["tls_fingerprint"] = _cert_fingerprint()
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(
                    f"{BACKEND_URL}/api/runners/register",
                    json={
                        "hostname": NODE,
                        "address": AGENT_ADDRESS,
                        "port": int(AGENT_ADDRESS.rsplit(":", 1)[-1]) if ":" in AGENT_ADDRESS.rsplit("/", 1)[-1] else 8090,
                        "capabilities": caps,
                    },
                    headers={"X-Agent-PSK": AGENT_PSK} if AGENT_PSK else {},
                )
                r.raise_for_status()
                _RUNNER_ID = r.json()["runner_id"]
                logger.info("Registered with backend as runner_id=%d (address=%s)", _RUNNER_ID, AGENT_ADDRESS)
        except Exception as e:
            logger.warning("Could not register with backend at startup: %s", e)

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    yield
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


async def _build_capabilities() -> dict:
    gpu = _gpu_stats()
    loaded = await _ollama_loaded_models()
    comfyui_ok = await _comfyui_ok()
    caps = {
        "gpu_vendor": gpu.get("gpu_vendor", "none"),
        "gpu_vram_total_bytes": gpu["vram_total_bytes"],
        "gpu_vram_used_bytes": gpu["vram_used_bytes"],
        "gpu_vram_free_bytes": max(0, gpu["vram_total_bytes"] - gpu["vram_used_bytes"]),
        "comfyui_running": comfyui_ok,
        "loaded_models": [m["name"] for m in loaded],
    }
    # Always include TLS cert so heartbeats don't wipe it
    try:
        caps["tls_cert"] = _read_cert_pem()
        caps["tls_fingerprint"] = _cert_fingerprint()
    except Exception:
        pass
    return caps


async def _heartbeat_loop():
    while True:
        await asyncio.sleep(30)
        if not _RUNNER_ID or not BACKEND_URL:
            continue
        # Auto-rotate TLS cert if expiring within 30 days
        try:
            _ensure_tls_cert(_DETECTED_IP)
        except Exception:
            pass
        try:
            caps = await _build_capabilities()
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(
                    f"{BACKEND_URL}/api/runners/heartbeat",
                    json={"runner_id": _RUNNER_ID, "capabilities": caps},
                    headers={"X-Agent-PSK": AGENT_PSK} if AGENT_PSK else {},
                )
        except Exception as e:
            logger.debug("Heartbeat failed: %s", e)


app = FastAPI(title="LLM Agent", version="1.0.0", lifespan=lifespan)


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


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _gpu_stats() -> dict:
    _zero = {"vram_used_bytes": 0, "vram_total_bytes": 0, "vram_pct": 0.0, "gpu_vendor": _GPU_BACKEND}
    if _GPU_BACKEND == "nvidia":
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pct = round(info.used / info.total * 100, 1) if info.total else 0.0
            return {
                "vram_used_bytes": info.used,
                "vram_total_bytes": info.total,
                "vram_pct": pct,
                "gpu_vendor": "nvidia",
            }
        except Exception:
            return _zero
    elif _GPU_BACKEND == "amd":
        try:
            handle = _amd_handles[0]
            vram_info = amdsmi.amdsmi_get_gpu_vram_usage(handle)
            used = vram_info["vram_used"]
            total = vram_info["vram_total"]
            pct = round(used / total * 100, 1) if total else 0.0
            return {
                "vram_used_bytes": used,
                "vram_total_bytes": total,
                "vram_pct": pct,
                "gpu_vendor": "amd",
            }
        except Exception:
            return _zero
    return _zero


def _sys_stats() -> dict:
    cpu = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    return {
        "cpu_pct": cpu,
        "mem_used_bytes": vm.used,
        "mem_total_bytes": vm.total,
        "mem_used_gb": round(vm.used / 1e9, 2),
        "mem_total_gb": round(vm.total / 1e9, 2),
    }


def _update_gauges(gpu: dict, sys: dict, loaded_count: int, comfyui_ok: bool):
    gpu_vram_used.labels(node=NODE).set(gpu["vram_used_bytes"])
    gpu_vram_total.labels(node=NODE).set(gpu["vram_total_bytes"])
    gpu_vram_util.labels(node=NODE).set(gpu["vram_pct"])
    cpu_usage.labels(node=NODE).set(sys["cpu_pct"])
    mem_used.labels(node=NODE).set(sys["mem_used_bytes"])
    mem_total.labels(node=NODE).set(sys["mem_total_bytes"])
    ollama_models_loaded.labels(node=NODE).set(loaded_count)
    comfyui_running_gauge.labels(node=NODE).set(1 if comfyui_ok else 0)


# ── Ollama helpers ─────────────────────────────────────────────────────────────

async def _ollama_ok() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _comfyui_ok() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{COMFYUI_URL}/system_stats")
            return r.status_code == 200
    except Exception:
        return False


async def _ollama_loaded_models() -> list[dict]:
    """Return list of currently loaded (in-VRAM) models from Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/ps")
            if r.status_code == 200:
                models = r.json().get("models", [])
                return [
                    {
                        "name": m["name"],
                        "size_gb": round(m.get("size", 0) / 1e9, 2),
                    }
                    for m in models
                ]
    except Exception:
        pass
    return []


async def _comfyui_checkpoints() -> list[str]:
    checkpoint_dir = Path("/opt/models/checkpoints")
    if not checkpoint_dir.exists():
        return []
    return sorted(
        p.name for p in checkpoint_dir.iterdir()
        if p.suffix in (".safetensors", ".ckpt", ".pt")
    )


async def _comfyui_active_checkpoint() -> Optional[str]:
    """Try to get the currently loaded checkpoint from ComfyUI object_info."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple")
            if r.status_code == 200:
                info = r.json()
                node_info = info.get("CheckpointLoaderSimple", {})
                input_info = node_info.get("input", {}).get("required", {})
                ckpt_input = input_info.get("ckpt_name", [])
                if isinstance(ckpt_input, list) and ckpt_input:
                    options = ckpt_input[0]
                    if isinstance(options, list) and options:
                        return options[0]
    except Exception:
        pass
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ollama_up = await _ollama_ok()
    comfyui_up = await _comfyui_ok()
    return {"ok": True, "node": NODE, "gpu_vendor": _GPU_BACKEND, "ollama": ollama_up, "comfyui": comfyui_up}


@app.get("/v1/status")
async def status():
    t0 = time.time()
    gpu = _gpu_stats()
    sys = _sys_stats()
    loaded = await _ollama_loaded_models()
    comfyui_up = await _comfyui_ok()
    checkpoints = await _comfyui_checkpoints()
    active_ckpt = await _comfyui_active_checkpoint() if comfyui_up else None

    _update_gauges(gpu, sys, len(loaded), comfyui_up)
    requests_total.labels(node=NODE, endpoint="/v1/status", model="").inc()
    request_duration.labels(node=NODE, endpoint="/v1/status", model="").observe(time.time() - t0)

    return {
        "node": NODE,
        "gpu_vendor": gpu.get("gpu_vendor", "none"),
        "gpu_vram_used_bytes": gpu["vram_used_bytes"],
        "gpu_vram_total_bytes": gpu["vram_total_bytes"],
        "gpu_vram_pct": gpu["vram_pct"],
        "gpu_vram_used_gb": round(gpu["vram_used_bytes"] / 1e9, 2),
        "gpu_vram_total_gb": round(gpu["vram_total_bytes"] / 1e9, 2),
        "cpu_pct": sys["cpu_pct"],
        "mem_used_gb": sys["mem_used_gb"],
        "mem_total_gb": sys["mem_total_gb"],
        "loaded_ollama_models": loaded,
        "comfyui_running": comfyui_up,
        "comfyui_checkpoints": checkpoints,
        "comfyui_active_checkpoint": active_ckpt,
    }


@app.get("/v1/models")
async def list_models():
    t0 = time.time()
    result = []

    # Ollama text models
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    result.append({
                        "id": m["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama",
                        "type": "text",
                    })
    except Exception as e:
        logger.warning("Ollama models unavailable: %s", e)

    # ComfyUI image models
    for ckpt in await _comfyui_checkpoints():
        result.append({
            "id": ckpt,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "comfyui",
            "type": "image",
        })

    requests_total.labels(node=NODE, endpoint="/v1/models", model="").inc()
    request_duration.labels(node=NODE, endpoint="/v1/models", model="").observe(time.time() - t0)

    return {"object": "list", "data": result}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)
    t0 = time.time()

    requests_total.labels(node=NODE, endpoint="/v1/chat/completions", model=model).inc()

    # Convert OpenAI format to Ollama format
    ollama_body = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": stream,
    }
    if "temperature" in body:
        ollama_body.setdefault("options", {})["temperature"] = body["temperature"]
    if "max_tokens" in body:
        ollama_body.setdefault("options", {})["num_predict"] = body["max_tokens"]
    if "top_p" in body:
        ollama_body.setdefault("options", {})["top_p"] = body["top_p"]

    if stream:
        async def _stream_gen() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=300) as c:
                    async with c.stream(
                        "POST",
                        f"{OLLAMA_URL}/api/chat",
                        json=ollama_body,
                        timeout=300,
                    ) as resp:
                        if resp.status_code != 200:
                            err = await resp.aread()
                            yield b"data: " + json.dumps({"error": err.decode()}).encode() + b"\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            # Convert Ollama streaming chunk to OpenAI SSE format
                            content = chunk.get("message", {}).get("content", "")
                            done = chunk.get("done", False)
                            openai_chunk = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content} if not done else {},
                                    "finish_reason": "stop" if done else None,
                                }],
                            }
                            yield b"data: " + json.dumps(openai_chunk).encode() + b"\n\n"
                            if done:
                                yield b"data: [DONE]\n\n"
                                break
            except Exception as e:
                yield b"data: " + json.dumps({"error": str(e)}).encode() + b"\n\n"
            finally:
                request_duration.labels(
                    node=NODE, endpoint="/v1/chat/completions", model=model
                ).observe(time.time() - t0)

        return StreamingResponse(_stream_gen(), media_type="text/event-stream")

    else:
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                r = await c.post(f"{OLLAMA_URL}/api/chat", json=ollama_body)
                if r.status_code != 200:
                    raise HTTPException(status_code=r.status_code, detail=r.text)
                data = r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")

        request_duration.labels(
            node=NODE, endpoint="/v1/chat/completions", model=model
        ).observe(time.time() - t0)

        msg = data.get("message", {})
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            },
        }


def _build_txt2img_workflow(prompt: str, checkpoint: str, width: int, height: int) -> dict:
    """Build a minimal ComfyUI API workflow for txt2img."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time()) % 2**32,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": height, "width": width},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": "ugly, bad anatomy, blurry, deformed"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "llm_agent", "images": ["8", 0]},
        },
    }


class ImageGenRequest(BaseModel):
    prompt: str
    model: str = "v1-5-pruned-emaonly.safetensors"
    n: int = 1
    size: str = "512x512"


@app.post("/v1/images/generations")
async def generate_image(req: ImageGenRequest):
    t0 = time.time()
    requests_total.labels(node=NODE, endpoint="/v1/images/generations", model=req.model).inc()

    if not await _comfyui_ok():
        raise HTTPException(status_code=503, detail="ComfyUI is not running")

    try:
        width, height = (int(x) for x in req.size.split("x"))
    except ValueError:
        width, height = 512, 512

    workflow = _build_txt2img_workflow(req.prompt, req.model, width, height)
    client_id = uuid.uuid4().hex

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"ComfyUI error: {r.text}")
            prompt_id = r.json()["prompt_id"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI submit failed: {e}")

    # Poll for completion (up to 5 minutes)
    deadline = time.time() + 300
    output_filename: Optional[str] = None
    while time.time() < deadline:
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{COMFYUI_URL}/history/{prompt_id}")
                if r.status_code == 200:
                    history = r.json()
                    if prompt_id in history:
                        prompt_result = history[prompt_id]
                        outputs = prompt_result.get("outputs", {})
                        for node_id, node_output in outputs.items():
                            images = node_output.get("images", [])
                            if images:
                                output_filename = images[0]["filename"]
                                break
                        if output_filename:
                            break
        except Exception as e:
            logger.warning("ComfyUI poll error: %s", e)

    request_duration.labels(
        node=NODE, endpoint="/v1/images/generations", model=req.model
    ).observe(time.time() - t0)

    if not output_filename:
        raise HTTPException(status_code=504, detail="ComfyUI generation timed out")

    return {
        "created": int(time.time()),
        "data": [{"url": f"/v1/images/outputs/{output_filename}"}],
    }


@app.get("/v1/images/outputs/{filename}")
async def get_image(filename: str):
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    path = Path(COMFYUI_OUTPUT_DIR) / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))


class PullRequest(BaseModel):
    model: str


@app.post("/v1/models/pull")
async def pull_model(req: PullRequest):
    requests_total.labels(node=NODE, endpoint="/v1/models/pull", model=req.model).inc()

    async def _stream_pull() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": req.model, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield line.encode() + b"\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode() + b"\n"

    return StreamingResponse(_stream_pull(), media_type="application/x-ndjson")


@app.delete("/v1/models/{model:path}")
async def unload_model(model: str):
    """Unload a model from VRAM by setting keep_alive=0."""
    requests_total.labels(node=NODE, endpoint="/v1/models/delete", model=model).inc()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "keep_alive": 0},
            )
            if r.status_code not in (200, 404):
                raise HTTPException(status_code=r.status_code, detail=r.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    return {"ok": True, "message": f"Model {model} unloaded from VRAM"}


class LoadRequest(BaseModel):
    model: str
    keep_alive: int = -1  # -1 means forever (seconds), 0 means unload immediately


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    """Load a model into VRAM by sending a minimal generate request with keep_alive."""
    requests_total.labels(node=NODE, endpoint="/v1/models/load", model=req.model).inc()
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0)) as c:
            r = await c.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": req.model, "prompt": "", "keep_alive": req.keep_alive},
            )
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    finally:
        request_duration.labels(node=NODE, endpoint="/v1/models/load", model=req.model).observe(time.time() - t0)
    return {"ok": True, "message": f"Model {req.model} loaded into VRAM"}


class CheckpointRequest(BaseModel):
    name: str


@app.post("/v1/comfyui/checkpoint")
async def switch_checkpoint(req: CheckpointRequest):
    """Switch ComfyUI checkpoint by queuing a no-op workflow that loads the model."""
    if not await _comfyui_ok():
        raise HTTPException(status_code=503, detail="ComfyUI is not running")

    # Verify checkpoint exists
    ckpt_path = Path("/opt/models/checkpoints") / req.name
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {req.name}")

    # Submit a minimal workflow just to load the checkpoint
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": req.name},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow},
            )
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"ComfyUI error: {r.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI unavailable: {e}")

    return {"ok": True, "checkpoint": req.name}


# ── ComfyUI lifecycle ────────────────────────────────────────────────────────

@app.post("/v1/comfyui/start")
async def start_comfyui():
    """Start the ComfyUI container."""
    if not _DOCKER_OK:
        raise HTTPException(503, "Docker not available")

    try:
        container = _docker.containers.get(COMFYUI_CONTAINER)
        if container.status == "running":
            return {"ok": True, "message": "ComfyUI already running"}
        container.start()
        return {"ok": True, "message": "ComfyUI started"}
    except docker_lib.errors.NotFound:
        pass

    # Container doesn't exist — create and start it
    try:
        # First try to find the image
        try:
            _docker.images.get(COMFYUI_IMAGE)
        except docker_lib.errors.ImageNotFound:
            # Try building from the image project directory
            raise HTTPException(404, f"ComfyUI image '{COMFYUI_IMAGE}' not found. Build it first.")

        _docker.containers.run(
            COMFYUI_IMAGE,
            name=COMFYUI_CONTAINER,
            detach=True,
            ports={"8188/tcp": 8188},
            volumes={
                "/opt/models/checkpoints": {"bind": "/app/models/checkpoints", "mode": "rw"},
                "/opt/models/lora": {"bind": "/app/models/loras", "mode": "rw"},
                "/opt/models/vae": {"bind": "/app/models/vae", "mode": "rw"},
                COMFYUI_OUTPUT_DIR: {"bind": "/app/output", "mode": "rw"},
            },
            device_requests=[
                docker_lib.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ],
            restart_policy={"Name": "unless-stopped"},
        )
        return {"ok": True, "message": "ComfyUI container created and started"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start ComfyUI: {e}")


@app.post("/v1/comfyui/stop")
async def stop_comfyui():
    """Stop the ComfyUI container."""
    if not _DOCKER_OK:
        raise HTTPException(503, "Docker not available")

    try:
        container = _docker.containers.get(COMFYUI_CONTAINER)
        if container.status != "running":
            return {"ok": True, "message": "ComfyUI already stopped"}
        container.stop(timeout=10)
        return {"ok": True, "message": "ComfyUI stopped"}
    except docker_lib.errors.NotFound:
        return {"ok": True, "message": "ComfyUI container not found (already stopped)"}
    except Exception as e:
        raise HTTPException(500, f"Failed to stop ComfyUI: {e}")


@app.get("/metrics")
async def metrics():
    # Update gauges before scrape
    gpu = _gpu_stats()
    sys = _sys_stats()
    loaded = await _ollama_loaded_models()
    comfyui_up = await _comfyui_ok()
    _update_gauges(gpu, sys, len(loaded), comfyui_up)

    return StreamingResponse(
        iter([generate_latest().decode()]),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8090,
        log_level="info",
        ssl_certfile=_TLS_CERT_PATH,
        ssl_keyfile=_TLS_KEY_PATH,
    )
