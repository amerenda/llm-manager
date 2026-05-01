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
import random
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

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

import collections

_LOG_BUFFER: collections.deque = collections.deque(maxlen=500)


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _LOG_BUFFER.append(self.format(record))


_buf_handler = _BufferHandler()
_buf_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger().addHandler(_buf_handler)
logger = logging.getLogger(__name__)

# ── GPU backend detection (NVIDIA via pynvml, AMD via sysfs) ─────────────────
_GPU_BACKEND = "none"  # "nvidia", "amd", or "none"
_amd_sysfs_dir = ""  # path to /sys/class/drm/cardN/device/ for AMD

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_BACKEND = "nvidia"
    logger.info("GPU backend: nvidia (pynvml)")
except Exception as exc:
    logger.debug("pynvml init failed: %s", exc)

if _GPU_BACKEND == "none":
    # AMD: find the first drm card with amdgpu VRAM sysfs files
    import glob as _glob
    for card_dir in sorted(_glob.glob("/sys/class/drm/card[0-9]*/device")):
        vram_total_path = os.path.join(card_dir, "mem_info_vram_total")
        vram_used_path = os.path.join(card_dir, "mem_info_vram_used")
        if os.path.isfile(vram_total_path) and os.path.isfile(vram_used_path):
            _amd_sysfs_dir = card_dir
            _GPU_BACKEND = "amd"
            logger.info("GPU backend: amd (sysfs: %s)", card_dir)
            break
    if _GPU_BACKEND != "amd":
        # Also check /dev/kfd as a hint that AMD GPU exists but sysfs isn't exposed
        if os.path.exists("/dev/kfd"):
            logger.warning("AMD GPU detected (/dev/kfd) but no VRAM sysfs found — is /sys mounted in the container?")

if _GPU_BACKEND == "none":
    logger.warning("No GPU backend detected — VRAM stats will be unavailable")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Ollama model storage — used for disk space reporting
# MODEL_STORAGE_PATH takes priority, then OLLAMA_MODELS, then default
OLLAMA_MODELS_DIR = os.environ.get("MODEL_STORAGE_PATH") or os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
COMFYUI_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", "/outputs")
COMFYUI_IMAGE = os.environ.get("COMFYUI_IMAGE", "murderbot-image-comfyui:latest")
COMFYUI_CONTAINER = os.environ.get("COMFYUI_CONTAINER", "comfyui")
OLLAMA_CONTAINER = os.environ.get("OLLAMA_CONTAINER", "")
# Docker default hostname is the container id (short hex). Prefer a stable host name.
NODE = (
    os.environ.get("RUNNER_HOSTNAME")
    or os.environ.get("AGENT_NODE_NAME")
    or socket.gethostname()
)

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
ALLOW_INSECURE_NO_PSK = os.environ.get("ALLOW_INSECURE_NO_PSK", "").lower() in ("1", "true", "yes")
BACKEND_URL = os.environ.get("BACKEND_URL", "")
AGENT_VERSION = os.environ.get("AGENT_VERSION", "unknown")
COMPOSE_DIR = os.environ.get("COMPOSE_DIR", "")  # host path — used in docker compose -f
COMPOSE_DIR_LOCAL = os.environ.get("COMPOSE_DIR_LOCAL", "")  # container-mounted path — used for file I/O
COMPOSE_PROFILE = os.environ.get("COMPOSE_PROFILE", "")  # nvidia or amd


def _unified_vram_enabled() -> bool:
    """Apple Silicon / unified memory: no NVML/AMD sysfs in the agent container."""
    return os.environ.get("AGENT_UNIFIED_MEMORY_VRAM", "").lower() in ("1", "true", "yes")


def _unified_vram_total_override_bytes() -> int | None:
    raw = (os.environ.get("AGENT_UNIFIED_VRAM_TOTAL_BYTES") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid AGENT_UNIFIED_VRAM_TOTAL_BYTES=%r", raw)
        return None


def _detect_host_ip() -> str:
    """Return the host's outbound IP — the interface used to reach the internet.
    Works correctly when the agent runs with network_mode: host.
    Falls back to hostname if detection fails.
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostname()
    finally:
        if s is not None:
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
_register_retry_task: Optional[asyncio.Task] = None


def _agent_auth_headers() -> dict:
    return {"X-Agent-PSK": AGENT_PSK}


def _agent_address_port() -> int:
    parsed = urlparse(AGENT_ADDRESS)
    return parsed.port or 8090

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


async def _register_once() -> bool:
    """Attempt a single registration. Returns True if successful, False on
    any failure. Sets _RUNNER_ID on success. Kept separate so it can be
    retried from a background task when the first attempt fails."""
    global _RUNNER_ID
    if not BACKEND_URL or not AGENT_ADDRESS:
        return False
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
                    "port": _agent_address_port(),
                    "capabilities": caps,
                },
                headers=_agent_auth_headers(),
            )
            r.raise_for_status()
            _RUNNER_ID = r.json()["runner_id"]
            logger.info("Registered with backend as runner_id=%d (address=%s)", _RUNNER_ID, AGENT_ADDRESS)
            return True
    except Exception as e:
        logger.warning("Could not register with backend: %s", e)
        return False


async def _register_retry_loop():
    """Keep retrying registration until it succeeds. Runs as a background
    task when the startup attempt fails — otherwise the agent would sit
    forever with _RUNNER_ID=None and no heartbeat (observed on murderbot
    during a backend rollout: register hit a 502, then the agent was
    invisible to the backend until someone manually restarted it).

    Backoff starts at 5s, doubles up to 5min, then stays there. Stops as
    soon as _RUNNER_ID is set."""
    delay = 5.0
    global _register_retry_task
    while _RUNNER_ID is None:
        await asyncio.sleep(delay)
        ok = await _register_once()
        if ok:
            # Fire an immediate heartbeat so the backend sees fresh state now
            asyncio.create_task(_heartbeat_once())
            _register_retry_task = None
            return
        # Jitter avoids synchronized retries after backend outages.
        delay = min(delay * 2, 300.0)
        delay = random.uniform(0.0, delay)


def _ensure_register_retry_task() -> None:
    global _register_retry_task
    if _register_retry_task and not _register_retry_task.done():
        return
    _register_retry_task = asyncio.create_task(_register_retry_loop())


def _check_ollama_compose_managed() -> None:
    """Warn loudly if the `ollama` container exists but isn't compose-managed
    under the same project as this agent. Symptom of a pre-migration orphan
    container: UI-driven tunable edits write to ollama.env but the running
    container was started with a different env/mounts and ignores the file.
    Seen on murderbot 2026-04-21 — ollama was pinned to /home/alex/.ollama
    while compose expected /opt/ollama; agent rewrites did nothing."""
    if not _DOCKER_OK or not OLLAMA_CONTAINER:
        return
    try:
        c = _docker.containers.get(OLLAMA_CONTAINER)
    except Exception:
        logger.info("Ollama container %r not found — compose will create it.", OLLAMA_CONTAINER)
        return
    labels = c.attrs.get("Config", {}).get("Labels") or {}
    project = labels.get("com.docker.compose.project")
    if project != "agent":
        logger.error(
            "Ollama container %r is NOT compose-managed (project=%r). "
            "UI tunable edits will NOT take effect. "
            "Fix: `docker rm -f %s && docker compose --profile %s up -d ollama` "
            "from %s",
            OLLAMA_CONTAINER, project or "<none>", OLLAMA_CONTAINER,
            COMPOSE_PROFILE or "<profile>", COMPOSE_DIR or "<compose dir>",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not AGENT_PSK and not ALLOW_INSECURE_NO_PSK:
        raise RuntimeError(
            "LLM_MANAGER_AGENT_PSK is required. "
            "Set ALLOW_INSECURE_NO_PSK=true only for local development."
        )
    _check_ollama_compose_managed()
    ok = await _register_once()
    if not ok:
        # Background retry — the periodic heartbeat loop already skips when
        # _RUNNER_ID is None, but without this loop the agent never recovers.
        _ensure_register_retry_task()

    # Fire an eager heartbeat immediately so the backend has fresh state
    # (downloaded_models, loaded_models, GPU counters) right after register,
    # instead of waiting up to 30s for the periodic loop's first tick. The
    # library UI was showing stale "downloaded on runnerX" info for that
    # first half-minute after every install/restart.
    if ok:
        asyncio.create_task(_heartbeat_once())
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    yield
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


async def _build_capabilities() -> dict:
    loaded = await _ollama_loaded_models()
    ollama_loaded_bytes = sum(int(m.get("size_bytes", 0) or 0) for m in loaded)
    gpu = _gpu_stats(ollama_loaded_bytes=ollama_loaded_bytes)
    disk = _disk_stats()
    downloaded = await _ollama_downloaded_models()
    comfyui_ok = await _comfyui_ok()
    caps = {
        "gpu_vendor": gpu.get("gpu_vendor", "none"),
        "gpu_vram_total_bytes": gpu["vram_total_bytes"],
        "gpu_vram_used_bytes": gpu["vram_used_bytes"],
        "gpu_vram_free_bytes": max(0, gpu["vram_total_bytes"] - gpu["vram_used_bytes"]),
        "disk_total_bytes": disk["disk_total_bytes"],
        "disk_used_bytes": disk["disk_used_bytes"],
        "disk_free_bytes": disk["disk_free_bytes"],
        "comfyui_running": comfyui_ok,
        "loaded_models": [m["name"] for m in loaded],
        # Per-runner authoritative "what's on disk" set, with digests so the
        # backend can detect out-of-date tags vs registry.ollama.ai without
        # polling each runner on every library view request. Replaces live
        # polling of /v1/models from library_routes.py.
        "downloaded_models": downloaded,
        "agent_version": AGENT_VERSION,
    }
    # Always include TLS cert so heartbeats don't wipe it
    try:
        caps["tls_cert"] = _read_cert_pem()
        caps["tls_fingerprint"] = _cert_fingerprint()
    except Exception:
        pass
    return caps


async def _heartbeat_once():
    """One heartbeat POST. Silently skips if not yet registered or no backend.
    Extracted so both the periodic loop and the eager post-register call go
    through the same code path."""
    global _RUNNER_ID
    if not _RUNNER_ID or not BACKEND_URL:
        return
    # Auto-rotate TLS cert if expiring within 30 days
    try:
        _ensure_tls_cert(_DETECTED_IP)
    except Exception:
        pass
    try:
        caps = await _build_capabilities()
        async with httpx.AsyncClient(timeout=5) as c:
            resp = await c.post(
                f"{BACKEND_URL}/api/runners/heartbeat",
                json={"runner_id": _RUNNER_ID, "capabilities": caps},
                headers=_agent_auth_headers(),
            )
            resp.raise_for_status()
            try:
                data = resp.json()
                if data.get("update_to") and data["update_to"] != AGENT_VERSION:
                    if data.get("auto_update"):
                        logger.info("Auto-updating: %s -> %s", AGENT_VERSION, data["update_to"])
                        asyncio.create_task(_self_update(data["update_to"]))
                    else:
                        logger.info("Update available: %s -> %s (manual trigger required)", AGENT_VERSION, data["update_to"])
            except Exception:
                pass
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        logger.warning("Heartbeat rejected (%d): %s", code, e.response.text[:200])
        if code in (401, 403, 404, 410):
            _RUNNER_ID = None
            _ensure_register_retry_task()
    except Exception as e:
        logger.debug("Heartbeat failed: %s", e)


async def _heartbeat_loop():
    while True:
        await asyncio.sleep(30)
        await _heartbeat_once()


_updating = False


def _upsert_env_key(env_path: str, key: str, value: str) -> None:
    lines = []
    found = False
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)


def _write_env_tag(env_path: str, tag: str):
    """Set AGENT_IMAGE_TAG in the .env file so compose uses the pinned tag."""
    _upsert_env_key(env_path, "AGENT_IMAGE_TAG", tag)


def _get_own_image_prefix() -> str:
    """Determine image repo + tag prefix from our own container's image name.
    e.g. 'amerenda/llm-manager:agent-amd-sha-abc' -> ('amerenda/llm-manager', 'agent-amd-')
    """
    try:
        own = _docker.containers.get(socket.gethostname())
        img = own.image.tags[0] if own.image.tags else own.attrs["Config"]["Image"]
    except Exception:
        try:
            own = _docker.containers.get("llm-agent")
            img = own.image.tags[0] if own.image.tags else own.attrs["Config"]["Image"]
        except Exception:
            return ""
    # img looks like 'amerenda/llm-manager:agent-amd-sha-abc1234'
    if ":" not in img:
        return ""
    repo, tag = img.rsplit(":", 1)
    # Strip the version suffix to get the prefix: 'agent-amd-sha-abc1234' -> 'agent-amd-'
    # The version is the AGENT_VERSION baked at build time
    if AGENT_VERSION != "unknown" and tag.endswith(AGENT_VERSION):
        prefix = tag[: -len(AGENT_VERSION)]
    elif tag.startswith("agent-amd-"):
        prefix = "agent-amd-"
    elif tag.startswith("agent-"):
        prefix = "agent-"
    else:
        prefix = tag.rsplit("-", 1)[0] + "-" if "-" in tag else ""
    return f"{repo}:{prefix}"


async def _self_update(target_version: str):
    """Pull new image via Docker SDK, then run a helper container to
    ``docker compose up -d --force-recreate`` with the new tag."""
    global _updating
    if _updating:
        return
    if not COMPOSE_DIR or not COMPOSE_PROFILE:
        logger.warning("Cannot self-update: COMPOSE_DIR or COMPOSE_PROFILE not set")
        return
    if not _DOCKER_OK:
        logger.warning("Cannot self-update: Docker client not available")
        return
    _updating = True
    try:
        logger.info("Self-updating to %s ...", target_version)

        # Pin the target version in .env for future docker compose up
        local_dir = COMPOSE_DIR_LOCAL or COMPOSE_DIR
        env_file = os.path.join(local_dir, ".env")
        _write_env_tag(env_file, target_version)
        logger.info("Pinned AGENT_IMAGE_TAG=%s in %s", target_version, env_file)

        # Determine new image tag from our current image name
        prefix = _get_own_image_prefix()
        if not prefix:
            logger.error("Cannot determine own image name for update")
            _updating = False
            return
        new_image = f"{prefix}{target_version}"
        logger.info("Pulling %s ...", new_image)

        # Pull new image via Docker SDK (talks to daemon over the socket)
        repo, tag = new_image.rsplit(":", 1)
        _docker.images.pull(repo, tag=tag)
        logger.info("Pull complete: %s", new_image)

        # Run a helper container with docker compose to recreate us.
        # The helper mounts the Docker socket and the compose dir (host paths),
        # then runs `docker compose up -d --force-recreate` targeted at ONLY
        # the llm-agent service. Must not recreate the whole profile —
        # `nvidia-full` / `amd-full` now include the ollama container, and
        # recreating it hits a name conflict (ollama already running) that
        # silently rolls back the entire update. The agent was stuck in a
        # 35s retry loop for a day because of this.
        compose_file = os.path.join(COMPOSE_DIR, "compose.yaml")
        agent_service = "llm-agent-amd" if COMPOSE_PROFILE.startswith("amd") else "llm-agent"
        cmd = (
            f"sleep 2 && "
            f"docker compose -f {compose_file} --profile {COMPOSE_PROFILE} "
            f"up -d --no-deps --force-recreate {agent_service}"
        )
        logger.info("Launching helper container to recreate with new image...")
        _docker.containers.run(
            "docker:cli",
            ["sh", "-c", cmd],
            remove=True,
            detach=True,
            volumes={
                "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
                COMPOSE_DIR: {"bind": COMPOSE_DIR, "mode": "ro"},
            },
            environment={"AGENT_IMAGE_TAG": target_version},
            network_mode="host",
        )
        logger.info("Helper dispatched, exiting for restart...")
        await asyncio.sleep(3)
        os._exit(0)
    except Exception as e:
        logger.error("Self-update failed: %s", e)
        _updating = False


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

def _gpu_stats(*, ollama_loaded_bytes: int = 0) -> dict:
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
        except Exception as exc:
            logger.warning("NVIDIA VRAM read failed: %s", exc)
            return _zero
    elif _GPU_BACKEND == "amd":
        try:
            total_path = os.path.join(_amd_sysfs_dir, "mem_info_vram_total")
            used_path = os.path.join(_amd_sysfs_dir, "mem_info_vram_used")
            with open(total_path) as f:
                total = int(f.read().strip())
            with open(used_path) as f:
                used = int(f.read().strip())
            pct = round(used / total * 100, 1) if total else 0.0
            return {
                "vram_used_bytes": used,
                "vram_total_bytes": total,
                "vram_pct": pct,
                "gpu_vendor": "amd",
            }
        except Exception as exc:
            logger.warning("AMD VRAM read failed: %s", exc, exc_info=True)
            return _zero
    if _unified_vram_enabled():
        vm = psutil.virtual_memory()
        total = _unified_vram_total_override_bytes() or int(vm.total)
        # Same physical pool as system RAM — use vm.used so VRAM matches /v1/status mem_*
        # and schedulers/UI don't treat Ollama weights-only as "free" RAM.
        used = min(max(0, int(vm.used)), total)
        pct = round(used / total * 100, 1) if total else 0.0
        return {
            "vram_used_bytes": used,
            "vram_total_bytes": total,
            "vram_pct": pct,
            "gpu_vendor": "unified",
        }
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


def _disk_stats() -> dict:
    """Disk usage for the Ollama model storage path."""
    # The container mounts the host root at /hostfs.  MODEL_STORAGE_PATH is a
    # *host* path (e.g. /mnt/storage/models) which won't exist inside the
    # container.  Prefer the /hostfs-prefixed version for the actual statvfs
    # call so we report the correct mount point instead of falling back to /.
    path = OLLAMA_MODELS_DIR
    hostfs_path = f"/hostfs{path}"
    if not os.path.exists(path) and os.path.exists(hostfs_path):
        path = hostfs_path
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        usage = shutil.disk_usage(path)
        return {
            "disk_total_bytes": usage.total,
            "disk_used_bytes": usage.used,
            "disk_free_bytes": usage.free,
            "disk_total_gb": round(usage.total / 1e9, 2),
            "disk_used_gb": round(usage.used / 1e9, 2),
            "disk_free_gb": round(usage.free / 1e9, 2),
            "disk_path": OLLAMA_MODELS_DIR,
        }
    except Exception as exc:
        logger.warning("Disk stats failed for %s: %s", OLLAMA_MODELS_DIR, exc)
        return {
            "disk_total_bytes": 0, "disk_used_bytes": 0, "disk_free_bytes": 0,
            "disk_total_gb": 0, "disk_used_gb": 0, "disk_free_gb": 0,
            "disk_path": OLLAMA_MODELS_DIR,
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


async def _ollama_downloaded_models() -> list[dict]:
    """Return list of on-disk Ollama models with manifest digest + size.
    Used by the backend to authoritatively track what each runner has
    downloaded — replaces the old pattern of the backend polling each runner
    on every /api/library request. The digest is what we'll later compare
    against registry.ollama.ai to detect out-of-date tags."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                return [
                    {
                        "name": m["name"],
                        "digest": m.get("digest", ""),
                        "size_bytes": m.get("size", 0),
                        "modified_at": m.get("modified_at", ""),
                    }
                    for m in models
                ]
    except Exception:
        logger.warning("Failed to list Ollama downloaded models", exc_info=False)
    return []


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
                        "size_bytes": int(m.get("size", 0) or 0),
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
    gpu_vendor = _gpu_stats(ollama_loaded_bytes=0).get("gpu_vendor", _GPU_BACKEND)
    return {"ok": True, "node": NODE, "gpu_vendor": gpu_vendor, "ollama": ollama_up, "comfyui": comfyui_up}


@app.get("/v1/status")
async def status():
    t0 = time.time()
    loaded = await _ollama_loaded_models()
    ollama_loaded_bytes = sum(int(m.get("size_bytes", 0) or 0) for m in loaded)
    gpu = _gpu_stats(ollama_loaded_bytes=ollama_loaded_bytes)
    sys = _sys_stats()
    disk = _disk_stats()
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
        "disk_total_gb": disk["disk_total_gb"],
        "disk_used_gb": disk["disk_used_gb"],
        "disk_free_gb": disk["disk_free_gb"],
        "disk_path": disk["disk_path"],
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

    # When tools are present, use Ollama's OpenAI-compatible endpoint directly
    # to avoid format conversion issues with tool_calls
    use_openai_compat = bool(body.get("tools"))

    # Convert OpenAI format to Ollama format (only for non-tool requests)
    ollama_body = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": stream,
    }
    if "tools" in body:
        ollama_body["tools"] = body["tools"]
    if "temperature" in body:
        if use_openai_compat:
            ollama_body["temperature"] = body["temperature"]
        else:
            ollama_body.setdefault("options", {})["temperature"] = body["temperature"]
    if "max_tokens" in body:
        if use_openai_compat:
            ollama_body["max_tokens"] = body["max_tokens"]
        else:
            ollama_body.setdefault("options", {})["num_predict"] = body["max_tokens"]
    if "top_p" in body:
        if use_openai_compat:
            ollama_body["top_p"] = body["top_p"]
        else:
            ollama_body.setdefault("options", {})["top_p"] = body["top_p"]

    if stream:
        stream_endpoint = f"{OLLAMA_URL}/v1/chat/completions" if use_openai_compat else f"{OLLAMA_URL}/api/chat"
        async def _stream_gen() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=300) as c:
                    async with c.stream(
                        "POST",
                        stream_endpoint,
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
        endpoint = f"{OLLAMA_URL}/v1/chat/completions" if use_openai_compat else f"{OLLAMA_URL}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                r = await c.post(endpoint, json=ollama_body)
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

        # If we used the OpenAI-compatible endpoint, pass through directly
        if use_openai_compat:
            return data

        msg = data.get("message", {})
        openai_msg = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
        }
        # Forward tool calls from Ollama response
        if msg.get("tool_calls"):
            openai_msg["tool_calls"] = [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"]),
                    },
                }
                for tc in msg["tool_calls"]
            ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": openai_msg,
                "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
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
    if req.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")
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
            # Long read timeout — large models on slow links; connect stays bounded.
            t = httpx.Timeout(30.0, read=3600.0)
            async with httpx.AsyncClient(timeout=t) as c:
                async with c.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": req.model, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread()).decode(errors="replace")[:800]
                        yield (
                            json.dumps(
                                {
                                    "error": (
                                        f"Ollama pull failed HTTP {resp.status_code}: {detail or resp.reason_phrase}. "
                                        "On Mac, ensure Ollama listens on 0.0.0.0:11434 (OLLAMA_HOST), "
                                        "not only 127.0.0.1, so the agent container can reach it via host.docker.internal."
                                    )
                                }
                            ).encode()
                            + b"\n"
                        )
                        return
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


@app.post("/v1/models/{model:path}/delete-from-disk")
async def delete_model_from_disk(model: str):
    """Delete a model from disk via Ollama's DELETE /api/delete."""
    requests_total.labels(node=NODE, endpoint="/v1/models/delete-disk", model=model).inc()
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.request(
                "DELETE",
                f"{OLLAMA_URL}/api/delete",
                json={"name": model},
            )
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Model {model} not found")
            if r.status_code not in (200,):
                raise HTTPException(status_code=r.status_code, detail=r.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
    return {"ok": True, "message": f"Model {model} deleted from disk"}


class LoadRequest(BaseModel):
    model: str
    keep_alive: int = -1  # -1 means forever (seconds), 0 means unload immediately


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    """Load a model into VRAM by sending a minimal generate request with keep_alive."""
    requests_total.labels(node=NODE, endpoint="/v1/models/load", model=req.model).inc()
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0)) as c:
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


# ── Agent update ──────────────────────────────────────────────────────────────

class UpdateRequest(BaseModel):
    target_version: str


@app.post("/v1/update")
async def trigger_update(req: UpdateRequest):
    """Trigger a self-update to the given version (called by backend)."""
    if req.target_version == AGENT_VERSION:
        return {"ok": True, "message": "Already at target version"}
    if _updating:
        return {"ok": True, "message": "Update already in progress"}
    asyncio.create_task(_self_update(req.target_version))
    return {"ok": True, "message": f"Updating to {req.target_version}"}


# ── ComfyUI ───────────────────────────────────────────────────────────────────

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


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

@app.post("/v1/ollama/restart")
async def restart_ollama():
    """Restart the Ollama container to clear stuck VRAM. Requires OLLAMA_CONTAINER env var."""
    if not OLLAMA_CONTAINER:
        raise HTTPException(503, "OLLAMA_CONTAINER not configured — cannot restart Ollama via Docker")
    if not _DOCKER_OK:
        raise HTTPException(503, "Docker not available")
    try:
        container = _docker.containers.get(OLLAMA_CONTAINER)
        container.restart(timeout=15)
        return {"ok": True, "message": f"Ollama container '{OLLAMA_CONTAINER}' restarted"}
    except docker_lib.errors.NotFound:
        raise HTTPException(404, f"Container '{OLLAMA_CONTAINER}' not found")
    except Exception as e:
        raise HTTPException(500, f"Failed to restart Ollama: {e}")


# ── Ollama runtime settings (server-level env vars) ──────────────────────────
#
# Tunables the UI is allowed to set. The agent rewrites ollama.env on the
# host-mounted compose dir, then recreates the ollama container via docker-py
# (not docker compose — agent image has no docker CLI). On recreate, the
# container's image, network mode, restart policy, mounts, devices, and GPU
# reservations are preserved; only the env vars change.

_OLLAMA_SETTINGS_ALLOWLIST = {
    # key → (type, validator or None)
    "OLLAMA_NUM_CTX":            ("int", lambda v: 1 <= int(v) <= 1_048_576),
    "OLLAMA_FLASH_ATTENTION":    ("bool", None),          # "0" / "1" / "true" / "false"
    "OLLAMA_KV_CACHE_TYPE":      ("enum", {"f16", "q8_0", "q4_0"}),
    "OLLAMA_NUM_GPU":            ("int", lambda v: 0 <= int(v) <= 999),
    "OLLAMA_NUM_PARALLEL":       ("int", lambda v: 1 <= int(v) <= 64),
    "OLLAMA_MAX_LOADED_MODELS":  ("int", lambda v: 1 <= int(v) <= 64),
    "OLLAMA_KEEP_ALIVE":         ("duration", None),      # "30s", "5m", "2h", "-1"
    "OLLAMA_MAX_QUEUE":          ("int", lambda v: 1 <= int(v) <= 65535),
    "OLLAMA_HOST":               ("str", None),           # "127.0.0.1:11434" etc
    # ROCm/HIP override — needed for new AMD GPUs (e.g. gfx1201 RX 9070 XT) whose
    # Tensile libs aren't bundled yet. Set to "11.0.0" to use gfx1100 (RDNA 3) libs.
    "HSA_OVERRIDE_GFX_VERSION":  ("str", None),
}

_OLLAMA_ENV_FILE = "/host-compose/ollama.env"


def _normalize_bool(v) -> str:
    """Return '1' or '0' for truthy / falsy input. Raises ValueError on garbage."""
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return "1"
    if s in ("0", "false", "no", "off", ""):
        return "0"
    raise ValueError(f"not a boolean: {v!r}")


def _validate_ollama_setting(key: str, value) -> str:
    """Validate key+value against the allowlist. Returns the stringified value
    ready to write to ollama.env. Raises ValueError on invalid input."""
    if key not in _OLLAMA_SETTINGS_ALLOWLIST:
        raise ValueError(f"unknown setting: {key}")
    kind, rule = _OLLAMA_SETTINGS_ALLOWLIST[key]
    if value is None or value == "":
        return ""  # empty = unset (comment out)
    if kind == "int":
        try:
            iv = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be an integer, got {value!r}") from e
        if rule and not rule(iv):
            raise ValueError(f"{key}={iv} is out of range")
        return str(iv)
    if kind == "bool":
        return _normalize_bool(value)
    if kind == "enum":
        s = str(value).strip()
        if s not in rule:
            raise ValueError(f"{key} must be one of {sorted(rule)}, got {value!r}")
        return s
    if kind == "duration":
        s = str(value).strip()
        if s == "-1":
            return s
        import re
        if not re.fullmatch(r"\d+(ms|s|m|h)", s):
            raise ValueError(f"{key} must be a duration (e.g. 30s, 5m, 2h) or -1, got {value!r}")
        return s
    # str: pass through with light sanity
    s = str(value).strip()
    if "\n" in s or "\r" in s:
        raise ValueError(f"{key} contains a newline")
    return s


def _parse_env_file(path: str) -> dict:
    """Parse a simple shell-style env file. Returns {key: value}. Comments
    and blank lines are dropped. Keys outside the allowlist are dropped too
    (prevents leaking unrelated env into the UI)."""
    result = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                # Strip matching quotes from value
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                if k in _OLLAMA_SETTINGS_ALLOWLIST:
                    result[k] = v
    except FileNotFoundError:
        pass
    return result


def _write_env_file(path: str, settings: dict) -> None:
    """Rewrite ollama.env preserving the commented-out template and setting
    only keys with non-empty values. Keys with empty value are commented out."""
    # Known order for stability
    ordered_keys = list(_OLLAMA_SETTINGS_ALLOWLIST.keys())
    lines = [
        "# Ollama server-level runtime tunables (managed by llm-manager).",
        "# Edit via the Runners page in the UI, not by hand — the agent will",
        "# overwrite this file when settings are applied.",
        "",
    ]
    for key in ordered_keys:
        val = settings.get(key, "")
        if val == "" or val is None:
            lines.append(f"# {key}=")
        else:
            lines.append(f"{key}={val}")
    # Ensure parent dir exists (docker-compose dir is bind-mounted from host)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def _get_ollama_tag_env_key() -> str:
    """Return the .env key for the Ollama image tag based on the compose profile."""
    return "OLLAMA_AMD_IMAGE_TAG" if COMPOSE_PROFILE.startswith("amd") else "OLLAMA_IMAGE_TAG"


def _read_ollama_image_tag() -> str:
    """Read the current Ollama image tag from the host .env file."""
    local_dir = COMPOSE_DIR_LOCAL or COMPOSE_DIR
    if not local_dir:
        return ""
    key = _get_ollama_tag_env_key()
    env_file = os.path.join(local_dir, ".env")
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line[len(key) + 1:].strip()
    except FileNotFoundError:
        pass
    return ""


def _write_ollama_image_tag(tag: str) -> None:
    """Update the Ollama image tag in the host .env file."""
    local_dir = COMPOSE_DIR_LOCAL or COMPOSE_DIR
    if not local_dir:
        raise RuntimeError("COMPOSE_DIR not configured")
    key = _get_ollama_tag_env_key()
    env_file = os.path.join(local_dir, ".env")
    _upsert_env_key(env_file, key, tag)


def _ollama_image_commit(image: str) -> Optional[str]:
    """Return the git commit SHA from Docker image OCI labels, if present."""
    if not _DOCKER_OK:
        return None
    try:
        img = _docker.images.get(image)
        labels = img.labels or {}
        return labels.get("org.opencontainers.image.revision") or None
    except Exception:
        return None


def _recreate_ollama_container(new_env: dict, new_image: Optional[str] = None) -> None:
    """Stop + remove the existing ollama container, then recreate it with the
    same image/mounts/devices/GPU config but a new env dict.

    Preserves: image, NetworkMode, RestartPolicy, Binds (volumes), Devices,
    DeviceRequests (nvidia GPU), GroupAdd (amd video/render groups), PortBindings.
    Only the env (PATH, OLLAMA_HOST, HSA_OVERRIDE_GFX_VERSION + allowlisted
    tunables) changes.  Pass new_image to swap the image tag (for upgrades).

    Raises RuntimeError with a message on failure — caller should surface it.
    """
    if not _DOCKER_OK:
        raise RuntimeError("Docker not available inside agent container")

    try:
        inspect = _docker.api.inspect_container(OLLAMA_CONTAINER)
    except docker_lib.errors.NotFound as e:
        raise RuntimeError(f"container {OLLAMA_CONTAINER!r} not found") from e

    cfg = inspect.get("Config") or {}
    host = inspect.get("HostConfig") or {}
    image = new_image or cfg.get("Image")

    # Preserve env vars that belong to the runtime (not user tunables).
    preserve = {"PATH", "OLLAMA_HOST", "HSA_OVERRIDE_GFX_VERSION",
                "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                "LANG", "LC_ALL", "HOME"}
    existing_env = {}
    for item in (cfg.get("Env") or []):
        if "=" in item:
            k, v = item.split("=", 1)
            if k in preserve:
                existing_env[k] = v
    # Override/augment with the user's new tunables
    merged_env = {**existing_env, **new_env}

    kwargs = dict(
        image=image,
        name=OLLAMA_CONTAINER,
        detach=True,
        environment=merged_env,
        network_mode=host.get("NetworkMode") or "host",
        restart_policy=host.get("RestartPolicy") or {"Name": "unless-stopped"},
    )

    binds = host.get("Binds") or []
    if binds:
        # docker-py accepts the same "src:dst[:opts]" string format
        kwargs["volumes"] = binds

    # Device list (AMD /dev/kfd + /dev/dri mounts)
    devices = host.get("Devices") or []
    if devices:
        kwargs["devices"] = [
            f"{d['PathOnHost']}:{d['PathInContainer']}:{d.get('CgroupPermissions', 'rwm')}"
            for d in devices
        ]

    # GPU requests (NVIDIA)
    drequests = host.get("DeviceRequests") or []
    if drequests:
        kwargs["device_requests"] = [
            docker_lib.types.DeviceRequest(
                driver=d.get("Driver") or "",
                count=d.get("Count", -1),
                device_ids=d.get("DeviceIDs") or [],
                capabilities=d.get("Capabilities") or [],
                options=d.get("Options") or {},
            )
            for d in drequests
        ]

    group_add = host.get("GroupAdd") or []
    if group_add:
        kwargs["group_add"] = group_add

    # Stop + remove old. Timeout keeps Ollama's HTTP handler graceful.
    try:
        old = _docker.containers.get(OLLAMA_CONTAINER)
        old.stop(timeout=15)
        old.remove(force=True)
    except docker_lib.errors.NotFound:
        pass

    # Pull image if missing (image tag may have been updated)
    try:
        _docker.images.get(image)
    except docker_lib.errors.ImageNotFound:
        logger.info("Pulling %s before recreate", image)
        _docker.images.pull(image)

    _docker.containers.run(**kwargs)


@app.get("/v1/ollama/settings")
async def get_ollama_settings():
    """Read the current ollama.env tunings + show which keys are controllable."""
    settings = _parse_env_file(_OLLAMA_ENV_FILE)
    return {
        "settings": settings,
        "allowlist": {k: v[0] for k, v in _OLLAMA_SETTINGS_ALLOWLIST.items()},
        "env_file": _OLLAMA_ENV_FILE,
    }


class OllamaSettingsRequest(BaseModel):
    # Only keys in _OLLAMA_SETTINGS_ALLOWLIST are accepted; others rejected.
    # Values may be str/int/bool — normalized server-side.
    settings: dict


@app.put("/v1/ollama/settings")
async def put_ollama_settings(req: OllamaSettingsRequest):
    """Validate, write ollama.env, then recreate the ollama container so the
    new env takes effect."""
    if not _DOCKER_OK:
        raise HTTPException(503, "Docker not available inside agent container")

    # Validate + normalize every incoming value first; bail before touching disk.
    normalized: dict = {}
    for k, v in (req.settings or {}).items():
        try:
            normalized[k] = _validate_ollama_setting(k, v)
        except ValueError as e:
            raise HTTPException(400, f"{k}: {e}")

    # Keep any currently-set keys the caller didn't include (partial updates).
    # Caller can explicitly clear a key by passing empty string.
    current = _parse_env_file(_OLLAMA_ENV_FILE)
    merged = {**current, **normalized}
    # Drop empty-string values from the container env (they mean "unset"),
    # but keep them commented-out in the file for visibility.
    container_env = {k: v for k, v in merged.items() if v != ""}

    # Write new file first (on disk), then recreate. If recreate fails, the
    # file is still the desired state — user can fix and re-apply.
    _write_env_file(_OLLAMA_ENV_FILE, merged)

    try:
        _recreate_ollama_container(container_env)
    except RuntimeError as e:
        raise HTTPException(500, f"Wrote ollama.env but failed to recreate container: {e}")
    except Exception as e:
        logger.exception("recreate failed")
        raise HTTPException(500, f"Wrote ollama.env but failed to recreate container: {e}")

    return {
        "ok": True,
        "applied": container_env,
        "message": f"Ollama restarted with {len(container_env)} tuning(s).",
    }


async def _ollama_version() -> Optional[str]:
    """Fetch the running Ollama version string from its API."""
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/version")
            if r.status_code == 200:
                return r.json().get("version")
    except Exception:
        pass
    return None


@app.get("/v1/ollama/version")
async def get_ollama_version():
    """Return running Ollama version, configured image tag, and git commit hash."""
    version = await _ollama_version()
    image_tag = _read_ollama_image_tag()
    commit: Optional[str] = None
    if _DOCKER_OK and OLLAMA_CONTAINER:
        try:
            container = _docker.containers.get(OLLAMA_CONTAINER)
            image_name = container.attrs.get("Config", {}).get("Image", "")
            if image_name:
                commit = _ollama_image_commit(image_name)
        except Exception:
            pass
    return {"version": version, "image_tag": image_tag, "commit": commit}


class OllamaUpgradeRequest(BaseModel):
    tag: str


@app.post("/v1/ollama/upgrade")
async def upgrade_ollama(req: OllamaUpgradeRequest):
    """Pull a new Ollama image version and recreate the container.
    Writes the new tag to .env so it persists across compose restarts."""
    if not OLLAMA_CONTAINER:
        raise HTTPException(503, "OLLAMA_CONTAINER not configured")
    if not _DOCKER_OK:
        raise HTTPException(503, "Docker not available")

    import re
    tag = req.tag.strip()
    if not tag:
        raise HTTPException(400, "tag is required")
    if not re.fullmatch(r"[\w.\-]+", tag):
        raise HTTPException(400, f"Invalid tag format: {tag!r}")

    try:
        inspect = _docker.api.inspect_container(OLLAMA_CONTAINER)
    except docker_lib.errors.NotFound:
        raise HTTPException(404, f"Container '{OLLAMA_CONTAINER}' not found")

    current_image = inspect.get("Config", {}).get("Image", "ollama/ollama")
    repo = current_image.rsplit(":", 1)[0] if ":" in current_image else current_image
    new_image = f"{repo}:{tag}"

    try:
        _write_ollama_image_tag(tag)
    except RuntimeError as e:
        raise HTTPException(500, f"Failed to update .env: {e}")

    logger.info("Pulling Ollama image %s ...", new_image)
    try:
        pull_repo, pull_tag = new_image.rsplit(":", 1)
        _docker.images.pull(pull_repo, tag=pull_tag)
    except Exception as e:
        raise HTTPException(500, f"Failed to pull {new_image}: {e}")

    current_env = _parse_env_file(_OLLAMA_ENV_FILE)
    container_env = {k: v for k, v in current_env.items() if v != ""}
    try:
        _recreate_ollama_container(container_env, new_image=new_image)
    except RuntimeError as e:
        raise HTTPException(500, f"Pulled image but failed to recreate container: {e}")

    commit = _ollama_image_commit(new_image)
    return {
        "ok": True,
        "image": new_image,
        "tag": tag,
        "commit": commit,
        "message": f"Ollama upgraded to {tag}",
    }


@app.get("/v1/logs")
async def get_logs(tail: int = 200, service: str = "all"):
    """Return recent log lines for agent and/or ollama container.

    service: "all" | "agent" | "ollama"
    tail: number of lines to return per service (max 500)
    """
    tail = max(1, min(tail, 500))
    agent_lines: list[str] = []
    ollama_lines: list[str] = []
    ollama_available = False

    if service in ("all", "agent"):
        agent_lines = list(_LOG_BUFFER)[-tail:]

    if service in ("all", "ollama") and OLLAMA_CONTAINER and _DOCKER_OK:
        try:
            container = _docker.containers.get(OLLAMA_CONTAINER)
            raw: bytes = container.logs(tail=tail, timestamps=True, stream=False)
            ollama_available = True
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if line.strip():
                    ollama_lines.append(line)
        except Exception as e:
            ollama_lines = [f"[error reading ollama logs: {e}]"]
            ollama_available = True

    return {
        "agent_logs": agent_lines,
        "ollama_logs": ollama_lines,
        "ollama_available": ollama_available,
    }


@app.get("/metrics")
async def metrics():
    # Update gauges before scrape
    loaded = await _ollama_loaded_models()
    ollama_loaded_bytes = sum(int(m.get("size_bytes", 0) or 0) for m in loaded)
    gpu = _gpu_stats(ollama_loaded_bytes=ollama_loaded_bytes)
    sys = _sys_stats()
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

