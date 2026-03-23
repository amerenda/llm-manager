"""
LLM Manager Backend API.
Runs as a stateless k8s Deployment on port 8081.
LLM proxy, model management, app registry, and queue scheduler.
LLM runners self-register via PSK instead of being polled.
"""
import asyncio
import logging
import os
import re
import socket
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel

import db
from db import (
    init_db,
    register_app, heartbeat_app, get_apps, deregister_app,
)
from gpu import vram_for_model
from llm_agent import LLMAgentClient
from scheduler import Scheduler
from queue_routes import router as queue_router, model_router
from library_routes import router as library_router, safety_router
from library import classify_models_batch, parse_param_count, parse_quantization, refresh_library_cache
import queue_db
import auth
from auth import (
    GITHUB_CLIENT_ID, COOKIE_NAME, SESSION_TTL,
    create_session_token, get_current_user, require_admin,
    exchange_code_for_user,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

AGENT_PSK = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
REGISTRATION_SECRET = os.environ.get("LLM_MANAGER_REGISTRATION_SECRET", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
NODE = socket.gethostname()

# Background operations tracker (pull, load, unload)
_ops: dict[str, dict] = {}

# ── Prometheus metrics ────────────────────────────────────────────────────────

api_requests_total = Counter(
    "llm_backend_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
registered_apps_gauge = Gauge("llm_backend_registered_apps", "Number of registered apps")


def _inc_request(endpoint: str, method: str, status: int):
    api_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()


# ── Runner helpers ─────────────────────────────────────────────────────────────

async def _get_runner_client(pool: asyncpg.Pool, runner_id: Optional[int] = None) -> LLMAgentClient:
    """Return an LLMAgentClient pointed at an active runner."""
    runners_list = await db.get_active_runners(pool)
    if not runners_list:
        raise HTTPException(503, "No active llm-runners available")
    if runner_id is not None:
        r = next((x for x in runners_list if x["id"] == runner_id), None)
        if not r:
            raise HTTPException(404, "Runner not found or inactive")
    else:
        r = runners_list[0]
    url = r["address"].rstrip("/")
    if "://" in url:
        url = url.split("://", 1)[1]
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        port = int(port_str)
    else:
        host = url
        port = 8090
    # Extract TLS cert from capabilities for cert-pinned HTTPS
    caps = r.get("capabilities", {})
    tls_cert_pem = caps.get("tls_cert") if isinstance(caps, dict) else None
    return LLMAgentClient(host=host, port=port, psk=AGENT_PSK, tls_cert_pem=tls_cert_pem)


async def _get_runner_ollama_base(pool: asyncpg.Pool, runner_id: Optional[int] = None) -> str:
    """Get Ollama URL for a runner. Replaces the runner port with 11434.
    Ollama always uses plain HTTP regardless of agent protocol."""
    runners_list = await db.get_active_runners(pool)
    if not runners_list:
        raise HTTPException(503, "No active llm-runners available")
    if runner_id is not None:
        r = next((x for x in runners_list if x["id"] == runner_id), None)
        if not r:
            r = runners_list[0]
    else:
        r = runners_list[0]
    # runner address is like https://10.x.x.x:8090
    # ollama is on the same host at port 11434, always plain HTTP
    addr = r["address"]
    # Strip scheme and port, rebuild as http with Ollama port
    host = re.sub(r'^https?://', '', addr)
    host = re.sub(r':\d+$', '', host)
    return f"http://{host}:11434"


# ── Lifespan ──────────────────────────────────────────────────────────────────

SCHEDULER_LOCK_ID = 900001  # Postgres advisory lock ID for scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    app.state.db = pool
    await init_db(pool)
    await queue_db.init_queue_tables(pool)
    logger.info("Database connected: %s", DATABASE_URL)

    # Acquire advisory lock for scheduler — only one pod runs it
    lock_conn = await asyncpg.connect(DATABASE_URL)
    got_lock = await lock_conn.fetchval(
        "SELECT pg_try_advisory_lock($1)", SCHEDULER_LOCK_ID
    )
    app.state.lock_conn = lock_conn

    async def get_ollama():
        return await _get_runner_ollama_base(pool)
    scheduler = Scheduler(pool, get_ollama, lock_conn=lock_conn if got_lock else None)
    app.state.scheduler = scheduler

    if got_lock:
        scheduler.start()
        logger.info("Queue scheduler started (advisory lock acquired)")
    else:
        logger.info("Scheduler skipped — another pod holds the lock")

    yield

    scheduler.stop()
    if got_lock:
        try:
            await lock_conn.execute(
                "SELECT pg_advisory_unlock($1)", SCHEDULER_LOCK_ID
            )
        except Exception:
            pass
    try:
        await lock_conn.close()
    except Exception:
        pass
    await pool.close()


app = FastAPI(title="LLM Manager Backend", version="3.0.0", lifespan=lifespan)
UI_ORIGIN = os.environ.get("UI_ORIGIN", "https://llm-manager.amer.dev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[UI_ORIGIN, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(queue_router)
app.include_router(model_router)
app.include_router(library_router)
app.include_router(safety_router)

# ── Auth middleware ───────────────────────────────────────────────────────────
# Routes that DON'T need admin auth (they have their own auth or are public)
_PUBLIC_PATHS = {
    "/health", "/metrics", "/auth/login", "/auth/callback", "/auth/me",
    "/auth/logout", "/api/stats",
}
_PUBLIC_PREFIXES = (
    "/v1/",                    # OpenAI-compat proxy (app API key auth)
    "/api/runners/",           # Agent PSK auth
    "/api/apps/discover",      # Registration secret auth
    "/api/apps/heartbeat",     # App API key auth
    "/api/queue/jobs/",        # Job status (app API key or public)
    "/api/queue/batches/",     # Batch status
    "/api/queue/submit",       # Job submission (app API key)
    "/api/profiles/list",      # App profile discovery (API key auth)
    "/api/queue/status",       # Queue overview
)


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    """Require admin auth for management API routes."""
    path = request.url.path

    # Skip auth for public/self-authenticated routes
    if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)

    # Skip auth for non-API routes (docs, openapi.json, etc.)
    if not path.startswith("/api/"):
        return await call_next(request)

    # Allow through if request has a valid app API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
        app_row = await db.get_app_by_api_key(request.app.state.db, api_key)
        if app_row and app_row.get("status") == "active":
            return await call_next(request)

    # Admin routes: require session cookie
    user = get_current_user(request)
    if not user or user not in auth.GITHUB_ALLOWED_USERS:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

    return await call_next(request)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = app.state.db is not None
    return {"ok": True, "service": "llm-manager-backend", "node": NODE, "db": db_ok}


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/auth/login")
async def auth_login():
    """Redirect to GitHub OAuth authorization page."""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(500, "GitHub OAuth not configured")
    params = f"client_id={GITHUB_CLIENT_ID}&scope=read:user"
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")


@app.get("/auth/callback")
async def auth_callback(code: str = ""):
    """Handle GitHub OAuth callback."""
    if not code:
        raise HTTPException(400, "Missing code parameter")

    username = await exchange_code_for_user(code)
    if not username:
        raise HTTPException(401, "GitHub authentication failed")

    if username not in auth.GITHUB_ALLOWED_USERS:
        logger.warning("Login rejected for GitHub user: %s", username)
        raise HTTPException(403, f"User '{username}' is not authorized")

    logger.info("Admin login: %s", username)
    token = create_session_token(username)
    response = RedirectResponse(UI_ORIGIN, status_code=302)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@app.get("/auth/me")
async def auth_me(request: Request):
    """Return current authenticated user, or 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"user": user, "admin": user in auth.GITHUB_ALLOWED_USERS}


@app.get("/auth/logout")
async def auth_logout():
    """Clear the session cookie."""
    response = RedirectResponse(UI_ORIGIN, status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


# ── Public stats (no auth required) ──────────────────────────────────────────

@app.get("/api/stats")
async def public_stats():
    """Aggregate stats visible without auth. No model names or app details."""
    pool = app.state.db
    gpu = {"name": "Unknown", "vram_total_gb": 0, "vram_used_gb": 0, "vram_free_gb": 0}
    loaded_count = 0

    try:
        client = await _get_runner_client(pool)
        status = await client.status()
        total = status.get("gpu_vram_total_gb", 0)
        used = status.get("gpu_vram_used_gb", 0)
        gpu = {
            "vram_total_gb": round(total, 2),
            "vram_used_gb": round(used, 2),
            "vram_free_gb": round(total - used, 2),
        }
        loaded_count = len(status.get("loaded_ollama_models", []))
    except Exception:
        pass

    app_count = 0
    try:
        apps_list = await get_apps(pool)
        app_count = len([a for a in apps_list if a.get("status") == "active"])
    except Exception:
        pass

    runner_count = 0
    try:
        runners = await db.get_active_runners(pool)
        runner_count = len(runners)
    except Exception:
        pass

    return {
        "gpu": gpu,
        "active_models": loaded_count,
        "connected_apps": app_count,
        "active_runners": runner_count,
    }


# ── GPU info (proxied from active runner) ────────────────────────────────────

@app.get("/api/gpu")
async def gpu_info():
    """GPU info from the primary active runner."""
    try:
        client = await _get_runner_client(app.state.db)
        status = await client.status()
        total = status.get("gpu_vram_total_gb", 0)
        used = status.get("gpu_vram_used_gb", 0)
        return {
            "name": status.get("node", "GPU"),
            "vram_total_gb": round(total, 2),
            "vram_used_gb": round(used, 2),
            "vram_free_gb": round(total - used, 2),
        }
    except HTTPException:
        return {"name": "Unknown", "vram_total_gb": 0, "vram_used_gb": 0, "vram_free_gb": 0}


@app.get("/api/models")
async def models_for_agents():
    """Ollama model list with VRAM estimates and fit info per runner."""
    try:
        pool = app.state.db
        runners_list = await db.get_active_runners(pool)
        if not runners_list:
            return []

        # Get GPU info per runner
        runner_vram = {}
        for r in runners_list:
            caps = r.get("capabilities", {})
            if isinstance(caps, dict):
                total = caps.get("gpu_vram_total_bytes", 0) / (1024**3)
                runner_vram[r["hostname"]] = round(total, 1)

        # Get models from the first runner's Ollama
        ollama_base = await _get_runner_ollama_base(pool)
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{ollama_base}/api/tags")
            r.raise_for_status()
            data = r.json()

        # Get loaded models
        loaded_names = set()
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                ps_resp = await c.get(f"{ollama_base}/api/ps")
                if ps_resp.status_code == 200:
                    for lm in ps_resp.json().get("models", []):
                        loaded_names.add(lm["name"])
        except Exception:
            pass

        # Classify safety
        model_names = [m.get("name", "") for m in data.get("models", [])]
        safety_map = await classify_models_batch(pool, model_names)

        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_gb = round(m.get("size", 0) / 1e9, 2)
            vram_est = round(vram_for_model(name), 2)
            fits_on = [
                {"runner": hostname, "vram_total_gb": vram}
                for hostname, vram in runner_vram.items()
                if vram >= vram_est
            ]
            models.append({
                "name": name,
                "size_gb": size_gb,
                "vram_estimate_gb": vram_est,
                "parameter_count": parse_param_count(name),
                "quantization": parse_quantization(name),
                "safety": safety_map.get(name, "safe"),
                "downloaded": True,
                "loaded": name in loaded_names,
                "fits": len(fits_on) > 0,
                "fits_on": fits_on,
            })
        # Sort: models that fit first, then by VRAM estimate
        models.sort(key=lambda m: (not m["fits"], m["vram_estimate_gb"]))
        return models
    except Exception as e:
        logger.exception("Error fetching models")
        return []


class VramCheckRequest(BaseModel):
    models: list[str]


@app.post("/api/vram-check")
async def vram_check(req: VramCheckRequest):
    """Check if a set of models fit in GPU VRAM."""
    try:
        client = await _get_runner_client(app.state.db)
        status = await client.status()
        gpu_vram = status.get("gpu_vram_total_gb", 0)
    except HTTPException:
        gpu_vram = 0

    per_model = []
    total = 0.0
    for model in req.models:
        est = vram_for_model(model)
        per_model.append({"model": model, "vram_gb": round(est, 2)})
        total += est

    fits = gpu_vram > 0 and total <= gpu_vram
    return {
        "total_vram_needed_gb": round(total, 2),
        "gpu_vram_gb": round(gpu_vram, 2),
        "fits_simultaneously": fits,
        "per_model": per_model,
        "warning": (
            None if fits or not req.models
            else f"Selected models need {total:.1f} GB but GPU has {gpu_vram:.1f} GB. "
                 "Agents will be scheduled to run at different times."
        ),
    }


# ── LLM Runner registration ───────────────────────────────────────────────────

class RunnerRegisterRequest(BaseModel):
    hostname: str
    address: str
    port: int = 8090
    capabilities: dict = {}


@app.post("/api/runners/register")
async def register_runner(
    req: RunnerRegisterRequest,
    x_agent_psk: Optional[str] = Header(None),
):
    if AGENT_PSK and x_agent_psk != AGENT_PSK:
        raise HTTPException(401, "Invalid PSK")
    runner_id = await db.register_runner(
        app.state.db, req.hostname, req.address, req.port, req.capabilities
    )
    return {"ok": True, "runner_id": runner_id}


class RunnerHeartbeatRequest(BaseModel):
    runner_id: int
    capabilities: dict = {}


@app.post("/api/runners/heartbeat")
async def runner_heartbeat(
    req: RunnerHeartbeatRequest,
    x_agent_psk: Optional[str] = Header(None),
):
    if AGENT_PSK and x_agent_psk != AGENT_PSK:
        raise HTTPException(401, "Invalid PSK")
    found = await db.heartbeat_runner(app.state.db, req.runner_id, req.capabilities)
    if not found:
        raise HTTPException(404, "Runner not found")
    return {"ok": True}


@app.get("/api/runners")
async def list_runners():
    return await db.get_active_runners(app.state.db)


# ── LLM proxy — all ops target active runner(s) ───────────────────────────────

def _agent_unavailable(detail: str = "No active llm-runner available") -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


@app.get("/api/llm/status")
async def llm_status(runner_id: Optional[int] = None):
    _inc_request("/api/llm/status", "GET", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.status()
    except HTTPException:
        raise
    except Exception as e:
        _inc_request("/api/llm/status", "GET", 503)
        raise _agent_unavailable(f"Runner error: {e}")


@app.get("/api/llm/models")
async def llm_models(runner_id: Optional[int] = None):
    _inc_request("/api/llm/models", "GET", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.models()
    except HTTPException:
        raise
    except Exception as e:
        _inc_request("/api/llm/models", "GET", 503)
        raise _agent_unavailable(f"Runner error: {e}")


class LLMPullRequest(BaseModel):
    model: str


@app.post("/api/llm/models/pull")
async def llm_pull_model(req: LLMPullRequest, runner_id: Optional[int] = None):
    """Pull a model. Runs in background — returns immediately with operation ID."""
    _inc_request("/api/llm/models/pull", "POST", 200)
    op_id = f"pull-{req.model}-{id(req)}"
    _ops[op_id] = {"status": "running", "model": req.model, "type": "pull", "progress": ""}

    async def _do_pull():
        try:
            client = await _get_runner_client(app.state.db, runner_id)
            last_status = ""
            async for chunk in client.pull_model(req.model):
                last_status = chunk.decode().strip()
                _ops[op_id]["progress"] = last_status
            _ops[op_id]["status"] = "completed"
        except Exception as e:
            _ops[op_id]["status"] = "failed"
            _ops[op_id]["error"] = str(e)

    asyncio.create_task(_do_pull())
    return {"ok": True, "op_id": op_id, "message": f"Pulling {req.model} in background"}


@app.post("/api/models/{model:path}/update")
async def update_model(model: str, runner_id: Optional[int] = None):
    """Pull the latest version of an already-downloaded model. Runs in background."""
    _inc_request("/api/models/update", "POST", 200)
    op_id = f"update-{model}-{id(model)}"
    _ops[op_id] = {"status": "running", "model": model, "type": "update", "progress": ""}

    async def _do_update():
        try:
            client = await _get_runner_client(app.state.db, runner_id)
            async for chunk in client.pull_model(model):
                _ops[op_id]["progress"] = chunk.decode().strip()
            _ops[op_id]["status"] = "completed"
        except Exception as e:
            _ops[op_id]["status"] = "failed"
            _ops[op_id]["error"] = str(e)

    asyncio.create_task(_do_update())
    return {"ok": True, "op_id": op_id, "message": f"Updating {model} in background"}


@app.get("/api/ops")
async def list_operations():
    """List all background operations and their status."""
    # Clean up old completed ops (keep last 20)
    completed = [k for k, v in _ops.items() if v["status"] in ("completed", "failed")]
    for k in completed[:-20]:
        del _ops[k]
    return list(_ops.values())


@app.get("/api/ops/{op_id}")
async def get_operation(op_id: str):
    """Get status of a background operation."""
    if op_id not in _ops:
        raise HTTPException(404, "Operation not found")
    return _ops[op_id]


@app.delete("/api/llm/models/{model:path}")
async def llm_delete_model(model: str, runner_id: Optional[int] = None):
    _inc_request("/api/llm/models/delete", "DELETE", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.delete_model(model)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


class CheckpointSwitchRequest(BaseModel):
    name: str


@app.post("/api/llm/comfyui/checkpoint")
async def llm_switch_checkpoint(req: CheckpointSwitchRequest, runner_id: Optional[int] = None):
    _inc_request("/api/llm/comfyui/checkpoint", "POST", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.switch_checkpoint(req.name)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


@app.post("/api/llm/comfyui/start")
async def llm_start_comfyui(runner_id: Optional[int] = None):
    _inc_request("/api/llm/comfyui/start", "POST", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.start_comfyui()
    except HTTPException:
        raise
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


@app.post("/api/llm/comfyui/stop")
async def llm_stop_comfyui(runner_id: Optional[int] = None):
    _inc_request("/api/llm/comfyui/stop", "POST", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.stop_comfyui()
    except HTTPException:
        raise
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


@app.get("/api/llm/checkpoints")
async def llm_checkpoints(runner_id: Optional[int] = None):
    _inc_request("/api/llm/checkpoints", "GET", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        status = await client.status()
        return {"checkpoints": status.get("comfyui_checkpoints", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


# ── OpenAI-compatible proxy ───────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, runner_id: Optional[int] = None):
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    # Enforce per-app model restrictions
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        api_key = auth.removeprefix("Bearer ").strip()
        allowed = await db.check_model_allowed(app.state.db, api_key, model)
        if not allowed:
            raise HTTPException(403, f"Model '{model}' is not allowed for this application")

    _inc_request("/v1/chat/completions", "POST", 200)

    try:
        client = await _get_runner_client(app.state.db, runner_id)

        if stream:
            async def _forward_stream():
                stream_ctx = await client.chat(
                    messages=body.get("messages", []),
                    model=model,
                    stream=True,
                    **{k: v for k, v in body.items()
                       if k not in ("model", "messages", "stream")},
                )
                async with stream_ctx as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(_forward_stream(), media_type="text/event-stream")
        else:
            result = await client.chat(
                messages=body.get("messages", []),
                model=model,
                stream=False,
                **{k: v for k, v in body.items()
                   if k not in ("model", "messages", "stream")},
            )
            return result
    except HTTPException:
        raise
    except Exception as e:
        _inc_request("/v1/chat/completions", "POST", 503)
        raise _agent_unavailable(f"Runner error: {e}")


@app.post("/v1/images/generations")
async def proxy_image_generations(request: Request, runner_id: Optional[int] = None):
    body = await request.json()
    _inc_request("/v1/images/generations", "POST", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        return await client.generate_image(
            prompt=body.get("prompt", ""),
            model=body.get("model", "v1-5-pruned-emaonly.safetensors"),
            n=body.get("n", 1),
            size=body.get("size", "512x512"),
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        _inc_request("/v1/images/generations", "POST", e.response.status_code)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _inc_request("/v1/images/generations", "POST", 503)
        raise _agent_unavailable(f"Runner error: {e}")


# ── App registry ──────────────────────────────────────────────────────────────

@app.get("/api/apps")
async def list_apps():
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    _inc_request("/api/apps", "GET", 200)
    apps = await get_apps(app.state.db)
    registered_apps_gauge.set(len(apps))
    result = []
    for a in apps:
        a_copy = dict(a)
        key = a_copy.get("api_key", "")
        a_copy["api_key_preview"] = key[:8] + "..." if len(key) >= 8 else key
        del a_copy["api_key"]
        # Ensure new fields are present (for backwards compatibility)
        a_copy.setdefault("status", "active")
        a_copy.setdefault("allow_profile_switch", False)
        a_copy["allowed_models"] = await db.get_app_allowed_models(app.state.db, a_copy["id"])
        result.append(a_copy)
    return result


class AppRegisterRequest(BaseModel):
    name: str
    base_url: Optional[str] = None


@app.post("/api/apps/register")
async def register_new_app(req: AppRegisterRequest):
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    _inc_request("/api/apps/register", "POST", 200)
    api_key = await register_app(app.state.db, req.name, req.base_url)
    return {
        "ok": True,
        "api_key": api_key,
        "message": f"App '{req.name}' registered. Store your API key — it won't be shown again.",
    }


class AppHeartbeatRequest(BaseModel):
    metadata: dict = {}


@app.post("/api/apps/heartbeat")
async def app_heartbeat(request: Request, body: AppHeartbeatRequest):
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    api_key = auth.removeprefix("Bearer ").strip()

    found = await heartbeat_app(app.state.db, api_key, body.metadata)
    if not found:
        raise HTTPException(status_code=404, detail="App not found or invalid API key")

    _inc_request("/api/apps/heartbeat", "POST", 200)
    return {"ok": True}


@app.delete("/api/apps/{api_key}")
async def remove_app(api_key: str):
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    removed = await deregister_app(app.state.db, api_key)
    if not removed:
        raise HTTPException(status_code=404, detail="App not found")
    _inc_request("/api/apps/delete", "DELETE", 200)
    return {"ok": True}


# ── App discovery ────────────────────────────────────────────────────────────

class AppDiscoverRequest(BaseModel):
    name: str
    base_url: str
    registration_secret: str
    capabilities: list[str] = []


@app.post("/api/apps/discover")
async def discover_app_endpoint(req: AppDiscoverRequest):
    """Auto-discovery endpoint. Apps call this on startup to register or retrieve their key."""
    _inc_request("/api/apps/discover", "POST", 200)
    if not REGISTRATION_SECRET:
        raise HTTPException(400, "Registration secret not configured on server")
    if req.registration_secret != REGISTRATION_SECRET:
        raise HTTPException(403, "Invalid registration secret")
    result = await db.discover_app(
        app.state.db, req.name, req.base_url, req.capabilities
    )
    return result


@app.post("/api/apps/{app_id}/approve")
async def approve_app_endpoint(app_id: int):
    """Approve a pending app and push the API key to it."""
    _inc_request("/api/apps/approve", "POST", 200)
    pool = app.state.db
    api_key = await db.approve_app(pool, app_id)
    if api_key is None:
        raise HTTPException(404, "App not found")
    # Get the app's base_url to push the key
    app_row = await db.get_app_by_api_key(pool, api_key)
    push_result = None
    if app_row and app_row.get("base_url"):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(
                    f"{app_row['base_url'].rstrip('/')}/.well-known/llm-manager/register",
                    json={"api_key": api_key},
                )
                push_result = {"pushed": True, "status": r.status_code}
        except Exception as e:
            push_result = {"pushed": False, "error": str(e)}
    return {"ok": True, "api_key": api_key, "push_result": push_result}


class AppPermissionsRequest(BaseModel):
    allow_profile_switch: bool


@app.patch("/api/apps/{app_id}/permissions")
async def update_app_permissions_endpoint(app_id: int, req: AppPermissionsRequest):
    """Update permissions for an app."""
    _inc_request("/api/apps/permissions", "PATCH", 200)
    found = await db.update_app_permissions(app.state.db, app_id, req.allow_profile_switch)
    if not found:
        raise HTTPException(404, "App not found")
    return {"ok": True}


@app.get("/api/apps/{app_id}/allowed-models")
async def get_app_allowed_models_endpoint(app_id: int):
    models = await db.get_app_allowed_models(app.state.db, app_id)
    return {"app_id": app_id, "allowed_models": models, "unrestricted": len(models) == 0}


class AppAllowedModelsRequest(BaseModel):
    allowed_models: list[str]


@app.put("/api/apps/{app_id}/allowed-models")
async def set_app_allowed_models_endpoint(app_id: int, req: AppAllowedModelsRequest):
    await db.set_app_allowed_models(app.state.db, app_id, req.allowed_models)
    return {"ok": True}


# ── Model load/unload (ad-hoc) ───────────────────────────────────────────────

class ModelLoadRequest(BaseModel):
    model: str
    keep_alive: int = -1


@app.post("/api/llm/models/load")
async def llm_load_model(req: ModelLoadRequest, runner_id: Optional[int] = None):
    """Load a model into VRAM. Runs in background."""
    _inc_request("/api/llm/models/load", "POST", 200)
    op_id = f"load-{req.model}-{id(req)}"
    _ops[op_id] = {"status": "running", "model": req.model, "type": "load"}

    async def _do_load():
        try:
            client = await _get_runner_client(app.state.db, runner_id)
            await client.load_model(req.model, req.keep_alive)
            _ops[op_id]["status"] = "completed"
        except Exception as e:
            _ops[op_id]["status"] = "failed"
            _ops[op_id]["error"] = str(e)

    asyncio.create_task(_do_load())
    return {"ok": True, "op_id": op_id, "message": f"Loading {req.model} in background"}


class ModelUnloadRequest(BaseModel):
    model: str


@app.post("/api/llm/models/unload")
async def llm_unload_model(req: ModelUnloadRequest, runner_id: Optional[int] = None):
    """Unload a model from VRAM. Runs in background."""
    _inc_request("/api/llm/models/unload", "POST", 200)
    op_id = f"unload-{req.model}-{id(req)}"
    _ops[op_id] = {"status": "running", "model": req.model, "type": "unload"}

    async def _do_unload():
        try:
            client = await _get_runner_client(app.state.db, runner_id)
            await client.unload_model(req.model)
            _ops[op_id]["status"] = "completed"
        except Exception as e:
            _ops[op_id]["status"] = "failed"
            _ops[op_id]["error"] = str(e)

    asyncio.create_task(_do_unload())
    return {"ok": True, "op_id": op_id, "message": f"Unloading {req.model} in background"}


# ── Profiles ──────────────────────────────────────────────────────────────────

class ProfileCreateRequest(BaseModel):
    name: str
    unsafe_enabled: bool = False


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    unsafe_enabled: Optional[bool] = None


class ProfileModelEntryRequest(BaseModel):
    model_safe: str
    model_unsafe: Optional[str] = None
    count: int = 1
    label: Optional[str] = None
    parameters: dict = {}


class ProfileImageEntryRequest(BaseModel):
    checkpoint_safe: str
    checkpoint_unsafe: Optional[str] = None
    label: Optional[str] = None
    parameters: dict = {}


class ProfileActivateRequest(BaseModel):
    runner_id: int
    force: bool = False


@app.get("/api/profiles")
async def list_profiles():
    _inc_request("/api/profiles", "GET", 200)
    return await db.get_all_profiles(app.state.db)


@app.post("/api/profiles")
async def create_profile(req: ProfileCreateRequest):
    _inc_request("/api/profiles", "POST", 200)
    profile_id = await db.create_profile(app.state.db, req.name, req.unsafe_enabled)
    return {"ok": True, "id": profile_id}


# Static profile routes must come before {profile_id} parameterized routes
@app.get("/api/profiles/activations")
async def list_activations():
    _inc_request("/api/profiles/activations", "GET", 200)
    return await db.get_all_activations(app.state.db)


@app.get("/api/profiles/list")
async def list_profiles_for_apps(request: Request):
    """Endpoint for apps to discover available profiles (API key auth)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    api_key = auth.removeprefix("Bearer ").strip()
    app_row = await db.get_app_by_api_key(app.state.db, api_key)
    if not app_row:
        raise HTTPException(401, "Invalid API key")
    profiles = await db.get_all_profiles(app.state.db)
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "is_default": p["is_default"],
            "unsafe_enabled": p["unsafe_enabled"],
            "model_entry_count": p.get("model_entry_count", 0),
            "image_entry_count": p.get("image_entry_count", 0),
        }
        for p in profiles
    ]


@app.get("/api/profiles/{profile_id}")
async def get_profile(profile_id: int):
    _inc_request("/api/profiles/detail", "GET", 200)
    profile = await db.get_profile(app.state.db, profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile


@app.patch("/api/profiles/{profile_id}")
async def update_profile(profile_id: int, req: ProfileUpdateRequest):
    _inc_request("/api/profiles/update", "PATCH", 200)
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    if not kwargs:
        return {"ok": True}
    found = await db.update_profile(app.state.db, profile_id, **kwargs)
    if not found:
        raise HTTPException(404, "Profile not found")
    return {"ok": True}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: int):
    _inc_request("/api/profiles/delete", "DELETE", 200)
    try:
        found = await db.delete_profile(app.state.db, profile_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not found:
        raise HTTPException(404, "Profile not found")
    return {"ok": True}


# ── Profile model entries ────────────────────────────────────────────────────

@app.post("/api/profiles/{profile_id}/models")
async def add_profile_model(profile_id: int, req: ProfileModelEntryRequest):
    _inc_request("/api/profiles/models", "POST", 200)
    entry_id = await db.add_profile_model_entry(
        app.state.db, profile_id,
        model_safe=req.model_safe,
        model_unsafe=req.model_unsafe,
        count=req.count,
        label=req.label,
        parameters=req.parameters,
    )
    return {"ok": True, "id": entry_id}


@app.patch("/api/profiles/{profile_id}/models/{entry_id}")
async def update_profile_model(profile_id: int, entry_id: int, req: ProfileModelEntryRequest):
    _inc_request("/api/profiles/models/update", "PATCH", 200)
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    found = await db.update_profile_model_entry(app.state.db, entry_id, **kwargs)
    if not found:
        raise HTTPException(404, "Entry not found")
    return {"ok": True}


@app.delete("/api/profiles/{profile_id}/models/{entry_id}")
async def delete_profile_model(profile_id: int, entry_id: int):
    _inc_request("/api/profiles/models/delete", "DELETE", 200)
    found = await db.delete_profile_model_entry(app.state.db, entry_id)
    if not found:
        raise HTTPException(404, "Entry not found")
    return {"ok": True}


# ── Profile image entries ────────────────────────────────────────────────────

@app.post("/api/profiles/{profile_id}/images")
async def add_profile_image(profile_id: int, req: ProfileImageEntryRequest):
    _inc_request("/api/profiles/images", "POST", 200)
    entry_id = await db.add_profile_image_entry(
        app.state.db, profile_id,
        checkpoint_safe=req.checkpoint_safe,
        checkpoint_unsafe=req.checkpoint_unsafe,
        label=req.label,
        parameters=req.parameters,
    )
    return {"ok": True, "id": entry_id}


@app.patch("/api/profiles/{profile_id}/images/{entry_id}")
async def update_profile_image(profile_id: int, entry_id: int, req: ProfileImageEntryRequest):
    _inc_request("/api/profiles/images/update", "PATCH", 200)
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    found = await db.update_profile_image_entry(app.state.db, entry_id, **kwargs)
    if not found:
        raise HTTPException(404, "Entry not found")
    return {"ok": True}


@app.delete("/api/profiles/{profile_id}/images/{entry_id}")
async def delete_profile_image(profile_id: int, entry_id: int):
    _inc_request("/api/profiles/images/delete", "DELETE", 200)
    found = await db.delete_profile_image_entry(app.state.db, entry_id)
    if not found:
        raise HTTPException(404, "Entry not found")
    return {"ok": True}


# ── Profile activation ──────────────────────────────────────────────────────

@app.post("/api/profiles/{profile_id}/activate")
async def activate_profile_endpoint(
    profile_id: int,
    req: ProfileActivateRequest,
    request: Request,
):
    """Activate a profile on a runner. Supports both UI calls and app API key auth."""
    _inc_request("/api/profiles/activate", "POST", 200)
    pool = app.state.db

    # Check if this is an app calling via API key
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        api_key = auth.removeprefix("Bearer ").strip()
        app_row = await db.get_app_by_api_key(pool, api_key)
        if not app_row:
            raise HTTPException(401, "Invalid API key")
        if not app_row.get("allow_profile_switch"):
            raise HTTPException(403, "App does not have profile switching permission")

    # Validate profile exists
    profile = await db.get_profile(pool, profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    # Validate runner exists
    runner = await db.get_runner_by_id(pool, req.runner_id)
    if not runner:
        raise HTTPException(404, "Runner not found or inactive")

    # VRAM check
    caps = runner.get("capabilities", {})
    vram_total = caps.get("gpu_vram_total_bytes", 0)
    vram_free = caps.get("gpu_vram_free_bytes", 0)
    warnings = []

    # Estimate VRAM needed (rough: query Ollama model sizes would be better, but
    # for now we rely on the user's judgment + warning if many models)
    model_count = sum(e.get("count", 1) for e in profile.get("model_entries", []))
    if model_count > 3:
        warnings.append(f"Profile loads {model_count} model instances — may exceed VRAM")

    # Get the runner client for actual operations
    try:
        client = await _get_runner_client(pool, req.runner_id)
    except HTTPException:
        raise HTTPException(503, "Runner not reachable")

    # Set activation status
    await db.activate_profile(pool, req.runner_id, profile_id)

    try:
        # Step 1: Unload current models
        if req.force:
            # Force unload all loaded models
            try:
                status = await client.status()
                for m in status.get("loaded_ollama_models", []):
                    try:
                        await client.unload_model(m["name"])
                    except Exception as e:
                        logger.warning("Failed to unload %s: %s", m["name"], e)
            except Exception as e:
                logger.warning("Could not get status for force-unload: %s", e)

        # Step 2: Load profile models
        use_unsafe = profile.get("unsafe_enabled", False)
        for entry in profile.get("model_entries", []):
            model = entry.get("model_unsafe") if use_unsafe and entry.get("model_unsafe") else entry["model_safe"]
            for _ in range(entry.get("count", 1)):
                try:
                    await client.load_model(model)
                except Exception as e:
                    warnings.append(f"Failed to load {model}: {e}")

        # Step 3: Switch image checkpoint if specified
        for entry in profile.get("image_entries", []):
            ckpt = entry.get("checkpoint_unsafe") if use_unsafe and entry.get("checkpoint_unsafe") else entry["checkpoint_safe"]
            try:
                await client.switch_checkpoint(ckpt)
            except Exception as e:
                warnings.append(f"Failed to switch checkpoint to {ckpt}: {e}")

        await db.update_activation_status(pool, req.runner_id, "active")
    except Exception as e:
        await db.update_activation_status(pool, req.runner_id, "error")
        raise HTTPException(500, f"Activation failed: {e}")

    return {
        "ok": True,
        "profile": profile["name"],
        "runner": runner["hostname"],
        "warnings": warnings,
        "vram_total_gb": round(vram_total / 1e9, 2) if vram_total else 0,
        "vram_free_gb": round(vram_free / 1e9, 2) if vram_free else 0,
    }


@app.post("/api/profiles/{profile_id}/deactivate")
async def deactivate_profile_endpoint(profile_id: int, runner_id: int):
    _inc_request("/api/profiles/deactivate", "POST", 200)
    await db.deactivate_profile(app.state.db, runner_id)
    return {"ok": True}


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics_endpoint():
    if app.state.db:
        try:
            apps = await get_apps(app.state.db)
            registered_apps_gauge.set(len(apps))
        except Exception:
            pass

    backend_metrics = generate_latest().decode()
    return StreamingResponse(
        iter([backend_metrics]),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
