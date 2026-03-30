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
from cloud_providers import (
    detect_provider, ModelProvider, get_anthropic_models, anthropic_chat,
    get_anthropic_api_key,
)
import api_keys
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
DISABLE_SCHEDULER = os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes")
DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")  # "production" or "uat"
UAT_TEST_RUNNER = os.environ.get("UAT_TEST_RUNNER", "")  # runner_id for UAT connectivity tests
UAT_TEST_MODEL = os.environ.get("UAT_TEST_MODEL", "")  # model name for UAT connectivity tests
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

async def _get_runner_client(
    pool: asyncpg.Pool,
    runner_id: Optional[int] = None,
    allowed_runner_ids: Optional[list[int]] = None,
) -> LLMAgentClient:
    """Return an LLMAgentClient pointed at an active (enabled) runner.
    If allowed_runner_ids is set, only those runners are candidates."""
    runners_list = await db.get_active_runners(pool)
    if allowed_runner_ids:
        runners_list = [r for r in runners_list if r["id"] in allowed_runner_ids]
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

    if DISABLE_SCHEDULER:
        logger.info("Scheduler disabled via DISABLE_SCHEDULER env var")
    elif got_lock:
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
    "/api/runners",            # Agent PSK auth + runner list
    "/api/apps/discover",      # Registration secret auth
    "/api/apps/heartbeat",     # App API key auth
    "/api/queue/jobs/",        # Job status (app API key or public)
    "/api/queue/batches/",     # Batch status
    "/api/queue/submit",       # Job submission (app API key)
    "/api/profiles/list",      # App profile discovery (API key auth)
    "/api/queue/status",       # Queue overview
    "/api/agents",             # Ecdysis agent management (behind its own Tailscale ingress)
    "/api/gpu",                # GPU info (used by ecdysis)
    "/api/models",             # Model list (used by ecdysis config)
    "/api/vram-check",         # VRAM check (used by ecdysis)
    "/api/llm/",               # LLM status, models, load/unload (UI + ecdysis)
    "/api/ops",                # Background operations status
    "/api/profiles",           # Profile management
)


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    """Require admin auth for management API routes."""
    if DISABLE_AUTH:
        return await call_next(request)

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


@app.post("/api/uat/test-model")
async def uat_test_model():
    """Send a tiny prompt to the configured UAT runner/model to verify connectivity.
    Requires UAT_TEST_RUNNER and UAT_TEST_MODEL env vars to be set."""
    if not UAT_TEST_RUNNER or not UAT_TEST_MODEL:
        raise HTTPException(
            400,
            "UAT test not configured — set UAT_TEST_RUNNER and UAT_TEST_MODEL env vars",
        )
    pool = app.state.db
    runners_list = await db.get_active_runners(pool)
    runner = next(
        (r for r in runners_list if r["name"] == UAT_TEST_RUNNER),
        None,
    )
    if not runner:
        raise HTTPException(
            503,
            f"UAT test runner '{UAT_TEST_RUNNER}' not found or not active",
        )
    ollama_base = await _get_runner_ollama_base(pool, runner["id"])
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                f"{ollama_base}/api/chat",
                json={
                    "model": UAT_TEST_MODEL,
                    "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
                    "stream": False,
                    "options": {"num_predict": 20},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return {
                "ok": True,
                "runner": UAT_TEST_RUNNER,
                "model": UAT_TEST_MODEL,
                "response": content,
                "eval_duration_ms": data.get("eval_duration", 0) // 1_000_000,
            }
    except httpx.TimeoutException:
        raise HTTPException(504, f"Model '{UAT_TEST_MODEL}' timed out on runner '{UAT_TEST_RUNNER}'")
    except Exception as e:
        raise HTTPException(502, f"Model test failed: {e}")


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
    if DISABLE_AUTH:
        return {"user": "uat", "admin": True, "environment": ENVIRONMENT}
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"user": user, "admin": user in auth.GITHUB_ALLOWED_USERS, "environment": ENVIRONMENT}


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
async def gpu_info(runner_id: Optional[int] = None):
    """GPU info from a specific runner, or the first active runner."""
    try:
        client = await _get_runner_client(app.state.db, runner_id)
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

        # Get models from ALL runners via their agent API
        all_models_map: dict = {}  # name -> model dict
        loaded_names = set()
        for runner in runners_list:
            try:
                client = await _get_runner_client(pool, runner["id"])
                result = await client.models()
                for m in result.get("data", []):
                    name = m.get("id", "")
                    if name:
                        all_models_map[name] = m
                status = await client.status()
                for lm in status.get("loaded_ollama_models", []):
                    loaded_names.add(lm["name"])
            except Exception:
                pass

        # Classify safety
        model_names = list(all_models_map.keys())
        safety_map = await classify_models_batch(pool, model_names)

        # Load library cache for enrichment (fallback for :latest tags)
        library_cache = {}
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT name, parameter_sizes FROM ollama_library_cache")
                for row in rows:
                    library_cache[row["name"]] = row["parameter_sizes"]
        except Exception:
            pass

        models = []
        for name, m in all_models_map.items():
            # Agent API returns OpenAI format (no size), fall back to VRAM estimate
            size_gb = round(m.get("size", 0) / 1e9, 2) if m.get("size") else 0
            vram_est = round(vram_for_model(name), 2)
            if size_gb == 0:
                size_gb = vram_est  # approximate
            fits_on = [
                {"runner": hostname, "vram_total_gb": vram}
                for hostname, vram in runner_vram.items()
                if vram >= vram_est
            ]
            param_count = parse_param_count(name)
            quant = parse_quantization(name)
            # For :latest tags, try to infer from library cache
            if not param_count:
                base = name.split(":")[0]
                sizes = library_cache.get(base, [])
                if isinstance(sizes, str):
                    import json
                    sizes = json.loads(sizes)
                if len(sizes) == 1:
                    param_count = sizes[0]
                elif sizes:
                    # :latest is typically the smallest or default size
                    param_count = f"{sizes[0]} (default)"
            models.append({
                "name": name,
                "size_gb": size_gb,
                "vram_estimate_gb": vram_est,
                "parameter_count": param_count,
                "quantization": quant,
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


# ── Cloud models ─────────────────────────────────────────────────────────────

@app.get("/api/cloud/models")
async def cloud_models():
    """List available cloud models with their config."""
    pool = app.state.db
    # Fetch from Anthropic API (auto-discovery), using DB key if available
    key = await get_anthropic_api_key(pool)
    api_models = await get_anthropic_models(api_key=key)
    # Merge with DB config (admin overrides)
    configs = {c["model_id"]: c for c in await db.get_cloud_model_configs(pool)}

    result = []
    for m in api_models:
        model_id = m["id"]
        cfg = configs.get(model_id, {})
        result.append({
            "id": model_id,
            "display_name": cfg.get("display_name") or m.get("display_name", model_id),
            "provider": "anthropic",
            "enabled": cfg.get("enabled", True),
            "max_tokens": cfg.get("max_tokens", 4096),
            "temperature": cfg.get("temperature"),
            "config": cfg.get("config", {}),
        })

    # Include any manually-configured models not in the API list
    api_ids = {m["id"] for m in api_models}
    for model_id, cfg in configs.items():
        if model_id not in api_ids:
            result.append({
                "id": model_id,
                "display_name": cfg.get("display_name", model_id),
                "provider": cfg.get("provider", "anthropic"),
                "enabled": cfg.get("enabled", True),
                "max_tokens": cfg.get("max_tokens", 4096),
                "temperature": cfg.get("temperature"),
                "config": cfg.get("config", {}),
            })

    return result


class CloudModelConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    display_name: Optional[str] = None
    config: Optional[dict] = None


@app.patch("/api/cloud/models/{model_id:path}")
async def update_cloud_model_config(model_id: str, req: CloudModelConfigRequest):
    """Update config for a cloud model (max_tokens, temperature, etc.)."""
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    await db.upsert_cloud_model_config(app.state.db, model_id, **kwargs)
    return {"ok": True}


@app.get("/api/cloud/status")
async def cloud_status():
    """Check cloud provider connectivity."""
    pool = app.state.db
    key = await get_anthropic_api_key(pool)
    providers = {}
    models = await get_anthropic_models(api_key=key)
    providers["anthropic"] = {
        "configured": bool(key),
        "reachable": len(models) > 0,
        "model_count": len(models),
    }
    return providers


# ── API key management ────────────────────────────────────────────────────────

@app.get("/api/cloud/keys")
async def list_cloud_keys():
    """List stored API keys (masked, never returns plaintext)."""
    return await api_keys.list_api_keys(app.state.db)


class StoreKeyRequest(BaseModel):
    provider: str
    key: str
    label: str = ""


@app.post("/api/cloud/keys")
async def store_cloud_key(req: StoreKeyRequest):
    """Encrypt and store an API key."""
    key_id = await api_keys.store_api_key(
        app.state.db, req.provider, req.key, req.label
    )
    return {"ok": True, "id": key_id}


@app.delete("/api/cloud/keys/{key_id}")
async def delete_cloud_key(key_id: int):
    """Delete a stored API key."""
    found = await api_keys.delete_api_key(app.state.db, key_id)
    if not found:
        raise HTTPException(404, "Key not found")
    return {"ok": True}


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
    """Return all recent runners (including disabled) for UI display."""
    return await db.get_all_runners(app.state.db)


class RunnerUpdateRequest(BaseModel):
    enabled: Optional[bool] = None


@app.patch("/api/runners/{runner_id}")
async def update_runner(runner_id: int, req: RunnerUpdateRequest):
    """Update runner settings (enable/disable)."""
    if req.enabled is not None:
        found = await db.set_runner_enabled(app.state.db, runner_id, req.enabled)
        if not found:
            raise HTTPException(404, "Runner not found")
    return {"ok": True}


# ── LLM proxy — all ops target active runner(s) ───────────────────────────────

def _agent_unavailable(detail: str = "No active llm-runner available") -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


@app.get("/api/llm/status")
async def llm_status(request: Request, runner_id: Optional[int] = None):
    """Status from a single runner, or aggregated from all runners.
    If a Bearer token is provided, filter to runners allowed for that app."""
    _inc_request("/api/llm/status", "GET", 200)
    pool = app.state.db

    if runner_id is not None:
        try:
            client = await _get_runner_client(pool, runner_id)
            return await client.status()
        except Exception as e:
            raise _agent_unavailable(f"Runner error: {e}")

    # If caller provides a Bearer token, filter to their allowed runners
    allowed_runner_ids: list[int] = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.removeprefix("Bearer ").strip()
        allowed_runner_ids = await db.get_app_allowed_runners(pool, api_key)

    # Aggregate from all runners
    runners_list = await db.get_active_runners(pool)
    if allowed_runner_ids:
        runners_list = [r for r in runners_list if r["id"] in allowed_runner_ids]
    if not runners_list:
        return {"runners": [], "gpu_vram_total_gb": 0, "gpu_vram_used_gb": 0,
                "loaded_ollama_models": [], "cpu_pct": 0, "mem_total_gb": 0,
                "mem_used_gb": 0, "gpu_vram_pct": 0}

    runner_statuses = []
    all_loaded_models = []
    total_vram = 0.0
    used_vram = 0.0
    total_cpu = 0.0
    total_mem = 0.0
    used_mem = 0.0

    for r in runners_list:
        try:
            client = await _get_runner_client(pool, r["id"])
            status = await client.status()
            status["runner_id"] = r["id"]
            status["runner_hostname"] = r["hostname"]
            runner_statuses.append(status)

            rv_total = status.get("gpu_vram_total_gb", 0)
            rv_used = status.get("gpu_vram_used_gb", 0)
            total_vram += rv_total
            used_vram += rv_used
            total_cpu += status.get("cpu_pct", 0)
            total_mem += status.get("mem_total_gb", 0)
            used_mem += status.get("mem_used_gb", 0)

            for m in status.get("loaded_ollama_models", []):
                m["runner"] = r["hostname"]
                all_loaded_models.append(m)
        except Exception:
            runner_statuses.append({
                "runner_id": r["id"],
                "runner_hostname": r["hostname"],
                "error": "unreachable",
                "gpu_vram_total_gb": 0,
                "gpu_vram_used_gb": 0,
            })

    vram_pct = round((used_vram / total_vram * 100) if total_vram > 0 else 0, 1)

    return {
        "runners": runner_statuses,
        "gpu_vram_total_gb": round(total_vram, 2),
        "gpu_vram_used_gb": round(used_vram, 2),
        "gpu_vram_pct": vram_pct,
        "loaded_ollama_models": all_loaded_models,
        "cpu_pct": round(total_cpu / len(runners_list), 1) if runners_list else 0,
        "mem_total_gb": round(total_mem, 2),
        "mem_used_gb": round(used_mem, 2),
    }


@app.get("/api/llm/models")
async def llm_models(runner_id: Optional[int] = None):
    """Models from a single runner or aggregated from all.

    When aggregating, returns per-model runner presence so the UI can show
    which models are on which runners.
    """
    _inc_request("/api/llm/models", "GET", 200)
    pool = app.state.db

    if runner_id is not None:
        try:
            runners_list = await db.get_active_runners(pool)
            r = next((x for x in runners_list if x["id"] == runner_id), None)
            hostname = r["hostname"] if r else "unknown"
            client = await _get_runner_client(pool, runner_id)
            result = await client.models()
            for m in result.get("data", []):
                m["runners"] = [{"runner_id": runner_id, "hostname": hostname}]
            return result
        except Exception as e:
            raise _agent_unavailable(f"Runner error: {e}")

    # Aggregate from all runners — track which runners have each model
    runners_list = await db.get_active_runners(pool)
    model_map: dict[str, dict] = {}  # model_id -> model data with runners list

    for r in runners_list:
        try:
            client = await _get_runner_client(pool, r["id"])
            result = await client.models()
            for m in result.get("data", []):
                mid = m.get("id")
                if mid not in model_map:
                    model_map[mid] = {**m, "runners": []}
                model_map[mid]["runners"].append({
                    "runner_id": r["id"],
                    "hostname": r["hostname"],
                })
        except Exception:
            pass

    return {"data": list(model_map.values())}


class LLMPullRequest(BaseModel):
    model: str


@app.post("/api/llm/models/pull")
async def llm_pull_model(req: LLMPullRequest, runner_id: Optional[int] = None):
    """Pull a model. Runs in background — returns immediately with operation ID."""
    _inc_request("/api/llm/models/pull", "POST", 200)

    # Pre-check: warn if disk is low on the target runner
    disk_warning = None
    try:
        client = await _get_runner_client(app.state.db, runner_id)
        st = await client.status()
        disk_free_gb = st.get("disk_free_gb", None)
        model_vram = vram_for_model(req.model)
        # Rough heuristic: disk needed ≈ VRAM estimate (quantized weights on disk)
        if disk_free_gb is not None and model_vram > 0 and disk_free_gb < model_vram:
            raise HTTPException(
                507,
                f"Not enough disk space on {st.get('node', 'runner')}: "
                f"{disk_free_gb:.1f} GB free, ~{model_vram:.1f} GB needed for {req.model}"
            )
        elif disk_free_gb is not None and disk_free_gb < 5:
            disk_warning = f"Low disk space: {disk_free_gb:.1f} GB free"
    except HTTPException:
        raise
    except Exception:
        pass  # non-critical, proceed with pull

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
    msg = f"Pulling {req.model} in background"
    if disk_warning:
        msg += f" (warning: {disk_warning})"
    return {"ok": True, "op_id": op_id, "message": msg}


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


@app.post("/api/llm/models/sync")
@app.post("/api/llm/models/mirror")  # backwards compat
async def sync_models():
    """Sync models across all runners.

    For each model base name (e.g. 'qwen2.5') downloaded on any runner,
    ensure every other runner that can fit it has the biggest weight variant
    that fits in its VRAM. If the runner already has that model (any weight),
    skip it.
    """
    _inc_request("/api/llm/models/sync", "POST", 200)
    pool = app.state.db
    runners_list = await db.get_active_runners(pool)
    if len(runners_list) < 2:
        return {"ok": True, "pulls": [], "message": "Need at least 2 runners to sync"}

    # Gather per-runner downloaded models and VRAM capacity
    runner_models: dict[int, set[str]] = {}
    runner_vram: dict[int, float] = {}
    runner_names: dict[int, str] = {}

    for r in runners_list:
        rid = r["id"]
        hostname = r["hostname"]
        runner_names[rid] = hostname
        runner_models[rid] = set()
        caps = r.get("capabilities", {})
        if isinstance(caps, dict):
            runner_vram[rid] = caps.get("gpu_vram_total_bytes", 0) / (1024**3)
        else:
            runner_vram[rid] = 0

        try:
            addr = r["address"]
            host = re.sub(r'^https?://', '', addr)
            host = re.sub(r':\d+$', '', host)
            ollama_base = f"http://{host}:11434"
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get(f"{ollama_base}/api/tags")
                if resp.status_code == 200:
                    for m in resp.json().get("models", []):
                        runner_models[rid].add(m["name"])
        except Exception:
            logger.warning("sync: failed to query runner %s", hostname)

    # Extract base model names (e.g. 'qwen2.5' from 'qwen2.5:7b')
    def _base_name(model: str) -> str:
        return model.split(":")[0]

    # For each base model present on any runner, find the best weight for each target
    all_base_names = set()
    for models in runner_models.values():
        for m in models:
            all_base_names.add(_base_name(m))

    pulls = []
    for base in all_base_names:
        # Collect all weight variants of this model across all runners
        variants = set()
        for models in runner_models.values():
            for m in models:
                if _base_name(m) == base:
                    variants.add(m)

        for target_rid in runner_models:
            # Skip if this runner already has any variant of this base model
            target_has = [m for m in runner_models[target_rid] if _base_name(m) == base]
            if target_has:
                continue

            # Find the biggest variant that fits on this runner's VRAM
            target_cap = runner_vram.get(target_rid, 0)
            best_variant = None
            best_vram = 0.0
            for v in variants:
                v_vram = vram_for_model(v)
                if v_vram <= target_cap and v_vram > best_vram:
                    best_variant = v
                    best_vram = v_vram

            if best_variant:
                # Find a source runner that has this variant
                source_name = "unknown"
                for src_rid, models in runner_models.items():
                    if best_variant in models:
                        source_name = runner_names[src_rid]
                        break
                pulls.append({
                    "model": best_variant,
                    "target_runner_id": target_rid,
                    "target_runner": runner_names[target_rid],
                    "source_runner": source_name,
                })

    # Deduplicate (same model+target)
    seen = set()
    unique_pulls = []
    for p in pulls:
        key = (p["model"], p["target_runner_id"])
        if key not in seen:
            seen.add(key)
            unique_pulls.append(p)

    # Trigger pulls in background
    for p in unique_pulls:
        op_id = f"sync-{p['model']}-{p['target_runner_id']}"
        _ops[op_id] = {"status": "running", "model": p["model"], "type": "sync",
                       "target": p["target_runner"], "progress": ""}

        async def _do_sync_pull(model=p["model"], rid=p["target_runner_id"], oid=op_id):
            try:
                client = await _get_runner_client(pool, rid)
                async for chunk in client.pull_model(model):
                    _ops[oid]["progress"] = chunk.decode().strip()
                _ops[oid]["status"] = "completed"
            except Exception as e:
                _ops[oid]["status"] = "failed"
                _ops[oid]["error"] = str(e)

        asyncio.create_task(_do_sync_pull())

    return {
        "ok": True,
        "pulls": [{"model": p["model"], "target": p["target_runner"]} for p in unique_pulls],
        "message": f"Syncing {len(unique_pulls)} model(s) across runners",
    }


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
    """Delete a model from disk on a runner."""
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

    # Enforce per-app model + runner restrictions
    app_allowed_runners: list[int] = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.removeprefix("Bearer ").strip()
        allowed = await db.check_model_allowed(app.state.db, api_key, model)
        if not allowed:
            raise HTTPException(403, f"Model '{model}' is not allowed for this application")
        app_allowed_runners = await db.get_app_allowed_runners(app.state.db, api_key)

    provider = detect_provider(model)
    _inc_request("/v1/chat/completions", "POST", 200)

    # ── Cloud model routing ──────────────────────────────────────────────
    if provider == ModelProvider.ANTHROPIC:
        # Check if model is enabled
        cfg = await db.get_cloud_model_config(app.state.db, model)
        if cfg and not cfg.get("enabled", True):
            raise HTTPException(403, f"Cloud model '{model}' is disabled")

        config_overrides = {}
        if cfg:
            if cfg.get("max_tokens"):
                config_overrides["max_tokens"] = cfg["max_tokens"]
            if cfg.get("temperature") is not None:
                config_overrides["temperature"] = cfg["temperature"]

        try:
            key = await get_anthropic_api_key(app.state.db)
            result = await anthropic_chat(body, api_key=key, stream=stream, config_overrides=config_overrides)
            if stream:
                return StreamingResponse(result, media_type="text/event-stream")
            return result
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            _inc_request("/v1/chat/completions", "POST", status)
            raise HTTPException(status, f"Cloud model error: {e.response.text[:200]}")
        except ValueError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            _inc_request("/v1/chat/completions", "POST", 503)
            raise HTTPException(503, f"Cloud model error: {e}")

    # ── Local model routing (Ollama) ─────────────────────────────────────
    try:
        client = await _get_runner_client(
            app.state.db, runner_id,
            allowed_runner_ids=app_allowed_runners or None,
        )

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
        a_copy.setdefault("allowed_runner_ids", [])
        a_copy["allowed_runner_ids"] = list(a_copy["allowed_runner_ids"] or [])
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


class AppAllowedRunnersRequest(BaseModel):
    allowed_runner_ids: list[int]


@app.get("/api/apps/{app_id}/allowed-runners")
async def get_app_allowed_runners_endpoint(app_id: int):
    apps = await db.get_apps(app.state.db)
    a = next((x for x in apps if x["id"] == app_id), None)
    if not a:
        raise HTTPException(404, "App not found")
    runner_ids = list(a.get("allowed_runner_ids") or [])
    return {"app_id": app_id, "allowed_runner_ids": runner_ids, "unrestricted": len(runner_ids) == 0}


@app.put("/api/apps/{app_id}/allowed-runners")
async def set_app_allowed_runners_endpoint(app_id: int, req: AppAllowedRunnersRequest):
    found = await db.set_app_allowed_runners(app.state.db, app_id, req.allowed_runner_ids)
    if not found:
        raise HTTPException(404, "App not found")
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
            await client.unload_model_from_vram(req.model)
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
                        await client.unload_model_from_vram(m["name"])
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
