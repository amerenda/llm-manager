"""
LLM Manager Backend API.
Runs as a stateless k8s Deployment on port 8081.
LLM proxy, model management, app registry, and queue scheduler.
LLM runners self-register via PSK instead of being polled.
"""

import asyncio
import json
import logging
import os
import re
import socket
import time
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
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

import db
from db import (
    init_db,
    register_app, heartbeat_app, get_apps, deregister_app, deregister_app_by_id,
)
from gpu import vram_for_model
from agent_version_compare import agent_versions_equivalent
from llm_agent import LLMAgentClient, client_from_runner_row
from scheduler_v2 import SimplifiedScheduler as Scheduler
from scheduler_v2 import fastpath_duration_seconds, fastpath_requests_total

_SCHEDULER_VARIANT = "v2"
from queue_routes import router as queue_router, model_router, alias_router
from library_routes import router as library_router, safety_router
from library import classify_models_batch, parse_param_count, parse_quantization, refresh_library_cache
import queue_db
import queue_policy
import auth
from cloud_providers import (
    detect_provider, ModelProvider, get_anthropic_models, anthropic_chat,
    get_anthropic_api_key,
)
import api_keys
from auth import (
    GITHUB_CLIENT_ID, COOKIE_NAME, SESSION_TTL,
    create_session_token, get_current_user,
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

# Background operations are now stored in PostgreSQL (background_ops table)
# to ensure consistent state across multiple backend replicas.

# ── Prometheus metrics ────────────────────────────────────────────────────────

api_requests_total = Counter(
    "llm_backend_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
registered_apps_gauge = Gauge("llm_backend_registered_apps", "Number of registered apps")
active_runners_gauge = Gauge("llm_backend_active_runners", "Number of active runners")
runner_last_seen_seconds = Gauge(
    "llm_backend_runner_last_seen_seconds", "Seconds since runner last heartbeat", ["runner"])
library_cache_age_seconds = Gauge(
    "llm_library_cache_age_seconds",
    "Seconds since the Ollama library cache was last successfully refreshed. "
    "Alert if > 48h — the refresh CronJob has likely been failing silently.",
)

# Queue metrics
queue_jobs_submitted = Counter("llm_queue_jobs_submitted_total", "Total jobs submitted to queue", ["model", "app"])
queue_jobs_completed = Counter("llm_queue_jobs_completed_total", "Total completed queue jobs", ["model", "status"])
queue_depth_gauge = Gauge("llm_queue_depth", "Current queue depth")
queue_active_jobs_gauge = Gauge("llm_queue_active_jobs", "Currently running jobs")
queue_loading_jobs_gauge = Gauge("llm_queue_loading_jobs", "Jobs waiting for model load")
queue_job_duration_seconds = Histogram("llm_queue_job_duration_seconds", "Job processing time", ["model"], buckets=[1, 5, 10, 30, 60, 120, 300])
queue_job_wait_seconds = Histogram("llm_queue_job_wait_seconds", "Time job spent waiting in queue", ["model"], buckets=[0.5, 1, 5, 10, 30, 60, 120])
queue_submission_errors_total = Counter(
    "llm_queue_submission_errors_total", "Queue submission errors", ["reason"])

# Config info metrics — refreshed by background task every 30s.
# Numeric params exposed as gauge values so they can be graphed/alerted on.
# String metadata (alias→base_model) carried as labels on the info gauge.
alias_info_gauge = Gauge(
    "llm_alias_info",
    "Alias metadata (value=1, use label values for config display)",
    ["alias", "base_model", "has_system_prompt"],
)
alias_param_gauge = Gauge(
    "llm_alias_param",
    "Numeric parameter for an alias (temperature, num_ctx, etc.)",
    ["alias", "param"],
)
runner_model_param_gauge = Gauge(
    "llm_runner_model_param",
    "Per-runner model parameter override",
    ["model", "runner", "param"],
)


def _inc_request(endpoint: str, method: str, status: int):
    api_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()


# Shorter timeouts when aggregating many runners — avoids 30s × N sequential stalls.
_AGENT_AGG_TIMEOUT = httpx.Timeout(5.0, read=25.0)


def _llm_agent_client_for_runner_row(r: dict) -> LLMAgentClient:
    """Build an agent client from a DB runner row (no extra query)."""
    return client_from_runner_row(r, AGENT_PSK)


# ── Runner helpers ─────────────────────────────────────────────────────────────

async def _get_runner_client(
    pool: asyncpg.Pool,
    runner_id: Optional[int] = None,
    allowed_runner_ids: Optional[list[int]] = None,
) -> LLMAgentClient:
    """Return an LLMAgentClient pointed at an active (enabled) runner.
    If allowed_runner_ids is set, only those runners are candidates.

    When runner_id is None (fallback path used by /api/llm/models/load,
    /api/llm/models/unload, library pull helpers, etc.) draining runners
    are excluded. An explicit runner_id overrides — admins can still
    target a drained runner for maintenance ops.

    Regression 2026-04-22: archlinux was drained, alphabetical ordering
    in get_active_runners put it first, and a UI "load model" test
    landed a 35b model on it anyway — archlinux's 17 GB GPU overflowed
    to RAM and the host died."""
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
        eligible = [x for x in runners_list if not x.get("draining")]
        if not eligible:
            raise HTTPException(503, "No active llm-runners available (all are draining)")
        r = eligible[0]
    return _llm_agent_client_for_runner_row(r)


async def _get_runner_ollama_base(pool: asyncpg.Pool, runner_id: Optional[int] = None) -> str:
    """Get Ollama URL for a runner. Replaces the runner port with 11434.
    Ollama always uses plain HTTP regardless of agent protocol.

    Draining-aware: see _get_runner_client for the reasoning."""
    runners_list = await db.get_active_runners(pool)
    if not runners_list:
        raise HTTPException(503, "No active llm-runners available")
    if runner_id is not None:
        r = next((x for x in runners_list if x["id"] == runner_id), None)
        if not r:
            eligible = [x for x in runners_list if not x.get("draining")]
            r = eligible[0] if eligible else runners_list[0]
    else:
        eligible = [x for x in runners_list if not x.get("draining")]
        if not eligible:
            raise HTTPException(503, "No active llm-runners available (all are draining)")
        r = eligible[0]
    # runner address is like https://10.x.x.x:8090
    # ollama is on the same host at port 11434, always plain HTTP
    addr = r["address"]
    # Strip scheme and port, rebuild as http with Ollama port
    host = re.sub(r'^https?://', '', addr)
    host = re.sub(r':\d+$', '', host)
    return f"http://{host}:11434"


# ── Lifespan ──────────────────────────────────────────────────────────────────

SCHEDULER_LOCK_ID = 900001  # Postgres advisory lock ID for scheduler
STALE_SWEEP_LOCK_ID = 900002  # Only one replica runs stale in-progress recovery per tick

@asynccontextmanager
async def lifespan(app: FastAPI):
    # TCP keepalives on every pooled connection reduce stale half-open TCP after
    # network blips; max_inactive_connection_lifetime retires idle pool conns.
    _pg_server_settings = {
        "tcp_keepalives_idle": "60",
        "tcp_keepalives_interval": "10",
        "tcp_keepalives_count": "3",
    }
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        max_inactive_connection_lifetime=float(
            os.environ.get("DB_POOL_MAX_INACTIVE_SEC", "300")
        ),
        server_settings=_pg_server_settings,
    )
    app.state.db = pool
    await init_db(pool)
    await queue_db.init_queue_tables(pool)
    logger.info("Database connected: %s", DATABASE_URL)

    # Acquire advisory lock for scheduler — only one pod runs it.
    # TCP keepalives ensure the server-side session dies within ~90s if the
    # client disappears (network blip). Without this, PostgreSQL keeps the
    # zombie session alive for hours, holding the advisory lock and silently
    # blocking all watchdog re-acquisition attempts.
    lock_conn = await asyncpg.connect(
        DATABASE_URL, server_settings=_pg_server_settings
    )
    got_lock = await lock_conn.fetchval(
        "SELECT pg_try_advisory_lock($1)", SCHEDULER_LOCK_ID
    )
    app.state.lock_conn = lock_conn

    async def get_runner(runner_id=None):
        return await _get_runner_client(pool, runner_id=runner_id)
    scheduler = Scheduler(pool, get_runner, lock_conn=lock_conn if got_lock else None)
    app.state.scheduler = scheduler
    logger.info("Scheduler variant: %s", _SCHEDULER_VARIANT)

    if DISABLE_SCHEDULER:
        logger.info("Scheduler disabled via DISABLE_SCHEDULER env var")
    elif got_lock:
        recovered = await queue_db.recover_stuck_jobs(pool)
        if recovered:
            logger.warning("Recovered %d jobs stuck in loading_model/running → queued", recovered)
        orphaned = await db.recover_stuck_ops(pool)
        if orphaned:
            logger.warning("Marked %d background ops as failed (orphaned by previous pod)", orphaned)
        scheduler.start()
        logger.info("Queue scheduler started (advisory lock acquired)")
    else:
        logger.info("Scheduler skipped — another pod holds the lock")

    # Background watchdog: acquire the advisory lock (if not held) and start the
    # scheduler. Runs on ALL pods for the lifetime of the process so that:
    # 1. A standby pod picks up the lock when the primary loses it.
    # 2. The primary pod can recover if its own lock_conn dies while idle.
    #
    # Root-cause of the 2026-04-28 outage: the old _retry_lock returned after
    # first acquiring the lock, leaving the scheduler unmonitored. When the
    # primary pod's lock_conn was eventually closed by PostgreSQL (idle
    # connection cleanup), the advisory lock was released but no pod attempted
    # to re-acquire it. The standby pod's _retry_lock had a dead lock_conn from
    # an earlier disconnect and silently ate every exception, making it
    # permanently stuck. Result: scheduler dead on all pods, jobs stuck queued.
    async def _retry_lock():
        nonlocal got_lock, lock_conn
        while not DISABLE_SCHEDULER:
            await asyncio.sleep(15)

            # If we hold the lock and the scheduler is running, nothing to do.
            if got_lock and scheduler._running:
                continue

            # We either don't hold the lock, or the scheduler stopped. In either
            # case, try to acquire the lock on a fresh connection — the existing
            # lock_conn may be dead (that's often why we're here).
            if got_lock and not scheduler._running:
                logger.warning(
                    "Scheduler stopped unexpectedly — releasing lock state "
                    "and attempting to re-acquire"
                )
                got_lock = False

            fresh_conn = None
            try:
                fresh_conn = await asyncpg.connect(
                    DATABASE_URL, server_settings=_pg_server_settings
                )
                acquired = await fresh_conn.fetchval(
                    "SELECT pg_try_advisory_lock($1)", SCHEDULER_LOCK_ID
                )
                if not acquired:
                    # Another pod holds the lock — check again next tick.
                    await fresh_conn.close()
                    continue

                # We acquired the lock. Replace the old (possibly dead) lock_conn.
                try:
                    await lock_conn.close()
                except Exception:
                    pass
                lock_conn = fresh_conn
                app.state.lock_conn = lock_conn
                scheduler.lock_conn = lock_conn
                got_lock = True
                fresh_conn = None  # don't double-close in finally

                # Recover jobs the previous lock holder may have orphaned
                # mid-swap/mid-run. Without this, jobs stuck in
                # 'loading_model' or 'running' are never picked up since
                # get_pending_jobs only returns 'queued' statuses.
                try:
                    recovered = await queue_db.recover_stuck_jobs(pool)
                    if recovered:
                        logger.warning(
                            "Recovered %d jobs stuck in loading_model/running → queued",
                            recovered,
                        )
                except Exception:
                    logger.exception("recover_stuck_jobs failed on lock retry")
                # Same for background_ops (pull/update/sync): asyncio tasks
                # don't survive a pod restart, so any op still marked
                # 'running' from the previous pod is a zombie.
                try:
                    orphaned = await db.recover_stuck_ops(pool)
                    if orphaned:
                        logger.warning(
                            "Marked %d background ops as failed (orphaned by previous pod)",
                            orphaned,
                        )
                except Exception:
                    logger.exception("recover_stuck_ops failed on lock retry")
                scheduler.start()
                logger.info("Queue scheduler started (advisory lock acquired on retry)")

            except Exception:
                logger.exception("lock_retry: error attempting to acquire scheduler lock")
                if fresh_conn is not None:
                    try:
                        await fresh_conn.close()
                    except Exception:
                        pass

    lock_retry_task = asyncio.create_task(_retry_lock()) if not DISABLE_SCHEDULER else None

    async def _stale_queue_sweeper():
        """Re-queue wedged loading_model/running rows. Uses a short-lived
        advisory lock so only one replica performs the sweep per interval."""
        interval = float(os.environ.get("STALE_QUEUE_SWEEP_INTERVAL_SEC", "60"))
        lm = int(os.environ.get("SCHEDULER_STALE_LOADING_MIN", "45"))
        rm = int(os.environ.get("SCHEDULER_STALE_RUNNING_MIN", "360"))
        fb = int(os.environ.get("SCHEDULER_STALE_LOADING_FALLBACK_HOURS", "24"))
        while True:
            await asyncio.sleep(interval)
            if DISABLE_SCHEDULER:
                continue
            try:
                async with pool.acquire() as conn:
                    got = await conn.fetchval(
                        "SELECT pg_try_advisory_lock($1)", STALE_SWEEP_LOCK_ID
                    )
                    if not got:
                        continue
                    try:
                        await queue_db.recover_stale_in_progress_jobs(
                            pool,
                            loading_minutes=lm,
                            running_minutes=rm,
                            loading_fallback_hours=fb,
                        )
                    finally:
                        try:
                            await conn.execute(
                                "SELECT pg_advisory_unlock($1)", STALE_SWEEP_LOCK_ID
                            )
                        except Exception:
                            logger.warning(
                                "stale_queue_sweeper: advisory unlock failed",
                                exc_info=True,
                            )
            except Exception:
                logger.exception("stale_queue_sweeper failed")

    stale_sweep_task = (
        asyncio.create_task(_stale_queue_sweeper()) if not DISABLE_SCHEDULER else None
    )

    # Track which label sets we've written so we can remove stale ones on refresh
    _alias_info_labels: set[tuple] = set()
    _alias_param_labels: set[tuple] = set()
    _runner_param_labels: set[tuple] = set()

    async def _refresh_config_metrics():
        nonlocal _alias_info_labels, _alias_param_labels, _runner_param_labels
        _NUMERIC_PARAMS = ("temperature", "top_p", "top_k", "num_ctx",
                           "repeat_penalty", "seed", "max_tokens", "num_predict",
                           "top_k", "frequency_penalty", "presence_penalty")
        while True:
            try:
                aliases = await queue_db.get_all_model_aliases(pool)
                new_info: set[tuple] = set()
                new_param: set[tuple] = set()

                for a in aliases:
                    alias = a["alias_name"]
                    base = a["base_model"]
                    has_sp = "1" if a.get("system_prompt") else "0"
                    key = (alias, base, has_sp)
                    alias_info_gauge.labels(alias=alias, base_model=base, has_system_prompt=has_sp).set(1)
                    new_info.add(key)

                    params = a.get("parameters") or {}
                    if isinstance(params, str):
                        import json as _j; params = _j.loads(params)
                    for p in _NUMERIC_PARAMS:
                        if p in params:
                            try:
                                alias_param_gauge.labels(alias=alias, param=p).set(float(params[p]))
                                new_param.add((alias, p))
                            except (TypeError, ValueError):
                                pass

                # Remove stale alias entries
                for key in _alias_info_labels - new_info:
                    try:
                        alias_info_gauge.remove(*key)
                    except Exception:
                        pass
                for key in _alias_param_labels - new_param:
                    try:
                        alias_param_gauge.remove(*key)
                    except Exception:
                        pass
                _alias_info_labels = new_info
                _alias_param_labels = new_param

                # Runner-level model param overrides
                # Query all model_runner_params rows via the DB directly
                new_rp: set[tuple] = set()
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT mrp.model_name, r.hostname, mrp.parameters
                           FROM model_runner_params mrp
                           JOIN llm_runners r ON r.id = mrp.runner_id
                           WHERE mrp.parameters IS NOT NULL AND mrp.parameters != 'null'"""
                    )
                for row in rows:
                    m = row["model_name"]
                    runner = row["hostname"]
                    params = row["parameters"]
                    if isinstance(params, str):
                        import json as _j
                        try:
                            params = _j.loads(params)
                        except Exception:
                            params = {}
                    if not isinstance(params, dict):
                        continue
                    for p in _NUMERIC_PARAMS:
                        if p in params:
                            try:
                                runner_model_param_gauge.labels(model=m, runner=runner, param=p).set(float(params[p]))
                                new_rp.add((m, runner, p))
                            except (TypeError, ValueError):
                                pass
                for key in _runner_param_labels - new_rp:
                    try:
                        runner_model_param_gauge.remove(*key)
                    except Exception:
                        pass
                _runner_param_labels = new_rp

            except Exception:
                logger.exception("config metrics refresh failed")
            await asyncio.sleep(30)

    config_metrics_task = asyncio.create_task(_refresh_config_metrics())

    yield

    if lock_retry_task and not lock_retry_task.done():
        lock_retry_task.cancel()
    if stale_sweep_task and not stale_sweep_task.done():
        stale_sweep_task.cancel()
    config_metrics_task.cancel()

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
app.include_router(alias_router)
app.include_router(library_router)
app.include_router(safety_router)

# ── Auth middleware ───────────────────────────────────────────────────────────
# Routes that DON'T need admin auth (they have their own auth or are public)
_PUBLIC_PATHS = {
    "/health", "/metrics", "/auth/login", "/auth/callback", "/auth/me",
    "/auth/logout", "/api/stats",
}

_APP_KEY_AUTH_PREFIXES = (
    "/v1/",
    "/api/apps/heartbeat",
    "/api/queue/submit",
    "/api/queue/jobs/",
    "/api/queue/batches/",
    "/api/profiles/list",
)


def _session_admin_user(request: Request) -> Optional[str]:
    user = get_current_user(request)
    if user and user in auth.GITHUB_ALLOWED_USERS:
        return user
    return None


def _allow_anonymous_request(path: str, method: str) -> bool:
    """True if this request may proceed without a GitHub admin session.

    Mutations (load/unload, runner settings, profiles, library writes, etc.)
    still require admin auth or an app API key on the specific routes below.
    """
    m = method.upper()

    # Runner agents (PSK validated inside the route handlers)
    if path == "/api/runners/register" and m == "POST":
        return True
    if path == "/api/runners/heartbeat" and m == "POST":
        return True

    # Public read — fleet / catalog (GET /api/runners redacts internal agent URLs
    # when the caller is not an admin; see list_runners).
    if path == "/api/runners" and m == "GET":
        return True
    if path == "/api/runners/target-version" and m == "GET":
        return True

    if path == "/api/models" and m == "GET":
        return True
    if path == "/api/gpu" and m == "GET":
        return True
    if path == "/api/vram-check" and m == "POST":
        return True

    # LLM aggregate read-only (per-runner admin tools stay session-gated)
    if path == "/api/llm/status" and m == "GET":
        return True
    if path == "/api/llm/models" and m == "GET":
        return True
    if path == "/api/llm/checkpoints" and m == "GET":
        return True

    # Cron / automation — exact path only (avoid exposing other /api/library/* writes)
    if path == "/api/library/refresh" and m == "POST":
        return True

    # OpenAI-compatible proxy (app API key inside route / v1 handlers)
    if path.startswith("/v1/"):
        return True

    # Queue — reads and app-key submission; cancel/priority need admin (handled below)
    if path.startswith("/api/queue/jobs/") and m not in ("DELETE", "PATCH"):
        return True
    if path.startswith("/api/queue/batches/"):
        return True
    if path.startswith("/api/queue/submit") and m == "POST":
        return True
    if path == "/api/queue/status" and m == "GET":
        return True

    # App self-service (credentials in body or Bearer token)
    if path == "/api/apps/discover" and m == "POST":
        return True
    if path == "/api/apps/heartbeat" and m == "POST":
        return True

    # App profile discovery (Bearer app API key)
    if path == "/api/profiles/list" and m == "GET":
        return True

    # Moltbook / agent slots (separate ingress hardening in prod)
    if path.startswith("/api/agents"):
        return True

    return False


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    """Require admin auth for management API routes."""
    if DISABLE_AUTH:
        return await call_next(request)

    path = request.url.path
    method = request.method

    # Skip auth for public/self-authenticated routes
    # Queue job cancel (DELETE) and priority (PATCH) require admin auth
    if path in _PUBLIC_PATHS or _allow_anonymous_request(path, method):
        if path.startswith("/api/queue/jobs/") and method in ("DELETE", "PATCH"):
            pass  # fall through to admin auth check below
        else:
            return await call_next(request)

    # Skip auth for non-API routes (docs, openapi.json, etc.)
    if not path.startswith("/api/"):
        return await call_next(request)

    # Allow app API key auth only on explicitly app-authenticated routes.
    auth_header = request.headers.get("Authorization", "")
    if path.startswith(_APP_KEY_AUTH_PREFIXES) and auth_header.startswith("Bearer "):
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

        # Get models from ALL runners via their agent API (parallel — was sequential 30s×2×N)
        all_models_map: dict = {}  # name -> model dict
        model_runners: dict = {}  # name -> [{runner_id, hostname}]
        loaded_names = set()

        async def _runner_catalog(runner: dict) -> None:
            try:
                client = _llm_agent_client_for_runner_row(runner)
                result, status = await asyncio.gather(
                    client.models(timeout=_AGENT_AGG_TIMEOUT),
                    client.status(timeout=_AGENT_AGG_TIMEOUT),
                )
                for m in result.get("data", []):
                    name = m.get("id", "")
                    if name:
                        all_models_map[name] = m
                        model_runners.setdefault(name, []).append({
                            "runner_id": runner["id"],
                            "hostname": runner["hostname"],
                        })
                for lm in status.get("loaded_ollama_models", []):
                    loaded_names.add(lm["name"])
            except Exception:
                pass

        await asyncio.gather(*(_runner_catalog(r) for r in runners_list))

        # Classify safety
        model_names = list(all_models_map.keys())
        safety_map = await classify_models_batch(pool, model_names)

        # Load model_settings for categories/safety overrides
        model_settings_map: dict = {}
        try:
            rows = await queue_db.get_all_model_settings(pool)
            for r in rows:
                model_settings_map[r["model_name"]] = r
        except Exception:
            pass

        # Load library cache for enrichment (fallback for :latest tags) and categories
        library_cache = {}
        library_categories: dict = {}
        library_descriptions: dict = {}
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT name, description, parameter_sizes, categories FROM ollama_library_cache")
                for row in rows:
                    library_cache[row["name"]] = row["parameter_sizes"]
                    library_descriptions[row["name"]] = row["description"] or ""
                    cats = row["categories"]
                    if isinstance(cats, str):
                        import json
                        cats = json.loads(cats)
                    library_categories[row["name"]] = cats or []
        except Exception:
            pass

        def _is_chat_model(model_name: str) -> bool:
            """Return False for embedding and vision-only models — not usable for chat."""
            n = model_name.lower().split(":")[0]
            # Embedding models (bert-family, nomic, minilm, bge, etc.)
            if "embed" in n or "minilm" in n or n.startswith("bge-"):
                return False
            # Vision-only (non-chat) models
            if n in ("clip",) or n.startswith("clip-"):
                return False
            return True

        models = []
        for name, m in all_models_map.items():
            if not _is_chat_model(name):
                continue
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
            # Categories: prefer model_settings override, else library cache by base name
            ms = model_settings_map.get(name, {})
            base = name.split(":")[0]
            categories = list(ms.get("categories") or [])
            if not categories:
                categories = library_categories.get(base, [])
            # Safety: prefer model_settings override, else pattern-based classification
            safety = ms.get("safety") or safety_map.get(name, "safe")
            # Description from library cache
            description = library_descriptions.get(base, "")
            models.append({
                "name": name,
                "size_gb": size_gb,
                "vram_estimate_gb": vram_est,
                "parameter_count": param_count,
                "quantization": quant,
                "safety": safety,
                "categories": categories,
                "description": description,
                "runners": model_runners.get(name, []),
                "downloaded": True,
                "loaded": name in loaded_names,
                "fits": len(fits_on) > 0,
                "fits_on": fits_on,
                "do_not_evict": bool(ms.get("do_not_evict", False)),
            })
        # Sort: models that fit first, then by VRAM estimate
        models.sort(key=lambda m: (not m["fits"], m["vram_estimate_gb"]))

        # Append alias entries so the UI can show/manage them
        try:
            aliases = await queue_db.get_all_model_aliases(pool)
            base_map = {m["name"]: m for m in models}
            for alias in aliases:
                base = base_map.get(alias["base_model"])
                params = alias.get("parameters") or {}
                if isinstance(params, str):
                    params = json.loads(params)
                models.append({
                    "name": alias["alias_name"],
                    "is_alias": True,
                    "base_model": alias["base_model"],
                    "alias_description": alias.get("description", ""),
                    "alias_parameters": params,
                    "alias_system_prompt": alias.get("system_prompt"),
                    "size_gb": base["size_gb"] if base else 0,
                    "vram_estimate_gb": base["vram_estimate_gb"] if base else 0,
                    "parameter_count": base["parameter_count"] if base else None,
                    "quantization": base["quantization"] if base else None,
                    "safety": base["safety"] if base else "safe",
                    "categories": base["categories"] if base else [],
                    "description": alias.get("description", ""),
                    "runners": base["runners"] if base else [],
                    "downloaded": bool(base),
                    "loaded": base["loaded"] if base else False,
                    "fits": base["fits"] if base else False,
                    "fits_on": base["fits_on"] if base else [],
                    "do_not_evict": base["do_not_evict"] if base else False,
                })
        except Exception:
            pass

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
    # Check if there's a target version for this runner
    response: dict = {"ok": True}
    target = await db.get_global_setting(app.state.db, "agent_target_version")
    agent_version = req.capabilities.get("agent_version", "")
    if target and agent_version and not agent_versions_equivalent(target, agent_version):
        response["update_to"] = target.strip()
    # Include auto_update preference so agent knows whether to self-update
    runner = await db.get_runner_by_id(app.state.db, req.runner_id)
    if runner:
        response["auto_update"] = runner.get("auto_update", False)
    return response


@app.get("/api/runners/target-version")
async def get_target_version():
    target = await db.get_global_setting(app.state.db, "agent_target_version")
    return {"target_version": target or ""}


class SetTargetVersionRequest(BaseModel):
    target_version: str


@app.put("/api/runners/target-version")
async def set_target_version(req: SetTargetVersionRequest):
    ver = (req.target_version or "").strip()
    await db.set_global_setting(app.state.db, "agent_target_version", ver)
    return {"ok": True, "target_version": ver}


def _public_runner_row(r: dict) -> dict:
    """Strip internal agent URLs for unauthenticated /api/runners consumers."""
    out = dict(r)
    out["address"] = None
    out["port"] = None
    return out


@app.get("/api/runners")
async def list_runners(request: Request):
    """Return all recent runners (including disabled) for UI display.

    Enriches each row with live scheduler state (current_model,
    in_flight_job_id) when this pod is the scheduler-holding pod. The
    non-scheduler replica returns those fields as None — the UI polls
    the scheduler pod often enough that it picks up the right state.

    Unauthenticated callers receive the same fields except ``address`` and
    ``port`` are nulled so internal agent endpoints are not exposed publicly."""
    rows = await db.get_all_runners(app.state.db)
    scheduler = getattr(app.state, "scheduler", None)
    runner_state = getattr(scheduler, "_runners", {}) if scheduler else {}
    for r in rows:
        rs = runner_state.get(r["id"])
        r["current_model"] = rs.current_model if rs else None
        r["in_flight_job_id"] = rs.in_flight_job_id if rs else None
    if _session_admin_user(request):
        return rows
    return [_public_runner_row(r) for r in rows]


class RunnerUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    auto_update: Optional[bool] = None
    # pinned_model: dedicate a runner to a model — the scheduler (post-Phase 1)
    # will never swap its loaded model. Pass "" or null to unpin.
    # Sentinel "__unset__" means "leave unchanged."
    pinned_model: Optional[str] = "__unset__"
    # draining: graceful takedown flag. True = scheduler stops assigning new
    # jobs to this runner; in-flight job runs to completion. False = resume
    # normal scheduling. Persists in llm_runners.draining.
    draining: Optional[bool] = None


@app.patch("/api/runners/{runner_id}")
async def update_runner(runner_id: int, req: RunnerUpdateRequest):
    """Update runner settings (enable/disable, auto_update, pinned_model, draining)."""
    if req.enabled is not None:
        found = await db.set_runner_enabled(app.state.db, runner_id, req.enabled)
        if not found:
            raise HTTPException(404, "Runner not found")
    if req.auto_update is not None:
        found = await db.set_runner_auto_update(app.state.db, runner_id, req.auto_update)
        if not found:
            raise HTTPException(404, "Runner not found")
    if req.pinned_model != "__unset__":
        model = req.pinned_model or None  # "" → unpin
        found = await db.set_runner_pinned_model(app.state.db, runner_id, model)
        if not found:
            raise HTTPException(404, "Runner not found")
    if req.draining is not None:
        found = await db.set_runner_draining(app.state.db, runner_id, req.draining)
        if not found:
            raise HTTPException(404, "Runner not found")
    return {"ok": True}


@app.delete("/api/runners/{runner_id}")
async def delete_runner_endpoint(runner_id: int):
    """Remove a runner from the fleet (DB row + app allowlist scrub)."""
    ok = await db.delete_runner(app.state.db, runner_id)
    if not ok:
        raise HTTPException(404, "Runner not found")
    return {"ok": True, "deleted_id": runner_id}


@app.post("/api/runners/delete-stale")
async def delete_stale_runners_endpoint():
    """Delete all runners in the admin \"stale\" bucket (disabled or no ~90s heartbeat)."""
    ids = await db.delete_stale_runners(app.state.db)
    return {"ok": True, "deleted": len(ids), "runner_ids": ids}


class TriggerRunnerUpdateRequest(BaseModel):
    target_version: Optional[str] = None


@app.post("/api/runners/{runner_id}/update")
async def trigger_runner_update(
    runner_id: int,
    req: TriggerRunnerUpdateRequest = TriggerRunnerUpdateRequest(),
):
    """Trigger a manual update on a specific runner.
    If target_version is provided, use it; otherwise fall back to global target."""
    target = req.target_version
    if not target:
        target = await db.get_global_setting(app.state.db, "agent_target_version")
    if not target:
        raise HTTPException(400, "No target version specified and no global target set")
    runner = await db.get_runner_by_id(app.state.db, runner_id)
    if not runner:
        raise HTTPException(404, "Runner not found")
    agent_version = runner.get("capabilities", {}).get("agent_version", "")
    if agent_versions_equivalent(agent_version, target):
        return {"ok": True, "message": "Runner already at target version", "version": target}
    client = await _get_runner_client(app.state.db, runner_id)
    try:
        resp = await client.trigger_update(target)
        return {"ok": True, "message": f"Update to {target} triggered", "agent_response": resp}
    except Exception as e:
        raise HTTPException(502, f"Failed to trigger update on runner: {e}")


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

    async def _one_status(r: dict) -> dict:
        try:
            client = _llm_agent_client_for_runner_row(r)
            status = await client.status(timeout=_AGENT_AGG_TIMEOUT)
            status["runner_id"] = r["id"]
            status["runner_hostname"] = r["hostname"]
            return status
        except Exception:
            return {
                "runner_id": r["id"],
                "runner_hostname": r["hostname"],
                "error": "unreachable",
                "gpu_vram_total_gb": 0,
                "gpu_vram_used_gb": 0,
            }

    runner_statuses = await asyncio.gather(*(_one_status(r) for r in runners_list))
    all_loaded_models = []
    total_vram = 0.0
    used_vram = 0.0
    total_cpu = 0.0
    total_mem = 0.0
    used_mem = 0.0

    for i, r in enumerate(runners_list):
        status = runner_statuses[i]
        if "error" not in status:
            rv_total = status.get("gpu_vram_total_gb", 0)
            rv_used = status.get("gpu_vram_used_gb", 0)
            total_vram += rv_total
            used_vram += rv_used
            total_cpu += status.get("cpu_pct", 0)
            # Unified-memory runners: pool is already counted in gpu_vram_*; skip mem_* to avoid double sum.
            if status.get("gpu_vendor") != "unified":
                total_mem += status.get("mem_total_gb", 0)
                used_mem += status.get("mem_used_gb", 0)

            for m in status.get("loaded_ollama_models", []):
                m["runner"] = r["hostname"]
                m["do_not_evict"] = (r.get("pinned_model") == m["name"])
                all_loaded_models.append(m)

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

    async def _one_models(r: dict) -> Optional[dict]:
        try:
            client = _llm_agent_client_for_runner_row(r)
            return await client.models(timeout=_AGENT_AGG_TIMEOUT)
        except Exception:
            return None

    model_results = await asyncio.gather(*(_one_models(r) for r in runners_list))
    for r, result in zip(runners_list, model_results):
        if not result:
            continue
        for m in result.get("data", []):
            mid = m.get("id")
            if mid not in model_map:
                model_map[mid] = {**m, "runners": []}
            model_map[mid]["runners"].append({
                "runner_id": r["id"],
                "hostname": r["hostname"],
            })

    # Enrich with categories from model_settings or library cache
    for mid, m in model_map.items():
        try:
            settings = await queue_db.get_model_settings(pool, mid)
            cats = list(settings.get("categories") or [])
            if not cats:
                base = mid.split(":")[0]
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT categories FROM ollama_library_cache WHERE name = $1", base)
                if row and row["categories"]:
                    c = row["categories"]
                    cats = json.loads(c) if isinstance(c, str) else list(c)
            m["categories"] = cats
        except Exception:
            m["categories"] = []

    return {"data": list(model_map.values())}


class LLMPullRequest(BaseModel):
    model: str


def _exception_detail(exc: BaseException) -> str:
    """Non-empty string for storing on background_ops.error (empty str breaks the UI)."""
    text = str(exc).strip()
    if text:
        return text
    name = exc.__class__.__name__
    mod = getattr(exc.__class__, "__module__", "") or ""
    if mod and mod != "builtins":
        return f"{mod}.{name}"
    return name


def _extract_pull_error(line: str) -> Optional[str]:
    """Parse one NDJSON line from an Ollama /api/pull stream (or agent proxy).

    Pull can fail with HTTP 200 and an `{"error":...}` line; some builds use
    `status: error` without a dedicated error string."""
    if not line or not line.strip():
        return None
    try:
        parsed = json.loads(line)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    if err is not None and str(err).strip():
        return str(err).strip()
    if parsed.get("status") == "error":
        msg = parsed.get("message")
        if msg is not None and str(msg).strip():
            return str(msg).strip()
        try:
            return json.dumps(parsed, ensure_ascii=False)[:800]
        except Exception:
            return "Ollama reported status=error (unparseable JSON)"
    return None


async def _consume_agent_pull_stream(
    pool: asyncpg.Pool,
    op_id: str,
    client: LLMAgentClient,
    model: str,
) -> None:
    """Run pull on agent, stream progress into background_ops, set terminal status."""
    last_status = ""
    stream_err: Optional[str] = None
    last_write = 0.0
    try:
        async for chunk in client.pull_model(model):
            line = chunk.decode(errors="replace").strip()
            if not line:
                continue
            last_status = line
            e = _extract_pull_error(line)
            if e:
                stream_err = e
            now = asyncio.get_event_loop().time()
            if now - last_write >= 1.0:
                await db.update_op(pool, op_id, progress=last_status)
                last_write = now
        err = stream_err or _extract_pull_error(last_status)
        if err:
            detail = err.strip() or "Pull failed (Ollama returned an error with no message)"
            await db.update_op(
                pool, op_id,
                status="failed",
                progress=last_status or None,
                error=detail,
            )
        else:
            await db.update_op(pool, op_id, status="completed", progress=last_status)
    except Exception as e:
        await db.update_op(
            pool, op_id,
            status="failed",
            error=_exception_detail(e),
        )


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
        disk_total_gb = st.get("disk_total_gb", None)
        # Agent returns disk_* == 0 when disk stats fail (mount/path quirks on some hosts).
        # Do not treat that as "full disk" or pulls break (e.g. Mac Mini bind mounts).
        disk_known = (
            disk_free_gb is not None
            and disk_total_gb is not None
            and float(disk_total_gb) > 0
        )
        model_vram = vram_for_model(req.model)
        # Rough heuristic: disk needed ≈ VRAM estimate (quantized weights on disk)
        if disk_known and model_vram > 0 and disk_free_gb < model_vram:
            raise HTTPException(
                507,
                f"Not enough disk space on {st.get('node', 'runner')}: "
                f"{disk_free_gb:.1f} GB free, ~{model_vram:.1f} GB needed for {req.model}"
            )
        elif disk_known and disk_free_gb < 5:
            disk_warning = f"Low disk space: {disk_free_gb:.1f} GB free"
    except HTTPException:
        raise
    except Exception:
        pass  # non-critical, proceed with pull

    op_id = f"pull-{req.model}-{id(req)}"
    pool = app.state.db
    await db.create_op(pool, op_id, "pull", req.model)

    async def _do_pull():
        client = await _get_runner_client(pool, runner_id)
        await _consume_agent_pull_stream(pool, op_id, client, req.model)

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
    pool = app.state.db
    await db.create_op(pool, op_id, "update", model)

    async def _do_update():
        client = await _get_runner_client(pool, runner_id)
        await _consume_agent_pull_stream(pool, op_id, client, model)

    asyncio.create_task(_do_update())
    return {"ok": True, "op_id": op_id, "message": f"Updating {model} in background"}


@app.post("/api/llm/models/sync")
@app.post("/api/llm/models/mirror")  # backwards compat
async def sync_models():
    """Sync models across all runners.

    For each exact model tag (e.g. 'qwen2.5:14b') downloaded on any runner,
    ensure every other runner that can fit that same tag in VRAM has it too.
    If the runner already has the exact tag, skip it.
    """
    _inc_request("/api/llm/models/sync", "POST", 200)
    pool = app.state.db
    runners_list = await db.get_active_runners(pool)
    if len(runners_list) < 2:
        return {"ok": True, "pulls": [], "message": "Need at least 2 runners to sync"}

    # Gather per-runner downloaded models and VRAM capacity.
    runner_models: dict[int, set[str]] = {}
    runner_vram: dict[int, float] = {}
    runner_names: dict[int, str] = {}

    for r in runners_list:
        rid = r["id"]
        hostname = r["hostname"]
        runner_names[rid] = hostname
        runner_models[rid] = set()
        caps = r.get("capabilities", {})
        if isinstance(caps, str):
            try:
                caps = json.loads(caps)
            except Exception:
                caps = {}
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

    # For each exact model tag present on any runner, pull it to every target
    # runner that does not already have it and can fit it.
    all_models = set()
    for models in runner_models.values():
        all_models.update(models)

    pulls = []
    for model in all_models:
        need = vram_for_model(model)
        for target_rid in runner_models:
            # Skip if this runner already has this exact tag.
            if model in runner_models[target_rid]:
                continue

            # Skip if this model does not fit on target runner VRAM.
            target_cap = runner_vram.get(target_rid, 0)
            if need > target_cap:
                continue

            # Find a source runner that has this model tag.
            source_name = "unknown"
            for src_rid, models in runner_models.items():
                if model in models:
                    source_name = runner_names[src_rid]
                    break
            pulls.append({
                "model": model,
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
        await db.create_op(pool, op_id, "sync", p["model"], target=p["target_runner"])

        async def _do_sync_pull(model=p["model"], rid=p["target_runner_id"], oid=op_id):
            client = await _get_runner_client(pool, rid)
            await _consume_agent_pull_stream(pool, oid, client, model)

        asyncio.create_task(_do_sync_pull())

    return {
        "ok": True,
        "pulls": [{"model": p["model"], "target": p["target_runner"]} for p in unique_pulls],
        "message": f"Syncing {len(unique_pulls)} model(s) across runners",
    }


@app.get("/api/ops")
async def list_operations():
    """List all background operations and their status."""
    return await db.get_ops(app.state.db)


@app.get("/api/ops/{op_id:path}")
async def get_operation(op_id: str):
    """Get status of a background operation. op_id uses :path matcher so
    ids that include slashes (model names like MFDoom/deepseek-r1-...)
    still route correctly once URL-decoded."""
    op = await db.get_op(app.state.db, op_id)
    if op is None:
        raise HTTPException(404, "Operation not found")
    return op


@app.delete("/api/ops/{op_id:path}")
async def dismiss_operation(op_id: str):
    """Dismiss (delete) a background operation record. Used by the UI to
    clear completed/failed ops after the user has acknowledged them.
    Refuses to delete ops still in 'running' status — those must finish
    first to avoid orphaning a live pull."""
    op = await db.get_op(app.state.db, op_id)
    if op is None:
        raise HTTPException(404, "Operation not found")
    if op.get("status") == "running":
        raise HTTPException(409, "Cannot dismiss a running operation")
    await db.delete_op(app.state.db, op_id)
    return {"ok": True, "op_id": op_id}


@app.post("/api/llm/models/delete")
async def llm_delete_model(model: str, runner_id: Optional[int] = None):
    """Delete a model from disk on a runner."""
    _inc_request("/api/llm/models/delete", "POST", 200)
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

    # Resolve model alias → base model, injecting alias params/system prompt
    alias_row = await queue_db.get_model_alias(app.state.db, model)
    if alias_row:
        body = dict(body)
        alias_params = alias_row.get("parameters") or {}
        if isinstance(alias_params, str):
            alias_params = json.loads(alias_params)
        for k, v in alias_params.items():
            if k not in body:
                body[k] = v
        alias_system = alias_row.get("system_prompt")
        if alias_system:
            msgs = list(body.get("messages", []))
            if not any(m.get("role") == "system" for m in msgs):
                msgs.insert(0, {"role": "system", "content": alias_system})
                body["messages"] = msgs
        model = alias_row["base_model"]
        body["model"] = model

    # Enforce per-app model + runner restrictions
    pool = app.state.db
    app_id: Optional[int] = None
    app_allowed_runners: list[int] = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.removeprefix("Bearer ").strip()
        allowed = await db.check_model_allowed(pool, api_key, model)
        if not allowed:
            raise HTTPException(403, f"Model '{model}' is not allowed for this application")
        app_allowed_runners = await db.get_app_allowed_runners(pool, api_key)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE registered_apps SET last_seen = NOW() WHERE api_key = $1", api_key)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM registered_apps WHERE api_key = $1 AND status = 'active'",
                api_key,
            )
        if row:
            app_id = row["id"]

    if app_id is not None:
        await queue_policy.ensure_category_access(pool, app_id, model)

    provider = detect_provider(model)
    _inc_request("/v1/chat/completions", "POST", 200)

    # ── Cloud model routing ──────────────────────────────────────────────
    if provider == ModelProvider.ANTHROPIC:
        if app_id is not None:
            await queue_policy.check_queue_rate_limit(pool, app_id)
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

    # ── Local model routing (Ollama) ──────────────────────────────────────
    import uuid as _uuid

    scheduler: Scheduler = app.state.scheduler

    # Pre-check VRAM (respect app runner affinity like the queue path)
    check = await scheduler.check_submission(
        model,
        allowed_runner_ids=app_allowed_runners if app_allowed_runners else None,
    )
    if not check["ok"]:
        raise HTTPException(422, check)

    if app_id is not None:
        await queue_policy.check_queue_rate_limit(pool, app_id)

    queue_request = {
        "messages": body.get("messages", []),
    }
    for key in ("temperature", "max_tokens", "tools", "top_p", "top_k",
                "frequency_penalty", "presence_penalty", "stop"):
        if key in body:
            queue_request[key] = body[key]

    # ── Fast path: direct proxy when model is already loaded + runner idle ─
    # Bypasses the job queue entirely — no DB write, no poll loop, real streaming.
    # Falls through to the queue when no idle runner has the model loaded (swap needed).
    if hasattr(scheduler, "try_claim_for_fast_path"):
        fp_runner = scheduler.try_claim_for_fast_path(
            model, app_allowed_runners if app_allowed_runners else None
        )
        if fp_runner is not None:
            t0_fp = time.time()
            try:
                fp_client = await _get_runner_client(pool, runner_id=fp_runner.runner_id)
                fp_messages, fp_kwargs = await scheduler._apply_runner_params(
                    queue_request, model, fp_runner
                )
                logger.info(
                    "fast-path: %s on %s (stream=%s, msgs=%d)",
                    model, fp_runner.hostname, stream, len(fp_messages),
                )
            except Exception:
                scheduler.release_fast_path_claim(fp_runner)
                logger.exception("fast-path setup failed for %s — falling back to queue", model)
                fp_runner = None  # fall through

            if fp_runner is not None:
                if stream:
                    async def _fast_stream():
                        nonlocal t0_fp
                        try:
                            stream_ctx = await fp_client.chat(
                                messages=fp_messages, model=model, stream=True, **fp_kwargs
                            )
                            async with stream_ctx as resp:
                                async for chunk in resp.aiter_bytes():
                                    yield chunk
                            elapsed = time.time() - t0_fp
                            fastpath_duration_seconds.labels(model=model).observe(elapsed)
                            fastpath_requests_total.labels(model=model, status="completed").inc()
                            logger.info("fast-path done: %s on %s (%.1fs)", model, fp_runner.hostname, elapsed)
                        except Exception:
                            fastpath_requests_total.labels(model=model, status="failed").inc()
                            logger.exception("fast-path stream error: %s on %s", model, fp_runner.hostname)
                        finally:
                            scheduler.release_fast_path_claim(fp_runner)
                    return StreamingResponse(_fast_stream(), media_type="text/event-stream")
                else:
                    try:
                        fp_result = await fp_client.chat(
                            messages=fp_messages, model=model, stream=False, **fp_kwargs
                        )
                        elapsed = time.time() - t0_fp
                        fastpath_duration_seconds.labels(model=model).observe(elapsed)
                        fastpath_requests_total.labels(model=model, status="completed").inc()
                        logger.info("fast-path done: %s on %s (%.1fs)", model, fp_runner.hostname, elapsed)
                        return fp_result
                    except Exception:
                        fastpath_requests_total.labels(model=model, status="failed").inc()
                        logger.exception("fast-path failed: %s on %s", model, fp_runner.hostname)
                        raise
                    finally:
                        scheduler.release_fast_path_claim(fp_runner)

    # ── Queue path: scheduler handles model loading and dispatch ───────────
    job_id = str(_uuid.uuid4())[:12]
    job_metadata = {"allowed_runner_ids": app_allowed_runners} if app_allowed_runners else None
    priority = await queue_policy.priority_for_app(pool, app_id)
    logger.info("queue: submitting job %s (model=%s, app_id=%s, runners=%s)",
                job_id, model, app_id, app_allowed_runners or "any")
    await queue_db.insert_job(
        pool, job_id, None, app_id, model, queue_request, job_metadata, priority=priority
    )

    # Poll until the scheduler completes the job (timeout after 10 min)
    deadline = time.time() + 600
    result = None
    while time.time() < deadline:
        job = await queue_db.get_job(pool, job_id)
        if not job:
            raise HTTPException(500, "Queue job disappeared")
        status = job["status"]
        if status == "completed":
            result = job.get("result")
            if isinstance(result, str):
                result = json.loads(result)
            break
        if status in ("failed", "cancelled"):
            error = job.get("error", "Job failed")
            raise HTTPException(502, f"Inference failed: {error}")
        await asyncio.sleep(0.5)

    if result is None:
        await queue_db.update_job_status(pool, job_id, "cancelled", error="Proxy timeout")
        raise HTTPException(504, "Inference timed out waiting for queue")

    if not stream:
        return result

    # Convert completed result into an OpenAI streaming response so
    # streaming clients (Forge, etc.) can consume it unchanged.
    async def _queue_to_stream():
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message", {})
        chunk = {
            "id": result.get("id", f"chatcmpl-{job_id}"),
            "object": "chat.completion.chunk",
            "created": result.get("created", int(time.time())),
            "model": result.get("model", model),
            "choices": [{
                "index": 0,
                "delta": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }],
        }
        if message.get("tool_calls"):
            chunk["choices"][0]["delta"]["tool_calls"] = message["tool_calls"]
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_queue_to_stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def proxy_list_models(request: Request):
    """OpenAI-compatible model list. Enumerates local models across active
    runners plus enabled cloud models. Filtered by the calling app's
    allowed-model patterns when an API key is provided."""
    _inc_request("/v1/models", "GET", 200)
    pool = app.state.db
    model_ids: set[str] = set()

    # Local models from active runners
    try:
        runners_list = await db.get_active_runners(pool)

        async def _ids_for_runner(runner: dict) -> set[str]:
            out: set[str] = set()
            try:
                client = _llm_agent_client_for_runner_row(runner)
                result = await client.models(timeout=_AGENT_AGG_TIMEOUT)
                for m in result.get("data", []):
                    name = m.get("id", "")
                    if name:
                        out.add(name)
            except Exception:
                pass
            return out

        id_sets = await asyncio.gather(*(_ids_for_runner(r) for r in runners_list))
        for s in id_sets:
            model_ids |= s
    except Exception:
        pass

    # Enabled cloud models
    try:
        cloud_configs = await db.get_cloud_model_configs(pool)
        for c in cloud_configs:
            if c.get("enabled", True):
                model_ids.add(c["model_id"])
    except Exception:
        pass

    # Model aliases appear as selectable models
    try:
        aliases = await queue_db.get_all_model_aliases(pool)
        for a in aliases:
            model_ids.add(a["alias_name"])
    except Exception:
        pass

    # Per-app filtering: if Bearer token present, hide models the app can't use
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.removeprefix("Bearer ").strip()
        allowed: list[str] = []
        for name in sorted(model_ids):
            if await db.check_model_allowed(pool, api_key, name):
                allowed.append(name)
        model_ids = set(allowed)

    data = [
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "llm-manager",
        }
        for name in sorted(model_ids)
    ]
    return {"object": "list", "data": data}


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
        a_copy.setdefault("allowed_categories", [])
        a_copy["allowed_categories"] = list(a_copy["allowed_categories"] or [])
        a_copy.setdefault("excluded_categories", [])
        a_copy["excluded_categories"] = list(a_copy["excluded_categories"] or [])
        a_copy["allowed_models"] = await db.get_app_allowed_models(app.state.db, a_copy["id"])
        a_copy["excluded_models"] = await db.get_app_excluded_models(app.state.db, a_copy["id"])
        limits = await queue_db.get_rate_limit(app.state.db, a_copy["id"])
        a_copy["max_queue_depth"] = int(limits.get("max_queue_depth", 50))
        a_copy["max_jobs_per_minute"] = int(limits.get("max_jobs_per_minute", 10))
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


@app.delete("/api/apps/by-id/{app_id}")
async def remove_app_by_id(app_id: int):
    """Id-keyed delete for the admin UI (which only has api_key_preview,
    not the full key). No route collision with /api/apps/{api_key}:
    api_keys are single path segments so `/apps/by-id/3` — two segments —
    won't match the api_key pattern."""
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    removed = await deregister_app_by_id(app.state.db, app_id)
    if not removed:
        raise HTTPException(status_code=404, detail="App not found")
    _inc_request("/api/apps/delete", "DELETE", 200)
    return {"ok": True, "app_id": app_id}


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
    allowed_categories: Optional[list[str]] = None
    excluded_categories: Optional[list[str]] = None


class AppRateLimitsRequest(BaseModel):
    max_queue_depth: int
    max_jobs_per_minute: int


@app.patch("/api/apps/{app_id}/permissions")
async def update_app_permissions_endpoint(app_id: int, req: AppPermissionsRequest):
    """Update permissions for an app."""
    _inc_request("/api/apps/permissions", "PATCH", 200)
    found = await db.update_app_permissions(
        app.state.db, app_id, req.allow_profile_switch,
        req.allowed_categories, req.excluded_categories,
    )
    if not found:
        raise HTTPException(404, "App not found")
    return {"ok": True}


@app.put("/api/apps/{app_id}/rate-limits")
async def update_app_rate_limits_endpoint(app_id: int, req: AppRateLimitsRequest):
    if req.max_queue_depth < 1:
        raise HTTPException(422, "max_queue_depth must be >= 1")
    if req.max_jobs_per_minute < 1:
        raise HTTPException(422, "max_jobs_per_minute must be >= 1")
    app_row = await db.get_app_by_id(app.state.db, app_id)
    if not app_row:
        raise HTTPException(404, "App not found")
    await queue_db.set_rate_limit(
        app.state.db,
        app_id,
        req.max_queue_depth,
        req.max_jobs_per_minute,
    )
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


@app.get("/api/apps/{app_id}/excluded-models")
async def get_app_excluded_models_endpoint(app_id: int):
    models = await db.get_app_excluded_models(app.state.db, app_id)
    return {"app_id": app_id, "excluded_models": models}


class AppExcludedModelsRequest(BaseModel):
    excluded_models: list[str]


@app.put("/api/apps/{app_id}/excluded-models")
async def set_app_excluded_models_endpoint(app_id: int, req: AppExcludedModelsRequest):
    await db.set_app_excluded_models(app.state.db, app_id, req.excluded_models)
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
    pool = app.state.db
    alias_row = await queue_db.get_model_alias(pool, req.model)
    model = alias_row["base_model"] if alias_row else req.model
    op_id = f"load-{model}-{id(req)}"
    await db.create_op(pool, op_id, "load", model)

    async def _do_load():
        try:
            client = await _get_runner_client(pool, runner_id)
            await client.load_model(model, req.keep_alive)
            await db.update_op(pool, op_id, status="completed")
        except Exception as e:
            await db.update_op(pool, op_id, status="failed", error=_exception_detail(e))

    asyncio.create_task(_do_load())
    return {"ok": True, "op_id": op_id, "message": f"Loading {req.model} in background"}


class ModelUnloadRequest(BaseModel):
    model: str


@app.post("/api/llm/models/unload")
async def llm_unload_model(req: ModelUnloadRequest, runner_id: Optional[int] = None):
    """Unload a model from VRAM. Runs in background."""
    _inc_request("/api/llm/models/unload", "POST", 200)
    op_id = f"unload-{req.model}-{id(req)}"
    pool = app.state.db
    await db.create_op(pool, op_id, "unload", req.model)

    async def _do_unload():
        try:
            client = await _get_runner_client(pool, runner_id)
            await client.unload_model_from_vram(req.model)
            await db.update_op(pool, op_id, status="completed")
        except Exception as e:
            await db.update_op(pool, op_id, status="failed", error=_exception_detail(e))

    asyncio.create_task(_do_unload())
    return {"ok": True, "op_id": op_id, "message": f"Unloading {req.model} in background"}


@app.post("/api/llm/runners/{runner_id}/flush")
async def flush_runner_vram(runner_id: int):
    """Unload all models from VRAM on a specific runner. Synchronous — waits for completion."""
    _inc_request(f"/api/llm/runners/{runner_id}/flush", "POST", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    status = await client.status()
    loaded = status.get("loaded_ollama_models", [])

    if not loaded:
        return {
            "ok": True,
            "unloaded": [],
            "message": "No models reported as loaded by Ollama. If VRAM is still full, try restarting Ollama.",
        }

    unloaded = []
    errors = []
    for m in loaded:
        try:
            await client.unload_model_from_vram(m["name"])
            unloaded.append(m["name"])
        except Exception as e:
            logger.warning("Failed to unload %s during flush: %s", m["name"], e)
            errors.append({"model": m["name"], "error": str(e)})

    return {
        "ok": True,
        "unloaded": unloaded,
        "errors": errors,
        "message": f"Unloaded {len(unloaded)} model(s)" + (f", {len(errors)} error(s)" if errors else ""),
    }


@app.post("/api/llm/runners/{runner_id}/restart-ollama")
async def restart_runner_ollama(runner_id: int):
    """Restart Ollama on the runner (requires OLLAMA_CONTAINER configured on agent)."""
    _inc_request(f"/api/llm/runners/{runner_id}/restart-ollama", "POST", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    result = await client.restart_ollama()
    return result


@app.get("/api/llm/runners/{runner_id}/ollama-settings")
async def get_runner_ollama_settings(runner_id: int):
    """Read a runner's Ollama tunables (contents of ollama.env)."""
    _inc_request(f"/api/llm/runners/{runner_id}/ollama-settings", "GET", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    return await client.get_ollama_settings()


class OllamaSettingsUpdateRequest(BaseModel):
    settings: dict


@app.put("/api/llm/runners/{runner_id}/ollama-settings")
async def put_runner_ollama_settings(runner_id: int, req: OllamaSettingsUpdateRequest):
    """Apply new Ollama tunables and recreate the container. Strongly suggest
    draining the runner first — the recreate interrupts any in-flight job."""
    _inc_request(f"/api/llm/runners/{runner_id}/ollama-settings", "PUT", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    try:
        return await client.put_ollama_settings(req.settings or {})
    except httpx.HTTPStatusError as e:
        # Propagate the agent's error message — usually a validation error
        # from its allowlist (e.g. "OLLAMA_KV_CACHE_TYPE must be one of ...")
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        raise HTTPException(e.response.status_code, detail)


@app.get("/api/llm/runners/{runner_id}/logs")
async def get_runner_logs(runner_id: int, tail: int = 200, service: str = "all"):
    """Return recent log lines from the agent and/or Ollama container on this runner."""
    _inc_request(f"/api/llm/runners/{runner_id}/logs", "GET", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    try:
        return await client.logs(tail=tail, service=service)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Agent is an older version that doesn't support /v1/logs yet
            return {"agent_logs": [], "ollama_logs": [], "ollama_available": False,
                    "_note": "Agent does not support log streaming — update the agent to enable this feature."}
        raise


@app.get("/api/llm/runners/{runner_id}/ollama-version")
async def get_runner_ollama_version(runner_id: int):
    """Return the running Ollama version, configured image tag, and git commit hash."""
    _inc_request(f"/api/llm/runners/{runner_id}/ollama-version", "GET", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    return await client.get_ollama_version()


class OllamaUpgradeRequest(BaseModel):
    tag: str


@app.post("/api/llm/runners/{runner_id}/ollama-upgrade")
async def upgrade_runner_ollama(runner_id: int, req: OllamaUpgradeRequest):
    """Pull a new Ollama image version and recreate the container on this runner.
    Updates .env so the tag persists across compose restarts.
    Strongly recommend draining the runner first."""
    _inc_request(f"/api/llm/runners/{runner_id}/ollama-upgrade", "POST", 200)
    client = await _get_runner_client(app.state.db, runner_id)
    try:
        return await client.upgrade_ollama(req.tag)
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        raise HTTPException(e.response.status_code, detail)


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
        try:
            queue_depth_gauge.set(await queue_db.count_pending_jobs(app.state.db))
            running = await queue_db.get_running_jobs(app.state.db)
            queue_active_jobs_gauge.set(len(running))
        except Exception:
            pass
        try:
            all_jobs = await queue_db.get_all_active_jobs(app.state.db)
            loading = sum(1 for j in all_jobs if j.get("status") == "loading_model")
            queue_loading_jobs_gauge.set(loading)
        except Exception:
            pass
        try:
            runners = await db.get_active_runners(app.state.db)
            active_runners_gauge.set(len(runners))
            now = time.time()
            for r in runners:
                ls = r.get("last_seen")
                if ls:
                    import datetime
                    if isinstance(ls, datetime.datetime):
                        age = now - ls.timestamp()
                    else:
                        age = 0
                    runner_last_seen_seconds.labels(runner=r["hostname"]).set(round(age, 1))
        except Exception:
            pass
        try:
            age_hours = await db.get_library_cache_age_hours(app.state.db)
            # None means no refresh has ever been recorded — treat as very stale
            # so an alert fires. Use a year as the sentinel.
            age_seconds = age_hours * 3600 if age_hours is not None else 365 * 24 * 3600
            library_cache_age_seconds.set(age_seconds)
        except Exception:
            pass

    backend_metrics = generate_latest().decode()
    return StreamingResponse(
        iter([backend_metrics]),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
