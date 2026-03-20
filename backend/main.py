"""
LLM Manager Backend API.
Runs as a k8s DaemonSet on GPU-labeled nodes.
Combines Moltbook agent management with LLM proxy + app registry.
Port 8081.
"""
import asyncio
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel

from config import (
    AgentConfig, AgentPersona, AgentSchedule, AgentBehavior,
    load_config, save_config, load_all_configs,
    load_state, read_activity,
)
from db import (
    init_db, upsert_agent, get_agents as db_get_agents,
    register_app, heartbeat_app, get_apps, deregister_app,
)
from gpu import detect_gpu, check_model_fit, vram_for_model
from agent_runner import AgentRunner
from llm_agent import LLMAgentClient

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
API_BASE = "https://www.moltbook.com/api/v1"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
AGENT_URL = os.environ.get("AGENT_URL", "http://localhost:8090")
NODE = socket.gethostname()

# Global agent runners (slot 1-3)
runners: dict[int, AgentRunner] = {}

# ── Prometheus metrics ────────────────────────────────────────────────────────

api_requests_total = Counter(
    "llm_backend_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
registered_apps_gauge = Gauge("llm_backend_registered_apps", "Number of registered apps")
moltbook_agents_running_gauge = Gauge(
    "llm_backend_moltbook_agents_running", "Number of running moltbook agents"
)


def _inc_request(endpoint: str, method: str, status: int):
    api_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()


# ── Agent client helper ───────────────────────────────────────────────────────

def _agent_client() -> LLMAgentClient:
    """Parse AGENT_URL and return an LLMAgentClient."""
    url = AGENT_URL.rstrip("/")
    # strip http(s)://
    if "://" in url:
        url = url.split("://", 1)[1]
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        port = int(port_str)
    else:
        host = url
        port = 8090
    return LLMAgentClient(host=host, port=port)


def _make_runner(config: AgentConfig) -> AgentRunner:
    return AgentRunner(config, ollama_base=OLLAMA_BASE, ollama_model=config.model)


# ── Background task: agent heartbeat ─────────────────────────────────────────

async def _agent_heartbeat_loop(app: FastAPI):
    """Periodically check the local llm-agent and update the DB."""
    while True:
        await asyncio.sleep(60)
        try:
            client = _agent_client()
            if await client.is_reachable():
                status = await client.status()
                capabilities = {
                    "gpu_vram_total_bytes": status.get("gpu_vram_total_bytes", 0),
                    "gpu_vram_used_bytes": status.get("gpu_vram_used_bytes", 0),
                    "comfyui_running": status.get("comfyui_running", False),
                    "loaded_models": [m["name"] for m in status.get("loaded_ollama_models", [])],
                }
                await upsert_agent(
                    app.state.db,
                    node_name=NODE,
                    host="localhost",
                    port=8090,
                    capabilities=capabilities,
                )
                logger.debug("Agent heartbeat OK")
            else:
                logger.warning("Local llm-agent not reachable at %s", AGENT_URL)
        except Exception as e:
            logger.error("Agent heartbeat loop error: %s", e)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect to PostgreSQL
    try:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        app.state.db = pool
        await init_db(pool)
        logger.info("Database connected: %s", DATABASE_URL)
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        app.state.db = None

    # 2. Try to register local agent in DB
    if app.state.db:
        try:
            client = _agent_client()
            if await client.is_reachable():
                status = await client.status()
                capabilities = {
                    "gpu_vram_total_bytes": status.get("gpu_vram_total_bytes", 0),
                    "comfyui_running": status.get("comfyui_running", False),
                }
                await upsert_agent(
                    pool,
                    node_name=NODE,
                    host="localhost",
                    port=8090,
                    capabilities=capabilities,
                )
                logger.info("Registered local llm-agent in DB")
            else:
                logger.warning("Local llm-agent not reachable at startup; will retry in background")
        except Exception as e:
            logger.warning("Could not register agent at startup: %s", e)

    # 3. Start agent heartbeat background task
    heartbeat_task = asyncio.create_task(_agent_heartbeat_loop(app))

    # 4. Auto-start moltbook agents that are enabled and have an API key
    for config in load_all_configs():
        if config.enabled and config.api_key:
            r = _make_runner(config)
            runners[config.slot] = r
            r.start()
            logger.info("Auto-started moltbook agent %d (%s)", config.slot, config.persona.name)

    yield

    # Shutdown
    heartbeat_task.cancel()
    for r in runners.values():
        r.stop()
    if app.state.db:
        await app.state.db.close()


app = FastAPI(title="LLM Manager Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_ok = app.state.db is not None
    return {"ok": True, "service": "llm-manager-backend", "node": NODE, "db": db_ok}


# ── Existing: GPU & Ollama models ─────────────────────────────────────────────

@app.get("/api/gpu")
async def gpu_info():
    return detect_gpu() or {"name": "No NVIDIA GPU detected", "vram_total_gb": 0}


@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            r.raise_for_status()
            models = r.json().get("models", [])
            return [
                {
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                    "vram_estimate_gb": vram_for_model(m["name"]),
                }
                for m in models
            ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


@app.post("/api/vram-check")
async def vram_check(body: dict):
    models = body.get("models", [])
    gpu = detect_gpu()
    return check_model_fit(models, gpu)


# ── Existing: Moltbook Agent Config ───────────────────────────────────────────

@app.get("/api/agents")
async def get_moltbook_agents():
    configs = load_all_configs()
    result = []
    for c in configs:
        state = load_state(c.slot)
        result.append({
            "slot": c.slot,
            "enabled": c.enabled,
            "model": c.model,
            "registered": c.registered,
            "claimed": c.claimed,
            "running": c.slot in runners and runners[c.slot].running,
            "persona": c.persona.model_dump(),
            "schedule": c.schedule.model_dump(),
            "behavior": c.behavior.model_dump(),
            "state": state.model_dump(),
        })
    moltbook_agents_running_gauge.set(
        sum(1 for r in runners.values() if r.running)
    )
    return result


class AgentUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    persona: Optional[dict] = None
    schedule: Optional[dict] = None
    behavior: Optional[dict] = None


@app.patch("/api/agents/{slot}")
async def update_moltbook_agent(slot: int, req: AgentUpdateRequest):
    if slot not in range(1, 7):
        raise HTTPException(status_code=404, detail="Slot must be 1-6")
    config = load_config(slot)

    if req.enabled is not None:
        config.enabled = req.enabled
    if req.model is not None:
        config.model = req.model
    if req.persona:
        config.persona = AgentPersona(**{**config.persona.model_dump(), **req.persona})
    if req.schedule:
        config.schedule = AgentSchedule(**{**config.schedule.model_dump(), **req.schedule})
    if req.behavior:
        config.behavior = AgentBehavior(**{**config.behavior.model_dump(), **req.behavior})

    save_config(config)

    # Restart runner if it's active
    if slot in runners:
        runners[slot].stop()
        del runners[slot]
    if config.enabled and config.api_key:
        r = _make_runner(config)
        runners[slot] = r
        r.start()

    return {"ok": True}


# ── Existing: Moltbook Agent Lifecycle ────────────────────────────────────────

@app.post("/api/agents/{slot}/start")
async def start_moltbook_agent(slot: int):
    config = load_config(slot)
    if not config.api_key:
        raise HTTPException(status_code=400, detail="Agent not registered — no API key")
    if slot in runners and runners[slot].running:
        return {"ok": True, "message": "Already running"}
    r = _make_runner(config)
    runners[slot] = r
    r.start()
    config.enabled = True
    save_config(config)
    return {"ok": True, "message": f"Agent {slot} started"}


@app.post("/api/agents/{slot}/stop")
async def stop_moltbook_agent(slot: int):
    if slot in runners:
        runners[slot].stop()
        del runners[slot]
    config = load_config(slot)
    config.enabled = False
    save_config(config)
    return {"ok": True, "message": f"Agent {slot} stopped"}


@app.post("/api/agents/{slot}/heartbeat")
async def trigger_moltbook_heartbeat(slot: int):
    if slot not in runners:
        raise HTTPException(status_code=400, detail="Agent not running")
    asyncio.create_task(runners[slot].run_heartbeat())
    return {"ok": True}


@app.post("/api/agents/{slot}/interact-with-peers")
async def interact_with_peers(slot: int):
    if slot not in runners:
        raise HTTPException(status_code=400, detail="Agent not running")
    peer_names = [
        runners[s].config.persona.name
        for s in runners
        if s != slot and runners[s].running
    ]
    if not peer_names:
        return {"ok": True, "message": "No other running agents to interact with"}
    asyncio.create_task(runners[slot].interact_with_peers(peer_names))
    return {"ok": True, "message": f"Interacting with posts by: {', '.join(peer_names)}"}


@app.get("/api/agents/{slot}/activity")
async def get_agent_activity(slot: int, n: int = 50):
    return read_activity(slot, n)


class PostRequest(BaseModel):
    submolt: str
    title: str
    content: str


@app.post("/api/agents/{slot}/post")
async def manual_post(slot: int, req: PostRequest):
    if slot not in runners:
        raise HTTPException(status_code=400, detail="Agent not running")
    r = runners[slot]
    try:
        result = await r._post_with_challenge(
            r.client.create_post, req.submolt, req.title, req.content
        )
        r.log("manual_post", f"Posted: '{req.title}'")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RegisterRequest(BaseModel):
    name: str
    description: str


@app.post("/api/agents/{slot}/register")
async def register_moltbook_agent(slot: int, req: RegisterRequest):
    if slot not in range(1, 7):
        raise HTTPException(status_code=404)
    config = load_config(slot)
    if config.registered and config.api_key:
        raise HTTPException(status_code=400, detail="Already registered")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{API_BASE}/agents/register",
                json={"name": req.name, "description": req.description},
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
        api_key = data.get("api_key") or data.get("token") or data.get("key")
        if not api_key:
            raise HTTPException(status_code=502, detail=f"No API key in response: {data}")
        config.api_key = api_key
        config.registered = True
        config.persona.name = req.name
        config.persona.description = req.description
        save_config(config)
        return {
            "ok": True,
            "api_key_preview": api_key[:8] + "...",
            "message": "Registered! Check your X (Twitter) DMs for the claim link.",
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.post("/api/agents/{slot}/mark-claimed")
async def mark_claimed(slot: int):
    config = load_config(slot)
    config.claimed = True
    save_config(config)
    return {"ok": True}


@app.post("/api/agents/{slot}/dm/approve/{conv_id}")
async def approve_dm(slot: int, conv_id: str):
    if slot not in runners:
        raise HTTPException(status_code=400, detail="Agent not running")
    r = runners[slot]
    result = await r.client.dm_approve(conv_id)
    if conv_id in r.state.pending_dm_requests:
        r.state.pending_dm_requests.remove(conv_id)
    from config import save_state
    save_state(r.state)
    r.log("dm_approved", f"Approved DM {conv_id}")
    return result


@app.delete("/api/agents/{slot}")
async def delete_moltbook_agent(slot: int):
    if slot not in range(1, 7):
        raise HTTPException(status_code=404, detail="Slot must be 1-6")
    if slot in runners:
        runners[slot].stop()
        del runners[slot]
    # Reset to defaults
    config = AgentConfig(slot=slot)
    save_config(config)
    # Clear state, activity, and peer database files
    from config import _state_path, _activity_path, _peer_db_path
    for path in (_state_path(slot), _activity_path(slot), _peer_db_path(slot)):
        if path.exists():
            path.unlink()
    return {"ok": True}


# ── New: LLM Management ───────────────────────────────────────────────────────

def _agent_unavailable(detail: str = "Local llm-agent is not reachable") -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


@app.get("/api/llm/status")
async def llm_status():
    _inc_request("/api/llm/status", "GET", 200)
    try:
        client = _agent_client()
        return await client.status()
    except Exception as e:
        _inc_request("/api/llm/status", "GET", 503)
        raise _agent_unavailable(f"Agent error: {e}")


@app.get("/api/llm/models")
async def llm_models():
    _inc_request("/api/llm/models", "GET", 200)
    try:
        client = _agent_client()
        return await client.models()
    except Exception as e:
        _inc_request("/api/llm/models", "GET", 503)
        raise _agent_unavailable(f"Agent error: {e}")


class LLMPullRequest(BaseModel):
    model: str


@app.post("/api/llm/models/pull")
async def llm_pull_model(req: LLMPullRequest):
    _inc_request("/api/llm/models/pull", "POST", 200)
    try:
        client = _agent_client()

        async def _stream():
            async for chunk in client.pull_model(req.model):
                yield chunk

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    except Exception as e:
        raise _agent_unavailable(f"Agent error: {e}")


@app.delete("/api/llm/models/{model:path}")
async def llm_delete_model(model: str):
    _inc_request("/api/llm/models/delete", "DELETE", 200)
    try:
        client = _agent_client()
        return await client.delete_model(model)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise _agent_unavailable(f"Agent error: {e}")


class CheckpointSwitchRequest(BaseModel):
    name: str


@app.post("/api/llm/comfyui/checkpoint")
async def llm_switch_checkpoint(req: CheckpointSwitchRequest):
    _inc_request("/api/llm/comfyui/checkpoint", "POST", 200)
    try:
        client = _agent_client()
        return await client.switch_checkpoint(req.name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise _agent_unavailable(f"Agent error: {e}")


@app.get("/api/llm/checkpoints")
async def llm_checkpoints():
    _inc_request("/api/llm/checkpoints", "GET", 200)
    try:
        client = _agent_client()
        status = await client.status()
        return {"checkpoints": status.get("comfyui_checkpoints", [])}
    except Exception as e:
        raise _agent_unavailable(f"Agent error: {e}")


# ── New: OpenAI-compatible proxy ──────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", False)

    _inc_request("/v1/chat/completions", "POST", 200)

    try:
        client = _agent_client()
        if not await client.is_reachable():
            raise _agent_unavailable()

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
        raise _agent_unavailable(f"Agent error: {e}")


@app.post("/v1/images/generations")
async def proxy_image_generations(request: Request):
    body = await request.json()
    _inc_request("/v1/images/generations", "POST", 200)
    try:
        client = _agent_client()
        return await client.generate_image(
            prompt=body.get("prompt", ""),
            model=body.get("model", "v1-5-pruned-emaonly.safetensors"),
            n=body.get("n", 1),
            size=body.get("size", "512x512"),
        )
    except httpx.HTTPStatusError as e:
        _inc_request("/v1/images/generations", "POST", e.response.status_code)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        _inc_request("/v1/images/generations", "POST", 503)
        raise _agent_unavailable(f"Agent error: {e}")


# ── New: App Registry ─────────────────────────────────────────────────────────

@app.get("/api/apps")
async def list_apps():
    if not app.state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    _inc_request("/api/apps", "GET", 200)
    apps = await get_apps(app.state.db)
    registered_apps_gauge.set(len(apps))
    # Redact full api_key, show only prefix
    result = []
    for a in apps:
        a_copy = dict(a)
        key = a_copy.get("api_key", "")
        a_copy["api_key_preview"] = key[:8] + "..." if len(key) >= 8 else key
        del a_copy["api_key"]
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

    # Extract Bearer token from Authorization header
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


# ── New: Prometheus metrics ───────────────────────────────────────────────────

@app.get("/metrics")
async def metrics_endpoint():
    # Update gauges
    running_count = sum(1 for r in runners.values() if r.running)
    moltbook_agents_running_gauge.set(running_count)

    if app.state.db:
        try:
            apps = await get_apps(app.state.db)
            registered_apps_gauge.set(len(apps))
        except Exception:
            pass

    # Backend prometheus metrics
    backend_metrics = generate_latest().decode()

    # Forward agent metrics if available
    agent_metrics = ""
    try:
        client = _agent_client()
        agent_metrics = await client.metrics_raw()
    except Exception:
        pass

    combined = backend_metrics + "\n" + agent_metrics
    return StreamingResponse(
        iter([combined]),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
