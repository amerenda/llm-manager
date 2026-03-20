"""
LLM Manager Backend API.
Runs as a stateless k8s Deployment on port 8081.
Combines Moltbook agent management with LLM proxy + app registry.
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
from fastapi.responses import StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel

import db
from config import (
    AgentConfig, AgentPersona, AgentSchedule, AgentBehavior,
    config_from_db, state_from_db,
)
from db import (
    init_db,
    register_app, heartbeat_app, get_apps, deregister_app,
)
from agent_runner import AgentRunner
from llm_agent import LLMAgentClient

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

AGENT_PSK = os.environ.get("AGENT_PSK", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
NODE = socket.gethostname()
API_BASE = "https://www.moltbook.com/api/v1"

# Global agent runners (slot 1-6)
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
    return LLMAgentClient(host=host, port=port, psk=AGENT_PSK)


async def _get_runner_ollama_base(pool: asyncpg.Pool, runner_id: Optional[int] = None) -> str:
    """Get Ollama URL for a runner. Replaces the runner port with 11434."""
    runners_list = await db.get_active_runners(pool)
    if not runners_list:
        raise HTTPException(503, "No active llm-runners available")
    if runner_id is not None:
        r = next((x for x in runners_list if x["id"] == runner_id), None)
        if not r:
            r = runners_list[0]
    else:
        r = runners_list[0]
    # runner address is like http://murderbot.amer.home:8090
    # ollama is on the same host at port 11434
    addr = r["address"]
    return re.sub(r':\d+$', ':11434', addr)


def _make_runner(config: AgentConfig, pool: asyncpg.Pool, ollama_base: str) -> AgentRunner:
    return AgentRunner(
        config,
        pool=pool,
        ollama_base=ollama_base,
        ollama_model=config.model,
        psk=AGENT_PSK,
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    app.state.db = pool
    await init_db(pool)
    logger.info("Database connected: %s", DATABASE_URL)

    # Auto-start enabled moltbook agents from DB
    for row in await db.get_all_moltbook_configs(pool):
        if row["enabled"] and row["api_key"]:
            config = config_from_db(row)
            try:
                ollama_base = await _get_runner_ollama_base(pool, row.get("llm_runner_id"))
            except HTTPException:
                logger.warning(
                    "No runners available for slot %d, deferring start", row["slot"]
                )
                continue
            r = _make_runner(config, pool, ollama_base)
            runners[config.slot] = r
            r.start()
            logger.info(
                "Auto-started moltbook agent %d (%s)", config.slot, config.persona.name
            )

    yield

    for r in runners.values():
        r.stop()
    await pool.close()


app = FastAPI(title="LLM Manager Backend", version="2.0.0", lifespan=lifespan)
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


# ── GPU info (placeholder — backend is no longer on the GPU node) ─────────────

@app.get("/api/gpu")
async def gpu_info():
    return {"message": "GPU info available via /api/llm/status"}


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


# ── Moltbook agent config ─────────────────────────────────────────────────────

@app.get("/api/agents")
async def get_moltbook_agents():
    pool = app.state.db
    configs = await db.get_all_moltbook_configs(pool)
    result = []
    for row in configs:
        state_row = await db.get_moltbook_state(pool, row["slot"])
        state = state_from_db(state_row)
        result.append({
            "slot": row["slot"],
            "enabled": row["enabled"],
            "model": row["model"],
            "api_key": row["api_key"],
            "registered": row["registered"],
            "claimed": row["claimed"],
            "llm_runner_id": row.get("llm_runner_id"),
            "running": row["slot"] in runners and runners[row["slot"]].running,
            "persona": {
                "name": row["name"],
                "description": row["description"],
                "tone": row["tone"],
                "topics": row["topics"],
            },
            "schedule": {
                "post_interval_minutes": row["post_interval_minutes"],
                "active_hours_start": row["active_hours_start"],
                "active_hours_end": row["active_hours_end"],
            },
            "behavior": {
                "max_post_length": row["max_post_length"],
                "auto_reply": row["auto_reply"],
                "auto_like": row["auto_like"],
                "reply_to_own_threads": row["reply_to_own_threads"],
                "post_jitter_pct": row["post_jitter_pct"],
                "karma_throttle": row["karma_throttle"],
                "karma_throttle_threshold": row["karma_throttle_threshold"],
                "karma_throttle_multiplier": row["karma_throttle_multiplier"],
                "target_submolts": row["target_submolts"],
                "auto_dm_approve": row["auto_dm_approve"],
                "receive_peer_likes": row["receive_peer_likes"],
                "receive_peer_comments": row["receive_peer_comments"],
                "send_peer_likes": row["send_peer_likes"],
                "send_peer_comments": row["send_peer_comments"],
            },
            "state": state.model_dump(),
        })
    moltbook_agents_running_gauge.set(
        sum(1 for r in runners.values() if r.running)
    )
    return result


class AgentUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    llm_runner_id: Optional[int] = None
    persona: Optional[dict] = None
    schedule: Optional[dict] = None
    behavior: Optional[dict] = None


@app.patch("/api/agents/{slot}")
async def update_moltbook_agent(slot: int, req: AgentUpdateRequest):
    if slot not in range(1, 7):
        raise HTTPException(status_code=404, detail="Slot must be 1-6")
    pool = app.state.db

    # Load current row to merge partial updates
    row = await db.get_moltbook_config(pool, slot)

    updates: dict = {}

    if req.enabled is not None:
        updates["enabled"] = req.enabled
    if req.model is not None:
        updates["model"] = req.model
    if req.llm_runner_id is not None:
        updates["llm_runner_id"] = req.llm_runner_id

    if req.persona:
        if "name" in req.persona:
            updates["name"] = req.persona["name"]
        if "description" in req.persona:
            updates["description"] = req.persona["description"]
        if "tone" in req.persona:
            updates["tone"] = req.persona["tone"]
        if "topics" in req.persona:
            updates["topics"] = req.persona["topics"]

    if req.schedule:
        for field in ("post_interval_minutes", "active_hours_start", "active_hours_end"):
            if field in req.schedule:
                updates[field] = req.schedule[field]

    if req.behavior:
        for field in (
            "max_post_length", "auto_reply", "auto_like", "reply_to_own_threads",
            "post_jitter_pct", "karma_throttle", "karma_throttle_threshold",
            "karma_throttle_multiplier", "target_submolts", "auto_dm_approve",
            "receive_peer_likes", "receive_peer_comments", "send_peer_likes",
            "send_peer_comments",
        ):
            if field in req.behavior:
                updates[field] = req.behavior[field]

    if updates:
        await db.upsert_moltbook_config(pool, slot, **updates)

    # Restart runner if it's active
    if slot in runners:
        runners[slot].stop()
        del runners[slot]

    # Re-load updated config and restart if enabled
    updated_row = await db.get_moltbook_config(pool, slot)
    if updated_row["enabled"] and updated_row["api_key"]:
        config = config_from_db(updated_row)
        try:
            ollama_base = await _get_runner_ollama_base(pool, updated_row.get("llm_runner_id"))
            r = _make_runner(config, pool, ollama_base)
            runners[slot] = r
            r.start()
        except HTTPException:
            logger.warning("No runners available for slot %d after update", slot)

    return {"ok": True}


# ── Moltbook agent lifecycle ──────────────────────────────────────────────────

@app.post("/api/agents/{slot}/start")
async def start_moltbook_agent(slot: int):
    pool = app.state.db
    row = await db.get_moltbook_config(pool, slot)
    if not row["api_key"]:
        raise HTTPException(status_code=400, detail="Agent not registered — no API key")
    if slot in runners and runners[slot].running:
        return {"ok": True, "message": "Already running"}
    config = config_from_db(row)
    try:
        ollama_base = await _get_runner_ollama_base(pool, row.get("llm_runner_id"))
    except HTTPException:
        raise HTTPException(503, "No active llm-runners available to start agent")
    r = _make_runner(config, pool, ollama_base)
    runners[slot] = r
    r.start()
    await db.upsert_moltbook_config(pool, slot, enabled=True)
    return {"ok": True, "message": f"Agent {slot} started"}


@app.post("/api/agents/{slot}/stop")
async def stop_moltbook_agent(slot: int):
    pool = app.state.db
    if slot in runners:
        runners[slot].stop()
        del runners[slot]
    await db.upsert_moltbook_config(pool, slot, enabled=False)
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
    return await db.read_moltbook_activity(app.state.db, slot, n)


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
        await r.log("manual_post", f"Posted: '{req.title}'")
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
    pool = app.state.db
    row = await db.get_moltbook_config(pool, slot)
    if row["registered"] and row["api_key"]:
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
        await db.upsert_moltbook_config(
            pool,
            slot,
            api_key=api_key,
            registered=True,
            name=req.name,
            description=req.description,
        )
        return {
            "ok": True,
            "api_key_preview": api_key[:8] + "...",
            "message": "Registered! Check your X (Twitter) DMs for the claim link.",
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.post("/api/agents/{slot}/mark-claimed")
async def mark_claimed(slot: int):
    await db.upsert_moltbook_config(app.state.db, slot, claimed=True)
    return {"ok": True}


@app.post("/api/agents/{slot}/dm/approve/{conv_id}")
async def approve_dm(slot: int, conv_id: str):
    pool = app.state.db
    if slot not in runners:
        raise HTTPException(status_code=400, detail="Agent not running")
    r = runners[slot]
    result = await r.client.dm_approve(conv_id)
    if conv_id in r.state.pending_dm_requests:
        r.state.pending_dm_requests.remove(conv_id)
    await db.upsert_moltbook_state(
        pool,
        slot,
        pending_dm_requests=r.state.pending_dm_requests,
    )
    await r.log("dm_approved", f"Approved DM {conv_id}")
    return result


@app.delete("/api/agents/{slot}")
async def delete_moltbook_agent(slot: int):
    if slot not in range(1, 7):
        raise HTTPException(status_code=404, detail="Slot must be 1-6")
    if slot in runners:
        runners[slot].stop()
        del runners[slot]
    await db.delete_moltbook_config(app.state.db, slot)
    return {"ok": True}


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
    _inc_request("/api/llm/models/pull", "POST", 200)
    try:
        client = await _get_runner_client(app.state.db, runner_id)

        async def _stream():
            async for chunk in client.pull_model(req.model):
                yield chunk

        return StreamingResponse(_stream(), media_type="application/x-ndjson")
    except HTTPException:
        raise
    except Exception as e:
        raise _agent_unavailable(f"Runner error: {e}")


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


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics_endpoint():
    running_count = sum(1 for r in runners.values() if r.running)
    moltbook_agents_running_gauge.set(running_count)

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
