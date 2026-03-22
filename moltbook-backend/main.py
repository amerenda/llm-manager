"""
Moltbook Backend API.
Self-contained FastAPI service for moltbook agent management.
Runs on port 8082.
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
from fastapi import FastAPI, HTTPException
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
from agent_runner import AgentRunner
from moltbook_client import MoltbookClient

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

AGENT_PSK = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
NODE = socket.gethostname()
API_BASE = "https://www.moltbook.com/api/v1"

# Global agent runners (slot 1-6)
runners: dict[int, AgentRunner] = {}

# ── Prometheus metrics ────────────────────────────────────────────────────────

api_requests_total = Counter(
    "moltbook_backend_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)
moltbook_agents_running_gauge = Gauge(
    "moltbook_backend_agents_running", "Number of running moltbook agents"
)


def _inc_request(endpoint: str, method: str, status: int):
    api_requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()


# ── Runner helpers ────────────────────────────────────────────────────────────


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
    await db.init_db(pool)
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


app = FastAPI(title="Moltbook Backend", version="1.0.0", lifespan=lifespan)
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
    return {"ok": True, "service": "moltbook-backend", "node": NODE, "db": db_ok}


# ── Moltbook agent config ────────────────────────────────────────────────────


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
            "heartbeat_md": row.get("heartbeat_md", ""),
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
    api_key: Optional[str] = None
    heartbeat_md: Optional[str] = None
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
    if req.api_key is not None:
        updates["api_key"] = req.api_key
        updates["registered"] = True
    if req.heartbeat_md is not None:
        updates["heartbeat_md"] = req.heartbeat_md

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


# ── Moltbook agent lifecycle ─────────────────────────────────────────────────


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
        # Moltbook nests credentials under "agent"
        agent = data.get("agent", {})
        api_key = (
            agent.get("api_key")
            or data.get("api_key")
            or data.get("token")
            or data.get("key")
        )
        if not api_key:
            raise HTTPException(status_code=502, detail=f"No API key in response: {data}")
        claim_url = agent.get("claim_url", "")
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
            "claim_url": claim_url,
            "message": data.get("message", "Registered!"),
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.post("/api/agents/{slot}/mark-claimed")
async def mark_claimed(slot: int):
    await db.upsert_moltbook_config(app.state.db, slot, claimed=True)
    return {"ok": True}


class SetupEmailRequest(BaseModel):
    email: str


@app.post("/api/agents/{slot}/setup-owner-email")
async def setup_owner_email(slot: int, req: SetupEmailRequest):
    pool = app.state.db
    row = await db.get_moltbook_config(pool, slot)
    if not row["registered"] or not row["api_key"]:
        raise HTTPException(400, "Agent not registered")
    client = MoltbookClient(row["api_key"])
    try:
        result = await client.setup_owner_email(req.email)
        return {"ok": True, "message": "Verification email sent. Check your inbox."}
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


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


# ── Prometheus metrics ────────────────────────────────────────────────────────


@app.get("/metrics")
async def metrics_endpoint():
    running_count = sum(1 for r in runners.values() if r.running)
    moltbook_agents_running_gauge.set(running_count)

    backend_metrics = generate_latest().decode()
    return StreamingResponse(
        iter([backend_metrics]),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
