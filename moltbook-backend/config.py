"""Agent configuration models (Pydantic only — no file I/O)."""
from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel


class AgentPersona(BaseModel):
    name: str = "Agent"
    description: str = "A curious AI agent on Moltbook."
    tone: str = "friendly, conversational"
    topics: list[str] = ["technology", "daily life"]


class AgentSchedule(BaseModel):
    post_interval_minutes: int = 120
    active_hours_start: int = 8
    active_hours_end: int = 22


class AgentBehavior(BaseModel):
    max_post_length: int = 280
    auto_reply: bool = True
    auto_like: bool = False
    reply_to_own_threads: bool = False
    post_jitter_pct: int = 20
    karma_throttle: bool = False
    karma_throttle_threshold: int = 10
    karma_throttle_multiplier: float = 2.0
    target_submolts: list[str] = []
    auto_dm_approve: bool = False
    receive_peer_likes: bool = False
    receive_peer_comments: bool = False
    send_peer_likes: bool = True
    send_peer_comments: bool = True


class AgentConfig(BaseModel):
    slot: int
    enabled: bool = False
    model: str = "qwen2.5:7b"
    api_key: str = ""
    registered: bool = False
    claimed: bool = False
    heartbeat_md: str = ""
    persona: AgentPersona = AgentPersona()
    schedule: AgentSchedule = AgentSchedule()
    behavior: AgentBehavior = AgentBehavior()


class AgentState(BaseModel):
    slot: int
    karma: int = 0
    last_heartbeat: Optional[Union[datetime, str]] = None
    last_post_time: float = 0
    next_post_time: float = 0
    pending_dm_requests: list[str] = []


class PeerPost(BaseModel):
    post_id: str
    title: str
    content_preview: str  # first 200 chars
    seen_at: str  # ISO timestamp


class PeerDatabase(BaseModel):
    slot: int
    peers: dict[str, list[PeerPost]] = {}  # keyed by peer agent name
    # tracks posts we've already liked/commented on to avoid repeating
    liked_post_ids: list[str] = []
    commented_post_ids: list[str] = []


# ── DB reconstruction helpers ─────────────────────────────────────────────────

def config_from_db(row: dict) -> AgentConfig:
    """Reconstruct AgentConfig from a moltbook_configs DB row dict."""
    persona = AgentPersona(
        name=row.get("name", "Agent"),
        description=row.get("description", "A curious AI agent on Moltbook."),
        tone=row.get("tone", "friendly, conversational"),
        topics=row.get("topics", ["technology", "daily life"]),
    )
    schedule = AgentSchedule(
        post_interval_minutes=row.get("post_interval_minutes", 120),
        active_hours_start=row.get("active_hours_start", 8),
        active_hours_end=row.get("active_hours_end", 22),
    )
    behavior = AgentBehavior(
        max_post_length=row.get("max_post_length", 280),
        auto_reply=row.get("auto_reply", True),
        auto_like=row.get("auto_like", False),
        reply_to_own_threads=row.get("reply_to_own_threads", False),
        post_jitter_pct=row.get("post_jitter_pct", 20),
        karma_throttle=row.get("karma_throttle", False),
        karma_throttle_threshold=row.get("karma_throttle_threshold", 10),
        karma_throttle_multiplier=row.get("karma_throttle_multiplier", 2.0),
        target_submolts=row.get("target_submolts", []),
        auto_dm_approve=row.get("auto_dm_approve", False),
        receive_peer_likes=row.get("receive_peer_likes", False),
        receive_peer_comments=row.get("receive_peer_comments", False),
        send_peer_likes=row.get("send_peer_likes", True),
        send_peer_comments=row.get("send_peer_comments", True),
    )
    return AgentConfig(
        slot=row["slot"],
        enabled=row.get("enabled", False),
        model=row.get("model", "qwen2.5:7b"),
        api_key=row.get("api_key", ""),
        registered=row.get("registered", False),
        claimed=row.get("claimed", False),
        heartbeat_md=row.get("heartbeat_md", ""),
        persona=persona,
        schedule=schedule,
        behavior=behavior,
    )


def state_from_db(row: dict) -> AgentState:
    """Reconstruct AgentState from a moltbook_state DB row dict."""
    last_hb = row.get("last_heartbeat")
    if last_hb is not None and not isinstance(last_hb, str):
        # asyncpg returns datetime objects for TIMESTAMPTZ
        last_hb = last_hb.isoformat()
    return AgentState(
        slot=row["slot"],
        karma=row.get("karma", 0),
        last_heartbeat=last_hb,
        last_post_time=float(row.get("last_post_time", 0)),
        next_post_time=float(row.get("next_post_time", 0)),
        pending_dm_requests=row.get("pending_dm_requests", []),
    )
