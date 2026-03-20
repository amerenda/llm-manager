"""Agent configuration persistence."""
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

CONFIG_DIR = Path("/data/agents")


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
    persona: AgentPersona = AgentPersona()
    schedule: AgentSchedule = AgentSchedule()
    behavior: AgentBehavior = AgentBehavior()


class AgentState(BaseModel):
    slot: int
    karma: int = 0
    last_heartbeat: Optional[str] = None
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


def _config_path(slot: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"agent_{slot}_config.json"


def _state_path(slot: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"agent_{slot}_state.json"


def _activity_path(slot: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"agent_{slot}_activity.jsonl"


def load_config(slot: int) -> AgentConfig:
    path = _config_path(slot)
    if path.exists():
        return AgentConfig.model_validate_json(path.read_text())
    return AgentConfig(slot=slot)


def save_config(config: AgentConfig) -> None:
    _config_path(config.slot).write_text(config.model_dump_json(indent=2))


def load_state(slot: int) -> AgentState:
    path = _state_path(slot)
    if path.exists():
        return AgentState.model_validate_json(path.read_text())
    return AgentState(slot=slot)


def save_state(state: AgentState) -> None:
    _state_path(state.slot).write_text(state.model_dump_json(indent=2))


def append_activity(slot: int, action: str, detail: str) -> None:
    from datetime import datetime, timezone
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "detail": detail,
    }
    with _activity_path(slot).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_activity(slot: int, n: int = 50) -> list[dict]:
    path = _activity_path(slot)
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]][::-1]


def _peer_db_path(slot: int) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"agent_{slot}_peers.json"


def load_peer_db(slot: int) -> PeerDatabase:
    path = _peer_db_path(slot)
    if path.exists():
        return PeerDatabase.model_validate_json(path.read_text())
    return PeerDatabase(slot=slot)


def save_peer_db(db: PeerDatabase) -> None:
    _peer_db_path(db.slot).write_text(db.model_dump_json(indent=2))


def load_all_configs() -> list[AgentConfig]:
    return [load_config(i) for i in range(1, 7)]
