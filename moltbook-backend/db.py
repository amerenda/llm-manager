"""
Database layer for moltbook-backend.
Manages only moltbook tables: moltbook_configs, moltbook_state,
moltbook_activity, moltbook_peer_posts, moltbook_peer_interactions.
Read-only access to llm_runners for runner lookups.
"""
import json
import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS moltbook_configs (
    slot INT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Agent',
    description TEXT NOT NULL DEFAULT 'A curious AI agent on Moltbook.',
    tone TEXT NOT NULL DEFAULT 'friendly, conversational',
    topics JSONB NOT NULL DEFAULT '["technology","daily life"]'::jsonb,
    model TEXT NOT NULL DEFAULT 'qwen2.5:7b',
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    api_key TEXT NOT NULL DEFAULT '',
    registered BOOLEAN NOT NULL DEFAULT FALSE,
    claimed BOOLEAN NOT NULL DEFAULT FALSE,
    llm_runner_id INT,
    post_interval_minutes INT NOT NULL DEFAULT 120,
    active_hours_start INT NOT NULL DEFAULT 8,
    active_hours_end INT NOT NULL DEFAULT 22,
    max_post_length INT NOT NULL DEFAULT 280,
    auto_reply BOOLEAN NOT NULL DEFAULT TRUE,
    auto_like BOOLEAN NOT NULL DEFAULT FALSE,
    reply_to_own_threads BOOLEAN NOT NULL DEFAULT FALSE,
    post_jitter_pct INT NOT NULL DEFAULT 20,
    karma_throttle BOOLEAN NOT NULL DEFAULT FALSE,
    karma_throttle_threshold INT NOT NULL DEFAULT 10,
    karma_throttle_multiplier FLOAT NOT NULL DEFAULT 2.0,
    target_submolts JSONB NOT NULL DEFAULT '[]'::jsonb,
    auto_dm_approve BOOLEAN NOT NULL DEFAULT FALSE,
    receive_peer_likes BOOLEAN NOT NULL DEFAULT FALSE,
    receive_peer_comments BOOLEAN NOT NULL DEFAULT FALSE,
    send_peer_likes BOOLEAN NOT NULL DEFAULT TRUE,
    send_peer_comments BOOLEAN NOT NULL DEFAULT TRUE,
    heartbeat_md TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS moltbook_state (
    slot INT PRIMARY KEY REFERENCES moltbook_configs(slot) ON DELETE CASCADE,
    karma INT NOT NULL DEFAULT 0,
    last_heartbeat TIMESTAMPTZ,
    last_post_time FLOAT NOT NULL DEFAULT 0,
    next_post_time FLOAT NOT NULL DEFAULT 0,
    pending_dm_requests JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS moltbook_activity (
    id BIGSERIAL PRIMARY KEY,
    slot INT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS moltbook_peer_posts (
    id BIGSERIAL PRIMARY KEY,
    slot INT NOT NULL,
    peer_name TEXT NOT NULL,
    post_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content_preview TEXT NOT NULL DEFAULT '',
    seen_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(slot, post_id)
);

CREATE TABLE IF NOT EXISTS moltbook_peer_interactions (
    slot INT NOT NULL,
    post_id TEXT NOT NULL,
    action TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (slot, post_id, action)
);
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create moltbook tables if they don't exist, and run migrations."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)

        # Migration: add heartbeat_md to moltbook_configs
        try:
            await conn.execute(
                "ALTER TABLE moltbook_configs ADD COLUMN heartbeat_md TEXT NOT NULL DEFAULT ''"
            )
            logger.info("Added column moltbook_configs.heartbeat_md")
        except asyncpg.DuplicateColumnError:
            pass


# ── Read-only access to llm_runners ──────────────────────────────────────────


async def get_active_runners(pool: asyncpg.Pool) -> list[dict]:
    """Return runners seen in the last 90 seconds, ordered by hostname."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, hostname, address, port, capabilities, last_seen, created_at
            FROM llm_runners
            WHERE last_seen > NOW() - INTERVAL '90 seconds'
            ORDER BY hostname
            """
        )
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("capabilities"), str):
            d["capabilities"] = json.loads(d["capabilities"])
        result.append(d)
    return result


async def get_runner_by_id(pool: asyncpg.Pool, runner_id: int) -> Optional[dict]:
    """Return a runner by id, or None if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, hostname, address, port, capabilities, last_seen, created_at
            FROM llm_runners
            WHERE id = $1
            """,
            runner_id,
        )
    if row is None:
        return None
    d = dict(row)
    if isinstance(d.get("capabilities"), str):
        d["capabilities"] = json.loads(d["capabilities"])
    return d


# ── moltbook_configs ─────────────────────────────────────────────────────────


def _default_config_dict(slot: int) -> dict:
    return {
        "slot": slot,
        "name": "Agent",
        "description": "A curious AI agent on Moltbook.",
        "tone": "friendly, conversational",
        "topics": ["technology", "daily life"],
        "model": "qwen2.5:7b",
        "enabled": False,
        "api_key": "",
        "registered": False,
        "claimed": False,
        "llm_runner_id": None,
        "post_interval_minutes": 120,
        "active_hours_start": 8,
        "active_hours_end": 22,
        "max_post_length": 280,
        "auto_reply": True,
        "auto_like": False,
        "reply_to_own_threads": False,
        "post_jitter_pct": 20,
        "karma_throttle": False,
        "karma_throttle_threshold": 10,
        "karma_throttle_multiplier": 2.0,
        "target_submolts": [],
        "auto_dm_approve": False,
        "receive_peer_likes": False,
        "receive_peer_comments": False,
        "send_peer_likes": True,
        "send_peer_comments": True,
        "heartbeat_md": "",
    }


def _row_to_config_dict(row) -> dict:
    d = dict(row)
    for field in ("topics", "target_submolts"):
        if isinstance(d.get(field), str):
            d[field] = json.loads(d[field])
    return d


async def get_moltbook_config(pool: asyncpg.Pool, slot: int) -> dict:
    """Return config row for slot, or default dict if not in DB (does NOT insert)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM moltbook_configs WHERE slot = $1",
            slot,
        )
    if row is None:
        return _default_config_dict(slot)
    return _row_to_config_dict(row)


async def get_all_moltbook_configs(pool: asyncpg.Pool) -> list[dict]:
    """Return configs for slots 1-6. Missing slots get default dicts. Sorted by slot."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM moltbook_configs WHERE slot BETWEEN 1 AND 6 ORDER BY slot"
        )
    db_map = {row["slot"]: _row_to_config_dict(row) for row in rows}
    result = []
    for slot in range(1, 7):
        if slot in db_map:
            result.append(db_map[slot])
        else:
            result.append(_default_config_dict(slot))
    return result


async def upsert_moltbook_config(pool: asyncpg.Pool, slot: int, **kwargs) -> None:
    """Insert or update a moltbook agent config. Always sets updated_at=NOW()."""
    # Serialize JSON fields
    for field in ("topics", "target_submolts"):
        if field in kwargs and not isinstance(kwargs[field], str):
            kwargs[field] = json.dumps(kwargs[field])

    # Build column list from kwargs
    if not kwargs:
        # Ensure row exists with defaults
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO moltbook_configs (slot)
                VALUES ($1)
                ON CONFLICT (slot) DO UPDATE SET updated_at = NOW()
                """,
                slot,
            )
        return

    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    # Param placeholders: $2, $3, ...
    set_clauses = ", ".join(
        f"{col} = ${i + 2}" for i, col in enumerate(cols)
    )
    insert_cols = "slot, " + ", ".join(cols)
    insert_placeholders = "$1, " + ", ".join(f"${i + 2}" for i in range(len(cols)))

    sql = f"""
        INSERT INTO moltbook_configs ({insert_cols}, updated_at)
        VALUES ({insert_placeholders}, NOW())
        ON CONFLICT (slot) DO UPDATE SET
            {set_clauses},
            updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, slot, *vals)


async def delete_moltbook_config(pool: asyncpg.Pool, slot: int) -> None:
    """Delete a moltbook agent config (cascades to state, activity, peers via FK)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM moltbook_configs WHERE slot = $1",
            slot,
        )
    # Also delete activity and peer data (not FK-cascaded since they don't reference moltbook_configs)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM moltbook_activity WHERE slot = $1", slot)
        await conn.execute("DELETE FROM moltbook_peer_posts WHERE slot = $1", slot)
        await conn.execute("DELETE FROM moltbook_peer_interactions WHERE slot = $1", slot)


# ── moltbook_state ───────────────────────────────────────────────────────────


def _default_state_dict(slot: int) -> dict:
    return {
        "slot": slot,
        "karma": 0,
        "last_heartbeat": None,
        "last_post_time": 0.0,
        "next_post_time": 0.0,
        "pending_dm_requests": [],
    }


async def get_moltbook_state(pool: asyncpg.Pool, slot: int) -> dict:
    """Return state row for slot, or default dict if not in DB."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM moltbook_state WHERE slot = $1",
            slot,
        )
    if row is None:
        return _default_state_dict(slot)
    d = dict(row)
    if isinstance(d.get("pending_dm_requests"), str):
        d["pending_dm_requests"] = json.loads(d["pending_dm_requests"])
    return d


async def upsert_moltbook_state(pool: asyncpg.Pool, slot: int, **kwargs) -> None:
    """Insert or update moltbook agent state."""
    if "pending_dm_requests" in kwargs and not isinstance(kwargs["pending_dm_requests"], str):
        kwargs["pending_dm_requests"] = json.dumps(kwargs["pending_dm_requests"])

    if not kwargs:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO moltbook_state (slot)
                VALUES ($1)
                ON CONFLICT (slot) DO UPDATE SET updated_at = NOW()
                """,
                slot,
            )
        return

    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    set_clauses = ", ".join(
        f"{col} = ${i + 2}" for i, col in enumerate(cols)
    )
    insert_cols = "slot, " + ", ".join(cols)
    insert_placeholders = "$1, " + ", ".join(f"${i + 2}" for i in range(len(cols)))

    sql = f"""
        INSERT INTO moltbook_state ({insert_cols}, updated_at)
        VALUES ({insert_placeholders}, NOW())
        ON CONFLICT (slot) DO UPDATE SET
            {set_clauses},
            updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, slot, *vals)


# ── moltbook_activity ────────────────────────────────────────────────────────


async def append_moltbook_activity(
    pool: asyncpg.Pool,
    slot: int,
    action: str,
    detail: str,
) -> None:
    """Append an activity log entry for a slot."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO moltbook_activity (slot, action, detail)
            VALUES ($1, $2, $3)
            """,
            slot,
            action,
            detail,
        )


async def read_moltbook_activity(
    pool: asyncpg.Pool,
    slot: int,
    n: int = 50,
) -> list[dict]:
    """Return the most recent n activity entries for a slot, newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, slot, action, detail, created_at
            FROM moltbook_activity
            WHERE slot = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            slot,
            n,
        )
    return [dict(r) for r in rows]


# ── moltbook_peer_posts ──────────────────────────────────────────────────────


async def get_peer_posts(pool: asyncpg.Pool, slot: int) -> dict[str, list[dict]]:
    """Return all peer posts for a slot, keyed by peer_name."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, slot, peer_name, post_id, title, content_preview, seen_at
            FROM moltbook_peer_posts
            WHERE slot = $1
            ORDER BY peer_name, seen_at ASC
            """,
            slot,
        )
    result: dict[str, list[dict]] = {}
    for row in rows:
        d = dict(row)
        peer = d["peer_name"]
        if peer not in result:
            result[peer] = []
        result[peer].append(d)
    return result


async def upsert_peer_post(
    pool: asyncpg.Pool,
    slot: int,
    peer_name: str,
    post_id: str,
    title: str,
    content_preview: str,
) -> None:
    """Insert a peer post if not already tracked (ON CONFLICT DO NOTHING)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO moltbook_peer_posts (slot, peer_name, post_id, title, content_preview)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (slot, post_id) DO NOTHING
            """,
            slot,
            peer_name,
            post_id,
            title,
            content_preview,
        )


async def has_interacted(
    pool: asyncpg.Pool,
    slot: int,
    post_id: str,
    action: str,
) -> bool:
    """Return True if the agent has already taken the given action on this post."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM moltbook_peer_interactions
            WHERE slot = $1 AND post_id = $2 AND action = $3
            """,
            slot,
            post_id,
            action,
        )
    return row is not None


async def record_interaction(
    pool: asyncpg.Pool,
    slot: int,
    post_id: str,
    action: str,
) -> None:
    """Record that the agent took an action on a post (INSERT OR IGNORE)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO moltbook_peer_interactions (slot, post_id, action)
            VALUES ($1, $2, $3)
            ON CONFLICT (slot, post_id, action) DO NOTHING
            """,
            slot,
            post_id,
            action,
        )


async def get_interacted_post_ids(
    pool: asyncpg.Pool,
    slot: int,
    action: str,
) -> list[str]:
    """Return list of post_ids where the given action was taken by this slot."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT post_id FROM moltbook_peer_interactions
            WHERE slot = $1 AND action = $2
            ORDER BY created_at ASC
            """,
            slot,
            action,
        )
    return [r["post_id"] for r in rows]


async def prune_peer_posts(
    pool: asyncpg.Pool,
    slot: int,
    keep_per_peer: int = 20,
) -> None:
    """Delete oldest peer posts beyond keep_per_peer per peer_name for a slot."""
    async with pool.acquire() as conn:
        # For each peer_name, delete rows ranked beyond keep_per_peer (oldest first)
        await conn.execute(
            """
            DELETE FROM moltbook_peer_posts
            WHERE id IN (
                SELECT id FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY slot, peer_name
                               ORDER BY seen_at ASC
                           ) AS rn
                    FROM moltbook_peer_posts
                    WHERE slot = $1
                ) ranked
                WHERE rn > $2
            )
            """,
            slot,
            keep_per_peer,
        )
