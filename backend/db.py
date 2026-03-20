"""
Database layer using asyncpg directly.
Manages llm_agents and registered_apps tables.
"""
import secrets
import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS llm_agents (
    id SERIAL PRIMARY KEY,
    node_name TEXT UNIQUE NOT NULL,
    host TEXT NOT NULL,
    port INT NOT NULL DEFAULT 8090,
    last_seen TIMESTAMPTZ,
    capabilities JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS registered_apps (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    base_url TEXT,
    api_key TEXT UNIQUE NOT NULL,
    last_seen TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create tables if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)
    logger.info("Database tables initialized")


async def upsert_agent(
    pool: asyncpg.Pool,
    node_name: str,
    host: str,
    port: int,
    capabilities: dict,
) -> None:
    """Insert or update an agent record, refreshing last_seen."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO llm_agents (node_name, host, port, last_seen, capabilities)
            VALUES ($1, $2, $3, NOW(), $4::jsonb)
            ON CONFLICT (node_name) DO UPDATE SET
                host = EXCLUDED.host,
                port = EXCLUDED.port,
                last_seen = NOW(),
                capabilities = EXCLUDED.capabilities
            """,
            node_name,
            host,
            port,
            __import__("json").dumps(capabilities),
        )


async def get_agents(pool: asyncpg.Pool) -> list[dict]:
    """Return all agents seen in the last 5 minutes."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, node_name, host, port, last_seen, capabilities, created_at
            FROM llm_agents
            WHERE last_seen > NOW() - INTERVAL '5 minutes'
            ORDER BY node_name
            """
        )
    return [dict(r) for r in rows]


async def register_app(
    pool: asyncpg.Pool,
    name: str,
    base_url: Optional[str],
) -> str:
    """Register a new app and return its generated API key."""
    api_key = secrets.token_urlsafe(32)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO registered_apps (name, base_url, api_key)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO UPDATE SET
                base_url = EXCLUDED.base_url,
                api_key = EXCLUDED.api_key
            """,
            name,
            base_url,
            api_key,
        )
    return api_key


async def heartbeat_app(
    pool: asyncpg.Pool,
    api_key: str,
    metadata: dict,
) -> bool:
    """Update last_seen and metadata for an app. Returns False if not found."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE registered_apps
            SET last_seen = NOW(), metadata = $1::jsonb
            WHERE api_key = $2
            """,
            __import__("json").dumps(metadata),
            api_key,
        )
    # result is like "UPDATE 1" or "UPDATE 0"
    return result.endswith("1")


async def get_apps(pool: asyncpg.Pool) -> list[dict]:
    """Return all registered apps."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, base_url, api_key, last_seen, metadata, created_at
            FROM registered_apps
            ORDER BY name
            """
        )
    return [dict(r) for r in rows]


async def deregister_app(pool: asyncpg.Pool, api_key: str) -> bool:
    """Remove a registered app by API key. Returns False if not found."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM registered_apps WHERE api_key = $1",
            api_key,
        )
    return result.endswith("1")
