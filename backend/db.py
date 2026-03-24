"""
Database layer using asyncpg directly.
Manages all tables: llm_agents, registered_apps, llm_runners,
profiles, profile_model_entries, profile_image_entries,
profile_activations, app_allowed_models, ollama_library_cache,
model_safety_tags, library_cache_meta.
"""
from __future__ import annotations

import json
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
    status TEXT NOT NULL DEFAULT 'active',
    allow_profile_switch BOOLEAN NOT NULL DEFAULT FALSE,
    last_seen TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm_runners (
    id SERIAL PRIMARY KEY,
    hostname TEXT UNIQUE NOT NULL,
    address TEXT NOT NULL,
    port INT NOT NULL DEFAULT 8090,
    capabilities JSONB DEFAULT '{}'::jsonb,
    last_seen TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profiles (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    unsafe_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profile_model_entries (
    id SERIAL PRIMARY KEY,
    profile_id INT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    model_safe TEXT NOT NULL,
    model_unsafe TEXT,
    count INT NOT NULL DEFAULT 1,
    label TEXT,
    parameters JSONB DEFAULT '{}'::jsonb,
    sort_order INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profile_image_entries (
    id SERIAL PRIMARY KEY,
    profile_id INT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    checkpoint_safe TEXT NOT NULL,
    checkpoint_unsafe TEXT,
    label TEXT,
    parameters JSONB DEFAULT '{}'::jsonb,
    sort_order INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profile_activations (
    id SERIAL PRIMARY KEY,
    runner_id INT NOT NULL REFERENCES llm_runners(id) ON DELETE CASCADE,
    profile_id INT REFERENCES profiles(id) ON DELETE SET NULL,
    activation_status TEXT NOT NULL DEFAULT 'idle',
    activated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(runner_id)
);

CREATE TABLE IF NOT EXISTS app_allowed_models (
    id SERIAL PRIMARY KEY,
    app_id INT NOT NULL REFERENCES registered_apps(id) ON DELETE CASCADE,
    model_pattern TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(app_id, model_pattern)
);

CREATE TABLE IF NOT EXISTS ollama_library_cache (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    pulls TEXT NOT NULL DEFAULT '',
    tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    parameter_sizes JSONB NOT NULL DEFAULT '[]'::jsonb,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_scraped TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_safety_tags (
    id SERIAL PRIMARY KEY,
    pattern TEXT NOT NULL UNIQUE,
    classification TEXT NOT NULL DEFAULT 'unsafe',
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS library_cache_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    user_id INTEGER,
    encrypted_key TEXT NOT NULL,
    key_preview TEXT DEFAULT '',
    label TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_provider_user
    ON api_keys (provider, COALESCE(user_id, 0));

CREATE TABLE IF NOT EXISTS cloud_model_config (
    model_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'anthropic',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    max_tokens INTEGER DEFAULT 4096,
    temperature REAL,
    display_name TEXT,
    config JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_profile_model_entries_profile
    ON profile_model_entries (profile_id, sort_order);

CREATE INDEX IF NOT EXISTS idx_profile_image_entries_profile
    ON profile_image_entries (profile_id, sort_order);
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create tables and indexes if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)
        await conn.execute(CREATE_INDEXES_SQL)

        # Migrations: add columns to existing tables if missing
        for col, default in [
            ("status", "'active'"),
            ("allow_profile_switch", "FALSE"),
            ("allow_unsafe", "FALSE"),
        ]:
            try:
                await conn.execute(
                    f"ALTER TABLE registered_apps ADD COLUMN {col} "
                    f"{'TEXT' if col == 'status' else 'BOOLEAN'} NOT NULL DEFAULT {default}"
                )
                logger.info("Added column registered_apps.%s", col)
            except asyncpg.DuplicateColumnError:
                pass

        # Ensure the Default profile always exists
        await conn.execute(
            """
            INSERT INTO profiles (name, is_default)
            VALUES ('Default', TRUE)
            ON CONFLICT (name) DO NOTHING
            """
        )

        # Seed default safety tags
        for pattern, reason in [
            ("*uncensored*", "Model trained without safety restrictions"),
            ("dolphin-*", "Dolphin models are uncensored by design"),
            ("wizard-vicuna*", "WizardVicuna uncensored variant"),
            ("*abliterated*", "Model with safety training removed"),
        ]:
            await conn.execute(
                """
                INSERT INTO model_safety_tags (pattern, classification, reason)
                VALUES ($1, 'unsafe', $2)
                ON CONFLICT (pattern) DO NOTHING
                """,
                pattern, reason,
            )
    logger.info("Database tables initialized")


# ── Existing: llm_agents ──────────────────────────────────────────────────────

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
            json.dumps(capabilities),
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


# ── Existing: registered_apps ─────────────────────────────────────────────────

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
            INSERT INTO registered_apps (name, base_url, api_key, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (name) DO UPDATE SET
                base_url = EXCLUDED.base_url,
                api_key = EXCLUDED.api_key,
                status = 'active'
            """,
            name,
            base_url,
            api_key,
        )
    return api_key


async def discover_app(
    pool: asyncpg.Pool,
    name: str,
    base_url: str,
    capabilities: list[str],
) -> dict:
    """Handle app discovery. Returns status and api_key if approved."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, api_key, status, allow_profile_switch FROM registered_apps WHERE name = $1",
            name,
        )
    if row is None:
        # New app — create as pending
        api_key = secrets.token_urlsafe(32)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO registered_apps (name, base_url, api_key, status, metadata)
                VALUES ($1, $2, $3, 'pending', $4::jsonb)
                ON CONFLICT (name) DO UPDATE SET
                    base_url = EXCLUDED.base_url,
                    status = 'pending',
                    metadata = EXCLUDED.metadata
                """,
                name,
                base_url,
                api_key,
                json.dumps({"capabilities": capabilities}),
            )
        return {"status": "pending"}
    # Always update base_url and capabilities on re-discovery
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE registered_apps SET base_url = $2, metadata = $3::jsonb WHERE name = $1",
            name, base_url, json.dumps({"capabilities": capabilities}),
        )
    if row["status"] == "pending":
        return {"status": "pending"}
    # Already approved/active — return the api_key
    return {"status": "approved", "api_key": row["api_key"]}


async def approve_app(pool: asyncpg.Pool, app_id: int) -> Optional[str]:
    """Approve a pending app. Returns the api_key or None if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE registered_apps SET status = 'active'
            WHERE id = $1 RETURNING api_key, base_url
            """,
            app_id,
        )
    if row is None:
        return None
    return row["api_key"]


async def update_app_permissions(
    pool: asyncpg.Pool,
    app_id: int,
    allow_profile_switch: bool,
) -> bool:
    """Update app permissions. Returns False if not found."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE registered_apps SET allow_profile_switch = $1 WHERE id = $2",
            allow_profile_switch,
            app_id,
        )
    return result.endswith("1")


async def get_app_by_api_key(pool: asyncpg.Pool, api_key: str) -> Optional[dict]:
    """Look up an app by its API key."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, base_url, api_key, status, allow_profile_switch,
                   last_seen, metadata, created_at
            FROM registered_apps WHERE api_key = $1
            """,
            api_key,
        )
    return dict(row) if row else None


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
            json.dumps(metadata),
            api_key,
        )
    return result.endswith("1")


async def get_apps(pool: asyncpg.Pool) -> list[dict]:
    """Return all registered apps."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, base_url, api_key, status, allow_profile_switch,
                   last_seen, metadata, created_at
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


# ── New: llm_runners ──────────────────────────────────────────────────────────

async def register_runner(
    pool: asyncpg.Pool,
    hostname: str,
    address: str,
    port: int,
    capabilities: dict,
) -> int:
    """Upsert a runner by hostname. Returns the runner id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO llm_runners (hostname, address, port, capabilities, last_seen)
            VALUES ($1, $2, $3, $4::jsonb, NOW())
            ON CONFLICT (hostname) DO UPDATE SET
                address = EXCLUDED.address,
                port = EXCLUDED.port,
                capabilities = EXCLUDED.capabilities,
                last_seen = NOW()
            RETURNING id
            """,
            hostname,
            address,
            port,
            json.dumps(capabilities),
        )
    return row["id"]


async def heartbeat_runner(
    pool: asyncpg.Pool,
    runner_id: int,
    capabilities: dict,
) -> bool:
    """Update last_seen and capabilities for a runner. Returns False if not found."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE llm_runners
            SET last_seen = NOW(), capabilities = $1::jsonb
            WHERE id = $2
            """,
            json.dumps(capabilities),
            runner_id,
        )
    return result.endswith("1")


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


# ── Profiles ─────────────────────────────────────────────────────────────────

async def create_profile(
    pool: asyncpg.Pool,
    name: str,
    unsafe_enabled: bool = False,
) -> int:
    """Create a new profile. Returns the profile id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO profiles (name, unsafe_enabled)
            VALUES ($1, $2)
            RETURNING id
            """,
            name,
            unsafe_enabled,
        )
    return row["id"]


async def get_profile(pool: asyncpg.Pool, profile_id: int) -> Optional[dict]:
    """Return a profile with its model and image entries."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM profiles WHERE id = $1",
            profile_id,
        )
        if row is None:
            return None
        profile = dict(row)
        model_rows = await conn.fetch(
            """
            SELECT * FROM profile_model_entries
            WHERE profile_id = $1 ORDER BY sort_order, id
            """,
            profile_id,
        )
        image_rows = await conn.fetch(
            """
            SELECT * FROM profile_image_entries
            WHERE profile_id = $1 ORDER BY sort_order, id
            """,
            profile_id,
        )
    profile["model_entries"] = [dict(r) for r in model_rows]
    profile["image_entries"] = [dict(r) for r in image_rows]
    return profile


async def get_all_profiles(pool: asyncpg.Pool) -> list[dict]:
    """Return all profiles with entry counts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT p.*,
                   COALESCE(mc.cnt, 0) AS model_entry_count,
                   COALESCE(ic.cnt, 0) AS image_entry_count
            FROM profiles p
            LEFT JOIN (
                SELECT profile_id, COUNT(*) AS cnt
                FROM profile_model_entries GROUP BY profile_id
            ) mc ON mc.profile_id = p.id
            LEFT JOIN (
                SELECT profile_id, COUNT(*) AS cnt
                FROM profile_image_entries GROUP BY profile_id
            ) ic ON ic.profile_id = p.id
            ORDER BY p.is_default DESC, p.name
            """
        )
    return [dict(r) for r in rows]


async def update_profile(pool: asyncpg.Pool, profile_id: int, **kwargs) -> bool:
    """Update profile fields. Returns False if not found."""
    if not kwargs:
        return True
    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    set_clauses = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(cols))
    sql = f"UPDATE profiles SET {set_clauses}, updated_at = NOW() WHERE id = $1"
    async with pool.acquire() as conn:
        result = await conn.execute(sql, profile_id, *vals)
    return result.endswith("1")


async def delete_profile(pool: asyncpg.Pool, profile_id: int) -> bool:
    """Delete a profile. Prevents deleting the default profile. Returns False if not found."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT is_default FROM profiles WHERE id = $1", profile_id
        )
        if row is None:
            return False
        if row["is_default"]:
            raise ValueError("Cannot delete the default profile")
        result = await conn.execute("DELETE FROM profiles WHERE id = $1", profile_id)
    return result.endswith("1")


# ── Profile model entries ────────────────────────────────────────────────────

async def add_profile_model_entry(
    pool: asyncpg.Pool,
    profile_id: int,
    model_safe: str,
    model_unsafe: Optional[str] = None,
    count: int = 1,
    label: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> int:
    """Add a model entry to a profile. Returns the entry id."""
    async with pool.acquire() as conn:
        # Get next sort_order
        row = await conn.fetchrow(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM profile_model_entries WHERE profile_id = $1",
            profile_id,
        )
        next_order = row["next_order"]
        row = await conn.fetchrow(
            """
            INSERT INTO profile_model_entries (profile_id, model_safe, model_unsafe, count, label, parameters, sort_order)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING id
            """,
            profile_id,
            model_safe,
            model_unsafe,
            count,
            label,
            json.dumps(parameters or {}),
            next_order,
        )
    return row["id"]


async def update_profile_model_entry(pool: asyncpg.Pool, entry_id: int, **kwargs) -> bool:
    """Update a model entry."""
    if not kwargs:
        return True
    if "parameters" in kwargs and not isinstance(kwargs["parameters"], str):
        kwargs["parameters"] = json.dumps(kwargs["parameters"])
    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    set_clauses = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(cols))
    sql = f"UPDATE profile_model_entries SET {set_clauses} WHERE id = $1"
    async with pool.acquire() as conn:
        result = await conn.execute(sql, entry_id, *vals)
    return result.endswith("1")


async def delete_profile_model_entry(pool: asyncpg.Pool, entry_id: int) -> bool:
    """Delete a model entry."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM profile_model_entries WHERE id = $1", entry_id
        )
    return result.endswith("1")


# ── Profile image entries ────────────────────────────────────────────────────

async def add_profile_image_entry(
    pool: asyncpg.Pool,
    profile_id: int,
    checkpoint_safe: str,
    checkpoint_unsafe: Optional[str] = None,
    label: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> int:
    """Add an image entry to a profile. Returns the entry id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM profile_image_entries WHERE profile_id = $1",
            profile_id,
        )
        next_order = row["next_order"]
        row = await conn.fetchrow(
            """
            INSERT INTO profile_image_entries (profile_id, checkpoint_safe, checkpoint_unsafe, label, parameters, sort_order)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            RETURNING id
            """,
            profile_id,
            checkpoint_safe,
            checkpoint_unsafe,
            label,
            json.dumps(parameters or {}),
            next_order,
        )
    return row["id"]


async def update_profile_image_entry(pool: asyncpg.Pool, entry_id: int, **kwargs) -> bool:
    """Update an image entry."""
    if not kwargs:
        return True
    if "parameters" in kwargs and not isinstance(kwargs["parameters"], str):
        kwargs["parameters"] = json.dumps(kwargs["parameters"])
    cols = list(kwargs.keys())
    vals = list(kwargs.values())
    set_clauses = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(cols))
    sql = f"UPDATE profile_image_entries SET {set_clauses} WHERE id = $1"
    async with pool.acquire() as conn:
        result = await conn.execute(sql, entry_id, *vals)
    return result.endswith("1")


async def delete_profile_image_entry(pool: asyncpg.Pool, entry_id: int) -> bool:
    """Delete an image entry."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM profile_image_entries WHERE id = $1", entry_id
        )
    return result.endswith("1")


# ── Profile activations ──────────────────────────────────────────────────────

async def activate_profile(
    pool: asyncpg.Pool,
    runner_id: int,
    profile_id: int,
) -> None:
    """Set or update the active profile for a runner."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO profile_activations (runner_id, profile_id, activation_status, activated_at)
            VALUES ($1, $2, 'activating', NOW())
            ON CONFLICT (runner_id) DO UPDATE SET
                profile_id = EXCLUDED.profile_id,
                activation_status = 'activating',
                activated_at = NOW()
            """,
            runner_id,
            profile_id,
        )


async def update_activation_status(
    pool: asyncpg.Pool,
    runner_id: int,
    status: str,
) -> None:
    """Update activation status for a runner (activating, active, error, idle)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE profile_activations SET activation_status = $1 WHERE runner_id = $2",
            status,
            runner_id,
        )


async def deactivate_profile(pool: asyncpg.Pool, runner_id: int) -> None:
    """Remove the active profile for a runner (return to ad-hoc mode)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM profile_activations WHERE runner_id = $1",
            runner_id,
        )


async def get_active_profile_for_runner(
    pool: asyncpg.Pool,
    runner_id: int,
) -> Optional[dict]:
    """Return the active profile for a runner, or None if ad-hoc."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT pa.runner_id, pa.profile_id, pa.activation_status, pa.activated_at,
                   p.name AS profile_name, p.unsafe_enabled
            FROM profile_activations pa
            JOIN profiles p ON p.id = pa.profile_id
            WHERE pa.runner_id = $1
            """,
            runner_id,
        )
    return dict(row) if row else None


async def get_all_activations(pool: asyncpg.Pool) -> list[dict]:
    """Return all runner-to-profile mappings."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pa.runner_id, pa.profile_id, pa.activation_status, pa.activated_at,
                   p.name AS profile_name
            FROM profile_activations pa
            LEFT JOIN profiles p ON p.id = pa.profile_id
            ORDER BY pa.runner_id
            """
        )
    return [dict(r) for r in rows]


# ── App allowed models ──────────────────────────────────────────────────────

async def get_app_allowed_models(pool: asyncpg.Pool, app_id: int) -> list[str]:
    """Return list of allowed model patterns for an app. Empty = unrestricted."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_pattern FROM app_allowed_models WHERE app_id = $1 ORDER BY model_pattern",
            app_id,
        )
    return [r["model_pattern"] for r in rows]


async def set_app_allowed_models(pool: asyncpg.Pool, app_id: int, patterns: list[str]) -> None:
    """Replace the allowed model list for an app."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM app_allowed_models WHERE app_id = $1", app_id)
        for p in patterns:
            await conn.execute(
                "INSERT INTO app_allowed_models (app_id, model_pattern) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                app_id, p,
            )


async def check_model_allowed(pool: asyncpg.Pool, api_key: str, model: str) -> bool:
    """Check if a model is allowed for the app identified by api_key.

    Local models: no restrictions = unrestricted (allow all safe local models).
    Cloud models: no restrictions = denied. Must have an explicit grant (e.g., 'claude-*').
    """
    import fnmatch
    from cloud_providers import detect_provider, ModelProvider

    async with pool.acquire() as conn:
        app_row = await conn.fetchrow(
            "SELECT id FROM registered_apps WHERE api_key = $1", api_key
        )
        if not app_row:
            return False
        patterns = await conn.fetch(
            "SELECT model_pattern FROM app_allowed_models WHERE app_id = $1", app_row["id"]
        )

    provider = detect_provider(model)
    pattern_list = [r["model_pattern"] for r in patterns]

    if provider == ModelProvider.LOCAL:
        # Local: no patterns = unrestricted (backward compatible)
        if not pattern_list:
            return True
        return any(fnmatch.fnmatch(model, p) for p in pattern_list)

    # Cloud: must have an explicit grant
    if not pattern_list:
        return False
    return any(fnmatch.fnmatch(model, p) for p in pattern_list)


# ── Library cache ────────────────────────────────────────────────────────────

async def upsert_library_model(pool: asyncpg.Pool, name: str, description: str,
                                pulls: str, tags_json: list, parameter_sizes: list,
                                categories: list):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO ollama_library_cache (name, description, pulls, tags_json, parameter_sizes, categories, last_scraped)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, NOW())
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                pulls = EXCLUDED.pulls,
                tags_json = EXCLUDED.tags_json,
                parameter_sizes = EXCLUDED.parameter_sizes,
                categories = EXCLUDED.categories,
                last_scraped = NOW()
        """, name, description, pulls, json.dumps(tags_json),
            json.dumps(parameter_sizes), json.dumps(categories))


async def get_library_models(pool: asyncpg.Pool, search: str = None) -> list[dict]:
    async with pool.acquire() as conn:
        if search:
            rows = await conn.fetch(
                "SELECT * FROM ollama_library_cache WHERE name ILIKE $1 ORDER BY name",
                f"%{search}%")
        else:
            rows = await conn.fetch("SELECT * FROM ollama_library_cache ORDER BY name")
    return [dict(r) for r in rows]


async def get_library_model(pool: asyncpg.Pool, name: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM ollama_library_cache WHERE name = $1", name)
    return dict(row) if row else None


async def get_library_cache_age_hours(pool: asyncpg.Pool) -> Optional[float]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT updated_at FROM library_cache_meta WHERE key = 'last_full_refresh'")
    if not row:
        return None
    import datetime
    age = datetime.datetime.now(datetime.timezone.utc) - row["updated_at"]
    return age.total_seconds() / 3600


async def set_library_cache_meta(pool: asyncpg.Pool, key: str, value: str):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO library_cache_meta (key, value, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """, key, value)


# ── Safety tags ──────────────────────────────────────────────────────────────

async def get_safety_tags(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM model_safety_tags ORDER BY id")
    return [dict(r) for r in rows]


async def create_safety_tag(pool: asyncpg.Pool, pattern: str, classification: str,
                             reason: str = "") -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO model_safety_tags (pattern, classification, reason)
            VALUES ($1, $2, $3) RETURNING id
        """, pattern, classification, reason)
    return row["id"]


async def update_safety_tag(pool: asyncpg.Pool, tag_id: int, pattern: str,
                             classification: str, reason: str = "") -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE model_safety_tags SET pattern = $2, classification = $3, reason = $4
            WHERE id = $1
        """, tag_id, pattern, classification, reason)
    return "UPDATE 1" in result


async def delete_safety_tag(pool: asyncpg.Pool, tag_id: int) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM model_safety_tags WHERE id = $1", tag_id)
    return "DELETE 1" in result


# ── App unsafe permission ────────────────────────────────────────────────────

async def set_app_allow_unsafe(pool: asyncpg.Pool, app_id: int, allow_unsafe: bool) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE registered_apps SET allow_unsafe = $2 WHERE id = $1",
            app_id, allow_unsafe)
    return "UPDATE 1" in result


# ── Cloud model config ────────────────────────────────────────────────────────

async def get_cloud_model_configs(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM cloud_model_config ORDER BY provider, model_id")
    return [dict(r) for r in rows]


async def get_cloud_model_config(pool: asyncpg.Pool, model_id: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM cloud_model_config WHERE model_id = $1", model_id)
    return dict(row) if row else None


async def upsert_cloud_model_config(
    pool: asyncpg.Pool,
    model_id: str,
    provider: str = "anthropic",
    enabled: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    display_name: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO cloud_model_config (model_id, provider, enabled, max_tokens,
                                            temperature, display_name, config, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW())
            ON CONFLICT (model_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                enabled = EXCLUDED.enabled,
                max_tokens = COALESCE(EXCLUDED.max_tokens, cloud_model_config.max_tokens),
                temperature = COALESCE(EXCLUDED.temperature, cloud_model_config.temperature),
                display_name = COALESCE(EXCLUDED.display_name, cloud_model_config.display_name),
                config = COALESCE(EXCLUDED.config, cloud_model_config.config),
                updated_at = NOW()
        """, model_id, provider, enabled,
            max_tokens, temperature, display_name,
            json.dumps(config) if config else None)


async def delete_cloud_model_config(pool: asyncpg.Pool, model_id: str) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM cloud_model_config WHERE model_id = $1", model_id)
    return "DELETE 1" in result
