#!/usr/bin/env python3
"""
Reset the UAT database with seed data.

SAFETY: Only operates on databases/tables containing "uat" in the name.
Usage: python3 reset-uat.py <DATABASE_URL>
   or: DATABASE_URL=... python3 reset-uat.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import asyncpg


SEED_RUNNERS = [
    {
        "hostname": "murderbot-uat",
        "address": "https://10.100.20.19:8090",
        "port": 8090,
        "capabilities": json.dumps({
            "gpu_vram_total_bytes": 8589934592,
            "gpu_vram_used_bytes": 0,
            "gpu_vram_free_bytes": 8589934592,
            "comfyui_running": False,
            "loaded_models": [],
            "gpu_vendor": "nvidia",
        }),
    },
    {
        "hostname": "archlinux-uat",
        "address": "https://10.100.20.18:8090",
        "port": 8090,
        "capabilities": json.dumps({
            "gpu_vram_total_bytes": 4294967296,
            "gpu_vram_used_bytes": 0,
            "gpu_vram_free_bytes": 4294967296,
            "comfyui_running": False,
            "loaded_models": [],
            "gpu_vendor": "amd",
        }),
    },
]

SEED_APPS = [
    {"name": "ecdysis-uat", "base_url": "http://ecdysis-backend-uat:8081", "status": "active"},
    {"name": "test-chatbot", "base_url": "http://chatbot-uat:3000", "status": "active"},
    {"name": "pending-app", "base_url": "http://pending:8080", "status": "pending"},
]

SEED_PROFILES = [
    {"name": "Default", "is_default": True, "unsafe_enabled": False},
    {"name": "Development", "is_default": False, "unsafe_enabled": True},
]


async def main():
    db_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("Usage: python3 reset-uat.py <DATABASE_URL>")
        print("   or: DATABASE_URL=... python3 reset-uat.py")
        sys.exit(1)

    # SAFETY CHECK: only operate on UAT databases
    if "uat" not in db_url.lower():
        print(f"SAFETY: Database URL does not contain 'uat': {db_url}")
        print("This script only operates on UAT databases. Aborting.")
        sys.exit(1)

    conn = await asyncpg.connect(db_url)
    print(f"Connected to: {db_url.split('@')[1] if '@' in db_url else db_url}")

    # Get all tables
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    table_names = [t["tablename"] for t in tables]
    print(f"Found {len(table_names)} tables: {', '.join(sorted(table_names))}")

    # Truncate all tables (cascade)
    for t in table_names:
        await conn.execute(f"TRUNCATE TABLE {t} CASCADE")
    print("Truncated all tables")

    # Seed runners
    for r in SEED_RUNNERS:
        await conn.execute("""
            INSERT INTO llm_runners (hostname, address, port, capabilities, last_seen)
            VALUES ($1, $2, $3, $4::jsonb, NOW())
        """, r["hostname"], r["address"], r["port"], r["capabilities"])
    print(f"Seeded {len(SEED_RUNNERS)} runners")

    # Seed apps
    import secrets as sec
    for a in SEED_APPS:
        api_key = sec.token_urlsafe(32)
        await conn.execute("""
            INSERT INTO registered_apps (name, base_url, api_key, status, last_seen)
            VALUES ($1, $2, $3, $4, NOW())
        """, a["name"], a["base_url"], api_key, a["status"])
        print(f"  App '{a['name']}': key={api_key[:12]}...")
    print(f"Seeded {len(SEED_APPS)} apps")

    # Seed profiles
    for p in SEED_PROFILES:
        await conn.execute("""
            INSERT INTO profiles (name, is_default, unsafe_enabled)
            VALUES ($1, $2, $3)
            ON CONFLICT (name) DO NOTHING
        """, p["name"], p["is_default"], p["unsafe_enabled"])
    print(f"Seeded {len(SEED_PROFILES)} profiles")

    # Seed safety tags
    for pattern, reason in [
        ("*uncensored*", "Model trained without safety restrictions"),
        ("dolphin-*", "Dolphin models are uncensored by design"),
        ("*abliterated*", "Model with safety training removed"),
    ]:
        await conn.execute("""
            INSERT INTO model_safety_tags (pattern, classification, reason)
            VALUES ($1, 'unsafe', $2)
            ON CONFLICT (pattern) DO NOTHING
        """, pattern, reason)
    print("Seeded safety tags")

    await conn.close()
    print("\nUAT database reset complete!")


if __name__ == "__main__":
    asyncio.run(main())
