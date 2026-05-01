"""Shared queue admission policy for /api/queue/* and /v1/chat/completions."""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import HTTPException

import queue_db

logger = logging.getLogger(__name__)


def model_passes_category_filter(
    model_categories: list,
    allowed: list,
    excluded: list,
) -> bool:
    """False if the model is blocked by the app's category permissions."""
    if not model_categories:
        return True
    if allowed and not any(c in allowed for c in model_categories):
        return False
    if excluded and all(c in excluded for c in model_categories):
        return False
    return True


async def ensure_category_access(
    pool,
    app_id: Optional[int],
    model: str,
    *,
    batch_model_label: Optional[str] = None,
) -> None:
    """403 if the app's category rules block this model."""
    if app_id is None:
        return
    perms = await queue_db.get_app_category_perms(pool, app_id)
    if not perms["allowed_categories"] and not perms["excluded_categories"]:
        return
    model_settings = await queue_db.get_model_settings(pool, model)
    model_cats = list(model_settings.get("categories") or [])
    if model_passes_category_filter(
        model_cats, perms["allowed_categories"], perms["excluded_categories"]
    ):
        return
    if batch_model_label:
        raise HTTPException(
            403,
            f"Model {batch_model_label} not accessible under app category restrictions",
        )
    raise HTTPException(403, "Model not accessible under app category restrictions")


async def check_queue_rate_limit(pool, app_id: Optional[int]) -> None:
    """429 if the app exceeded queue depth or jobs-per-minute caps."""
    if app_id is None:
        return
    limits = await queue_db.get_rate_limit(pool, app_id)
    queued = await queue_db.count_app_queued_jobs(pool, app_id)
    if queued >= limits["max_queue_depth"]:
        raise HTTPException(
            429, f"Queue depth limit reached ({limits['max_queue_depth']})"
        )
    recent = await queue_db.count_app_recent_jobs(pool, app_id)
    if recent >= limits["max_jobs_per_minute"]:
        raise HTTPException(
            429, f"Rate limit reached ({limits['max_jobs_per_minute']}/min)"
        )


async def priority_for_app(pool, app_id: Optional[int]) -> int:
    """Map the calling app to a queue priority (see HIGH_PRIORITY_APPS)."""
    if app_id is None:
        return 0
    high = [
        s.strip().lower()
        for s in os.getenv("HIGH_PRIORITY_APPS", "").split(",")
        if s.strip()
    ]
    if not high:
        return 0
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT name FROM registered_apps WHERE id = $1", app_id
        )
    if not row:
        return 0
    return 1 if (row["name"] or "").lower() in high else 0
