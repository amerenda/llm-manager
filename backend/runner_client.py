"""Resolve DB runner rows to LLMAgentClient — shared by FastAPI and scheduler_worker.

The scheduler must not ``import main`` (full app import: duplicate Prometheus metrics,
heavy module graph, lifespan wiring). Use this module instead.
"""

from __future__ import annotations

import os
from typing import Optional

import asyncpg
from fastapi import HTTPException

import db
from llm_agent import LLMAgentClient, client_from_runner_row

AGENT_PSK = os.environ.get("LLM_MANAGER_AGENT_PSK", "")


def llm_agent_client_for_runner_row(r: dict) -> LLMAgentClient:
    """Build an agent client from a DB runner row (no extra query)."""
    return client_from_runner_row(r, AGENT_PSK)


async def get_runner_llm_client(
    pool: asyncpg.Pool,
    runner_id: Optional[int] = None,
    allowed_runner_ids: Optional[list[int]] = None,
) -> LLMAgentClient:
    """Return an LLMAgentClient pointed at an active (enabled) runner.
    If allowed_runner_ids is set, only those runners are candidates.

    When runner_id is None, draining runners are excluded from the fallback pick.
    An explicit runner_id still allows a drained runner (maintenance).
    """
    runners_list = await db.get_active_runners(pool)
    if allowed_runner_ids:
        runners_list = [r for r in runners_list if r["id"] in allowed_runner_ids]
    if not runners_list:
        raise HTTPException(503, "No active llm-runners available")
    if runner_id is not None:
        r = next((x for x in runners_list if x["id"] == runner_id), None)
        if not r:
            raise HTTPException(404, "Runner not found or inactive")
    else:
        eligible = [x for x in runners_list if not x.get("draining")]
        if not eligible:
            raise HTTPException(503, "No active llm-runners available (all are draining)")
        r = eligible[0]
    return llm_agent_client_for_runner_row(r)
