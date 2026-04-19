"""
Queue strategies for the simplified scheduler.

A strategy decides *which* jobs to dispatch next from the pending queue.
The scheduler then figures out *where* to run them (runner selection,
model swap). This keeps policy (strategy) separate from mechanism
(scheduler).

Strategies:

- FifoStrategy: current behavior — return the single oldest queued job.
  Strict FIFO within priority.
- PriorityBatchingStrategy: return up to N consecutive same-model jobs
  from the head of the priority-ordered queue. Reduces swap churn when
  one app submits bursts of same-model jobs. Priority+age ordering is
  handled by queue_db.get_pending_jobs, so batching just consumes from
  the already-ordered head.

Switch via QUEUE_STRATEGY env var. Default: priority_batching.
Switching is "destructive" in the sense that in-flight state doesn't
migrate — the scheduler restarts and recover_stuck_jobs reverts
anything orphaned.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import asyncpg

import queue_db

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 5


class QueueStrategy(Protocol):
    """A strategy returns the next set of jobs to dispatch together.

    Empty list = nothing ready to run right now. The scheduler will sleep
    briefly and try again.
    """
    name: str

    async def next_jobs(self, pool: asyncpg.Pool) -> list[dict]: ...


class FifoStrategy:
    """Single-job strict FIFO (same as the v1 simplified scheduler)."""

    name = "fifo"

    async def next_jobs(self, pool: asyncpg.Pool) -> list[dict]:
        jobs = await queue_db.get_pending_jobs(pool, limit=1)
        return jobs


class PriorityBatchingStrategy:
    """Take up to batch_size consecutive same-model jobs from the head of
    the priority-ordered queue.

    Example: head is [qwen3:14b, qwen3:14b, qwen3:14b, deepseek-r1:14b, ...].
    Returns the first three; the next call (after they run) will see
    deepseek at the head and batch from there.

    Priority/age ordering is applied by get_pending_jobs, so this strategy
    is purely "how many adjacent same-model jobs to grab." Within a batch,
    FIFO by created_at (that's how the query orders them).
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.name = f"priority_batching(N={batch_size})"

    async def next_jobs(self, pool: asyncpg.Pool) -> list[dict]:
        # Fetch a bit more than batch_size so we can see ahead without pulling
        # the whole queue. 3x is plenty — the head block of same-model is
        # usually short.
        fetch_limit = max(self.batch_size * 3, 10)
        jobs = await queue_db.get_pending_jobs(pool, limit=fetch_limit)
        if not jobs:
            return []

        head_model = jobs[0]["model"]
        batch = []
        for job in jobs:
            if job["model"] != head_model:
                break
            batch.append(job)
            if len(batch) >= self.batch_size:
                break
        return batch


def make_strategy(name: str | None = None) -> QueueStrategy:
    """Construct the configured strategy. Reads QUEUE_STRATEGY env var if
    name is None. Unknown names fall back to priority_batching with a warning."""
    name = name or os.getenv("QUEUE_STRATEGY", "priority_batching")
    name = name.strip().lower()

    if name in ("fifo", "strict_fifo"):
        return FifoStrategy()
    if name in ("priority_batching", "batching", "priority-batching"):
        size = int(os.getenv("QUEUE_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
        return PriorityBatchingStrategy(batch_size=size)

    logger.warning("Unknown QUEUE_STRATEGY=%r — defaulting to priority_batching", name)
    return PriorityBatchingStrategy()
