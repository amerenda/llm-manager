"""
Dedicated queue scheduler worker (HA): runs SimplifiedScheduler under pluggable leader election.

Use when LLM_MANAGER_PROCESS=api on API pods and SCHEDULER_K8s_LEASE / postgres lease
outside the API Deployment. Started as: python -m scheduler_worker
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

import asyncpg

import db
import queue_db
from db import init_db
from leader_election import (
    init_scheduler_lease_table,
    install_sigterm,
    make_leader_elector,
)
from scheduler_v2 import SimplifiedScheduler

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
DISABLE_SCHEDULER = os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes")


async def _wait_shutdown_event() -> None:
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _wake(*_a):
        loop.call_soon_threadsafe(done.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _wake)
        except NotImplementedError:
            pass
    await done.wait()


async def main() -> None:
    if DISABLE_SCHEDULER:
        logger.info("DISABLE_SCHEDULER set; scheduler_worker idle (no DB, no election).")
        await _wait_shutdown_event()
        return

    _pg_server_settings = {
        "tcp_keepalives_idle": "60",
        "tcp_keepalives_interval": "10",
        "tcp_keepalives_count": "3",
    }
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        max_inactive_connection_lifetime=float(
            os.environ.get("DB_POOL_MAX_INACTIVE_SEC", "300")
        ),
        server_settings=_pg_server_settings,
    )
    elector = None
    scheduler: SimplifiedScheduler | None = None
    try:
        await init_db(pool)
        await queue_db.init_queue_tables(pool)
        logger.info("Database connected: %s", DATABASE_URL)

        backend = os.environ.get("SCHEDULER_LEADER_BACKEND", "none").strip().lower()
        if backend in ("postgres", "pg", "sql"):
            await init_scheduler_lease_table(pool)

        import main as api_main  # noqa: E402

        async def get_runner(runner_id=None):
            return await api_main._get_runner_client(pool, runner_id=runner_id)

        scheduler = SimplifiedScheduler(pool, get_runner, lock_conn=None)
        elector = make_leader_elector(
            os.environ.get("SCHEDULER_LEADER_BACKEND", "none"), pool
        )

        async def on_leadership_gained() -> None:
            recovered = await queue_db.recover_stuck_jobs(pool)
            if recovered:
                logger.warning(
                    "Recovered %d jobs stuck in loading_model/running → queued",
                    recovered,
                )
            orphaned = await db.recover_stuck_ops(pool)
            if orphaned:
                logger.warning(
                    "Marked %d background ops as failed (orphaned by previous holder)",
                    orphaned,
                )
            scheduler.start()

        async def on_leadership_lost() -> None:
            scheduler.stop()

        loop = asyncio.get_running_loop()
        install_sigterm(elector, loop)
        logger.info(
            "Scheduler worker running (leader backend=%s)",
            os.environ.get("SCHEDULER_LEADER_BACKEND", "none"),
        )
        await elector.run(on_leadership_gained, on_leadership_lost)
    finally:
        if scheduler is not None:
            scheduler.stop()
        if elector is not None:
            await elector.shutdown()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
