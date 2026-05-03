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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import asyncpg

import db
import queue_db
from db import init_db
from leader_election import (
    init_scheduler_lease_table,
    install_sigterm,
    make_leader_elector,
)
from runner_client import get_runner_llm_client
from scheduler_v2 import SimplifiedScheduler

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")

# If the dispatch asyncio task exits while we still hold the K8s/Postgres lease, jobs
# stay queued forever (lease renews on the elector coroutine). Watchdog restarts the loop.
DISPATCH_WATCHDOG_SEC = float(os.environ.get("SCHEDULER_DISPATCH_WATCHDOG_SEC", "45"))
DISABLE_SCHEDULER = os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes")

_health_httpd: Optional[HTTPServer] = None
_health_thread: Optional[threading.Thread] = None


class _SchedulerHealthHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health" or self.path.startswith("/health?"):
            body = b'{"ok":true,"service":"llm-manager-scheduler"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "not found")


def _start_health_http(port: int) -> None:
    """Serve /health on a background thread so kube probes are not starved by the asyncio elector."""
    global _health_httpd, _health_thread
    try:
        _health_httpd = HTTPServer(("0.0.0.0", port), _SchedulerHealthHandler)
    except OSError as e:
        logger.error(
            "Scheduler health bind failed on 0.0.0.0:%s (is the port in use?): %s",
            port,
            e,
        )
        raise
    _health_thread = threading.Thread(
        target=_health_httpd.serve_forever,
        name="scheduler-health",
        daemon=True,
    )
    _health_thread.start()
    logger.info("Scheduler health server on 0.0.0.0:%s (thread)", port)


def _stop_health_http() -> None:
    global _health_httpd, _health_thread
    if _health_httpd is not None:
        try:
            _health_httpd.shutdown()
        except Exception:
            logger.warning("health server shutdown failed", exc_info=True)
        try:
            _health_httpd.server_close()
        except Exception:
            pass
        if _health_thread is not None and _health_thread.is_alive():
            _health_thread.join(timeout=10)
        _health_httpd = None
        _health_thread = None


async def _dispatch_loop_watchdog(
    elector,
    scheduler: SimplifiedScheduler,
    pool: asyncpg.Pool,
) -> None:
    if DISPATCH_WATCHDOG_SEC <= 0:
        return
    while True:
        await asyncio.sleep(DISPATCH_WATCHDOG_SEC)
        try:
            if not elector.is_leader():
                continue
            if scheduler.is_dispatch_loop_running:
                continue
            recovered = await queue_db.recover_stuck_jobs(pool)
            if recovered:
                logger.warning(
                    "Dispatch watchdog: recovered %d stuck jobs before restart",
                    recovered,
                )
            logger.error(
                "Dispatch loop not running while leader — restarting SimplifiedScheduler "
                "(likely loop exited without lease loss; see prior logs for exceptions)"
            )
            await scheduler.stop_and_wait()
            scheduler.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("scheduler dispatch watchdog tick failed")


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
    health_port = int(os.environ.get("SCHEDULER_HEALTH_PORT", "8081"))
    if health_port > 0:
        _start_health_http(health_port)

    try:
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

            async def get_runner(runner_id=None):
                return await get_runner_llm_client(pool, runner_id=runner_id)

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
                await scheduler.stop_and_wait()

            loop = asyncio.get_running_loop()
            install_sigterm(elector, loop)
            logger.info(
                "Scheduler worker running (leader backend=%s)",
                os.environ.get("SCHEDULER_LEADER_BACKEND", "none"),
            )
            watchdog: asyncio.Task | None = None
            if DISPATCH_WATCHDOG_SEC > 0:
                watchdog = asyncio.create_task(
                    _dispatch_loop_watchdog(elector, scheduler, pool),
                    name="scheduler-dispatch-watchdog",
                )
            try:
                await elector.run(on_leadership_gained, on_leadership_lost)
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    try:
                        await watchdog
                    except asyncio.CancelledError:
                        pass
        finally:
            if scheduler is not None:
                await scheduler.stop_and_wait()
            if elector is not None:
                await elector.shutdown()
            await pool.close()

    finally:
        if health_port > 0:
            _stop_health_http()


if __name__ == "__main__":
    asyncio.run(main())
