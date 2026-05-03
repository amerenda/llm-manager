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
from scheduler_v2 import SimplifiedScheduler

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://llm:llm@localhost:5432/llmmanager")
DISABLE_SCHEDULER = os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes")


_health_tcp_server: Optional[asyncio.AbstractServer] = None

_HEALTH_BODY = (
    b'{"ok":true,"service":"llm-manager-scheduler"}',
)


async def _health_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        header = await reader.readline()
        # Drain rest of request (keep-alive not used)
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        path = header.split()[1] if len(header.split()) > 1 else b""
        if header.startswith(b"GET ") and path.startswith(b"/health"):
            h = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: "
                + str(len(_HEALTH_BODY)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + _HEALTH_BODY
            )
            writer.write(h)
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _run_health_server(port: int) -> None:
    """Minimal HTTP so k8s can probe :8081 (avoid uvicorn: it steals SIGTERM)."""
    global _health_tcp_server
    server = await asyncio.start_server(
        _health_client,
        host="0.0.0.0",
        port=port,
    )
    _health_tcp_server = server
    try:
        await server.serve_forever()
    finally:
        _health_tcp_server = None


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
    health_task: asyncio.Task[None] | None = None
    if health_port > 0:
        health_task = asyncio.create_task(_run_health_server(health_port))
        await asyncio.sleep(0)  # allow health server to start scheduling

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

    finally:
        if health_task is not None:
            srv = _health_tcp_server
            if srv is not None:
                srv.close()
                try:
                    await srv.wait_closed()
                except Exception:
                    pass
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
