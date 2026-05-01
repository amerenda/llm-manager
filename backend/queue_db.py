"""Database operations for the job queue and model settings."""
import logging
from typing import Optional
import asyncpg

logger = logging.getLogger(__name__)

QUEUE_TABLES_SQL = """

CREATE TABLE IF NOT EXISTS queue_jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT,
    app_id INTEGER REFERENCES registered_apps(id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER DEFAULT 0,
    retried INTEGER DEFAULT 0,
    request JSONB NOT NULL,
    metadata JSONB,
    result JSONB,
    error TEXT,
    runner_id INTEGER REFERENCES llm_runners(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_queue_jobs_status ON queue_jobs(status);
CREATE INDEX IF NOT EXISTS idx_queue_jobs_model ON queue_jobs(model, status);
CREATE INDEX IF NOT EXISTS idx_queue_jobs_batch ON queue_jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_queue_jobs_created ON queue_jobs(created_at);

CREATE TABLE IF NOT EXISTS model_settings (
    model_name TEXT PRIMARY KEY,
    do_not_evict BOOLEAN DEFAULT false,
    evictable BOOLEAN DEFAULT true,
    wait_for_completion BOOLEAN DEFAULT true,
    vram_estimate_gb REAL,
    categories TEXT[] NOT NULL DEFAULT '{}',
    safety TEXT NOT NULL DEFAULT 'safe',
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_rate_limits (
    app_id INTEGER PRIMARY KEY REFERENCES registered_apps(id) ON DELETE CASCADE,
    max_queue_depth INTEGER DEFAULT 50,
    max_jobs_per_minute INTEGER DEFAULT 10,
    updated_at TIMESTAMPTZ DEFAULT now()
);
"""


async def init_queue_tables(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(QUEUE_TABLES_SQL)
        # Migrations for model_settings
        for stmt in [
            "ALTER TABLE model_settings ADD COLUMN IF NOT EXISTS categories TEXT[] NOT NULL DEFAULT '{}'",
            "ALTER TABLE model_settings ADD COLUMN IF NOT EXISTS safety TEXT NOT NULL DEFAULT 'safe'",
            # Added 2026-04: expose which runner a job is dispatched to so the
            # Active Queue UI can show it. Set by the scheduler on
            # loading_model/running transitions.
            "ALTER TABLE queue_jobs ADD COLUMN IF NOT EXISTS runner_id INTEGER REFERENCES llm_runners(id) ON DELETE SET NULL",
            # Added 2026-04: track how many times a job has been re-queued after
            # transient failures (5xx from runner). Scheduler fails permanently after MAX_RETRIES.
            "ALTER TABLE queue_jobs ADD COLUMN IF NOT EXISTS retried INTEGER DEFAULT 0",
            # When a job entered loading_model (swap). Used for time-based recovery
            # if the process dies mid-load — started_at is only set at running.
            "ALTER TABLE queue_jobs ADD COLUMN IF NOT EXISTS loading_model_at TIMESTAMPTZ",
            # Terminal rows without completed_at sorted first in ORDER BY completed_at DESC
            # (Postgres NULLS FIRST default) and broke Recent History; backfill once.
            """UPDATE queue_jobs SET completed_at = COALESCE(started_at, created_at)
               WHERE status IN ('completed', 'failed', 'cancelled') AND completed_at IS NULL""",
            # Queued work is not bound to a runner; stale runner_id from retries/recovery
            # made the Active Queue show misleading hostnames.
            "UPDATE queue_jobs SET runner_id = NULL WHERE status = 'queued' AND runner_id IS NOT NULL",
        ]:
            try:
                await conn.execute(stmt)
            except Exception:
                logger.exception("queue table migration failed: %s", stmt[:120])
                raise
    logger.info("Queue tables initialized")


async def resolved_vram_gb_for_model(pool: asyncpg.Pool, model: str) -> float:
    """VRAM estimate for a model: ``model_settings.vram_estimate_gb`` if set,
    otherwise the heuristic in ``gpu.vram_for_model``."""
    from gpu import vram_for_model

    settings = await get_model_settings(pool, model)
    v = settings.get("vram_estimate_gb")
    if v is not None:
        return float(v)
    return vram_for_model(model)


# ── Job CRUD ──────────────────────────────────────────────────────────────────

async def insert_job(pool: asyncpg.Pool, job_id: str, batch_id: Optional[str],
                     app_id: Optional[int], model: str, request: dict,
                     metadata: Optional[dict], priority: int = 0) -> dict:
    import json
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO queue_jobs (id, batch_id, app_id, model, request, metadata, priority)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
            RETURNING *
        """, job_id, batch_id, app_id, model,
            json.dumps(request), json.dumps(metadata) if metadata else None, priority)
    return dict(row)


async def get_job(pool: asyncpg.Pool, job_id: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM queue_jobs WHERE id = $1", job_id)
    return dict(row) if row else None


async def get_batch_jobs(pool: asyncpg.Pool, batch_id: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM queue_jobs WHERE batch_id = $1 ORDER BY created_at", batch_id)
    return [dict(r) for r in rows]


async def update_job_status(pool: asyncpg.Pool, job_id: str, status: str,
                            result: Optional[dict] = None, error: Optional[str] = None,
                            runner_id: Optional[int] = None):
    """Update job status. When runner_id is provided it's written in the same
    UPDATE — used by the scheduler to stamp the dispatched runner on the
    loading_model → running transitions so the Active Queue UI can show it.
    Terminal transitions (completed/failed/cancelled) leave runner_id alone."""
    import json
    async with pool.acquire() as conn:
        if status == "running":
            if runner_id is not None:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, runner_id = $3,
                        started_at = now(), loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status, runner_id)
            else:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, started_at = now(),
                        loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status)
        elif status in ("completed", "failed", "cancelled"):
            if result is not None and status == "completed":
                from result_slim import slim_stored_result

                result = slim_stored_result(result)
            await conn.execute("""
                UPDATE queue_jobs SET status = $2, result = $3::jsonb, error = $4,
                    completed_at = now(), loading_model_at = NULL
                WHERE id = $1
                AND status NOT IN ('completed', 'failed', 'cancelled')
            """, job_id, status, json.dumps(result) if result else None, error)
        elif status == "queued":
            # Re-queue after retry: must not keep a runner assignment — UI and
            # placement logic assume queued jobs have no host yet.
            if error is not None:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, error = $3,
                        loading_model_at = NULL, runner_id = NULL
                    WHERE id = $1
                """, job_id, status, error)
            else:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2,
                        loading_model_at = NULL, runner_id = NULL
                    WHERE id = $1
                """, job_id, status)
        else:
            if status == "loading_model":
                if runner_id is not None:
                    await conn.execute("""
                        UPDATE queue_jobs SET status = $2, runner_id = $3,
                            loading_model_at = now()
                        WHERE id = $1
                    """, job_id, status, runner_id)
                else:
                    await conn.execute("""
                        UPDATE queue_jobs SET status = $2, loading_model_at = now()
                        WHERE id = $1
                    """, job_id, status)
            elif runner_id is not None and error is not None:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, runner_id = $3, error = $4,
                        loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status, runner_id, error)
            elif runner_id is not None:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, runner_id = $3,
                        loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status, runner_id)
            elif error is not None:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, error = $3, loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status, error)
            else:
                await conn.execute("""
                    UPDATE queue_jobs SET status = $2, loading_model_at = NULL
                    WHERE id = $1
                """, job_id, status)


async def increment_job_retry(pool: asyncpg.Pool, job_id: str) -> int:
    """Atomically increment and return the new retry count for a job."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE queue_jobs SET retried = retried + 1 WHERE id = $1 RETURNING retried",
            job_id)
    return row["retried"] if row else 1


async def get_pending_jobs(pool: asyncpg.Pool, limit: int = 100) -> list[dict]:
    """Get queued jobs ordered by effective priority (priority + age bonus)
    then created_at.

    The effective priority adds 1 point per 600s (10 min) of waiting. This
    gives higher-priority apps a soft edge: a fresh priority=1 job beats a
    fresh priority=0 job, but a priority=0 job that has waited 10 minutes
    catches up, and one that has waited longer wins. Prevents starvation
    without a separate age-bump pass.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT q.*, a.name AS app_name
            FROM queue_jobs q
            LEFT JOIN registered_apps a ON a.id = q.app_id
            WHERE q.status IN ('queued', 'waiting_for_eviction')
            ORDER BY
                (q.priority + EXTRACT(EPOCH FROM (now() - q.created_at)) / 600) DESC,
                q.created_at ASC
            LIMIT $1
        """, limit)
    return [dict(r) for r in rows]


async def get_active_jobs_for_model(pool: asyncpg.Pool, model: str) -> list[dict]:
    """Get jobs currently running for a specific model."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM queue_jobs WHERE model = $1 AND status = 'running'
        """, model)
    return [dict(r) for r in rows]


async def get_running_jobs(pool: asyncpg.Pool, limit: int = 10) -> list[dict]:
    """Get currently running jobs."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM queue_jobs WHERE status = 'running'
            ORDER BY started_at ASC LIMIT $1
        """, limit)
    return [dict(r) for r in rows]


async def get_all_active_jobs(pool: asyncpg.Pool) -> list[dict]:
    """Get all non-terminal jobs (queued, loading, running, waiting)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, status, model FROM queue_jobs
            WHERE status IN ('queued', 'running', 'loading_model', 'waiting_for_eviction')
        """)
    return [dict(r) for r in rows]


async def count_app_queued_jobs(pool: asyncpg.Pool, app_id: int) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT COUNT(*) as cnt FROM queue_jobs
            WHERE app_id = $1 AND status IN ('queued', 'running', 'loading_model', 'waiting_for_eviction')
        """, app_id)
    return row["cnt"]


async def count_app_recent_jobs(pool: asyncpg.Pool, app_id: int) -> int:
    """Count jobs submitted in the last minute for rate limiting."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT COUNT(*) as cnt FROM queue_jobs
            WHERE app_id = $1 AND created_at > now() - interval '1 minute'
        """, app_id)
    return row["cnt"]


async def recover_stuck_jobs(pool: asyncpg.Pool) -> int:
    """Reset jobs stuck in loading_model/running back to queued on startup."""
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE queue_jobs SET status = 'queued', started_at = NULL,
                loading_model_at = NULL, runner_id = NULL
            WHERE status IN ('loading_model', 'running')
        """)
    count = int(result.split()[-1])
    if count:
        logger.info("Recovered %d stuck jobs → queued", count)
    return count


async def recover_stale_in_progress_jobs(
    pool: asyncpg.Pool,
    loading_minutes: int = 45,
    running_minutes: int = 360,
    loading_fallback_hours: int = 24,
) -> int:
    """Re-queue jobs stuck in loading_model or running longer than thresholds.

    get_pending_jobs only returns queued/waiting — without this, a wedged swap
    or crashed worker leaves rows invisible until process restart or lock
    handoff. ``loading_model_at`` is set when entering loading_model; rows with
    NULL (pre-migration) use ``created_at`` and loading_fallback_hours.
    """
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE queue_jobs SET status = 'queued', started_at = NULL,
                loading_model_at = NULL, runner_id = NULL
            WHERE (
                status = 'loading_model' AND (
                    (loading_model_at IS NOT NULL
                        AND loading_model_at < now() - make_interval(mins => $1::int))
                    OR (loading_model_at IS NULL
                        AND created_at < now() - make_interval(hours => $3::int))
                )
            )
            OR (
                status = 'running'
                AND started_at IS NOT NULL
                AND started_at < now() - make_interval(mins => $2::int)
            )
        """, loading_minutes, running_minutes, loading_fallback_hours)
    count = int(result.split()[-1])
    if count:
        logger.warning(
            "Recovered %d stale in-progress jobs (loading>%dm or running>%dm, "
            "or loading_model without timestamp >%dh) → queued",
            count,
            loading_minutes,
            running_minutes,
            loading_fallback_hours,
        )
    return count


async def cleanup_old_jobs(pool: asyncpg.Pool, hours: int = 24):
    """Delete completed/failed jobs older than N hours."""
    async with pool.acquire() as conn:
        deleted = await conn.execute("""
            DELETE FROM queue_jobs
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND completed_at < now() - make_interval(hours => $1)
        """, hours)
    logger.info("Cleaned up old jobs: %s", deleted)


# ── Model Settings ────────────────────────────────────────────────────────────

async def get_model_settings(pool: asyncpg.Pool, model_name: str) -> dict:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM model_settings WHERE model_name = $1", model_name)
    if row:
        return dict(row)
    return {
        "model_name": model_name,
        "do_not_evict": False,
        "evictable": True,
        "wait_for_completion": True,
        "vram_estimate_gb": None,
        "categories": [],
        "safety": "safe",
    }


async def get_all_model_settings(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM model_settings ORDER BY model_name")
    return [dict(r) for r in rows]


async def upsert_model_settings(pool: asyncpg.Pool, model_name: str, **kwargs):
    fields = {k: v for k, v in kwargs.items() if v is not None}
    if not fields:
        return
    async with pool.acquire() as conn:
        # Ensure row exists
        await conn.execute("""
            INSERT INTO model_settings (model_name) VALUES ($1)
            ON CONFLICT (model_name) DO NOTHING
        """, model_name)
        for key, value in fields.items():
            await conn.execute(f"""
                UPDATE model_settings SET {key} = $2, updated_at = now()
                WHERE model_name = $1
            """, model_name, value)


# ── Rate Limits ───────────────────────────────────────────────────────────────

async def update_job_priority(pool: asyncpg.Pool, job_id: str, priority: int):
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE queue_jobs SET priority = $2 WHERE id = $1", job_id, priority)


async def list_jobs(pool: asyncpg.Pool, status: Optional[str] = None, limit: int = 100) -> list[dict]:
    """List jobs with optional status filter. Includes app name and — when
    the scheduler has dispatched the job — runner hostname."""
    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch("""
                SELECT j.*, a.name as app_name, r.hostname as runner_hostname
                FROM queue_jobs j
                LEFT JOIN registered_apps a ON j.app_id = a.id
                LEFT JOIN llm_runners r ON j.runner_id = r.id
                WHERE j.status = $1
                ORDER BY j.priority DESC, j.created_at ASC
                LIMIT $2
            """, status, limit)
        else:
            rows = await conn.fetch("""
                SELECT j.*, a.name as app_name, r.hostname as runner_hostname
                FROM queue_jobs j
                LEFT JOIN registered_apps a ON j.app_id = a.id
                LEFT JOIN llm_runners r ON j.runner_id = r.id
                WHERE j.status NOT IN ('completed', 'failed', 'cancelled')
                ORDER BY
                    CASE j.status WHEN 'running' THEN 0 WHEN 'loading_model' THEN 1 ELSE 2 END,
                    j.priority DESC, j.created_at ASC
                LIMIT $1
            """, limit)
    result = []
    for r in rows:
        d = dict(r)
        # Convert timestamps to ISO strings
        for ts_field in ('created_at', 'started_at', 'completed_at'):
            if d.get(ts_field):
                d[ts_field] = d[ts_field].isoformat()
        # Don't send full request/result blobs in listings
        if d.get('request'):
            req = d['request']
            if isinstance(req, str):
                import json as _json
                req = _json.loads(req)
            d['request_summary'] = {
                'message_count': len(req.get('messages', [])),
                'temperature': req.get('temperature'),
                'max_tokens': req.get('max_tokens'),
            }
        d.pop('request', None)
        d.pop('result', None)
        if isinstance(d.get('metadata'), str):
            import json as _json
            d['metadata'] = _json.loads(d['metadata'])
        result.append(d)
    return result


async def get_queue_metrics(pool: asyncpg.Pool) -> dict:
    """Get comprehensive queue metrics."""
    async with pool.acquire() as conn:
        # Current counts by status
        status_counts = await conn.fetch("""
            SELECT status, COUNT(*) as cnt FROM queue_jobs
            WHERE status NOT IN ('completed', 'failed', 'cancelled')
            GROUP BY status
        """)

        # Completed/failed in last hour
        hourly = await conn.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'completed') as completed_1h,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_1h,
                COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_1h
            FROM queue_jobs
            WHERE completed_at > now() - interval '1 hour'
        """)

        # Average processing time (last hour, completed only)
        avg_times = await conn.fetchrow("""
            SELECT
                AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_secs,
                AVG(EXTRACT(EPOCH FROM (started_at - created_at))) as avg_wait_secs,
                MAX(EXTRACT(EPOCH FROM (completed_at - started_at))) as max_processing_secs,
                MIN(EXTRACT(EPOCH FROM (completed_at - started_at))) as min_processing_secs
            FROM queue_jobs
            WHERE status = 'completed'
            AND completed_at > now() - interval '1 hour'
            AND started_at IS NOT NULL
        """)

        # Jobs per model (last hour)
        model_breakdown = await conn.fetch("""
            SELECT model,
                   COUNT(*) as total,
                   COUNT(*) FILTER (WHERE status = 'completed') as completed,
                   COUNT(*) FILTER (WHERE status = 'failed') as failed,
                   AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))
                       FILTER (WHERE status = 'completed' AND started_at IS NOT NULL) as avg_secs
            FROM queue_jobs
            WHERE created_at > now() - interval '1 hour'
            GROUP BY model
            ORDER BY total DESC
        """)

        # Jobs per app (last hour)
        app_breakdown = await conn.fetch("""
            SELECT a.name as app_name, COUNT(*) as total,
                   COUNT(*) FILTER (WHERE j.status = 'completed') as completed,
                   COUNT(*) FILTER (WHERE j.status = 'failed') as failed
            FROM queue_jobs j
            LEFT JOIN registered_apps a ON j.app_id = a.id
            WHERE j.created_at > now() - interval '1 hour'
            GROUP BY a.name
            ORDER BY total DESC
        """)

        # Total all-time
        totals = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_all_time,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_all_time,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_all_time
            FROM queue_jobs
        """)

    active_statuses = {r['status']: r['cnt'] for r in status_counts}

    return {
        "active": {
            "queued": active_statuses.get('queued', 0),
            "running": active_statuses.get('running', 0),
            "loading_model": active_statuses.get('loading_model', 0),
            "waiting_for_eviction": active_statuses.get('waiting_for_eviction', 0),
        },
        "last_hour": {
            "completed": hourly['completed_1h'],
            "failed": hourly['failed_1h'],
            "cancelled": hourly['cancelled_1h'],
        },
        "timing": {
            "avg_processing_secs": round(avg_times['avg_processing_secs'] or 0, 2),
            "avg_wait_secs": round(avg_times['avg_wait_secs'] or 0, 2),
            "max_processing_secs": round(avg_times['max_processing_secs'] or 0, 2),
            "min_processing_secs": round(avg_times['min_processing_secs'] or 0, 2),
        },
        "by_model": [
            {
                "model": r['model'],
                "total": r['total'],
                "completed": r['completed'],
                "failed": r['failed'],
                "avg_secs": round(r['avg_secs'] or 0, 2),
            }
            for r in model_breakdown
        ],
        "by_app": [
            {
                "app_name": r['app_name'] or 'unknown',
                "total": r['total'],
                "completed": r['completed'],
                "failed": r['failed'],
            }
            for r in app_breakdown
        ],
        "totals": {
            "all_time": totals['total_all_time'],
            "completed": totals['completed_all_time'],
            "failed": totals['failed_all_time'],
        },
    }


async def list_recent_jobs(pool: asyncpg.Pool, limit: int = 50) -> list[dict]:
    """List recently completed/failed jobs."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT j.*, a.name as app_name, r.hostname as runner_hostname
            FROM queue_jobs j
            LEFT JOIN registered_apps a ON j.app_id = a.id
            LEFT JOIN llm_runners r ON j.runner_id = r.id
            WHERE j.status IN ('completed', 'failed', 'cancelled')
            ORDER BY j.completed_at DESC NULLS LAST, j.created_at DESC
            LIMIT $1
        """, limit)
    result = []
    for r in rows:
        d = dict(r)
        for ts_field in ('created_at', 'started_at', 'completed_at'):
            if d.get(ts_field):
                d[ts_field] = d[ts_field].isoformat()
        d.pop('request', None)
        d.pop('result', None)
        if isinstance(d.get('metadata'), str):
            import json as _json
            d['metadata'] = _json.loads(d['metadata'])
        result.append(d)
    return result


# ── Model Aliases ─────────────────────────────────────────────────────────────

async def get_all_model_aliases(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM model_aliases ORDER BY alias_name")
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("parameters"), str):
            import json as _j
            d["parameters"] = _j.loads(d["parameters"])
        result.append(d)
    return result


async def get_aliases_for_base_model(pool: asyncpg.Pool, base_model: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM model_aliases WHERE base_model = $1 ORDER BY alias_name", base_model)
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("parameters"), str):
            import json as _j
            d["parameters"] = _j.loads(d["parameters"])
        result.append(d)
    return result


async def get_model_alias(pool: asyncpg.Pool, alias_name: str) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM model_aliases WHERE alias_name = $1", alias_name)
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("parameters"), str):
        import json as _j
        d["parameters"] = _j.loads(d["parameters"])
    return d


async def upsert_model_alias(
    pool: asyncpg.Pool, alias_name: str, base_model: str,
    system_prompt: Optional[str], parameters: dict, description: str = "",
) -> dict:
    import json as _j
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO model_aliases (alias_name, base_model, system_prompt, parameters, description)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (alias_name) DO UPDATE SET
                base_model = EXCLUDED.base_model,
                system_prompt = EXCLUDED.system_prompt,
                parameters = EXCLUDED.parameters,
                description = EXCLUDED.description,
                updated_at = NOW()
            RETURNING *
        """, alias_name, base_model, system_prompt, _j.dumps(parameters), description)
    d = dict(row)
    if isinstance(d.get("parameters"), str):
        d["parameters"] = _j.loads(d["parameters"])
    return d


async def delete_model_alias(pool: asyncpg.Pool, alias_name: str) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM model_aliases WHERE alias_name = $1", alias_name)
    return result == "DELETE 1"


# ── Model Runner Params ───────────────────────────────────────────────────────

async def get_model_runner_params(
    pool: asyncpg.Pool, model_name: str, runner_id: int
) -> Optional[dict]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM model_runner_params WHERE model_name = $1 AND runner_id = $2",
            model_name, runner_id)
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("parameters"), str):
        import json as _j
        d["parameters"] = _j.loads(d["parameters"])
    return d


async def get_all_runner_params_for_model(pool: asyncpg.Pool, model_name: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT mrp.*, r.hostname
            FROM model_runner_params mrp
            JOIN llm_runners r ON r.id = mrp.runner_id
            WHERE mrp.model_name = $1
            ORDER BY r.hostname
        """, model_name)
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("parameters"), str):
            import json as _j
            d["parameters"] = _j.loads(d["parameters"])
        result.append(d)
    return result


async def upsert_model_runner_params(
    pool: asyncpg.Pool, model_name: str, runner_id: int,
    system_prompt: Optional[str], parameters: dict,
    do_not_evict: bool = False,
) -> dict:
    import json as _j
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO model_runner_params (model_name, runner_id, system_prompt, parameters, do_not_evict)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (model_name, runner_id) DO UPDATE SET
                system_prompt = EXCLUDED.system_prompt,
                parameters = EXCLUDED.parameters,
                do_not_evict = EXCLUDED.do_not_evict,
                updated_at = NOW()
            RETURNING *
        """, model_name, runner_id, system_prompt, _j.dumps(parameters), do_not_evict)
    d = dict(row)
    if isinstance(d.get("parameters"), str):
        d["parameters"] = _j.loads(d["parameters"])
    return d


async def set_runner_model_pin(
    pool: asyncpg.Pool, model_name: str, runner_id: int, do_not_evict: bool
) -> None:
    """Create or update only the do_not_evict flag without touching system_prompt/parameters."""
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO model_runner_params (model_name, runner_id, do_not_evict)
            VALUES ($1, $2, $3)
            ON CONFLICT (model_name, runner_id) DO UPDATE SET
                do_not_evict = EXCLUDED.do_not_evict,
                updated_at = NOW()
        """, model_name, runner_id, do_not_evict)


async def delete_model_runner_params(
    pool: asyncpg.Pool, model_name: str, runner_id: int
) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM model_runner_params WHERE model_name = $1 AND runner_id = $2",
            model_name, runner_id)
    return result == "DELETE 1"


async def get_app_category_perms(pool: asyncpg.Pool, app_id: int) -> dict:
    """Return allowed_categories and excluded_categories for an app."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT allowed_categories, excluded_categories FROM registered_apps WHERE id = $1",
            app_id,
        )
    if not row:
        return {"allowed_categories": [], "excluded_categories": []}
    return {
        "allowed_categories": list(row["allowed_categories"] or []),
        "excluded_categories": list(row["excluded_categories"] or []),
    }


async def get_rate_limit(pool: asyncpg.Pool, app_id: int) -> dict:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM app_rate_limits WHERE app_id = $1", app_id)
    if row:
        return dict(row)
    return {"app_id": app_id, "max_queue_depth": 50, "max_jobs_per_minute": 10}


async def set_rate_limit(
    pool: asyncpg.Pool, app_id: int, max_queue_depth: int, max_jobs_per_minute: int
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO app_rate_limits (app_id, max_queue_depth, max_jobs_per_minute)
            VALUES ($1, $2, $3)
            ON CONFLICT (app_id) DO UPDATE SET
                max_queue_depth = EXCLUDED.max_queue_depth,
                max_jobs_per_minute = EXCLUDED.max_jobs_per_minute
            """,
            app_id,
            max_queue_depth,
            max_jobs_per_minute,
        )
