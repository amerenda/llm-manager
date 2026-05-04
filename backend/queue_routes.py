"""Queue API routes for llm-manager."""
import asyncio
import json
import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import queue_db
import db
import queue_policy
from queue_models import (
    QueueJobRequest, QueueBatchRequest, QueueJobResponse,
    QueueBatchResponse, QueueJobResult, QueueBatchStatus,
    QueueOverview, ModelSettingsUpdate, ModelSettings,
    ModelAlias, ModelAliasCreate, ModelRunnerParams, ModelRunnerParamsUpsert,
)
from scheduler_v2 import SimplifiedScheduler as Scheduler

logger = logging.getLogger(__name__)


def _parse_jsonb(val):
    """Parse JSONB value that asyncpg may return as a string."""
    if isinstance(val, str):
        return json.loads(val)
    return val

router = APIRouter(prefix="/api/queue", tags=["queue"])
model_router = APIRouter(prefix="/api/models", tags=["models"])
alias_router = APIRouter(prefix="/api/model-aliases", tags=["aliases"])


def _get_pool(request: Request):
    return request.app.state.db


def _get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


async def _resolve_app(request: Request, authorization: Optional[str]) -> Optional[int]:
    """Resolve app_id from Bearer token. Returns None if no auth.
    Also touches last_seen so the app shows as online in the UI."""

    if not authorization or not authorization.startswith("Bearer "):
        return None
    api_key = authorization[7:]
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM registered_apps WHERE api_key = $1 AND status = 'active'", api_key)
    if not row:
        raise HTTPException(401, "Invalid API key")
    # Touch last_seen — app is online if it's making API calls
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE registered_apps SET last_seen = NOW() WHERE id = $1", row["id"])
    return row["id"]


async def _resolve_queue_job_alias(pool, body: QueueJobRequest) -> QueueJobRequest:
    """Resolve model alias to base model and merge alias parameters (parity with /v1/chat/completions)."""
    row = await queue_db.get_model_alias(pool, body.model)
    if not row:
        return body
    fields_set = body.model_fields_set
    data = body.model_dump()
    ap = row.get("parameters") or {}
    if isinstance(ap, str):
        ap = json.loads(ap)
    for k, v in ap.items():
        if k in QueueJobRequest.model_fields and k != "model" and k not in fields_set:
            data[k] = v
    sp = row.get("system_prompt")
    if sp:
        msgs = list(data.get("messages") or [])
        if not any(m.get("role") == "system" for m in msgs):
            msgs.insert(0, {"role": "system", "content": sp})
            data["messages"] = msgs
    data["model"] = row["base_model"]
    return QueueJobRequest(**data)


def _enriched_job_metadata(
    metadata: Optional[dict], allowed_runner_ids: Optional[list[int]]
) -> Optional[dict]:
    """Merge app runner affinity into queue metadata so the scheduler and
    batching strategy match registered_apps.allowed_runner_ids."""
    out: dict = {}
    if metadata:
        out.update(metadata)
    if allowed_runner_ids:
        out["allowed_runner_ids"] = list(allowed_runner_ids)
    return out if out else None


async def _check_rate_limit(pool, app_id: Optional[int]):
    """Check per-app rate limits (thin wrapper for tests that patch this path)."""
    await queue_policy.check_queue_rate_limit(pool, app_id)


# ── Submit jobs ───────────────────────────────────────────────────────────────

@router.post("/submit", response_model=QueueJobResponse)
async def submit_job(body: QueueJobRequest, request: Request,
                     authorization: Optional[str] = Header(None)):
    if os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes"):
        raise HTTPException(503, "Queue submissions disabled — scheduler is not running")
    pool = _get_pool(request)
    scheduler = _get_scheduler(request)
    try:
        app_id = await _resolve_app(request, authorization)
        allowed_runner_ids = None
        if app_id is not None:
            app_row = await db.get_app_by_id(pool, app_id)
            allowed_runner_ids = list((app_row or {}).get("allowed_runner_ids") or [])

        label_model = body.model
        body = await _resolve_queue_job_alias(pool, body)
        await queue_policy.ensure_category_access(
            pool, app_id, body.model, batch_model_label=label_model,
        )

        # Pre-check VRAM
        check = await scheduler.check_submission(body.model, allowed_runner_ids=allowed_runner_ids)
        if not check["ok"]:
            raise HTTPException(422, check)
        await _check_rate_limit(pool, app_id)

        job_id = str(uuid.uuid4())[:12]
        priority = await queue_policy.priority_for_app(pool, app_id)
        meta = _enriched_job_metadata(
            body.metadata,
            allowed_runner_ids if allowed_runner_ids else None,
        )
        await queue_db.insert_job(
            pool, job_id, None, app_id, body.model,
            body.model_dump(exclude={"model", "metadata"}),
            meta,
            priority=priority,
        )

        position = await queue_db.get_job_queue_position(pool, job_id)
        if position is None:
            position = 0

        resp = QueueJobResponse(
            job_id=job_id,
            status="queued",
            model=body.model,
            position=position,
        )
        if check.get("warning"):
            resp.warning = check["message"]
            resp.evicting = check.get("evicting")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("queue submit failed model=%s", getattr(body, "model", ""))
        raise HTTPException(500, detail=f"queue submit failed: {type(e).__name__}: {e}") from e


@router.post("/submit-batch", response_model=QueueBatchResponse)
async def submit_batch(body: QueueBatchRequest, request: Request,
                       authorization: Optional[str] = Header(None)):
    if os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes"):
        raise HTTPException(503, "Queue submissions disabled — scheduler is not running")
    pool = _get_pool(request)
    scheduler = _get_scheduler(request)
    app_id = await _resolve_app(request, authorization)
    allowed_runner_ids = None
    if app_id is not None:
        app_row = await db.get_app_by_id(pool, app_id)
        allowed_runner_ids = list((app_row or {}).get("allowed_runner_ids") or [])

    batch_id = f"batch_{str(uuid.uuid4())[:8]}"
    jobs = []

    # Resolve priority once per batch
    priority = await queue_policy.priority_for_app(pool, app_id)

    check: dict = {"ok": True}
    resolved: list[QueueJobRequest] = []
    for job_req in body.jobs:
        label_model = job_req.model
        job_eff = await _resolve_queue_job_alias(pool, job_req)
        resolved.append(job_eff)
        await queue_policy.ensure_category_access(
            pool, app_id, job_eff.model, batch_model_label=label_model,
        )

        # Pre-check VRAM for each job (resolved base model)
        check = await scheduler.check_submission(job_eff.model, allowed_runner_ids=allowed_runner_ids)
        if not check["ok"]:
            raise HTTPException(422, {
                "error": check["error"],
                "message": f"Model {label_model}: {check['message']}",
                "batch_id": batch_id,
            })
    await _check_rate_limit(pool, app_id)

    for job_eff in resolved:
        job_id = str(uuid.uuid4())[:12]
        meta = _enriched_job_metadata(
            job_eff.metadata,
            allowed_runner_ids if allowed_runner_ids else None,
        )
        await queue_db.insert_job(
            pool, job_id, batch_id, app_id, job_eff.model,
            job_eff.model_dump(exclude={"model", "metadata"}),
            meta,
            priority=priority,
        )
        resp = QueueJobResponse(job_id=job_id, status="queued", model=job_eff.model)
        if check.get("warning"):
            resp.warning = check["message"]
            resp.evicting = check.get("evicting")
        jobs.append(resp)

    return QueueBatchResponse(batch_id=batch_id, jobs=jobs)


# ── Job status ────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=QueueJobResult)
async def get_job(job_id: str, request: Request):
    pool = _get_pool(request)
    job = await queue_db.get_job(pool, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return QueueJobResult(
        job_id=job["id"],
        status=job["status"],
        model=job["model"],
        result=_parse_jsonb(job.get("result")),
        error=job.get("error"),
        retried=job.get("retried") or 0,
        created_at=job["created_at"].isoformat() if job.get("created_at") else None,
        started_at=job["started_at"].isoformat() if job.get("started_at") else None,
        completed_at=job["completed_at"].isoformat() if job.get("completed_at") else None,
        metadata=_parse_jsonb(job.get("metadata")),
    )


@router.get("/batches/{batch_id}", response_model=QueueBatchStatus)
async def get_batch(batch_id: str, request: Request):
    pool = _get_pool(request)
    jobs = await queue_db.get_batch_jobs(pool, batch_id)
    if not jobs:
        raise HTTPException(404, "Batch not found")

    statuses = [j["status"] for j in jobs]
    return QueueBatchStatus(
        batch_id=batch_id,
        total=len(jobs),
        completed=statuses.count("completed"),
        failed=statuses.count("failed"),
        running=statuses.count("running"),
        queued=statuses.count("queued") + statuses.count("waiting_for_eviction"),
        jobs=[
            QueueJobResult(
                job_id=j["id"],
                status=j["status"],
                model=j["model"],
                result=_parse_jsonb(j.get("result")),
                error=j.get("error"),
                created_at=j["created_at"].isoformat() if j.get("created_at") else None,
                started_at=j["started_at"].isoformat() if j.get("started_at") else None,
                completed_at=j["completed_at"].isoformat() if j.get("completed_at") else None,
                metadata=_parse_jsonb(j.get("metadata")),
            ) for j in jobs
        ],
    )


# ── Cancel ────────────────────────────────────────────────────────────────────

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    pool = _get_pool(request)
    job = await queue_db.get_job(pool, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(400, f"Job already {job['status']}")
    await queue_db.update_job_status(pool, job_id, "cancelled")
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


# ── Queue overview ────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/wait")
async def wait_for_job(job_id: str, request: Request):
    """SSE stream that waits for a job to complete. Streams status updates."""
    pool = _get_pool(request)

    async def event_stream():
        while True:
            job = await queue_db.get_job(pool, job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            status = job["status"]
            event = {
                "job_id": job_id,
                "status": status,
                "model": job["model"],
            }
            if status == "completed":
                event["result"] = _parse_jsonb(job.get("result"))
                yield f"data: {json.dumps(event)}\n\n"
                break
            elif status in ("failed", "cancelled"):
                event["error"] = job.get("error")
                yield f"data: {json.dumps(event)}\n\n"
                break
            else:
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/batches/{batch_id}/wait")
async def wait_for_batch(batch_id: str, request: Request):
    """SSE stream that waits for all batch jobs to complete."""
    pool = _get_pool(request)

    async def event_stream():
        while True:
            jobs = await queue_db.get_batch_jobs(pool, batch_id)
            if not jobs:
                yield f"data: {json.dumps({'error': 'Batch not found'})}\n\n"
                break
            statuses = [j["status"] for j in jobs]
            event = {
                "batch_id": batch_id,
                "total": len(jobs),
                "completed": statuses.count("completed"),
                "failed": statuses.count("failed"),
                "running": statuses.count("running"),
                "queued": statuses.count("queued") + statuses.count("waiting_for_eviction"),
            }
            all_done = all(s in ("completed", "failed", "cancelled") for s in statuses)
            if all_done:
                event["jobs"] = [
                    {"job_id": j["id"], "status": j["status"],
                     "result": j.get("result"), "error": j.get("error")}
                    for j in jobs
                ]
                yield f"data: {json.dumps(event)}\n\n"
                break
            else:
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.patch("/jobs/{job_id}/priority")
async def set_job_priority(job_id: str, body: dict, request: Request):
    """Set the priority of a queued job. Higher priority = processed first."""
    pool = _get_pool(request)
    job = await queue_db.get_job(pool, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("queued", "waiting_for_eviction"):
        raise HTTPException(400, f"Cannot reprioritize job in '{job['status']}' state")
    priority = body.get("priority", 0)
    await queue_db.update_job_priority(pool, job_id, priority)
    return {"ok": True, "job_id": job_id, "priority": priority}


@router.get("/jobs")
async def list_queue_jobs(request: Request, status: Optional[str] = None, limit: int = 100):
    """List jobs in the queue with optional status filter."""
    pool = _get_pool(request)
    jobs = await queue_db.list_jobs(pool, status=status, limit=limit)
    return jobs


@router.get("/metrics")
async def queue_metrics(request: Request):
    """Queue performance metrics for the dashboard."""
    pool = _get_pool(request)
    metrics = await queue_db.get_queue_metrics(pool)
    return metrics


@router.get("/history")
async def queue_history(request: Request, limit: int = 50):
    """Get recently completed/failed jobs."""
    pool = _get_pool(request)
    jobs = await queue_db.list_recent_jobs(pool, limit=limit)
    return jobs


@router.get("/status", response_model=QueueOverview)
async def queue_status(request: Request):
    pool = _get_pool(request)
    scheduler = _get_scheduler(request)

    queue_depth = await queue_db.count_pending_jobs(pool)
    models_queued = await queue_db.pending_queued_models(pool)

    # Get loaded models from scheduler cache or runner agent
    models_loaded = list(scheduler.loaded_models.keys())
    if not models_loaded:
        try:
            client = await scheduler.get_runner_client()
            status = await client.status()
            for m in status.get("loaded_ollama_models", []):
                models_loaded.append(m.get("name", ""))
        except Exception:
            pass

    # Prefer Postgres — correct on every replica; scheduler memory only on lock holder
    current_job = None
    running_jobs = await queue_db.get_running_jobs(pool, limit=1)
    if running_jobs:
        current_job = running_jobs[0]["id"]
    if not current_job:
        current_job = scheduler.current_job_id

    # GPU info from scheduler if available, otherwise estimate from Ollama
    gpu_info = await scheduler._get_gpu_info()

    return QueueOverview(
        queue_depth=queue_depth,
        models_queued=models_queued,
        models_loaded=models_loaded,
        current_job=current_job,
        gpu_vram_total_gb=gpu_info["total"],
        gpu_vram_used_gb=gpu_info["used"],
        gpu_vram_free_gb=gpu_info["free"],
    )


# ── Model settings ────────────────────────────────────────────────────────────

@model_router.get("/settings", response_model=list[ModelSettings])
async def list_model_settings(request: Request):
    pool = _get_pool(request)
    rows = await queue_db.get_all_model_settings(pool)
    return [ModelSettings(**r) for r in rows]


# Community model names contain slashes (e.g. MFDoom/deepseek-r1-tool-calling:
# 14b-qwen-distill-q4_K_M). Traefik / ingress decodes %2F in the URL back to a
# literal '/' before FastAPI routes, so a default {model_name} matcher only
# captures the first segment and the rest fails /settings. :path tells Starlette
# to match everything up to the trailing /settings and URL-decode it.
@model_router.get("/{model_name:path}/settings", response_model=ModelSettings)
async def get_model_settings(model_name: str, request: Request):
    pool = _get_pool(request)
    settings = await queue_db.get_model_settings(pool, model_name)
    return ModelSettings(**settings)


@model_router.patch("/{model_name:path}/settings", response_model=ModelSettings)
async def update_model_settings(model_name: str, body: ModelSettingsUpdate, request: Request):
    pool = _get_pool(request)
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if updates:
        await queue_db.upsert_model_settings(pool, model_name, **updates)
    settings = await queue_db.get_model_settings(pool, model_name)
    return ModelSettings(**settings)


# ── Runner params (per model per runner) ──────────────────────────────────────

@model_router.get("/{model_name:path}/runner-params", response_model=list[ModelRunnerParams])
async def list_runner_params(model_name: str, request: Request):
    pool = _get_pool(request)
    rows = await queue_db.get_all_runner_params_for_model(pool, model_name)
    return [ModelRunnerParams(
        model_name=r["model_name"],
        runner_id=r["runner_id"],
        hostname=r.get("hostname"),
        system_prompt=r.get("system_prompt"),
        parameters=r.get("parameters") or {},
    ) for r in rows]


@model_router.put("/{model_name:path}/runner-params/{runner_id}", response_model=ModelRunnerParams)
async def upsert_runner_params(
    model_name: str, runner_id: int, body: ModelRunnerParamsUpsert, request: Request
):
    pool = _get_pool(request)
    do_not_evict = body.do_not_evict if body.do_not_evict is not None else False
    row = await queue_db.upsert_model_runner_params(
        pool, model_name, runner_id, body.system_prompt, body.parameters, do_not_evict)
    # Re-fetch with hostname
    rows = await queue_db.get_all_runner_params_for_model(pool, model_name)
    for r in rows:
        if r["runner_id"] == runner_id:
            return ModelRunnerParams(
                model_name=r["model_name"],
                runner_id=r["runner_id"],
                hostname=r.get("hostname"),
                system_prompt=r.get("system_prompt"),
                parameters=r.get("parameters") or {},
                do_not_evict=bool(r.get("do_not_evict", False)),
            )
    return ModelRunnerParams(
        model_name=row["model_name"], runner_id=row["runner_id"],
        system_prompt=row.get("system_prompt"), parameters=row.get("parameters") or {},
        do_not_evict=bool(row.get("do_not_evict", False)),
    )


class PinRequest(BaseModel):
    do_not_evict: bool


@model_router.patch("/{model_name:path}/runner-params/{runner_id}/pin")
async def pin_model_on_runner(model_name: str, runner_id: int, body: PinRequest, request: Request):
    """Set do_not_evict for a specific model/runner without touching system_prompt/parameters."""
    pool = _get_pool(request)
    await queue_db.set_runner_model_pin(pool, model_name, runner_id, body.do_not_evict)
    return {"ok": True, "model_name": model_name, "runner_id": runner_id, "do_not_evict": body.do_not_evict}


@model_router.delete("/{model_name:path}/runner-params/{runner_id}")
async def delete_runner_params(model_name: str, runner_id: int, request: Request):
    pool = _get_pool(request)
    deleted = await queue_db.delete_model_runner_params(pool, model_name, runner_id)
    if not deleted:
        raise HTTPException(404, "Runner params not found")
    return {"ok": True}


# ── Model aliases ─────────────────────────────────────────────────────────────

@alias_router.get("", response_model=list[ModelAlias])
async def list_aliases(request: Request):
    pool = _get_pool(request)
    rows = await queue_db.get_all_model_aliases(pool)
    return [ModelAlias(**{k: v for k, v in r.items() if k in ModelAlias.model_fields}) for r in rows]


@alias_router.get("/{alias_name:path}", response_model=ModelAlias)
async def get_alias(alias_name: str, request: Request):
    pool = _get_pool(request)
    row = await queue_db.get_model_alias(pool, alias_name)
    if not row:
        raise HTTPException(404, "Alias not found")
    return ModelAlias(**{k: v for k, v in row.items() if k in ModelAlias.model_fields})


@alias_router.put("/{alias_name:path}", response_model=ModelAlias)
async def upsert_alias(alias_name: str, body: ModelAliasCreate, request: Request):
    pool = _get_pool(request)
    row = await queue_db.upsert_model_alias(
        pool, alias_name, body.base_model,
        body.system_prompt, body.parameters, body.description,
    )
    return ModelAlias(**{k: v for k, v in row.items() if k in ModelAlias.model_fields})


@alias_router.delete("/{alias_name:path}")
async def delete_alias(alias_name: str, request: Request):
    pool = _get_pool(request)
    deleted = await queue_db.delete_model_alias(pool, alias_name)
    if not deleted:
        raise HTTPException(404, "Alias not found")
    return {"ok": True}


@model_router.get("/{model_name:path}/aliases", response_model=list[ModelAlias])
async def list_aliases_for_model(model_name: str, request: Request):
    pool = _get_pool(request)
    rows = await queue_db.get_aliases_for_base_model(pool, model_name)
    return [ModelAlias(**{k: v for k, v in r.items() if k in ModelAlias.model_fields}) for r in rows]
