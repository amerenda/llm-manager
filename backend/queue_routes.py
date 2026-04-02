"""Queue API routes for llm-manager."""
import asyncio
import json
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse

import queue_db
from queue_models import (
    QueueJobRequest, QueueBatchRequest, QueueJobResponse,
    QueueBatchResponse, QueueJobResult, QueueBatchStatus,
    QueueOverview, ModelSettingsUpdate, ModelSettings,
)
from scheduler import Scheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/queue", tags=["queue"])
model_router = APIRouter(prefix="/api/models", tags=["models"])


def _get_pool(request: Request):
    return request.app.state.db


def _get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


async def _resolve_app(request: Request, authorization: Optional[str]) -> Optional[int]:
    """Resolve app_id from Bearer token. Returns None if no auth."""

    if not authorization or not authorization.startswith("Bearer "):
        return None
    api_key = authorization[7:]
    pool = _get_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM registered_apps WHERE api_key = $1 AND status = 'active'", api_key)
    if not row:
        raise HTTPException(401, "Invalid API key")
    return row["id"]


async def _check_rate_limit(pool, app_id: int):
    """Check per-app rate limits."""
    if app_id is None:
        return
    limits = await queue_db.get_rate_limit(pool, app_id)
    queued = await queue_db.count_app_queued_jobs(pool, app_id)
    if queued >= limits["max_queue_depth"]:
        raise HTTPException(429, f"Queue depth limit reached ({limits['max_queue_depth']})")
    recent = await queue_db.count_app_recent_jobs(pool, app_id)
    if recent >= limits["max_jobs_per_minute"]:
        raise HTTPException(429, f"Rate limit reached ({limits['max_jobs_per_minute']}/min)")


# ── Submit jobs ───────────────────────────────────────────────────────────────

@router.post("/submit", response_model=QueueJobResponse)
async def submit_job(body: QueueJobRequest, request: Request,
                     authorization: Optional[str] = Header(None)):
    import os
    if os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes"):
        raise HTTPException(503, "Queue submissions disabled — scheduler is not running")
    pool = _get_pool(request)
    scheduler = _get_scheduler(request)
    app_id = await _resolve_app(request, authorization)
    await _check_rate_limit(pool, app_id)

    # Pre-check VRAM
    check = await scheduler.check_submission(body.model)
    if not check["ok"]:
        raise HTTPException(422, check)

    job_id = str(uuid.uuid4())[:12]
    await queue_db.insert_job(
        pool, job_id, None, app_id, body.model,
        body.model_dump(exclude={"model", "metadata"}),
        body.metadata,
    )

    # Count position in queue
    pending = await queue_db.get_pending_jobs(pool)
    position = next((i for i, j in enumerate(pending) if j["id"] == job_id), len(pending))

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


@router.post("/submit-batch", response_model=QueueBatchResponse)
async def submit_batch(body: QueueBatchRequest, request: Request,
                       authorization: Optional[str] = Header(None)):
    import os
    if os.environ.get("DISABLE_SCHEDULER", "").lower() in ("true", "1", "yes"):
        raise HTTPException(503, "Queue submissions disabled — scheduler is not running")
    pool = _get_pool(request)
    scheduler = _get_scheduler(request)
    app_id = await _resolve_app(request, authorization)
    await _check_rate_limit(pool, app_id)

    batch_id = f"batch_{str(uuid.uuid4())[:8]}"
    jobs = []

    for job_req in body.jobs:
        # Pre-check VRAM for each unique model
        check = await scheduler.check_submission(job_req.model)
        if not check["ok"]:
            raise HTTPException(422, {
                "error": check["error"],
                "message": f"Model {job_req.model}: {check['message']}",
                "batch_id": batch_id,
            })

        job_id = str(uuid.uuid4())[:12]
        await queue_db.insert_job(
            pool, job_id, batch_id, app_id, job_req.model,
            job_req.model_dump(exclude={"model", "metadata"}),
            job_req.metadata,
        )
        resp = QueueJobResponse(job_id=job_id, status="queued", model=job_req.model)
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
    meta = job.get("metadata")
    if isinstance(meta, str):
        meta = json.loads(meta)
    return QueueJobResult(
        job_id=job["id"],
        status=job["status"],
        model=job["model"],
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"].isoformat() if job.get("created_at") else None,
        started_at=job["started_at"].isoformat() if job.get("started_at") else None,
        completed_at=job["completed_at"].isoformat() if job.get("completed_at") else None,
        metadata=meta,
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
                result=j.get("result"),
                error=j.get("error"),
                created_at=j["created_at"].isoformat() if j.get("created_at") else None,
                started_at=j["started_at"].isoformat() if j.get("started_at") else None,
                completed_at=j["completed_at"].isoformat() if j.get("completed_at") else None,
                metadata=json.loads(j["metadata"]) if isinstance(j.get("metadata"), str) else j.get("metadata"),
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
                event["result"] = job.get("result")
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

    pending = await queue_db.get_pending_jobs(pool)
    models_queued = list(set(j["model"] for j in pending))

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

    # Get current running job from DB instead of scheduler state
    current_job = scheduler.current_job_id
    if not current_job:
        running_jobs = await queue_db.get_running_jobs(pool, limit=1)
        if running_jobs:
            current_job = running_jobs[0]["id"]

    # GPU info from scheduler if available, otherwise estimate from Ollama
    gpu_info = await scheduler._get_gpu_info()

    return QueueOverview(
        queue_depth=len(pending),
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


@model_router.get("/{model_name}/settings", response_model=ModelSettings)
async def get_model_settings(model_name: str, request: Request):
    pool = _get_pool(request)
    settings = await queue_db.get_model_settings(pool, model_name)
    return ModelSettings(**settings)


@model_router.patch("/{model_name}/settings", response_model=ModelSettings)
async def update_model_settings(model_name: str, body: ModelSettingsUpdate, request: Request):
    pool = _get_pool(request)
    updates = body.model_dump(exclude_none=True)
    if updates:
        await queue_db.upsert_model_settings(pool, model_name, **updates)
    settings = await queue_db.get_model_settings(pool, model_name)
    return ModelSettings(**settings)
