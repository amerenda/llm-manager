"""
Simplified queue scheduler: one model per GPU, strict FIFO.

Each runner tracks at most one current_model. Jobs are processed oldest-first
from the queue (within priority). If the head job's model matches the current
model on an idle runner, it runs there. Otherwise the scheduler swaps that
runner to the needed model (unload + load) and then runs it. No VRAM
accounting, no eviction math, no mid-transition snapshots.

Deliberately dumb v1 — see plans/queue-one-model-per-gpu.md for the rationale
and the future-work list.

Public surface matches the old Scheduler so main.py and queue_routes.py can
swap between them via the SIMPLIFIED_SCHEDULER env flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import asyncpg
from prometheus_client import Counter, Histogram

import queue_db
from gpu import vram_for_model
from cloud_providers import detect_provider, ModelProvider, anthropic_chat
from queue_strategies import make_strategy, QueueStrategy

logger = logging.getLogger(__name__)


SCHEDULER_LOCK_ID = 900001  # Must match main.py


# ── Metrics ──────────────────────────────────────────────────────────────────
# v2-specific. Old scheduler metrics (loaded_models gauge, eviction counter,
# etc.) are kept live by the old scheduler when the flag is off; when the flag
# flips on they simply stop incrementing.

scheduler_model_swap_total = Counter(
    "llm_scheduler_v2_model_swap_total",
    "Model swaps on a runner (from_model -> to_model)",
    ["runner", "from_model", "to_model"],
)
scheduler_model_swap_seconds = Histogram(
    "llm_scheduler_v2_model_swap_seconds",
    "Time spent swapping a runner's model (unload+load)",
    ["runner"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)
scheduler_job_wait_seconds = Histogram(
    "llm_scheduler_v2_job_wait_seconds",
    "Time a job spent queued before starting",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
scheduler_job_wait_by_app_seconds = Histogram(
    "llm_scheduler_v2_job_wait_by_app_seconds",
    "Time a job spent queued before starting, broken down by app",
    ["app"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
scheduler_jobs_completed_total = Counter(
    "llm_scheduler_v2_jobs_completed_total",
    "Jobs completed",
    ["model", "status"],  # status: completed|failed
)
scheduler_runner_current_model_v2 = None  # populated only after start; see init


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class RunnerState:
    runner_id: int
    hostname: str
    gpu_total_gb: float = 0.0
    current_model: Optional[str] = None
    pinned_model: Optional[str] = None
    model_loaded_at: Optional[float] = None
    in_flight_job_id: Optional[str] = None
    # Draining: admin has asked the runner to stop accepting new work. The
    # current in-flight job (if any) finishes normally; new jobs skip this
    # runner. The flag persists in llm_runners.draining; it clears only when
    # an admin explicitly un-drains.
    draining: bool = False

    @property
    def is_idle(self) -> bool:
        return self.in_flight_job_id is None


# ── Scheduler ────────────────────────────────────────────────────────────────

class SimplifiedScheduler:
    """Drop-in replacement for Scheduler with one-model-per-GPU semantics."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        get_runner_client: callable,
        lock_conn: Optional[asyncpg.Connection] = None,
        strategy: Optional[QueueStrategy] = None,
    ):
        self.pool = pool
        self.get_runner_client = get_runner_client
        self.lock_conn = lock_conn
        self.strategy: QueueStrategy = strategy or make_strategy()

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._current_job_id: Optional[str] = None
        self._runners: dict[int, RunnerState] = {}  # keyed by runner_id

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._loop())
            logger.info(
                "SimplifiedScheduler started (one-model-per-GPU, strategy=%s)",
                self.strategy.name,
            )

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("SimplifiedScheduler stopped")

    # ── public compat surface (used by queue_routes.py) ──────────────────────

    @property
    def current_job_id(self) -> Optional[str]:
        return self._current_job_id

    @property
    def loaded_models(self) -> dict[str, dict]:
        """Shape-compat with old Scheduler. Each runner contributes at most one entry."""
        out: dict[str, dict] = {}
        for r in self._runners.values():
            if r.current_model:
                out[r.current_model] = {
                    "runner_id": r.runner_id,
                    "loaded_at": r.model_loaded_at or 0,
                    "vram_gb": vram_for_model(r.current_model),
                    "runner_hostname": r.hostname,
                }
        return out

    async def _get_gpu_info(self, runner_id: Optional[int] = None) -> dict:
        """Compat helper for queue_routes.QueueOverview. Returns first runner's
        VRAM snapshot (or a specific runner if runner_id is given)."""
        import db as _db
        try:
            rows = await _db.get_active_runners(self.pool)
            if runner_id:
                row = next((r for r in rows if r["id"] == runner_id), None)
            else:
                row = rows[0] if rows else None
            if not row:
                return {"total": 0, "used": 0, "free": 0}
            client = await self.get_runner_client(runner_id=row["id"])
            status = await client.status()
            total = status.get("gpu_vram_total_gb", 0)
            used = status.get("gpu_vram_used_gb", 0)
            return {
                "total": round(total, 1),
                "used": round(used, 1),
                "free": round(max(0.0, total - used), 1),
            }
        except Exception:
            return {"total": 0, "used": 0, "free": 0}

    # ── lock verification ────────────────────────────────────────────────────

    async def _verify_lock(self) -> bool:
        if not self.lock_conn:
            return True  # single-replica or no lock configured
        try:
            held = await self.lock_conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", SCHEDULER_LOCK_ID
            )
            return bool(held)
        except Exception:
            logger.warning("Failed to verify scheduler lock")
            return False

    # ── runner state management ──────────────────────────────────────────────

    async def _reconcile_runners(self):
        """One-shot: poll each active runner and seed RunnerState. Runners we
        can't reach are added with unknown GPU total — they'll get filled in
        later by _refresh_runners once they come back."""
        import db as _db
        rows = await _db.get_active_runners(self.pool)
        new_state: dict[int, RunnerState] = {}
        for r in rows:
            rs = RunnerState(
                runner_id=r["id"],
                hostname=r["hostname"],
                pinned_model=r.get("pinned_model"),
                draining=bool(r.get("draining")),
            )
            # Seed gpu_total_gb from capabilities (populated by agent heartbeat)
            caps = r.get("capabilities") or {}
            if isinstance(caps, str):
                try:
                    caps = json.loads(caps)
                except Exception:
                    caps = {}
            total_bytes = caps.get("gpu_vram_total_bytes", 0)
            if total_bytes:
                rs.gpu_total_gb = round(total_bytes / 1e9, 2)

            # Ask the runner what it currently has loaded (best effort)
            try:
                client = await self.get_runner_client(runner_id=r["id"])
                status = await client.status()
                if status.get("gpu_vram_total_gb"):
                    rs.gpu_total_gb = status.get("gpu_vram_total_gb")
                loaded = status.get("loaded_ollama_models") or []
                if loaded:
                    rs.current_model = loaded[0].get("name") or None
                    if rs.current_model:
                        rs.model_loaded_at = time.time()
            except Exception:
                logger.warning(
                    "reconcile: runner %s unreachable — gpu_total_gb=%s from capabilities",
                    r["hostname"], rs.gpu_total_gb,
                )

            new_state[r["id"]] = rs

        self._runners = new_state
        summary = {rs.hostname: {"gpu_gb": rs.gpu_total_gb, "model": rs.current_model, "pin": rs.pinned_model}
                   for rs in new_state.values()}
        logger.info("Runners reconciled: %s", summary)

    async def _refresh_runners(self):
        """Add newly-registered runners, drop gone ones, update pinned_model.
        Does NOT change current_model / in_flight state for existing runners."""
        import db as _db
        rows = await _db.get_active_runners(self.pool)
        active_ids = {r["id"] for r in rows}
        # Drop runners that are no longer active
        self._runners = {k: v for k, v in self._runners.items() if k in active_ids}
        # Add / update
        for r in rows:
            rs = self._runners.get(r["id"])
            caps = r.get("capabilities") or {}
            if isinstance(caps, str):
                try:
                    caps = json.loads(caps)
                except Exception:
                    caps = {}
            if rs is None:
                rs = RunnerState(
                    runner_id=r["id"],
                    hostname=r["hostname"],
                    pinned_model=r.get("pinned_model"),
                    draining=bool(r.get("draining")),
                )
                # Try to poll for initial state
                try:
                    client = await self.get_runner_client(runner_id=r["id"])
                    status = await client.status()
                    rs.gpu_total_gb = status.get("gpu_vram_total_gb", 0.0)
                    loaded = status.get("loaded_ollama_models") or []
                    if loaded:
                        rs.current_model = loaded[0].get("name") or None
                        if rs.current_model:
                            rs.model_loaded_at = time.time()
                except Exception:
                    total_bytes = caps.get("gpu_vram_total_bytes", 0)
                    if total_bytes:
                        rs.gpu_total_gb = round(total_bytes / 1e9, 2)
                self._runners[r["id"]] = rs
            else:
                # Update mutable admin-controlled flags (changed via API).
                # current_model and in_flight_job_id are scheduler-owned —
                # don't touch those.
                rs.pinned_model = r.get("pinned_model")
                rs.draining = bool(r.get("draining"))

    # ── main loop ────────────────────────────────────────────────────────────

    async def _loop(self):
        """Main scheduler loop. Dispatches batches chosen by the strategy."""
        await self._reconcile_runners()
        idle_counter = 0
        while self._running:
            try:
                if not await self._verify_lock():
                    logger.warning("Lost scheduler advisory lock — stopping")
                    self._running = False
                    break

                await self._refresh_runners()

                batch = await self.strategy.next_jobs(self.pool)
                if not batch:
                    await asyncio.sleep(1)
                    idle_counter += 1
                    if idle_counter > 3600:  # hourly cleanup
                        await queue_db.cleanup_old_jobs(self.pool)
                        idle_counter = 0
                    continue
                idle_counter = 0

                head = batch[0]
                model = head["model"]
                provider = detect_provider(model)

                if provider != ModelProvider.LOCAL:
                    # Cloud: batch isn't meaningful — run each inline.
                    for job in batch:
                        if not self._running:
                            break
                        await self._run_job(job, runner=None)
                    continue

                runner = self._pick_runner(model)
                if runner is None:
                    # No runner ready right now — wait briefly before retrying.
                    await asyncio.sleep(2)
                    continue

                # Swap (if needed) then run every job in the batch against the
                # now-loaded runner. Mark every job loading_model so clients
                # see progress during the swap.
                if runner.current_model != model:
                    for job in batch:
                        await queue_db.update_job_status(self.pool, job["id"], "loading_model")
                    ok = await self._swap_model(runner, model)
                    if not ok:
                        for job in batch:
                            await queue_db.update_job_status(
                                self.pool, job["id"], "failed",
                                error=f"Could not load {model} on {runner.hostname}",
                            )
                            scheduler_jobs_completed_total.labels(model=model, status="failed").inc()
                        continue

                if len(batch) > 1:
                    logger.info("Running batch of %d jobs for %s on %s",
                                len(batch), model, runner.hostname)
                for job in batch:
                    if not self._running:
                        break
                    await self._run_job(job, runner=runner)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")
                await asyncio.sleep(5)

    # ── routing ──────────────────────────────────────────────────────────────

    def _pick_runner(self, model: str) -> Optional[RunnerState]:
        """Return a runner to use for `model`, or None if none is immediately usable.

        Policy:
          1. Pinned runner for this model → use if idle; if busy, return None
             (don't fall through — a pinned runner's jobs must go to it).
             Pinned + draining: don't route new work here — admin intent wins.
          2. Any non-pinned, non-draining idle runner already on this model.
          3. Any non-pinned, non-draining idle runner that can fit the model (swap).
          4. None.

        In all cases, a runner marked `draining` is excluded from new work.
        Its current in-flight job (if any) is unaffected; it finishes and the
        runner goes idle with draining=True, still excluded.
        """
        # 1. Pinned match (skip draining pinned runners — wait for drain to clear)
        pinned = [r for r in self._runners.values() if r.pinned_model == model]
        if pinned:
            for r in pinned:
                if r.draining:
                    continue
                if r.is_idle:
                    return r
            return None  # pinned but busy or draining — wait

        # 2. Already-loaded + idle
        for r in self._runners.values():
            if r.pinned_model is not None or r.draining:
                continue
            if r.current_model == model and r.is_idle:
                return r

        # 3. Idle + fits
        need = vram_for_model(model)
        for r in self._runners.values():
            if r.pinned_model is not None or r.draining:
                continue
            if not r.is_idle:
                continue
            if r.gpu_total_gb <= 0:
                continue  # unknown — skip defensively
            if need <= r.gpu_total_gb:
                return r

        return None

    # ── swap ─────────────────────────────────────────────────────────────────

    async def _swap_model(self, runner: RunnerState, new_model: str) -> bool:
        """Unload runner.current_model (if any) then load new_model. Returns True
        on success. Runner state is updated in place."""
        t0 = time.time()
        old = runner.current_model
        try:
            client = await self.get_runner_client(runner_id=runner.runner_id)
        except Exception as e:
            logger.error("swap: can't get client for %s: %s", runner.hostname, e)
            return False

        if old:
            try:
                logger.info("swap: unloading %s on %s", old, runner.hostname)
                await client.unload_model(old)
            except Exception:
                logger.exception("swap: unload %s on %s failed — proceeding anyway",
                                 old, runner.hostname)
            runner.current_model = None
            runner.model_loaded_at = None

        # Verify the model is downloaded on this runner
        try:
            models_resp = await client.models()
            model_list = models_resp.get("data", models_resp.get("models", []))
            names = [m.get("id", m.get("name", "")) for m in model_list]
            if new_model not in names:
                logger.error("swap: model %s not downloaded on %s; falling back",
                             new_model, runner.hostname)
                # Try any other runner that has it downloaded
                other = await self._find_runner_with_model_downloaded(new_model, exclude_id=runner.runner_id)
                if other is None:
                    return False
                # Swap context to that runner
                runner = other
                try:
                    client = await self.get_runner_client(runner_id=runner.runner_id)
                except Exception:
                    return False
        except Exception:
            logger.exception("swap: couldn't list models on %s", runner.hostname)
            return False

        try:
            logger.info("swap: loading %s on %s", new_model, runner.hostname)
            await client.load_model(new_model, keep_alive=-1)
            runner.current_model = new_model
            runner.model_loaded_at = time.time()
            elapsed = time.time() - t0
            scheduler_model_swap_total.labels(
                runner=runner.hostname,
                from_model=old or "none",
                to_model=new_model,
            ).inc()
            scheduler_model_swap_seconds.labels(runner=runner.hostname).observe(elapsed)
            logger.info("swap: %s → %s on %s in %.1fs",
                        old or "none", new_model, runner.hostname, elapsed)
            return True
        except Exception:
            logger.exception("swap: load %s on %s failed", new_model, runner.hostname)
            return False

    async def _find_runner_with_model_downloaded(
        self, model: str, exclude_id: Optional[int] = None,
    ) -> Optional[RunnerState]:
        """Fallback: probe other runners to see who has the model on disk."""
        for rs in self._runners.values():
            if exclude_id is not None and rs.runner_id == exclude_id:
                continue
            if not rs.is_idle:
                continue
            try:
                client = await self.get_runner_client(runner_id=rs.runner_id)
                resp = await client.models()
                model_list = resp.get("data", resp.get("models", []))
                names = [m.get("id", m.get("name", "")) for m in model_list]
                if model in names:
                    return rs
            except Exception:
                continue
        return None

    # ── job execution ────────────────────────────────────────────────────────

    async def _run_job(self, job: dict, runner: Optional[RunnerState]):
        """Execute one job. runner=None means cloud inference."""
        job_id = job["id"]
        model = job["model"]
        request = job["request"] if isinstance(job["request"], dict) else json.loads(job["request"])

        self._current_job_id = job_id

        # Observe queue wait
        created_at = job.get("created_at")
        if created_at is not None:
            try:
                wait = time.time() - created_at.timestamp()
                if wait > 0:
                    scheduler_job_wait_seconds.observe(wait)
                    app_name = (job.get("app_name") or "unknown").lower()
                    scheduler_job_wait_by_app_seconds.labels(app=app_name).observe(wait)
            except Exception:
                pass

        await queue_db.update_job_status(self.pool, job_id, "running")
        if runner is not None:
            runner.in_flight_job_id = job_id

        try:
            if runner is None:
                await self._run_cloud(job_id, model, request)
            else:
                await self._run_local(job_id, model, request, runner)
            scheduler_jobs_completed_total.labels(model=model, status="completed").inc()
        except Exception as e:
            await queue_db.update_job_status(self.pool, job_id, "failed", error=str(e))
            scheduler_jobs_completed_total.labels(model=model, status="failed").inc()
            logger.exception("Job %s failed", job_id)
        finally:
            self._current_job_id = None
            if runner is not None:
                runner.in_flight_job_id = None

    async def _run_cloud(self, job_id: str, model: str, request: dict):
        body = {"model": model, "messages": request.get("messages", []), "stream": False}
        if "temperature" in request: body["temperature"] = request["temperature"]
        if "max_tokens" in request: body["max_tokens"] = request["max_tokens"]
        if request.get("tools"): body["tools"] = request["tools"]
        result = await anthropic_chat(body, stream=False)
        await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
        logger.info("Job %s completed (model=%s, cloud)", job_id, model)

    async def _run_local(self, job_id: str, model: str, request: dict, runner: RunnerState):
        client = await self.get_runner_client(runner_id=runner.runner_id)
        kwargs = {}
        if "temperature" in request: kwargs["temperature"] = request["temperature"]
        if "max_tokens" in request: kwargs["max_tokens"] = request["max_tokens"]
        if request.get("tools"): kwargs["tools"] = request["tools"]
        t0 = time.time()
        result = await client.chat(
            messages=request.get("messages", []),
            model=model, stream=False, **kwargs,
        )
        elapsed = time.time() - t0
        await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
        logger.info("Job %s completed (model=%s, %.1fs on %s)",
                    job_id, model, elapsed, runner.hostname)

    # ── submission pre-check ─────────────────────────────────────────────────

    async def check_submission(self, model: str) -> dict:
        """Trivial: accept if the model fits on any runner's total VRAM.

        Reads gpu_total from the scheduler's RunnerState cache (if populated)
        or from the DB runner capabilities (for replicas that aren't running
        the scheduler). No live runner polling — that's the whole point.
        """
        if detect_provider(model) != ModelProvider.LOCAL:
            return {"ok": True, "provider": detect_provider(model).value}

        need = vram_for_model(model)

        # Prefer in-memory state if we have it
        totals = [rs.gpu_total_gb for rs in self._runners.values() if rs.gpu_total_gb > 0]

        if not totals:
            # Fall back to DB (the replica without the scheduler won't have _runners populated)
            import db as _db
            rows = await _db.get_active_runners(self.pool)
            if not rows:
                return {"ok": True}  # no runners visible yet — defer
            for r in rows:
                caps = r.get("capabilities") or {}
                if isinstance(caps, str):
                    try:
                        caps = json.loads(caps)
                    except Exception:
                        caps = {}
                tb = caps.get("gpu_vram_total_bytes", 0)
                if tb:
                    totals.append(round(tb / 1e9, 2))

        if not totals:
            return {"ok": True}  # still no data — accept, let runtime figure it out

        max_total = max(totals)
        if need > max_total:
            return {
                "ok": False,
                "error": "model_too_large",
                "message": (
                    f"{model} requires {need:.1f}GB VRAM, "
                    f"largest GPU has {max_total:.1f}GB"
                ),
                "vram_required_gb": need,
                "vram_available_gb": max_total,
            }
        return {"ok": True}
