"""
Simplified queue scheduler: one model per GPU, strict FIFO.

Each runner tracks at most one current_model. Jobs are processed oldest-first
from the queue (within priority). If the head job's model matches the current
model on an idle runner, it runs there. Otherwise the scheduler swaps that
runner to the needed model (unload + load) and then runs it. No VRAM
accounting, no eviction math, no mid-transition snapshots.

Deliberately dumb v1 — see plans/queue-one-model-per-gpu.md for the rationale
and the future-work list.

Public surface matches the historical Scheduler name in main.py and
queue_routes.py (imported as ``Scheduler``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import asyncpg
import httpx
from prometheus_client import Counter, Gauge, Histogram

import queue_db
from gpu import vram_for_model
from cloud_providers import detect_provider, ModelProvider, anthropic_chat
from queue_strategies import make_strategy, QueueStrategy

logger = logging.getLogger(__name__)


SCHEDULER_LOCK_ID = 900001  # Must match main.py


def _queued_age_seconds(created) -> Optional[float]:
    """Wall-clock seconds since ``created_at`` (UTC), or None if unknown."""
    if created is None or not isinstance(created, datetime):
        return None
    c = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - c).total_seconds()


# ── Metrics ──────────────────────────────────────────────────────────────────
# v2-specific Prometheus metrics (swap, wait, completion counters).

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

# Per-runner live state gauges (updated every _refresh_runners tick)
runner_vram_used_bytes = Gauge(
    "llm_runner_vram_used_bytes",
    "VRAM currently used on runner (bytes)",
    ["runner"],
)
runner_vram_total_bytes = Gauge(
    "llm_runner_vram_total_bytes",
    "Total VRAM capacity of runner (bytes)",
    ["runner"],
)
runner_is_idle = Gauge(
    "llm_runner_is_idle",
    "1 if runner has no in-flight job, 0 if busy",
    ["runner"],
)

# Inference token accounting (from Ollama response usage field)
inference_prompt_tokens_total = Counter(
    "llm_inference_prompt_tokens_total",
    "Prompt tokens processed",
    ["model", "runner"],
)
inference_completion_tokens_total = Counter(
    "llm_inference_completion_tokens_total",
    "Completion tokens generated",
    ["model", "runner"],
)

# Fast-path (direct proxy, bypassing the job queue)
fastpath_requests_total = Counter(
    "llm_fastpath_requests_total",
    "Requests served via direct-proxy fast path (no queue)",
    ["model", "status"],  # status: completed|failed
)
fastpath_duration_seconds = Histogram(
    "llm_fastpath_duration_seconds",
    "End-to-end wall time of fast-path requests",
    ["model"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)


def _dl_set_from_caps(caps: dict) -> set[str]:
    """Extract the set of downloaded model names from a runner's capabilities
    dict. Accepts the new-style list-of-dicts shape and the legacy list-of-
    strings shape as a safety net."""
    out: set[str] = set()
    for entry in (caps.get("downloaded_models") or []):
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = entry
        if isinstance(name, str) and name:
            out.add(name)
    return out


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
    # When in_flight_job_id was last set (monotonic not required; used for stale heal).
    in_flight_since: Optional[float] = None
    # Draining: admin has asked the runner to stop accepting new work. The
    # current in-flight job (if any) finishes normally; new jobs skip this
    # runner. The flag persists in llm_runners.draining; it clears only when
    # an admin explicitly un-drains.
    draining: bool = False
    # Names (base + tag, e.g. "qwen3:14b") of models currently on disk. Sourced
    # from agent heartbeat capabilities.downloaded_models. Used by _pick_runner
    # so we don't pick a runner for a model it doesn't have — that would force
    # a doomed swap and then a fallback that hammered the drained runner (seen
    # 2026-04-22: archlinux drained + only-copy of qwen3.6:35b → every job
    # fallback-loaded on it, making it unresponsive).
    downloaded_models: set[str] = field(default_factory=set)
    # Ollama cooldown: set when Ollama returns 503 on this runner. _pick_runner
    # skips the runner until the timestamp expires, giving Ollama time to recover.
    ollama_down_until: Optional[float] = None

    @property
    def is_idle(self) -> bool:
        return self.in_flight_job_id is None

    @property
    def is_ollama_down(self) -> bool:
        return self.ollama_down_until is not None and time.time() < self.ollama_down_until

    def has_downloaded(self, model: str) -> bool:
        """Model-tag downloaded. Also matches :latest tag as an alias for the
        base name — keeps parity with the UI's `downloadedNames.has(name) ||
        has(name:latest)` pattern."""
        if model in self.downloaded_models:
            return True
        if ":" not in model and f"{model}:latest" in self.downloaded_models:
            return True
        return False


# ── Scheduler ────────────────────────────────────────────────────────────────

class SimplifiedScheduler:
    """Drop-in replacement for Scheduler with one-model-per-GPU semantics."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        get_runner_client: callable,
        lock_conn: Optional[asyncpg.Connection] = None,
        strategy: Optional[QueueStrategy] = None,
        lock_session_app_name: Optional[str] = None,
        on_lock_verify_failed: Optional[Callable[[str], None]] = None,
    ):
        self.pool = pool
        self.get_runner_client = get_runner_client
        self.lock_conn = lock_conn
        self._lock_session_app_name = lock_session_app_name
        self._on_lock_verify_failed = on_lock_verify_failed
        self.strategy: QueueStrategy = strategy or make_strategy()

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._current_job_id: Optional[str] = None
        self._runners: dict[int, RunnerState] = {}  # keyed by runner_id
        self._last_idle_model_sync_mono: float = 0.0
        self._last_no_runner_log_mono: float = 0.0
        # Monotonic time at the start of the current / last `_loop` iteration (watchdog liveness).
        self._last_loop_tick_mono: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._task is not None and not self._task.done():
            logger.warning(
                "SimplifiedScheduler start() skipped: dispatch loop still running",
            )
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "SimplifiedScheduler started (one-model-per-GPU, strategy=%s)",
            self.strategy.name,
        )

    def stop(self):
        """Request stop; the asyncio task may still be winding down. Prefer
        :meth:`stop_and_wait` when you must start again in the same process
        (K8s/postgres leader handoff)."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("SimplifiedScheduler stopped")

    async def stop_and_wait(self) -> None:
        """Cancel the dispatch loop and wait until it has exited."""
        self._running = False
        t = self._task
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("SimplifiedScheduler stopped (loop finished)")

    def abandon_dispatch_task(self) -> None:
        """Clear dispatch bookkeeping without joining the loop task.

        Used when :meth:`stop_and_wait` hits a timeout (watchdog / shutdown): the
        old ``_loop`` task may be stuck in an await that does not yield to cancel
        promptly. We cancel it and drop the handle so :meth:`start` can spawn a
        new loop. If the old task later wakes, it sees ``_running`` False and exits.
        """
        self._running = False
        t = self._task
        self._task = None
        if t is not None and not t.done():
            t.cancel()
        logger.warning(
            "SimplifiedScheduler: dispatch task handle cleared without join "
            "(stuck loop; cancelled best-effort)"
        )

    async def bounded_stop_and_wait(self, *, context: str) -> None:
        """Like :meth:`stop_and_wait` but never blocks past ``SCHEDULER_WATCHDOG_STOP_TIMEOUT_SEC``.

        If the dispatch task ignores cancel inside a long await, :meth:`abandon_dispatch_task`
        clears bookkeeping so a new loop can be started (watchdog / shutdown).
        """
        lim = float(os.environ.get("SCHEDULER_WATCHDOG_STOP_TIMEOUT_SEC", "120"))
        if lim <= 0:
            await self.stop_and_wait()
            return
        try:
            await asyncio.wait_for(self.stop_and_wait(), timeout=lim)
        except asyncio.TimeoutError:
            logger.critical(
                "%s: stop_and_wait exceeded %.0fs — abandoning stuck dispatch task",
                context,
                lim,
            )
            self.abandon_dispatch_task()

    def bind_lock_session(
        self,
        conn: Optional[asyncpg.Connection],
        session_app_name: Optional[str],
    ) -> None:
        """Attach the Postgres session used for the scheduler advisory lock.

        ``session_app_name`` must match ``server_settings['application_name']``
        on that connection so a dead client can be identified server-side for
        zombie lock cleanup (see main._retry_lock)."""
        self.lock_conn = conn
        self._lock_session_app_name = session_app_name

    # ── public compat surface (used by queue_routes.py) ──────────────────────

    @property
    def current_job_id(self) -> Optional[str]:
        return self._current_job_id

    @property
    def is_dispatch_loop_running(self) -> bool:
        """True while the asyncio task for :meth:`_loop` exists and has not finished."""
        t = self._task
        return t is not None and not t.done()

    @property
    def dispatch_loop_tick_stale_seconds(self) -> float:
        """Seconds since :meth:`_loop` last began an iteration (top of ``while`` body).

        Returns ``0`` when no dispatch task is running so the worker watchdog can
        treat "task dead" vs "task alive but wedged" separately.
        """
        t = self._task
        if t is None or t.done():
            return 0.0
        tick = self._last_loop_tick_mono
        if tick <= 0:
            return 0.0
        return max(0.0, time.monotonic() - tick)

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
            # Simple liveness check — if the connection is alive, the session-level
            # advisory lock is still held. Don't call pg_try_advisory_lock here:
            # it increments the lock nesting count on every call (advisory locks are
            # reentrant within a session), bloating it to thousands after hours of
            # operation. When the connection eventually dies, PostgreSQL releases all
            # nested acquisitions at once and the lock becomes free — meaning the
            # standby pod can acquire it.
            await self.lock_conn.fetchval("SELECT 1")
            return True
        except Exception:
            logger.warning("Failed to verify scheduler lock — connection lost")
            fn = self._on_lock_verify_failed
            name = self._lock_session_app_name
            if fn and name:
                try:
                    fn(name)
                except Exception:
                    logger.exception("on_lock_verify_failed callback raised")
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
            rs.downloaded_models = _dl_set_from_caps(caps)

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
                # Seed live VRAM gauges from status (caps don't carry used_bytes)
                runner_vram_total_bytes.labels(runner=r["hostname"]).set(
                    (status.get("gpu_vram_total_gb") or 0) * 1e9
                )
                runner_vram_used_bytes.labels(runner=r["hostname"]).set(
                    (status.get("gpu_vram_used_gb") or 0) * 1e9
                )
                runner_is_idle.labels(runner=r["hostname"]).set(1)
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
                rs.downloaded_models = _dl_set_from_caps(caps)
                self._runners[r["id"]] = rs
            else:
                # Update mutable admin-controlled flags (changed via API).
                # current_model and in_flight_job_id are scheduler-owned —
                # don't touch those.
                rs.pinned_model = r.get("pinned_model")
                rs.draining = bool(r.get("draining"))
                # Downloaded set refreshes every heartbeat — the agent pulls
                # models in the background and we need _pick_runner to see it
                # as soon as the heartbeat lands.
                rs.downloaded_models = _dl_set_from_caps(caps)

            # Update static gauges from capabilities (applies to new and existing runners).
            # vram_used is NOT in capabilities — it's live state updated by
            # _reconcile_runners and _swap_model via client.status().
            vram_total = caps.get("gpu_vram_total_bytes", 0) or 0
            runner_vram_total_bytes.labels(runner=rs.hostname).set(vram_total)
            runner_is_idle.labels(runner=rs.hostname).set(1 if rs.is_idle else 0)

    async def _sync_idle_loaded_models_from_agents(self) -> None:
        """Reconcile ``current_model`` with live Ollama on *idle* runners.

        Admin/UI unload or host-side changes otherwise leave scheduler state
        thinking a model is still loaded, skipping the swap path."""
        interval = float(os.environ.get("SCHEDULER_LIVE_MODEL_SYNC_SEC", "30"))
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_idle_model_sync_mono < interval:
            return
        self._last_idle_model_sync_mono = now
        for rs in self._runners.values():
            if not rs.is_idle:
                continue
            try:
                client = await self.get_runner_client(runner_id=rs.runner_id)
                status = await client.status()
            except Exception:
                logger.debug("live model sync: status failed for %s", rs.hostname, exc_info=True)
                continue
            loaded = status.get("loaded_ollama_models") or []
            live = (loaded[0].get("name") or None) if loaded else None
            if live != rs.current_model:
                logger.info(
                    "scheduler: idle runner %s loaded model %r → %r (agent status)",
                    rs.hostname, rs.current_model, live,
                )
                rs.current_model = live
                rs.model_loaded_at = time.time() if live else None
            if status.get("gpu_vram_total_gb"):
                rs.gpu_total_gb = status.get("gpu_vram_total_gb", rs.gpu_total_gb)
            try:
                runner_vram_used_bytes.labels(runner=rs.hostname).set(
                    (status.get("gpu_vram_used_gb") or 0) * 1e9
                )
            except Exception:
                pass

    def _any_eligible_idle_runner(
        self, allowed_runner_ids: list[int] | None
    ) -> bool:
        """True if some runner could accept work (idle, not draining/down, allowed)."""
        for r in self._runners.values():
            if r.draining or r.is_ollama_down:
                continue
            if allowed_runner_ids and r.runner_id not in allowed_runner_ids:
                continue
            if r.is_idle:
                return True
        return False

    def _waiting_on_pinned_busy_runner(self, model: str) -> bool:
        """True when the model is pinned to a runner that is currently busy."""
        for r in self._runners.values():
            if r.pinned_model != model:
                continue
            if r.draining or r.is_ollama_down:
                continue
            if not r.is_idle:
                return True
        return False

    def _model_has_exclusive_pin(self, model: str) -> bool:
        """True if any runner is pinned to this model (work must go there only).

        When the pinned runner is idle but the blob is still pulling, or the
        runner is draining, _pick_runner returns None — not the same as fleet
        unplaceable; do not fail-fast those head jobs."""
        return any(r.pinned_model == model for r in self._runners.values())

    def _log_pick_runner_miss_throttled(
        self,
        model: str,
        head_job_id: str,
        allowed_runner_ids: list[int] | None,
        has_idle: bool,
        pinned_wait: bool,
        exclusive_pin: bool,
    ) -> None:
        now = time.monotonic()
        if now - self._last_no_runner_log_mono < 30:
            return
        self._last_no_runner_log_mono = now
        summary = [
            (rs.hostname, rs.is_idle, rs.draining, rs.current_model, rs.pinned_model)
            for rs in sorted(self._runners.values(), key=lambda x: x.runner_id)
        ]
        logger.info(
            "scheduler: no runner for model=%r head_job=%s allowed_runners=%s "
            "eligible_idle=%s pinned_busy_wait=%s exclusive_pin=%s runners=%s",
            model,
            head_job_id,
            allowed_runner_ids,
            has_idle,
            pinned_wait,
            exclusive_pin,
            summary,
        )

    async def _heal_stale_in_flight_claims(self) -> None:
        """Drop bogus busy flags when Postgres shows no active work for that job.

        If a background dispatch dies without clearing `in_flight_job_id`, the
        runner stays non-idle forever and the queue head blocks (often for all
        apps behind it) until pod restart. See 2026-05-01 incident."""
        queued_grace = float(os.environ.get("SCHEDULER_STALE_QUEUED_GRACE_SEC", "45"))
        fast_path_max = float(os.environ.get("SCHEDULER_FAST_PATH_STALE_SEC", "300"))
        now = time.time()
        for rs in self._runners.values():
            jid = rs.in_flight_job_id
            if jid is None:
                continue
            since = rs.in_flight_since or now
            if jid == "__fast_path__":
                if now - since > fast_path_max:
                    logger.warning(
                        "Healing stale fast-path claim on %s (held %.0fs > %.0fs)",
                        rs.hostname, now - since, fast_path_max,
                    )
                    rs.in_flight_job_id = None
                    rs.in_flight_since = None
                    runner_is_idle.labels(runner=rs.hostname).set(1)
                continue
            try:
                job = await queue_db.get_job(self.pool, jid)
            except Exception:
                logger.debug("stale heal: get_job failed for %s", jid, exc_info=True)
                continue
            st = (job or {}).get("status")
            if job is None or st in ("failed", "completed", "cancelled"):
                logger.warning(
                    "Healing stale in_flight on %s (job=%s status=%s)",
                    rs.hostname, jid, st,
                )
                rs.in_flight_job_id = None
                rs.in_flight_since = None
                runner_is_idle.labels(runner=rs.hostname).set(1)
            elif st == "queued" and (now - since) > queued_grace:
                logger.warning(
                    "Healing stale in_flight on %s (job=%s still queued after %.0fs)",
                    rs.hostname, jid, now - since,
                )
                rs.in_flight_job_id = None
                rs.in_flight_since = None
                runner_is_idle.labels(runner=rs.hostname).set(1)

    # ── main loop ────────────────────────────────────────────────────────────

    async def _loop(self):
        """Main scheduler loop. Picks batches via the strategy and hands each
        one off to a background task so the loop never blocks on inference.

        Without backgrounding, one busy runner serialized work across the whole
        fleet — queued jobs for different models sat waiting until the current
        runner's chat call returned, even when other runners were idle. Each
        batch runs sequentially on its assigned runner (Ollama is serial
        anyway); parallelism is only across runners."""
        self._last_loop_tick_mono = time.monotonic()
        await self._reconcile_runners()
        idle_counter = 0
        active_tasks: set[asyncio.Task] = set()
        consecutive_errors = 0
        last_stale_heal_mono = 0.0
        stale_heal_interval = float(os.environ.get("SCHEDULER_STALE_HEAL_INTERVAL_SEC", "15"))
        head_unplaceable_since: float | None = None
        head_unplaceable_key: str | None = None
        while self._running:
            self._last_loop_tick_mono = time.monotonic()
            try:
                # Reap completed dispatch tasks so the set doesn't grow unbounded
                active_tasks = {t for t in active_tasks if not t.done()}

                if not await self._verify_lock():
                    logger.warning("Lost scheduler advisory lock — stopping")
                    self._running = False
                    break

                await self._refresh_runners()
                await self._sync_idle_loaded_models_from_agents()

                now_mono = time.monotonic()
                if now_mono - last_stale_heal_mono >= stale_heal_interval:
                    await self._heal_stale_in_flight_claims()
                    last_stale_heal_mono = now_mono

                batch = await self.strategy.next_jobs(self.pool)
                if not batch:
                    head_unplaceable_key = None
                    head_unplaceable_since = None
                    await asyncio.sleep(1)
                    idle_counter += 1
                    if idle_counter > 3600:  # hourly cleanup
                        await queue_db.cleanup_old_jobs(self.pool)
                        idle_counter = 0
                    continue
                idle_counter = 0

                # get_pending_jobs omits request to avoid loading 100 huge JSON
                # bodies every tick; load only what this batch will execute.
                _req_map = await queue_db.fetch_pending_job_requests(
                    self.pool, [j["id"] for j in batch]
                )
                for _job in batch:
                    _job["request"] = _req_map[_job["id"]]

                head = batch[0]
                model = head["model"]

                max_q = float(os.environ.get("SCHEDULER_FAIL_QUEUED_OLDER_THAN_SEC", "0"))
                if max_q > 0:
                    age_sec = _queued_age_seconds(head.get("created_at"))
                    if age_sec is not None and age_sec > max_q:
                        err = (
                            f"Exceeded SCHEDULER_FAIL_QUEUED_OLDER_THAN_SEC={max_q:.0f}s "
                            f"(queued ~{age_sec / 3600:.2f}h). Common causes: every eligible GPU "
                            f"busy with other work, allowed_runner_ids excludes idle hosts, "
                            f"runners offline/draining/Ollama down, or model/vram rules block "
                            f"placement. Model={model!r} job={head['id']!r}."
                        )
                        logger.error("scheduler: %s", err)
                        for job in batch:
                            await queue_db.update_job_status(
                                self.pool, job["id"], "failed", error=err,
                            )
                            scheduler_jobs_completed_total.labels(
                                model=job["model"], status="failed"
                            ).inc()
                        await asyncio.sleep(0)
                        continue

                provider = detect_provider(model)

                if provider != ModelProvider.LOCAL:
                    # Cloud jobs don't need a runner — dispatch each as its own
                    # background task so they run in parallel.
                    head_unplaceable_key = None
                    head_unplaceable_since = None
                    for job in batch:
                        t = asyncio.create_task(self._run_job(job, runner=None))
                        active_tasks.add(t)
                    continue

                # Extract per-app runner restriction from job metadata
                _meta = head.get("metadata") or {}
                if isinstance(_meta, str):
                    try:
                        _meta = json.loads(_meta)
                    except Exception:
                        _meta = {}
                allowed_runner_ids: list[int] | None = _meta.get("allowed_runner_ids") or None
                if allowed_runner_ids:
                    logger.debug("job %s: restricting to runners %s", head["id"], allowed_runner_ids)

                runner = await self._pick_runner(
                    model, allowed_runner_ids=allowed_runner_ids
                )
                if runner is None:
                    await self._heal_stale_in_flight_claims()
                    fail_sec = float(
                        os.environ.get("SCHEDULER_UNPLACEABLE_FAIL_SEC", "180")
                    )
                    has_idle = self._any_eligible_idle_runner(allowed_runner_ids)
                    pinned_wait = self._waiting_on_pinned_busy_runner(model)
                    exclusive_pin = self._model_has_exclusive_pin(model)
                    self._log_pick_runner_miss_throttled(
                        model,
                        head["id"],
                        allowed_runner_ids,
                        has_idle,
                        pinned_wait,
                        exclusive_pin,
                    )
                    if (
                        fail_sec <= 0
                        or not has_idle
                        or pinned_wait
                        or exclusive_pin
                    ):
                        head_unplaceable_key = None
                        head_unplaceable_since = None
                    else:
                        key = head["id"]
                        if head_unplaceable_key != key:
                            head_unplaceable_key = key
                            head_unplaceable_since = time.monotonic()
                        elif head_unplaceable_since is not None:
                            if time.monotonic() - head_unplaceable_since >= fail_sec:
                                err = (
                                    f"No runner could be selected for model {model!r} "
                                    f"after {fail_sec:.0f}s while other runners were idle "
                                    f"(model may be missing on disk, VRAM estimate too large, "
                                    f"or placement rules block all idle GPUs). "
                                    f"Head job {head['id']!r}."
                                )
                                logger.error("scheduler: %s", err)
                                for job in batch:
                                    await queue_db.update_job_status(
                                        self.pool, job["id"], "failed", error=err,
                                    )
                                    scheduler_jobs_completed_total.labels(
                                        model=job["model"], status="failed"
                                    ).inc()
                                head_unplaceable_key = None
                                head_unplaceable_since = None
                                await asyncio.sleep(0)
                                continue
                    await asyncio.sleep(2)
                    continue

                head_unplaceable_key = None
                head_unplaceable_since = None

                # Eagerly mark the runner busy with the first job in the batch
                # so the very next loop iteration's _pick_runner skips it.
                runner.in_flight_job_id = head["id"]
                runner.in_flight_since = time.time()

                # If we're about to swap, mark the jobs loading_model now so
                # they don't stay "queued" while the task handles the swap.
                swap_needed = runner.current_model != model
                if swap_needed:
                    for job in batch:
                        await queue_db.update_job_status(
                            self.pool, job["id"], "loading_model",
                            runner_id=runner.runner_id,
                        )

                task = asyncio.create_task(
                    self._dispatch_batch(runner, batch, swap_needed)
                )
                active_tasks.add(task)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")
                consecutive_errors += 1
                # If the loop keeps failing (e.g. pool connection reset that
                # asyncpg can't auto-recover from), release the advisory lock
                # explicitly so the standby pod can take over immediately —
                # no pod restart required. lock_conn is a dedicated connection
                # separate from the pool; it's usually still alive when the pool
                # breaks, so the unlock should succeed.
                if consecutive_errors >= 10:
                    logger.critical(
                        "Scheduler loop: %d consecutive errors — releasing advisory "
                        "lock so standby pod can take over",
                        consecutive_errors,
                    )
                    if self.lock_conn:
                        try:
                            await self.lock_conn.execute(
                                "SELECT pg_advisory_unlock($1)", SCHEDULER_LOCK_ID
                            )
                        except Exception:
                            logger.warning(
                                "Failed to release advisory lock explicitly — "
                                "standby pod will acquire it on next pod restart"
                            )
                        self.lock_conn = None
                    self._running = False
                    break
                await asyncio.sleep(5)
            else:
                consecutive_errors = 0

    async def _dispatch_batch(
        self,
        runner: RunnerState,
        batch: list[dict],
        swap_needed: bool,
    ) -> None:
        """Run a batch of jobs sequentially on a specific runner. Swaps the
        model first if needed. Runs as a background task so the scheduler loop
        can continue picking up work for other runners.

        Ownership contract: runner.in_flight_job_id was set to batch[0]['id']
        by the caller before we started. This function is responsible for
        clearing it (via _run_job's finally) and for marking any unrun jobs
        failed if the swap breaks."""
        model = batch[0]["model"]
        original_runner = runner
        try:
            if swap_needed:
                landed = await self._swap_model(runner, model)
                if landed is None:
                    for job in batch:
                        await queue_db.update_job_status(
                            self.pool, job["id"], "failed",
                            error=f"Could not load {model} on {runner.hostname}",
                        )
                        scheduler_jobs_completed_total.labels(model=model, status="failed").inc()
                    return
                # Swap's live-probe fallback may have put the model on a
                # different runner than we picked. Transfer ownership so the
                # rest of this dispatch (and the DB/UI) refer to the actual
                # runner. Without this, _run_job would call chat() against the
                # original (wrong) runner and every job in the batch would
                # fail with "model not found".
                if landed.runner_id != runner.runner_id:
                    logger.warning(
                        "dispatch: swap redirected %s from %s to %s — "
                        "retargeting batch",
                        model, runner.hostname, landed.hostname,
                    )
                    # Release the originally-picked runner (we never used it)
                    original_runner.in_flight_job_id = None
                    original_runner.in_flight_since = None
                    landed.in_flight_job_id = batch[0]["id"]
                    landed.in_flight_since = time.time()
                    runner = landed
                    for job in batch:
                        await queue_db.update_job_status(
                            self.pool, job["id"], "loading_model",
                            runner_id=runner.runner_id,
                        )

            if len(batch) > 1:
                logger.info("Running batch of %d jobs for %s on %s",
                            len(batch), model, runner.hostname)
            for job in batch:
                if not self._running:
                    break
                await self._run_job(job, runner=runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dispatch_batch failed on %s for %s", runner.hostname, model)
        finally:
            # _run_job clears in_flight_job_id in its own finally, but if the
            # swap failed we never entered _run_job — clear it here defensively.
            if runner.in_flight_job_id is not None:
                runner.in_flight_job_id = None
                runner.in_flight_since = None

    # ── routing ──────────────────────────────────────────────────────────────

    async def _pick_runner(
        self, model: str, allowed_runner_ids: list[int] | None = None
    ) -> Optional[RunnerState]:
        """Return a runner to use for `model`, or None if none is immediately usable.

        Policy:
          1. Pinned runner for this model → use if idle; if busy, return None
             (don't fall through — a pinned runner's jobs must go to it).
             Pinned + draining: don't route new work here — admin intent wins.
          2. Any non-pinned, non-draining idle runner already on this model.
          3. Any non-pinned, non-draining idle runner that HAS THE MODEL
             DOWNLOADED and can fit it (swap on that runner).

        allowed_runner_ids restricts which runners are eligible (from app config).
        None means all runners are eligible.
        """
        def _eligible(r: RunnerState) -> bool:
            return not allowed_runner_ids or r.runner_id in allowed_runner_ids

        def _skip(r: RunnerState) -> bool:
            return r.draining or r.is_ollama_down or not _eligible(r)

        # 1. Pinned match (skip draining/down pinned runners — wait for drain to clear)
        # If the model is already loaded, trust live scheduler state even if
        # downloaded_models from heartbeat is stale/missing that tag.
        pinned = [r for r in self._runners.values() if r.pinned_model == model]
        if pinned:
            for r in pinned:
                if _skip(r):
                    continue
                if r.is_idle and (r.current_model == model or r.has_downloaded(model)):
                    return r
            return None  # pinned but busy / draining / down / not eligible — wait

        # 2. Already-loaded + idle
        for r in self._runners.values():
            if r.pinned_model is not None or _skip(r):
                continue
            if r.current_model == model and r.is_idle:
                return r

        # 3. Idle + has it downloaded + fits (VRAM-wise).
        #
        # Some agents omit gpu_vram_total_gb in /v1/status while still reporting
        # models via heartbeat; gpu_total_gb stays 0 and we used to skip here
        # forever — jobs sat queued with idle GPUs and the model on disk.
        need = await queue_db.resolved_vram_gb_for_model(self.pool, model)
        swap_candidates: list[RunnerState] = []
        for r in self._runners.values():
            if r.pinned_model is not None or _skip(r):
                continue
            if not r.is_idle:
                continue
            if not r.has_downloaded(model):
                # Non-empty heartbeat inventory that omits this model → runner
                # truly does not have the blob; picking it would force a doomed swap.
                if r.downloaded_models:
                    continue
                # Empty inventory: agent has not yet reported /api/tags (restart
                # race, transient Ollama error). _swap_model live-lists models
                # before load — better than stalling the queue forever.
            cap = r.gpu_total_gb
            if cap > 0:
                if need <= cap:
                    return r
            else:
                swap_candidates.append(r)
        # Unknown VRAM: still attempt swap/load — Ollama refuses if it truly
        # cannot fit; better than an infinite queue when status is incomplete.
        if swap_candidates:
            r = min(swap_candidates, key=lambda x: x.runner_id)
            logger.info(
                "scheduler: picking %s for model=%r with unknown VRAM "
                "(need=%.1fGiB per settings/heuristic)",
                r.hostname,
                model,
                need,
            )
            return r

        return None

    # ── swap ─────────────────────────────────────────────────────────────────

    async def _swap_model(self, runner: RunnerState, new_model: str) -> Optional[RunnerState]:
        """Unload runner.current_model (if any) then load new_model. Returns
        the RunnerState the model ended up on (usually == runner), or None on
        failure. Normally same as the input; the live-probe fallback below
        can return a different runner if capabilities drifted."""
        t0 = time.time()
        old = runner.current_model
        try:
            client = await self.get_runner_client(runner_id=runner.runner_id)
        except Exception as e:
            logger.error("swap: can't get client for %s: %s", runner.hostname, e)
            return None

        if old:
            try:
                logger.info("swap: unloading %s on %s", old, runner.hostname)
                await client.unload_model(old)
            except Exception:
                logger.exception("swap: unload %s on %s failed — proceeding anyway",
                                 old, runner.hostname)
            runner.current_model = None
            runner.model_loaded_at = None

        # Live-probe the runner — capabilities may be a heartbeat behind. If
        # the model vanished from the picked runner (admin deleted it between
        # pick and swap), fall back to another non-draining runner that has it.
        try:
            models_resp = await client.models()
            model_list = models_resp.get("data", models_resp.get("models", []))
            names = [m.get("id", m.get("name", "")) for m in model_list]
            if new_model not in names:
                logger.error("swap: model %s vanished on %s; looking for another runner",
                             new_model, runner.hostname)
                other = await self._find_runner_with_model_downloaded(
                    new_model, exclude_id=runner.runner_id,
                )
                if other is None:
                    return None
                runner = other
                try:
                    client = await self.get_runner_client(runner_id=runner.runner_id)
                except Exception:
                    return None
        except Exception:
            logger.exception("swap: couldn't list models on %s", runner.hostname)
            return None

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
            # Refresh VRAM used gauge — VRAM state changed significantly after swap
            try:
                post_status = await client.status()
                runner_vram_used_bytes.labels(runner=runner.hostname).set(
                    (post_status.get("gpu_vram_used_gb") or 0) * 1e9
                )
            except Exception:
                pass
            return runner
        except Exception:
            logger.exception("swap: load %s on %s failed", new_model, runner.hostname)
            return None

    async def _find_runner_with_model_downloaded(
        self, model: str, exclude_id: Optional[int] = None,
    ) -> Optional[RunnerState]:
        """Fallback: probe other runners to see who has the model on disk.
        Skips draining runners (admin intent wins) and non-idle ones."""
        for rs in self._runners.values():
            if exclude_id is not None and rs.runner_id == exclude_id:
                continue
            if rs.draining or not rs.is_idle:
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

        await queue_db.update_job_status(
            self.pool, job_id, "running",
            runner_id=runner.runner_id if runner is not None else None,
        )
        if runner is not None:
            runner.in_flight_job_id = job_id

        OLLAMA_COOLDOWN = 300  # seconds to skip a runner after a 503
        MAX_RETRIES = 3       # permanent failure after this many re-queues

        try:
            if runner is None:
                await self._run_cloud(job_id, model, request)
            else:
                await self._run_local(job_id, model, request, runner)
            scheduler_jobs_completed_total.labels(model=model, status="completed").inc()
        except Exception as e:
            is_5xx = isinstance(e, httpx.HTTPStatusError) and 500 <= e.response.status_code < 600
            is_503 = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 503

            if isinstance(e, httpx.HTTPStatusError):
                error_msg = f"{e}: {e.response.text}"
            else:
                error_msg = str(e) or repr(e)  # str() is empty for some exceptions (e.g. ReadTimeout)

            if is_5xx and runner is not None:
                # Apply 503-specific runner cooldown (Ollama service down)
                if is_503:
                    runner.ollama_down_until = time.time() + OLLAMA_COOLDOWN
                    # Fetch Ollama logs for the error record
                    try:
                        client = await self.get_runner_client(runner_id=runner.runner_id)
                        logs = await client.logs(tail=30, service="ollama")
                        log_lines = "\n".join(logs.get("ollama_logs") or [])
                        if log_lines:
                            error_msg += f"\n\nOllama logs ({runner.hostname}):\n{log_lines}"
                    except Exception:
                        pass

                retry_count = await queue_db.increment_job_retry(self.pool, job_id)
                if retry_count > MAX_RETRIES:
                    logger.warning(
                        "Job %s exhausted %d retries on %s (HTTP %d), failing permanently",
                        job_id, MAX_RETRIES, runner.hostname, e.response.status_code,
                    )
                    await queue_db.update_job_status(self.pool, job_id, "failed", error=error_msg)
                    scheduler_jobs_completed_total.labels(model=model, status="failed").inc()
                else:
                    logger.warning(
                        "Runner %s returned HTTP %d — re-queuing job %s (retry %d/%d)",
                        runner.hostname, e.response.status_code, job_id, retry_count, MAX_RETRIES,
                    )
                    await queue_db.update_job_status(self.pool, job_id, "queued", error=error_msg)
                    scheduler_jobs_completed_total.labels(model=model, status="retried").inc()
            else:
                await queue_db.update_job_status(self.pool, job_id, "failed", error=error_msg)
                scheduler_jobs_completed_total.labels(model=model, status="failed").inc()
                logger.exception("Job %s failed", job_id)
        finally:
            self._current_job_id = None
            if runner is not None:
                runner.in_flight_job_id = None
                runner.in_flight_since = None

    async def _run_cloud(self, job_id: str, model: str, request: dict):
        body = {"model": model, "messages": request.get("messages", []), "stream": False}
        if "temperature" in request: body["temperature"] = request["temperature"]
        if "max_tokens" in request: body["max_tokens"] = request["max_tokens"]
        if request.get("tools"): body["tools"] = request["tools"]
        result = await anthropic_chat(body, stream=False)
        await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
        logger.info("Job %s completed (model=%s, cloud)", job_id, model)

    async def _apply_runner_params(
        self, request: dict, model: str, runner: RunnerState
    ) -> tuple[list, dict]:
        """Return (messages, kwargs) with runner-level param overrides applied.
        Runner params take precedence over alias/request params (physical constraints)."""
        messages = list(request.get("messages", []))
        kwargs: dict = {}
        for key in ("temperature", "max_tokens", "tools", "top_p", "top_k",
                    "frequency_penalty", "presence_penalty", "stop",
                    "seed", "repeat_penalty", "num_ctx", "num_predict"):
            if key in request:
                kwargs[key] = request[key]
        try:
            rp = await queue_db.get_model_runner_params(self.pool, model, runner.runner_id)
            if rp:
                params = rp.get("parameters") or {}
                if isinstance(params, str):
                    params = json.loads(params)
                kwargs.update(params)
                sys_prompt = rp.get("system_prompt")
                if sys_prompt:
                    messages = [m for m in messages if m.get("role") != "system"]
                    messages.insert(0, {"role": "system", "content": sys_prompt})
        except Exception:
            logger.debug("Failed to fetch runner params for %s on runner %d", model, runner.runner_id)
        return messages, kwargs

    def _record_token_usage(self, result: dict, model: str, runner_hostname: str) -> None:
        """Extract token counts from an Ollama response and update counters."""
        try:
            usage = result.get("usage") or {}
            pt = usage.get("prompt_tokens") or usage.get("prompt_eval_count", 0)
            ct = usage.get("completion_tokens") or usage.get("eval_count", 0)
            if pt:
                inference_prompt_tokens_total.labels(model=model, runner=runner_hostname).inc(pt)
            if ct:
                inference_completion_tokens_total.labels(model=model, runner=runner_hostname).inc(ct)
        except Exception:
            pass

    async def _run_local(self, job_id: str, model: str, request: dict, runner: RunnerState):
        client = await self.get_runner_client(runner_id=runner.runner_id)
        messages, kwargs = await self._apply_runner_params(request, model, runner)

        logger.info("Job %s starting on %s (model=%s, msgs=%d)",
                    job_id, runner.hostname, model, len(messages))
        t0 = time.time()
        result = await client.chat(messages=messages, model=model, stream=False, **kwargs)
        elapsed = time.time() - t0
        self._record_token_usage(result, model, runner.hostname)
        await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
        logger.info("Job %s done (model=%s, %.1fs, runner=%s)",
                    job_id, model, elapsed, runner.hostname)

    # ── fast-path (direct proxy, no queue) ───────────────────────────────────

    def try_claim_for_fast_path(
        self, model: str, allowed_runner_ids: list[int] | None = None
    ) -> Optional[RunnerState]:
        """Atomically claim an idle runner that already has `model` loaded.
        Returns the RunnerState with in_flight_job_id set, or None if no
        suitable runner is available (caller should fall back to the queue).

        Mirrors _pick_runner steps 1+2 (already-loaded only — no swap)."""
        # Step 1: pinned runner for this model. If one exists but is busy,
        # return None — same as _pick_runner; don't fall through to other runners.
        pinned = [r for r in self._runners.values() if r.pinned_model == model]
        if pinned:
            for r in pinned:
                if r.draining or not r.is_idle or r.current_model != model:
                    continue
                if allowed_runner_ids and r.runner_id not in allowed_runner_ids:
                    continue
                r.in_flight_job_id = "__fast_path__"
                r.in_flight_since = time.time()
                runner_is_idle.labels(runner=r.hostname).set(0)
                logger.debug("fast-path: claimed pinned runner %s for %s", r.hostname, model)
                return r
            return None  # pinned runner exists but not ready — don't use others

        # Step 2: any non-pinned, non-draining idle runner with model loaded
        for r in self._runners.values():
            if r.pinned_model is not None or r.draining:
                continue
            if r.current_model != model or not r.is_idle:
                continue
            if allowed_runner_ids and r.runner_id not in allowed_runner_ids:
                continue
            r.in_flight_job_id = "__fast_path__"
            r.in_flight_since = time.time()
            runner_is_idle.labels(runner=r.hostname).set(0)
            logger.debug("fast-path: claimed runner %s for %s", r.hostname, model)
            return r
        return None

    def release_fast_path_claim(self, runner: RunnerState) -> None:
        if runner.in_flight_job_id == "__fast_path__":
            runner.in_flight_job_id = None
            runner.in_flight_since = None
            runner_is_idle.labels(runner=runner.hostname).set(1)
            logger.debug("fast-path: released runner %s", runner.hostname)

    # ── submission pre-check ─────────────────────────────────────────────────

    async def check_submission(
        self, model: str, allowed_runner_ids: list[int] | None = None
    ) -> dict:
        """Trivial: accept if the model fits on any runner's total VRAM.

        Reads gpu_total from the scheduler's RunnerState cache (if populated)
        or from the DB runner capabilities (for replicas that aren't running
        the scheduler). No live runner polling — that's the whole point.
        """
        if detect_provider(model) != ModelProvider.LOCAL:
            return {"ok": True, "provider": detect_provider(model).value}

        need = await queue_db.resolved_vram_gb_for_model(self.pool, model)

        # Prefer in-memory state if we have it
        candidate_runners = list(self._runners.values())
        if allowed_runner_ids:
            allowed = set(allowed_runner_ids)
            candidate_runners = [rs for rs in candidate_runners if rs.runner_id in allowed]
            if not candidate_runners and self._runners:
                return {
                    "ok": False,
                    "error": "no_schedulable_runners",
                    "message": "No schedulable runners available for this app.",
                }

        totals = [rs.gpu_total_gb for rs in candidate_runners if rs.gpu_total_gb > 0]

        if not totals:
            # Fall back to DB (the replica without the scheduler won't have _runners populated)
            import db as _db
            rows = await _db.get_active_runners(self.pool)
            if not rows:
                return {"ok": True}  # no runners visible yet — defer
            if allowed_runner_ids:
                allowed = set(allowed_runner_ids)
                rows = [r for r in rows if r.get("id") in allowed]
                if not rows:
                    return {
                        "ok": False,
                        "error": "no_schedulable_runners",
                        "message": "No schedulable runners available for this app.",
                    }
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
