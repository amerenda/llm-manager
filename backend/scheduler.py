"""
Queue scheduler with VRAM-aware model management.

Single async worker that processes jobs grouped by model,
handles model loading/unloading with eviction policies.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import asyncpg
import httpx
from prometheus_client import Counter, Gauge, Histogram

import queue_db
from gpu import vram_for_model
from cloud_providers import detect_provider, ModelProvider, anthropic_chat

logger = logging.getLogger(__name__)

# ── Scheduler Prometheus metrics ─────────────────────────────────────────────

scheduler_model_load_total = Counter(
    "llm_scheduler_model_load_total", "Model load attempts", ["model", "status"])
scheduler_model_load_seconds = Histogram(
    "llm_scheduler_model_load_seconds", "Time to load model into VRAM", ["model"],
    buckets=[5, 10, 30, 60, 120, 300, 600])
scheduler_model_eviction_total = Counter(
    "llm_scheduler_model_eviction_total", "Model evictions", ["model"])
scheduler_job_inference_seconds = Histogram(
    "llm_scheduler_job_inference_seconds", "Inference time (excluding queue wait)", ["model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120])
scheduler_job_tokens_total = Counter(
    "llm_scheduler_job_tokens_total", "Total tokens generated", ["model", "type"])
scheduler_job_errors_total = Counter(
    "llm_scheduler_job_errors_total", "Job failures", ["model", "reason"])
scheduler_loaded_models = Gauge(
    "llm_scheduler_loaded_models", "Number of models currently loaded in VRAM")
scheduler_loaded_vram_gb = Gauge(
    "llm_scheduler_loaded_vram_gb", "Total VRAM used by loaded models")
scheduler_submission_rejected_total = Counter(
    "llm_scheduler_submission_rejected_total", "Pre-check rejections", ["model", "reason"])
scheduler_loop_iterations_total = Counter(
    "llm_scheduler_loop_iterations_total", "Scheduler loop iterations")
scheduler_batch_size = Histogram(
    "llm_scheduler_batch_size", "Jobs per batch", ["model"],
    buckets=[1, 2, 3, 5, 10, 20])
# Phase 0 scaffolding for "one model per GPU" scheduler rewrite. Exports the
# currently-loaded model per runner (value is always 1 when set). Populated
# from the existing _sync_loaded_models data — no scheduler behavior change.
scheduler_runner_current_model = Gauge(
    "llm_scheduler_runner_current_model",
    "Model currently loaded on a runner (value=1). Absence = no model loaded.",
    ["runner", "model"],
)


SCHEDULER_LOCK_ID = 900001  # Must match main.py


class Scheduler:
    def __init__(self, pool: asyncpg.Pool, get_runner_client: callable,
                 lock_conn=None):
        self.pool = pool
        self.get_runner_client = get_runner_client  # async callable(runner_id=None) -> LLMAgentClient
        self.lock_conn = lock_conn  # dedicated connection for advisory lock
        self._task: Optional[asyncio.Task] = None
        self._current_job_id: Optional[str] = None
        self._running = False
        # Track loaded models: {model_name: {"loaded_at": float, "vram_gb": float, "runner_id": int}}
        self._loaded_models: dict[str, dict] = {}
        # Models with a load in flight — prevents duplicate dispatch across iterations
        # when a prior load is slow, or when sync transiently misses a runner.
        self._loading_models: set[str] = set()
        # Cache: model -> runner_id (which runner has it downloaded)
        self._model_runner_cache: dict[str, int] = {}
        # Tracks which (hostname, model) label combos are currently set on the
        # scheduler_runner_current_model gauge, so we can remove stale series.
        self._runner_model_gauge_labels: set[tuple[str, str]] = set()

    def start(self):
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("Scheduler started")

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Scheduler stopped")

    @property
    def current_job_id(self) -> Optional[str]:
        return self._current_job_id

    @property
    def loaded_models(self) -> dict[str, dict]:
        return dict(self._loaded_models)

    async def _get_runner_for_model(self, model: str):
        """Find which runner has a model downloaded and return its client.
        Checks cache first, then queries all runners."""
        import db as _db

        # Check cache
        if model in self._model_runner_cache:
            try:
                return await self.get_runner_client(runner_id=self._model_runner_cache[model])
            except Exception:
                del self._model_runner_cache[model]

        # Query all runners
        runners = await _db.get_active_runners(self.pool)
        for r in runners:
            try:
                client = await self.get_runner_client(runner_id=r["id"])
                models_resp = await client.models()
                model_list = models_resp.get("data", models_resp.get("models", []))
                names = [m.get("id", m.get("name", "")) for m in model_list]
                if model in names:
                    self._model_runner_cache[model] = r["id"]
                    logger.info("Model %s found on runner %s (id=%d)", model, r["hostname"], r["id"])
                    # Cache actual model size in DB if not already set
                    await self._cache_model_size(model, model_list)
                    return client
            except Exception:
                continue

        # Not found on any runner
        raise RuntimeError(f"Model {model} not found on any runner")

    async def _cache_model_size(self, model: str, model_list: list[dict]):
        """Store actual model size from runner's model list in model_settings DB."""
        existing = await queue_db.get_model_settings(self.pool, model)
        if existing.get("vram_estimate_gb"):
            return  # already set, don't overwrite
        for m in model_list:
            if m.get("id", m.get("name", "")) != model:
                continue
            size_bytes = m.get("size", 0)
            if size_bytes and size_bytes > 0:
                size_gb = round(size_bytes / (1024 ** 3), 2)
                await queue_db.upsert_model_settings(self.pool, model, vram_estimate_gb=size_gb)
                logger.info("Cached VRAM estimate for %s from runner: %.2f GB", model, size_gb)
            break

    async def _verify_lock(self) -> bool:
        """Verify we still hold the advisory lock. Returns False if lost."""
        if not self.lock_conn:
            return True  # No lock connection = single-replica mode
        try:
            # Liveness only — pg_try_advisory_lock nests per call and will leak
            # session lock depth until the connection dies (see scheduler_v2).
            await self.lock_conn.fetchval("SELECT 1")
            return True
        except Exception:
            logger.warning("Failed to verify scheduler lock — connection lost")
            return False

    async def _loop(self):
        """Main scheduler loop. Processes jobs grouped by model."""
        cleanup_counter = 0
        while self._running:
            try:
                # Verify we still hold the lock before processing
                if not await self._verify_lock():
                    logger.warning("Lost scheduler advisory lock — stopping scheduler")
                    self._running = False
                    break

                scheduler_loop_iterations_total.inc()
                jobs = await queue_db.get_pending_jobs(self.pool)
                if not jobs:
                    await asyncio.sleep(1)
                    cleanup_counter += 1
                    if cleanup_counter > 3600:  # hourly cleanup
                        await queue_db.cleanup_old_jobs(self.pool)
                        cleanup_counter = 0
                    continue

                # Split cloud vs local jobs
                cloud_jobs = [j for j in jobs if detect_provider(j["model"]) != ModelProvider.LOCAL]
                local_jobs = [j for j in jobs if detect_provider(j["model"]) == ModelProvider.LOCAL]

                # Process cloud jobs immediately (no VRAM management needed)
                for job in cloud_jobs:
                    if not self._running:
                        break
                    await self._run_job(job)

                if not local_jobs:
                    continue

                # Refresh loaded model list from Ollama
                await self._sync_loaded_models()

                # Group local jobs by model
                model_groups: dict[str, list[dict]] = {}
                for job in local_jobs:
                    model_groups.setdefault(job["model"], []).append(job)

                # Process already-loaded models first (no swap needed)
                for model in list(model_groups.keys()):
                    if model in self._loaded_models:
                        batch = model_groups.pop(model)
                        await self._process_batch(model, batch)

                # Skip models with a load already in flight — their jobs will be
                # picked up next iteration once the load completes.
                for model in list(model_groups.keys()):
                    if model in self._loading_models:
                        logger.debug("Skipping %s: load already in flight", model)
                        model_groups.pop(model)

                # Process remaining models (need loading)
                any_failed = False
                for model, batch in model_groups.items():
                    # Mark jobs as loading_model so clients see progress
                    for job in batch:
                        await queue_db.update_job_status(self.pool, job["id"], "loading_model")
                    try:
                        success = await self._ensure_model_loaded(model)
                    except RuntimeError as e:
                        # Permanent failure (e.g. model not found) — fail all jobs
                        logger.error("Permanent model load failure: %s", e)
                        for job in batch:
                            await queue_db.update_job_status(
                                self.pool, job["id"], "failed", error=str(e))
                        continue
                    if success:
                        await self._process_batch(model, batch)
                    else:
                        # Revert to queued so they're retried
                        for job in batch:
                            await queue_db.update_job_status(self.pool, job["id"], "queued")
                        any_failed = True

                # Back off if we couldn't load any models — avoid tight loop
                if any_failed:
                    await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler loop error")
                await asyncio.sleep(5)

    async def _sync_loaded_models(self):
        """Sync loaded model list from all runner agents.

        Only drops cached entries for runners we successfully polled. A transient
        failure on one runner must not wipe the cache — doing so causes the
        scheduler to re-dispatch loads for models that are already loaded.
        """
        import db as _db
        try:
            runners = await _db.get_active_runners(self.pool)
            polled_runner_ids: set[int] = set()
            polled_runner_hostnames: set[str] = set()
            fresh: dict[str, dict] = {}
            current_per_runner: list[tuple[str, str]] = []  # (hostname, model)
            for r in runners:
                try:
                    client = await self.get_runner_client(runner_id=r["id"])
                    status = await client.status()
                except Exception:
                    continue
                polled_runner_ids.add(r["id"])
                polled_runner_hostnames.add(r["hostname"])
                for m in status.get("loaded_ollama_models", []):
                    name = m.get("name", "")
                    vram_gb = m.get("size_gb", vram_for_model(name))
                    fresh[name] = {
                        "loaded_at": self._loaded_models.get(name, {}).get("loaded_at", time.time()),
                        "vram_gb": vram_gb,
                        "active_jobs": len(await queue_db.get_active_jobs_for_model(self.pool, name)),
                        "runner_id": r["id"],
                    }
                    current_per_runner.append((r["hostname"], name))
            # Keep cached entries for runners we couldn't poll this cycle.
            merged = {
                name: info
                for name, info in self._loaded_models.items()
                if info.get("runner_id") not in polled_runner_ids
            }
            merged.update(fresh)
            self._loaded_models = merged

            # Refresh the per-runner current-model gauge. Only touch series for
            # runners we successfully polled — a transient unreachable shouldn't
            # drop a runner's gauge and create false "no model loaded" readings.
            fresh_series = set(current_per_runner)
            stale = {
                s for s in self._runner_model_gauge_labels
                if s[0] in polled_runner_hostnames and s not in fresh_series
            }
            for hostname, model in stale:
                try:
                    scheduler_runner_current_model.remove(hostname, model)
                except KeyError:
                    pass
                self._runner_model_gauge_labels.discard((hostname, model))
            for hostname, model in fresh_series:
                scheduler_runner_current_model.labels(runner=hostname, model=model).set(1)
                self._runner_model_gauge_labels.add((hostname, model))
        except Exception:
            logger.warning("Failed to sync loaded models from runner agents", exc_info=True)

    async def _get_gpu_info(self, runner_id: int | None = None) -> dict:
        """Get GPU VRAM info from a specific runner or the first available."""
        import db as _db
        try:
            if runner_id:
                client = await self.get_runner_client(runner_id=runner_id)
            else:
                client = await self.get_runner_client()
            status = await client.status()
            total = status.get("gpu_vram_total_gb", 0)
            used = status.get("gpu_vram_used_gb", 0)
            return {"total": round(total, 1), "used": round(used, 1), "free": round(total - used, 1)}
        except Exception:
            return {"total": 0, "used": 0, "free": 0}

    async def _vram_for_model(self, model: str) -> float:
        """VRAM estimate: DB setting > hardcoded lookup > heuristic."""
        settings = await queue_db.get_model_settings(self.pool, model)
        if settings.get("vram_estimate_gb"):
            return float(settings["vram_estimate_gb"])
        return vram_for_model(model)

    async def _ensure_model_loaded(self, model: str) -> bool:
        """Load a model, evicting others if needed. Returns True if successful."""
        if model in self._loaded_models:
            return True

        self._loading_models.add(model)
        try:
            vram_needed = await self._vram_for_model(model)
            gpu = await self._get_gpu_info()

            # Check if model fits at all
            if vram_needed > gpu["total"] and gpu["total"] > 0:
                logger.error("Model %s needs %.1fGB but GPU has %.1fGB total",
                             model, vram_needed, gpu["total"])
                return False

            # Check if we have free VRAM
            if gpu["free"] >= vram_needed:
                return await self._load_model(model)

            # Need to evict
            vram_to_free = vram_needed - gpu["free"]
            evicted = await self._evict_for_vram(vram_to_free)
            if not evicted:
                logger.error("Cannot free %.1fGB for model %s", vram_to_free, model)
                return False

            return await self._load_model(model)
        finally:
            self._loading_models.discard(model)

    async def _evict_for_vram(self, vram_needed: float) -> bool:
        """Evict models to free up VRAM. Returns True if enough was freed."""
        candidates = []
        for name, info in self._loaded_models.items():
            settings = await queue_db.get_model_settings(self.pool, name)
            runner_id = info.get("runner_id")
            runner_params = (await queue_db.get_model_runner_params(self.pool, name, runner_id)) if runner_id else {}
            if settings.get("do_not_evict", False) or (runner_params or {}).get("do_not_evict", False):
                continue
            if not settings.get("evictable", True):
                continue
            candidates.append({
                "name": name,
                "vram_gb": info.get("vram_gb", vram_for_model(name)),
                "active_jobs": info.get("active_jobs", 0),
                "loaded_at": info.get("loaded_at", 0),
                "wait_for_completion": settings.get("wait_for_completion", True),
            })

        # Sort: idle models first, then oldest loaded
        candidates.sort(key=lambda m: (m["active_jobs"] > 0, m["loaded_at"]))

        freed = 0.0
        for candidate in candidates:
            if freed >= vram_needed:
                break

            # Wait for active jobs if configured
            if candidate["active_jobs"] > 0 and candidate["wait_for_completion"]:
                logger.info("Waiting for %d jobs on %s to complete before evicting",
                            candidate["active_jobs"], candidate["name"])
                # Wait up to 5 minutes for jobs to finish
                for _ in range(60):
                    active = await queue_db.get_active_jobs_for_model(self.pool, candidate["name"])
                    if not active:
                        break
                    await asyncio.sleep(5)

            success = await self._unload_model(candidate["name"])
            if success:
                freed += candidate["vram_gb"]
                logger.info("Evicted %s, freed %.1fGB (total freed: %.1fGB / %.1fGB needed)",
                            candidate["name"], candidate["vram_gb"], freed, vram_needed)

        return freed >= vram_needed

    async def _load_model(self, model: str) -> bool:
        """Load a model via the runner that has it downloaded. Returns True on
        success, False on retriable error. Raises RuntimeError for permanent
        failures (model not on any runner, or runner rejected load)."""
        t0 = time.time()
        try:
            client = await self._get_runner_for_model(model)
            runner_id = self._model_runner_cache.get(model)
            logger.info("Loading model %s", model)
            await client.load_model(model, keep_alive=-1)
            elapsed = time.time() - t0
            self._loaded_models[model] = {
                "loaded_at": time.time(),
                "vram_gb": vram_for_model(model),
                "active_jobs": 0,
                "runner_id": runner_id,
            }
            scheduler_model_load_total.labels(model=model, status="success").inc()
            scheduler_model_load_seconds.labels(model=model).observe(elapsed)
            self._update_loaded_gauges()
            logger.info("Model %s loaded in %.1fs", model, elapsed)
            return True
        except RuntimeError:
            scheduler_model_load_total.labels(model=model, status="permanent_failure").inc()
            raise  # model not found — permanent failure
        except httpx.HTTPStatusError as e:
            scheduler_model_load_total.labels(model=model, status="permanent_failure").inc()
            if e.response.status_code in (400, 404, 422):
                raise RuntimeError(
                    f"Model {model} cannot be loaded by runner "
                    f"(HTTP {e.response.status_code})"
                ) from e
            logger.exception("Error loading model %s", model)
            return False
        except Exception:
            scheduler_model_load_total.labels(model=model, status="error").inc()
            logger.exception("Error loading model %s", model)
            return False

    def _update_loaded_gauges(self):
        scheduler_loaded_models.set(len(self._loaded_models))
        scheduler_loaded_vram_gb.set(
            sum(m.get("vram_gb", 0) for m in self._loaded_models.values()))

    async def _unload_model(self, model: str) -> bool:
        """Unload a model via runner agent."""
        try:
            client = await self._get_runner_for_model(model)
            logger.info("Unloading model %s", model)
            await client.unload_model(model)
            self._loaded_models.pop(model, None)
            scheduler_model_eviction_total.labels(model=model).inc()
            self._update_loaded_gauges()
            logger.info("Model %s unloaded", model)
            return True
        except Exception:
            logger.exception("Error unloading model %s", model)
            return False

    async def _process_batch(self, model: str, jobs: list[dict]):
        """Process a batch of jobs for a single model."""
        scheduler_batch_size.labels(model=model).observe(len(jobs))
        logger.info("Processing %d jobs for model %s", len(jobs), model)
        for job in jobs:
            if not self._running:
                break
            await self._run_job(job)

    async def _run_job(self, job: dict):
        """Execute a single inference job (local or cloud)."""
        job_id = job["id"]
        model = job["model"]
        request = job["request"] if isinstance(job["request"], dict) else json.loads(job["request"])

        self._current_job_id = job_id
        await queue_db.update_job_status(self.pool, job_id, "running")

        try:
            provider = detect_provider(model)
            if provider == ModelProvider.ANTHROPIC:
                await self._run_cloud_job(job_id, model, request)
            else:
                await self._run_local_job(job_id, model, request)
        except Exception as e:
            await queue_db.update_job_status(self.pool, job_id, "failed", error=str(e))
            logger.exception("Job %s error", job_id)
        finally:
            self._current_job_id = None

    async def _run_cloud_job(self, job_id: str, model: str, request: dict):
        """Execute a cloud model inference job."""
        body = {"model": model, "messages": request.get("messages", []), "stream": False}
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        if "max_tokens" in request:
            body["max_tokens"] = request["max_tokens"]
        if request.get("tools"):
            body["tools"] = request["tools"]

        result = await anthropic_chat(body, stream=False)
        await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
        usage = result.get("usage", {})
        logger.info("Job %s completed (model=%s, cloud, tokens=%d)",
                    job_id, model, usage.get("completion_tokens", 0))

    async def _run_local_job(self, job_id: str, model: str, request: dict):
        """Execute a local inference job via runner agent."""
        client = await self._get_runner_for_model(model)
        kwargs = {}
        if "temperature" in request:
            kwargs["temperature"] = request["temperature"]
        if "max_tokens" in request:
            kwargs["max_tokens"] = request["max_tokens"]
        if request.get("tools"):
            kwargs["tools"] = request["tools"]

        t0 = time.time()
        try:
            result = await client.chat(
                messages=request.get("messages", []),
                model=model,
                stream=False,
                **kwargs,
            )
            elapsed = time.time() - t0
            scheduler_job_inference_seconds.labels(model=model).observe(elapsed)
            await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            scheduler_job_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
            scheduler_job_tokens_total.labels(model=model, type="completion").inc(completion_tokens)
            logger.info("Job %s completed (model=%s, tokens=%d, %.1fs)",
                        job_id, model, completion_tokens, elapsed)
        except Exception as e:
            scheduler_job_errors_total.labels(model=model, reason="inference_error").inc()
            error = f"Runner agent error: {e}"
            await queue_db.update_job_status(self.pool, job_id, "failed", error=error)
            logger.error("Job %s failed: %s", job_id, error)

    # ── Public methods for VRAM analysis ──────────────────────────────────────

    async def check_submission(
        self, model: str, allowed_runner_ids: list[int] | None = None
    ) -> dict:
        """Pre-check a job submission against ALL runners. A model is accepted
        if at least one runner can fit it (directly, or after eviction)."""
        if detect_provider(model) != ModelProvider.LOCAL:
            return {"ok": True, "provider": detect_provider(model).value}

        import db as _db
        vram_needed = await self._vram_for_model(model)

        runners = await _db.get_active_runners(self.pool)
        if allowed_runner_ids:
            allowed = set(allowed_runner_ids)
            runners = [r for r in runners if r.get("id") in allowed]
            if not runners:
                return {
                    "ok": False,
                    "error": "no_schedulable_runners",
                    "message": "No schedulable runners available for this app.",
                }
        if not runners:
            return {"ok": True}

        best_result = None
        max_gpu_total = 0
        unreachable_count = 0
        unreachable_hostnames: list[str] = []

        for r in runners:
            try:
                client = await self.get_runner_client(runner_id=r["id"])
                status = await client.status()
            except Exception as e:
                unreachable_count += 1
                unreachable_hostnames.append(r.get("hostname", f"id={r.get('id')}"))
                logger.warning(
                    "check_submission: runner %s (id=%s) unreachable for model=%s: %s",
                    r.get("hostname"), r.get("id"), model, e,
                )
                continue

            gpu_total = status.get("gpu_vram_total_gb", 0)
            gpu_used = status.get("gpu_vram_used_gb", 0)
            gpu_free = round(gpu_total - gpu_used, 1)
            loaded_models = {m["name"]: m.get("size_gb", 0) for m in status.get("loaded_ollama_models", [])}

            # Inconsistent-state detection: the driver shows significant VRAM
            # used but Ollama's loaded-models list is empty. Happens briefly
            # during an unload/reload when the driver still holds the pages.
            # We can't reason about eviction with this snapshot (evictable will
            # compute to 0, causing a spurious insufficient_vram rejection),
            # so treat the runner as unreachable for pre-check purposes — the
            # unreachable-aware acceptance path then accepts optimistically
            # and runtime eviction re-checks with fresh state.
            VRAM_IDLE_THRESHOLD_GB = 1.5
            if gpu_used > VRAM_IDLE_THRESHOLD_GB and not loaded_models:
                unreachable_count += 1
                hostname = r.get("hostname", f"id={r.get('id')}")
                unreachable_hostnames.append(f"{hostname} (stale snapshot)")
                logger.warning(
                    "check_submission: runner %s inconsistent snapshot — "
                    "gpu_used=%.1fGB but loaded=[]; treating as unreachable",
                    hostname, gpu_used,
                )
                continue

            if gpu_total > max_gpu_total:
                max_gpu_total = gpu_total

            if vram_needed > gpu_total:
                logger.info(
                    "check_submission: runner=%s model=%s too large (need=%.1f total=%.1f)",
                    r.get("hostname"), model, vram_needed, gpu_total,
                )
                continue

            if model in loaded_models:
                logger.info(
                    "check_submission: runner=%s model=%s already loaded",
                    r.get("hostname"), model,
                )
                return {"ok": True}

            if gpu_free >= vram_needed:
                logger.info(
                    "check_submission: runner=%s model=%s fits free (need=%.1f free=%.1f)",
                    r.get("hostname"), model, vram_needed, gpu_free,
                )
                return {"ok": True}

            evictable_vram = 0
            non_evictable_vram = 0
            loaded_info = []
            for name, size_gb in loaded_models.items():
                settings = await queue_db.get_model_settings(self.pool, name)
                model_runner_id = self._loaded_models.get(name, {}).get("runner_id")
                runner_params = (await queue_db.get_model_runner_params(self.pool, name, model_runner_id)) if model_runner_id else {}
                model_vram = size_gb or vram_for_model(name)
                pinned = settings.get("do_not_evict", False) or (runner_params or {}).get("do_not_evict", False) or not settings.get("evictable", True)
                if pinned:
                    non_evictable_vram += model_vram
                    loaded_info.append({"model": name, "vram_gb": model_vram, "do_not_evict": True})
                else:
                    evictable_vram += model_vram
                    loaded_info.append({"model": name, "vram_gb": model_vram, "do_not_evict": False})

            available_after_eviction = gpu_free + evictable_vram
            logger.info(
                "check_submission: runner=%s model=%s need=%.1f total=%.1f used=%.1f free=%.1f "
                "evictable=%.1f non_evictable=%.1f after_evict=%.1f loaded=%s",
                r.get("hostname"), model, vram_needed, gpu_total, gpu_used, gpu_free,
                evictable_vram, non_evictable_vram, available_after_eviction,
                list(loaded_models.keys()),
            )
            if available_after_eviction >= vram_needed:
                to_evict = [m["model"] for m in loaded_info if not m["do_not_evict"]]
                best_result = {
                    "ok": True,
                    "warning": "eviction_required",
                    "message": f"Will evict {', '.join(to_evict)} to free VRAM for {model}",
                    "evicting": to_evict,
                }

        if best_result:
            return best_result

        if max_gpu_total == 0:
            return {"ok": True}

        # model_too_large only rejects if we were confident about the answer —
        # i.e. every runner responded and none of them had a big enough GPU.
        # An unreachable runner might have had the capacity.
        if vram_needed > max_gpu_total and unreachable_count == 0:
            scheduler_submission_rejected_total.labels(model=model, reason="model_too_large").inc()
            return {
                "ok": False,
                "error": "model_too_large",
                "message": f"{model} requires {vram_needed:.1f}GB VRAM, largest GPU has {max_gpu_total:.1f}GB",
                "vram_required_gb": vram_needed,
                "vram_available_gb": max_gpu_total,
            }

        # If any runner was unreachable during this pre-check, we lacked the
        # data to reject confidently. Accept optimistically — the scheduler's
        # runtime eviction will re-evaluate with fresh state when the job is
        # processed.
        if unreachable_count > 0:
            logger.info(
                "check_submission: accepting model=%s optimistically — %d runner(s) unreachable: %s",
                model, unreachable_count, ", ".join(unreachable_hostnames),
            )
            return {
                "ok": True,
                "warning": "some_runners_unreachable",
                "message": (
                    f"{unreachable_count} runner(s) unreachable during pre-check; "
                    f"accepting {model} optimistically"
                ),
                "unreachable_runners": unreachable_hostnames,
            }

        scheduler_submission_rejected_total.labels(model=model, reason="insufficient_vram").inc()
        return {
            "ok": False,
            "error": "insufficient_vram",
            "message": f"{model} requires {vram_needed:.1f}GB, no runner can fit it after eviction",
            "vram_required_gb": vram_needed,
        }
