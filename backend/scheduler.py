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

import queue_db
from gpu import vram_for_model
from cloud_providers import detect_provider, ModelProvider, anthropic_chat

logger = logging.getLogger(__name__)


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
        # Cache: model -> runner_id (which runner has it downloaded)
        self._model_runner_cache: dict[str, int] = {}

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
                names = [m.get("id", m.get("name", "")) for m in models_resp.get("data", models_resp.get("models", []))]
                if model in names:
                    self._model_runner_cache[model] = r["id"]
                    logger.info("Model %s found on runner %s (id=%d)", model, r["hostname"], r["id"])
                    return client
            except Exception:
                continue

        # Not found on any runner
        raise RuntimeError(f"Model {model} not found on any runner")

    async def _verify_lock(self) -> bool:
        """Verify we still hold the advisory lock. Returns False if lost."""
        if not self.lock_conn:
            return True  # No lock connection = single-replica mode
        try:
            held = await self.lock_conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", SCHEDULER_LOCK_ID
            )
            # pg_try_advisory_lock returns True if we acquired it (or already hold it)
            return held
        except Exception:
            logger.warning("Failed to verify scheduler lock")
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
        """Sync loaded model list from all runner agents."""
        import db as _db
        try:
            runners = await _db.get_active_runners(self.pool)
            current = {}
            for r in runners:
                try:
                    client = await self.get_runner_client(runner_id=r["id"])
                    status = await client.status()
                    for m in status.get("loaded_ollama_models", []):
                        name = m.get("name", "")
                        vram_gb = m.get("size_gb", vram_for_model(name))
                        current[name] = {
                            "loaded_at": self._loaded_models.get(name, {}).get("loaded_at", time.time()),
                            "vram_gb": vram_gb,
                            "active_jobs": len(await queue_db.get_active_jobs_for_model(self.pool, name)),
                            "runner_id": r["id"],
                        }
                except Exception:
                    continue
            self._loaded_models = current
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

    async def _ensure_model_loaded(self, model: str) -> bool:
        """Load a model, evicting others if needed. Returns True if successful."""
        if model in self._loaded_models:
            return True

        vram_needed = vram_for_model(model)
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

    async def _evict_for_vram(self, vram_needed: float) -> bool:
        """Evict models to free up VRAM. Returns True if enough was freed."""
        candidates = []
        for name, info in self._loaded_models.items():
            settings = await queue_db.get_model_settings(self.pool, name)
            if settings.get("do_not_evict", False):
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
        failures (model not on any runner)."""
        try:
            client = await self._get_runner_for_model(model)
            logger.info("Loading model %s", model)
            await client.load_model(model, keep_alive=-1)
            self._loaded_models[model] = {
                "loaded_at": time.time(),
                "vram_gb": vram_for_model(model),
                "active_jobs": 0,
            }
            logger.info("Model %s loaded", model)
            return True
        except RuntimeError:
            raise  # model not found — permanent failure
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(f"Model {model} not found on runner") from e
            logger.exception("Error loading model %s", model)
            return False
        except Exception:
            logger.exception("Error loading model %s", model)
            return False

    async def _unload_model(self, model: str) -> bool:
        """Unload a model via runner agent."""
        try:
            client = await self._get_runner_for_model(model)
            logger.info("Unloading model %s", model)
            await client.unload_model(model)
            self._loaded_models.pop(model, None)
            logger.info("Model %s unloaded", model)
            return True
        except Exception:
            logger.exception("Error unloading model %s", model)
            return False

    async def _process_batch(self, model: str, jobs: list[dict]):
        """Process a batch of jobs for a single model."""
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

        try:
            result = await client.chat(
                messages=request.get("messages", []),
                model=model,
                stream=False,
                **kwargs,
            )
            # Runner agent returns OpenAI-format response
            await queue_db.update_job_status(self.pool, job_id, "completed", result=result)
            usage = result.get("usage", {})
            logger.info("Job %s completed (model=%s, tokens=%d)",
                        job_id, model, usage.get("completion_tokens", 0))
        except Exception as e:
            error = f"Runner agent error: {e}"
            await queue_db.update_job_status(self.pool, job_id, "failed", error=error)
            logger.error("Job %s failed: %s", job_id, error)

    # ── Public methods for VRAM analysis ──────────────────────────────────────

    async def check_submission(self, model: str) -> dict:
        """Pre-check a job submission. Returns warnings/errors about VRAM."""
        # Cloud models don't need VRAM checks
        if detect_provider(model) != ModelProvider.LOCAL:
            return {"ok": True, "provider": detect_provider(model).value}

        vram_needed = vram_for_model(model)

        # Get live GPU info and loaded models from runner
        try:
            client = await self.get_runner_client()
            status = await client.status()
            gpu_total = status.get("gpu_vram_total_gb", 0)
            gpu_used = status.get("gpu_vram_used_gb", 0)
            gpu_free = round(gpu_total - gpu_used, 1)
            loaded_models = {m["name"]: m.get("size_gb", 0) for m in status.get("loaded_ollama_models", [])}
        except Exception:
            # Can't reach runner — accept optimistically
            return {"ok": True}

        if gpu_total == 0:
            return {"ok": True}

        if vram_needed > gpu_total:
            return {
                "ok": False,
                "error": "model_too_large",
                "message": f"{model} requires {vram_needed:.1f}GB VRAM, GPU has {gpu_total:.1f}GB total",
                "vram_required_gb": vram_needed,
                "vram_available_gb": gpu_total,
            }

        # Model already loaded — no swap needed
        if model in loaded_models:
            return {"ok": True}

        if gpu_free >= vram_needed:
            return {"ok": True}

        # Check eviction feasibility using live loaded model data
        evictable_vram = 0
        non_evictable_vram = 0
        loaded_info = []
        for name, size_gb in loaded_models.items():
            settings = await queue_db.get_model_settings(self.pool, name)
            model_vram = size_gb or vram_for_model(name)
            if settings.get("do_not_evict", False) or not settings.get("evictable", True):
                non_evictable_vram += model_vram
                loaded_info.append({"model": name, "vram_gb": model_vram, "do_not_evict": True})
            else:
                evictable_vram += model_vram
                loaded_info.append({"model": name, "vram_gb": model_vram, "do_not_evict": False})

        available_after_eviction = gpu_free + evictable_vram
        if available_after_eviction < vram_needed:
            return {
                "ok": False,
                "error": "insufficient_vram",
                "message": (f"{model} requires {vram_needed:.1f}GB, only {available_after_eviction:.1f}GB "
                            f"available after eviction. {non_evictable_vram:.1f}GB held by non-evictable models."),
                "vram_required_gb": vram_needed,
                "vram_available_gb": available_after_eviction,
                "non_evictable_gb": non_evictable_vram,
                "loaded_models": loaded_info,
            }

        # Can evict, return warning
        to_evict = [m["model"] for m in loaded_info if not m["do_not_evict"]]
        return {
            "ok": True,
            "warning": "eviction_required",
            "message": f"Will evict {', '.join(to_evict)} to free VRAM for {model}",
            "evicting": to_evict,
        }
