"""
Unit tests for the one-model-per-GPU scheduler (scheduler_v2).

Focus: _pick_runner policy truth table and check_submission's new trivial
shape. Swap and _run_job are exercised via integration; here we mock them.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from scheduler_v2 import SimplifiedScheduler, RunnerState


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_sched(runners=None):
    pool = MagicMock()
    get_client = AsyncMock()
    sched = SimplifiedScheduler(pool, get_client)
    if runners:
        sched._runners = {r.runner_id: r for r in runners}
    return sched


# ── _pick_runner ─────────────────────────────────────────────────────────────

class TestPickRunner:
    def test_pins_win(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        pinned_model="qwen3:14b",
                        downloaded_models={"qwen3:14b"})
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=24, current_model="qwen3:14b")
        sched = _make_sched([a, b])
        assert sched._pick_runner("qwen3:14b") is a

    def test_pinned_busy_blocks_fallback(self):
        # Pinned runner is busy → return None, DON'T steal capacity from another runner
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        pinned_model="qwen3:14b", in_flight_job_id="j1",
                        downloaded_models={"qwen3:14b"})
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=24,
                        downloaded_models={"qwen3:14b"})
        sched = _make_sched([a, b])
        assert sched._pick_runner("qwen3:14b") is None

    def test_pin_without_download_not_picked(self):
        # A pin that refers to a model the runner hasn't downloaded yet shouldn't
        # force a doomed swap. Wait for the pull to finish.
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        pinned_model="qwen3:14b")  # no downloads
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is None

    def test_pinned_loaded_without_download_still_picked(self):
        # Regression: heartbeat downloaded_models can be stale/missing a tag.
        # If the model is already loaded on an idle pinned runner, use it.
        a = RunnerState(
            runner_id=1,
            hostname="a",
            gpu_total_gb=17,
            pinned_model="qwen3:14b",
            current_model="qwen3:14b",
        )
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is a

    def test_already_loaded_idle_wins_over_idle_empty(self):
        # Rule 2: current_model match bypasses the downloaded_models check
        # (loaded implies downloaded).
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17)  # empty
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=17,
                        current_model="qwen3:14b")  # loaded
        sched = _make_sched([a, b])
        assert sched._pick_runner("qwen3:14b") is b

    def test_busy_same_model_rejected(self):
        # Runner on the right model but running another job for it → can't reuse
        # (one-model-one-GPU serializes on the runner; wait).
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        current_model="qwen3:14b", in_flight_job_id="j1")
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is None

    def test_idle_fits_picked_for_swap(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        current_model="deepseek-r1:14b",
                        downloaded_models={"qwen3:14b", "deepseek-r1:14b"})
        sched = _make_sched([a])
        picked = sched._pick_runner("qwen3:14b")
        assert picked is a  # will swap

    def test_idle_fits_but_no_download_skipped(self):
        # Core regression fix: runner fits VRAM but doesn't have the model on
        # disk. Pre-fix this was picked, the swap failed, the fallback loaded
        # on the drained runner anyway. Post-fix: no pick — job waits.
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=40,
                        current_model="something-else")
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3.6:35b-a3b") is None

    def test_too_large_skipped(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=8.6,
                        downloaded_models={"qwen3:70b"})
        sched = _make_sched([a])
        # qwen3:70b → 40GB estimate > 8.6
        assert sched._pick_runner("qwen3:70b") is None

    def test_unknown_gpu_total_skipped(self):
        # Runner whose gpu_total_gb=0 hasn't been reconciled yet — defensively skip
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=0,
                        downloaded_models={"qwen3:14b"})
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is None

    def test_no_runners(self):
        sched = _make_sched([])
        assert sched._pick_runner("qwen3:14b") is None

    def test_draining_runner_skipped(self):
        # Only runner is draining → no one to pick, even if it's idle and has the model loaded
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        current_model="qwen3:14b", draining=True)
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is None

    def test_draining_runner_skipped_fallback_to_other(self):
        # One draining runner has the model, another non-draining idle has it
        # downloaded → pick the non-draining for swap.
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        current_model="qwen3:14b", draining=True)
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=24,
                        downloaded_models={"qwen3:14b"})
        sched = _make_sched([a, b])
        assert sched._pick_runner("qwen3:14b") is b  # will swap b to qwen3:14b

    def test_draining_pinned_blocks_new_work(self):
        # Pinned but draining → don't fall through to other runners (admin intent wins)
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        pinned_model="qwen3:14b", draining=True,
                        downloaded_models={"qwen3:14b"})
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=24,
                        downloaded_models={"qwen3:14b"})
        sched = _make_sched([a, b])
        assert sched._pick_runner("qwen3:14b") is None

    def test_draining_allows_inflight_to_finish(self):
        # A draining runner with an in-flight job is NOT picked for new work.
        # (The in-flight job itself runs to completion via _run_job — this test
        # only asserts the "no new work" half; the other half is structural.)
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        current_model="qwen3:14b", in_flight_job_id="j1", draining=True)
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3:14b") is None

    def test_latest_tag_alias(self):
        # Catalog-style requests come in as "qwen3" or "qwen3:14b"; agents often
        # report "qwen3:latest" for the default tag. has_downloaded() aliases
        # base → base:latest so _pick_runner still matches.
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17,
                        downloaded_models={"qwen3:latest"})
        sched = _make_sched([a])
        assert sched._pick_runner("qwen3") is a


# ── check_submission ────────────────────────────────────────────────────────

class TestCheckSubmission:
    def test_cloud_models_always_ok(self):
        sched = _make_sched([])
        result = _run(sched.check_submission("claude-sonnet-4-5"))
        assert result["ok"] is True

    def test_fits_on_biggest_runner(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17)
        sched = _make_sched([a])
        result = _run(sched.check_submission("qwen3:14b"))
        assert result["ok"] is True

    def test_too_large_for_any(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=8.6)
        sched = _make_sched([a])
        result = _run(sched.check_submission("qwen3:70b"))  # ~40GB estimate
        assert result["ok"] is False
        assert result["error"] == "model_too_large"
        assert result["vram_available_gb"] == 8.6

    def test_no_runners_defer(self):
        """No runners visible and no cache → defer accept (let runtime sort it)."""
        sched = _make_sched([])
        import db as _db
        _db.get_active_runners = AsyncMock(return_value=[])
        result = _run(sched.check_submission("qwen3:14b"))
        assert result["ok"] is True

    def test_uses_db_capabilities_when_cache_empty(self):
        """The non-scheduler replica has _runners empty — should fall back to DB row capabilities."""
        sched = _make_sched([])
        import db as _db
        _db.get_active_runners = AsyncMock(return_value=[
            {"id": 1, "hostname": "archlinux",
             "capabilities": {"gpu_vram_total_bytes": 17_100_000_000}},
        ])
        result = _run(sched.check_submission("qwen3:14b"))
        assert result["ok"] is True

    def test_uses_db_capabilities_json_string(self):
        """capabilities can come back as a JSON string from asyncpg depending on version."""
        sched = _make_sched([])
        import db as _db
        _db.get_active_runners = AsyncMock(return_value=[
            {"id": 1, "hostname": "a",
             "capabilities": json.dumps({"gpu_vram_total_bytes": 17_100_000_000})},
        ])
        result = _run(sched.check_submission("qwen3:14b"))
        assert result["ok"] is True

    def test_prefers_in_memory_cache_over_db(self):
        """If RunnerState has values, we should use them (they're fresher)."""
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17)
        sched = _make_sched([a])
        # DB has different (stale) total — should be ignored
        import db as _db
        _db.get_active_runners = AsyncMock(return_value=[
            {"id": 1, "hostname": "a",
             "capabilities": {"gpu_vram_total_bytes": 1_000_000_000}},
        ])
        result = _run(sched.check_submission("qwen3:14b"))
        assert result["ok"] is True

    def test_allowed_runner_filter_no_candidates_returns_unschedulable(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17)
        sched = _make_sched([a])
        result = _run(sched.check_submission("qwen3:14b", allowed_runner_ids=[99]))
        assert result["ok"] is False
        assert result["error"] == "no_schedulable_runners"


# ── loaded_models property compat ────────────────────────────────────────────

class TestLoadedModelsProperty:
    def test_empty_when_no_runners_loaded(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17)
        sched = _make_sched([a])
        assert sched.loaded_models == {}

    def test_returns_current_models(self):
        a = RunnerState(runner_id=1, hostname="a", gpu_total_gb=17, current_model="qwen3:14b",
                        model_loaded_at=1000.0)
        b = RunnerState(runner_id=2, hostname="b", gpu_total_gb=8.6)
        sched = _make_sched([a, b])
        loaded = sched.loaded_models
        assert "qwen3:14b" in loaded
        assert loaded["qwen3:14b"]["runner_id"] == 1
        assert loaded["qwen3:14b"]["loaded_at"] == 1000.0
        assert loaded["qwen3:14b"]["runner_hostname"] == "a"


# ── lifecycle ────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_stop(self):
        sched = _make_sched([])
        # Don't actually run the loop — it'll fail without a real pool
        sched._running = True
        sched._task = MagicMock()
        sched._task.done = MagicMock(return_value=True)
        sched.stop()
        assert sched._running is False

    def test_current_job_id_starts_none(self):
        sched = _make_sched([])
        assert sched.current_job_id is None
