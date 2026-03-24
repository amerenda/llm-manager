
"""
Tests for the queue system: models, DB operations, scheduler, and routes.
Covers queue_models.py, queue_db.py, scheduler.py, and queue_routes.py.
"""
import asyncio
import datetime
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Queue models unit tests ──────────────────────────────────────────────────

from queue_models import (
    QueueJobRequest, QueueBatchRequest, QueueJobResponse,
    QueueJobResult, QueueBatchStatus, QueueOverview,
    ModelSettingsUpdate, ModelSettings, EvictionError,
)


class TestQueueJobRequest:
    """Test QueueJobRequest validation and defaults."""

    def test_minimal_request(self):
        req = QueueJobRequest(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.model == "qwen2.5:7b"
        assert req.temperature == 0.7
        assert req.max_tokens == 512
        assert req.stream is False
        assert req.metadata is None

    def test_full_request(self):
        req = QueueJobRequest(
            model="llama3:8b",
            messages=[{"role": "system", "content": "You are helpful"},
                      {"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=1024,
            stream=True,
            metadata={"app": "test", "ref_id": 42},
        )
        assert req.temperature == 0.3
        assert req.max_tokens == 1024
        assert req.stream is True
        assert req.metadata == {"app": "test", "ref_id": 42}

    def test_missing_required_fields_raises(self):
        with pytest.raises(Exception):
            QueueJobRequest(model="qwen2.5:7b")  # missing messages


class TestQueueBatchRequest:
    def test_batch_with_multiple_jobs(self):
        batch = QueueBatchRequest(jobs=[
            QueueJobRequest(model="a:7b", messages=[{"role": "user", "content": "1"}]),
            QueueJobRequest(model="b:3b", messages=[{"role": "user", "content": "2"}]),
        ])
        assert len(batch.jobs) == 2
        assert batch.jobs[0].model == "a:7b"
        assert batch.jobs[1].model == "b:3b"

    def test_empty_batch_is_valid(self):
        batch = QueueBatchRequest(jobs=[])
        assert batch.jobs == []


class TestQueueJobResponse:
    def test_defaults(self):
        resp = QueueJobResponse(job_id="abc", status="queued", model="m:7b")
        assert resp.position is None
        assert resp.warning is None
        assert resp.evicting is None


class TestModelSettings:
    def test_defaults(self):
        ms = ModelSettings(model_name="qwen2.5:7b")
        assert ms.do_not_evict is False
        assert ms.evictable is True
        assert ms.wait_for_completion is True
        assert ms.vram_estimate_gb is None


class TestModelSettingsUpdate:
    def test_all_none_by_default(self):
        u = ModelSettingsUpdate()
        assert u.do_not_evict is None
        assert u.evictable is None
        assert u.wait_for_completion is None

    def test_partial_update(self):
        u = ModelSettingsUpdate(do_not_evict=True)
        assert u.do_not_evict is True
        assert u.evictable is None


class TestEvictionError:
    def test_fields(self):
        e = EvictionError(
            error="model_too_large",
            message="Too big",
            vram_required_gb=40.0,
            vram_available_gb=24.0,
        )
        assert e.non_evictable_gb is None
        assert e.loaded_models is None


class TestQueueOverview:
    def test_defaults(self):
        ov = QueueOverview(
            queue_depth=3,
            models_queued=["a"],
            models_loaded=["b"],
        )
        assert ov.current_job is None
        assert ov.gpu_vram_total_gb == 0
        assert ov.gpu_vram_free_gb == 0


# ── Queue DB unit tests ──────────────────────────────────────────────────────

import queue_db


def _make_mock_pool():
    """Create a mock asyncpg pool with an acquire context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


class TestInsertJob:
    def test_insert_job_calls_fetchrow(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "id": "job1", "batch_id": None, "app_id": None,
            "model": "qwen2.5:7b", "status": "queued", "priority": 0,
            "request": {}, "metadata": None, "result": None,
            "error": None, "created_at": None, "started_at": None,
            "completed_at": None,
        }
        result = _run(queue_db.insert_job(
            pool, "job1", None, None, "qwen2.5:7b", {"messages": []}, None))
        assert result["id"] == "job1"
        assert result["model"] == "qwen2.5:7b"
        conn.fetchrow.assert_called_once()

    def test_insert_job_with_metadata(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "id": "j2", "batch_id": "b1", "app_id": 5,
            "model": "m:3b", "status": "queued", "priority": 1,
            "request": {}, "metadata": {"foo": "bar"}, "result": None,
            "error": None, "created_at": None, "started_at": None,
            "completed_at": None,
        }
        result = _run(queue_db.insert_job(
            pool, "j2", "b1", 5, "m:3b", {"messages": []}, {"foo": "bar"}, priority=1))
        assert result["batch_id"] == "b1"
        assert result["metadata"] == {"foo": "bar"}
        assert result["priority"] == 1


class TestGetJob:
    def test_get_existing_job(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {"id": "j1", "status": "queued", "model": "m"}
        result = _run(queue_db.get_job(pool, "j1"))
        assert result["id"] == "j1"

    def test_get_missing_job_returns_none(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        result = _run(queue_db.get_job(pool, "missing"))
        assert result is None


class TestUpdateJobStatus:
    def test_update_to_running_sets_started_at(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.update_job_status(pool, "j1", "running"))
        sql = conn.execute.call_args[0][0]
        assert "started_at" in sql
        assert "now()" in sql

    def test_update_to_completed_sets_result(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.update_job_status(pool, "j1", "completed", result={"choices": []}))
        sql = conn.execute.call_args[0][0]
        assert "completed_at" in sql
        # Args: sql, job_id, status, result_json, error
        args = conn.execute.call_args[0]
        assert json.loads(args[3]) == {"choices": []}

    def test_update_to_failed_sets_error(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.update_job_status(pool, "j1", "failed", error="timeout"))
        args = conn.execute.call_args[0]
        # Args: sql, job_id, status, result_json(None), error
        assert args[4] == "timeout"

    def test_update_to_cancelled_sets_completed_at(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.update_job_status(pool, "j1", "cancelled"))
        sql = conn.execute.call_args[0][0]
        assert "completed_at" in sql

    def test_update_to_other_status(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.update_job_status(pool, "j1", "waiting_for_eviction"))
        sql = conn.execute.call_args[0][0]
        assert "started_at" not in sql
        assert "completed_at" not in sql


class TestGetPendingJobs:
    def test_returns_list_of_dicts(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {"id": "j1", "model": "a", "status": "queued", "priority": 0},
            {"id": "j2", "model": "b", "status": "queued", "priority": 1},
        ]
        result = _run(queue_db.get_pending_jobs(pool))
        assert len(result) == 2
        assert result[0]["id"] == "j1"

    def test_empty_queue(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        result = _run(queue_db.get_pending_jobs(pool))
        assert result == []

    def test_respects_limit(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        _run(queue_db.get_pending_jobs(pool, limit=5))
        args = conn.fetch.call_args[0]
        assert args[1] == 5


class TestGetBatchJobs:
    def test_returns_list(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {"id": "j1", "batch_id": "b1"},
            {"id": "j2", "batch_id": "b1"},
        ]
        result = _run(queue_db.get_batch_jobs(pool, "b1"))
        assert len(result) == 2

    def test_empty_batch(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        result = _run(queue_db.get_batch_jobs(pool, "nonexistent"))
        assert result == []


class TestModelSettingsCrud:
    def test_get_model_settings_existing(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "model_name": "qwen2.5:7b",
            "do_not_evict": True,
            "evictable": False,
            "wait_for_completion": True,
            "vram_estimate_gb": 4.5,
        }
        result = _run(queue_db.get_model_settings(pool, "qwen2.5:7b"))
        assert result["do_not_evict"] is True
        assert result["evictable"] is False

    def test_get_model_settings_default(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        result = _run(queue_db.get_model_settings(pool, "unknown:7b"))
        assert result["model_name"] == "unknown:7b"
        assert result["do_not_evict"] is False
        assert result["evictable"] is True
        assert result["wait_for_completion"] is True
        assert result["vram_estimate_gb"] is None

    def test_upsert_model_settings(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.upsert_model_settings(pool, "qwen2.5:7b", do_not_evict=True))
        # Should call execute for insert + one update
        assert conn.execute.call_count == 2

    def test_upsert_multiple_fields(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.upsert_model_settings(
            pool, "qwen2.5:7b", do_not_evict=True, evictable=False))
        # insert + two field updates
        assert conn.execute.call_count == 3

    def test_upsert_with_no_fields_is_noop(self):
        pool, conn = _make_mock_pool()
        _run(queue_db.upsert_model_settings(pool, "x:7b"))
        conn.execute.assert_not_called()

    def test_get_all_model_settings(self):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {"model_name": "a", "do_not_evict": False, "evictable": True,
             "wait_for_completion": True, "vram_estimate_gb": None},
        ]
        result = _run(queue_db.get_all_model_settings(pool))
        assert len(result) == 1
        assert result[0]["model_name"] == "a"


class TestGetRateLimit:
    def test_existing_rate_limit(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "app_id": 1, "max_queue_depth": 100, "max_jobs_per_minute": 20,
        }
        result = _run(queue_db.get_rate_limit(pool, 1))
        assert result["max_queue_depth"] == 100

    def test_default_rate_limit(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        result = _run(queue_db.get_rate_limit(pool, 99))
        assert result["max_queue_depth"] == 50
        assert result["max_jobs_per_minute"] == 10


# ── Scheduler unit tests ─────────────────────────────────────────────────────

from scheduler import Scheduler


def _make_scheduler(loaded_models=None, gpu_info=None):
    """Create a Scheduler with mocked pool and ollama base."""
    pool = MagicMock()
    get_ollama = AsyncMock(return_value="http://localhost:11434")
    sched = Scheduler(pool, get_ollama)
    if loaded_models:
        sched._loaded_models = loaded_models
    if gpu_info is None:
        gpu_info = {"total": 24.0, "used": 0, "free": 24.0}
    sched._get_gpu_info = AsyncMock(return_value=gpu_info)
    return sched


class TestCheckSubmission:
    def test_model_too_large(self):
        sched = _make_scheduler(gpu_info={"total": 8.0, "used": 0, "free": 8.0})
        result = _run(sched.check_submission("llama2:70b"))
        assert result["ok"] is False
        assert result["error"] == "model_too_large"
        assert result["vram_required_gb"] == 40.0

    def test_model_already_loaded(self):
        sched = _make_scheduler(
            loaded_models={"qwen2.5:7b": {"loaded_at": 1.0, "vram_gb": 4.5}},
            gpu_info={"total": 24.0, "used": 4.5, "free": 19.5},
        )
        result = _run(sched.check_submission("qwen2.5:7b"))
        assert result["ok"] is True
        assert "error" not in result

    def test_enough_free_vram(self):
        sched = _make_scheduler(gpu_info={"total": 24.0, "used": 0, "free": 24.0})
        result = _run(sched.check_submission("qwen2.5:7b"))
        assert result["ok"] is True

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    def test_eviction_needed_and_possible(self, mock_settings):
        mock_settings.return_value = {
            "do_not_evict": False, "evictable": True, "wait_for_completion": True,
        }
        sched = _make_scheduler(
            loaded_models={
                "old:7b": {"loaded_at": 1.0, "vram_gb": 4.5, "active_jobs": 0},
            },
            gpu_info={"total": 24.0, "used": 20.0, "free": 4.0},
        )
        result = _run(sched.check_submission("qwen2.5:14b"))
        # 4.0 free + 4.5 evictable = 8.5, exactly enough for 14b
        assert result["ok"] is True
        assert result.get("warning") == "eviction_required"
        assert "old:7b" in result["evicting"]

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    def test_insufficient_vram_after_eviction(self, mock_settings):
        mock_settings.return_value = {
            "do_not_evict": True, "evictable": False, "wait_for_completion": True,
        }
        sched = _make_scheduler(
            loaded_models={
                "pinned:14b": {"loaded_at": 1.0, "vram_gb": 8.5, "active_jobs": 0},
            },
            gpu_info={"total": 24.0, "used": 20.0, "free": 4.0},
        )
        result = _run(sched.check_submission("qwen2.5:14b"))
        assert result["ok"] is False
        assert result["error"] == "insufficient_vram"
        assert result["non_evictable_gb"] == 8.5

    def test_zero_gpu_total_skips_too_large_check(self):
        """If GPU total is 0 (unknown), the model_too_large check is skipped."""
        sched = _make_scheduler(gpu_info={"total": 0, "used": 0, "free": 0})
        result = _run(sched.check_submission("llama2:70b"))
        # free (0) < needed (40), no loaded models -> insufficient_vram
        assert result["ok"] is False
        assert result["error"] == "insufficient_vram"

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    def test_mixed_evictable_and_pinned(self, mock_settings):
        async def side_effect(pool, name):
            if name == "pinned:7b":
                return {"do_not_evict": True, "evictable": False}
            return {"do_not_evict": False, "evictable": True}

        mock_settings.side_effect = side_effect
        sched = _make_scheduler(
            loaded_models={
                "pinned:7b": {"loaded_at": 1.0, "vram_gb": 4.5, "active_jobs": 0},
                "evictable:3b": {"loaded_at": 2.0, "vram_gb": 2.5, "active_jobs": 0},
            },
            gpu_info={"total": 24.0, "used": 17.0, "free": 7.0},
        )
        # Need 8.5, free=7.0, evictable=2.5 -> total 9.5 >= 8.5
        result = _run(sched.check_submission("qwen2.5:14b"))
        assert result["ok"] is True
        assert result.get("warning") == "eviction_required"
        assert "evictable:3b" in result["evicting"]
        assert "pinned:7b" not in result.get("evicting", [])

    def test_model_too_large_message_includes_sizes(self):
        sched = _make_scheduler(gpu_info={"total": 12.0, "used": 0, "free": 12.0})
        result = _run(sched.check_submission("llama2:70b"))
        assert "40.0" in result["message"]
        assert "12.0" in result["message"]


class TestSchedulerProperties:
    def test_current_job_id_default_none(self):
        sched = _make_scheduler()
        assert sched.current_job_id is None

    def test_loaded_models_returns_copy(self):
        sched = _make_scheduler(loaded_models={"a": {"vram_gb": 1}})
        models = sched.loaded_models
        models["b"] = {"vram_gb": 2}
        assert "b" not in sched.loaded_models

    def test_running_flag_default_false(self):
        sched = _make_scheduler()
        assert sched._running is False

    def test_stop_sets_running_false(self):
        sched = _make_scheduler()
        sched._running = True
        sched.stop()
        assert sched._running is False


class TestEnsureModelLoaded:
    def test_already_loaded_returns_true(self):
        sched = _make_scheduler(
            loaded_models={"qwen2.5:7b": {"loaded_at": 1.0, "vram_gb": 4.5}},
        )
        result = _run(sched._ensure_model_loaded("qwen2.5:7b"))
        assert result is True

    def test_model_too_large_returns_false(self):
        sched = _make_scheduler(gpu_info={"total": 8.0, "used": 0, "free": 8.0})
        result = _run(sched._ensure_model_loaded("llama2:70b"))
        assert result is False


# ── FastAPI route tests ───────────────────────────────────────────────────────

import auth
import main


@asynccontextmanager
async def _noop_lifespan(a):
    """No-op lifespan that sets app.state.db and scheduler to mocks."""
    a.state.db = MagicMock()
    a.state.scheduler = MagicMock(spec=Scheduler)
    a.state.scheduler.check_submission = AsyncMock(return_value={"ok": True})
    a.state.scheduler.loaded_models = {}
    a.state.scheduler.current_job_id = None
    a.state.scheduler._get_gpu_info = AsyncMock(
        return_value={"total": 24.0, "used": 0, "free": 24.0})
    yield
    a.state.db = None
    a.state.scheduler = None


_original_lifespan = main.app.router.lifespan_context


@pytest.fixture
def client():
    """TestClient with mocked lifespan, DB pool, scheduler, and admin auth."""
    main.app.router.lifespan_context = _noop_lifespan
    auth.SESSION_SECRET = "test-secret"
    auth.GITHUB_ALLOWED_USERS = {"testuser"}
    token = auth.create_session_token("testuser")
    with TestClient(main.app, raise_server_exceptions=False, cookies={auth.COOKIE_NAME: token}) as c:
        yield c
    main.app.router.lifespan_context = _original_lifespan


class TestSubmitJob:
    @patch("queue_db.get_pending_jobs", new_callable=AsyncMock, return_value=[])
    @patch("queue_db.insert_job", new_callable=AsyncMock, return_value={"id": "abc123"})
    def test_submit_returns_queued(self, mock_insert, mock_pending, client):
        resp = client.post("/api/queue/submit", json={
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["model"] == "qwen2.5:7b"
        assert "job_id" in data

    @patch("queue_db.get_pending_jobs", new_callable=AsyncMock, return_value=[])
    @patch("queue_db.insert_job", new_callable=AsyncMock, return_value={"id": "abc123"})
    def test_submit_with_eviction_warning(self, mock_insert, mock_pending, client):
        client.app.state.scheduler.check_submission = AsyncMock(return_value={
            "ok": True,
            "warning": "eviction_required",
            "message": "Will evict old:7b to free VRAM",
            "evicting": ["old:7b"],
        })
        resp = client.post("/api/queue/submit", json={
            "model": "qwen2.5:14b",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["warning"] == "Will evict old:7b to free VRAM"
        assert data["evicting"] == ["old:7b"]

    def test_submit_model_too_large_returns_422(self, client):
        client.app.state.scheduler.check_submission = AsyncMock(return_value={
            "ok": False,
            "error": "model_too_large",
            "message": "Model needs 40GB, GPU has 24GB",
            "vram_required_gb": 40.0,
            "vram_available_gb": 24.0,
        })
        resp = client.post("/api/queue/submit", json={
            "model": "llama2:70b",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 422

    def test_submit_missing_messages_returns_422(self, client):
        resp = client.post("/api/queue/submit", json={
            "model": "qwen2.5:7b",
        })
        assert resp.status_code == 422


class TestSubmitBatch:
    @patch("queue_db.insert_job", new_callable=AsyncMock, return_value={"id": "x"})
    def test_submit_batch_returns_batch_id(self, mock_insert, client):
        resp = client.post("/api/queue/submit-batch", json={
            "jobs": [
                {"model": "a:7b", "messages": [{"role": "user", "content": "1"}]},
                {"model": "b:3b", "messages": [{"role": "user", "content": "2"}]},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"].startswith("batch_")
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["status"] == "queued"
        assert data["jobs"][1]["status"] == "queued"

    def test_submit_batch_fails_if_model_too_large(self, client):
        client.app.state.scheduler.check_submission = AsyncMock(return_value={
            "ok": False,
            "error": "model_too_large",
            "message": "Too big",
            "vram_required_gb": 40.0,
            "vram_available_gb": 24.0,
        })
        resp = client.post("/api/queue/submit-batch", json={
            "jobs": [
                {"model": "huge:70b", "messages": [{"role": "user", "content": "x"}]},
            ]
        })
        assert resp.status_code == 422

    def test_submit_empty_batch(self, client):
        resp = client.post("/api/queue/submit-batch", json={"jobs": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []


class TestGetJobRoute:
    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_get_existing_job(self, mock_get, client):
        mock_get.return_value = {
            "id": "j1", "status": "completed", "model": "m:7b",
            "result": {"choices": [{"message": {"content": "hi"}}]},
            "error": None,
            "created_at": datetime.datetime(2025, 1, 1, 12, 0, 0),
            "started_at": datetime.datetime(2025, 1, 1, 12, 0, 1),
            "completed_at": datetime.datetime(2025, 1, 1, 12, 0, 5),
            "metadata": None,
        }
        resp = client.get("/api/queue/jobs/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "j1"
        assert data["status"] == "completed"
        assert data["result"]["choices"][0]["message"]["content"] == "hi"
        assert data["created_at"] is not None
        assert data["started_at"] is not None
        assert data["completed_at"] is not None

    @patch("queue_db.get_job", new_callable=AsyncMock, return_value=None)
    def test_get_missing_job_returns_404(self, mock_get, client):
        resp = client.get("/api/queue/jobs/nonexistent")
        assert resp.status_code == 404

    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_get_job_with_error(self, mock_get, client):
        mock_get.return_value = {
            "id": "j2", "status": "failed", "model": "m:7b",
            "result": None, "error": "Ollama returned 500",
            "created_at": datetime.datetime(2025, 1, 1, 12, 0, 0),
            "started_at": None, "completed_at": None, "metadata": None,
        }
        resp = client.get("/api/queue/jobs/j2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "Ollama returned 500"
        assert data["result"] is None


class TestGetBatch:
    @patch("queue_db.get_batch_jobs", new_callable=AsyncMock)
    def test_get_batch_status(self, mock_batch, client):
        mock_batch.return_value = [
            {"id": "j1", "status": "completed", "model": "m", "result": {},
             "error": None, "created_at": datetime.datetime.now(),
             "started_at": None, "completed_at": None, "metadata": None},
            {"id": "j2", "status": "queued", "model": "m", "result": None,
             "error": None, "created_at": datetime.datetime.now(),
             "started_at": None, "completed_at": None, "metadata": None},
            {"id": "j3", "status": "running", "model": "m", "result": None,
             "error": None, "created_at": datetime.datetime.now(),
             "started_at": None, "completed_at": None, "metadata": None},
        ]
        resp = client.get("/api/queue/batches/batch_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "batch_abc"
        assert data["total"] == 3
        assert data["completed"] == 1
        assert data["queued"] == 1
        assert data["running"] == 1
        assert data["failed"] == 0
        assert len(data["jobs"]) == 3

    @patch("queue_db.get_batch_jobs", new_callable=AsyncMock, return_value=[])
    def test_get_missing_batch_returns_404(self, mock_batch, client):
        resp = client.get("/api/queue/batches/nonexistent")
        assert resp.status_code == 404

    @patch("queue_db.get_batch_jobs", new_callable=AsyncMock)
    def test_batch_counts_waiting_for_eviction_as_queued(self, mock_batch, client):
        mock_batch.return_value = [
            {"id": "j1", "status": "waiting_for_eviction", "model": "m",
             "result": None, "error": None, "created_at": datetime.datetime.now(),
             "started_at": None, "completed_at": None, "metadata": None},
        ]
        resp = client.get("/api/queue/batches/batch_x")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queued"] == 1


class TestCancelJob:
    @patch("queue_db.update_job_status", new_callable=AsyncMock)
    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_cancel_queued_job(self, mock_get, mock_update, client):
        mock_get.return_value = {"id": "j1", "status": "queued", "model": "m"}
        resp = client.delete("/api/queue/jobs/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["status"] == "cancelled"
        mock_update.assert_called_once()

    @patch("queue_db.get_job", new_callable=AsyncMock, return_value=None)
    def test_cancel_missing_job_returns_404(self, mock_get, client):
        resp = client.delete("/api/queue/jobs/missing")
        assert resp.status_code == 404

    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_cancel_completed_job_returns_400(self, mock_get, client):
        mock_get.return_value = {"id": "j1", "status": "completed", "model": "m"}
        resp = client.delete("/api/queue/jobs/j1")
        assert resp.status_code == 400

    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_cancel_failed_job_returns_400(self, mock_get, client):
        mock_get.return_value = {"id": "j1", "status": "failed", "model": "m"}
        resp = client.delete("/api/queue/jobs/j1")
        assert resp.status_code == 400

    @patch("queue_db.update_job_status", new_callable=AsyncMock)
    @patch("queue_db.get_job", new_callable=AsyncMock)
    def test_cancel_running_job_succeeds(self, mock_get, mock_update, client):
        mock_get.return_value = {"id": "j1", "status": "running", "model": "m"}
        resp = client.delete("/api/queue/jobs/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"


class TestQueueStatusRoute:
    @patch("queue_db.get_pending_jobs", new_callable=AsyncMock)
    def test_queue_status_overview(self, mock_pending, client):
        mock_pending.return_value = [
            {"id": "j1", "model": "a:7b", "status": "queued"},
            {"id": "j2", "model": "b:3b", "status": "queued"},
            {"id": "j3", "model": "a:7b", "status": "queued"},
        ]
        resp = client.get("/api/queue/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue_depth"] == 3
        assert set(data["models_queued"]) == {"a:7b", "b:3b"}
        assert data["gpu_vram_total_gb"] == 24.0

    @patch("queue_db.get_pending_jobs", new_callable=AsyncMock, return_value=[])
    def test_queue_status_empty(self, mock_pending, client):
        resp = client.get("/api/queue/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue_depth"] == 0
        assert data["models_queued"] == []
        assert data["current_job"] is None


class TestModelSettingsRoutes:
    @patch("queue_db.get_all_model_settings", new_callable=AsyncMock)
    def test_list_model_settings(self, mock_all, client):
        mock_all.return_value = [
            {"model_name": "qwen2.5:7b", "do_not_evict": False,
             "evictable": True, "wait_for_completion": True, "vram_estimate_gb": 4.5},
        ]
        resp = client.get("/api/models/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["model_name"] == "qwen2.5:7b"

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    def test_get_single_model_settings(self, mock_get, client):
        mock_get.return_value = {
            "model_name": "qwen2.5:7b", "do_not_evict": False,
            "evictable": True, "wait_for_completion": True, "vram_estimate_gb": None,
        }
        resp = client.get("/api/models/qwen2.5:7b/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "qwen2.5:7b"
        assert data["do_not_evict"] is False

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    @patch("queue_db.upsert_model_settings", new_callable=AsyncMock)
    def test_update_model_settings(self, mock_upsert, mock_get, client):
        mock_get.return_value = {
            "model_name": "qwen2.5:7b", "do_not_evict": True,
            "evictable": True, "wait_for_completion": True, "vram_estimate_gb": None,
        }
        resp = client.patch("/api/models/qwen2.5:7b/settings", json={
            "do_not_evict": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["do_not_evict"] is True
        mock_upsert.assert_called_once()

    @patch("queue_db.get_model_settings", new_callable=AsyncMock)
    def test_update_model_settings_no_changes(self, mock_get, client):
        mock_get.return_value = {
            "model_name": "qwen2.5:7b", "do_not_evict": False,
            "evictable": True, "wait_for_completion": True, "vram_estimate_gb": None,
        }
        resp = client.patch("/api/models/qwen2.5:7b/settings", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "qwen2.5:7b"
