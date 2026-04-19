"""Tests for queue_strategies.py."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from queue_strategies import (
    FifoStrategy, PriorityBatchingStrategy, make_strategy, DEFAULT_BATCH_SIZE,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _job(id_, model, priority=0):
    return {"id": id_, "model": model, "priority": priority}


# ── FifoStrategy ────────────────────────────────────────────────────────────

class TestFifo:
    def test_empty(self):
        s = FifoStrategy()
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=[])):
            assert _run(s.next_jobs(None)) == []

    def test_returns_single_job(self):
        s = FifoStrategy()
        j = _job("a", "qwen3:14b")
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=[j])):
            assert _run(s.next_jobs(None)) == [j]

    def test_never_batches(self):
        """FIFO passes limit=1 to get_pending_jobs, so only the head ever comes back."""
        s = FifoStrategy()
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=[_job("a", "m1")])) as m:
            _run(s.next_jobs(None))
            m.assert_called_once()
            assert m.call_args.kwargs.get("limit") == 1 or m.call_args.args[-1] == 1


# ── PriorityBatchingStrategy ────────────────────────────────────────────────

class TestPriorityBatching:
    def test_empty_queue(self):
        s = PriorityBatchingStrategy(batch_size=5)
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=[])):
            assert _run(s.next_jobs(None)) == []

    def test_single_job(self):
        s = PriorityBatchingStrategy(batch_size=5)
        j = _job("a", "qwen3:14b")
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=[j])):
            assert _run(s.next_jobs(None)) == [j]

    def test_batches_same_model_jobs(self):
        s = PriorityBatchingStrategy(batch_size=5)
        jobs = [
            _job("a", "qwen3:14b"),
            _job("b", "qwen3:14b"),
            _job("c", "qwen3:14b"),
            _job("d", "deepseek-r1:14b"),
            _job("e", "qwen3:14b"),  # would match but comes after a different model
        ]
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=jobs)):
            batch = _run(s.next_jobs(None))
        assert [j["id"] for j in batch] == ["a", "b", "c"]

    def test_respects_batch_size_cap(self):
        s = PriorityBatchingStrategy(batch_size=3)
        jobs = [_job(str(i), "qwen3:14b") for i in range(10)]
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=jobs)):
            batch = _run(s.next_jobs(None))
        assert len(batch) == 3

    def test_different_models_at_head_only_returns_first(self):
        s = PriorityBatchingStrategy(batch_size=5)
        jobs = [
            _job("a", "qwen3:14b"),
            _job("b", "deepseek-r1:14b"),
            _job("c", "deepseek-r1:14b"),
        ]
        with patch("queue_db.get_pending_jobs", AsyncMock(return_value=jobs)):
            batch = _run(s.next_jobs(None))
        assert [j["id"] for j in batch] == ["a"]

    def test_fetch_limit_is_larger_than_batch(self):
        """Strategy must fetch more than batch_size so it can see when the
        same-model run ends."""
        s = PriorityBatchingStrategy(batch_size=5)
        mock = AsyncMock(return_value=[])
        with patch("queue_db.get_pending_jobs", mock):
            _run(s.next_jobs(None))
        # Check that limit > batch_size
        call_kwargs = mock.call_args.kwargs
        call_args = mock.call_args.args
        limit = call_kwargs.get("limit") or (call_args[-1] if call_args else None)
        assert limit >= 10  # max(batch_size * 3, 10)


# ── make_strategy ────────────────────────────────────────────────────────────

class TestMakeStrategy:
    def test_default_is_priority_batching(self, monkeypatch):
        monkeypatch.delenv("QUEUE_STRATEGY", raising=False)
        s = make_strategy()
        assert isinstance(s, PriorityBatchingStrategy)
        assert s.batch_size == DEFAULT_BATCH_SIZE

    def test_explicit_fifo(self):
        s = make_strategy("fifo")
        assert isinstance(s, FifoStrategy)

    def test_priority_batching_alias(self):
        s = make_strategy("batching")
        assert isinstance(s, PriorityBatchingStrategy)

    def test_unknown_falls_back(self):
        s = make_strategy("does-not-exist")
        assert isinstance(s, PriorityBatchingStrategy)

    def test_env_var_respected(self, monkeypatch):
        monkeypatch.setenv("QUEUE_STRATEGY", "fifo")
        s = make_strategy()
        assert isinstance(s, FifoStrategy)

    def test_batch_size_env(self, monkeypatch):
        monkeypatch.setenv("QUEUE_STRATEGY", "priority_batching")
        monkeypatch.setenv("QUEUE_BATCH_SIZE", "3")
        s = make_strategy()
        assert isinstance(s, PriorityBatchingStrategy)
        assert s.batch_size == 3
