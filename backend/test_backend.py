"""
Tests for the LLM Manager backend.
Covers gpu.py helpers, config.py reconstruction, and FastAPI endpoints.
"""
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── gpu.py unit tests ────────────────────────────────────────────────────────

from gpu import vram_for_model, _estimate_vram, check_model_fit


class TestVramForModel:
    """Test VRAM estimation for Ollama models."""

    def test_known_model_exact(self):
        assert vram_for_model("qwen2.5:7b") == 4.5

    def test_known_model_14b(self):
        assert vram_for_model("qwen2.5:14b") == 8.5

    def test_known_model_dolphin_phi(self):
        assert vram_for_model("dolphin-phi") == 2.5

    def test_known_model_with_latest_tag(self):
        # "dolphin-phi:latest" is in the table directly
        assert vram_for_model("dolphin-phi:latest") == 2.5

    def test_unknown_model_falls_back_to_estimate(self):
        # Not in MODEL_VRAM, should use _estimate_vram
        result = vram_for_model("some-custom-model:7b")
        assert result == 4.5  # 7b -> 4.5

    def test_unknown_model_no_size_hint(self):
        # No parameter count in name -> default 4.5
        result = vram_for_model("totally-unknown-model:latest")
        assert result == 4.5


class TestEstimateVram:
    """Test the _estimate_vram fallback function."""

    def test_70b(self):
        assert _estimate_vram("llama2:70b") == 40.0

    def test_34b(self):
        assert _estimate_vram("codellama:34b") == 20.0

    def test_32b(self):
        assert _estimate_vram("qwen2.5:32b") == 18.0

    def test_14b(self):
        assert _estimate_vram("model:14b") == 8.5

    def test_13b(self):
        assert _estimate_vram("llama2:13b") == 8.0

    def test_8b(self):
        assert _estimate_vram("llama3:8b") == 5.0

    def test_7b(self):
        assert _estimate_vram("mistral:7b") == 4.5

    def test_3b(self):
        assert _estimate_vram("llama3.2:3b") == 2.5

    def test_2b(self):
        assert _estimate_vram("gemma2:2b") == 2.0

    def test_1b(self):
        assert _estimate_vram("llama3.2:1b") == 1.5

    def test_unknown_default(self):
        assert _estimate_vram("mystery") == 4.5

    def test_case_insensitive(self):
        assert _estimate_vram("MyModel-70B-GGUF") == 40.0


class TestCheckModelFit:
    """Test check_model_fit with and without GPU info."""

    def test_models_fit(self):
        gpu = {"name": "RTX 4090", "vram_total_gb": 24.0, "vram_used_gb": 0, "vram_free_gb": 24.0}
        result = check_model_fit(["qwen2.5:7b", "llama3.2:3b"], gpu)
        assert result["fits_simultaneously"] is True
        assert result["total_vram_needed_gb"] == 7.0  # 4.5 + 2.5
        assert result["gpu_vram_gb"] == 24.0
        assert result["warning"] is None
        assert len(result["per_model"]) == 2

    def test_models_do_not_fit(self):
        gpu = {"name": "RTX 3060", "vram_total_gb": 12.0, "vram_used_gb": 0, "vram_free_gb": 12.0}
        result = check_model_fit(["qwen2.5:14b", "qwen2.5:7b"], gpu)
        assert result["fits_simultaneously"] is False
        assert result["total_vram_needed_gb"] == 13.0  # 8.5 + 4.5
        assert result["warning"] is not None
        assert "13.0 GB" in result["warning"]

    def test_no_gpu(self):
        result = check_model_fit(["qwen2.5:7b"], None)
        assert result["fits_simultaneously"] is False
        assert result["gpu_vram_gb"] == 0.0

    def test_empty_models(self):
        gpu = {"name": "RTX 4090", "vram_total_gb": 24.0, "vram_used_gb": 0, "vram_free_gb": 24.0}
        result = check_model_fit([], gpu)
        assert result["fits_simultaneously"] is True
        assert result["total_vram_needed_gb"] == 0
        assert result["warning"] is None


# ── config.py unit tests ─────────────────────────────────────────────────────

from config import config_from_db, state_from_db, AgentConfig, AgentState


class TestConfigFromDb:
    """Test reconstruction of AgentConfig from a DB row dict."""

    def test_full_row(self):
        row = {
            "slot": 1,
            "name": "TestBot",
            "description": "A test agent",
            "tone": "sarcastic",
            "topics": ["ai", "memes"],
            "model": "llama3:8b",
            "enabled": True,
            "api_key": "abc123",
            "registered": True,
            "claimed": True,
            "post_interval_minutes": 60,
            "active_hours_start": 9,
            "active_hours_end": 21,
            "max_post_length": 500,
            "auto_reply": False,
            "auto_like": True,
            "reply_to_own_threads": True,
            "post_jitter_pct": 30,
            "karma_throttle": True,
            "karma_throttle_threshold": 5,
            "karma_throttle_multiplier": 1.5,
            "target_submolts": ["tech", "humor"],
            "auto_dm_approve": True,
            "receive_peer_likes": True,
            "receive_peer_comments": True,
            "send_peer_likes": False,
            "send_peer_comments": False,
        }
        config = config_from_db(row)
        assert isinstance(config, AgentConfig)
        assert config.slot == 1
        assert config.model == "llama3:8b"
        assert config.enabled is True
        assert config.persona.name == "TestBot"
        assert config.persona.tone == "sarcastic"
        assert config.persona.topics == ["ai", "memes"]
        assert config.schedule.post_interval_minutes == 60
        assert config.schedule.active_hours_start == 9
        assert config.behavior.max_post_length == 500
        assert config.behavior.auto_reply is False
        assert config.behavior.karma_throttle is True
        assert config.behavior.target_submolts == ["tech", "humor"]
        assert config.behavior.send_peer_likes is False

    def test_minimal_row_uses_defaults(self):
        row = {"slot": 3}
        config = config_from_db(row)
        assert config.slot == 3
        assert config.model == "qwen2.5:7b"
        assert config.enabled is False
        assert config.persona.name == "Agent"
        assert config.schedule.post_interval_minutes == 120
        assert config.behavior.max_post_length == 280


class TestStateFromDb:
    """Test reconstruction of AgentState from a DB row dict."""

    def test_full_state(self):
        row = {
            "slot": 2,
            "karma": 42,
            "last_heartbeat": datetime.datetime(2025, 1, 1, 12, 0, 0),
            "last_post_time": 1700000000.0,
            "next_post_time": 1700003600.0,
            "pending_dm_requests": ["conv1", "conv2"],
        }
        state = state_from_db(row)
        assert isinstance(state, AgentState)
        assert state.slot == 2
        assert state.karma == 42
        assert "2025-01-01" in state.last_heartbeat
        assert state.last_post_time == 1700000000.0
        assert state.pending_dm_requests == ["conv1", "conv2"]

    def test_minimal_state_uses_defaults(self):
        row = {"slot": 5}
        state = state_from_db(row)
        assert state.slot == 5
        assert state.karma == 0
        assert state.last_heartbeat is None
        assert state.pending_dm_requests == []

    def test_string_heartbeat_passes_through(self):
        row = {"slot": 1, "last_heartbeat": "2025-06-15T10:00:00+00:00"}
        state = state_from_db(row)
        assert state.last_heartbeat == "2025-06-15T10:00:00+00:00"


# ── FastAPI endpoint tests ───────────────────────────────────────────────────

# The endpoints reference main.app.state.db directly, so we mock that object
# and also mock _get_runner_client (which queries the DB for active runners).
# We swap out the lifespan so TestClient doesn't try to connect to PostgreSQL.

from contextlib import asynccontextmanager
import main


@asynccontextmanager
async def _noop_lifespan(a):
    """No-op lifespan that sets app.state.db to a mock."""
    a.state.db = MagicMock()
    yield
    a.state.db = None


# Patch the lifespan on the real app before constructing TestClient.
_original_lifespan = main.app.router.lifespan_context


@pytest.fixture
def client():
    """TestClient with mocked lifespan and runner client."""
    main.app.router.lifespan_context = _noop_lifespan
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c
    main.app.router.lifespan_context = _original_lifespan


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["service"] == "llm-manager-backend"
        assert "node" in data
        assert data["db"] is True


class TestGpuEndpoint:
    def test_gpu_returns_zeros_when_no_runner(self, client):
        """With no real runners, the endpoint catches HTTPException and returns zeros."""
        resp = client.get("/api/gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert "vram_total_gb" in data
        assert "vram_used_gb" in data
        assert "vram_free_gb" in data
        assert "name" in data
        # No runner available -> zeros
        assert data["vram_total_gb"] == 0
        assert data["vram_used_gb"] == 0
        assert data["vram_free_gb"] == 0


class TestModelsEndpoint:
    def test_models_returns_empty_list_on_error(self, client):
        """With no real runners, the endpoint returns an empty list."""
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data == []


class TestVramCheckEndpoint:
    def test_vram_check_single_model(self, client):
        resp = client.post("/api/vram-check", json={"models": ["qwen2.5:7b"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vram_needed_gb" in data
        assert "gpu_vram_gb" in data
        assert "fits_simultaneously" in data
        assert "per_model" in data
        assert len(data["per_model"]) == 1
        assert data["per_model"][0]["model"] == "qwen2.5:7b"
        assert data["per_model"][0]["vram_gb"] == 4.5
        # No runner -> gpu_vram_gb == 0 -> fits_simultaneously is False
        assert data["fits_simultaneously"] is False

    def test_vram_check_multiple_models(self, client):
        resp = client.post(
            "/api/vram-check",
            json={"models": ["qwen2.5:7b", "llama3:8b", "gemma2:2b"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["per_model"]) == 3
        # 4.5 + 5.0 + 2.0 = 11.5
        assert data["total_vram_needed_gb"] == 11.5

    def test_vram_check_empty_models(self, client):
        resp = client.post("/api/vram-check", json={"models": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_vram_needed_gb"] == 0
        assert data["per_model"] == []
        assert data["warning"] is None

    def test_vram_check_unknown_model(self, client):
        resp = client.post("/api/vram-check", json={"models": ["custom-model:13b"]})
        assert resp.status_code == 200
        data = resp.json()
        # Falls back to _estimate_vram: 13b -> 8.0
        assert data["per_model"][0]["vram_gb"] == 8.0
