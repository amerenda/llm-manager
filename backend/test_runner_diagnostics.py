from datetime import datetime, timedelta, timezone

from main import _runner_offline_diagnostics


def test_runner_offline_diagnostics_reports_likely_causes():
    last_seen = datetime.now(timezone.utc) - timedelta(seconds=95)
    caps = {
        "config_diagnostics": {
            "agent_address_configured": False,
            "backend_url_configured": False,
            "compose_dir_configured": True,
            "ollama_defaults_present": False,
        }
    }
    out = _runner_offline_diagnostics(last_seen, caps)
    assert out["age_seconds"] is not None
    assert "AGENT_ADDRESS missing or stale" in out["likely_causes"]
    assert "BACKEND_URL missing" in out["likely_causes"]
    assert "ollama.env missing from compose directory" in out["likely_causes"]


def test_runner_offline_diagnostics_uses_empty_defaults():
    out = _runner_offline_diagnostics(None, {})
    assert out["age_seconds"] is None
    assert out["likely_causes"] == []
