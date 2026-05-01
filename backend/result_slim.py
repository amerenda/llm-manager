"""Trim completion payloads before persisting to queue_jobs.result (JSONB).

The scheduler loads full agent/Ollama responses into Python; large optional
fields (notably ``logprobs``) can multiply RAM during ``json.dumps``. We keep
only what API consumers need for chat completions. Image and other shapes are
left unchanged.
"""

from __future__ import annotations

from typing import Any


def slim_stored_result(raw: Any) -> Any:
    """Return a smaller copy suitable for ``queue_jobs.result`` when possible."""
    if not isinstance(raw, dict):
        return raw
    if _is_chat_completion_payload(raw):
        return _slim_chat_completion(raw)
    return raw


def _is_chat_completion_payload(d: dict) -> bool:
    if d.get("object") == "chat.completion":
        return True
    choices = d.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    c0 = choices[0]
    return isinstance(c0, dict) and "message" in c0


def _slim_chat_completion(raw: dict) -> dict:
    out: dict[str, Any] = {}
    for key in ("id", "object", "created", "model", "system_fingerprint"):
        if key in raw:
            out[key] = raw[key]

    choices = raw.get("choices")
    if isinstance(choices, list):
        out["choices"] = [_slim_choice(c) for c in choices]

    usage = raw.get("usage")
    if isinstance(usage, dict):
        allowed = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        )
        slim_usage = {k: usage[k] for k in allowed if k in usage}
        if slim_usage:
            out["usage"] = slim_usage

    return out


def _slim_choice(c: Any) -> Any:
    if not isinstance(c, dict):
        return c
    slim: dict[str, Any] = {}
    for key in ("index", "finish_reason"):
        if key in c:
            slim[key] = c[key]
    msg = c.get("message")
    if isinstance(msg, dict):
        keep_keys = ("role", "content", "tool_calls", "name", "function_call")
        slim["message"] = {k: msg[k] for k in keep_keys if k in msg}
    return slim
