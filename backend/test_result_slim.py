"""Tests for result_slim trimming of queue job payloads."""

import result_slim


def test_chat_completion_drops_logprobs():
    raw = {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "a", "bytes": list(range(10000))}]},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "extra_ollama_field": {"huge": True},
    }
    out = result_slim.slim_stored_result(raw)
    assert out["id"] == "x"
    assert len(out["choices"]) == 1
    assert "logprobs" not in out["choices"][0]
    assert out["choices"][0]["message"]["content"] == "hi"
    assert out["usage"]["total_tokens"] == 3
    assert "extra_ollama_field" not in out


def test_non_chat_payload_unchanged():
    raw = {"object": "list", "data": [{"url": "http://x"}]}
    assert result_slim.slim_stored_result(raw) == raw


def test_detects_chat_by_choices_shape():
    raw = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
    }
    out = result_slim.slim_stored_result(raw)
    assert "choices" in out
    assert out["choices"][0]["message"]["content"] == "x"
