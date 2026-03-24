"""
Cloud model providers for llm-manager.

Translates OpenAI-compatible chat requests to provider-native APIs
and back. Currently supports Anthropic (Claude).
"""

import logging
import os
import time
from enum import Enum
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


async def get_anthropic_api_key(pool=None) -> str:
    """Get Anthropic API key — DB first, then env var fallback."""
    if pool:
        try:
            from api_keys import get_api_key
            db_key = await get_api_key(pool, "anthropic")
            if db_key:
                return db_key
        except Exception:
            pass
    return ANTHROPIC_API_KEY
ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class ModelProvider(str, Enum):
    LOCAL = "local"
    ANTHROPIC = "anthropic"


# ── Model registry & auto-discovery ──────────────────────────────────────────

# Cached model list from Anthropic API
_anthropic_models_cache: list[dict] = []
_anthropic_models_cache_time: float = 0
_CACHE_TTL = 3600  # 1 hour


def detect_provider(model: str) -> ModelProvider:
    """Determine which provider handles this model."""
    if model.startswith("claude-"):
        return ModelProvider.ANTHROPIC
    return ModelProvider.LOCAL


async def get_anthropic_models(api_key: str = "") -> list[dict]:
    """Fetch available models from Anthropic API. Caches for 1 hour."""
    global _anthropic_models_cache, _anthropic_models_cache_time

    key = api_key or ANTHROPIC_API_KEY
    if not key:
        return []

    now = time.time()
    if _anthropic_models_cache and (now - _anthropic_models_cache_time) < _CACHE_TTL:
        return _anthropic_models_cache

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{ANTHROPIC_API_URL}/v1/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    models.append({
                        "id": m["id"],
                        "display_name": m.get("display_name", m["id"]),
                        "provider": "anthropic",
                        "created_at": m.get("created_at"),
                    })
                _anthropic_models_cache = models
                _anthropic_models_cache_time = now
                logger.info("Refreshed Anthropic model list: %d models", len(models))
                return models
            else:
                logger.warning("Anthropic models API returned %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Failed to fetch Anthropic models: %s", e)

    return _anthropic_models_cache  # return stale cache on error


# ── OpenAI → Anthropic format translation ────────────────────────────────────

def _translate_request(body: dict) -> dict:
    """Translate OpenAI chat completion request to Anthropic Messages API format."""
    messages = list(body.get("messages", []))

    # Extract system messages → top-level system param
    system_parts = []
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            non_system.append(msg)

    payload: dict = {
        "model": body["model"],
        "messages": non_system,
        "max_tokens": body.get("max_tokens", 4096),
    }

    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    # Pass through compatible params
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if "top_p" in body:
        payload["top_p"] = body["top_p"]
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            stop = [stop]
        payload["stop_sequences"] = stop
    if body.get("stream"):
        payload["stream"] = True

    return payload


def _translate_response(data: dict, model: str) -> dict:
    """Translate Anthropic Messages response to OpenAI chat completion format."""
    # Extract text from content blocks
    content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")

    usage = data.get("usage", {})

    return {
        "id": data.get("id", ""),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": _map_stop_reason(data.get("stop_reason")),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def _map_stop_reason(reason: Optional[str]) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
    }
    return mapping.get(reason, "stop")


# ── Anthropic API calls ──────────────────────────────────────────────────────

async def anthropic_chat(
    body: dict,
    api_key: str = "",
    stream: bool = False,
    config_overrides: Optional[dict] = None,
) -> dict | AsyncGenerator[bytes, None]:
    """
    Send a chat completion to Anthropic.

    Args:
        body: OpenAI-format request body
        api_key: Anthropic API key (falls back to env var)
        stream: Whether to stream the response
        config_overrides: Optional overrides for max_tokens, temperature, etc.
                          from cloud model config in the DB.
    """
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        raise ValueError("No Anthropic API key configured")

    payload = _translate_request(body)

    # Apply config overrides (admin-set defaults per model)
    if config_overrides:
        if "max_tokens" in config_overrides and "max_tokens" not in body:
            payload["max_tokens"] = config_overrides["max_tokens"]
        if "temperature" in config_overrides and "temperature" not in body:
            payload["temperature"] = config_overrides["temperature"]

    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    if stream:
        return _stream_anthropic(payload, headers)
    else:
        return await _call_anthropic(payload, headers)


async def _call_anthropic(payload: dict, headers: dict) -> dict:
    """Non-streaming Anthropic API call."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0)) as client:
        resp = await client.post(
            f"{ANTHROPIC_API_URL}/v1/messages",
            json=payload,
            headers=headers,
        )
        if resp.status_code != 200:
            error_body = resp.text[:500]
            logger.error("Anthropic API error %d: %s", resp.status_code, error_body)
            raise httpx.HTTPStatusError(
                f"Anthropic API error: {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        return _translate_response(resp.json(), payload["model"])


async def _stream_anthropic(payload: dict, headers: dict) -> AsyncGenerator[bytes, None]:
    """Streaming Anthropic API call, translating SSE to OpenAI format."""
    import json

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0)) as client:
        async with client.stream(
            "POST",
            f"{ANTHROPIC_API_URL}/v1/messages",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                error = await resp.aread()
                logger.error("Anthropic streaming error %d: %s", resp.status_code, error[:500])
                yield f"data: {json.dumps({'error': f'Anthropic API error: {resp.status_code}'})}\n\n".encode()
                return

            model = payload["model"]
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    return

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()

                elif event_type == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
                    chunk = {
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": _map_stop_reason(stop_reason),
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                elif event_type == "message_stop":
                    yield b"data: [DONE]\n\n"
                    return
