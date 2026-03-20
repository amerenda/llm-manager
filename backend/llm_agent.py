"""
HTTP client for the llm-agent service.
"""
import logging
from typing import Any, AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMAgentClient:
    def __init__(self, host: str = "localhost", port: int = 8090, psk: str = ""):
        self.base_url = f"http://{host}:{port}"
        self._timeout = httpx.Timeout(30.0, read=300.0)
        self._headers = {"X-Agent-PSK": psk} if psk else {}

    # ── Read operations ────────────────────────────────────────────────────────

    async def status(self) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/v1/status", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def models(self) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/v1/models", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def metrics_raw(self) -> str:
        """Return raw Prometheus text from the agent's /metrics endpoint."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/metrics", headers=self._headers)
            r.raise_for_status()
            return r.text

    # ── LLM operations ────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        POST /v1/chat/completions.
        When stream=True, returns an async httpx response that must be used as
        a context manager to stream SSE lines. Caller is responsible for
        managing the client lifecycle in that case.
        """
        body = {"model": model, "messages": messages, "stream": stream, **kwargs}

        if stream:
            # Return a streaming context — the caller must use async with
            client = httpx.AsyncClient(timeout=self._timeout)
            return client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers,
            )

        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── Image operations ───────────────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        model: str = "v1-5-pruned-emaonly.safetensors",
        n: int = 1,
        size: str = "512x512",
    ) -> dict:
        body = {"prompt": prompt, "model": model, "n": n, "size": size}
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=360.0)) as c:
            r = await c.post(
                f"{self.base_url}/v1/images/generations",
                json=body,
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── Model management ──────────────────────────────────────────────────────

    async def pull_model(self, model: str) -> AsyncGenerator[bytes, None]:
        """Stream NDJSON progress lines while pulling a model."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0)) as c:
            async with c.stream(
                "POST",
                f"{self.base_url}/v1/models/pull",
                json={"model": model},
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield line.encode() + b"\n"

    async def delete_model(self, model: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.delete(
                f"{self.base_url}/v1/models/{model}",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── ComfyUI operations ────────────────────────────────────────────────────

    async def switch_checkpoint(self, name: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{self.base_url}/v1/comfyui/checkpoint",
                json={"name": name},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as c:
                r = await c.get(f"{self.base_url}/health", headers=self._headers)
                return r.status_code == 200
        except Exception:
            return False
