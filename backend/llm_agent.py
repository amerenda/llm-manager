"""
HTTP client for the llm-agent service.
Supports TLS with certificate pinning when a PEM cert is provided.
"""

import logging
import os
import ssl
import tempfile
from typing import Any, AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)


def _make_ssl_context(cert_pem: str) -> ssl.SSLContext:
    """Create an SSL context that trusts only the given PEM certificate."""
    ctx = ssl.create_default_context()
    # Write cert to a temp file so we can load it into the context
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
        f.write(cert_pem)
        cert_path = f.name
    try:
        ctx.load_verify_locations(cert_path)
    finally:
        os.unlink(cert_path)
    return ctx


class LLMAgentClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8090,
        psk: str = "",
        tls_cert_pem: Optional[str] = None,
    ):
        if tls_cert_pem:
            self.base_url = f"https://{host}:{port}"
            self._ssl_context = _make_ssl_context(tls_cert_pem)
        else:
            self.base_url = f"http://{host}:{port}"
            self._ssl_context = None
        self._timeout = httpx.Timeout(30.0, read=300.0)
        self._headers = {"X-Agent-PSK": psk} if psk else {}

    def _client(self, **overrides) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with TLS verify if configured."""
        kwargs: dict[str, Any] = {}
        if self._ssl_context:
            kwargs["verify"] = self._ssl_context
        kwargs.update(overrides)
        return httpx.AsyncClient(**kwargs)

    # ── Read operations ────────────────────────────────────────────────────────

    async def status(self) -> dict:
        async with self._client(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/v1/status", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def models(self) -> dict:
        async with self._client(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/v1/models", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def metrics_raw(self) -> str:
        """Return raw Prometheus text from the agent's /metrics endpoint."""
        async with self._client(timeout=self._timeout) as c:
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
            client = self._client(timeout=self._timeout)
            return client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers,
            )

        async with self._client(timeout=self._timeout) as c:
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
        async with self._client(timeout=httpx.Timeout(30.0, read=360.0)) as c:
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
        async with self._client(timeout=httpx.Timeout(30.0, read=600.0)) as c:
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

    async def load_model(self, model: str, keep_alive: int = -1) -> dict:
        """Load a model into VRAM with the given keep_alive (seconds, -1=forever)."""
        async with self._client(timeout=httpx.Timeout(30.0, read=600.0)) as c:
            r = await c.post(
                f"{self.base_url}/v1/models/load",
                json={"model": model, "keep_alive": keep_alive},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def unload_model(self, model: str) -> dict:
        """Alias for backwards compatibility."""
        return await self.unload_model_from_vram(model)

    async def _old_unload(self, model: str) -> dict:
        async with self._client(timeout=self._timeout) as c:
            r = await c.delete(
                f"{self.base_url}/v1/models/{model}",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def unload_model_from_vram(self, model: str) -> dict:
        """Unload a model from VRAM (keep on disk)."""
        async with self._client(timeout=self._timeout) as c:
            r = await c.delete(
                f"{self.base_url}/v1/models/{model}",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def delete_model(self, model: str) -> dict:
        """Delete a model from disk via the agent's delete-from-disk endpoint."""
        async with self._client(timeout=httpx.Timeout(10, read=60)) as c:
            r = await c.post(
                f"{self.base_url}/v1/models/{model}/delete-from-disk",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def restart_ollama(self) -> dict:
        """Restart the Ollama container on this runner (requires OLLAMA_CONTAINER on agent)."""
        async with self._client(timeout=httpx.Timeout(30.0)) as c:
            r = await c.post(f"{self.base_url}/v1/ollama/restart", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def get_ollama_version(self) -> dict:
        """Return running Ollama version, configured image tag, and commit hash."""
        async with self._client(timeout=httpx.Timeout(10.0)) as c:
            r = await c.get(f"{self.base_url}/v1/ollama/version", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def upgrade_ollama(self, tag: str) -> dict:
        """Pull a new Ollama image and recreate the container. Long timeout — image pull can be slow."""
        async with self._client(timeout=httpx.Timeout(30.0, read=600.0)) as c:
            r = await c.post(
                f"{self.base_url}/v1/ollama/upgrade",
                json={"tag": tag},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def get_ollama_settings(self) -> dict:
        """Read the runner's ollama.env tunables."""
        async with self._client(timeout=httpx.Timeout(10.0)) as c:
            r = await c.get(f"{self.base_url}/v1/ollama/settings", headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def put_ollama_settings(self, settings: dict) -> dict:
        """Apply ollama.env tunables on the runner. Rewrites the file + recreates
        the ollama container. Allow up to 2 min — image pull + container boot."""
        async with self._client(timeout=httpx.Timeout(10.0, read=120.0)) as c:
            r = await c.put(
                f"{self.base_url}/v1/ollama/settings",
                json={"settings": settings},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── Agent management ───────────────────────────────────────────────────────

    async def trigger_update(self, target_version: str) -> dict:
        """Tell the agent to self-update to the given version."""
        async with self._client(timeout=httpx.Timeout(10.0)) as c:
            r = await c.post(
                f"{self.base_url}/v1/update",
                json={"target_version": target_version},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── ComfyUI operations ────────────────────────────────────────────────────

    async def switch_checkpoint(self, name: str) -> dict:
        async with self._client(timeout=self._timeout) as c:
            r = await c.post(
                f"{self.base_url}/v1/comfyui/checkpoint",
                json={"name": name},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── ComfyUI lifecycle ─────────────────────────────────────────────────────

    async def start_comfyui(self) -> dict:
        async with self._client(timeout=httpx.Timeout(30.0, read=60.0)) as c:
            r = await c.post(
                f"{self.base_url}/v1/comfyui/start",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    async def stop_comfyui(self) -> dict:
        async with self._client(timeout=self._timeout) as c:
            r = await c.post(
                f"{self.base_url}/v1/comfyui/stop",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_reachable(self) -> bool:
        try:
            async with self._client(timeout=httpx.Timeout(3.0)) as c:
                r = await c.get(f"{self.base_url}/health", headers=self._headers)
                return r.status_code == 200
        except Exception:
            return False
