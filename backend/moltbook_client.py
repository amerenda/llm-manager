"""Moltbook API client. All endpoints from skill.md / heartbeat.md / messaging.md."""

import logging
import httpx

API_BASE = "https://www.moltbook.com/api/v1"
logger = logging.getLogger(__name__)


class MoltbookClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{API_BASE}{path}", headers=self._headers, params=params)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, data: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{API_BASE}{path}", headers=self._headers, json=data or {})
            r.raise_for_status()
            return r.json()

    # ── Core ────────────────────────────────────────────────────────────────

    async def home(self) -> dict:
        """GET /home — everything you need in one call."""
        return await self._get("/home")

    async def feed(self, sort: str = "new", limit: int = 15) -> dict:
        return await self._get("/feed", {"sort": sort, "limit": limit})

    # ── Posts ───────────────────────────────────────────────────────────────

    async def create_post(
        self,
        submolt: str,
        title: str,
        content: str,
        challenge_answer: dict = None,
    ) -> dict:
        data = {"submolt_name": submolt, "title": title, "content": content}
        if challenge_answer:
            data["challenge_answer"] = challenge_answer
        return await self._post("/posts", data)

    async def get_comments(self, post_id: str, sort: str = "new", limit: int = 35) -> dict:
        return await self._get(f"/posts/{post_id}/comments", {"sort": sort, "limit": limit})

    async def create_comment(
        self,
        post_id: str,
        content: str,
        parent_id: str = None,
        challenge_answer: dict = None,
    ) -> dict:
        data: dict = {"content": content}
        if parent_id:
            data["parent_id"] = parent_id
        if challenge_answer:
            data["challenge_answer"] = challenge_answer
        return await self._post(f"/posts/{post_id}/comments", data)

    async def upvote_post(self, post_id: str) -> dict:
        return await self._post(f"/posts/{post_id}/upvote")

    async def upvote_comment(self, comment_id: str) -> dict:
        return await self._post(f"/comments/{comment_id}/upvote")

    async def mark_notifications_read(self, post_id: str) -> dict:
        return await self._post(f"/notifications/read-by-post/{post_id}")

    async def follow_agent(self, agent_name: str) -> dict:
        return await self._post(f"/agents/{agent_name}/follow")

    # ── DMs ────────────────────────────────────────────────────────────────

    async def dm_check(self) -> dict:
        return await self._get("/agents/dm/check")

    async def dm_requests(self) -> dict:
        return await self._get("/agents/dm/requests")

    async def dm_approve(self, conv_id: str) -> dict:
        return await self._post(f"/agents/dm/requests/{conv_id}/approve")

    async def dm_reject(self, conv_id: str, block: bool = False) -> dict:
        data = {"block": True} if block else {}
        return await self._post(f"/agents/dm/requests/{conv_id}/reject", data)

    async def dm_conversations(self) -> dict:
        return await self._get("/agents/dm/conversations")

    async def dm_read(self, conv_id: str) -> dict:
        return await self._get(f"/agents/dm/conversations/{conv_id}")

    async def dm_send(self, conv_id: str, message: str, needs_human: bool = False) -> dict:
        data: dict = {"message": message}
        if needs_human:
            data["needs_human_input"] = True
        return await self._post(f"/agents/dm/conversations/{conv_id}/send", data)

    async def dm_request(self, to: str = None, to_owner: str = None, message: str = "") -> dict:
        data: dict = {"message": message}
        if to:
            data["to"] = to
        elif to_owner:
            data["to_owner"] = to_owner
        return await self._post("/agents/dm/request", data)

    # ── Owner management ────────────────────────────────────────────────────

    async def setup_owner_email(self, email: str) -> dict:
        """POST /agents/me/setup-owner-email — sends verification email to owner."""
        return await self._post("/agents/me/setup-owner-email", {"email": email})

    # ── Registration (one-time) ─────────────────────────────────────────────

    @staticmethod
    async def register(name: str, description: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{API_BASE}/agents/register",
                json={"name": name, "description": description},
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            return r.json()
