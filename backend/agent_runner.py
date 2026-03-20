"""Runs a single Moltbook agent as an asyncio task. Extracted from original main.py."""
import asyncio
import logging
import random
import re
import time
from datetime import datetime, timezone

import asyncpg
import httpx

import db
from config import (
    AgentConfig, AgentState, PeerDatabase, PeerPost,
    config_from_db, state_from_db,
)
from moltbook_client import MoltbookClient

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL = 30 * 60  # 30 min


class AgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        pool: asyncpg.Pool,
        ollama_base: str,
        ollama_model: str,
        psk: str = "",
    ):
        self.config = config
        self.slot = config.slot
        self.pool = pool
        self.ollama_base = ollama_base
        self.ollama_model = ollama_model
        self.client = MoltbookClient(config.api_key)
        # State is loaded lazily on first heartbeat
        self.state: AgentState = AgentState(slot=self.slot)
        self._task: asyncio.Task | None = None
        self.running = False
        # PSK stored for potential future use (e.g. agent API calls)
        self._psk = psk

    async def log(self, action: str, detail: str):
        await db.append_moltbook_activity(self.pool, self.slot, action, detail)
        logger.info("[agent-%d] [%s] %s", self.slot, action, detail)

    async def _llm(self, prompt: str, system: str | None = None) -> str:
        p = self.config.persona
        sys_prompt = system or (
            f"You are {p.name} on Moltbook, a social network for AI agents.\n"
            f"Description: {p.description}\n"
            f"Tone: {p.tone}\n"
            f"Topics: {', '.join(p.topics)}\n\n"
            "Be genuine, concise, and thoughtful. Don't be sycophantic or robotic. "
            "Write like a real community member who actually has opinions."
        )
        try:
            async with httpx.AsyncClient(timeout=60) as http:
                r = await http.post(
                    f"{self.ollama_base}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                    },
                )
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error("LLM error slot %d: %s", self.slot, e)
            return ""

    async def _solve_challenge(self, problem: str) -> str:
        answer = await self._llm(
            f"Solve this math problem. Return ONLY the numeric answer:\n{problem}",
            system="You are a precise math solver. Return only the number.",
        )
        nums = re.findall(r"\d+", answer)
        return nums[0] if nums else "0"

    async def _post_with_challenge(self, fn, *args, **kwargs) -> dict:
        try:
            result = await fn(*args, **kwargs)
            if isinstance(result, dict) and result.get("verification"):
                ch = result["verification"]
                kwargs["challenge_answer"] = {
                    "id": ch["id"],
                    "answer": await self._solve_challenge(ch.get("problem", "")),
                }
                result = await fn(*args, **kwargs)
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                try:
                    body = e.response.json()
                    if "verification" in body:
                        ch = body["verification"]
                        kwargs["challenge_answer"] = {
                            "id": ch["id"],
                            "answer": await self._solve_challenge(ch.get("problem", "")),
                        }
                        return await fn(*args, **kwargs)
                except Exception:
                    pass
            raise

    async def _load_state(self) -> None:
        """Load state from DB into self.state."""
        row = await db.get_moltbook_state(self.pool, self.slot)
        self.state = state_from_db(row)

    async def _save_state(self) -> None:
        """Persist self.state to DB."""
        await db.upsert_moltbook_state(
            self.pool,
            self.slot,
            karma=self.state.karma,
            last_heartbeat=self.state.last_heartbeat,
            last_post_time=self.state.last_post_time,
            next_post_time=self.state.next_post_time,
            pending_dm_requests=self.state.pending_dm_requests,
        )

    async def _load_peer_db(self) -> PeerDatabase:
        """Build PeerDatabase from DB tables."""
        raw_posts = await db.get_peer_posts(self.pool, self.slot)
        liked_ids = await db.get_interacted_post_ids(self.pool, self.slot, "liked")
        commented_ids = await db.get_interacted_post_ids(self.pool, self.slot, "commented")

        peers: dict[str, list[PeerPost]] = {}
        for peer_name, posts in raw_posts.items():
            peers[peer_name] = [
                PeerPost(
                    post_id=p["post_id"],
                    title=p["title"],
                    content_preview=p["content_preview"],
                    seen_at=(
                        p["seen_at"].isoformat()
                        if not isinstance(p["seen_at"], str)
                        else p["seen_at"]
                    ),
                )
                for p in posts
            ]

        return PeerDatabase(
            slot=self.slot,
            peers=peers,
            liked_post_ids=liked_ids,
            commented_post_ids=commented_ids,
        )

    async def _save_peer_db(self, peer_db: PeerDatabase) -> None:
        """Persist peer posts and interactions to DB."""
        for peer_name, posts in peer_db.peers.items():
            for pp in posts:
                await db.upsert_peer_post(
                    self.pool,
                    self.slot,
                    peer_name,
                    pp.post_id,
                    pp.title,
                    pp.content_preview,
                )

        # Prune to keep_per_peer=20
        await db.prune_peer_posts(self.pool, self.slot, keep_per_peer=20)

        # Interactions are recorded in place during operations — nothing extra to flush here

    async def run_heartbeat(self):
        # Load fresh state from DB each heartbeat
        await self._load_state()
        await self.log("heartbeat", "Starting")
        try:
            home = await self.client.home()
            self.state.karma = home.get("your_account", {}).get("karma", self.state.karma)
            self.state.last_heartbeat = datetime.now(timezone.utc).isoformat()
            await self._save_state()

            if self.config.behavior.auto_reply:
                for activity in home.get("activity_on_your_posts", []):
                    await self._handle_post_activity(activity)

            dm_data = await self.client.dm_check()
            if dm_data.get("has_activity"):
                await self._handle_dms(dm_data)

            await self._browse_and_engage()
            if self.config.behavior.reply_to_own_threads:
                await self._reply_to_own_threads()
            await self._maybe_post_new()

            # Passive: keep peer database updated from feed observations
            await self._update_peer_db()

            await self.log("heartbeat", f"Done — karma: {self.state.karma}")
        except Exception as e:
            logger.error("Heartbeat error slot %d: %s", self.slot, e)
            await self.log("error", str(e))

    async def _handle_post_activity(self, activity: dict):
        post_id = activity.get("post_id") or activity.get("id")
        if not post_id:
            return
        try:
            data = await self.client.get_comments(post_id, sort="new", limit=35)
            replied = 0
            own_name = self.config.persona.name
            for comment in data.get("comments", [])[:5]:
                author = comment.get("author", {}).get("name", "someone")
                if author == own_name:
                    continue  # never reply to own comments
                content = comment.get("content", "")
                if not content:
                    continue
                reply = await self._llm(
                    f'{author} replied to your post:\n"{content}"\n\n'
                    "Write a thoughtful reply (1-3 sentences). No filler."
                )
                if reply:
                    await self._post_with_challenge(
                        self.client.create_comment, post_id, reply,
                        parent_id=comment.get("id"),
                    )
                    replied += 1
                    await asyncio.sleep(25)
            await self.client.mark_notifications_read(post_id)
            if replied:
                await self.log("replied", f"Replied to {replied} comments on {post_id}")
        except Exception as e:
            logger.error("Post activity error: %s", e)

    async def _handle_dms(self, dm_data: dict):
        for req in dm_data.get("requests", {}).get("items", []):
            requester = req.get("from", {}).get("name", "unknown")
            preview = req.get("message_preview", "")
            cid = req["conversation_id"]
            if self.config.behavior.auto_dm_approve:
                try:
                    await self.client.dm_approve(cid)
                    await self.log("dm_approved", f"Auto-approved DM from {requester}: '{preview}'")
                except Exception as e:
                    logger.error("Auto DM approve error: %s", e)
            else:
                await self.log("dm_request_pending", f"DM from {requester}: '{preview}'")
                if cid not in self.state.pending_dm_requests:
                    self.state.pending_dm_requests.append(cid)
        await self._save_state()

    async def _browse_and_engage(self):
        try:
            feed = await self.client.feed(sort="new", limit=15)
            own_name = self.config.persona.name
            upvoted = commented = 0
            peer_db = await self._load_peer_db()
            for post in feed.get("posts", []):
                pid = post.get("id")
                if not pid:
                    continue
                post_author = post.get("author", {}).get("name", "")
                if post_author == own_name:
                    continue  # never upvote or comment on own posts
                # skip if we already liked this post via peer interaction
                if pid not in peer_db.liked_post_ids:
                    try:
                        await self.client.upvote_post(pid)
                        upvoted += 1
                    except Exception:
                        pass
                if commented < 2 and len(post.get("content", "")) > 50:
                    decision = await self._llm(
                        f'Post "{post.get("title")}" by {post.get("author", {}).get("name")}:\n'
                        f'{post.get("content", "")[:300]}\n\nComment? YES or NO.',
                    )
                    if decision.upper().startswith("YES"):
                        comment = await self._llm(
                            f'Write one thoughtful comment on this post (1-2 sentences):\n'
                            f'"{post.get("title")}"\n{post.get("content", "")[:500]}'
                        )
                        if comment:
                            try:
                                await self._post_with_challenge(self.client.create_comment, pid, comment)
                                commented += 1
                                await asyncio.sleep(25)
                            except Exception:
                                pass
            if upvoted or commented:
                await self.log("browsed", f"Upvoted {upvoted}, commented {commented}")
        except Exception as e:
            logger.error("Browse error: %s", e)

    async def _reply_to_own_threads(self):
        """Continue agent's own recent posts as threads."""
        try:
            feed = await self.client.feed(sort="new", limit=30)
            own_name = self.config.persona.name
            own_posts = [
                p for p in feed.get("posts", [])
                if p.get("author", {}).get("name") == own_name
            ][:3]  # at most 3 recent posts to consider

            for post in own_posts:
                pid = post.get("id")
                if not pid:
                    continue
                # Only thread ~30% of the time per post to avoid spam
                if random.random() > 0.3:
                    continue
                continuation = await self._llm(
                    f'You previously posted:\nTitle: "{post.get("title")}"\n'
                    f'{post.get("content", "")[:300]}\n\n'
                    f"Write a short follow-up thought to continue this thread (1-2 sentences). "
                    "Add something new — don't just restate the original."
                )
                if continuation:
                    try:
                        await self._post_with_challenge(
                            self.client.create_comment, pid, continuation
                        )
                        await self.log("thread_reply", f"Continued thread on '{post.get('title')}'")
                        await asyncio.sleep(20)
                    except Exception as e:
                        logger.error("Thread reply error: %s", e)
        except Exception as e:
            logger.error("Reply to own threads error: %s", e)

    async def _maybe_post_new(self):
        sched = self.config.schedule
        beh = self.config.behavior
        now = time.time()

        # Determine effective interval with karma throttle
        interval_secs = sched.post_interval_minutes * 60
        if beh.karma_throttle and self.state.karma < beh.karma_throttle_threshold:
            interval_secs *= beh.karma_throttle_multiplier

        # Use next_post_time for jitter-aware scheduling; seed it on first run
        if self.state.next_post_time == 0:
            jitter = 1.0 + random.uniform(-beh.post_jitter_pct / 100, beh.post_jitter_pct / 100)
            self.state.next_post_time = self.state.last_post_time + interval_secs * jitter
            await self._save_state()

        if now < self.state.next_post_time:
            return

        hour = datetime.now().hour
        if not (sched.active_hours_start <= hour < sched.active_hours_end):
            return

        # Choose submolt
        if beh.target_submolts:
            submolt = random.choice(beh.target_submolts)
        else:
            topics = self.config.persona.topics
            submolt = (topics[0] if topics else "general").lower().replace(" ", "")

        topics = self.config.persona.topics
        max_len = beh.max_post_length
        content = await self._llm(
            f"Choose one topic from {topics} and write a genuine post. "
            f"Title on first line, content below. Max {max_len} chars. No hashtags."
        )
        if not content:
            return
        lines = content.strip().splitlines()
        title = lines[0].strip().lstrip("#").strip()
        body = "\n".join(lines[1:]).strip() or title
        try:
            await self._post_with_challenge(
                self.client.create_post, submolt=submolt, title=title, content=body
            )
            self.state.last_post_time = now
            # Schedule next post with fresh jitter
            jitter = 1.0 + random.uniform(-beh.post_jitter_pct / 100, beh.post_jitter_pct / 100)
            self.state.next_post_time = now + interval_secs * jitter
            await self._save_state()
            await self.log("posted", f"New post: '{title}' → m/{submolt}")
        except Exception as e:
            logger.error("Post error: %s", e)

    async def _update_peer_db(self, peer_names: list[str] | None = None) -> None:
        """Scan feed and record posts from peer agents. If peer_names is None, tracks all non-self authors."""
        try:
            feed = await self.client.feed(sort="new", limit=30)
            own_name = self.config.persona.name
            for post in feed.get("posts", []):
                author = post.get("author", {}).get("name", "")
                if peer_names is not None:
                    if author not in peer_names:
                        continue
                else:
                    if author == own_name:
                        continue
                pid = post.get("id")
                if not pid:
                    continue
                await db.upsert_peer_post(
                    self.pool,
                    self.slot,
                    author,
                    pid,
                    post.get("title", ""),
                    post.get("content", "")[:200],
                )
            # Prune to keep last 20 per peer
            await db.prune_peer_posts(self.pool, self.slot, keep_per_peer=20)
        except Exception as e:
            logger.error("Peer DB update error slot %d: %s", self.slot, e)

    async def interact_with_peers(self, peer_names: list[str]) -> None:
        """Interact with known peer agent posts using the peer database."""
        await self.log("peer_interact", f"Engaging with peers: {', '.join(peer_names)}")
        await self._update_peer_db(peer_names)
        peer_db = await self._load_peer_db()
        beh = self.config.behavior
        own_name = self.config.persona.name
        liked = 0
        commented = 0

        for peer_name, posts in peer_db.peers.items():
            if peer_name == own_name:
                continue
            for pp in posts[-5:]:  # most recent 5 posts per peer
                pid = pp.post_id

                if beh.send_peer_likes and pid not in peer_db.liked_post_ids:
                    try:
                        await self.client.upvote_post(pid)
                        await db.record_interaction(self.pool, self.slot, pid, "liked")
                        peer_db.liked_post_ids.append(pid)
                        liked += 1
                    except Exception:
                        pass

                if beh.send_peer_comments and pid not in peer_db.commented_post_ids:
                    comment = await self._llm(
                        f'Your peer agent {peer_name} posted:\n'
                        f'"{pp.title}"\n{pp.content_preview}\n\n'
                        "Write a thoughtful comment (1-2 sentences)."
                    )
                    if comment:
                        try:
                            await self._post_with_challenge(self.client.create_comment, pid, comment)
                            await db.record_interaction(self.pool, self.slot, pid, "commented")
                            peer_db.commented_post_ids.append(pid)
                            commented += 1
                            await asyncio.sleep(20)
                        except Exception as e:
                            logger.error("Peer comment error: %s", e)

        await self.log("peer_interact", f"Liked {liked}, commented {commented} peer posts")

    async def _loop(self):
        self.running = True
        try:
            while True:
                await self.run_heartbeat()
                await asyncio.sleep(HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False

    def start(self):
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("Agent %d started", self.slot)

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Agent %d stopped", self.slot)
