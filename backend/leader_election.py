"""
Pluggable leader election for the queue scheduler worker.

Backends (SCHEDULER_LEADER_BACKEND):
  - k8s: coordination.k8s.io Lease (in-cluster, httpx + SA token)
  - postgres: single-row lease table, short transactional heartbeats
  - none: always leader (single-replica deployments only)

See scheduler_worker.py and README for env vars.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import signal
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import asyncpg
import httpx

logger = logging.getLogger(__name__)


async def init_scheduler_lease_table(pool: asyncpg.Pool) -> None:
    """Create Postgres lease row for SCHEDULER_LEADER_BACKEND=postgres."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduler_leader_lease (
                id smallint PRIMARY KEY CHECK (id = 1),
                holder_id text NOT NULL DEFAULT '',
                lease_until timestamptz NOT NULL DEFAULT 'epoch'::timestamptz
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO scheduler_leader_lease (id, holder_id, lease_until)
            VALUES (1, '', 'epoch'::timestamptz)
            ON CONFLICT (id) DO NOTHING
            """
        )


class LeaderElector(ABC):
    """Drive scheduler start/stop from leadership changes."""

    @abstractmethod
    async def run(
        self,
        on_leadership_gained: Callable[[], Awaitable[None]],
        on_leadership_lost: Callable[[], Awaitable[None]],
    ) -> None:
        """Block until the process should exit (SIGTERM / cancel)."""

    async def shutdown(self) -> None:
        """Optional: signal run() to exit. Default no-op."""
        pass


class NoopLeaderElector(LeaderElector):
    """Always leader. Use only when exactly one scheduler replica runs."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(
        self,
        on_leadership_gained: Callable[[], Awaitable[None]],
        on_leadership_lost: Callable[[], Awaitable[None]],
    ) -> None:
        await on_leadership_gained()
        await self._stop.wait()
        await on_leadership_lost()

    async def shutdown(self) -> None:
        self._stop.set()


class PostgresLeaderElector(LeaderElector):
    """Single-row lease: steal when expired; extend when holder matches."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        holder_id: str,
        ttl_sec: int = 30,
        tick_sec: float = 5.0,
    ) -> None:
        self.pool = pool
        self.holder_id = holder_id
        self.ttl_sec = ttl_sec
        self.tick_sec = tick_sec
        self._stop = asyncio.Event()
        self._is_leader = False

    async def shutdown(self) -> None:
        self._stop.set()

    async def _try_acquire_or_renew(self) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE scheduler_leader_lease
                SET holder_id = $1,
                    lease_until = NOW() + ($2 * interval '1 second')
                WHERE id = 1
                  AND (
                      lease_until < NOW()
                      OR holder_id = $1
                      OR holder_id = ''
                  )
                RETURNING holder_id
                """,
                self.holder_id,
                self.ttl_sec,
            )
        return row is not None and row["holder_id"] == self.holder_id

    async def _release(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE scheduler_leader_lease
                SET lease_until = NOW()
                WHERE id = 1 AND holder_id = $1
                """,
                self.holder_id,
            )

    async def run(
        self,
        on_leadership_gained: Callable[[], Awaitable[None]],
        on_leadership_lost: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            while not self._stop.is_set():
                ok = await self._try_acquire_or_renew()
                if ok:
                    if not self._is_leader:
                        self._is_leader = True
                        await on_leadership_gained()
                else:
                    if self._is_leader:
                        self._is_leader = False
                        await on_leadership_lost()
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.tick_sec
                    )
                    break
                except asyncio.TimeoutError:
                    pass
        finally:
            if self._is_leader:
                self._is_leader = False
                await on_leadership_lost()
                await self._release()


def _rfc3339_micros() -> str:
    """Kubernetes Lease renewTime / acquireTime format."""
    now = datetime.now(timezone.utc)
    s = now.strftime("%Y-%m-%dT%H:%M:%S")
    micro = f"{now.microsecond:06d}"
    return f"{s}.{micro[:3]}Z"


def _parse_rfc3339(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class K8sLeaseLeaderElector(LeaderElector):
    """Leader election via coordination.k8s.io/v1 Lease (async httpx)."""

    def __init__(
        self,
        namespace: str,
        lease_name: str,
        identity: str,
        lease_duration_sec: int = 15,
        tick_sec: float = 5.0,
    ) -> None:
        self.namespace = namespace
        self.lease_name = lease_name
        self.identity = identity
        self.lease_duration_sec = lease_duration_sec
        self.tick_sec = tick_sec
        self._stop = asyncio.Event()
        self._is_leader = False
        self._token_path = os.environ.get(
            "KUBERNETES_SERVICE_ACCOUNT_TOKEN_PATH",
            "/var/run/secrets/kubernetes.io/serviceaccount/token",
        )
        self._ca_path = os.environ.get(
            "KUBERNETES_SERVICE_ACCOUNT_CA_PATH",
            "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
        )
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self._api_base = f"https://{host}:{port}"
        self._lease_url = (
            f"{self._api_base}/apis/coordination.k8s.io/v1/namespaces/"
            f"{namespace}/leases/{lease_name}"
        )

    async def shutdown(self) -> None:
        self._stop.set()

    def _read_token(self) -> str:
        with open(self._token_path, encoding="utf-8") as f:
            return f.read().strip()

    def _lease_put_body_from_get(
        self,
        lease: dict,
        *,
        renew_time: str,
        holder_identity: Optional[str] = None,
        acquire_time: Optional[str] = None,
    ) -> dict:
        """Full object for PUT replace: copy GET response, update spec, strip read-only metadata."""
        body = copy.deepcopy(lease)
        md = body.setdefault("metadata", {})
        # Omit server-populated clutter; apiserver merges from resourceVersion + name/namespace.
        md.pop("managedFields", None)
        spec = body.setdefault("spec") or {}
        holder = holder_identity if holder_identity is not None else self.identity
        spec["holderIdentity"] = holder
        spec["leaseDurationSeconds"] = self.lease_duration_sec
        spec["renewTime"] = renew_time
        if acquire_time is not None:
            spec["acquireTime"] = acquire_time
        body["spec"] = spec
        body["apiVersion"] = lease.get("apiVersion") or "coordination.k8s.io/v1"
        body["kind"] = "Lease"
        return body

    def _lease_create_body(self, renew_time: str) -> dict:
        return {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": self.lease_name, "namespace": self.namespace},
            "spec": {
                "holderIdentity": self.identity,
                "leaseDurationSeconds": self.lease_duration_sec,
                "renewTime": renew_time,
                "acquireTime": renew_time,
            },
        }

    async def _try_acquire_or_renew(self, client: httpx.AsyncClient) -> bool:
        r = await client.get(self._lease_url)
        now_s = _rfc3339_micros()

        if r.status_code == 404:
            body = self._lease_create_body(now_s)
            cr = await client.post(
                f"{self._api_base}/apis/coordination.k8s.io/v1/namespaces/"
                f"{self.namespace}/leases",
                json=body,
            )
            if cr.status_code in (200, 201):
                return True
            logger.warning("K8s Lease create failed: %s %s", cr.status_code, cr.text[:500])
            return False

        if r.status_code != 200:
            logger.warning("K8s Lease get failed: %s %s", r.status_code, r.text[:500])
            return False

        lease = r.json()
        spec = lease.get("spec") or {}
        holder = spec.get("holderIdentity") or ""
        renew_s = spec.get("renewTime") or ""
        renew_dt = _parse_rfc3339(renew_s)
        lease_dur = int(spec.get("leaseDurationSeconds") or self.lease_duration_sec)

        is_expired = True
        if renew_dt:
            age_sec = (datetime.now(timezone.utc) - renew_dt).total_seconds()
            is_expired = age_sec > float(lease_dur)

        if holder == self.identity:
            body = self._lease_put_body_from_get(lease, renew_time=now_s)
            pr = await client.put(self._lease_url, json=body)
            if pr.status_code == 200:
                return True
            logger.warning("K8s Lease renew failed: %s %s", pr.status_code, pr.text[:400])
            return False

        if not is_expired:
            return False

        body = self._lease_put_body_from_get(
            lease,
            renew_time=now_s,
            acquire_time=now_s,
        )
        pr = await client.put(self._lease_url, json=body)
        if pr.status_code == 200:
            return True
        logger.debug("K8s Lease steal conflict: %s %s", pr.status_code, pr.text[:300])
        return False

    async def _release_lease(self, client: httpx.AsyncClient) -> None:
        r = await client.get(self._lease_url)
        if r.status_code != 200:
            return
        lease = r.json()
        spec = lease.get("spec") or {}
        if spec.get("holderIdentity") != self.identity:
            return
        now_s = _rfc3339_micros()
        body = self._lease_put_body_from_get(lease, renew_time=now_s, holder_identity="")
        await client.put(self._lease_url, json=body)

    async def run(
        self,
        on_leadership_gained: Callable[[], Awaitable[None]],
        on_leadership_lost: Callable[[], Awaitable[None]],
    ) -> None:
        async with httpx.AsyncClient(
            verify=self._ca_path,
            headers={"Authorization": f"Bearer {self._read_token()}"},
            timeout=httpx.Timeout(20.0),
        ) as client:
            try:
                while not self._stop.is_set():
                    ok = await self._try_acquire_or_renew(client)
                    if ok:
                        if not self._is_leader:
                            self._is_leader = True
                            await on_leadership_gained()
                    else:
                        if self._is_leader:
                            self._is_leader = False
                            await on_leadership_lost()
                    try:
                        await asyncio.wait_for(
                            self._stop.wait(), timeout=self.tick_sec
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
            finally:
                if self._is_leader:
                    self._is_leader = False
                    await on_leadership_lost()
                    try:
                        await self._release_lease(client)
                    except Exception:
                        logger.warning("K8s Lease release failed", exc_info=True)


def make_leader_elector(
    backend: str,
    pool: asyncpg.Pool,
) -> LeaderElector:
    b = (backend or "none").strip().lower()
    if b in ("none", "off", "disabled", ""):
        return NoopLeaderElector()
    if b in ("postgres", "pg", "sql"):
        holder = os.environ.get(
            "SCHEDULER_LEADER_IDENTITY",
            os.environ.get("HOSTNAME", "scheduler"),
        )
        ttl = int(os.environ.get("SCHEDULER_POSTGRES_LEASE_TTL_SEC", "30"))
        tick = float(os.environ.get("SCHEDULER_LEADER_TICK_SEC", "5"))
        return PostgresLeaderElector(pool, holder_id=holder, ttl_sec=ttl, tick_sec=tick)
    if b in ("k8s", "kubernetes", "kube"):
        ns_path = os.environ.get(
            "KUBERNETES_NAMESPACE_PATH",
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
        )
        try:
            with open(ns_path, encoding="utf-8") as f:
                namespace = f.read().strip()
        except OSError:
            namespace = os.environ.get("POD_NAMESPACE", "default")
        lease_name = os.environ.get("SCHEDULER_K8S_LEASE_NAME", "llm-manager-scheduler")
        identity = os.environ.get(
            "SCHEDULER_LEADER_IDENTITY",
            os.environ.get("HOSTNAME", "scheduler"),
        )
        dur = int(os.environ.get("SCHEDULER_K8S_LEASE_DURATION_SEC", "15"))
        tick = float(os.environ.get("SCHEDULER_LEADER_TICK_SEC", "5"))
        return K8sLeaseLeaderElector(
            namespace=namespace,
            lease_name=lease_name,
            identity=identity,
            lease_duration_sec=dur,
            tick_sec=tick,
        )
    raise ValueError(f"Unknown SCHEDULER_LEADER_BACKEND={backend!r}")


def install_sigterm(elector: LeaderElector, loop: asyncio.AbstractEventLoop) -> None:
    def _handler():
        asyncio.ensure_future(elector.shutdown(), loop=loop)

    try:
        loop.add_signal_handler(signal.SIGTERM, _handler)
    except NotImplementedError:
        pass
