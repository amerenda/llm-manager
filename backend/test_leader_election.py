"""Unit tests for leader election helpers (mocked Postgres / K8s merge logic)."""

import asyncio

from leader_election import (
    K8sLeaseLeaderElector,
    PostgresLeaderElector,
)


def _run(coro):
    """Do not use asyncio.run() here — it mutates the default loop policy and breaks
    other tests that call asyncio.get_event_loop().run_until_complete on the main thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeConn:
    def __init__(self, fetchrow_result):
        self.fetchrow_result = fetchrow_result
        self.fetchrow_calls = []

    async def fetchrow(self, sql, holder_id, ttl_sec):
        self.fetchrow_calls.append((holder_id, ttl_sec))
        return self.fetchrow_result


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *_a):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


def test_postgres_lease_acquire_ok():
    holder = "pod-abc123"
    conn = _FakeConn({"holder_id": holder})
    elector = PostgresLeaderElector(_FakePool(conn), holder_id=holder, ttl_sec=30, tick_sec=1)

    async def job():
        assert await elector._try_acquire_or_renew()

    _run(job())
    assert len(conn.fetchrow_calls) == 1
    assert conn.fetchrow_calls[0][0] == holder
    assert conn.fetchrow_calls[0][1] == 30


def test_postgres_lease_steal_blocked():
    """UPDATE returns nothing when lease is held by another identity."""
    conn = _FakeConn(None)
    elector = PostgresLeaderElector(_FakePool(conn), holder_id="a", ttl_sec=15)

    async def job():
        assert not await elector._try_acquire_or_renew()

    _run(job())


def test_k8s_put_merge_keeps_resource_version_and_sets_spec():
    el = K8sLeaseLeaderElector(
        namespace="ns-x",
        lease_name="scheduler",
        identity="candidate",
        lease_duration_sec=40,
        tick_sec=1,
    )
    lease_from_api = {
        "apiVersion": "coordination.k8s.io/v1",
        "kind": "Lease",
        "metadata": {
            "name": "scheduler",
            "namespace": "ns-x",
            "resourceVersion": "999",
            "managedFields": [{"fake": True}],
            "creationTimestamp": "2020-01-01T00:00:00Z",
            "uid": "abc",
        },
        "spec": {
            "holderIdentity": "candidate",
            "leaseDurationSeconds": 40,
            "renewTime": "2020-01-01T00:00:00.000000Z",
            "acquireTime": "2019-12-31T00:00:00.000000Z",
        },
    }
    merged = el._lease_put_body_from_get(lease_from_api, renew_time="2026-05-03T01:02:03.004Z")

    assert merged["metadata"]["resourceVersion"] == "999"
    assert "managedFields" not in merged["metadata"]
    assert merged["spec"]["holderIdentity"] == "candidate"
    assert merged["spec"]["leaseDurationSeconds"] == 40
    assert merged["spec"]["renewTime"] == "2026-05-03T01:02:03.004Z"
    # Renew path: omit acquire_time → preserves prior acquireTime from GET
    assert merged["spec"]["acquireTime"] == "2019-12-31T00:00:00.000000Z"


def test_k8s_put_merge_steal_sets_acquire():
    el = K8sLeaseLeaderElector("ns-x", "sched", "thief", lease_duration_sec=15, tick_sec=1)
    lease = {"metadata": {"resourceVersion": "1"}, "spec": {}}
    merged = el._lease_put_body_from_get(
        lease,
        renew_time="2026-05-03T02:02:02.020Z",
        acquire_time="2026-05-03T02:02:03.030Z",
    )
    assert merged["spec"]["acquireTime"] == "2026-05-03T02:02:03.030Z"
    assert merged["spec"]["holderIdentity"] == "thief"


def test_k8s_release_clears_holder():
    el = K8sLeaseLeaderElector("ns", "l", "me", lease_duration_sec=25, tick_sec=1)
    lease = {
        "metadata": {"resourceVersion": "42"},
        "spec": {"holderIdentity": "me", "renewTime": "...", "leaseDurationSeconds": 25},
    }
    merged = el._lease_put_body_from_get(lease, renew_time="2099-01-01T00:00:00.000Z", holder_identity="")
    assert merged["spec"]["holderIdentity"] == ""
