# llm-manager deploy snippets

Example Resource limits for the backend: [llm-manager-backend.resources.yaml](llm-manager-backend.resources.yaml).

## HA: API replicas + dedicated scheduler worker

Split the legacy single-process backend into API pods and scheduler worker pods:

1. **API Deployment** (`LLM_MANAGER_PROCESS=api`): runs FastAPI only. Keeps Prometheus and queue REST routes; periodically refreshes in-memory runner state (`_refresh_runners` + `_sync_idle_loaded_models_from_agents`) so `loaded_models` and queue previews stay sane without owning the advisory lock.

2. **Scheduler Deployment**: same container image as the API, **command** `python -m scheduler_worker`. Exactly one replica is active leader at a time.

**Leader backends** (`SCHEDULER_LEADER_BACKEND`):

| Value | Behaviour |
| --- | --- |
| `k8s` | `coordination.k8s.io` Lease inside the cluster (SA token); see [scheduler-rbac.yaml.example](scheduler-rbac.yaml.example) |
| `postgres` | Row in `scheduler_leader_lease`; create table once (`init_scheduler_lease_table`) |
| `none` | Always leader (**only for one replica**) |

Examples: [scheduler-deployment.yaml.example](scheduler-deployment.yaml.example), [scheduler-rbac.yaml.example](scheduler-rbac.yaml.example).

Configure the API Deployment with env `LLM_MANAGER_PROCESS=api` (and omit the scheduler Deployment if you migrate from `combined`; keep `combined` as default for backwards compatibility).

**Invalid:** `LLM_MANAGER_PROCESS=scheduler` on `main.py` — that value is refused with a hint to run `python -m scheduler_worker` instead.

### Rollout pitfalls (lessons learned)

- **`maxUnavailable: 0`** on the scheduler means the **new** pod must become **Ready** (startup + readiness probes) **before** any old pod is removed. If probes are too aggressive (`timeoutSeconds` defaults to 1s), increase them (e.g. 3–5s) in your real Deployment manifests.
- **Same image tag** on **both** `llm-manager-backend` and `llm-manager-scheduler` (CI should bump both; the worker is only a different `command`).
- **Never run `combined` API and a separate scheduler** against the same DB — you risk **two dispatch loops**. Prod API should use `LLM_MANAGER_PROCESS=api` when scheduler pods exist.
- **K8s Lease** `renewTime` / `acquireTime` must match **MicroTime** (six digits after the decimal). Older images that emitted millisecond-only strings will get **400** from the apiserver until upgraded.
- **Leadership handoff**: the dispatch loop must **fully exit** before becoming leader again; the scheduler uses `stop_and_wait()` on leadership loss so restarts are safe.
