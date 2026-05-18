# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (from `backend/`)

```bash
uv sync --group dev          # install deps + dev deps
uv run pytest -q             # run all tests
uv run pytest test_scheduler_v2.py -v   # run a single test file
uv run uvicorn main:app --port 8081     # run locally (needs DATABASE_URL set)
```

### Frontend (from `frontend/`)

```bash
npm install
npm run dev      # dev server with HMR
npm run build    # tsc + vite build (CI does this inside Docker)
```

### Agent (from `agent/`)

```bash
docker compose --profile nvidia up -d   # start with NVIDIA GPU
docker compose --profile amd up -d      # start with AMD GPU
docker compose logs -f
```

## Architecture

Three independently deployed components:

```
GPU host (bare metal)         k3s cluster
─────────────────────         ───────────────────────────────
agent :8090 (Docker)    →     backend :8081 (Deployment, 2 replicas)
  wraps Ollama :11434          └── PostgreSQL :5432 (Longhorn PVC)
  wraps ComfyUI :8188          └── frontend nginx :80
  PSK auth on all routes
  self-registers on boot
  heartbeat every 30s
```

### Backend (`backend/`)

Flat module layout — all `.py` files are in `backend/`, no subdirectories. `main.py` wires everything together. Key modules:

- **`scheduler_v2.py`** — the core scheduler. Fast-path: if a model is already loaded on an idle runner, claims it atomically and streams directly (no DB write). Queue-path: inserts to DB, background loop dispatches batches, swaps models. PostgreSQL advisory lock (`SCHEDULER_LOCK_ID = 900001`) ensures only one pod runs the scheduler loop.
- **`queue_db.py` / `queue_models.py` / `queue_routes.py` / `queue_strategies.py`** — queue subsystem. `PriorityBatchingStrategy` (default) batches consecutive same-model jobs at the head of the queue to minimize swaps. `fifo` is the alternative.
- **`db.py`** — all DDL and DB helpers using `asyncpg` directly (no ORM). Schema is created by `init_db()` on startup.
- **`auth.py`** — GitHub OAuth for admin session (cookie). Apps use `Authorization: Bearer <api_key>`.
- **`runner_client.py`** — thin wrapper around `LLMAgentClient` for per-runner operations.
- **`cloud_providers.py`** — Anthropic cloud fallback path (detect by model name, proxy to Anthropic API).
- **`result_slim.py`** — strips bulky fields (`logprobs`, unknown extras) from completed job results before writing to PostgreSQL to reduce peak RAM and row size.
- **`leader_election.py`** — advisory lock helpers.

### Frontend (`frontend/src/`)

Vite + React 18 + TailwindCSS + TanStack Query. No Redux. `useBackend.ts` is the central data-fetching hook. nginx proxies `/api/*` and `/v1/*` to the backend container — no CORS config needed in the backend for the UI.

### Agent (`agent/main.py`)

Single FastAPI app. PSK required on all routes except `/health` and `/metrics`. Self-registers with the backend on startup, sends heartbeat every 30s. Wraps Ollama and ComfyUI — calls their local HTTP APIs and forwards responses.

## CI / Deployment

Push to `main` → CI builds changed components only (detected by path prefix: `backend/`, `frontend/`, `agent/Dockerfile|main.py|requirements`). Builds multi-arch (amd64 on ARC runners, arm64 on Mac Mini self-hosted runner), creates multi-arch manifest, then opens a deploy PR on `k3s-dean-gitops`. **The PR must be manually merged** to deploy to prod.

Agent-only changes don't create a GitOps PR — instead CI calls `PUT /api/runners/target-version` on the backend. Agents self-update on the next heartbeat.

Image names: `amerenda/llm-manager:backend-<tag>`, `:frontend-<tag>`, `:agent-<tag>`, `:agent-amd-<tag>`.

## Key Non-Obvious Behaviours

- **Backend memory**: each completed job buffers the full response in RAM before the DB write. The `result_slim.py` stripping step reduces this, but pods need **~2Gi** RAM limit — 512Mi causes OOMKill under load (exit 137 → 502s from the UI).
- **Two replicas, one scheduler**: advisory lock means only one pod runs the scheduler loop at a time. The other pod handles HTTP.
- **Runner health**: runners are considered active if heartbeat < 90s. `GET /api/runners` returns a ~7-day lookback; the scheduler only considers the live subset.
- **Model alias resolution**: aliases in `/v1/chat/completions` are resolved to their base model + parameter overrides before dispatch. `num_ctx` changes require a model reload; all other alias parameters are injected per-request.
- **UAT env**: shares the same GPU runners as prod (read-only) but has `DISABLE_SCHEDULER=true` and rejects queue submissions with 503.
- **DNS on GPU hosts**: Debian's `nsswitch.conf` default (`mdns4_minimal [NOTFOUND=return]`) breaks resolution of `amer.dev` names inside Docker. Remove the `[NOTFOUND=return]` — see README.
- **`AGENT_ADDRESS`**: must be a stable LAN IP (e.g. `http://10.100.20.19:8090`), not an mDNS name — k8s pods can't resolve `.local`/`.amer.home`.
