# llm-manager

Centralized GPU resource manager for a k3s home lab. Routes all AI inference
workloads (Ollama text models, ComfyUI image generation) through a stateless
k8s backend. GPU nodes self-register; the backend tracks them in PostgreSQL and
proxies requests with load balancing.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GPU Host (murderbot, bare metal, 10.100.20.19)                          │
│                                                                          │
│  Ollama :11434 ──┐                                                       │
│  ComfyUI :8188 ──┤── llm-agent :8090 (docker compose)                   │
│                  │   - PSK-authenticated REST API                         │
│                  │   - Self-registers with backend on startup             │
│                  │   - Sends heartbeat every 30s                          │
└──────────────────┼───────────────────────────────────────────────────────┘
                   │
              HTTPS (TLS via cert-manager)
              https://llm-manager-backend.amer.dev
                   │
┌──────────────────┼───────────────────────────────────────────────────────┐
│  k3s cluster     ▼                                                       │
│                                                                          │
│  Traefik (10.100.20.203:443) ──► llm-manager-backend :8081               │
│      ClusterIP Service                                                   │
│                                                                          │
│  llm-manager-backend (Deployment, 1 replica)                             │
│  ├── Runner registry (active GPU nodes, heartbeat < 90s)                 │
│  ├── Moltbook agent runner (slots 1–6, DB-backed state)                  │
│  ├── OpenAI-compatible proxy: /v1/chat/completions, /v1/images/...       │
│  ├── App registry (API key auth for external apps)                       │
│  └── Prometheus metrics                                                  │
│                                                                          │
│  PostgreSQL :5432 (Longhorn-backed PVC, daily S3 backup)                 │
│  llm-manager UI  :80  → https://llm-manager.amer.dev (tailscale-only)   │
└──────────────────────────────────────────────────────────────────────────┘
```

### External app integration

```
moltbook-frontend → moltbook-controller → llm-manager-backend (https)
home-assistant    → /v1/chat/completions (API key)
voice assistants  → /v1/chat/completions (API key)
```

## Components

### `agent/` — LLM Agent (runs on each GPU host)

Docker Compose service deployed on the GPU machine alongside Ollama and ComfyUI.

- **Port:** 8090 (PSK-authenticated on all endpoints except `/health`, `/metrics`)
- **Wraps:** Ollama (text inference) + ComfyUI (image generation)
- **Auth:** All requests require `X-Agent-PSK` header matching `LLM_MANAGER_AGENT_PSK`

See [agent/README.md](agent/README.md) for installation.

### `backend/` — Backend (k8s Deployment)

Stateless service running in k8s. Discovers GPU runners from DB (heartbeat < 90s).

- **Port:** 8081
- **State:** PostgreSQL only — no local files, no volumes
- **Replicas:** 2 (PostgreSQL advisory lock ensures only one pod runs the scheduler)

### `ui/` — React dashboard

Vite+React SPA served by nginx in k8s. Proxies `/api/*` to the backend container.
Accessible at `https://llm-manager.amer.dev` — tailscale-only middleware applied.

---

## Networking & DNS

### How agents reach the backend

All agent → backend traffic goes over **HTTPS** via `llm-manager-backend.amer.dev`.
This hostname is created automatically by `external-dns` from the backend `Ingress`
annotation (`external-dns.alpha.kubernetes.io/hostname`). DNS A records point to the
Traefik load balancer IP (10.100.20.203).

```
agent → https://llm-manager-backend.amer.dev → Traefik → ClusterIP → backend pod
```

No NodePort is exposed. The backend ingress does **not** apply the tailscale-only
middleware — PSK authentication (header `X-Agent-PSK`) protects all runner endpoints.

### DNS resolution requirement on GPU hosts

`amer.dev` records are hosted on **DigitalOcean DNS** (public authoritative DNS).
However, Debian's default `nsswitch.conf` has:

```
hosts: files myhostname mdns4_minimal [NOTFOUND=return] dns
```

The `[NOTFOUND=return]` causes `mdns4_minimal` to short-circuit lookups before the
real DNS server is consulted. This prevents `llm-manager-backend.amer.dev` from
resolving on the GPU host and inside Docker containers.

**Fix** — edit `/etc/nsswitch.conf` on each GPU host:

```diff
-hosts: files myhostname mdns4_minimal [NOTFOUND=return] dns
+hosts: files myhostname mdns4_minimal dns
```

No daemon restart needed. Verify with: `getent hosts llm-manager-backend.amer.dev`

### Agent `AGENT_ADDRESS` must use a stable IP

The backend pod needs to reach back to the agent for proxied requests. Use the
host's stable LAN IP, not an mDNS `.amer.home` or `.local` name — those won't
resolve from inside k8s pods.

```bash
# Good
AGENT_ADDRESS=http://10.100.20.19:8090

# Bad — mDNS names don't resolve from k8s pods
AGENT_ADDRESS=http://murderbot.amer.home:8090
```

### Traffic path summary

| Direction | Path | Auth | TLS |
|-----------|------|------|-----|
| agent → backend (register/heartbeat) | `https://llm-manager-backend.amer.dev` | `X-Agent-PSK` header | Yes (cert-manager) |
| backend → agent (proxy inference) | `http://<agent-ip>:8090` | `X-Agent-PSK` header | No (LAN) |
| browser → UI | `https://llm-manager.amer.dev` | Tailscale IP allowlist | Yes |
| apps → backend API | `https://llm-manager-backend.amer.dev` | `Authorization: Bearer <api_key>` | Yes |

> **Note:** Backend → agent traffic is unencrypted (LAN only, PSK still validates identity).
> If the agent host is on a different network segment, consider a tunnel or adding TLS
> to the agent container.

---

## Agent Installation

### Prerequisites

- Docker + Docker Compose with NVIDIA runtime
- Ollama running on the host (not in Docker) at port 11434
- LAN access to the k3s cluster

### Setup

```bash
cd agent/
cp .env.example .env
# Edit .env — set LLM_MANAGER_AGENT_PSK, BACKEND_URL, AGENT_ADDRESS
nano .env

docker compose up -d
docker compose logs -f   # confirm "Registered with backend as runner_id=N"
```

### `.env` reference

```bash
# PSK — must match llm-manager-agent-psk Bitwarden secret / k8s ExternalSecret
LLM_MANAGER_AGENT_PSK=your-psk-here

# Backend URL — always use HTTPS (PSK would be in cleartext over HTTP)
BACKEND_URL=https://llm-manager-backend.amer.dev

# This host's stable LAN IP (mDNS names won't resolve from k8s pods)
AGENT_ADDRESS=http://10.100.20.19:8090
```

### Auto-start on boot

```bash
bash install.sh   # installs systemd user service
```

---

## Backend API

### Runner Management

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/runners/register` | PSK | Agent self-registration on startup |
| POST | `/api/runners/heartbeat` | PSK | Agent heartbeat (every 30s) |
| GET  | `/api/runners` | PSK | List active runners (last 90s) |
| PATCH | `/api/runners/{id}` | Bearer | Update runner config (e.g. `pinned_model`) |

**Runner pinning** — set `pinned_model` to dedicate a runner exclusively to one model:

```bash
# Pin qwen3:14b to murderbot (runner_id=1)
curl -X PATCH https://llm-manager-backend.amer.dev/api/runners/1 \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"pinned_model": "qwen3:14b"}'

# Unpin
curl -X PATCH https://llm-manager-backend.amer.dev/api/runners/1 \
  -d '{"pinned_model": ""}'
```

**Pinning guarantees (simplified scheduler):**
- The pinned model can never be evicted from that runner
- No other model can ever be scheduled on that runner
- The fast-path will only use the pinned runner for the pinned model; if it's busy, requests wait in queue rather than going to another runner

### LLM Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/llm/status` | GPU stats, loaded models, ComfyUI state |
| GET | `/api/llm/models` | List all models (text + image) |
| POST | `/api/llm/models/pull` | Pull an Ollama model |
| POST | `/api/llm/models/load` | Load model into VRAM (keep_alive=-1) |
| POST | `/api/llm/models/unload` | Unload model from VRAM |
| DELETE | `/api/llm/models/{model}` | Unload model from VRAM |
| POST | `/api/llm/comfyui/checkpoint` | Switch ComfyUI checkpoint |
| GET | `/api/llm/checkpoints` | List available checkpoints |

All LLM endpoints accept `?runner_id=N` to target a specific GPU node.

### Profiles

Profiles define named configurations of models/checkpoints to load on GPU runners.
Each profile supports safe/unsafe model variants, instance counts, and advanced
parameters (context window, temperature, etc).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/profiles` | List all profiles |
| POST | `/api/profiles` | Create profile |
| GET | `/api/profiles/{id}` | Get profile with entries |
| PATCH | `/api/profiles/{id}` | Update profile (name, unsafe_enabled) |
| DELETE | `/api/profiles/{id}` | Delete profile (not default) |
| POST | `/api/profiles/{id}/models` | Add model entry |
| PATCH | `/api/profiles/{id}/models/{eid}` | Update model entry |
| DELETE | `/api/profiles/{id}/models/{eid}` | Remove model entry |
| POST | `/api/profiles/{id}/images` | Add image checkpoint entry |
| DELETE | `/api/profiles/{id}/images/{eid}` | Remove image entry |
| POST | `/api/profiles/{id}/activate` | Activate profile on runner |
| POST | `/api/profiles/{id}/deactivate` | Return runner to ad-hoc mode |
| GET | `/api/profiles/activations` | List all runner-profile mappings |
| GET | `/api/profiles/list` | App-authenticated profile discovery |

### Model Aliases

Aliases map a short name to a base model with optional parameter overrides (system prompt, temperature, num_ctx, etc.). Parameters are injected per-request at inference time — no model reload needed when you change them.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models/aliases` | List all aliases |
| POST | `/api/models/aliases` | Create alias |
| GET | `/api/models/aliases/{alias}` | Get alias config |
| PATCH | `/api/models/aliases/{alias}` | Update alias parameters |
| DELETE | `/api/models/aliases/{alias}` | Delete alias |

Apps can use an alias name anywhere a model name is accepted (e.g. `/v1/chat/completions`, queue submit). The backend resolves the alias to its base model and merges parameters before dispatching.

**Note:** `num_ctx` changes require a model reload (Ollama limitation). All other parameters (temperature, system prompt, top_p, etc.) are applied without any reload.

### OpenAI-Compatible Inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming supported) |
| POST | `/v1/images/generations` | Image generation via ComfyUI |

Apps authenticate with `Authorization: Bearer <api_key>` (get key from `/api/apps/register`).

### App Registry & Auto-Discovery

Apps can register manually via the UI or auto-discover via the discovery endpoint.
Auto-discovery requires a shared registration secret (Bitwarden: `llm-manager-registration-secret`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/apps` | List registered apps |
| POST | `/api/apps/register` | Manual register → returns `api_key` |
| POST | `/api/apps/discover` | Auto-discovery (registration secret required) |
| POST | `/api/apps/{id}/approve` | Approve pending app → pushes key to app |
| PATCH | `/api/apps/{id}/permissions` | Toggle profile switching permission |
| POST | `/api/apps/heartbeat` | Update app `last_seen` (Bearer auth) |
| DELETE | `/api/apps/{api_key}` | Deregister app |

**Auto-discovery flow:**

1. App starts with `LLM_MANAGER_URL` and `LLM_MANAGER_REGISTRATION_SECRET` env vars
2. App calls `POST /api/apps/discover` with `{name, base_url, registration_secret}`
3. First time: app appears as "pending" in llm-manager UI
4. Admin approves in UI → llm-manager pushes API key to `{base_url}/.well-known/llm-manager/register`
5. On subsequent restarts, discover returns `{status: "approved", api_key: "..."}` — app holds key in memory
6. App uses key for heartbeats and API calls (fully stateless)

### Moltbook Agents

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/agents` | List agent slots 1–6 |
| PATCH | `/api/agents/{slot}` | Update agent config |
| POST | `/api/agents/{slot}/start` | Start agent |
| POST | `/api/agents/{slot}/stop` | Stop agent |
| GET | `/api/agents/{slot}/activity` | Recent activity log |
| POST | `/api/agents/{slot}/register` | Register with moltbook |

### Metrics

| Path | Description |
|------|-------------|
| GET `/metrics` | Prometheus — backend + forwarded agent metrics |
| GET `/health` | Liveness/readiness probe |

---

## Environment Variables

### Agent (`agent/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MANAGER_AGENT_PSK` | *(required)* | Shared PSK — must match backend |
| `BACKEND_URL` | *(required)* | Backend HTTPS URL for self-registration |
| `AGENT_ADDRESS` | *(required)* | This host's address (stable LAN IP) |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama URL |
| `COMFYUI_URL` | `http://host.docker.internal:8188` | ComfyUI URL |
| `COMFYUI_OUTPUT_DIR` | `/outputs` | Output directory (in container) |

### Backend (`gitops/backend/deployment.yaml`)

| Variable | Source | Description |
|----------|--------|-------------|
| `DATABASE_URL` | ExternalSecret `postgres-credentials` | PostgreSQL DSN |
| `LLM_MANAGER_AGENT_PSK` | ExternalSecret `agent-psk` | PSK for runner auth |
| `LLM_MANAGER_REGISTRATION_SECRET` | ExternalSecret `registration-secret` | Shared secret for app auto-discovery |
| `GITHUB_CLIENT_ID` | ExternalSecret `github-oauth` | GitHub OAuth app client ID |
| `GITHUB_CLIENT_SECRET` | ExternalSecret `github-oauth` | GitHub OAuth app client secret |
| `SESSION_SECRET` | ExternalSecret `session-secret` | JWT session signing secret |
| `GITHUB_ALLOWED_USERS` | Deployment env | Comma-separated GitHub usernames for admin access |
| `API_KEY_ENCRYPTION_KEY` | ExternalSecret `api-key-encryption-key` | Fernet key for encrypting stored API keys |
| `DISABLE_SCHEDULER` | Deployment env | Set `true` to disable the job scheduler (UAT) |
| `UAT_TEST_RUNNER` | Deployment env | Runner name for UAT connectivity tests |
| `UAT_TEST_MODEL` | Deployment env | Model name for UAT connectivity tests |
| `SIMPLIFIED_SCHEDULER` | Deployment env | Set `1` to use the v2 simplified scheduler (recommended) |
| `QUEUE_STRATEGY` | Deployment env | `priority_batching` (default) or `fifo` — controls job dispatch order |
| `QUEUE_BATCH_SIZE` | Deployment env | Max same-model jobs to dispatch together (default: 5, priority_batching only) |

---

## UAT Environment

UAT runs alongside prod with a separate database, disabled scheduler, and its own UI.

| Component | Prod | UAT |
|-----------|------|-----|
| Backend | `llm-manager-backend:8081` | `llm-manager-backend-uat:8081` |
| UI | `llm-manager.amer.dev` | `llm-manager-uat.amer.dev` |
| Database | `llmmanager` | `llm_manager_uat` |
| Scheduler | Enabled (advisory lock) | Disabled (`DISABLE_SCHEDULER=true`) |
| Queue submissions | Accepted | Rejected (503) |

UAT shares the same GPU runners as prod (read-only access for GPU stats and model lists) but cannot load/evict models or process queue jobs.

### UAT Test Endpoint

`POST /api/uat/test-model` sends a tiny prompt to a configured runner/model to verify connectivity. Requires `UAT_TEST_RUNNER` and `UAT_TEST_MODEL` env vars. Returns the model response and eval timing.

### Resetting UAT Database

A k8s Job wipes and seeds the UAT database with test data. The seed SQL has a safety check that aborts if the database name doesn't contain "uat".

```bash
# Delete previous job run (if any), then create new one
kubectl delete job uat-db-reset -n llm-manager --ignore-not-found
kubectl apply -f k3s-dean-gitops/apps/llm-manager/backend-uat/jobs/reset-db-job.yaml
kubectl logs -n llm-manager -l app=llm-manager-uat-db-reset -f
```

Seed data includes:
- 3 profiles (Default, Creative, Safe Only) with model entries
- 4 model safety tags (uncensored, dolphin, abliterated patterns)
- 2 runners (murderbot, archbox)
- 3 apps (ecdysis, home-assistant, dev-notebook) with API keys, rate limits, and model restrictions

---

## Database

PostgreSQL 16 in the `llm-manager` namespace. Schema is created automatically by
`init_db()` on first backend startup.

**Storage:** Longhorn PVC (`postgres-data`, 10Gi, 3 replicas)

**Backup:** Daily at 2am to `s3://amerenda-backups@us/k3s/dean` via Longhorn
recurring job. The PVC has `recurring-job-group.longhorn.io/default: enabled`.

**Disaster recovery:** If the PVC is lost, restore from the last Longhorn backup.
All runner registrations are ephemeral (agents re-register on restart). Moltbook
agent configs, state, conversation history, and peer data are lost without a backup.

---

## Secrets

All secrets are in Bitwarden and synced via External Secrets Operator:

| Bitwarden key | k8s Secret | Field | Used by |
|--------------|-----------|-------|---------|
| `llm-manager-agent-psk` | `agent-psk` | `psk` | backend deployment, agent `.env` |
| `llm-manager-registration-secret` | `registration-secret` | `secret` | backend deployment, client apps |
| `llm-manager-postgres-password` | `postgres-credentials` | `postgres-password` | postgres deployment |
| `llm-manager-postgres-url` | `postgres-credentials` | `postgres-url` | backend deployment |

---

## GitOps

Manifests live in `gitops/`. ArgoCD (via `root-app.yaml` in `amerenda-k3s`) applies
them with `directory.recurse: true`.

```
gitops/
├── namespace.yaml
├── middleware.yaml          # tailscale-only IP allowlist
├── ingress.yaml             # UI: llm-manager.amer.dev (tailscale-only)
├── servicemonitor.yaml      # Prometheus scrape config
├── backend/
│   ├── deployment.yaml
│   ├── service.yaml         # ClusterIP :8081
│   ├── ingress.yaml         # Backend: llm-manager-backend.amer.dev (no tailscale, PSK auth)
│   └── externalsecret.yaml  # agent-psk
├── postgres/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── pvc.yaml             # Longhorn, 10Gi
│   └── externalsecret.yaml  # postgres-credentials
└── ui/
    ├── deployment.yaml
    └── service.yaml
```

CI (`.github/workflows/build.yaml`) builds and pushes Docker images on every push
to `main`, then updates the image tags in `gitops/backend/deployment.yaml` and
`gitops/ui/deployment.yaml` via a commit.

---

## Scheduler

### Simplified Scheduler v2 (recommended)

Enable with `SIMPLIFIED_SCHEDULER=1`. Designed around one-model-per-GPU semantics:

- **Fast-path** — when the requested model is already loaded on an idle runner, the request bypasses the queue entirely. The scheduler atomically claims the runner, proxies directly to Ollama with true streaming (token-by-token), then releases. Zero DB overhead.
- **Queue path** — when the model is not loaded (swap required), the request enters the DB queue. The scheduler dispatches a batch, swaps the model, runs the jobs, then checks the queue again.
- **Batching** — `PriorityBatchingStrategy` (default) dequeues up to `QUEUE_BATCH_SIZE` consecutive same-model jobs from the head of the priority-ordered queue, minimizing swap frequency. Switch to `fifo` via `QUEUE_STRATEGY=fifo` for strict single-job dispatch.
- **Runner restrictions** — if an app has `allowed_runner_ids` configured, that restriction is enforced on both the fast-path and the queue path. Batching will not mix jobs with different runner restriction sets.
- **Runner pinning** — pinned runners are dedicated exclusively to their pinned model. A pinned runner will never accept any other model. A pinned model can never be evicted from its runner.

```
request → fast-path check → model loaded + runner idle?
              yes → claim runner → stream → release
              no  → insert into queue DB
                      ↓ scheduler loop
                    pick runner → swap if needed → run batch → mark complete
```

### Legacy Scheduler (v1)

The default scheduler (no `SIMPLIFIED_SCHEDULER` flag) supports multi-model concurrent loading and VRAM-based eviction. It batches jobs by model, handles eviction policies, and supports `do_not_evict` and `evictable` flags on model settings. Use this if running multiple models concurrently on a single GPU.

## Queue System

The queue system provides async job submission with VRAM-aware scheduling. Instead
of sending inference requests directly through `/v1/chat/completions`, apps can
submit jobs to a queue. The scheduler batches jobs by model to minimize model
swaps, automatically loads/unloads models, and manages VRAM eviction.

### Queue API

All queue endpoints are under `/api/queue`. Authenticate with `Authorization: Bearer <api_key>`.

#### Job Submission

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/queue/submit` | Submit a single inference job |
| POST | `/api/queue/submit-batch` | Submit multiple jobs as a batch |

**Single job request:**

```json
{
  "model": "llama3.2:latest",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 512,
  "metadata": {"app_context": "any passthrough data"}
}
```

**Response:**

```json
{
  "job_id": "abc123def456",
  "status": "queued",
  "model": "llama3.2:latest",
  "position": 3,
  "warning": "Will evict qwen2:7b to free VRAM for llama3.2:latest",
  "evicting": ["qwen2:7b"]
}
```

If the model is too large for the GPU or cannot fit even after eviction, the
submission is rejected with HTTP 422.

#### Job Status & Results

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/queue/jobs/{job_id}` | Get job status and result |
| GET | `/api/queue/jobs/{job_id}/wait` | SSE stream -- blocks until job completes |
| GET | `/api/queue/batches/{batch_id}` | Get batch status with all job results |
| GET | `/api/queue/batches/{batch_id}/wait` | SSE stream -- blocks until all batch jobs complete |
| DELETE | `/api/queue/jobs/{job_id}` | Cancel a queued or running job |
| GET | `/api/queue/status` | Queue overview (depth, loaded models, VRAM) |

**Job statuses:** `queued` -> `waiting_for_eviction` -> `running` -> `completed` | `failed` | `cancelled`

#### Queue Overview

`GET /api/queue/status` returns:

```json
{
  "queue_depth": 5,
  "models_queued": ["llama3.2:latest", "qwen2:7b"],
  "models_loaded": ["llama3.2:latest"],
  "current_job": "abc123def456",
  "gpu_vram_total_gb": 24.0,
  "gpu_vram_used_gb": 14.2,
  "gpu_vram_free_gb": 9.8
}
```

### VRAM Management

The scheduler handles model loading and VRAM automatically:

1. **Batching** -- Jobs are grouped by model. All jobs for an already-loaded model
   run first, avoiding unnecessary swaps.
2. **Auto-loading** -- If a job's model is not loaded, the scheduler loads it via Ollama.
3. **Eviction** -- When VRAM is insufficient, the scheduler evicts models using these rules:
   - Models marked `do_not_evict` or `evictable: false` are never evicted
   - Idle models (no active jobs) are evicted before busy ones
   - Among idle models, the oldest-loaded is evicted first
   - If `wait_for_completion` is set (default), the scheduler waits up to 5 minutes
     for active jobs to finish before evicting
4. **Pre-check** -- On submission, the scheduler validates that the model can fit
   (either immediately or after eviction) and returns a warning if eviction is needed.

### Model Settings (Legacy Scheduler)

Per-model eviction behavior for the legacy v1 scheduler. Not used by the simplified v2 scheduler — use runner pinning (`PATCH /api/runners/{id}`) instead.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models/settings` | List all model settings |
| GET | `/api/models/{model_name}/settings` | Get settings for a model |
| PATCH | `/api/models/{model_name}/settings` | Update model settings |

**Settings fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `do_not_evict` | `false` | Prevent eviction (legacy scheduler only) |
| `evictable` | `true` | Whether the scheduler can evict this model (legacy only) |
| `wait_for_completion` | `true` | Wait for active jobs before evicting (legacy only) |
| `vram_estimate_gb` | *(auto)* | Manual VRAM override for scheduling |

### Per-Runner Model Parameters

Override inference parameters (system prompt, temperature, num_ctx, etc.) for a specific model on a specific runner. Applied on both fast-path and queue-path requests.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models/runner-params` | List all runner-model overrides |
| POST | `/api/models/runner-params` | Create override |
| PATCH | `/api/models/runner-params/{id}` | Update override |
| DELETE | `/api/models/runner-params/{id}` | Remove override |

### App Integration

Apps integrate with the queue using their existing Bearer token from app registration:

```python
import httpx

API = "https://llm-manager-backend.amer.dev"
KEY = "your-api-key"
headers = {"Authorization": f"Bearer {KEY}"}

# Submit a job
resp = httpx.post(f"{API}/api/queue/submit", json={
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": "Summarize this text..."}],
}, headers=headers)
job_id = resp.json()["job_id"]

# Poll for result
result = httpx.get(f"{API}/api/queue/jobs/{job_id}", headers=headers)
# Or use SSE to wait: GET /api/queue/jobs/{job_id}/wait
```

Rate limits are enforced per-app: max queue depth and max jobs per minute. Exceeding
limits returns HTTP 429.

---

## Prometheus Metrics

The backend exposes metrics at `GET /metrics`. The `servicemonitor.yaml` in GitOps
configures Prometheus scraping.

### Scheduler v2 Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_queue_depth` | Gauge | — | Jobs currently in `queued` state |
| `llm_queue_active_jobs` | Gauge | — | Jobs currently in `running` state |
| `llm_backend_active_runners` | Gauge | — | Runners with heartbeat < 90s |
| `llm_backend_runner_last_seen_seconds` | Gauge | `runner` | Seconds since last heartbeat |
| `llm_queue_jobs_submitted_total` | Counter | `model`, `app` | Total jobs submitted to queue |
| `llm_scheduler_v2_jobs_completed_total` | Counter | `status` | Jobs completed (status: completed/failed) |
| `llm_scheduler_v2_job_wait_seconds` | Histogram | — | Time from submit to dispatch |
| `llm_scheduler_v2_job_wait_by_app_seconds` | Histogram | `app` | Wait time broken down by app |
| `llm_scheduler_v2_model_swap_total` | Counter | `runner`, `from_model`, `to_model` | Model swap events |
| `llm_scheduler_v2_model_swap_seconds` | Histogram | `runner` | Swap duration |
| `llm_queue_job_duration_seconds` | Histogram | — | Inference time for queue-path jobs |

### Fast-path Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_fastpath_requests_total` | Counter | `model`, `status` | Fast-path requests (status: completed/failed) |
| `llm_fastpath_duration_seconds` | Histogram | `model` | End-to-end fast-path latency |

### Runner Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_runner_is_idle` | Gauge | `runner` | 1 = idle, 0 = has in-flight job |
| `llm_runner_vram_used_bytes` | Gauge | `runner` | Live VRAM used (updated at reconcile + post-swap) |
| `llm_runner_vram_total_bytes` | Gauge | `runner` | Total VRAM capacity |
| `llm_inference_prompt_tokens_total` | Counter | `model`, `runner` | Prompt tokens processed |
| `llm_inference_completion_tokens_total` | Counter | `model`, `runner` | Completion tokens generated |

### Config Metrics (info-style)

These are refreshed every 30 seconds by a background task. Value is always 1; data is in labels.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llm_alias_info` | Gauge | `alias`, `base_model`, `has_system_prompt` | Active alias configuration |
| `llm_alias_param` | Gauge | `alias`, `param` | Numeric parameters per alias |
| `llm_runner_model_param` | Gauge | `model`, `runner`, `param` | Per-runner model parameter overrides |

---

## Grafana Dashboards

Two dashboards are committed to `k3s-dean-gitops/apps/llm-manager/backend/` as
ConfigMaps with the `grafana_dashboard: "1"` label, which the Grafana sidecar
auto-discovers.

### LLM Manager Overview (`llm-manager-overview`)

Fleet-wide view. Panels:

- **Queue Depth / Active Jobs / Online Runners / Fast-path req/min** — stat row
- **Job Completions** — rate of completed/failed queue jobs + fast-path
- **Fast-path vs Queue** — relative request volume split
- **Inference Duration** — fast-path and queue-path p50/p95 latency
- **Queue Wait Time** — p50/p95 wait, broken down by app
- **VRAM Used % by Runner** — horizontal bar gauge, all runners
- **Token Throughput** — prompt/completion tokens/s by model
- **Model Swaps / hr** — swap frequency by runner
- **Swap Duration P95** — swap time by runner
- **Runner Idle State** — BUSY/IDLE indicator per runner
- **Alias Config / Alias Parameters / Runner Model Parameter Overrides** — config tables (30s refresh)

### LLM Manager — Runner (`llm-manager-runner`)

Per-runner drill-down with `$runner` template variable. Panels:

- **Status / Heartbeat Age / VRAM Used / VRAM %** — stat row
- **VRAM Over Time** — used vs total over the time window
- **Token Throughput** — per-model tok/s for this runner
- **Inference Duration (fast-path)** — p50/p95 by model
- **Model Swap Rate / Swap Duration P95** — swap behavior on this runner
- **Queue Wait by App** — per-app wait time p50/p95
- **Recent Swaps** — table of swap pairs with count over last 6h

---

## Future: LLMRouter Integration

[LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (UIUC) and
[RouteLLM](https://github.com/lm-sys/RouteLLM) (LMSYS/Berkeley) are **query-level
routing** libraries. They analyze incoming prompts and decide which model should
handle them (e.g., "simple question → 7B model, complex reasoning → 32B model").

**They are complementary, not replacements** for llm-manager:

- LLMRouter/RouteLLM pick the best model per query (prompt classification)
- llm-manager manages the infrastructure (VRAM, model lifecycle, GPU runners)
- Neither routing library can load/unload models, manage VRAM, or orchestrate GPUs

**When it would make sense to add:**

- When running multiple models concurrently (e.g., second GPU added, enough VRAM
  for both 7B and 32B models loaded simultaneously)
- A routing layer would sit in front of `/v1/chat/completions` and auto-select
  the right model per query, while llm-manager ensures the models are loaded
- Not useful with a single model loaded — there's nothing to route between

**Key projects:**

- `llmrouter-lib` (pip) — 16+ routing algorithms (KNN, SVM, BERT, Elo), Gradio UI
- `routellm` (pip) — Strong/weak model routing, OpenAI-compatible drop-in server
- NVIDIA LLM Router Blueprint — Intent classification with Qwen 1.7B or CLIP
