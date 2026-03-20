# llm-manager

Manages local LLM inference (Ollama + ComfyUI) across GPU nodes in a k3s cluster.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  murderbot (GPU host, bare metal)                                   │
│                                                                     │
│  ┌───────────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ Ollama            │   │ ComfyUI      │   │ llm-agent        │   │
│  │ localhost:11434   │   │ localhost:   │   │ port: 8090       │   │
│  │ (bare metal)      │   │ 8188         │   │ (docker compose) │   │
│  │                   │   │ (docker      │   │                  │   │
│  │  text models      │   │  compose)    │   │ wraps both ^     │   │
│  └───────────────────┘   └──────────────┘   │ OpenAI-compat    │   │
│                                             │ API + metrics    │   │
│  /opt/models/                               └──────────────────┘   │
│    checkpoints/                                      ▲             │
│    lora/                                             │ HTTP        │
│    vae/                                              │             │
└─────────────────────────────────────────────────────│─────────────┘
                                                       │
                              ┌────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  k3s cluster (GPU-labeled nodes)                                    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  llm-manager backend  (DaemonSet, port 8081)                   │ │
│  │                                                                │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                   │ │
│  │  │ Moltbook agents  │  │ LLM proxy API    │                   │ │
│  │  │ (slots 1-6)      │  │ /v1/chat/...     │  ◄── apps        │ │
│  │  │                  │  │ /v1/images/...   │                   │ │
│  │  └──────────────────┘  └──────────────────┘                   │ │
│  │                                                                │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                   │ │
│  │  │ App registry     │  │ /metrics         │  ◄── Prometheus   │ │
│  │  │ (PostgreSQL)     │  │ (backend +       │                   │ │
│  │  │                  │  │  agent fwd)      │                   │ │
│  │  └──────────────────┘  └──────────────────┘                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  PostgreSQL  │  DATABASE_URL=postgresql://llm:llm@postgres/     │ │
│  │  (StatefulSet│  llmmanager                                      │ │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### `agent/` — LLM Agent (runs on GPU host)

Docker compose service that runs on murderbot alongside Ollama and ComfyUI.

- **Port:** 8090
- **Proxies:** Ollama (11434), ComfyUI (8188)
- **Metrics:** GPU VRAM, CPU, memory, request counters

See [agent/README.md](agent/README.md) for installation instructions.

### `backend/` — Backend (k8s DaemonSet)

Expanded from the moltbook-manager backend. Runs on each GPU-labeled k8s node.

- **Port:** 8081
- **Connects to:** local llm-agent at `http://localhost:8090`
- **Persists to:** PostgreSQL (`DATABASE_URL`)
- **Provides:** OpenAI-compatible API proxy, app registry, moltbook agent management

## Quick start (development)

### Agent (on GPU host)

```bash
cd agent/
bash install.sh
# or manually:
docker compose up -d
```

### Backend (local dev)

```bash
cd backend/
pip install -r requirements.txt
DATABASE_URL=postgresql://llm:llm@localhost:5432/llmmanager \
  AGENT_URL=http://localhost:8090 \
  python main.py
```

## Backend API

### LLM Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/llm/status` | Agent status (GPU, models, ComfyUI) |
| GET | `/api/llm/models` | List all models (text + image) |
| POST | `/api/llm/models/pull` | Pull an Ollama model |
| DELETE | `/api/llm/models/{model}` | Unload model from VRAM |
| POST | `/api/llm/comfyui/checkpoint` | Switch ComfyUI checkpoint |
| GET | `/api/llm/checkpoints` | List available checkpoints |

### OpenAI-Compatible Proxy

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming supported) |
| POST | `/v1/images/generations` | Image generation |

### App Registry

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/apps` | List registered apps |
| POST | `/api/apps/register` | Register app → returns api_key |
| POST | `/api/apps/heartbeat` | Update app last_seen (Bearer auth) |
| DELETE | `/api/apps/{api_key}` | Deregister app |

### Moltbook (existing)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/agents` | List moltbook agent slots |
| PATCH | `/api/agents/{slot}` | Update agent config |
| POST | `/api/agents/{slot}/start` | Start agent |
| POST | `/api/agents/{slot}/stop` | Stop agent |
| GET | `/api/agents/{slot}/activity` | Recent activity log |
| POST | `/api/agents/{slot}/register` | Register with moltbook.com |

### Metrics

| Path | Description |
|------|-------------|
| `GET /metrics` | Prometheus — backend metrics + forwarded agent metrics |

## Environment variables

### Agent

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama URL |
| `COMFYUI_URL` | `http://host.docker.internal:8188` | ComfyUI URL |
| `COMFYUI_OUTPUT_DIR` | `/outputs` | Output image directory (in container) |

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://llm:llm@localhost:5432/llmmanager` | PostgreSQL DSN |
| `AGENT_URL` | `http://localhost:8090` | Local llm-agent URL |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama URL (for direct VRAM checks) |

## Database schema

```sql
-- LLM nodes seen in the last 5 minutes
CREATE TABLE llm_agents (
    id SERIAL PRIMARY KEY,
    node_name TEXT UNIQUE NOT NULL,
    host TEXT NOT NULL,
    port INT NOT NULL DEFAULT 8090,
    last_seen TIMESTAMPTZ,
    capabilities JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Apps registered to use the LLM proxy API
CREATE TABLE registered_apps (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    base_url TEXT,
    api_key TEXT UNIQUE NOT NULL,
    last_seen TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Tables are created automatically on first startup via `init_db()`.
