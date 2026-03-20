# LLM Agent (llm-runner)

Runs on the GPU host machine (murderbot). Wraps Ollama and ComfyUI, exposes an authenticated HTTP API on port **8090** plus Prometheus metrics. Self-registers with the llm-manager backend on startup.

## Prerequisites

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Ollama running on the host at `localhost:11434`
- ComfyUI running via docker compose at `/home/alex/claude/projects/image/`
- Models at `/opt/models/checkpoints/`, `/opt/models/lora/`, `/opt/models/vae/`

## Setup

### 1. Create `.env` file

```bash
cp agent/.env.example agent/.env
```

Edit `.env`:
```
LLM_MANAGER_AGENT_PSK=<value from Bitwarden: llm-manager-agent-psk>
BACKEND_URL=http://llm-manager-backend.llm-manager.svc.cluster.local:8081
AGENT_ADDRESS=http://murderbot.amer.home:8090
```

The PSK must match the value stored in Bitwarden and synced to the backend pod via ExternalSecret.

### 2. Start

```bash
cd agent/
docker compose up -d
```

Or run the install script (sets up systemd auto-start):
```bash
bash install.sh
```

### 3. Verify

```bash
curl http://localhost:8090/health
# {"ok": true, "node": "murderbot", "ollama": true, "comfyui": false}

docker logs -f llm-agent
# Should show: Registered with backend as runner_id=1
```

## PSK Authentication

All endpoints except `/health` and `/metrics` require the header:
```
X-Agent-PSK: <psk>
```

The backend pod sends this automatically when proxying requests.

## PSK Rotation

1. Update the value in Bitwarden (`llm-manager-agent-psk`)
2. ExternalSecret refreshes the k8s secret within 1 hour (or force: `kubectl annotate externalsecret agent-psk -n llm-manager force-sync=$(date +%s)`)
3. Update `.env` on each GPU host and restart: `docker compose restart`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MANAGER_AGENT_PSK` | _(empty)_ | Pre-shared key for backend auth. If empty, auth is disabled (dev only) |
| `BACKEND_URL` | _(empty)_ | Backend URL for self-registration. If empty, registration is skipped |
| `AGENT_ADDRESS` | `http://<hostname>:8090` | Externally-reachable address of this agent |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama base URL |
| `COMFYUI_URL` | `http://host.docker.internal:8188` | ComfyUI base URL |
| `COMFYUI_OUTPUT_DIR` | `/outputs` | Path inside container to ComfyUI outputs |

## API Reference

All requests (except `/health` and `/metrics`) require header `X-Agent-PSK: <psk>`.

### `GET /health`
```json
{"ok": true, "node": "murderbot", "ollama": true, "comfyui": true}
```

### `GET /v1/status`
Full system status: GPU VRAM, CPU, memory, loaded Ollama models, ComfyUI checkpoint list.

### `GET /v1/models`
OpenAI-compatible model list.

### `POST /v1/chat/completions`
OpenAI-compatible chat endpoint. Supports streaming.

### `POST /v1/images/generations`
Generate an image via ComfyUI.

### `POST /v1/models/pull`
Pull an Ollama model. Streams NDJSON progress.

### `DELETE /v1/models/{model}`
Unload a model from VRAM.

### `POST /v1/comfyui/checkpoint`
Switch the active ComfyUI checkpoint.

### `GET /metrics`
Prometheus text format metrics (no PSK required).

## Logs

```bash
docker logs -f llm-agent
```
