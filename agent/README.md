# LLM Agent (`llm-runner`)

Runs on every GPU host in the fleet. Wraps an Ollama container plus (optionally) ComfyUI, exposes an mTLS/PSK-authenticated HTTP API on port **8090** with Prometheus metrics, and self-registers with the llm-manager backend.

Ollama is **always compose-managed** by this agent — host-managed Ollama is not supported. The Models page in the UI writes `ollama.ui.env` and triggers a container recreate via the agent; compose loads `ollama.env` (GitOps defaults) then `ollama.ui.env` (UI overrides), so UI values survive restarts/redeploys.

## Deployment (Komodo)

In this homelab, **the llm-agent and Ollama are installed and updated through Komodo**, not via helper scripts in this repo. Compose lives in **[`komodo-dean-gitops`](https://github.com/amerenda/komodo-dean-gitops)** under each host’s `llm/` directory (e.g. `murderbot/llm/`, `archlinux/llm/`, `mac-mini-m4/llm/`). **Periphery** pulls the repo, runs `pre-deploy` (Bitwarden → `.env`), and `docker compose up`. Host bootstrap (Docker, BWS token, Periphery) is in **[`ansible-playbooks`](https://github.com/amerenda/ansible-playbooks)** (`setup-debian-komodo.yml`, `setup-archlinux-komodo.yml`, `setup-macmini.yml`).

Redeploy or pull new images by syncing stacks in Komodo or pushing to the tracked branch (see `komodo-dean-gitops` docs).

## Mac Mini (`mac-mini-m4/llm/`)

The Mini uses a **bridge-network** stack (not `network_mode: host`): see [`komodo-dean-gitops/mac-mini-m4/llm/compose.yaml`](https://github.com/amerenda/komodo-dean-gitops/tree/main/mac-mini-m4/llm). Ollama and `llm-agent` are both compose services; the agent talks to Ollama over the compose network and uses **`AGENT_UNIFIED_MEMORY_VRAM=true`** for sensible “VRAM” metrics on Apple Silicon.

Set **`RUNNER_HOSTNAME`** (e.g. `mac-mini-m4`) so the runner name in llm-manager is stable. Registering under a new hostname creates a **new** runner row — remove stale entries in the UI if needed.

For **auto self-update** (heartbeat-driven image tag + compose recreate), the stack sets **`COMPOSE_DIR`** / **`COMPOSE_DIR_LOCAL`** and bind-mounts the compose directory; enable **auto update** for the runner in llm-manager.

## Prerequisites (manual / dev installs only)

- Docker + Docker Compose plugin
- NVIDIA ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) or AMD (`/dev/kfd`, `/dev/dri`, ROCm)
- Agent PSK from Bitwarden: `llm-manager-agent-psk`

## Manual install (`agent/compose.yaml` in this repo)

For development or a one-off GPU host **without** Komodo:

```bash
cd agent/
cp .env.example .env
# Set LLM_MANAGER_AGENT_PSK, BACKEND_URL, AGENT_ADDRESS, OLLAMA_MODELS_PATH as needed.
# AMD: set VIDEO_GID and RENDER_GID (e.g. getent group video; getent group render).
cp ollama.env.example ollama.env

docker compose --profile nvidia up -d    # or --profile amd
docker compose logs -f
```

This starts **llm-agent** and **Docker-managed Ollama** with `network_mode: host` on Linux (see `compose.yaml`).

## Models directory — single source of truth

`OLLAMA_MODELS_PATH` in `agent/.env` is **the** knob. It drives:

- the Ollama container bind mount (`${OLLAMA_MODELS_PATH}:/root/.ollama/models`)
- the agent's `OLLAMA_MODELS` env (used by future pull-size estimates)
- `MODEL_STORAGE_PATH` (the agent's mirror of the same path)

Default: `/opt/ollama/models`. Change it by editing `.env` and running `docker compose --profile <vendor> up -d --force-recreate ollama`.

The agent logs a loud error at startup if it finds an `ollama` container that isn't labeled `com.docker.compose.project=agent` — that's the "pre-migration orphan" state where UI edits silently do nothing.

## Ollama tunables (UI-wins contract)

`ollama.env` is deployment-owned defaults; `ollama.ui.env` is UI-owned overrides. The agent only writes `ollama.ui.env`.

Every value the UI can set (`OLLAMA_NUM_CTX`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_DATA_HOST_PATH`, `OLLAMA_MODELS_HOST_PATH`, etc.) is merged as defaults + overrides, then applied. The agent:

1. Validates incoming keys against the allowlist.
2. Rewrites `ollama.ui.env` with the requested overrides.
3. Recreates the Ollama container.

`.env` (agent credentials + stack identity) remains deployment-owned and is intentionally **not** touched by the UI.

## PSK auth

All endpoints except `/health` and `/metrics` require header:

```
X-Agent-PSK: <psk>
```

The backend pod injects this via the `agent-psk` ExternalSecret. To rotate:

1. Update `llm-manager-agent-psk` in Bitwarden.
2. Force-sync the ExternalSecret: `kubectl annotate externalsecret agent-psk -n llm-manager force-sync=$(date +%s)`.
3. On each host: update the agent’s `.env` (or BWS-fed env), then recreate the agent container — e.g. `docker compose --profile <vendor> up -d --no-deps --force-recreate llm-agent`, or redeploy the Komodo `llm` stack.

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_MANAGER_AGENT_PSK` | *(required)* | PSK for backend auth |
| `BACKEND_URL` | `https://llm-manager-backend.amer.dev` | Backend base URL |
| `AGENT_ADDRESS` | *(auto-detected)* | Externally reachable `http://<ip>:8090`. Override if auto-detect picks the wrong NIC |
| `COMPOSE_DIR` | *(optional)* | Host path to the compose project directory (agent self-update / UI-driven compose) |
| `COMPOSE_PROFILE` | `nvidia-full` / `amd-full` | Active compose profile on Linux; Mac Mini stack often uses `""` |
| `OLLAMA_MODELS_PATH` | `/opt/ollama/models` | Models directory (bind-mounted into Ollama) |
| `MODEL_STORAGE_PATH` | same as above | Agent's view of the same path |
| `AGENT_IMAGE_TAG` | `latest` | Pinned agent image tag (rewritten on self-update) |
| `OLLAMA_IMAGE_TAG` | `0.21.0` | NVIDIA Ollama tag |
| `OLLAMA_AMD_IMAGE_TAG` | `0.21.0-rocm` | AMD Ollama tag |
| `HSA_OVERRIDE_GFX_VERSION` | *(AMD only)* | Forces a ROCm GFX version (e.g. `11.0.0` for RDNA 4) |
| `VIDEO_GID`, `RENDER_GID` | *(AMD only)* | Host numeric GIDs for `/dev/dri` perms |

## Verify

```bash
curl -sk https://localhost:8090/health
# {"ok":true,"node":"murderbot","ollama":true,"comfyui":true}  # comfyui false if not deployed

docker inspect ollama --format '{{index .Config.Labels "com.docker.compose.project"}}'
# expect your compose project name (this repo's default layout uses project "agent")

docker inspect ollama --format '{{(index .Mounts 0).Source}}'
# should match OLLAMA_MODELS_PATH

docker logs llm-agent 2>&1 | grep -i 'not compose-managed' || echo "ok"
```

## Day-2 ops

| | Command |
|---|---|
| Logs | `docker logs -f llm-agent` / `docker logs -f ollama` |
| Stop | `docker compose --profile <vendor> down` |
| Restart agent only | `docker compose --profile <vendor> up -d --no-deps --force-recreate llm-agent` |
| Restart Ollama (apply tunable changes) | `docker compose --profile <vendor> up -d --force-recreate ollama` |
| Update agent image | Redeploy Komodo `llm` stack, or `docker compose pull && docker compose --profile <vendor> up -d`, or UI **auto update** |

## API

All routes except `/health` and `/metrics` require `X-Agent-PSK`.

- `GET /health` — quick liveness (no PSK)
- `GET /v1/status` — GPU, memory, loaded/downloaded models, ComfyUI checkpoints
- `GET /v1/models` — OpenAI-compatible model list
- `POST /v1/chat/completions` — OpenAI-compatible chat (streaming supported)
- `POST /v1/images/generations` — ComfyUI image generation
- `POST /v1/models/pull` — Ollama pull, streams NDJSON progress
- `DELETE /v1/models/{model}` — unload from VRAM
- `POST /v1/ollama/restart` — restart Ollama container (clears stuck VRAM)
- `GET /v1/ollama/settings` — read effective settings + source metadata (`default` vs `ui_override`)
- `PUT /v1/ollama/settings` — write `ollama.ui.env` + recreate container
- `GET /metrics` — Prometheus (no PSK)
