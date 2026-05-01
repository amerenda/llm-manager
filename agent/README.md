# LLM Agent (`llm-runner`)

Runs on every GPU host in the fleet. Wraps an Ollama container plus (optionally) ComfyUI, exposes an mTLS/PSK-authenticated HTTP API on port **8090** with Prometheus metrics, and self-registers with the llm-manager backend.

Ollama is **always compose-managed** by this agent — host-managed Ollama is not supported. The Models page in the UI edits `ollama.env` and triggers a container recreate via the agent; that round-trip only works when compose owns the Ollama container.

## Mac Mini (Apple Silicon, native Ollama)

On the home-lab Mac Mini, Ollama runs **on the host** for Metal. Deploy **only** the agent via Docker using the dedicated stack in [`amerenda/mac-mini-compose`](https://github.com/amerenda/mac-mini-compose) (`llm/compose.yaml`): bridge networking, `OLLAMA_URL=http://host.docker.internal:11434`, and **no** `OLLAMA_CONTAINER` so the agent skips compose-managed Ollama checks. GitOps and Komodo wiring are documented in that repo’s README (separate stack + webhook so `llm` deploys do not touch core/automation/monitoring/runners).

Set **`RUNNER_HOSTNAME`** (or `AGENT_NODE_NAME`) to the Mac’s stable name (e.g. `mac-mini-m4`). Otherwise Docker sets the container hostname to a short id and llm-manager shows that hex as the runner name.

Enable **`AGENT_UNIFIED_MEMORY_VRAM=true`** so VRAM bars use **container-visible RAM** as the unified pool and **Ollama `/api/ps` model sizes** as “used”. There is no NVML/AMD sysfs for Metal from a Linux agent container. If `psutil`’s total does not match real unified RAM (some VM limits), set **`AGENT_UNIFIED_VRAM_TOTAL_BYTES`** to the pool size in bytes.

**Note:** Registering under a new hostname creates a **new** llm-manager runner row; remove the old runner (short-id hostname) in the UI if it lingers.

For a manual compose file outside that repo, mirror the same pattern: publish `8090:8090`, `extra_hosts: host.docker.internal:host-gateway`, bind-mount the host Ollama models directory read-only at `/host-ollama-models`, set `MODEL_STORAGE_PATH=/host-ollama-models`, and leave `COMPOSE_PROFILE` / `COMPOSE_DIR` unset so fleet **self-update** (which assumes GPU compose profiles) stays disabled — use image bumps via compose or Komodo instead.

## Prerequisites

- Docker + Docker Compose plugin
- One supported GPU:
  - NVIDIA with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - AMD with `/dev/kfd` + `/dev/dri` present (ROCm)
- Agent PSK from Bitwarden: `llm-manager-agent-psk`

## Install

```bash
cd agent/
LLM_AGENT_PSK=<psk> bash install.sh
```

What it does:
1. Detects GPU vendor, sets compose profile (`nvidia` or `amd`).
2. Stops any host Ollama (systemd or standalone `docker run`) and removes the container so compose can create a fresh, labeled one.
3. Creates `/opt/ollama/models` (or `$MODEL_STORAGE_PATH` override) owned by the install user.
4. Writes `.env` with the PSK, backend URL, compose dir/profile, and `OLLAMA_MODELS_PATH` (single source of truth for the models directory).
5. Copies `ollama.env.example` → `ollama.env`. All tunables stay commented out; the UI writes into this file.
6. `docker compose --profile <vendor> up -d` starts both `llm-agent` and `ollama`.

Useful flags:

| Flag | Purpose |
|---|---|
| `--update` / `-u` | Re-pull agent image, preserve config |
| `--psk <value>` | Rotate PSK in existing `.env` |
| `--model-storage <path>` | Override `OLLAMA_MODELS_PATH` |
| `--migrate` | Stop + disable host systemd `ollama.service` without prompting |
| `--ollama-tag <tag>` | Pin a different Ollama image tag |

`--managed-ollama` and `--host-ollama` are accepted for backward compatibility but ignored — Ollama is always compose-managed now.

## Models directory — single source of truth

`OLLAMA_MODELS_PATH` in `agent/.env` is **the** knob. It drives:

- the Ollama container bind mount (`${OLLAMA_MODELS_PATH}:/root/.ollama/models`)
- the agent's `OLLAMA_MODELS` env (used by future pull-size estimates)
- `MODEL_STORAGE_PATH` (the agent's mirror of the same path)

Default: `/opt/ollama/models`. Change it by editing `.env` and running `docker compose --profile <vendor> up -d --force-recreate ollama`.

The agent logs a loud error at startup if it finds an `ollama` container that isn't labeled `com.docker.compose.project=agent` — that's the "pre-migration orphan" state where UI edits silently do nothing.

## Ollama tunables

Every value the UI can set (`OLLAMA_NUM_CTX`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`, etc.) lives in `ollama.env`, loaded by the Ollama service via compose's `env_file`. The agent:

1. Rewrites `ollama.env` with the new values.
2. Calls `docker compose -f <compose.yaml> --profile <profile> up -d --force-recreate ollama`.
3. Waits for `/api/tags` to come back.

`.env` (agent credentials + paths) is intentionally **not** touched by the UI — mixing the two files was what motivated the split.

## PSK auth

All endpoints except `/health` and `/metrics` require header:

```
X-Agent-PSK: <psk>
```

The backend pod injects this via the `agent-psk` ExternalSecret. To rotate:

1. Update `llm-manager-agent-psk` in Bitwarden.
2. Force-sync the ExternalSecret: `kubectl annotate externalsecret agent-psk -n llm-manager force-sync=$(date +%s)`.
3. Per host: `bash install.sh --psk <new>` then `docker compose --profile <vendor> restart llm-agent`.

## Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_MANAGER_AGENT_PSK` | *(required)* | PSK for backend auth |
| `BACKEND_URL` | `https://llm-manager-backend.amer.dev` | Backend base URL |
| `AGENT_ADDRESS` | *(auto-detected)* | Externally reachable `http://<ip>:8090`. Override if auto-detect picks the wrong NIC |
| `COMPOSE_DIR` | *(set by install.sh)* | Host path to this directory (needed for agent self-update) |
| `COMPOSE_PROFILE` | *(set by install.sh)* | `nvidia` or `amd` |
| `OLLAMA_MODELS_PATH` | `/opt/ollama/models` | Models directory (bind-mounted into Ollama) |
| `MODEL_STORAGE_PATH` | same as above | Agent's view of the same path |
| `AGENT_IMAGE_TAG` | `latest` | Pinned agent image tag (rewritten on self-update) |
| `OLLAMA_IMAGE_TAG` | `0.21.0` | NVIDIA Ollama tag |
| `OLLAMA_AMD_IMAGE_TAG` | `0.21.0-rocm` | AMD Ollama tag |
| `HSA_OVERRIDE_GFX_VERSION` | *(AMD only)* | Forces a ROCm GFX version (e.g. `11.0.0` for RDNA 4) |
| `VIDEO_GID`, `RENDER_GID` | *(AMD only)* | Host numeric GIDs for `/dev/dri` perms |

## Verify

```bash
# mTLS health (install.sh already fingerprinted the backend)
curl -sk https://localhost:8090/health
# {"ok":true,"node":"murderbot","ollama":true,"comfyui":true}

# Ollama is compose-managed
docker inspect ollama --format '{{index .Config.Labels "com.docker.compose.project"}}'
# agent

# Mount matches OLLAMA_MODELS_PATH
docker inspect ollama --format '{{(index .Mounts 0).Source}}'
# /opt/ollama/models

# Agent startup log should NOT contain:
#   "Ollama container 'ollama' is NOT compose-managed"
docker logs llm-agent 2>&1 | grep -i 'not compose-managed' || echo "ok"
```

## Day-2 ops

| | Command |
|---|---|
| Logs | `docker logs -f llm-agent` / `docker logs -f ollama` |
| Stop | `docker compose --profile <vendor> down` |
| Restart agent only | `docker compose --profile <vendor> up -d --no-deps --force-recreate llm-agent` |
| Restart Ollama (apply tunable changes) | `docker compose --profile <vendor> up -d --force-recreate ollama` |
| Update agent image | `bash install.sh --update` (or UI auto-update) |

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
- `GET /v1/ollama/settings` — read current Ollama tunables + allowlist
- `PUT /v1/ollama/settings` — rewrite `ollama.env` + recreate container
- `GET /metrics` — Prometheus (no PSK)
