# LLM Agent

Runs on the GPU host machine (murderbot). Wraps Ollama and ComfyUI, exposes an HTTP API on port **8090** plus Prometheus metrics.

## Prerequisites

- Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Ollama running on the host at `localhost:11434`
- ComfyUI running via docker compose at `/home/alex/claude/projects/image/`
- Models at `/opt/models/checkpoints/`, `/opt/models/lora/`, `/opt/models/vae/`

## Installation

```bash
cd agent/
bash install.sh
```

The install script will:
1. Verify Docker and nvidia-container-toolkit are present
2. Build the container image
3. Start the container via docker compose
4. Create and enable a systemd service (`llm-agent.service`) for auto-start on boot
5. Wait for the health endpoint to respond

## Manual start

```bash
docker compose up -d
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama base URL |
| `COMFYUI_URL` | `http://host.docker.internal:8188` | ComfyUI base URL |
| `COMFYUI_OUTPUT_DIR` | `/outputs` | Path inside container to ComfyUI outputs |

## API Reference

### `GET /health`
```json
{"ok": true, "node": "murderbot", "ollama": true, "comfyui": true}
```

### `GET /v1/status`
Full system status: GPU VRAM, CPU, memory, loaded Ollama models, ComfyUI checkpoint list.

### `GET /v1/models`
OpenAI-compatible model list. Text models from Ollama, image models from ComfyUI checkpoints.

### `POST /v1/chat/completions`
OpenAI-compatible chat endpoint. Proxies to Ollama. Supports streaming (`"stream": true`).

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### `POST /v1/images/generations`
Generate an image via ComfyUI. Polls until done (up to 5 minutes).

```bash
curl http://localhost:8090/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a sunset over mountains", "model": "v1-5-pruned-emaonly.safetensors", "size": "512x512"}'
```

Returns:
```json
{"created": 1700000000, "data": [{"url": "/v1/images/outputs/llm_agent_00001_.png"}]}
```

### `GET /v1/images/outputs/{filename}`
Serve a generated image file.

### `POST /v1/models/pull`
Pull an Ollama model. Streams NDJSON progress lines.

```bash
curl http://localhost:8090/v1/models/pull \
  -d '{"model": "qwen2.5:7b"}'
```

### `DELETE /v1/models/{model}`
Unload a model from VRAM (sets `keep_alive=0`).

### `POST /v1/comfyui/checkpoint`
Switch the active ComfyUI checkpoint.

```bash
curl -X POST http://localhost:8090/v1/comfyui/checkpoint \
  -d '{"name": "dreamshaper_8.safetensors"}'
```

### `GET /metrics`
Prometheus text format metrics.

## Metrics exposed

| Metric | Type | Description |
|---|---|---|
| `llm_agent_gpu_vram_used_bytes` | Gauge | GPU VRAM used |
| `llm_agent_gpu_vram_total_bytes` | Gauge | GPU VRAM total |
| `llm_agent_gpu_vram_utilization_pct` | Gauge | GPU VRAM % |
| `llm_agent_cpu_usage_pct` | Gauge | CPU usage % |
| `llm_agent_memory_used_bytes` | Gauge | RAM used |
| `llm_agent_memory_total_bytes` | Gauge | RAM total |
| `llm_agent_ollama_models_loaded` | Gauge | Models in VRAM |
| `llm_agent_comfyui_running` | Gauge | ComfyUI 0/1 |
| `llm_agent_requests_total` | Counter | Requests by endpoint+model |
| `llm_agent_request_duration_seconds` | Histogram | Latency by endpoint+model |

## Logs

```bash
docker logs -f llm-agent
```
