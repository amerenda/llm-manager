#!/usr/bin/env bash
# install.sh — Install and configure the llm-agent on the GPU host.
# Run as root or a user with sudo and docker access.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="llm-agent"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Check dependencies ────────────────────────────────────────────────────────

log "Checking dependencies..."

command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install from https://docs.docker.com/engine/install/"

# Check Docker is running
docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start it with: sudo systemctl start docker"

# Check nvidia-container-toolkit (for GPU passthrough)
if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-container-toolkit may not be installed or GPU unavailable."
    warn "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    warn "Continuing anyway — agent will run without GPU metrics."
fi

# Check nvidia-smi on host
if command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "nvidia-smi not found on host. GPU metrics may be unavailable."
fi

# ── Verify model directories ───────────────────────────────────────────────────

if [[ ! -d /opt/models ]]; then
    warn "/opt/models does not exist. Creating it."
    sudo mkdir -p /opt/models/checkpoints /opt/models/lora /opt/models/vae
fi

# ── Build and start the container ─────────────────────────────────────────────

log "Building llm-agent container..."
docker compose -f "$COMPOSE_FILE" build

log "Starting llm-agent container..."
docker compose -f "$COMPOSE_FILE" up -d

# ── Create systemd service for auto-start ─────────────────────────────────────

SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

log "Creating systemd service at $SYSTEMD_UNIT ..."

sudo tee "$SYSTEMD_UNIT" > /dev/null <<EOF
[Unit]
Description=LLM Agent (Ollama + ComfyUI proxy)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/usr/bin/docker compose -f ${COMPOSE_FILE} up -d
ExecStop=/usr/bin/docker compose -f ${COMPOSE_FILE} down
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"

log "Systemd service '${SERVICE_NAME}' enabled for auto-start on boot."

# ── Wait for health check ──────────────────────────────────────────────────────

log "Waiting for llm-agent to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
        log "llm-agent is up!"
        curl -s http://localhost:8090/health | python3 -m json.tool
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -sf http://localhost:8090/health >/dev/null 2>&1; then
    warn "Agent didn't respond after 60s. Check logs with: docker logs llm-agent"
fi

echo ""
log "Installation complete!"
echo ""
echo "  Health:  http://localhost:8090/health"
echo "  Status:  http://localhost:8090/v1/status"
echo "  Models:  http://localhost:8090/v1/models"
echo "  Metrics: http://localhost:8090/metrics"
echo ""
echo "  View logs:    docker logs -f llm-agent"
echo "  Stop:         docker compose -f $COMPOSE_FILE down"
echo "  Restart:      docker compose -f $COMPOSE_FILE restart"
