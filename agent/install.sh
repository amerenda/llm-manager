#!/usr/bin/env bash
# install.sh — Install and configure the llm-agent on the GPU host.
# Auto-detects NVIDIA or AMD GPU and uses the appropriate compose file.
# Run as root or a user with sudo and docker access.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="llm-agent"

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
docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start it with: sudo systemctl start docker"

# ── Detect GPU vendor ────────────────────────────────────────────────────────

GPU_VENDOR="none"

# Check for AMD first via device nodes (more reliable than CLI tools which
# can be installed without matching hardware, e.g. nvidia-smi from Ollama deps).
if [[ -e /dev/kfd ]] && [[ -d /dev/dri ]]; then
    GPU_VENDOR="amd"
    log "AMD GPU detected (ROCm devices present)"
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname 2>/dev/null || true
    fi
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
    log "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "No GPU detected. Agent will run without GPU metrics."
fi

log "GPU vendor: $GPU_VENDOR"

# ── Select compose file ──────────────────────────────────────────────────────

if [[ "$GPU_VENDOR" == "amd" ]]; then
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.amd.yml"
    log "Using AMD compose file: docker-compose.amd.yml"
else
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
    log "Using NVIDIA compose file: docker-compose.yml"
fi

# ── Verify GPU toolkit (vendor-specific) ─────────────────────────────────────

if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    if ! docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-container-toolkit may not be installed or GPU unavailable."
        warn "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        warn "Continuing anyway — agent will run without GPU metrics."
    fi
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    if [[ ! -e /dev/kfd ]]; then
        warn "/dev/kfd not found — ROCm kernel driver may not be loaded."
        warn "Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html"
    fi
    # Verify user is in video/render groups
    if ! groups | grep -qE '\b(video|render)\b'; then
        warn "Current user is not in video/render groups. Adding..."
        sudo usermod -aG render,video "$USER"
        warn "You may need to log out and back in for group changes to take effect."
    fi
fi

# ── Verify .env ──────────────────────────────────────────────────────────────

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        warn ".env created from .env.example — edit it and set LLM_MANAGER_AGENT_PSK before continuing."
        warn "Get the PSK from Bitwarden: llm-manager-agent-psk"
        echo ""
        echo "  $EDITOR $SCRIPT_DIR/.env"
        echo ""
        die "Edit .env and re-run this script."
    else
        die ".env file not found and no .env.example to copy from."
    fi
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
Description=LLM Agent (Ollama proxy — ${GPU_VENDOR} GPU)
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
    if curl -sf https://localhost:8090/health --insecure >/dev/null 2>&1; then
        log "llm-agent is up!"
        curl -s https://localhost:8090/health --insecure | python3 -m json.tool
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -sf https://localhost:8090/health --insecure >/dev/null 2>&1; then
    warn "Agent didn't respond after 60s. Check logs with: docker logs llm-agent"
fi

echo ""
log "Installation complete! (GPU: $GPU_VENDOR)"
echo ""
echo "  Health:  https://localhost:8090/health"
echo "  Status:  https://localhost:8090/v1/status"
echo "  Models:  https://localhost:8090/v1/models"
echo "  Metrics: https://localhost:8090/metrics"
echo ""
echo "  View logs:    docker logs -f llm-agent"
echo "  Stop:         docker compose -f $COMPOSE_FILE down"
echo "  Restart:      docker compose -f $COMPOSE_FILE restart"
