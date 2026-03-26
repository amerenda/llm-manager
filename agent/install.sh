#!/usr/bin/env bash
# install.sh — Set up the llm-agent on a GPU host.
#
# Prerequisites:
#   - Docker with Compose plugin (docker compose)
#   - Ollama running on the host (localhost:11434)
#
# The agent is a pre-built container image pulled from Docker Hub.
# Auto-detects NVIDIA or AMD GPU and selects the correct profile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/compose.yaml"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Check prerequisites ──────────────────────────────────────────────────────

log "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install from https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not installed."
docker info >/dev/null 2>&1 || die "Docker daemon is not running."

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    die "Ollama is not running on localhost:11434. Install and start Ollama first: https://ollama.com/download"
fi
log "Ollama is running."

# ── Detect GPU vendor ────────────────────────────────────────────────────────

GPU_VENDOR="none"

if [[ -e /dev/kfd ]] && [[ -d /dev/dri ]]; then
    GPU_VENDOR="amd"
    log "AMD GPU detected (ROCm devices present)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
    log "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    die "No GPU detected. The agent requires a GPU for VRAM monitoring."
fi

PROFILE="$GPU_VENDOR"

# ── Configure .env ───────────────────────────────────────────────────────────

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    log "No .env found — creating one."

    PSK="${LLM_AGENT_PSK:-}"
    if [[ -z "$PSK" ]]; then
        read -rp "Enter agent PSK (from Bitwarden: llm-manager-agent-psk): " PSK
    else
        log "Using PSK from \$LLM_AGENT_PSK"
    fi
    [[ -z "$PSK" ]] && die "PSK is required."

    read -rp "Backend URL [https://llm-manager-backend.amer.dev]: " BACKEND
    BACKEND="${BACKEND:-https://llm-manager-backend.amer.dev}"

    cat > "$SCRIPT_DIR/.env" <<EOF
LLM_MANAGER_AGENT_PSK=${PSK}
BACKEND_URL=${BACKEND}
EOF

    # AMD-specific: prompt for GFX version override
    if [[ "$GPU_VENDOR" == "amd" ]]; then
        echo ""
        warn "AMD GPUs may need HSA_OVERRIDE_GFX_VERSION set for ROCm compatibility."
        warn "Common values: 11.0.0 (RDNA 4 / RX 9070), 10.3.0 (RDNA 2), 11.0.0 (RDNA 3)"
        read -rp "HSA_OVERRIDE_GFX_VERSION (leave blank to skip): " GFX_VER
        if [[ -n "$GFX_VER" ]]; then
            echo "HSA_OVERRIDE_GFX_VERSION=${GFX_VER}" >> "$SCRIPT_DIR/.env"
        fi
    fi

    log ".env created."
else
    log "Using existing .env"
fi

# ── Pull and start ───────────────────────────────────────────────────────────

log "Pulling agent image (profile: $PROFILE)..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" pull

log "Starting llm-agent..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d

# ── Health check ─────────────────────────────────────────────────────────────

log "Waiting for agent to become healthy..."
for i in $(seq 1 30); do
    if curl -sf https://localhost:8090/health --insecure >/dev/null 2>&1; then
        echo ""
        log "llm-agent is up!"
        curl -s https://localhost:8090/health --insecure | python3 -m json.tool 2>/dev/null || true
        break
    fi
    echo -n "."
    sleep 2
done

if ! curl -sf https://localhost:8090/health --insecure >/dev/null 2>&1; then
    warn "Agent didn't respond after 60s. Check: docker logs llm-agent"
fi

echo ""
log "Done! (GPU: $GPU_VENDOR, profile: $PROFILE)"
echo ""
echo "  Health:   curl -sk https://localhost:8090/health"
echo "  Logs:     docker logs -f llm-agent"
echo "  Stop:     docker compose -f $COMPOSE_FILE --profile $PROFILE down"
echo "  Restart:  docker compose -f $COMPOSE_FILE --profile $PROFILE restart"
echo ""
