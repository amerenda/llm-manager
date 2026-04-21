#!/usr/bin/env bash
# install.sh — Set up the llm-agent (and optionally a docker-managed Ollama)
# on a GPU host.
#
# Prerequisites:
#   - Docker + Docker Compose plugin installed
#   - (Legacy mode) Ollama already running on the host at localhost:11434
#   - (Managed mode, default for new installs) No Ollama running, or a
#     systemd ollama.service we can safely take over
#
# Tested on Arch and Debian; expected to work on any modern systemd distro.
# AMD/NVIDIA GPU is auto-detected.
#
# Flags:
#   --update, -u          Re-pull agent image only; preserve existing config
#   --psk <value>         Rotate the agent PSK in the existing .env
#   --model-storage <p>   Path where Ollama models live (bind-mounted)
#   --install-dir <p>     Install into a directory other than the script dir
#   --managed-ollama      Run Ollama in Docker (default for fresh installs)
#   --host-ollama         Skip Docker Ollama; expect Ollama on localhost:11434
#   --migrate             Stop & disable host systemd ollama.service without prompting
#   --ollama-tag <t>      Override Ollama image tag (default: 0.21.0 / 0.21.0-rocm)
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CLI ──────────────────────────────────────────────────────────────────────

INSTALL_DIR="$SCRIPT_DIR"
UPDATE_ONLY=false
NEW_PSK=""
MODEL_STORAGE=""
MANAGED_OLLAMA=""     # empty=auto, true, false
MIGRATE=false
OLLAMA_TAG=""

DEFAULT_OLLAMA_TAG_NVIDIA="0.21.0"
DEFAULT_OLLAMA_TAG_AMD="0.21.0-rocm"

usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -35
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update|-u) UPDATE_ONLY=true; shift ;;
        --psk) NEW_PSK="$2"; shift 2 ;;
        --psk=*) NEW_PSK="${1#--psk=}"; shift ;;
        --model-storage) MODEL_STORAGE="$2"; shift 2 ;;
        --model-storage=*) MODEL_STORAGE="${1#--model-storage=}"; shift ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        --install-dir=*) INSTALL_DIR="${1#--install-dir=}"; shift ;;
        --managed-ollama) MANAGED_OLLAMA=true; shift ;;
        --host-ollama) MANAGED_OLLAMA=false; shift ;;
        --migrate) MIGRATE=true; shift ;;
        --ollama-tag) OLLAMA_TAG="$2"; shift 2 ;;
        --ollama-tag=*) OLLAMA_TAG="${1#--ollama-tag=}"; shift ;;
        -h|--help) usage ;;
        *) echo "unknown flag: $1" >&2; usage ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Prerequisite checks ──────────────────────────────────────────────────────

log "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install from https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is not installed."
docker info >/dev/null 2>&1 || die "Docker daemon is not running (or you lack permission to use it)."

# ── Install dir setup ────────────────────────────────────────────────────────
# Users can point install.sh at a separate dir (e.g. /opt/llm-agent) so the
# git checkout doesn't need to match the deployment location. Compose files
# are copied into INSTALL_DIR and run from there.

mkdir -p "$INSTALL_DIR"
INSTALL_DIR="$(cd "$INSTALL_DIR" && pwd)"

copy_if_newer() {
    # $1=src $2=dst — copy src to dst if dst doesn't exist or src is newer.
    local src="$1" dst="$2"
    if [[ ! -f "$dst" ]] || [[ "$src" -nt "$dst" ]]; then
        cp "$src" "$dst"
    fi
}

if [[ "$INSTALL_DIR" != "$SCRIPT_DIR" ]]; then
    log "Installing into $INSTALL_DIR (separate from checkout at $SCRIPT_DIR)"
fi

copy_if_newer "$SCRIPT_DIR/compose.yaml" "$INSTALL_DIR/compose.yaml"
copy_if_newer "$SCRIPT_DIR/ollama.env.example" "$INSTALL_DIR/ollama.env.example"

COMPOSE_FILE="$INSTALL_DIR/compose.yaml"
ENV_FILE="$INSTALL_DIR/.env"
OLLAMA_ENV_FILE="$INSTALL_DIR/ollama.env"

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

# ── Decide managed vs host Ollama ────────────────────────────────────────────
# Default rules:
#   - If .env doesn't exist (fresh install) → managed Ollama
#   - If .env exists (existing agent) → host Ollama unless --managed-ollama is passed
# --managed-ollama / --host-ollama override.

if [[ -z "$MANAGED_OLLAMA" ]]; then
    if [[ -f "$ENV_FILE" ]]; then
        MANAGED_OLLAMA=false
    else
        MANAGED_OLLAMA=true
    fi
fi

if "$MANAGED_OLLAMA"; then
    log "Mode: managed Ollama (Docker)"
    case "$GPU_VENDOR" in
        nvidia) PROFILE="nvidia-full" ;;
        amd)    PROFILE="amd-full" ;;
    esac
    [[ -z "$OLLAMA_TAG" ]] && OLLAMA_TAG="$([[ $GPU_VENDOR == amd ]] && echo $DEFAULT_OLLAMA_TAG_AMD || echo $DEFAULT_OLLAMA_TAG_NVIDIA)"
else
    log "Mode: host-managed Ollama"
    PROFILE="$GPU_VENDOR"
fi

# ── Handle existing host Ollama ──────────────────────────────────────────────
# If we're switching to managed Ollama, the host's systemd or port-11434
# Ollama has to step aside.

stop_host_ollama() {
    local systemd_ollama_active=false
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
        systemd_ollama_active=true
    fi

    if "$systemd_ollama_active"; then
        if ! "$MIGRATE"; then
            warn "Host systemd ollama.service is active and would conflict with Docker Ollama."
            read -rp "Stop and disable it now? [y/N] " ans
            case "$ans" in
                y|Y|yes) ;;
                *) die "Refusing to proceed while host Ollama is running. Re-run with --migrate to auto-stop, or --host-ollama to skip managed mode." ;;
            esac
        fi
        log "Stopping and disabling systemd ollama.service..."
        sudo systemctl stop ollama
        sudo systemctl disable ollama || true
        log "Host Ollama stopped. (Re-enable later with: sudo systemctl enable --now ollama)"
    fi

    # Port conflict check — covers non-systemd Ollamas (docker run without
    # compose, bare binary, etc.)
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        # Is it already a Docker container named 'ollama'? If so, remove it so
        # compose can create a fresh one with the new env. Just "restart" or
        # "compose up" can't change env on an existing container, and compose
        # refuses to create over the name conflict.
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx ollama; then
            log "Removing existing 'ollama' container so compose can recreate it..."
            docker rm -f ollama >/dev/null 2>&1 || true
        else
            die "Port 11434 is still in use after stopping systemd. Stop whatever is listening and re-run."
        fi
    else
        # Port is free but a stopped ollama container may linger with the name
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx ollama; then
            log "Removing stopped 'ollama' container so compose can recreate it..."
            docker rm -f ollama >/dev/null 2>&1 || true
        fi
    fi
}

if "$MANAGED_OLLAMA"; then
    stop_host_ollama
elif ! "$UPDATE_ONLY"; then
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        die "Ollama is not running on localhost:11434 and --host-ollama was chosen. Start Ollama or rerun with --managed-ollama."
    fi
    log "Host Ollama is running."
fi

# ── Models path resolution ──────────────────────────────────────────────────
# Priority: --model-storage → MODEL_STORAGE_PATH env → existing .env →
# detected systemd/user path → /opt/ollama/models default.

resolve_models_path() {
    if [[ -n "$MODEL_STORAGE" ]]; then echo "$MODEL_STORAGE"; return; fi
    if [[ -n "${MODEL_STORAGE_PATH:-}" ]]; then echo "$MODEL_STORAGE_PATH"; return; fi
    if [[ -f "$ENV_FILE" ]]; then
        local from_env
        from_env=$(grep -oP '^OLLAMA_MODELS_PATH=\K.*' "$ENV_FILE" | tr -d '"' || true)
        if [[ -n "$from_env" ]]; then echo "$from_env"; return; fi
    fi
    if [[ -n "${OLLAMA_MODELS:-}" ]]; then echo "$OLLAMA_MODELS"; return; fi
    # systemd drop-in may have set it
    if command -v systemctl >/dev/null 2>&1; then
        local sd
        sd=$(systemctl show ollama -p Environment 2>/dev/null | grep -oP 'OLLAMA_MODELS=\K[^ ]+' || true)
        if [[ -n "$sd" ]]; then echo "$sd"; return; fi
    fi
    if [[ -d "/usr/share/ollama/.ollama/models" ]]; then echo "/usr/share/ollama/.ollama/models"; return; fi
    if [[ -d "$HOME/.ollama/models" ]]; then echo "$HOME/.ollama/models"; return; fi
    echo "/opt/ollama/models"
}

OLLAMA_MODELS_PATH="$(resolve_models_path)"
log "Ollama models path: $OLLAMA_MODELS_PATH"

if [[ ! -d "$OLLAMA_MODELS_PATH" ]]; then
    log "Creating $OLLAMA_MODELS_PATH..."
    if [[ "$OLLAMA_MODELS_PATH" == /opt/* ]] || [[ "$OLLAMA_MODELS_PATH" == /usr/* ]]; then
        sudo mkdir -p "$OLLAMA_MODELS_PATH"
        sudo chown "$(id -u):$(id -g)" "$OLLAMA_MODELS_PATH"
    else
        mkdir -p "$OLLAMA_MODELS_PATH"
    fi
fi

# ── Configure .env ──────────────────────────────────────────────────────────
# .env is authoritative for agent creds + host-level paths. Ollama tunables
# live in ollama.env (separate file so the agent can rewrite it safely).

if [[ -n "$NEW_PSK" ]] && [[ -f "$ENV_FILE" ]]; then
    sed -i "s|^LLM_MANAGER_AGENT_PSK=.*|LLM_MANAGER_AGENT_PSK=${NEW_PSK}|" "$ENV_FILE"
    log "Updated PSK in existing .env"
fi

if [[ ! -f "$ENV_FILE" ]]; then
    log "No .env found — creating one."

    PSK="${LLM_AGENT_PSK:-}"
    if [[ -z "$PSK" ]]; then
        read -rp "Enter agent PSK (from Bitwarden: llm-manager-agent-psk): " PSK
    else
        log "Using PSK from \$LLM_AGENT_PSK"
    fi
    [[ -z "$PSK" ]] && die "PSK is required."

    BACKEND="${BACKEND_URL:-https://llm-manager-backend.amer.dev}"
    log "Backend URL: $BACKEND"

    cat > "$ENV_FILE" <<EOF
LLM_MANAGER_AGENT_PSK=${PSK}
BACKEND_URL=${BACKEND}
OLLAMA_MODELS_PATH=${OLLAMA_MODELS_PATH}
MODEL_STORAGE_PATH=${OLLAMA_MODELS_PATH}
COMPOSE_DIR=${INSTALL_DIR}
COMPOSE_PROFILE=${PROFILE}
EOF

    # AMD-specific: auto-detect GFX version and set override for ROCm compatibility
    if [[ "$GPU_VENDOR" == "amd" ]]; then
        GFX_VER=""
        for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            ver=$(awk '/gfx_target_version/ {print $2}' "$props" 2>/dev/null)
            if [[ -n "$ver" ]] && [[ "$ver" != "0" ]]; then
                major=$(( ver / 10000 ))
                minor=$(( (ver % 10000) / 100 ))
                patch=$(( ver % 100 ))
                GFX_VER="${major}.${minor}.${patch}"
                break
            fi
        done

        if [[ -n "$GFX_VER" ]]; then
            log "Detected GPU GFX version: $GFX_VER"
            # RDNA 4 (gfx12) isn't supported by ROCm yet — override to 11.0.0 (RDNA 3)
            if [[ "$GFX_VER" == 12.* ]]; then
                warn "RDNA 4 detected (gfx $GFX_VER) — overriding to 11.0.0 for ROCm compatibility"
                echo "HSA_OVERRIDE_GFX_VERSION=11.0.0" >> "$ENV_FILE"
            else
                echo "HSA_OVERRIDE_GFX_VERSION=${GFX_VER}" >> "$ENV_FILE"
            fi
        else
            warn "Could not detect GFX version — skipping HSA_OVERRIDE_GFX_VERSION"
        fi
    fi

    log ".env created."
else
    log "Using existing .env"
    # Idempotent refresh: ensure keys the new compose.yaml needs are present.
    upsert_env() {
        local key="$1" val="$2"
        if grep -q "^${key}=" "$ENV_FILE"; then
            return  # already set — don't clobber user edits
        fi
        echo "${key}=${val}" >> "$ENV_FILE"
        log "Added ${key} to .env"
    }
    upsert_env COMPOSE_DIR "$INSTALL_DIR"
    upsert_env COMPOSE_PROFILE "$PROFILE"
    upsert_env OLLAMA_MODELS_PATH "$OLLAMA_MODELS_PATH"
    upsert_env MODEL_STORAGE_PATH "$OLLAMA_MODELS_PATH"
fi

# Host GIDs for video/render groups (AMD only — ollama/ollama:*-rocm image
# doesn't define these groups, so we must pass numeric GIDs for /dev/dri
# device perms to work). NVIDIA path uses device_requests and doesn't need
# supplementary groups.
if [[ "$GPU_VENDOR" == "amd" ]]; then
    upsert_gid_env() {
        local key="$1" group="$2"
        local gid
        gid=$(getent group "$group" 2>/dev/null | cut -d: -f3 || true)
        if [[ -z "$gid" ]]; then
            warn "group '$group' not found on host — leaving ${key} unset (compose default will be used)"
            return
        fi
        # Rewrite if present, add if missing — GIDs should track the host, not
        # be treated as immutable user edits.
        if grep -q "^${key}=" "$ENV_FILE"; then
            sed -i "s|^${key}=.*|${key}=${gid}|" "$ENV_FILE"
        else
            echo "${key}=${gid}" >> "$ENV_FILE"
        fi
    }
    upsert_gid_env VIDEO_GID  video
    upsert_gid_env RENDER_GID render
    log "AMD host GIDs: video=$(grep -oP '^VIDEO_GID=\K.*' "$ENV_FILE" 2>/dev/null || echo '?'), render=$(grep -oP '^RENDER_GID=\K.*' "$ENV_FILE" 2>/dev/null || echo '?')"
fi

# Pin the Ollama image tag in .env so re-runs and `compose up` inherit it
if "$MANAGED_OLLAMA"; then
    if [[ "$GPU_VENDOR" == "amd" ]]; then
        key="OLLAMA_AMD_IMAGE_TAG"
    else
        key="OLLAMA_IMAGE_TAG"
    fi
    if grep -q "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${OLLAMA_TAG}|" "$ENV_FILE"
    else
        echo "${key}=${OLLAMA_TAG}" >> "$ENV_FILE"
    fi
    log "Pinned Ollama image: ollama/ollama:${OLLAMA_TAG}"
fi

# ── Configure ollama.env (managed mode only) ─────────────────────────────────

if "$MANAGED_OLLAMA"; then
    if [[ ! -f "$OLLAMA_ENV_FILE" ]]; then
        log "Creating ollama.env from ollama.env.example (all defaults commented out)."
        cp "$INSTALL_DIR/ollama.env.example" "$OLLAMA_ENV_FILE"
    else
        log "Preserving existing ollama.env tunings."
    fi
fi

# ── Pull and start ──────────────────────────────────────────────────────────

# Remove any stale llm-agent container from a previous profile/compose file
if docker inspect llm-agent >/dev/null 2>&1; then
    prev_labels=$(docker inspect llm-agent --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' 2>/dev/null || true)
    if [[ -n "$prev_labels" ]] && [[ "$prev_labels" != "$INSTALL_DIR" ]]; then
        log "Removing llm-agent container from previous install dir ($prev_labels)..."
        docker rm -f llm-agent >/dev/null 2>&1
    fi
fi

log "Pulling images (profile: $PROFILE)..."
(cd "$INSTALL_DIR" && docker compose --profile "$PROFILE" pull)

log "Starting services..."
(cd "$INSTALL_DIR" && docker compose --profile "$PROFILE" up -d)

# ── Health checks ────────────────────────────────────────────────────────────

log "Waiting for agent to become healthy..."
agent_ok=false
for _ in $(seq 1 30); do
    if curl -sf https://localhost:8090/health --insecure >/dev/null 2>&1; then
        agent_ok=true; echo ""; break
    fi
    echo -n "."
    sleep 2
done

if "$agent_ok"; then
    log "llm-agent is up!"
    curl -s https://localhost:8090/health --insecure | python3 -m json.tool 2>/dev/null || true
else
    warn "Agent didn't respond after 60s. Check: docker logs llm-agent"
fi

if "$MANAGED_OLLAMA"; then
    log "Waiting for Ollama..."
    for _ in $(seq 1 30); do
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            log "Ollama is up!"
            break
        fi
        echo -n "."
        sleep 2
    done
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        warn "Ollama didn't respond after 60s. Check: docker logs ollama"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
log "Done! (GPU: $GPU_VENDOR, profile: $PROFILE, install dir: $INSTALL_DIR)"
echo ""
echo "  Health:   curl -sk https://localhost:8090/health"
echo "  Logs:     docker logs -f llm-agent"
if "$MANAGED_OLLAMA"; then
    echo "  Ollama:   docker logs -f ollama"
    echo "  Tunings:  edit $OLLAMA_ENV_FILE then rerun: docker compose -f $COMPOSE_FILE --profile $PROFILE up -d --force-recreate ollama"
fi
echo "  Stop:     cd $INSTALL_DIR && docker compose --profile $PROFILE down"
echo "  Restart:  cd $INSTALL_DIR && docker compose --profile $PROFILE restart"
echo ""
