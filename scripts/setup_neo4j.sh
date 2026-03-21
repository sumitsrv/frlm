#!/usr/bin/env bash
# =============================================================================
# setup_neo4j.sh — Install and configure Neo4j for the FRLM project.
#
# Supports two methods:
#   1. Docker (recommended) — no system-level install needed
#   2. Native (apt)         — installs Neo4j Community via the official repo
#
# Usage:
#   ./scripts/setup_neo4j.sh docker   # Docker method (default)
#   ./scripts/setup_neo4j.sh native   # Native apt install
#   ./scripts/setup_neo4j.sh status   # Check if Neo4j is running
#   ./scripts/setup_neo4j.sh stop     # Stop the Docker container
#
# The script reads FRLM_NEO4J_PASSWORD from the environment (or uses default).
# =============================================================================

set -euo pipefail

# ── Defaults (match config/default.yaml) ─────────────────────────────────────
NEO4J_CONTAINER_NAME="frlm-neo4j"
NEO4J_VERSION="5.26.0"
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687
NEO4J_DB_NAME="frlm"
NEO4J_PASSWORD="${FRLM_NEO4J_PASSWORD:-frlm_dev_password}"
NEO4J_DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/neo4j_data"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Helpers ──────────────────────────────────────────────────────────────────
wait_for_neo4j() {
    local max_wait=60
    local waited=0
    info "Waiting for Neo4j to become ready (up to ${max_wait}s)..."
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${NEO4J_HTTP_PORT}" > /dev/null 2>&1; then
            ok "Neo4j is ready!"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "."
    done
    echo ""
    err "Neo4j did not start within ${max_wait}s."
    return 1
}

create_frlm_database() {
    info "Creating database '${NEO4J_DB_NAME}' (if it doesn't exist)..."
    # Neo4j Community Edition only supports the default 'neo4j' database.
    # If you're running Enterprise, uncomment the cypher-shell call below.
    # For Community, we'll use the default 'neo4j' database and note this.
    if docker exec "${NEO4J_CONTAINER_NAME}" neo4j-admin dbms list-databases 2>/dev/null | grep -q "${NEO4J_DB_NAME}"; then
        ok "Database '${NEO4J_DB_NAME}' already exists."
    else
        warn "Neo4j Community Edition only supports the default 'neo4j' database."
        warn "Your FRLM config should set neo4j.database to 'neo4j' (or use Enterprise for a custom DB name)."
        warn "Proceeding with the default 'neo4j' database."
    fi
}

print_connection_info() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Neo4j is running!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Browser UI:   ${CYAN}http://localhost:${NEO4J_HTTP_PORT}${NC}"
    echo -e "  Bolt URI:     ${CYAN}bolt://localhost:${NEO4J_BOLT_PORT}${NC}"
    echo -e "  Username:     ${CYAN}neo4j${NC}"
    echo -e "  Password:     ${CYAN}${NEO4J_PASSWORD}${NC}"
    echo -e "  Database:     ${CYAN}neo4j${NC}  (Community Edition default)"
    echo ""
    echo -e "  ${YELLOW}Set these in your environment:${NC}"
    echo ""
    echo -e "    export FRLM_NEO4J_PASSWORD=\"${NEO4J_PASSWORD}\""
    echo ""
    echo -e "  ${YELLOW}Or add to a .env file:${NC}"
    echo ""
    echo -e "    FRLM_NEO4J_PASSWORD=${NEO4J_PASSWORD}"
    echo ""
    echo -e "  ${YELLOW}Update config/default.yaml if needed:${NC}"
    echo ""
    echo -e "    neo4j.database: \"neo4j\"   # Community Edition uses the default DB"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
}

# ── Docker method ────────────────────────────────────────────────────────────
install_docker() {
    info "Setting up Neo4j ${NEO4J_VERSION} via Docker..."

    # Check Docker is available
    if ! command -v docker &> /dev/null; then
        err "Docker is not installed. Install it first:"
        echo ""
        echo "  # Ubuntu/Debian:"
        echo "  curl -fsSL https://get.docker.com | sh"
        echo "  sudo usermod -aG docker \$USER"
        echo "  # Then log out and back in"
        echo ""
        echo "  # Or use the native method instead:"
        echo "  ./scripts/setup_neo4j.sh native"
        exit 1
    fi

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER_NAME}$"; then
            ok "Neo4j container '${NEO4J_CONTAINER_NAME}' is already running."
            print_connection_info
            return 0
        else
            info "Starting existing container '${NEO4J_CONTAINER_NAME}'..."
            docker start "${NEO4J_CONTAINER_NAME}"
            wait_for_neo4j
            print_connection_info
            return 0
        fi
    fi

    # Create data directory for persistence
    mkdir -p "${NEO4J_DATA_DIR}/data" "${NEO4J_DATA_DIR}/logs" "${NEO4J_DATA_DIR}/plugins"

    info "Pulling Neo4j ${NEO4J_VERSION} image..."
    docker pull "neo4j:${NEO4J_VERSION}"

    info "Starting Neo4j container..."
    docker run -d \
        --name "${NEO4J_CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "${NEO4J_HTTP_PORT}:7474" \
        -p "${NEO4J_BOLT_PORT}:7687" \
        -v "${NEO4J_DATA_DIR}/data:/data" \
        -v "${NEO4J_DATA_DIR}/logs:/logs" \
        -v "${NEO4J_DATA_DIR}/plugins:/plugins" \
        -e "NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}" \
        -e "NEO4J_PLUGINS=[\"apoc\"]" \
        -e "NEO4J_dbms_security_procedures_unrestricted=apoc.*" \
        -e "NEO4J_dbms_memory_heap_initial__size=512m" \
        -e "NEO4J_dbms_memory_heap_max__size=2g" \
        -e "NEO4J_dbms_memory_pagecache_size=1g" \
        "neo4j:${NEO4J_VERSION}"

    wait_for_neo4j
    create_frlm_database
    print_connection_info
}

# ── Native apt method ────────────────────────────────────────────────────────
install_native() {
    info "Setting up Neo4j via native apt install..."

    # Check for Java 17+
    if ! command -v java &> /dev/null; then
        info "Installing Java 17 (required by Neo4j 5.x)..."
        sudo apt-get update
        sudo apt-get install -y openjdk-17-jre-headless
    fi

    JAVA_VER=$(java -version 2>&1 | head -1 | awk -F '"' '{print $2}' | cut -d. -f1)
    if [ "${JAVA_VER}" -lt 17 ]; then
        err "Neo4j 5.x requires Java 17+. Found Java ${JAVA_VER}."
        err "Install Java 17: sudo apt install openjdk-17-jre-headless"
        exit 1
    fi
    ok "Java ${JAVA_VER} found."

    # Add Neo4j apt repository
    if [ ! -f /etc/apt/sources.list.d/neo4j.list ]; then
        info "Adding Neo4j apt repository..."
        # Import the GPG key
        curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j-archive-keyring.gpg
        # Add the repository
        echo "deb [signed-by=/usr/share/keyrings/neo4j-archive-keyring.gpg] https://debian.neo4j.com stable latest" | \
            sudo tee /etc/apt/sources.list.d/neo4j.list > /dev/null
    fi

    info "Installing Neo4j Community Edition..."
    sudo apt-get update
    sudo apt-get install -y neo4j

    # Set the initial password
    info "Setting Neo4j password..."
    sudo neo4j-admin dbms set-initial-password "${NEO4J_PASSWORD}" 2>/dev/null || \
        warn "Password may already be set. Use the Neo4j Browser to change it if needed."

    # Enable and start the service
    info "Starting Neo4j service..."
    sudo systemctl enable neo4j
    sudo systemctl start neo4j

    # Wait for it
    local max_wait=60
    local waited=0
    info "Waiting for Neo4j to become ready..."
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${NEO4J_HTTP_PORT}" > /dev/null 2>&1; then
            ok "Neo4j is ready!"
            print_connection_info
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "."
    done
    echo ""
    err "Neo4j did not start within ${max_wait}s. Check: sudo systemctl status neo4j"
    exit 1
}

# ── Status check ─────────────────────────────────────────────────────────────
check_status() {
    echo ""
    info "Checking Neo4j status..."
    echo ""

    # Check Docker method
    if command -v docker &> /dev/null; then
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${NEO4J_CONTAINER_NAME}$"; then
            ok "Docker container '${NEO4J_CONTAINER_NAME}' is RUNNING"
            docker ps --filter "name=${NEO4J_CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${NEO4J_CONTAINER_NAME}$"; then
            warn "Docker container '${NEO4J_CONTAINER_NAME}' exists but is STOPPED"
            echo "  Start it with: docker start ${NEO4J_CONTAINER_NAME}"
        fi
    fi

    # Check native method
    if command -v neo4j &> /dev/null; then
        if systemctl is-active --quiet neo4j 2>/dev/null; then
            ok "Neo4j system service is RUNNING"
        else
            warn "Neo4j is installed but the service is NOT RUNNING"
            echo "  Start it with: sudo systemctl start neo4j"
        fi
    fi

    # Check connectivity
    echo ""
    if curl -s "http://localhost:${NEO4J_HTTP_PORT}" > /dev/null 2>&1; then
        ok "Neo4j HTTP endpoint is reachable at http://localhost:${NEO4J_HTTP_PORT}"
    else
        err "Cannot reach Neo4j at http://localhost:${NEO4J_HTTP_PORT}"
    fi
}

# ── Stop ─────────────────────────────────────────────────────────────────────
stop_neo4j() {
    info "Stopping Neo4j..."

    if command -v docker &> /dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${NEO4J_CONTAINER_NAME}$"; then
        docker stop "${NEO4J_CONTAINER_NAME}"
        ok "Docker container '${NEO4J_CONTAINER_NAME}' stopped."
    elif systemctl is-active --quiet neo4j 2>/dev/null; then
        sudo systemctl stop neo4j
        ok "Neo4j system service stopped."
    else
        warn "No running Neo4j instance found."
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [docker|native|status|stop]"
    echo ""
    echo "  docker  — Run Neo4j in a Docker container (default, recommended)"
    echo "  native  — Install Neo4j natively via apt"
    echo "  status  — Check if Neo4j is running"
    echo "  stop    — Stop the Neo4j instance"
}

case "${1:-docker}" in
    docker)  install_docker ;;
    native)  install_native ;;
    status)  check_status ;;
    stop)    stop_neo4j ;;
    -h|--help|help) usage ;;
    *)
        err "Unknown command: $1"
        usage
        exit 1
        ;;
esac

