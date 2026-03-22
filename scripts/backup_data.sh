#!/usr/bin/env bash
# =============================================================================
# backup_data.sh — Backup all generated FRLM pipeline data.
#
# Creates timestamped compressed archives of pipeline artifacts.
# Supports three modes:
#   1. local   — tar.gz to a local backup directory (default)
#   2. gcs     — upload to Google Cloud Storage bucket
#   3. s3      — upload to AWS S3 bucket
#
# Usage:
#   ./scripts/backup_data.sh                       # local backup
#   ./scripts/backup_data.sh local /mnt/backup     # local backup to custom dir
#   ./scripts/backup_data.sh gcs gs://my-bucket/frlm
#   ./scripts/backup_data.sh s3 s3://my-bucket/frlm
#   ./scripts/backup_data.sh restore <archive>     # restore from archive
#
# What gets backed up (in priority order):
#   CRITICAL  — data/processed/, data/labels/     (Claude API cost to regenerate)
#   IMPORTANT — neo4j_data/                        (hours to regenerate)
#   MODERATE  — data/kg/, data/faiss_indices/      (minutes to regenerate)
#   LOW       — data/corpus/, cache/               (free to re-download)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_BACKUP_DIR="${PROJECT_ROOT}/backups"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Backup targets ───────────────────────────────────────────────────────────
# Critical: expensive API calls to regenerate
CRITICAL_DIRS=(
    "data/processed"
    "data/labels"
)

# Important: time-consuming to regenerate
IMPORTANT_DIRS=(
    "neo4j_data"
    "data/kg"
)

# Moderate: quick to regenerate but convenient to have
MODERATE_DIRS=(
    "data/faiss_indices"
)

# Low priority: free to re-download
LOW_DIRS=(
    "data/corpus"
    "cache"
)

# ── Size report ──────────────────────────────────────────────────────────────
show_sizes() {
    info "Current data sizes:"
    local total=0
    for dir in "${CRITICAL_DIRS[@]}" "${IMPORTANT_DIRS[@]}" "${MODERATE_DIRS[@]}" "${LOW_DIRS[@]}"; do
        local full="${PROJECT_ROOT}/${dir}"
        if [ -d "$full" ]; then
            local size
            size=$(du -sh "$full" 2>/dev/null | cut -f1)
            printf "  %-30s %s\n" "$dir/" "$size"
        fi
    done
    echo ""
    du -sh "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/neo4j_data" "${PROJECT_ROOT}/cache" --total 2>/dev/null | tail -1 | awk '{print "  TOTAL: " $1}'
}

# ── Local backup ─────────────────────────────────────────────────────────────
backup_local() {
    local backup_dir="${1:-$DEFAULT_BACKUP_DIR}"
    mkdir -p "$backup_dir"

    info "Backing up to: ${backup_dir}"
    show_sizes

    # Critical data (always backed up)
    local critical_archive="${backup_dir}/frlm_critical_${TIMESTAMP}.tar.gz"
    info "Creating CRITICAL backup: ${critical_archive}"
    local critical_args=()
    for dir in "${CRITICAL_DIRS[@]}"; do
        [ -d "${PROJECT_ROOT}/${dir}" ] && critical_args+=("$dir")
    done
    if [ ${#critical_args[@]} -gt 0 ]; then
        tar -czf "$critical_archive" -C "$PROJECT_ROOT" "${critical_args[@]}"
        ok "Critical backup: $(du -sh "$critical_archive" | cut -f1)"
    else
        warn "No critical directories found to backup"
    fi

    # Full backup (everything)
    local full_archive="${backup_dir}/frlm_full_${TIMESTAMP}.tar.gz"
    info "Creating FULL backup: ${full_archive}"
    local all_args=()
    for dir in "${CRITICAL_DIRS[@]}" "${IMPORTANT_DIRS[@]}" "${MODERATE_DIRS[@]}"; do
        [ -d "${PROJECT_ROOT}/${dir}" ] && all_args+=("$dir")
    done
    if [ ${#all_args[@]} -gt 0 ]; then
        tar -czf "$full_archive" -C "$PROJECT_ROOT" "${all_args[@]}"
        ok "Full backup: $(du -sh "$full_archive" | cut -f1)"
    fi

    # Neo4j dump (if running)
    if command -v docker &>/dev/null; then
        local neo4j_container
        neo4j_container=$(docker ps --filter "name=neo4j\|frlm-neo4j" --format "{{.Names}}" 2>/dev/null | head -1)
        if [ -n "$neo4j_container" ]; then
            info "Creating Neo4j database dump..."
            local neo4j_dump="${backup_dir}/frlm_neo4j_dump_${TIMESTAMP}.tar.gz"
            tar -czf "$neo4j_dump" -C "$PROJECT_ROOT" neo4j_data/
            ok "Neo4j dump: $(du -sh "$neo4j_dump" | cut -f1)"
        fi
    fi

    # Manifest
    local manifest="${backup_dir}/manifest_${TIMESTAMP}.txt"
    {
        echo "FRLM Backup Manifest"
        echo "===================="
        echo "Timestamp: ${TIMESTAMP}"
        echo "Date: $(date -Iseconds)"
        echo "Host: $(hostname)"
        echo ""
        echo "Files:"
        ls -lh "${backup_dir}/"*"${TIMESTAMP}"* 2>/dev/null
        echo ""
        echo "Pipeline state:"
        echo "  Corpus files: $(ls "${PROJECT_ROOT}/data/corpus/"*.xml 2>/dev/null | wc -l)"
        echo "  Entity files: $(ls "${PROJECT_ROOT}/data/processed/entities_"*.json 2>/dev/null | wc -l)"
        echo "  Relation files: $(ls "${PROJECT_ROOT}/data/processed/relations_"*.json 2>/dev/null | wc -l)"
        echo "  Label files: $(ls "${PROJECT_ROOT}/data/labels/labels_"*.json 2>/dev/null | wc -l)"
        echo "  FAISS indices: $(ls "${PROJECT_ROOT}/data/faiss_indices/"*.faiss 2>/dev/null | wc -l)"
        echo "  KG facts exported: $(python3 -c "import json; print(len(json.load(open('${PROJECT_ROOT}/data/kg/exported_facts.json'))))" 2>/dev/null || echo "N/A")"
    } > "$manifest"
    ok "Manifest written: ${manifest}"

    echo ""
    info "=== Backup Summary ==="
    ls -lh "${backup_dir}/"*"${TIMESTAMP}"*
    echo ""
    ok "All backups saved to: ${backup_dir}"

    # Cleanup old backups (keep last 5)
    local old_count
    old_count=$(ls -1 "${backup_dir}/frlm_full_"*.tar.gz 2>/dev/null | wc -l)
    if [ "$old_count" -gt 5 ]; then
        info "Cleaning up old backups (keeping last 5)..."
        ls -1t "${backup_dir}/frlm_full_"*.tar.gz | tail -n +6 | xargs rm -f
        ls -1t "${backup_dir}/frlm_critical_"*.tar.gz | tail -n +6 | xargs rm -f
        ls -1t "${backup_dir}/frlm_neo4j_dump_"*.tar.gz 2>/dev/null | tail -n +6 | xargs rm -f
        ls -1t "${backup_dir}/manifest_"*.txt | tail -n +6 | xargs rm -f
    fi
}

# ── Cloud upload (GCS) ───────────────────────────────────────────────────────
backup_gcs() {
    local gcs_path="${1:?Usage: backup_data.sh gcs gs://bucket/path}"

    if ! command -v gsutil &>/dev/null; then
        err "gsutil not found. Install: pip install google-cloud-storage"
        exit 1
    fi

    # First create local backup
    backup_local "${DEFAULT_BACKUP_DIR}"

    # Upload
    info "Uploading to GCS: ${gcs_path}"
    gsutil -m cp "${DEFAULT_BACKUP_DIR}/"*"${TIMESTAMP}"* "${gcs_path}/"
    ok "Uploaded to ${gcs_path}"
}

# ── Cloud upload (S3) ────────────────────────────────────────────────────────
backup_s3() {
    local s3_path="${1:?Usage: backup_data.sh s3 s3://bucket/path}"

    if ! command -v aws &>/dev/null; then
        err "aws CLI not found. Install: pip install awscli"
        exit 1
    fi

    # First create local backup
    backup_local "${DEFAULT_BACKUP_DIR}"

    # Upload
    info "Uploading to S3: ${s3_path}"
    aws s3 cp "${DEFAULT_BACKUP_DIR}/" "${s3_path}/" --recursive --exclude "*" \
        --include "*${TIMESTAMP}*"
    ok "Uploaded to ${s3_path}"
}

# ── Restore ──────────────────────────────────────────────────────────────────
restore() {
    local archive="${1:?Usage: backup_data.sh restore <archive.tar.gz>}"

    if [ ! -f "$archive" ]; then
        err "Archive not found: $archive"
        exit 1
    fi

    warn "This will overwrite existing data in ${PROJECT_ROOT}"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Restore cancelled."
        exit 0
    fi

    info "Restoring from: ${archive}"
    tar -xzf "$archive" -C "$PROJECT_ROOT"
    ok "Restore complete!"

    info "Restored contents:"
    tar -tzf "$archive" | head -20
}

# ── Main ─────────────────────────────────────────────────────────────────────
case "${1:-local}" in
    local)
        backup_local "${2:-$DEFAULT_BACKUP_DIR}"
        ;;
    gcs)
        backup_gcs "${2:-}"
        ;;
    s3)
        backup_s3 "${2:-}"
        ;;
    restore)
        restore "${2:-}"
        ;;
    status)
        show_sizes
        ;;
    *)
        echo "Usage: $0 {local|gcs|s3|restore|status} [path]"
        exit 1
        ;;
esac

