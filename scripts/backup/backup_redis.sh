#!/bin/bash
# Hardened backup script for Redis in Docker environment
# Усиленное резервное копирование Redis данных в Docker окружении с enterprise-grade security

set -euo pipefail

# Security: Use environment variables for configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
REDIS_CONTAINER="${REDIS_CONTAINER:-redis}"

# Encryption configuration
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"
ENCRYPTION_ALGO="${BACKUP_ENCRYPTION_ALGO:-aes-256-cbc}"

# Validate required environment variables
if [[ -z "$ENCRYPTION_KEY" ]]; then
    echo "ERROR: BACKUP_ENCRYPTION_KEY environment variable is required" >&2
    exit 1
fi

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/redis_dump_${DATE}.rdb"
ENCRYPTED_FILE="${BACKUP_FILE}.enc"
CHECKSUM_FILE="${BACKUP_FILE}.sha256"

# Audit logging function
audit_log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date +%Y-%m-%dT%H:%M:%S%z)
    local user="${USER:-unknown}"
    local pid="$$"

    echo "{\"timestamp\":\"$timestamp\",\"level\":\"$level\",\"user\":\"$user\",\"pid\":\"$pid\",\"script\":\"backup_redis.sh\",\"operation\":\"backup\",\"message\":\"$message\"}" >> "${BACKUP_DIR}/audit.log"
}

# Encryption function
encrypt_backup() {
    local input_file="$1"
    local output_file="$2"

    if ! openssl enc -${ENCRYPTION_ALGO} -salt -in "$input_file" -out "$output_file" -k "$ENCRYPTION_KEY" 2>/dev/null; then
        audit_log "ERROR" "Failed to encrypt backup file: $input_file"
        return 1
    fi

    chmod 600 "$output_file"
    audit_log "INFO" "Backup file encrypted successfully: $output_file"
}

# Checksum function
generate_checksum() {
    local file="$1"
    local checksum_file="$2"

    if ! sha256sum "$file" > "$checksum_file" 2>/dev/null; then
        audit_log "ERROR" "Failed to generate checksum for: $file"
        return 1
    fi

    chmod 600 "$checksum_file"
    audit_log "INFO" "Checksum generated: $checksum_file"
}

# Secure command execution
secure_exec() {
    local cmd="$1"
    local description="$2"

    audit_log "INFO" "Executing: $description"

    if ! eval "$cmd" 2>&1; then
        audit_log "ERROR" "Command failed: $description"
        return 1
    fi
}

# Create backup directory with secure permissions
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

audit_log "INFO" "Starting Redis backup: $DATE"

# Verify Redis container is running and accessible
if ! docker ps --filter "name=^${REDIS_CONTAINER}$" --filter "status=running" | grep -q "$REDIS_CONTAINER"; then
    audit_log "ERROR" "Redis container is not running: $REDIS_CONTAINER"
    exit 1
fi

# Trigger Redis SAVE command securely
secure_exec "docker exec $REDIS_CONTAINER redis-cli SAVE" "Redis SAVE command"

# Wait for SAVE to complete with timeout
TIMEOUT=30
SECONDS=0
while (( SECONDS < TIMEOUT )); do
    if docker exec "$REDIS_CONTAINER" redis-cli INFO persistence | grep -q "rdb_bgsave_in_progress:0"; then
        break
    fi
    sleep 1
done

if (( SECONDS >= TIMEOUT )); then
    audit_log "ERROR" "Redis SAVE operation timed out"
    exit 1
fi

# Copy the dump file securely
secure_exec "docker cp $REDIS_CONTAINER:/data/dump.rdb $BACKUP_FILE" "Copy Redis dump file"

# Verify backup file exists and has size > 0
if [[ ! -s "$BACKUP_FILE" ]]; then
    audit_log "ERROR" "Redis backup file is empty or missing: $BACKUP_FILE"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Generate checksum before encryption
generate_checksum "$BACKUP_FILE" "$CHECKSUM_FILE"

# Encrypt the backup
encrypt_backup "$BACKUP_FILE" "$ENCRYPTED_FILE"

# Secure cleanup - remove unencrypted file
shred -u -z "$BACKUP_FILE" 2>/dev/null || rm -f "$BACKUP_FILE"

audit_log "INFO" "Redis backup completed and encrypted: $ENCRYPTED_FILE"

# Secure cleanup of old backups (keep last 30 days, max 50 files)
audit_log "INFO" "Starting cleanup of old Redis backups"

# Find and remove old encrypted backups
OLD_BACKUPS=$(find "$BACKUP_DIR" -name "redis_*.rdb.enc" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_BACKUPS" ]]; then
    echo "$OLD_BACKUPS" | while read -r file; do
        audit_log "INFO" "Removing old Redis backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

# Find and remove old checksums
OLD_CHECKSUMS=$(find "$BACKUP_DIR" -name "redis_*.rdb.sha256" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_CHECKSUMS" ]]; then
    echo "$OLD_CHECKSUMS" | while read -r file; do
        audit_log "INFO" "Removing old Redis checksum: $file"
        rm -f "$file"
    done
fi

# Keep only last 50 encrypted backups
EXCESS_BACKUPS=$(find "$BACKUP_DIR" -name "redis_*.rdb.enc" -type f -print0 2>/dev/null | xargs -0 ls -t | tail -n +51 2>/dev/null)
if [[ -n "$EXCESS_BACKUPS" ]]; then
    echo "$EXCESS_BACKUPS" | while read -r file; do
        audit_log "INFO" "Removing excess Redis backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

# Keep only last 50 checksums
EXCESS_CHECKSUMS=$(find "$BACKUP_DIR" -name "redis_*.rdb.sha256" -type f -print0 2>/dev/null | xargs -0 ls -t | tail -n +51 2>/dev/null)
if [[ -n "$EXCESS_CHECKSUMS" ]]; then
    echo "$EXCESS_CHECKSUMS" | while read -r file; do
        audit_log "INFO" "Removing excess Redis checksum: $file"
        rm -f "$file"
    done
fi

audit_log "INFO" "Redis backup cleanup completed"

# Log backup success with enhanced metrics
BACKUP_SIZE=$(stat -f%z "$ENCRYPTED_FILE" 2>/dev/null || stat -c%s "$ENCRYPTED_FILE")
CHECKSUM_SIZE=$(stat -f%z "$CHECKSUM_FILE" 2>/dev/null || stat -c%s "$CHECKSUM_FILE")

cat >> "${BACKUP_DIR}/metrics.prom" << EOF
backup_success{type="redis",encrypted="true",checksum="sha256"} 1
backup_size_bytes{type="redis"} $BACKUP_SIZE
backup_checksum_size_bytes{type="redis"} $CHECKSUM_SIZE
backup_timestamp{type="redis"} $(date +%s)
EOF

audit_log "INFO" "Redis backup completed successfully with encryption and integrity validation"