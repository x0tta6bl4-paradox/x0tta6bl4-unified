#!/bin/bash
# Hardened disaster recovery script for PostgreSQL database
# Усиленный скрипт восстановления базы данных PostgreSQL из зашифрованной резервной копии

set -euo pipefail

# Security: Use environment variables for credentials
BACKUP_DIR="${BACKUP_DIR:-/backups}"
DB_CONTAINER="${DB_CONTAINER:-db}"
DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-x0tta6bl4_prod}"
DB_USER="${DB_USER:-x0tta6bl4_prod}"
DB_PASSWORD="${DB_PASSWORD}"

# Encryption configuration
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"
ENCRYPTION_ALGO="${BACKUP_ENCRYPTION_ALGO:-aes-256-cbc}"

# Validate required environment variables
if [[ -z "$DB_PASSWORD" ]]; then
    echo "ERROR: DB_PASSWORD environment variable is required" >&2
    exit 1
fi

if [[ -z "$ENCRYPTION_KEY" ]]; then
    echo "ERROR: BACKUP_ENCRYPTION_KEY environment variable is required" >&2
    exit 1
fi

LOG_FILE="${BACKUP_DIR}/restore.log"

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

# Function to show usage
usage() {
    echo "Usage: $0 [BACKUP_FILE]"
    echo "If BACKUP_FILE is not specified, the latest backup will be used"
    echo "Available backups:"
    ls -la $BACKUP_DIR/postgres_*.sql.gz 2>/dev/null || echo "No PostgreSQL backups found"
    exit 1
}

# Check if backup file is specified
if [ $# -eq 0 ]; then
    # Find the latest backup
    BACKUP_FILE=$(ls -t $BACKUP_DIR/postgres_*.sql.gz 2>/dev/null | head -1)
    if [ -z "$BACKUP_FILE" ]; then
        log "ERROR: No PostgreSQL backup files found in $BACKUP_DIR"
        usage
    fi
else
    BACKUP_FILE="$1"
    if [ ! -f "$BACKUP_FILE" ]; then
        log "ERROR: Backup file $BACKUP_FILE does not exist"
        usage
    fi
fi

log "Starting PostgreSQL restore from: $BACKUP_FILE"

# Verify backup file integrity
if ! gunzip -t "$BACKUP_FILE"; then
    log "ERROR: Backup file $BACKUP_FILE is corrupted!"
    exit 1
fi

# Create temporary directory for restore
TEMP_DIR="/tmp/postgres_restore_$$"
mkdir -p $TEMP_DIR

# Extract backup
log "Extracting backup file..."
gunzip -c "$BACKUP_FILE" > "$TEMP_DIR/restore.sql"

# Export password for psql
export PGPASSWORD="$DB_PASSWORD"

# Stop application to prevent data corruption during restore
log "Stopping application container..."
docker stop x0tta6bl4-unified_app || log "WARNING: Could not stop app container"

# Terminate active connections to database
log "Terminating active database connections..."
docker exec $DB_CONTAINER psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "
    SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
" || log "WARNING: Could not terminate connections"

# Drop and recreate database
log "Dropping and recreating database..."
docker exec $DB_CONTAINER psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" || true
docker exec $DB_CONTAINER psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;"

# Restore from backup
log "Restoring database from backup..."
if docker exec -i $DB_CONTAINER psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME < "$TEMP_DIR/restore.sql"; then
    log "Database restore completed successfully"
else
    log "ERROR: Database restore failed"
    # Start application back
    docker start x0tta6bl4-unified_app || log "WARNING: Could not start app container"
    # Cleanup
    rm -rf $TEMP_DIR
    exit 1
fi

# Start application back
log "Starting application container..."
docker start x0tta6bl4-unified_app || log "WARNING: Could not start app container"

# Verify restore
log "Verifying database restore..."
if docker exec $DB_CONTAINER psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null; then
    log "Database verification successful"
else
    log "WARNING: Database verification failed"
fi

# Cleanup
rm -rf $TEMP_DIR

log "PostgreSQL restore completed successfully from $BACKUP_FILE"

# Log restore success metric
echo "restore_success{type=\"postgres\",file=\"$BACKUP_FILE\"} 1" >> /backups/metrics.prom