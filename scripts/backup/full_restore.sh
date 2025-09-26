#!/bin/bash
# Full disaster recovery script for x0tta6bl4 Unified Platform
# Полное восстановление всех компонентов системы из резервных копий

set -euo pipefail

BACKUP_DIR="/backups"
LOG_FILE="/backups/restore.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Encryption configuration
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"
ENCRYPTION_ALGO="${BACKUP_ENCRYPTION_ALGO:-aes-256-cbc}"

# Validate required environment variables
if [[ -z "$ENCRYPTION_KEY" ]]; then
    echo "ERROR: BACKUP_ENCRYPTION_KEY environment variable is required" >&2
    exit 1
fi

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

# Function to show usage
usage() {
    echo "Usage: $0 [BACKUP_DATE]"
    echo "BACKUP_DATE format: YYYYMMDD_HHMMSS (e.g., 20231225_143000)"
    echo "If not specified, the latest complete backup set will be used"
    echo ""
    echo "Available backup sets:"
    echo "PostgreSQL backups:"
    ls -la $BACKUP_DIR/postgres_*.sql.gz 2>/dev/null | head -5 || echo "  No PostgreSQL backups found"
    echo "Redis backups:"
    ls -la $BACKUP_DIR/redis_*.rdb 2>/dev/null | head -5 || echo "  No Redis backups found"
    exit 1
}

# Parse arguments
if [ $# -eq 0 ]; then
    # Find the latest complete backup set
    LATEST_POSTGRES=$(ls -t $BACKUP_DIR/postgres_*.sql.gz 2>/dev/null | head -1)
    LATEST_REDIS=$(ls -t $BACKUP_DIR/redis_*.rdb 2>/dev/null | head -1)

    if [ -z "$LATEST_POSTGRES" ] && [ -z "$LATEST_REDIS" ]; then
        log "ERROR: No backup files found"
        usage
    fi

    # Extract date from latest backup
    if [ -n "$LATEST_POSTGRES" ]; then
        BACKUP_DATE=$(basename "$LATEST_POSTGRES" | sed 's/postgres_x0tta6bl4_prod_\([0-9_]*\)\.sql\.gz/\1/')
    elif [ -n "$LATEST_REDIS" ]; then
        BACKUP_DATE=$(basename "$LATEST_REDIS" | sed 's/redis_dump_\([0-9_]*\)\.rdb/\1/')
    fi
else
    BACKUP_DATE="$1"
fi

log "Starting full system restore for backup date: $BACKUP_DATE"

# Initialize metrics file
echo "# Restore metrics for Prometheus" > /backups/restore_metrics.prom
echo "# Generated at $(date)" >> /backups/restore_metrics.prom

# Find backup files for the specified date (prefer encrypted versions)
POSTGRES_BACKUP=$(ls $BACKUP_DIR/postgres_*${BACKUP_DATE}*.sql.enc 2>/dev/null | head -1)
if [[ -z "$POSTGRES_BACKUP" ]]; then
    POSTGRES_BACKUP=$(ls $BACKUP_DIR/postgres_*${BACKUP_DATE}*.sql.gz 2>/dev/null | head -1)
fi

REDIS_BACKUP=$(ls $BACKUP_DIR/redis_*${BACKUP_DATE}*.rdb.enc 2>/dev/null | head -1)
if [[ -z "$REDIS_BACKUP" ]]; then
    REDIS_BACKUP=$(ls $BACKUP_DIR/redis_*${BACKUP_DATE}*.rdb 2>/dev/null | head -1)
fi

APP_DATA_BACKUP=$(ls $BACKUP_DIR/app_data_*${BACKUP_DATE}*.tar.gz 2>/dev/null | head -1)
CONFIG_BACKUP=$(ls $BACKUP_DIR/config_*${BACKUP_DATE}*.tar.gz 2>/dev/null | head -1)

log "Backup files to restore:"
[ -n "$POSTGRES_BACKUP" ] && log "  PostgreSQL: $POSTGRES_BACKUP" || log "  PostgreSQL: Not found"
[ -n "$REDIS_BACKUP" ] && log "  Redis: $REDIS_BACKUP" || log "  Redis: Not found"
[ -n "$APP_DATA_BACKUP" ] && log "  App Data: $APP_DATA_BACKUP" || log "  App Data: Not found"
[ -n "$CONFIG_BACKUP" ] && log "  Config: $CONFIG_BACKUP" || log "  Config: Not found"

# Stop all services
log "Stopping all services..."
docker-compose -f /host/docker-compose.production.yml down || log "WARNING: Could not stop services gracefully"

# Restore PostgreSQL
if [ -n "$POSTGRES_BACKUP" ]; then
    log "Restoring PostgreSQL database..."
    if $SCRIPT_DIR/restore_database.sh "$POSTGRES_BACKUP"; then
        log "PostgreSQL restore completed successfully"
    else
        log "ERROR: PostgreSQL restore failed"
        echo "restore_failure{type=\"postgres\"} 1" >> /backups/restore_metrics.prom
        exit 1
    fi
else
    log "WARNING: No PostgreSQL backup found, skipping database restore"
fi

# Restore Redis
if [ -n "$REDIS_BACKUP" ]; then
    log "Restoring Redis data..."
    if $SCRIPT_DIR/restore_redis.sh "$REDIS_BACKUP"; then
        log "Redis restore completed successfully"
    else
        log "ERROR: Redis restore failed"
        echo "restore_failure{type=\"redis\"} 1" >> /backups/restore_metrics.prom
        exit 1
    fi
else
    log "WARNING: No Redis backup found, skipping Redis restore"
fi

# Restore application data
if [ -n "$APP_DATA_BACKUP" ]; then
    log "Restoring application data..."
    if docker cp "$APP_DATA_BACKUP" x0tta6bl4-unified_app:/tmp/app_data.tar.gz && \
       docker exec x0tta6bl4-unified_app tar -xzf /tmp/app_data.tar.gz -C /app/data; then
        log "Application data restore completed successfully"
        echo "restore_success{type=\"app_data\"} 1" >> /backups/restore_metrics.prom
    else
        log "WARNING: Application data restore failed"
        echo "restore_failure{type=\"app_data\"} 1" >> /backups/restore_metrics.prom
    fi
fi

# Restore configuration files
if [ -n "$CONFIG_BACKUP" ]; then
    log "Restoring configuration files..."
    if tar -xzf "$CONFIG_BACKUP" -C /host; then
        log "Configuration restore completed successfully"
        echo "restore_success{type=\"config\"} 1" >> /backups/restore_metrics.prom
    else
        log "WARNING: Configuration restore failed"
        echo "restore_failure{type=\"config\"} 1" >> /backups/restore_metrics.prom
    fi
fi

# Start all services
log "Starting all services..."
if docker-compose -f /host/docker-compose.production.yml up -d; then
    log "Services started successfully"
else
    log "ERROR: Failed to start services"
    exit 1
fi

# Wait for services to be healthy
log "Waiting for services to be healthy..."
sleep 30

# Verify services
log "Verifying service health..."
if docker ps | grep -q x0tta6bl4-unified_app && \
   docker ps | grep -q db && \
   docker ps | grep -q redis; then
    log "All services are running"
else
    log "WARNING: Some services may not be running properly"
fi

# Final verification
log "Performing final system verification..."
# Add health checks here if needed

log "Full system restore completed successfully for backup date: $BACKUP_DATE"

# Update restore status metric
echo "restore_last_success $(date +%s)" >> /backups/restore_metrics.prom

# Send notification
if [ -f "/scripts/alerting/email_alert.py" ]; then
    python3 /scripts/alerting/email_alert.py << EOF
Full system restore completed successfully at $(date)
Backup date: $BACKUP_DATE
PostgreSQL: $([ -n "$POSTGRES_BACKUP" ] && echo "Restored" || echo "Skipped")
Redis: $([ -n "$REDIS_BACKUP" ] && echo "Restored" || echo "Skipped")
Application data: $([ -n "$APP_DATA_BACKUP" ] && echo "Restored" || echo "Skipped")
Configuration: $([ -n "$CONFIG_BACKUP" ] && echo "Restored" || echo "Skipped")
EOF
fi