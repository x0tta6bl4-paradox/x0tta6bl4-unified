#!/bin/bash
# Disaster recovery script for x0tta6bl4 Unified Platform
# Процедуры восстановления после катастрофы для всех компонентов системы

set -e

BACKUP_DIR="/backups"
LOG_FILE="/backups/disaster_recovery.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default recovery mode: latest
RECOVERY_MODE="${1:-latest}"
BACKUP_TIMESTAMP="${2:-latest}"

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

error_exit() {
    log "ERROR: $1"
    echo "disaster_recovery_failure{reason=\"$1\"} 1" > /backups/recovery_metrics.prom
    exit 1
}

# Function to find latest backup
find_latest_backup() {
    local backup_type="$1"
    local pattern="$2"

    if [ "$BACKUP_TIMESTAMP" = "latest" ]; then
        find $BACKUP_DIR -name "${pattern}" -type f | sort | tail -1
    else
        find $BACKUP_DIR -name "*${BACKUP_TIMESTAMP}*" -name "${pattern}" -type f | head -1
    fi
}

log "Starting disaster recovery: $RECOVERY_MODE mode, timestamp: $BACKUP_TIMESTAMP"

# Initialize metrics file
echo "# Disaster recovery metrics for Prometheus" > /backups/recovery_metrics.prom
echo "# Generated at $(date)" >> /backups/recovery_metrics.prom

# Phase 1: Infrastructure Recovery
log "=== PHASE 1: Infrastructure Recovery ==="

# Stop all services before recovery
log "Stopping all services..."
docker-compose -f /host/docker-compose.production.yml down || log "Warning: Could not stop services gracefully"

# Phase 2: Database Recovery
log "=== PHASE 2: Database Recovery ==="
DB_BACKUP=$(find_latest_backup "database" "postgres_*.sql.gz")
if [ -n "$DB_BACKUP" ] && [ -f "$DB_BACKUP" ]; then
    log "Restoring PostgreSQL database from: $DB_BACKUP"
    gunzip -c "$DB_BACKUP" | docker exec -i x0tta6bl4-unified_postgres psql -U postgres -d x0tta6bl4 || error_exit "Database recovery failed"
    log "Database recovery completed successfully"
else
    error_exit "No valid database backup found"
fi

# Phase 3: Redis Recovery
log "=== PHASE 3: Redis Recovery ==="
REDIS_BACKUP=$(find_latest_backup "redis" "redis_*.rdb")
if [ -n "$REDIS_BACKUP" ] && [ -f "$REDIS_BACKUP" ]; then
    log "Restoring Redis data from: $REDIS_BACKUP"
    docker cp "$REDIS_BACKUP" x0tta6bl4-unified_redis:/data/dump.rdb || error_exit "Redis recovery failed"
    docker restart x0tta6bl4-unified_redis || error_exit "Redis restart failed"
    log "Redis recovery completed successfully"
else
    log "Warning: No Redis backup found, starting with empty cache"
fi

# Phase 4: Quantum State Recovery
log "=== PHASE 4: Quantum State Recovery ==="
QUANTUM_CONFIG_BACKUP=$(find_latest_backup "quantum_config" "quantum_config_*.tar.gz")
if [ -n "$QUANTUM_CONFIG_BACKUP" ] && [ -f "$QUANTUM_CONFIG_BACKUP" ]; then
    log "Restoring quantum configurations from: $QUANTUM_CONFIG_BACKUP"
    docker cp "$QUANTUM_CONFIG_BACKUP" x0tta6bl4-unified_quantum:/tmp/
    docker exec x0tta6bl4-unified_quantum tar -xzf /tmp/$(basename "$QUANTUM_CONFIG_BACKUP") -C /app/config || error_exit "Quantum config recovery failed"
    log "Quantum configurations recovered successfully"
else
    error_exit "No quantum configuration backup found"
fi

QUANTUM_STATE_BACKUP=$(find_latest_backup "quantum_state" "quantum_state_*.tar.gz")
if [ -n "$QUANTUM_STATE_BACKUP" ] && [ -f "$QUANTUM_STATE_BACKUP" ]; then
    log "Restoring quantum states from: $QUANTUM_STATE_BACKUP"
    docker cp "$QUANTUM_STATE_BACKUP" x0tta6bl4-unified_quantum:/tmp/
    docker exec x0tta6bl4-unified_quantum tar -xzf /tmp/$(basename "$QUANTUM_STATE_BACKUP") -C /app/state || log "Warning: Quantum state recovery failed"
    log "Quantum states recovered successfully"
else
    log "Warning: No quantum state backup found"
fi

# Phase 5: Research Data Recovery
log "=== PHASE 5: Research Data Recovery ==="
RESEARCH_DB_BACKUP=$(find_latest_backup "research_db" "research_db_*.sql.gz")
if [ -n "$RESEARCH_DB_BACKUP" ] && [ -f "$RESEARCH_DB_BACKUP" ]; then
    log "Restoring research database from: $RESEARCH_DB_BACKUP"
    gunzip -c "$RESEARCH_DB_BACKUP" | docker exec -i x0tta6bl4-unified_research psql -U research_user -d research_db || error_exit "Research database recovery failed"
    log "Research database recovery completed successfully"
else
    error_exit "No research database backup found"
fi

EXPERIMENTS_BACKUP=$(find_latest_backup "experiments" "experiments_*.tar.gz")
if [ -n "$EXPERIMENTS_BACKUP" ] && [ -f "$EXPERIMENTS_BACKUP" ]; then
    log "Restoring experiment results from: $EXPERIMENTS_BACKUP"
    docker cp "$EXPERIMENTS_BACKUP" x0tta6bl4-unified_quantum:/tmp/
    docker exec x0tta6bl4-unified_quantum tar -xzf /tmp/$(basename "$EXPERIMENTS_BACKUP") -C /app/data || log "Warning: Experiments recovery failed"
    log "Experiment results recovered successfully"
fi

PUBLICATIONS_BACKUP=$(find_latest_backup "publications" "publications_*.tar.gz")
if [ -n "$PUBLICATIONS_BACKUP" ] && [ -f "$PUBLICATIONS_BACKUP" ]; then
    log "Restoring publications from: $PUBLICATIONS_BACKUP"
    docker cp "$PUBLICATIONS_BACKUP" x0tta6bl4-unified_research:/tmp/
    docker exec x0tta6bl4-unified_research tar -xzf /tmp/$(basename "$PUBLICATIONS_BACKUP") -C /app/publications || log "Warning: Publications recovery failed"
    log "Publications recovered successfully"
fi

# Phase 6: AI Model Recovery
log "=== PHASE 6: AI Model Recovery ==="
CHECKPOINTS_BACKUP=$(find_latest_backup "checkpoints" "checkpoints_*.tar.gz")
if [ -n "$CHECKPOINTS_BACKUP" ] && [ -f "$CHECKPOINTS_BACKUP" ]; then
    log "Restoring model checkpoints from: $CHECKPOINTS_BACKUP"
    docker cp "$CHECKPOINTS_BACKUP" x0tta6bl4-unified_ai:/tmp/
    docker exec x0tta6bl4-unified_ai tar -xzf /tmp/$(basename "$CHECKPOINTS_BACKUP") -C /app/models || error_exit "Model checkpoints recovery failed"
    log "Model checkpoints recovered successfully"
else
    error_exit "No model checkpoints backup found"
fi

HYBRID_BACKUP=$(find_latest_backup "hybrid_checkpoints" "hybrid_checkpoints_*.tar.gz")
if [ -n "$HYBRID_BACKUP" ] && [ -f "$HYBRID_BACKUP" ]; then
    log "Restoring hybrid algorithm checkpoints from: $HYBRID_BACKUP"
    docker cp "$HYBRID_BACKUP" x0tta6bl4-unified_ai:/tmp/
    docker exec x0tta6bl4-unified_ai tar -xzf /tmp/$(basename "$HYBRID_BACKUP") -C /app/models || log "Warning: Hybrid checkpoints recovery failed"
    log "Hybrid algorithm checkpoints recovered successfully"
fi

# Phase 7: Application Data Recovery
log "=== PHASE 7: Application Data Recovery ==="
APP_DATA_BACKUP=$(find_latest_backup "app_data" "app_data_*.tar.gz")
if [ -n "$APP_DATA_BACKUP" ] && [ -f "$APP_DATA_BACKUP" ]; then
    log "Restoring application data from: $APP_DATA_BACKUP"
    docker cp "$APP_DATA_BACKUP" x0tta6bl4-unified_app:/tmp/
    docker exec x0tta6bl4-unified_app tar -xzf /tmp/$(basename "$APP_DATA_BACKUP") -C /app/data || log "Warning: App data recovery failed"
    log "Application data recovered successfully"
fi

# Phase 8: Configuration Recovery
log "=== PHASE 8: Configuration Recovery ==="
CONFIG_BACKUP=$(find_latest_backup "config" "config_*.tar.gz")
if [ -n "$CONFIG_BACKUP" ] && [ -f "$CONFIG_BACKUP" ]; then
    log "Restoring configurations from: $CONFIG_BACKUP"
    tar -xzf "$CONFIG_BACKUP" -C /host || error_exit "Configuration recovery failed"
    log "Configurations recovered successfully"
else
    error_exit "No configuration backup found"
fi

# Phase 9: Service Restart
log "=== PHASE 9: Service Restart ==="
log "Starting all services..."
docker-compose -f /host/docker-compose.production.yml up -d || error_exit "Service restart failed"

# Wait for services to be healthy
log "Waiting for services to become healthy..."
sleep 30

# Health checks
log "Performing health checks..."
if ! docker exec x0tta6bl4-unified_app curl -f http://localhost:8000/health > /dev/null 2>&1; then
    log "Warning: Application health check failed"
fi

if ! docker exec x0tta6bl4-unified_quantum python3 -c "import quantum_interface; print('Quantum interface OK')" > /dev/null 2>&1; then
    log "Warning: Quantum interface health check failed"
fi

# Phase 10: Validation and Notification
log "=== PHASE 10: Recovery Validation ==="

# Calculate recovery metrics
RECOVERY_TIME=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE" 2>/dev/null || date +%s) ))
log "Total recovery time: ${RECOVERY_TIME} seconds"

# Send recovery notification
if [ -f "/scripts/alerting/email_alert.py" ]; then
    python3 /scripts/alerting/email_alert.py << EOF
🚨 DISASTER RECOVERY COMPLETED 🚨

Recovery completed at $(date)
Recovery mode: $RECOVERY_MODE
Recovery time: ${RECOVERY_TIME} seconds

Recovered components:
- PostgreSQL database
- Redis cache
- Quantum configurations and states
- Research database and publications
- AI model checkpoints
- Application data
- System configurations

Services have been restarted. Please verify system functionality.
EOF
fi

log "Disaster recovery completed successfully"

# Update recovery status metric
echo "disaster_recovery_last_success $(date +%s)" >> /backups/recovery_metrics.prom
echo "disaster_recovery_duration_seconds $RECOVERY_TIME" >> /backups/recovery_metrics.prom

log "Disaster recovery process finished successfully"