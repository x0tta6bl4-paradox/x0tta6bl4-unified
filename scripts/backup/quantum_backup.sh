#!/bin/bash
# Quantum state backup script for x0tta6bl4 Unified Platform
# Резервное копирование квантового состояния и критических конфигураций

set -e

BACKUP_DIR="/backups/quantum"
LOG_FILE="/backups/quantum_backup.log"
DATE=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create directories
mkdir -p $BACKUP_DIR

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

log "Starting quantum backup: $DATE"

# Initialize metrics file
echo "# Quantum backup metrics for Prometheus" > /backups/quantum_metrics.prom
echo "# Generated at $(date)" >> /backups/quantum_metrics.prom

# Backup quantum state configurations
log "Backing up quantum state configurations..."
QUANTUM_CONFIG_BACKUP="${BACKUP_DIR}/quantum_config_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_quantum tar -czf /tmp/quantum_config.tar.gz \
    -C /app/config quantum_provider_settings.json \
    quantum_hardware_config.yaml \
    entanglement_protocols.yaml \
    coherence_calibration.json 2>/dev/null && \
   docker cp x0tta6bl4-unified_quantum:/tmp/quantum_config.tar.gz $QUANTUM_CONFIG_BACKUP 2>/dev/null; then
    log "Quantum config backup completed: $QUANTUM_CONFIG_BACKUP"
    CONFIG_SIZE=$(stat -f%z "$QUANTUM_CONFIG_BACKUP" 2>/dev/null || stat -c%s "$QUANTUM_CONFIG_BACKUP")
    echo "backup_success{type=\"quantum_config\",size=\"$CONFIG_SIZE\"} 1" >> /backups/quantum_metrics.prom
else
    log "ERROR: Quantum config backup failed"
    echo "backup_failure{type=\"quantum_config\"} 1" >> /backups/quantum_metrics.prom
    exit 1
fi

# Backup quantum circuit states (if running)
log "Backing up quantum circuit states..."
QUANTUM_STATE_BACKUP="${BACKUP_DIR}/quantum_state_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_quantum tar -czf /tmp/quantum_state.tar.gz \
    -C /app/state active_circuits.json \
    quantum_registers_state.bin \
    entanglement_maps.json \
    coherence_history.log 2>/dev/null && \
   docker cp x0tta6bl4-unified_quantum:/tmp/quantum_state.tar.gz $QUANTUM_STATE_BACKUP 2>/dev/null; then
    log "Quantum state backup completed: $QUANTUM_STATE_BACKUP"
    STATE_SIZE=$(stat -f%z "$QUANTUM_STATE_BACKUP" 2>/dev/null || stat -c%s "$QUANTUM_STATE_BACKUP")
    echo "backup_success{type=\"quantum_state\",size=\"$STATE_SIZE\"} 1" >> /backups/quantum_metrics.prom
else
    log "WARNING: Quantum state backup failed or no active states to backup"
fi

# Backup quantum supremacy experiment data
log "Backing up quantum supremacy experiment data..."
SUPREMACY_BACKUP="${BACKUP_DIR}/quantum_supremacy_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_quantum tar -czf /tmp/supremacy_data.tar.gz \
    -C /app/experiments supremacy_circuits/ \
    benchmark_results/ \
    quantum_advantage_metrics.json 2>/dev/null && \
   docker cp x0tta6bl4-unified_quantum:/tmp/supremacy_data.tar.gz $SUPREMACY_BACKUP 2>/dev/null; then
    log "Quantum supremacy data backup completed: $SUPREMACY_BACKUP"
    SUPREMACY_SIZE=$(stat -f%z "$SUPREMACY_BACKUP" 2>/dev/null || stat -c%s "$SUPREMACY_BACKUP")
    echo "backup_success{type=\"quantum_supremacy\",size=\"$SUPREMACY_SIZE\"} 1" >> /backups/quantum_metrics.prom
else
    log "WARNING: Quantum supremacy data backup failed or no data to backup"
fi

# Backup hybrid algorithm checkpoints
log "Backing up hybrid algorithm checkpoints..."
HYBRID_BACKUP="${BACKUP_DIR}/hybrid_checkpoints_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_ai tar -czf /tmp/hybrid_checkpoints.tar.gz \
    -C /app/models hybrid_models/ \
    quantum_classical_interfaces/ \
    optimization_states/ 2>/dev/null && \
   docker cp x0tta6bl4-unified_ai:/tmp/hybrid_checkpoints.tar.gz $HYBRID_BACKUP 2>/dev/null; then
    log "Hybrid checkpoints backup completed: $HYBRID_BACKUP"
    HYBRID_SIZE=$(stat -f%z "$HYBRID_BACKUP" 2>/dev/null || stat -c%s "$HYBRID_BACKUP")
    echo "backup_success{type=\"hybrid_checkpoints\",size=\"$HYBRID_SIZE\"} 1" >> /backups/quantum_metrics.prom
else
    log "WARNING: Hybrid checkpoints backup failed or no checkpoints to backup"
fi

# Calculate total quantum backup sizes
TOTAL_SIZE=$(du -sh $BACKUP_DIR | awk '{print $1}')
log "Total quantum backup size: $TOTAL_SIZE"

# Verify backup integrity (basic check)
log "Verifying backup integrity..."
for backup_file in $QUANTUM_CONFIG_BACKUP $QUANTUM_STATE_BACKUP $SUPREMACY_BACKUP $HYBRID_BACKUP; do
    if [ -f "$backup_file" ]; then
        if tar -tzf "$backup_file" > /dev/null 2>&1; then
            log "Integrity check passed for: $(basename $backup_file)"
        else
            log "ERROR: Integrity check failed for: $(basename $backup_file)"
            echo "backup_integrity_failure{file=\"$(basename $backup_file)\"} 1" >> /backups/quantum_metrics.prom
        fi
    fi
done

# Send success notification
if [ -f "/scripts/alerting/email_alert.py" ]; then
    python3 /scripts/alerting/email_alert.py << EOF
Quantum backup completed successfully at $DATE
Total quantum backup size: $TOTAL_SIZE
Backed up: configs, states, supremacy data, hybrid checkpoints
EOF
fi

log "Quantum backup completed successfully: $DATE"

# Clean up old quantum backups (older than 14 days, keep max 10)
log "Cleaning up old quantum backups..."
find $BACKUP_DIR -name "quantum_*.tar.gz" -mtime +14 -delete
find $BACKUP_DIR -name "hybrid_*.tar.gz" -mtime +14 -delete

# Keep only last 10 quantum backup sets
find $BACKUP_DIR -name "quantum_*.tar.gz" -type f | sort | head -n -10 | xargs -r rm -f
find $BACKUP_DIR -name "hybrid_*.tar.gz" -type f | sort | head -n -10 | xargs -r rm -f

log "Quantum backup cleanup completed"

# Update backup status metric
echo "quantum_backup_last_success $(date +%s)" >> /backups/quantum_metrics.prom

log "Quantum backup process finished"