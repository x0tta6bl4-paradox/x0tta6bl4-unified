#!/bin/bash
# Research data backup script for x0tta6bl4 Unified Platform
# Резервное копирование исследовательских данных: эксперименты, публикации, чекпоинты моделей

set -e

BACKUP_DIR="/backups/research"
LOG_FILE="/backups/research_backup.log"
DATE=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create directories
mkdir -p $BACKUP_DIR

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

log "Starting research data backup: $DATE"

# Initialize metrics file
echo "# Research backup metrics for Prometheus" > /backups/research_metrics.prom
echo "# Generated at $(date)" >> /backups/research_metrics.prom

# Backup quantum experiment results
log "Backing up quantum experiment results..."
EXPERIMENT_BACKUP="${BACKUP_DIR}/experiments_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_quantum tar -czf /tmp/experiment_results.tar.gz \
    -C /app/data experiment_results/ \
    benchmark_data/ \
    quantum_measurements/ \
    performance_logs/ 2>/dev/null && \
   docker cp x0tta6bl4-unified_quantum:/tmp/experiment_results.tar.gz $EXPERIMENT_BACKUP 2>/dev/null; then
    log "Experiment results backup completed: $EXPERIMENT_BACKUP"
    EXP_SIZE=$(stat -f%z "$EXPERIMENT_BACKUP" 2>/dev/null || stat -c%s "$EXPERIMENT_BACKUP")
    echo "backup_success{type=\"experiments\",size=\"$EXP_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "WARNING: Experiment results backup failed or no data to backup"
fi

# Backup research publications and papers
log "Backing up research publications..."
PUBLICATIONS_BACKUP="${BACKUP_DIR}/publications_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_research tar -czf /tmp/publications.tar.gz \
    -C /app/publications papers/ \
    manuscripts/ \
    research_notes/ \
    citations.bib 2>/dev/null && \
   docker cp x0tta6bl4-unified_research:/tmp/publications.tar.gz $PUBLICATIONS_BACKUP 2>/dev/null; then
    log "Publications backup completed: $PUBLICATIONS_BACKUP"
    PUB_SIZE=$(stat -f%z "$PUBLICATIONS_BACKUP" 2>/dev/null || stat -c%s "$PUBLICATIONS_BACKUP")
    echo "backup_success{type=\"publications\",size=\"$PUB_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "WARNING: Publications backup failed or no publications to backup"
fi

# Backup trained model checkpoints
log "Backing up trained model checkpoints..."
CHECKPOINTS_BACKUP="${BACKUP_DIR}/checkpoints_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_ai tar -czf /tmp/checkpoints.tar.gz \
    -C /app/models checkpoints/ \
    trained_models/ \
    model_weights/ \
    training_history.json 2>/dev/null && \
   docker cp x0tta6bl4-unified_ai:/tmp/checkpoints.tar.gz $CHECKPOINTS_BACKUP 2>/dev/null; then
    log "Model checkpoints backup completed: $CHECKPOINTS_BACKUP"
    CHK_SIZE=$(stat -f%z "$CHECKPOINTS_BACKUP" 2>/dev/null || stat -c%s "$CHECKPOINTS_BACKUP")
    echo "backup_success{type=\"checkpoints\",size=\"$CHK_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "WARNING: Model checkpoints backup failed or no checkpoints to backup"
fi

# Backup research database (experiment metadata, results database)
log "Backing up research database..."
DB_BACKUP="${BACKUP_DIR}/research_db_${DATE}.sql.gz"
if docker exec x0tta6bl4-unified_research pg_dump -U research_user -h research_db research_db | gzip > $DB_BACKUP 2>/dev/null; then
    log "Research database backup completed: $DB_BACKUP"
    DB_SIZE=$(stat -f%z "$DB_BACKUP" 2>/dev/null || stat -c%s "$DB_BACKUP")
    echo "backup_success{type=\"research_db\",size=\"$DB_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "ERROR: Research database backup failed"
    echo "backup_failure{type=\"research_db\"} 1" >> /backups/research_metrics.prom
    exit 1
fi

# Backup quantum supremacy validation reports
log "Backing up quantum supremacy validation reports..."
VALIDATION_BACKUP="${BACKUP_DIR}/validation_reports_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_quantum tar -czf /tmp/validation_reports.tar.gz \
    -C /app/validation supremacy_validation_reports/ \
    quantum_advantage_proofs/ \
    classical_comparison_data/ 2>/dev/null && \
   docker cp x0tta6bl4-unified_quantum:/tmp/validation_reports.tar.gz $VALIDATION_BACKUP 2>/dev/null; then
    log "Validation reports backup completed: $VALIDATION_BACKUP"
    VAL_SIZE=$(stat -f%z "$VALIDATION_BACKUP" 2>/dev/null || stat -c%s "$VALIDATION_BACKUP")
    echo "backup_success{type=\"validation_reports\",size=\"$VAL_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "WARNING: Validation reports backup failed or no reports to backup"
fi

# Backup edge AI research data
log "Backing up edge AI research data..."
EDGE_AI_BACKUP="${BACKUP_DIR}/edge_ai_research_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_edge tar -czf /tmp/edge_ai_research.tar.gz \
    -C /app/research edge_experiments/ \
    latency_measurements/ \
    energy_profiles/ \
    optimization_results/ 2>/dev/null && \
   docker cp x0tta6bl4-unified_edge:/tmp/edge_ai_research.tar.gz $EDGE_AI_BACKUP 2>/dev/null; then
    log "Edge AI research backup completed: $EDGE_AI_BACKUP"
    EDGE_SIZE=$(stat -f%z "$EDGE_AI_BACKUP" 2>/dev/null || stat -c%s "$EDGE_AI_BACKUP")
    echo "backup_success{type=\"edge_ai_research\",size=\"$EDGE_SIZE\"} 1" >> /backups/research_metrics.prom
else
    log "WARNING: Edge AI research backup failed or no data to backup"
fi

# Calculate total research backup sizes
TOTAL_SIZE=$(du -sh $BACKUP_DIR | awk '{print $1}')
log "Total research backup size: $TOTAL_SIZE"

# Verify backup integrity
log "Verifying research backup integrity..."
for backup_file in $EXPERIMENT_BACKUP $PUBLICATIONS_BACKUP $CHECKPOINTS_BACKUP $DB_BACKUP $VALIDATION_BACKUP $EDGE_AI_BACKUP; do
    if [ -f "$backup_file" ]; then
        case "$backup_file" in
            *.tar.gz)
                if tar -tzf "$backup_file" > /dev/null 2>&1; then
                    log "Integrity check passed for: $(basename $backup_file)"
                else
                    log "ERROR: Integrity check failed for: $(basename $backup_file)"
                    echo "backup_integrity_failure{file=\"$(basename $backup_file)\"} 1" >> /backups/research_metrics.prom
                fi
                ;;
            *.sql.gz)
                if gzip -t "$backup_file" > /dev/null 2>&1; then
                    log "Integrity check passed for: $(basename $backup_file)"
                else
                    log "ERROR: Integrity check failed for: $(basename $backup_file)"
                    echo "backup_integrity_failure{file=\"$(basename $backup_file)\"} 1" >> /backups/research_metrics.prom
                fi
                ;;
        esac
    fi
done

# Send success notification
if [ -f "/scripts/alerting/email_alert.py" ]; then
    python3 /scripts/alerting/email_alert.py << EOF
Research data backup completed successfully at $DATE
Total research backup size: $TOTAL_SIZE
Backed up: experiments, publications, checkpoints, database, validation reports, edge AI research
EOF
fi

log "Research data backup completed successfully: $DATE"

# Clean up old research backups (older than 30 days, keep max 15)
log "Cleaning up old research backups..."
find $BACKUP_DIR -name "experiments_*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "publications_*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "checkpoints_*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "research_db_*.sql.gz" -mtime +30 -delete
find $BACKUP_DIR -name "validation_reports_*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "edge_ai_research_*.tar.gz" -mtime +30 -delete

# Keep only last 15 research backup sets
find $BACKUP_DIR -name "experiments_*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f
find $BACKUP_DIR -name "publications_*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f
find $BACKUP_DIR -name "checkpoints_*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f
find $BACKUP_DIR -name "research_db_*.sql.gz" -type f | sort | head -n -15 | xargs -r rm -f
find $BACKUP_DIR -name "validation_reports_*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f
find $BACKUP_DIR -name "edge_ai_research_*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f

log "Research backup cleanup completed"

# Update backup status metric
echo "research_backup_last_success $(date +%s)" >> /backups/research_metrics.prom

log "Research data backup process finished"