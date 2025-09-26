#!/bin/bash
# Full backup script for x0tta6bl4 Unified Platform in Docker environment
# Полное резервное копирование всех компонентов системы в Docker окружении

set -e

BACKUP_DIR="/backups"
LOG_FILE="/backups/backup.log"
DATE=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create directories
mkdir -p $BACKUP_DIR

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

log "Starting full backup: $DATE"

# Initialize metrics file
echo "# Backup metrics for Prometheus" > /backups/metrics.prom
echo "# Generated at $(date)" >> /backups/metrics.prom

# Backup database
log "Backing up PostgreSQL database..."
if $SCRIPT_DIR/backup_database.sh; then
    log "Database backup completed successfully"
else
    log "ERROR: Database backup failed"
    echo "backup_failure{type=\"postgres\"} 1" >> /backups/metrics.prom
    exit 1
fi

# Backup Redis
log "Backing up Redis data..."
if $SCRIPT_DIR/backup_redis.sh; then
    log "Redis backup completed successfully"
else
    log "ERROR: Redis backup failed"
    echo "backup_failure{type=\"redis\"} 1" >> /backups/metrics.prom
    exit 1
fi

# Backup quantum components
log "Backing up quantum components..."
if $SCRIPT_DIR/quantum_backup.sh; then
    log "Quantum backup completed successfully"
else
    log "ERROR: Quantum backup failed"
    echo "backup_failure{type=\"quantum\"} 1" >> /backups/metrics.prom
    exit 1
fi

# Backup research data
log "Backing up research data..."
if $SCRIPT_DIR/research_backup.sh; then
    log "Research data backup completed successfully"
else
    log "ERROR: Research data backup failed"
    echo "backup_failure{type=\"research\"} 1" >> /backups/metrics.prom
    exit 1
fi

# Backup application data (if exists)
log "Backing up application data..."
APP_DATA_BACKUP="${BACKUP_DIR}/app_data_${DATE}.tar.gz"
if docker exec x0tta6bl4-unified_app tar -czf /tmp/app_data.tar.gz -C /app/data . 2>/dev/null && \
   docker cp x0tta6bl4-unified_app:/tmp/app_data.tar.gz $APP_DATA_BACKUP 2>/dev/null; then
    log "Application data backup completed: $APP_DATA_BACKUP"
    APP_SIZE=$(stat -f%z "$APP_DATA_BACKUP" 2>/dev/null || stat -c%s "$APP_DATA_BACKUP")
    echo "backup_success{type=\"app_data\",size=\"$APP_SIZE\"} 1" >> /backups/metrics.prom
else
    log "WARNING: Application data backup failed or no data to backup"
fi

# Backup configuration files
log "Backing up configuration files..."
CONFIG_BACKUP="${BACKUP_DIR}/config_${DATE}.tar.gz"
if tar -czf $CONFIG_BACKUP -C /host .env.production docker-compose.production.yml 2>/dev/null; then
    log "Configuration backup completed: $CONFIG_BACKUP"
    CONFIG_SIZE=$(stat -f%z "$CONFIG_BACKUP" 2>/dev/null || stat -c%s "$CONFIG_BACKUP")
    echo "backup_success{type=\"config\",size=\"$CONFIG_SIZE\"} 1" >> /backups/metrics.prom
else
    log "WARNING: Configuration backup failed"
fi

# Calculate total backup sizes
TOTAL_SIZE=$(du -sh $BACKUP_DIR | awk '{print $1}')
log "Total backup size: $TOTAL_SIZE"

# Send success notification (if alerting is configured)
if [ -f "/scripts/alerting/email_alert.py" ]; then
    python3 /scripts/alerting/email_alert.py << EOF
Full backup completed successfully at $DATE
Total backup size: $TOTAL_SIZE
Components backed up: PostgreSQL, Redis, Quantum, Research Data, Application Data, Configurations
EOF
fi

log "Full backup completed successfully: $DATE"

# Secure cleanup of old backups (older than 30 days, keep max 20 full backups)
log "Cleaning up old backups with secure deletion..."

# Clean up encrypted database backups
OLD_DB_BACKUPS=$(find $BACKUP_DIR -name "postgres_*.sql.enc" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_DB_BACKUPS" ]]; then
    echo "$OLD_DB_BACKUPS" | while read -r file; do
        log "Securely removing old database backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

# Clean up database checksums
OLD_DB_CHECKSUMS=$(find $BACKUP_DIR -name "postgres_*.sql.sha256" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_DB_CHECKSUMS" ]]; then
    echo "$OLD_DB_CHECKSUMS" | while read -r file; do
        log "Removing old database checksum: $file"
        rm -f "$file"
    done
fi

# Clean up encrypted Redis backups
OLD_REDIS_BACKUPS=$(find $BACKUP_DIR -name "redis_*.rdb.enc" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_REDIS_BACKUPS" ]]; then
    echo "$OLD_REDIS_BACKUPS" | while read -r file; do
        log "Securely removing old Redis backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

# Clean up Redis checksums
OLD_REDIS_CHECKSUMS=$(find $BACKUP_DIR -name "redis_*.rdb.sha256" -mtime +30 -type f 2>/dev/null)
if [[ -n "$OLD_REDIS_CHECKSUMS" ]]; then
    echo "$OLD_REDIS_CHECKSUMS" | while read -r file; do
        log "Removing old Redis checksum: $file"
        rm -f "$file"
    done
fi

# Clean up other backup types (keeping existing logic for now)
find $BACKUP_DIR -name "app_data_*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find $BACKUP_DIR -name "config_*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find $BACKUP_DIR/quantum -name "quantum_*.tar.gz" -mtime +14 -delete 2>/dev/null || true
find $BACKUP_DIR/research -name "*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find $BACKUP_DIR/research -name "*.sql.gz" -mtime +30 -delete 2>/dev/null || true

# Keep only last 20 full backup sets (by date) - encrypted versions
EXCESS_DB_BACKUPS=$(find $BACKUP_DIR -name "postgres_*.sql.enc" -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | tail -n +21 2>/dev/null)
if [[ -n "$EXCESS_DB_BACKUPS" ]]; then
    echo "$EXCESS_DB_BACKUPS" | while read -r file; do
        log "Securely removing excess database backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

EXCESS_DB_CHECKSUMS=$(find $BACKUP_DIR -name "postgres_*.sql.sha256" -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | tail -n +21 2>/dev/null)
if [[ -n "$EXCESS_DB_CHECKSUMS" ]]; then
    echo "$EXCESS_DB_CHECKSUMS" | while read -r file; do
        log "Removing excess database checksum: $file"
        rm -f "$file"
    done
fi

EXCESS_REDIS_BACKUPS=$(find $BACKUP_DIR -name "redis_*.rdb.enc" -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | tail -n +21 2>/dev/null)
if [[ -n "$EXCESS_REDIS_BACKUPS" ]]; then
    echo "$EXCESS_REDIS_BACKUPS" | while read -r file; do
        log "Securely removing excess Redis backup: $file"
        shred -u -z "$file" 2>/dev/null || rm -f "$file"
    done
fi

EXCESS_REDIS_CHECKSUMS=$(find $BACKUP_DIR -name "redis_*.rdb.sha256" -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | tail -n +21 2>/dev/null)
if [[ -n "$EXCESS_REDIS_CHECKSUMS" ]]; then
    echo "$EXCESS_REDIS_CHECKSUMS" | while read -r file; do
        log "Removing excess Redis checksum: $file"
        rm -f "$file"
    done
fi

# Keep other backup types with existing limits
find $BACKUP_DIR -name "app_data_*.tar.gz" -type f | sort | head -n -20 | xargs -r rm -f 2>/dev/null || true
find $BACKUP_DIR -name "config_*.tar.gz" -type f | sort | head -n -20 | xargs -r rm -f 2>/dev/null || true
find $BACKUP_DIR/quantum -name "quantum_*.tar.gz" -type f | sort | head -n -10 | xargs -r rm -f 2>/dev/null || true
find $BACKUP_DIR/research -name "*.tar.gz" -type f | sort | head -n -15 | xargs -r rm -f 2>/dev/null || true
find $BACKUP_DIR/research -name "*.sql.gz" -type f | sort | head -n -15 | xargs -r rm -f 2>/dev/null || true

log "Backup cleanup completed"

# Update backup status metric
echo "backup_last_success $(date +%s)" >> /backups/metrics.prom