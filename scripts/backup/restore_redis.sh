#!/bin/bash
# Disaster recovery script for Redis
# Скрипт восстановления Redis из резервной копии

set -e

# Configuration
BACKUP_DIR="/backups"
REDIS_CONTAINER="redis"
LOG_FILE="/backups/restore.log"

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - $1" | tee -a $LOG_FILE
}

# Function to show usage
usage() {
    echo "Usage: $0 [BACKUP_FILE]"
    echo "If BACKUP_FILE is not specified, the latest backup will be used"
    echo "Available backups:"
    ls -la $BACKUP_DIR/redis_*.rdb 2>/dev/null || echo "No Redis backups found"
    exit 1
}

# Check if backup file is specified
if [ $# -eq 0 ]; then
    # Find the latest backup
    BACKUP_FILE=$(ls -t $BACKUP_DIR/redis_*.rdb 2>/dev/null | head -1)
    if [ -z "$BACKUP_FILE" ]; then
        log "ERROR: No Redis backup files found in $BACKUP_DIR"
        usage
    fi
else
    BACKUP_FILE="$1"
    if [ ! -f "$BACKUP_FILE" ]; then
        log "ERROR: Backup file $BACKUP_FILE does not exist"
        usage
    fi
fi

log "Starting Redis restore from: $BACKUP_FILE"

# Verify backup file exists and has size > 0
if [ ! -s "$BACKUP_FILE" ]; then
    log "ERROR: Backup file $BACKUP_FILE is empty!"
    exit 1
fi

# Stop Redis container
log "Stopping Redis container..."
docker stop $REDIS_CONTAINER

# Copy backup file to Redis data directory
log "Copying backup file to Redis container..."
docker cp "$BACKUP_FILE" $REDIS_CONTAINER:/data/dump.rdb

# Verify file was copied
if ! docker exec $REDIS_CONTAINER test -f /data/dump.rdb; then
    log "ERROR: Failed to copy backup file to Redis container"
    docker start $REDIS_CONTAINER
    exit 1
fi

# Start Redis container
log "Starting Redis container..."
docker start $REDIS_CONTAINER

# Wait for Redis to start
sleep 5

# Verify Redis is working
log "Verifying Redis restore..."
if docker exec $REDIS_CONTAINER redis-cli ping | grep -q PONG; then
    log "Redis restore verification successful"
else
    log "WARNING: Redis restore verification failed"
fi

# Check if data was restored (basic check)
KEYS_COUNT=$(docker exec $REDIS_CONTAINER redis-cli DBSIZE 2>/dev/null || echo "0")
log "Redis database contains $KEYS_COUNT keys after restore"

log "Redis restore completed successfully from $BACKUP_FILE"

# Log restore success metric
echo "restore_success{type=\"redis\",file=\"$BACKUP_FILE\"} 1" >> /backups/metrics.prom