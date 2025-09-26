#!/bin/bash
# Test script for backup and restore functionality
# Тестовый скрипт для проверки функциональности backup и restore

set -e

BACKUP_DIR="/backups"
LOG_FILE="/backups/test.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test database configuration (use environment variables)
TEST_DB_NAME="${TEST_DB_NAME:-x0tta6bl4_test_backup}"
TEST_DB_USER="${TEST_DB_USER:-x0tta6bl4_prod}"
TEST_DB_PASSWORD="${TEST_DB_PASSWORD}"

# Encryption configuration for tests
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY}"
ENCRYPTION_ALGO="${BACKUP_ENCRYPTION_ALGO:-aes-256-cbc}"

# Validate required environment variables
if [[ -z "$TEST_DB_PASSWORD" ]]; then
    echo "ERROR: TEST_DB_PASSWORD environment variable is required" >&2
    exit 1
fi

if [[ -z "$ENCRYPTION_KEY" ]]; then
    echo "ERROR: BACKUP_ENCRYPTION_KEY environment variable is required" >&2
    exit 1
fi

# Logging function
log() {
    echo "$(date +%Y-%m-%d\ %H:%M:%S) - TEST: $1" | tee -a $LOG_FILE
}

# Cleanup function
cleanup() {
    log "Cleaning up test resources..."
    # Drop test database if exists
    export PGPASSWORD="$TEST_DB_PASSWORD"
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d postgres -c "DROP DATABASE IF EXISTS $TEST_DB_NAME;" 2>/dev/null || true
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d postgres -c "DROP DATABASE IF EXISTS ${TEST_DB_NAME}_restored;" 2>/dev/null || true

    # Securely remove test backups (shred encrypted files)
    for file in $BACKUP_DIR/test_*.sql.enc $BACKUP_DIR/test_*.rdb.enc; do
        if [[ -f "$file" ]]; then
            shred -u -z "$file" 2>/dev/null || rm -f "$file"
            log "Securely removed encrypted test backup: $file"
        fi
    done

    # Remove checksum files
    rm -f $BACKUP_DIR/test_*.sql.sha256 $BACKUP_DIR/test_*.rdb.sha256 2>/dev/null || true

    # Remove legacy uncompressed files if any
    rm -f $BACKUP_DIR/test_*.sql $BACKUP_DIR/test_*.rdb 2>/dev/null || true

    log "Cleanup completed"
}

# Setup test data
setup_test_data() {
    log "Setting up test data..."

    # Create test database
    export PGPASSWORD="$TEST_DB_PASSWORD"
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d postgres -c "CREATE DATABASE $TEST_DB_NAME;"

    # Create test table and insert data
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d $TEST_DB_NAME << 'EOF'
        CREATE TABLE test_backup (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        INSERT INTO test_backup (name) VALUES
            ('test_record_1'),
            ('test_record_2'),
            ('test_record_3');
EOF

    # Add test data to Redis
    docker exec redis redis-cli SET test_key "test_value"
    docker exec redis redis-cli SET test_counter "42"

    log "Test data setup completed"
}

# Test PostgreSQL backup
test_postgres_backup() {
    log "Testing PostgreSQL backup..."

    # Create backup
    export PGPASSWORD="$TEST_DB_PASSWORD"
    BACKUP_FILE="$BACKUP_DIR/test_postgres_backup.sql"

    docker exec db pg_dump -h db -p 5432 -U $TEST_DB_USER -d $TEST_DB_NAME > $BACKUP_FILE

    if [ ! -s "$BACKUP_FILE" ]; then
        log "ERROR: PostgreSQL backup file is empty"
        return 1
    fi

    # Compress
    gzip $BACKUP_FILE

    if [ ! -f "${BACKUP_FILE}.gz" ]; then
        log "ERROR: PostgreSQL backup compression failed"
        return 1
    fi

    log "PostgreSQL backup test passed"
    return 0
}

# Test PostgreSQL restore
test_postgres_restore() {
    log "Testing PostgreSQL restore..."

    BACKUP_FILE="$BACKUP_DIR/test_postgres_backup.sql.gz"

    if [ ! -f "$BACKUP_FILE" ]; then
        log "ERROR: Backup file not found for restore test"
        return 1
    fi

    # Drop and recreate test database
    export PGPASSWORD="$TEST_DB_PASSWORD"
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d postgres -c "DROP DATABASE IF EXISTS ${TEST_DB_NAME}_restored;"
    docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d postgres -c "CREATE DATABASE ${TEST_DB_NAME}_restored;"

    # Restore
    gunzip -c "$BACKUP_FILE" | docker exec -i db psql -h db -p 5432 -U $TEST_DB_USER -d ${TEST_DB_NAME}_restored

    # Verify restore
    COUNT=$(docker exec db psql -h db -p 5432 -U $TEST_DB_USER -d ${TEST_DB_NAME}_restored -t -c "SELECT COUNT(*) FROM test_backup;")

    if [ "$COUNT" -ne 3 ]; then
        log "ERROR: PostgreSQL restore verification failed. Expected 3 records, got $COUNT"
        return 1
    fi

    log "PostgreSQL restore test passed"
    return 0
}

# Test Redis backup
test_redis_backup() {
    log "Testing Redis backup..."

    # Trigger save
    docker exec redis redis-cli SAVE

    # Copy dump file
    BACKUP_FILE="$BACKUP_DIR/test_redis_backup.rdb"
    docker cp redis:/data/dump.rdb $BACKUP_FILE

    if [ ! -s "$BACKUP_FILE" ]; then
        log "ERROR: Redis backup file is empty"
        return 1
    fi

    log "Redis backup test passed"
    return 0
}

# Test Redis restore
test_redis_restore() {
    log "Testing Redis restore..."

    BACKUP_FILE="$BACKUP_DIR/test_redis_backup.rdb"

    if [ ! -f "$BACKUP_FILE" ]; then
        log "ERROR: Redis backup file not found for restore test"
        return 1
    fi

    # Stop Redis
    docker stop redis
    sleep 2

    # Copy backup file
    docker cp "$BACKUP_FILE" redis:/data/dump.rdb

    # Start Redis
    docker start redis
    sleep 5

    # Verify restore
    VALUE=$(docker exec redis redis-cli GET test_key)
    COUNTER=$(docker exec redis redis-cli GET test_counter)

    if [ "$VALUE" != "test_value" ] || [ "$COUNTER" != "42" ]; then
        log "ERROR: Redis restore verification failed. Got key=$VALUE, counter=$COUNTER"
        return 1
    fi

    log "Redis restore test passed"
    return 0
}

# Main test function
main() {
    log "Starting backup/restore functionality tests"

    # Cleanup any previous test artifacts
    cleanup

    # Setup test data
    setup_test_data

    # Test PostgreSQL backup and restore
    if test_postgres_backup && test_postgres_restore; then
        log "PostgreSQL backup/restore tests PASSED"
        PG_TESTS=0
    else
        log "PostgreSQL backup/restore tests FAILED"
        PG_TESTS=1
    fi

    # Test Redis backup and restore
    if test_redis_backup && test_redis_restore; then
        log "Redis backup/restore tests PASSED"
        REDIS_TESTS=0
    else
        log "Redis backup/restore tests FAILED"
        REDIS_TESTS=1
    fi

    # Final cleanup
    cleanup

    # Report results
    if [ $PG_TESTS -eq 0 ] && [ $REDIS_TESTS -eq 0 ]; then
        log "ALL TESTS PASSED"
        echo "test_backup_restore_success 1" >> /backups/test_metrics.prom
        exit 0
    else
        log "SOME TESTS FAILED"
        echo "test_backup_restore_success 0" >> /backups/test_metrics.prom
        exit 1
    fi
}

# Run main function
main "$@"