# 🔄 Migration Scripts Guide

## Overview

This guide provides comprehensive instructions for using the x0tta6bl4 Unified migration scripts. The migration system is designed to safely and efficiently migrate data and configurations from x0tta6bl4 and x0tta6bl4-next into the unified platform.

## Migration Scripts Architecture

The migration system consists of three main scripts:

- **`migrate.py`**: Main migration orchestrator
- **`validation.py`**: Post-migration validation
- **`rollback.py`**: Migration rollback system

### Script Locations
```
scripts/migration/
├── migrate.py           # Main migration script
├── validation.py        # Validation script
├── rollback.py          # Rollback script
├── data_migration_x0tta6bl4.py
├── data_migration_x0tta6bl4_next.py
├── config_migration.py
├── phase1_setup.py
├── phase2_migration.py
├── phase3_integration.py
└── rollback.py
```

## Migration Process Overview

### Migration Phases
1. **Pre-migration Checks**: Environment validation
2. **Data Migration (x0tta6bl4)**: Migrate legacy quantum data
3. **Data Migration (x0tta6bl4-next)**: Migrate enterprise data
4. **Configuration Migration**: Merge configurations
5. **Service Migration**: Migrate service configurations
6. **Validation**: Verify migration success
7. **Cleanup**: Remove temporary files

### Safety Features
- **Dry Run Mode**: Test migration without changes
- **Backup Creation**: Automatic backup before migration
- **Rollback Capability**: Complete rollback support
- **Validation Checks**: Comprehensive post-migration validation
- **Logging**: Detailed logging throughout process

## Using migrate.py

### Basic Usage

```bash
# Full migration
python scripts/migration/migrate.py

# Dry run (recommended first)
python scripts/migration/migrate.py --dry-run

# Rollback migration
python scripts/migration/migrate.py --rollback

# Validate only
python scripts/migration/migrate.py --validate-only
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dry-run` | Show migration plan without executing | `--dry-run` |
| `--rollback` | Execute rollback instead of migration | `--rollback` |
| `--validate-only` | Run validation only | `--validate-only` |

### Migration Workflow

#### 1. Preparation
```bash
# Navigate to project root
cd x0tta6bl4-unified

# Ensure source projects exist
ls -la ../x0tta6bl4
ls -la ../x0tta6bl4-next

# Check prerequisites
python3 --version  # Should be 3.12+
docker --version
pip list | grep -E "(docker|kubernetes)"
```

#### 2. Dry Run
```bash
# Always run dry run first
python scripts/migration/migrate.py --dry-run

# Check the output for any issues
# Review migration plan and warnings
```

#### 3. Execute Migration
```bash
# Run actual migration
python scripts/migration/migrate.py

# Monitor progress in logs
tail -f migration.log

# Check migration report
cat migration_report.json | jq '.'
```

#### 4. Validation
```bash
# Run validation
python scripts/migration/migrate.py --validate-only

# Check validation report
cat validation_report.json | jq '.validation_report.summary'
```

### Migration Output

#### Migration Report Structure
```json
{
  "started_at": "2025-09-24T00:19:07.701Z",
  "completed_at": "2025-09-24T00:20:07.701Z",
  "status": "completed",
  "phases": [
    "pre_migration_checks",
    "data_migration_x0tta6bl4",
    "data_migration_x0tta6bl4_next",
    "config_migration",
    "service_migration",
    "validation",
    "cleanup"
  ],
  "completed_phases": ["pre_migration_checks", "data_migration_x0tta6bl4", ...],
  "failed_phases": [],
  "warnings": [],
  "errors": []
}
```

#### Log Output
```
2025-09-24 00:19:07,701 - INFO - 🚀 Начало миграции x0tta6bl4 Unified
2025-09-24 00:19:07,702 - INFO - 🔄 Выполнение фазы: pre_migration_checks
2025-09-24 00:19:07,703 - INFO - 🔍 Выполнение предварительных проверок...
2025-09-24 00:19:07,704 - INFO - ✅ Предварительные проверки пройдены
2025-09-24 00:19:07,705 - INFO - ✅ Фаза pre_migration_checks выполнена успешно
...
2025-09-24 00:20:07,700 - INFO - ✅ Миграция завершена успешно!
```

## Using validation.py

### Validation Checks Performed

The validation script performs comprehensive checks:

1. **Project Structure**: Required directories and files
2. **Configuration Files**: YAML/JSON config validation
3. **Python Imports**: Module import verification
4. **Service Integrity**: Service structure validation
5. **Database Connectivity**: DB connection checks
6. **API Endpoints**: Endpoint availability
7. **Monitoring Setup**: Monitoring configuration
8. **Security Config**: Security settings validation
9. **Quantum Components**: Quantum service validation
10. **AI Components**: AI/ML service validation
11. **Enterprise Components**: Enterprise service validation

### Running Validation

```bash
# Validate current setup
python scripts/migration/validation.py

# Or use migrate.py wrapper
python scripts/migration/migrate.py --validate-only
```

### Validation Report

```json
{
  "validation_report": {
    "timestamp": "2025-09-24T00:20:10.000Z",
    "summary": {
      "checks_passed": 11,
      "checks_failed": 0,
      "total_checks": 11,
      "success_rate": 100.0
    },
    "components_validated": ["structure", "config", "imports", "services"],
    "warnings": [],
    "errors": []
  }
}
```

### Common Validation Issues

#### Missing Directories
```
❌ Project structure: Отсутствуют директории: config, production
```
**Solution**: Ensure all required directories exist

#### Import Errors
```
❌ Python imports: Модуль production.quantum не найден
```
**Solution**: Check Python path and module installation

#### Configuration Issues
```
❌ Config files: Отсутствует секция 'system' в unified_config.yaml
```
**Solution**: Add missing configuration sections

## Using rollback.py

### Rollback Scenarios

Rollback should be used when:
- Migration fails partially
- Validation reveals critical issues
- Need to restore previous state
- Testing migration in staging

### Rollback Process

#### 1. Check Rollback Conditions
```bash
# Verify rollback is possible
python scripts/migration/rollback.py --check-only
```

#### 2. Execute Rollback
```bash
# Perform rollback
python scripts/migration/rollback.py

# Or use migrate.py wrapper
python scripts/migration/migrate.py --rollback
```

#### 3. Verify Rollback
```bash
# Check rollback report
cat rollback_report.json | jq '.rollback_report.summary'
```

### Rollback Steps

1. **Stop Services**: Gracefully stop all running services
2. **Restore Backup**: Restore from pre-migration backup
3. **Cleanup Files**: Remove unified-specific files
4. **Restore Configs**: Restore original configurations
5. **Restart Services**: Restart services with original config

### Rollback Safety

- **Backup Verification**: Ensures backup exists before rollback
- **Service Stopping**: Graceful shutdown prevents data corruption
- **File Preservation**: Preserves logs and reports
- **Configuration Restore**: Restores exact pre-migration state

## Migration Data Flow

### Source Systems
- **x0tta6bl4**: Legacy quantum computing platform
- **x0tta6bl4-next**: Enterprise SaaS platform

### Target System
- **x0tta6bl4-unified**: Unified platform combining both

### Data Migration Types

#### Database Migration
- **PostgreSQL**: User data, quantum jobs, billing
- **MongoDB**: Quantum circuits, AI models
- **Redis**: Sessions, cache, real-time data

#### Configuration Migration
- **Environment Configs**: Merge .env files
- **Service Configs**: Combine service configurations
- **Kubernetes Manifests**: Update K8s deployments

#### File Migration
- **Code Files**: Python modules and services
- **Documentation**: Merge docs from both systems
- **Assets**: Static files and resources

## Troubleshooting Migration

### Common Issues

#### Permission Errors
```
❌ Нет прав на запись в целевой каталог
```
**Solution**:
```bash
# Fix permissions
sudo chown -R $USER:$USER x0tta6bl4-unified/
chmod -R 755 x0tta6bl4-unified/
```

#### Missing Dependencies
```
❌ Команда не найдена: docker
```
**Solution**:
```bash
# Install missing tools
sudo apt update
sudo apt install docker.io python3-pip
```

#### Source Project Not Found
```
❌ Исходный проект x0tta6bl4 не найден
```
**Solution**:
```bash
# Verify source paths
ls -la /home/x0tta6bl4
ls -la /home/x0tta6bl4-next
# Update paths in migrate.py if needed
```

#### Database Connection Issues
```
❌ Ошибка подключения к базе данных
```
**Solution**:
```bash
# Check database connectivity
psql -h localhost -U user -d database
# Verify connection strings in config
```

### Recovery Procedures

#### Partial Migration Failure
```bash
# Stop migration
pkill -f migrate.py

# Check current state
python scripts/migration/validation.py

# Rollback if needed
python scripts/migration/rollback.py
```

#### Data Corruption
```bash
# Immediate rollback
python scripts/migration/rollback.py --force

# Restore from backup manually if needed
cp -r x0tta6bl4-unified-backup-*/* x0tta6bl4-unified/
```

#### Service Startup Issues
```bash
# Check service logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d

# Verify health
curl http://localhost:8000/health
```

## Best Practices

### Pre-Migration
1. **Backup Everything**: Create full backups of all systems
2. **Test Environment**: Use staging environment first
3. **Dry Run**: Always perform dry run before actual migration
4. **Team Coordination**: Inform all stakeholders of migration window

### During Migration
1. **Monitor Progress**: Watch logs continuously
2. **Have Rollback Plan**: Know how to rollback quickly
3. **Communication**: Keep team informed of progress
4. **Performance Monitoring**: Monitor system resources

### Post-Migration
1. **Validation**: Run all validation checks
2. **Testing**: Perform comprehensive testing
3. **Monitoring**: Monitor system health for 24-48 hours
4. **Documentation**: Update all documentation

### Maintenance
1. **Regular Backups**: Continue regular backup procedures
2. **Monitoring**: Keep monitoring systems active
3. **Documentation**: Maintain migration documentation
4. **Training**: Train team on migration procedures

## Migration Checklist

### Pre-Migration Checklist
- [ ] Source systems backed up
- [ ] Target system prepared
- [ ] Dependencies installed
- [ ] Access permissions verified
- [ ] Dry run completed successfully
- [ ] Rollback procedure tested
- [ ] Team notified of migration window

### Post-Migration Checklist
- [ ] Migration completed without errors
- [ ] Validation passed
- [ ] Services started successfully
- [ ] Data integrity verified
- [ ] Performance benchmarks met
- [ ] Monitoring systems active
- [ ] Documentation updated

## Support and Resources

### Getting Help
- **Logs**: Check `migration.log` for detailed information
- **Reports**: Review JSON reports for structured data
- **Documentation**: Refer to this guide and inline comments
- **Team**: Contact migration team for assistance

### Additional Resources
- [Architecture Overview](../architecture/overview.md)
- [Quick Start Guide](quick_start_guide.md)
- [Troubleshooting Guide](troubleshooting_guide.md)
- [API Documentation](../api/overview.md)

---

*Migration Scripts Guide - x0tta6bl4 Unified Training Materials*