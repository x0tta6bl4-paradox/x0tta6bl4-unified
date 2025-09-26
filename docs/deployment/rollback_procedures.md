# 🔄 Rollback Procedures - x0tta6bl4 Unified Platform

## Обзор

Этот guide содержит процедуры отката изменений в production среде x0tta6bl4 Unified Platform.

## 📋 Типы rollback

### 1. Application Rollback
Откат к предыдущей версии приложения при проблемах с deployment.

### 2. Database Rollback
Восстановление базы данных из backup при проблемах с миграциями.

### 3. Configuration Rollback
Откат конфигурационных изменений.

### 4. Infrastructure Rollback
Откат изменений в инфраструктуре (Kubernetes, Docker, etc.).

## 🚀 Application Rollback

### Автоматический Rollback (Blue-Green Deployment)

#### Подготовка
```bash
# Проверка текущего состояния
curl -f http://blue.x0tta6bl4.com/health
curl -f http://green.x0tta6bl4.com/health

# Определение активной среды
ACTIVE_ENV=$(curl -s http://load-balancer/status | jq -r '.active')
echo "Active environment: $ACTIVE_ENV"
```

#### Выполнение rollback
```bash
# Переключение трафика на предыдущую версию
curl -X POST http://load-balancer/switch \
  -H "Content-Type: application/json" \
  -d "{\"target\": \"$INACTIVE_ENV\"}"

# Ожидание стабилизации
sleep 300  # 5 минут

# Проверка здоровья
curl -f http://active.x0tta6bl4.com/health

# Остановка проблемной версии
docker-compose -f docker-compose.$PROBLEM_ENV.yml down
```

### Ручной Rollback (Rolling Deployment)

#### Подготовка
```bash
# Создание backup текущего состояния
./scripts/backup/create_rollback_backup.sh

# Определение версии для отката
TARGET_VERSION=$(git tag --sort=-version:refname | sed -n '2p')
echo "Rolling back to version: $TARGET_VERSION"
```

#### Выполнение
```bash
# Переход к целевой версии
git checkout $TARGET_VERSION

# Пересборка и перезапуск
docker-compose build --no-cache
docker-compose up -d --scale app=0  # Остановка всех инстансов
docker-compose up -d --scale app=3  # Запуск новых инстансов постепенно

# Проверка каждой инстанса
for i in {1..3}; do
  curl -f http://app-$i.x0tta6bl4.com/health
  echo "Instance $i healthy"
done
```

## 🗄️ Database Rollback

### Point-in-Time Recovery

#### Подготовка
```bash
# Остановка приложения
docker-compose stop app

# Создание backup текущего состояния (если возможно)
docker exec db pg_dump -U x0tta6bl4_prod x0tta6bl4_prod > pre_rollback_backup.sql
```

#### Восстановление из backup
```bash
# Найти подходящий backup
BACKUP_FILE=$(ls -t /opt/x0tta6bl4-production/backups/x0tta6bl4_prod_*.sql.gz | head -1)
echo "Using backup: $BACKUP_FILE"

# Распаковка
gunzip -c $BACKUP_FILE > rollback_backup.sql

# Остановка базы
docker-compose stop db

# Удаление текущего volume
docker volume rm x0tta6bl4-production_postgres_data

# Перезапуск базы
docker-compose up -d db
sleep 30

# Восстановление данных
docker exec -i db psql -U x0tta6bl4_prod -d x0tta6bl4_prod < rollback_backup.sql
```

#### Rollback миграций
```bash
# Определение проблемной миграции
docker exec app python -c "
from production.database.migrations import get_migration_history
history = get_migration_history()
print('Recent migrations:')
for mig in history[-5:]:
    print(f'{mig.id}: {mig.name} - {mig.applied_at}')
"

# Rollback конкретной миграции
docker exec app python -c "
from production.database.migrations import rollback_migration
rollback_migration('problem_migration_id')
"
```

## ⚙️ Configuration Rollback

### Git-based Rollback
```bash
# Проверка изменений
git log --oneline -10 .env.production

# Откат к предыдущей версии
git checkout HEAD~1 -- .env.production

# Перезапуск сервисов
docker-compose restart app
```

### Manual Configuration Backup
```bash
# Восстановление из backup
cp /opt/x0tta6bl4-production/config_backup/.env.production.backup .env.production

# Проверка синтаксиса
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env.production'); print('Config valid')"

# Перезапуск
docker-compose restart
```

## 🏗️ Infrastructure Rollback

### Kubernetes Rollback
```bash
# Проверка истории deployments
kubectl rollout history deployment/x0tta6bl4-app

# Rollback к предыдущей версии
kubectl rollout undo deployment/x0tta6bl4-app

# Проверка статуса
kubectl rollout status deployment/x0tta6bl4-app
```

### Docker Compose Rollback
```bash
# Сохранение текущего состояния
docker-compose config > docker-compose.current.yml

# Восстановление предыдущей конфигурации
cp docker-compose.previous.yml docker-compose.yml

# Перезапуск
docker-compose up -d
```

## ⏱️ Временные рамки Rollback

| Тип Rollback | Подготовка | Выполнение | Проверка | Общее время |
|--------------|------------|------------|----------|-------------|
| Application | 5 мин | 10-30 мин | 5 мин | 20-40 мин |
| Database | 10 мин | 30-60 мин | 15 мин | 55-85 мин |
| Configuration | 2 мин | 5 мин | 2 мин | 9 мин |
| Infrastructure | 5 мин | 15-45 мин | 10 мин | 30-60 мин |

## ✅ Проверка после Rollback

### Автоматические проверки
```bash
# Health checks
curl -f http://localhost/health
curl -f http://localhost/api/v1/quantum/status
curl -f http://localhost/api/v1/ai/status

# Database connectivity
docker exec db psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "SELECT 1;"

# Cache functionality
docker exec redis redis-cli ping

# Load testing
python load_test.py --url http://localhost --duration 300 --concurrency 10
```

### Ручные проверки
- [ ] API endpoints отвечают корректно
- [ ] Пользовательские функции работают
- [ ] Метрики собираются
- [ ] Логи не содержат ошибок
- [ ] Производительность в норме

## 📊 Мониторинг Rollback

### Ключевые метрики
- **Response Time**: < 500ms
- **Error Rate**: < 1%
- **CPU/Memory**: В пределах нормы
- **Database Connections**: Стабильны

### Alert Suppression
```bash
# Отключение алертов во время rollback
curl -X POST http://alertmanager/api/v1/silences \
  -H "Content-Type: application/json" \
  -d '{
    "matchers": [
      {"name": "alertname", "value": ".*", "isRegex": true}
    ],
    "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'",
    "endsAt": "'$(date -u -d "+1 hour" +%Y-%m-%dT%H:%M:%S.000Z)'",
    "comment": "Rollback in progress"
  }'
```

## 📝 Документация Rollback

### Rollback Report Template
```
# Rollback Report

Incident: [Описание проблемы]
Timestamp: [Время rollback]
Type: [Application/Database/Configuration/Infrastructure]
Previous Version: [Версия до rollback]
Target Version: [Версия после rollback]
Duration: [Время выполнения]
Impact: [Воздействие на пользователей]
Verification: [Результаты проверок]
Lessons Learned: [Уроки для будущих deployments]
```

### Post-Rollback Actions
1. **Анализ причины** первоначального сбоя
2. **Тестирование** исправления в staging
3. **Планирование** повторного deployment
4. **Обновление** документации

## 🔧 Автоматизация Rollback

### Rollback Scripts
```bash
#!/bin/bash
# automated_rollback.sh

set -e

echo "Starting automated rollback..."

# Validate input
if [ -z "$ROLLBACK_TYPE" ]; then
  echo "Error: ROLLBACK_TYPE not set"
  exit 1
fi

# Create backup
./scripts/backup/create_pre_rollback_backup.sh

# Execute rollback based on type
case $ROLLBACK_TYPE in
  application)
    ./scripts/rollback/application_rollback.sh
    ;;
  database)
    ./scripts/rollback/database_rollback.sh
    ;;
  configuration)
    ./scripts/rollback/config_rollback.sh
    ;;
  infrastructure)
    ./scripts/rollback/infra_rollback.sh
    ;;
  *)
    echo "Unknown rollback type: $ROLLBACK_TYPE"
    exit 1
    ;;
esac

# Verify rollback
./scripts/rollback/verify_rollback.sh

# Notify team
./scripts/alerting/send_rollback_notification.sh

echo "Rollback completed successfully"
```

### Rollback Testing
```bash
# Регулярное тестирование rollback процедур
./scripts/test_rollback_procedures.sh

# Проверка времени восстановления
time ./automated_rollback.sh
```

## 🚨 Экстренные ситуации

### Полный System Rollback
При катастрофическом сбое:
1. **Остановка** всех сервисов
2. **Восстановление** из полного backup
3. **Проверка** целостности данных
4. **Постепенный** запуск сервисов

### Emergency Contacts
- **Lead Engineer**: [Телефон] - [Email]
- **DevOps Lead**: [Телефон] - [Email]
- **Business Owner**: [Телефон] - [Email]

---

**Версия**: 1.0
**Обновлено**: $(date)
**Контакт для обновлений**: DevOps Team