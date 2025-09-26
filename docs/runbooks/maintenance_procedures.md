# 🔧 Maintenance Procedures - x0tta6bl4 Unified Platform

## Обзор

Этот runbook содержит процедуры планового обслуживания x0tta6bl4 Unified Platform в production среде.

## 📅 График обслуживания

### Ежедневные задачи
- **Время**: 06:00 - 07:00 (минимальное воздействие)
- **Продолжительность**: 30-60 минут
- **Ответственный**: On-call инженер

### Еженедельные задачи
- **Время**: Воскресенье 02:00 - 06:00
- **Продолжительность**: 2-4 часа
- **Ответственный**: DevOps Engineer

### Ежемесячные задачи
- **Время**: Первое воскресенье месяца 01:00 - 04:00
- **Продолжительность**: 4-6 часов
- **Ответственный**: SRE Team

### Квартальные задачи
- **Время**: По согласованию, обычно выходные
- **Продолжительность**: 8-12 часов
- **Ответственный**: Full Team

## 🔍 Ежедневные проверки

### Проверка здоровья системы
```bash
# Health checks
curl -f http://localhost/health
curl -f http://localhost/api/v1/quantum/status
curl -f http://localhost/api/v1/ai/status
curl -f http://localhost/api/v1/enterprise/status

# Проверка метрик
docker stats --no-stream
df -h
free -h
```

### Мониторинг логов
```bash
# Проверка на ошибки
tail -100 /opt/x0tta6bl4-production/logs/app.log | grep -i error

# Проверка системных логов
journalctl -u docker --since "1 hour ago" | grep -i error

# Проверка алертов
curl -s http://localhost:9090/api/v1/alerts | jq '.data[] | select(.state=="firing")'
```

### Проверка backup'ов
```bash
# Проверка последних backup'ов
ls -la /opt/x0tta6bl4-production/backups/ | tail -10

# Проверка размера backup'ов
du -sh /opt/x0tta6bl4-production/backups/*

# Тест восстановления (еженедельно)
# ./scripts/backup/test_restore.sh
```

## 🛠️ Еженедельные процедуры

### Очистка логов и временных файлов
```bash
# Очистка старых логов (старше 30 дней)
find /opt/x0tta6bl4-production/logs -name "*.log" -mtime +30 -delete

# Очистка Docker logs
docker system prune -f

# Очистка старых backup'ов (старше 90 дней)
find /opt/x0tta6bl4-production/backups -name "*.gz" -mtime +90 -delete
```

### Обновление зависимостей
```bash
# Проверка обновлений безопасности
cd /opt/x0tta6bl4-production

# Обновление Python зависимостей (тестировать предварительно!)
pip list --outdated
pip install --upgrade -r requirements.txt

# Обновление системных пакетов
sudo apt update && sudo apt upgrade -y
```

### Оптимизация базы данных
```bash
# PostgreSQL maintenance
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "VACUUM ANALYZE;"

# Проверка индексов
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
"
```

### Проверка и оптимизация Redis
```bash
# Redis memory usage
docker exec redis redis-cli INFO memory

# Очистка expired keys
docker exec redis redis-cli KEYS "*" | xargs -n 1 docker exec redis redis-cli DEL

# Проверка persistence
docker exec redis redis-cli SAVE
```

## 🔄 Ежемесячные процедуры

### Полное тестирование backup'ов
```bash
# Создание тестового окружения
docker run -d --name test-restore -p 5433:5432 postgres:15
sleep 30

# Восстановление из backup
LATEST_BACKUP=$(ls -t /opt/x0tta6bl4-production/backups/*x0tta6bl4_prod*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | docker exec -i test-restore psql -U postgres -d postgres

# Проверка целостности данных
docker exec test-restore psql -U postgres -d postgres -c "SELECT COUNT(*) FROM users;"

# Очистка тестового окружения
docker stop test-restore && docker rm test-restore
```

### Анализ производительности
```bash
# Сбор метрик за месяц
curl -s "http://localhost:9090/api/v1/query_range?query=rate(http_requests_total[30d])&start=$(date -d '30 days ago' +%s)&end=$(date +%s)" > monthly_metrics.json

# Анализ slow queries
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"
```

### Обновление системного ПО
```bash
# Безопасное обновление с проверкой
sudo apt update
sudo apt list --upgradable
sudo apt upgrade -y

# Перезапуск сервисов если необходимо
docker-compose -f docker-compose.production.yml restart
```

## 🚀 Квартальные процедуры

### Major version updates
```bash
# Планирование обновления
# 1. Тестирование в staging
# 2. Создание rollback плана
# 3. Планирование maintenance window

# Выполнение обновления
cd /opt/x0tta6bl4-production
git pull origin main
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d

# Проверка после обновления
curl -f http://localhost/health
python load_test.py --duration 1800  # 30 минут тестирования
```

### Архитектурные улучшения
- **Database optimization**: Добавление индексов, партиционирование
- **Cache optimization**: Настройка Redis кластера
- **Monitoring enhancement**: Добавление новых метрик
- **Security updates**: Обновление сертификатов, firewall правил

### Capacity planning
```bash
# Анализ трендов использования
# Прогнозирование роста
# Планирование масштабирования

# Текущее использование
docker stats --no-stream
df -h /
free -h
```

## ⚠️ Процедуры с downtime

### Планирование maintenance window
1. **Уведомление** заинтересованных сторон за 2 недели
2. **Документация** процедуры и rollback плана
3. **Тестирование** в staging окружении
4. **Резервный план** на случай проблем

### Безопасное проведение
```bash
# Предварительные проверки
curl -f http://localhost/health > /dev/null && echo "System healthy" || exit 1

# Создание pre-maintenance backup
./scripts/backup/full_backup.sh pre_maintenance_$(date +%Y%m%d_%H%M%S)

# Уведомление о начале
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚧 Maintenance window started"}' \
  $SLACK_WEBHOOK_URL

# Выполнение maintenance
# ... процедуры ...

# Проверка после maintenance
curl -f http://localhost/health > /dev/null && echo "Maintenance successful" || ./rollback.sh

# Уведомление о завершении
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"✅ Maintenance completed successfully"}' \
  $SLACK_WEBHOOK_URL
```

## 📊 Мониторинг во время обслуживания

### Ключевые метрики для отслеживания
- **Response time**: < 500ms
- **Error rate**: < 1%
- **CPU usage**: < 80%
- **Memory usage**: < 90%
- **Disk usage**: < 85%

### Автоматические алерты
- Отключение не-критических алертов во время планового обслуживания
- Специальные алерты для monitoring maintenance процесса
- Автоматическое уведомление о завершении

## 📝 Документация и отчетность

### Maintenance log
```bash
# Запись в maintenance log
echo "$(date): $MAINTENANCE_TYPE completed by $USER" >> /opt/x0tta6bl4-production/logs/maintenance.log
```

### Отчеты
- **Еженедельный**: Статус системы и выполненные задачи
- **Ежемесячный**: Анализ производительности и рекомендации
- **Квартальный**: План улучшений и capacity planning

## 🔄 Rollback процедуры

### Для каждого типа maintenance
- **Database changes**: Point-in-time recovery из backup
- **Application updates**: Rollback к предыдущей версии
- **Configuration changes**: Восстановление из Git
- **Infrastructure changes**: Terraform state rollback

### Автоматизированный rollback
```bash
#!/bin/bash
# rollback.sh
echo "Starting rollback procedure..."

# Stop services
docker-compose -f docker-compose.production.yml down

# Restore from backup
./scripts/backup/restore_from_backup.sh $PRE_MAINTENANCE_BACKUP

# Start services
docker-compose -f docker-compose.production.yml up -d

# Verify
curl -f http://localhost/health && echo "Rollback successful"
```

---

**Версия**: 1.0
**Обновлено**: $(date)
**Контакт для обновлений**: DevOps Team