# 📊 Monitoring Runbook - x0tta6bl4 Unified Platform

## Ежедневные проверки

### Утренние проверки (9:00)
```bash
# Проверка статуса всех сервисов
curl http://localhost/health
curl http://localhost/api/v1/quantum/status
curl http://localhost/api/v1/ai/status
curl http://localhost/api/v1/enterprise/status

# Проверка использования ресурсов
docker stats --no-stream
df -h
free -h
```

### Мониторинг ключевых метрик
- **Response Time**: < 500ms для API
- **Error Rate**: < 1%
- **CPU Usage**: < 80%
- **Memory Usage**: < 90%
- **Disk Usage**: < 85%

## Инструменты мониторинга

### Grafana Dashboards
- **System Overview**: http://localhost:3000/d/x0tta6bl4-system
- **Application Metrics**: http://localhost:3000/d/x0tta6bl4-app
- **Database Performance**: http://localhost:3000/d/x0tta6bl4-db

### Prometheus Queries
```promql
# CPU usage
rate(cpu_usage_percent[5m])

# Memory usage
memory_usage_percent

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

## Alert Response Procedures

### 🚨 Critical Alerts

#### Application Down
1. Проверить статус контейнера: `docker ps | grep app`
2. Посмотреть логи: `docker logs x0tta6bl4-production_app_1 --tail 50`
3. Перезапустить: `docker-compose restart app`
4. Если не помогает: `docker-compose up -d --force-recreate app`

#### Database Down
1. Проверить статус: `docker ps | grep db`
2. Проверить логи: `docker logs x0tta6bl4-production_db_1`
3. Перезапустить: `docker-compose restart db`

#### High Memory Usage
1. Проверить top процессы: `docker stats`
2. Очистить cache если Redis: `docker exec redis redis-cli FLUSHALL`
3. Перезапустить сервис с высоким потреблением

### ⚠️ Warning Alerts

#### High CPU Usage
1. Определить процесс: `docker stats`
2. Проверить логи на ошибки
3. Рассмотреть оптимизацию кода

#### Low Disk Space
1. Проверить использование: `du -sh /opt/x0tta6bl4-production/*`
2. Очистить старые логи: `find logs -name "*.log" -mtime +7 -delete`
3. Очистить старые backups: `find backups -mtime +30 -delete`

## Логирование

### Просмотр логов
```bash
# Application logs
tail -f logs/app.log

# Docker logs
docker logs -f x0tta6bl4-production_app_1

# System logs
journalctl -u docker -f
```

### Поиск ошибок
```bash
# Последние ошибки
grep "ERROR" logs/app.log | tail -10

# Поиск по паттерну
grep "exception" logs/app.log | tail -5
```

## Профилактическое обслуживание

### Еженедельные задачи
- [ ] Проверка backup'ов
- [ ] Очистка старых логов
- [ ] Обновление зависимостей
- [ ] Проверка дискового пространства

### Ежемесячные задачи
- [ ] Анализ трендов производительности
- [ ] Оптимизация запросов к БД
- [ ] Обновление системы безопасности
- [ ] Проверка резервных копий

## Контакты
- **Разработчик**: [Ваш контакт]
- **Мониторинг**: http://localhost:3000
- **Документация**: docs/runbooks/