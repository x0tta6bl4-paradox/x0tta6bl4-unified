# 🔧 Troubleshooting Guide - x0tta6bl4 Unified Platform

## Распространенные проблемы и решения

### 1. Application не запускается
**Симптомы**: Контейнер app не стартует
**Решение**:
```bash
# Проверить логи
docker logs x0tta6bl4-production_app_1

# Проверить конфигурацию
docker exec x0tta6bl4-production_app_1 cat /app/.env

# Перезапустить с новыми логами
docker-compose up -d --force-recreate app
```

### 2. Database connection error
**Симптомы**: Ошибки подключения к PostgreSQL
**Решение**:
```bash
# Проверить статус базы
docker ps | grep db

# Проверить логи базы
docker logs x0tta6bl4-production_db_1

# Проверить подключение
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "SELECT 1;"
```

### 3. High memory usage
**Симптомы**: Приложение использует >90% памяти
**Решение**:
```bash
# Проверить использование
docker stats

# Очистить Redis cache
docker exec redis redis-cli FLUSHALL

# Перезапустить приложение
docker-compose restart app
```

### 4. Slow response times
**Симптомы**: API отвечает медленно
**Решение**:
```bash
# Проверить нагрузку
docker stats

# Проверить логи на ошибки
tail -f logs/app.log | grep ERROR

# Проверить базу данных
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "SELECT * FROM pg_stat_activity;"
```

## Incident Response

### Шаги при инциденте:
1. **Оценка**: Определить влияние и серьезность
2. **Сдерживание**: Изолировать проблему
3. **Восстановление**: Вернуть систему в рабочее состояние
4. **Анализ**: Определить причину
5. **Документация**: Записать урок

### Escalation:
- **P1 (Critical)**: Немедленное реагирование, < 15 мин
- **P2 (High)**: Реагирование в течение часа
- **P3 (Medium)**: Реагирование в течение дня
- **P4 (Low)**: Реагирование по графику

## Контакты
- **Техническая поддержка**: [Ваш контакт]
- **Документация**: docs/runbooks/