# 📞 On-Call Training Guide - x0tta6bl4 Unified Platform

## Обзор

Этот training guide предназначен для подготовки членов команды к on-call обязанностям по поддержке x0tta6bl4 Unified Platform в production среде.

## 🎯 Цели обучения

После завершения этого курса вы сможете:
- Эффективно реагировать на алерты и инциденты
- Использовать инструменты мониторинга и диагностики
- Применять процедуры troubleshooting и recovery
- Координировать работу с командой при инцидентах
- Документировать инциденты и post-mortem анализ

## 📚 Модуль 1: Введение в On-Call

### 1.1 Что такое On-Call

On-call - это состояние готовности к реагированию на проблемы в production системах 24/7.

#### Обязанности On-Call инженера:
- **Мониторинг**: Отслеживание здоровья системы
- **Реагирование**: Быстрое реагирование на алерты
- **Диагностика**: Определение причин проблем
- **Исправление**: Применение процедур восстановления
- **Эскалация**: Привлечение дополнительных ресурсов при необходимости
- **Документация**: Запись всех действий и решений

### 1.2 Ротация и график

#### Структура ротации:
- **Продолжительность смены**: 1 неделя
- **Количество инженеров**: 3-5 человек в ротации
- **Передача смены**: Пятница 18:00
- **Резервный on-call**: Доступен для поддержки

#### Процедура передачи смены:
1. **Обзор текущего состояния** системы
2. **Передача открытых issues** и известных проблем
3. **Документация** недавних инцидентов
4. **Тестирование** доступа к инструментам

## 🛠️ Модуль 2: Инструменты и доступ

### 2.1 Необходимые инструменты

#### Мониторинг:
- **Grafana**: http://production-server:3000
- **Prometheus**: http://production-server:9090
- **Alertmanager**: Email/SMS уведомления

#### Доступ:
- **SSH**: К production серверам
- **Docker**: Управление контейнерами
- **Git**: Доступ к коду и конфигурациям
- **Slack/Teams**: Коммуникация с командой

### 2.2 Настройка рабочего места

#### Установка необходимого ПО:
```bash
# SSH client
sudo apt install openssh-client

# Docker client
sudo apt install docker.io

# Monitoring tools
sudo apt install curl jq

# Text editors
sudo apt install vim nano
```

#### Настройка доступа:
```bash
# SSH keys
ssh-keygen -t rsa -b 4096 -C "your-email@x0tta6bl4.com"
ssh-copy-id user@production-server

# VPN доступ (если требуется)
# Настройка OpenVPN или аналогичного

# Тестирование доступа
ssh user@production-server "docker ps"
```

## 🚨 Модуль 3: Реагирование на алерты

### 3.1 Классификация алертов

#### По severity:
- **P1 - Critical**: Немедленное реагирование (< 15 мин)
- **P2 - High**: Реагирование в течение 1 часа
- **P3 - Medium**: Реагирование в течение 4 часов
- **P4 - Low**: Реагирование в рабочее время

#### По типу:
- **Application**: Проблемы с приложением
- **Database**: Проблемы с базой данных
- **Infrastructure**: Проблемы с инфраструктурой
- **Monitoring**: Проблемы с системами мониторинга

### 3.2 Процедура реагирования

#### Шаг 1: Подтверждение алерта
```
1. Проверить получение алерта в течение 5 минут
2. Подтвердить в системе мониторинга
3. Оценить severity и impact
4. Начать первичную диагностику
```

#### Шаг 2: Диагностика
```bash
# Быстрая проверка здоровья
curl -f http://localhost/health || echo "Application DOWN"

# Проверка контейнеров
docker ps --filter "name=x0tta6bl4" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Просмотр логов
docker logs --tail 50 x0tta6bl4-production_app_1 | grep -i error

# Проверка ресурсов
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}"
```

#### Шаг 3: Реагирование
```bash
# Для распространенных проблем:

# Application crash
docker-compose -f docker-compose.production.yml restart app

# Database connection issues
docker-compose -f docker-compose.production.yml restart db

# High memory usage
docker exec redis redis-cli FLUSHALL
docker-compose -f docker-compose.production.yml restart app
```

## 🔍 Модуль 4: Диагностика проблем

### 4.1 Систематический подход

#### Шаги диагностики:
1. **Сбор информации**: Логи, метрики, конфигурация
2. **Воспроизведение**: Попытка воспроизвести проблему
3. **Изоляция**: Определение компонента с проблемой
4. **Анализ**: Поиск root cause
5. **Исправление**: Применение решения

### 4.2 Инструменты диагностики

#### Логи:
```bash
# Application logs
tail -f /opt/x0tta6bl4-production/logs/app.log

# Docker logs
docker logs -f x0tta6bl4-production_app_1 --tail 100

# System logs
journalctl -u docker -f --since "1 hour ago"
```

#### Метрики:
```bash
# CPU/Memory usage
docker stats

# Database connections
docker exec x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod -c "
SELECT count(*) as active_connections FROM pg_stat_activity;
"

# Redis info
docker exec redis redis-cli INFO
```

#### Сетевые проверки:
```bash
# Connectivity tests
curl -v http://localhost:8000/health

# DNS resolution
nslookup db

# Port availability
netstat -tlnp | grep :5432
```

## 🛠️ Модуль 5: Процедуры восстановления

### 5.1 Быстрые исправления

#### Application проблемы:
```bash
# Перезапуск сервиса
docker-compose -f docker-compose.production.yml restart app

# Масштабирование
docker-compose -f docker-compose.production.yml up -d --scale app=2

# Rollback (если доступен)
git checkout previous_version
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d
```

#### Database проблемы:
```bash
# Проверка статуса
docker ps | grep db

# Перезапуск
docker-compose -f docker-compose.production.yml restart db

# Восстановление из backup (при необходимости)
./scripts/backup/restore_db.sh latest_backup.sql
```

#### Infrastructure проблемы:
```bash
# Проверка дискового пространства
df -h

# Очистка логов
find /opt/x0tta6bl4-production/logs -name "*.log" -mtime +7 -delete

# Перезапуск системы мониторинга
docker-compose -f docker-compose.monitoring.yml restart
```

### 5.2 Disaster Recovery

#### При полном сбое:
1. **Оценка ситуации**: Определить масштаб проблемы
2. **Активация DR плана**: Следовать disaster_recovery.md
3. **Восстановление**: Использовать backup'ы
4. **Тестирование**: Проверка функциональности
5. **Коммуникация**: Информирование stakeholders

## 📞 Модуль 6: Коммуникация и эскалация

### 6.1 Внутренняя коммуникация

#### С командой:
- **Slack/Teams**: Для оперативной координации
- **GitHub Issues**: Для документирования инцидентов
- **Email**: Для формальных отчетов

#### Шаблоны сообщений:
```
🚨 INCIDENT: High CPU Usage
Status: Investigating
Impact: Degraded performance
ETA: 30 minutes
On-call: [Your Name]
```

### 6.2 Эскалация

#### Когда эскалировать:
- **P1 инциденты** не решены в 15 минут
- **P2 инциденты** не решены в 1 час
- **Требуется экспертиза** другого специалиста
- **Business impact** превышает допустимый уровень

#### Процесс эскалации:
1. **Уведомить** следующего уровня поддержки
2. **Передать контекст** проблемы
3. **Остаться involved** для поддержки
4. **Документировать** передачу

### 6.3 Внешняя коммуникация

#### С пользователями:
- **Status Page**: Обновление статуса сервиса
- **Email**: Уведомления о значительных инцидентах
- **Social Media**: При длительных простоях

#### С бизнесом:
- **Regular Updates**: Каждые 30 минут для P1
- **Post-mortem**: Анализ после разрешения
- **Impact Assessment**: Оценка влияния на бизнес

## 📝 Модуль 7: Документация и отчетность

### 7.1 Документация инцидентов

#### Incident Log Template:
```
# Incident Report

Incident ID: INC-2025-0925-001
Time Detected: 2025-09-25 14:30 UTC
Time Resolved: 2025-09-25 15:15 UTC
Duration: 45 minutes

Description:
High memory usage causing application slowdown

Impact:
- Response time increased by 200%
- 15% of requests failing
- User experience degraded

Root Cause:
Memory leak in AI processing module due to large dataset processing

Actions Taken:
1. Identified problematic process
2. Restarted application with memory limits
3. Implemented temporary throttling
4. Scheduled permanent fix for next deployment

Prevention:
- Add memory monitoring alerts
- Implement circuit breaker for large requests
- Add memory profiling to CI/CD

Lessons Learned:
- Need better monitoring for memory usage patterns
- Consider horizontal scaling for AI workloads
- Implement gradual rollout for memory-intensive features
```

### 7.2 Post-mortem процесс

#### Для значительных инцидентов:
1. **Сбор данных** в течение 24 часов
2. **Анализ timeline** событий
3. **Определение root cause**
4. **Разработка corrective actions**
5. **Внедрение улучшений** в течение 1 недели
6. **Обновление документации**

## 🎓 Модуль 8: Практические упражнения

### Упражнение 1: Симуляция алерта
```
Цель: Научиться быстрому реагированию
Шаги:
1. Получить симулированный алерт
2. Выполнить первичную диагностику
3. Применить соответствующее исправление
4. Задокументировать действия
```

### Упражнение 2: Troubleshooting сценарий
```
Цель: Развитие навыков диагностики
Сценарий: Application возвращает 500 ошибки
Шаги:
1. Проверить логи приложения
2. Проверить подключение к базе данных
3. Проверить использование ресурсов
4. Определить и исправить проблему
```

### Упражнение 3: Disaster Recovery
```
Цель: Практика восстановления
Сценарий: Полная потеря сервера
Шаги:
1. Оценить ситуацию
2. Активировать DR процедуры
3. Восстановить систему из backup
4. Проверить функциональность
```

## 📚 Модуль 9: Ресурсы и ссылки

### Документация:
- **Runbooks**: `docs/runbooks/`
- **Troubleshooting Guide**: `docs/runbooks/troubleshooting_guide.md`
- **Monitoring Runbook**: `docs/runbooks/monitoring_runbook.md`
- **Disaster Recovery**: `docs/runbooks/disaster_recovery.md`

### Инструменты:
- **Grafana**: http://production-server:3000
- **Prometheus**: http://production-server:9090
- **Alertmanager**: Конфигурация алертов

### Контакты:
- **Tech Lead**: [Имя] - [Телефон]
- **DevOps Lead**: [Имя] - [Телефон]
- **Business Owner**: [Имя] - [Телефон]

## ✅ Критерии готовности

### Знания:
- [ ] Понимание архитектуры системы
- [ ] Знание процедур реагирования
- [ ] Умение использовать инструменты диагностики
- [ ] Знание процедур восстановления

### Навыки:
- [ ] Быстрое реагирование на алерты
- [ ] Эффективная диагностика проблем
- [ ] Применение процедур восстановления
- [ ] Коммуникация с командой

### Опыт:
- [ ] Успешное прохождение симуляций
- [ ] Участие в shadow on-call
- [ ] Разрешение реальных инцидентов под supervision

## 🏆 Сертификат

После успешного завершения обучения и прохождения практических упражнений выдается сертификат "x0tta6bl4 On-Call Engineer".

### Требования для сертификации:
1. **Теоретический экзамен**: 80% правильных ответов
2. **Практические упражнения**: Успешное выполнение всех сценариев
3. **Shadow on-call**: 2 недели под руководством опытного инженера
4. **Оценка ментора**: Положительная рекомендация

---

**Версия курса**: 1.0
**Продолжительность**: 2 дня
**Обновлено**: $(date)
**Контакт для вопросов**: Training Team