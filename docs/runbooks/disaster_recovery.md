# 🚨 Disaster Recovery Runbook - x0tta6bl4 Unified Platform

## Обзор

Этот runbook содержит процедуры восстановления системы после катастрофических сбоев. Процедуры адаптированы для сценария с 1 разработчиком.

## 📋 Предварительные требования

- Доступ к backup хранилищу: `/opt/x0tta6bl4-production/backups/`
- Доступ к production серверу
- Копия production конфигурации (`.env.production`, `docker-compose.production.yml`)
- Копия исходного кода из репозитория

## 🚨 Сценарии катастроф и процедуры восстановления

### Сценарий 1: Полная потеря сервера (Hardware Failure)

#### Шаги восстановления:
1. **Подготовка нового сервера**
   ```bash
   # На новом сервере установить базовые компоненты
   sudo apt update
   sudo apt install -y docker.io docker-compose git curl
   ```

2. **Восстановление кода и конфигурации**
   ```bash
   # Клонировать репозиторий
   git clone https://github.com/x0tta6bl4/x0tta6bl4-unified.git /opt/x0tta6bl4-production
   cd /opt/x0tta6bl4-production

   # Восстановить конфигурацию из backup или репозитория
   cp .env.example .env.production
   # Отредактировать .env.production с production значениями
   ```

3. **Восстановление данных**
   ```bash
   # Создать директории для данных
   mkdir -p data logs backups

   # Восстановить последний backup базы данных
   LATEST_DB_BACKUP=$(ls -t backups/*x0tta6bl4_prod*.sql.gz | head -1)
   gunzip $LATEST_DB_BACKUP
   UNCOMPRESSED_BACKUP=${LATEST_DB_BACKUP%.gz}

   # Восстановить Redis данные
   LATEST_REDIS_BACKUP=$(ls -t backups/redis_*.rdb | head -1)
   cp $LATEST_REDIS_BACKUP redis_dump.rdb
   ```

4. **Запуск системы**
   ```bash
   # Запустить базу данных
   docker-compose -f docker-compose.production.yml up -d db

   # Дождаться готовности базы
   sleep 30

   # Восстановить данные базы
   docker exec -i x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod < $UNCOMPRESSED_BACKUP

   # Запустить Redis и восстановить данные
   docker-compose -f docker-compose.production.yml up -d redis
   docker cp redis_dump.rdb x0tta6bl4-production_redis_1:/data/dump.rdb
   docker exec x0tta6bl4-production_redis_1 redis-cli SHUTDOWN
   sleep 5
   docker-compose -f docker-compose.production.yml up -d redis

   # Запустить приложение
   docker-compose -f docker-compose.production.yml up -d app
   ```

5. **Проверка восстановления**
   ```bash
   # Проверить здоровье системы
   curl http://localhost/health
   curl http://localhost/api/v1/quantum/status
   curl http://localhost/api/v1/ai/status
   ```

**Ожидаемое время восстановления**: 2-4 часа

---

### Сценарий 2: Потеря базы данных (Database Corruption)

#### Шаги восстановления:
1. **Остановить приложение**
   ```bash
   docker-compose -f docker-compose.production.yml stop app
   ```

2. **Создать backup текущего состояния** (если возможно)
   ```bash
   # Попытаться сделать backup поврежденной базы для анализа
   docker exec x0tta6bl4-production_db_1 pg_dump -U x0tta6bl4_prod x0tta6bl4_prod > corrupted_backup.sql
   ```

3. **Восстановить из backup**
   ```bash
   # Найти последний успешный backup
   LATEST_BACKUP=$(ls -t backups/*x0tta6bl4_prod*.sql.gz | head -1)

   # Распаковать backup
   gunzip $LATEST_BACKUP
   BACKUP_FILE=${LATEST_BACKUP%.gz}

   # Остановить базу данных
   docker-compose -f docker-compose.production.yml stop db

   # Удалить старый volume (ОСТОРОЖНО!)
   docker volume rm x0tta6bl4-production_postgres_data

   # Перезапустить базу
   docker-compose -f docker-compose.production.yml up -d db
   sleep 30

   # Восстановить данные
   docker exec -i x0tta6bl4-production_db_1 psql -U x0tta6bl4_prod -d x0tta6bl4_prod < $BACKUP_FILE
   ```

4. **Запустить приложение**
   ```bash
   docker-compose -f docker-compose.production.yml up -d app
   ```

**Ожидаемое время восстановления**: 30-60 минут

---

### Сценарий 3: Потеря Redis данных (Cache Failure)

#### Шаги восстановления:
1. **Проверить возможность восстановления**
   ```bash
   # Проверить, есть ли Redis persistence включена
   docker exec x0tta6bl4-production_redis_1 redis-cli CONFIG GET save
   ```

2. **Восстановить из backup**
   ```bash
   # Остановить Redis
   docker-compose -f docker-compose.production.yml stop redis

   # Найти последний backup
   LATEST_REDIS=$(ls -t backups/redis_*.rdb | head -1)

   # Скопировать backup в volume
   docker cp $LATEST_REDIS x0tta6bl4-production_redis_1:/data/dump.rdb

   # Перезапустить Redis
   docker-compose -f docker-compose.production.yml up -d redis
   ```

3. **Если backup недоступен - холодный старт**
   ```bash
   # Redis начнется пустым - приложение должно перестроить cache постепенно
   docker-compose -f docker-compose.production.yml up -d redis
   ```

**Ожидаемое время восстановления**: 5-15 минут

---

### Сценарий 4: Повреждение конфигурации

#### Шаги восстановления:
1. **Восстановить из Git**
   ```bash
   cd /opt/x0tta6bl4-production
   git status  # Проверить изменения
   git checkout -- .env.production
   git checkout -- docker-compose.production.yml
   ```

2. **Восстановить из backup**
   ```bash
   # Если конфигурация backup'илась
   LATEST_CONFIG=$(ls -t backups/config_*.tar.gz | head -1)
   tar -xzf $LATEST_CONFIG -C /
   ```

3. **Перезапустить сервисы**
   ```bash
   docker-compose -f docker-compose.production.yml restart
   ```

**Ожидаемое время восстановления**: 5-10 минут

---

### Сценарий 5: Сбой приложения (Application Crash)

#### Шаги восстановления:
1. **Проверить статус**
   ```bash
   docker-compose -f docker-compose.production.yml ps
   docker logs x0tta6bl4-production_app_1 --tail 50
   ```

2. **Перезапустить приложение**
   ```bash
   docker-compose -f docker-compose.production.yml restart app
   ```

3. **Если не помогает - пересоздать контейнер**
   ```bash
   docker-compose -f docker-compose.production.yml up -d --force-recreate app
   ```

4. **Проверить логи после перезапуска**
   ```bash
   docker logs x0tta6bl4-production_app_1 --tail 100
   ```

**Ожидаемое время восстановления**: 2-5 минут

---

## 🔍 Диагностика проблем

### Проверка компонентов
```bash
# Проверка Docker
docker ps
docker stats

# Проверка сети
docker network ls
docker inspect x0tta6bl4-production_default

# Проверка дискового пространства
df -h
du -sh /opt/x0tta6bl4-production/*

# Проверка логов
tail -f /opt/x0tta6bl4-production/logs/app.log
docker logs x0tta6bl4-production_db_1 --tail 50
```

### Проверка backup'ов
```bash
# Проверить наличие backup'ов
ls -la /opt/x0tta6bl4-production/backups/

# Проверить целостность backup'а базы
LATEST_BACKUP=$(ls -t backups/*x0tta6bl4_prod*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | head -20  # Проверить заголовок
```

## 📞 Контакты для экстренных ситуаций

- **Основной разработчик**: [Ваш контакт]
- **Мониторинг**: http://your-server:3000
- **Системные логи**: `/opt/x0tta6bl4-production/logs/`

## ⏱️ RTO/RPO цели

- **Recovery Time Objective (RTO)**: 4 часа для полного восстановления
- **Recovery Point Objective (RPO)**: 1 час (частота backup'ов)

## 📝 Пост-восстановление

После успешного восстановления:

1. **Провести полное тестирование**
   ```bash
   python load_test.py --url http://localhost --duration 300
   ```

2. **Создать новый backup**
   ```bash
   cd /opt/x0tta6bl4-production/scripts/backup
   ./full_backup.sh
   ```

3. **Обновить документацию инцидента**
   - Записать причину сбоя
   - Задокументировать шаги восстановления
   - Определить меры предотвращения

4. **Отправить отчет**
   ```bash
   # Отправить email с отчетом о восстановлении
   python ../alerting/email_alert.py
   ```

---

**Последнее обновление**: $(date)
**Версия документа**: 1.0