# 🚀 Production Deployment Guide - x0tta6bl4 Unified Platform

## Обзор

Этот guide содержит пошаговые инструкции для развертывания x0tta6bl4 Unified Platform в production окружении.

## Предварительные требования

### Системные требования
- Ubuntu 20.04+ или аналогичная Linux система
- Docker 24.0+
- Docker Compose 2.0+
- Python 3.12+
- Минимум 4GB RAM, 8GB+ рекомендуется
- Минимум 20GB свободного места

### Необходимые инструменты
```bash
# Установка системных зависимостей
sudo apt update
sudo apt install -y curl wget git python3 python3-pip docker.io docker-compose postgresql-client redis-tools
```

## Шаг 1: Подготовка Production Окружения

### 1.1 Создание Production Директории
```bash
# Создание директории для production
mkdir -p /opt/x0tta6bl4-production
cd /opt/x0tta6bl4-production

# Клонирование репозитория
git clone https://github.com/x0tta6bl4/x0tta6bl4-unified.git .
git checkout main  # или production branch
```

### 1.2 Настройка Environment Variables
```bash
# Создание .env.production файла
cp .env.example .env.production

# Редактирование переменных окружения
nano .env.production
```

Пример `.env.production`:
```env
# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://x0tta6bl4_prod:secure_password@db:5432/x0tta6bl4_prod

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-production-secret-key-here
JWT_SECRET_KEY=your-production-jwt-secret-here

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Email для alerting
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=alerts@x0tta6bl4.com
```

## Шаг 2: Настройка Docker Production Compose

### 2.1 Создание docker-compose.production.yml
```yaml
version: '3.8'

services:
  # Application
  app:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "80:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Database
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: x0tta6bl4_prod
      POSTGRES_USER: x0tta6bl4_prod
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U x0tta6bl4_prod"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
      - ./redis.conf:/etc/redis/redis.conf
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: secure_grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 2.2 Создание Production Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt requirements_min.txt ./

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "main.py"]
```

## Шаг 3: Запуск Production Deployment

### 3.1 Инициализация Базы Данных
```bash
# Запуск только базы данных
docker-compose -f docker-compose.production.yml up -d db

# Ожидание готовности базы
sleep 30

# Создание схемы (если необходимо)
docker-compose -f docker-compose.production.yml exec app python -c "
import asyncio
from production.database.init_db import init_database
asyncio.run(init_database())
"
```

### 3.2 Полный Запуск Системы
```bash
# Запуск всех сервисов
docker-compose -f docker-compose.production.yml up -d

# Проверка статуса
docker-compose -f docker-compose.production.yml ps

# Просмотр логов
docker-compose -f docker-compose.production.yml logs -f app
```

### 3.3 Проверка Deployment
```bash
# Проверка API
curl http://localhost/health
curl http://localhost/api/v1/quantum/status
curl http://localhost/api/v1/ai/status

# Проверка мониторинга
curl http://localhost:9090/-/healthy
open http://localhost:3000  # Grafana
```

## Шаг 4: Настройка Backup

### 4.1 Создание Backup Скриптов
```bash
# Создание директории для скриптов
mkdir -p scripts/backup

# Database backup script
cat > scripts/backup/backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/x0tta6bl4-production/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="x0tta6bl4_prod"
DB_USER="x0tta6bl4_prod"

mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.sql

# Redis backup
docker exec redis redis-cli SAVE
docker cp redis:/data/dump.rdb $BACKUP_DIR/redis_${DATE}.rdb

# Сжатие и ротация
gzip $BACKUP_DIR/${DB_NAME}_${DATE}.sql
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x scripts/backup/backup_db.sh
```

### 4.2 Настройка Автоматического Backup
```bash
# Добавление в crontab
(crontab -l ; echo "0 2 * * * /opt/x0tta6bl4-production/scripts/backup/backup_db.sh") | crontab -
```

## Шаг 5: Настройка Security Features

### 5.1 Quantum-Resistant Cryptography
```bash
# Установка quantum-resistant crypto libraries
pip install pqcrypto

# Генерация quantum-resistant ключей
python -c "
from pqcrypto.sign.dilithium import generate_keypair, sign, verify
public_key, secret_key = generate_keypair()
print('Quantum-resistant keys generated')
"
```

### 5.2 AI-Powered Security Monitoring
```yaml
# monitoring/security_ai_config.yaml
ai_security:
   anomaly_detection:
      model: "quantum_anomaly_detector"
      threshold: 0.85
      features: ["request_rate", "response_time", "error_rate", "quantum_coherence"]

   threat_intelligence:
      quantum_ml_enabled: true
      consciousness_level: 0.8
      phi_harmonic_analysis: true

   automated_response:
      quantum_bypass_solver: true
      adaptive_rate_limiting: true
      consciousness_evolution: true
```

### 5.3 Zero-Trust Architecture Setup
```bash
# Настройка mTLS между сервисами
kubectl apply -f k8s/security/peer-authentication.yaml

# Настройка SPIFFE/SPIRE для identity
kubectl apply -f k8s/security/spire-setup.yaml

# Quantum-enhanced authentication
python -c "
from x0tta6bl4_security.quantum_resistant_crypto import QuantumAuth
auth = QuantumAuth()
token = auth.generate_quantum_token(user_id='admin')
print('Quantum-secured token generated')
"
```

### 5.4 Advanced Encryption Setup
```bash
# Настройка end-to-end encryption
export ENCRYPTION_KEY=$(openssl rand -hex 32)
export QUANTUM_ENCRYPTION_ENABLED=true

# Инициализация quantum encryption
python -c "
from x0tta6bl4_security.quantum_resistant_crypto import QuantumEncryption
qe = QuantumEncryption()
qe.initialize_quantum_key_exchange()
print('Quantum encryption initialized')
"
```

## Шаг 6: Настройка Monitoring и Alerting

### 6.1 Prometheus Конфигурация
```yaml
# monitoring/prometheus/prometheus.yml
global:
   scrape_interval: 15s

scrape_configs:
   - job_name: 'x0tta6bl4-app'
     static_configs:
       - targets: ['app:8000']

   - job_name: 'advanced-ai-ml'
     static_configs:
       - targets: ['ai-ml-service:8001']

   - job_name: 'quantum-supremacy'
     static_configs:
       - targets: ['quantum-service:8002']

   - job_name: 'postgres'
     static_configs:
       - targets: ['db:5432']

   - job_name: 'redis'
     static_configs:
       - targets: ['redis:6379']
```

### 5.2 Настройка Email Alerting
```yaml
# monitoring/prometheus/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@x0tta6bl4.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'

receivers:
  - name: 'email'
    email_configs:
      - to: 'admin@x0tta6bl4.com'
```

## Шаг 6: Post-Deployment Проверки

### 6.1 Validation Checklist
- [ ] API endpoints отвечают корректно
- [ ] База данных доступна и содержит данные
- [ ] Redis cache работает
- [ ] Мониторинг собирает метрики
- [ ] Логи пишутся в правильные файлы
- [ ] Backup скрипты выполняются
- [ ] SSL сертификаты установлены (если HTTPS)
- [ ] Firewall настроен правильно

### 6.2 Производительность Testing
```bash
# Load testing
python load_test.py --url http://localhost --concurrency 10 --duration 60

# Memory и CPU monitoring
docker stats
```

## Шаг 7: Обслуживание и Мониторинг

### 7.1 Ежедневные Задачи
```bash
# Проверка здоровья системы
curl http://localhost/health

# Проверка логов на ошибки
tail -f logs/app.log | grep ERROR

# Проверка использования ресурсов
docker stats
```

### 7.2 Еженедельные Задачи
- Проверка backup файлов
- Очистка старых логов
- Обновление зависимостей (если необходимо)

### 7.3 Troubleshooting

#### Распространенные проблемы:
1. **Высокое использование CPU**: Проверить логи, оптимизировать код
2. **Проблемы с памятью**: Проверить memory leaks, увеличить лимиты
3. **База данных недоступна**: Проверить подключение, перезапустить сервис
4. **Redis проблемы**: Проверить persistence, очистить cache

## Шаг 8: Масштабирование

### 8.1 Горизонтальное Масштабирование
```bash
# Добавление дополнительных app инстансов
docker-compose -f docker-compose.production.yml up -d --scale app=3

# Настройка load balancer (nginx)
```

### 8.2 Вертикальное Масштабирование
```bash
# Увеличение ресурсов для контейнеров
docker-compose -f docker-compose.production.yml up -d --scale app=1 -e DOCKER_MEMORY=4g
```

## Контакты и Поддержка

- **Документация**: docs.x0tta6bl4.com
- **Issues**: GitHub Issues
- **Мониторинг**: http://your-server:3000
- **Логи**: /opt/x0tta6bl4-production/logs/

---

**Production Deployment завершен!** 🎉