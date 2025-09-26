# Установка x0tta6bl4 Unified Platform

## Предварительные требования

### Системные требования
- **Операционная система**: Linux (Ubuntu 20.04+), macOS (10.15+), или Windows 10+
- **Python**: 3.12 или выше
- **Docker**: 24.0+ (для контейнеризации)
- **Docker Compose**: 2.0+ (для оркестрации)
- **Kubernetes**: 1.24+ (для production развертывания)
- **PostgreSQL**: 13+ (база данных)
- **Redis**: 6+ (кеширование)
- **RAM**: Минимум 8GB, рекомендуется 16GB+
- **CPU**: Минимум 4 ядра, рекомендуется 8+ ядер
- **Диск**: Минимум 50GB свободного места

### Необходимое ПО
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.12 python3.12-venv postgresql redis-server docker.io docker-compose

# macOS (с Homebrew)
brew install python@3.12 postgresql redis docker docker-compose

# Windows (с Chocolatey)
choco install python postgresql redis-64 docker-desktop
```

## Установка из исходного кода

### 1. Клонирование репозитория
```bash
git clone https://github.com/x0tta6bl4/x0tta6bl4-unified.git
cd x0tta6bl4-unified
```

### 2. Создание виртуального окружения
```bash
# Создание виртуального окружения
python3.12 -m venv .venv

# Активация виртуального окружения
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей
```bash
# Установка основных зависимостей
pip install -r requirements.txt

# Установка зависимостей для разработки (опционально)
pip install -r requirements-dev.txt
```

### 4. Настройка переменных окружения
```bash
# Копирование шаблона конфигурации
cp .env.example .env

# Редактирование переменных окружения
nano .env  # или используйте любой текстовый редактор
```

Пример файла `.env`:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/x0tta6bl4_unified

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Quantum Computing
QISKIT_TOKEN=your-qiskit-token
IBM_QUANTUM_BACKEND=ibmq_qasm_simulator

# AI/ML
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# Environment
ENVIRONMENT=development
DEBUG=True
```

### 5. Настройка базы данных
```bash
# Создание базы данных PostgreSQL
createdb x0tta6bl4_unified

# Запуск миграций
alembic upgrade head
```

### 6. Запуск платформы
```bash
# Запуск в режиме разработки
python main.py

# Или с uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker развертывание

### Быстрый старт с Docker Compose
```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка сервисов
docker-compose down
```

### Структура Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/x0tta6bl4_unified
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=x0tta6bl4_unified
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes развертывание

### Предварительная настройка
```bash
# Создание namespace
kubectl create namespace x0tta6bl4

# Применение конфигураций
kubectl apply -f k8s/base/

# Установка секретов
kubectl create secret generic x0tta6bl4-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --namespace=x0tta6bl4
```

### Развертывание компонентов
```bash
# Развертывание базы данных
kubectl apply -f k8s/base/postgres/

# Развертывание Redis
kubectl apply -f k8s/base/redis/

# Развертывание приложения
kubectl apply -f k8s/base/app/

# Развертывание мониторинга
kubectl apply -f monitoring/
```

### Масштабирование с KEDA
```bash
# Применение KEDA конфигураций
kubectl apply -f k8s/keda/

# Проверка состояния
kubectl get scaledobjects -n x0tta6bl4
```

## Проверка установки

### Проверка здоровья системы
```bash
# Проверка API
curl http://localhost:8000/health

# Проверка компонентов
curl http://localhost:8000/api/v1/quantum/status
curl http://localhost:8000/api/v1/ai/status
curl http://localhost:8000/api/v1/enterprise/status
curl http://localhost:8000/api/v1/billing/status
```

### Проверка мониторинга
```bash
# Prometheus метрики
curl http://localhost:9090/metrics

# Grafana dashboard
open http://localhost:3000
```

## Устранение неполадок

### Распространенные проблемы

#### Ошибка подключения к базе данных
```bash
# Проверка статуса PostgreSQL
sudo systemctl status postgresql

# Проверка логов
sudo tail -f /var/log/postgresql/postgresql-*.log
```

#### Ошибка подключения к Redis
```bash
# Проверка статуса Redis
redis-cli ping
```

#### Проблемы с Docker
```bash
# Очистка Docker
docker system prune -a

# Пересборка образов
docker-compose build --no-cache
```

#### Ошибки зависимостей Python
```bash
# Обновление pip
pip install --upgrade pip

# Переустановка зависимостей
pip install -r requirements.txt --force-reinstall
```

## Следующие шаги

После успешной установки:

1. **Изучите документацию**: Ознакомьтесь с [руководством разработчика](developer/getting-started.md)
2. **Настройте мониторинг**: Следуйте [руководству по мониторингу](deployment/monitoring.md)
3. **Разверните в production**: Используйте [production конфигурации](deployment/production.md)
4. **Присоединяйтесь к сообществу**: Задавайте вопросы в [Discord](https://discord.gg/x0tta6bl4)

## Поддержка

- **Документация**: [docs.x0tta6bl4.com](https://docs.x0tta6bl4.com)
- **Issues**: [GitHub Issues](https://github.com/x0tta6bl4/x0tta6bl4-unified/issues)
- **Сообщество**: [Discord](https://discord.gg/x0tta6bl4)
- **Email**: support@x0tta6bl4.com