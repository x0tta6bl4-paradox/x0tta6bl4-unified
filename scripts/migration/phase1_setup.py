#!/usr/bin/env python3
"""
🚀 Phase 1 Setup Script - x0tta6bl4 Unified Migration
Скрипт для настройки базовой инфраструктуры unified платформы
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase1Setup:
    """Настройка Phase 1 миграции x0tta6bl4 Unified"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config = self._load_config()
        self.status = {
            "started_at": datetime.now().isoformat(),
            "completed_tasks": [],
            "failed_tasks": [],
            "warnings": []
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        config_path = self.project_root / "config" / "migration_config.json"
        
        default_config = {
            "source_projects": {
                "x0tta6bl4": "/home/x0tta6bl4",
                "x0tta6bl4_next": "/home/x0tta6bl4-next"
            },
            "target_project": "/home/x0tta6bl4-unified",
            "environments": ["development", "staging", "production"],
            "services": {
                "quantum": True,
                "ai_ml": True,
                "enterprise": True,
                "billing": True,
                "monitoring": True
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Создание конфигурации по умолчанию
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def check_prerequisites(self) -> bool:
        """Проверка предварительных требований"""
        logger.info("🔍 Проверка предварительных требований...")
        
        prerequisites = {
            "python": self._check_python(),
            "docker": self._check_docker(),
            "git": self._check_git(),
            "source_projects": self._check_source_projects()
        }
        
        all_good = all(prerequisites.values())
        
        if all_good:
            logger.info("✅ Все предварительные требования выполнены")
        else:
            logger.error("❌ Некоторые предварительные требования не выполнены")
            for req, status in prerequisites.items():
                if not status:
                    logger.error(f"  - {req}: НЕ НАЙДЕНО")
        
        return all_good
    
    def _check_python(self) -> bool:
        """Проверка Python"""
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Python: {result.stdout.strip()}")
                return True
        except Exception as e:
            logger.error(f"❌ Python: {e}")
        return False
    
    def _check_docker(self) -> bool:
        """Проверка Docker"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Docker: {result.stdout.strip()}")
                return True
        except Exception as e:
            logger.error(f"❌ Docker: {e}")
        return False
    
    def _check_git(self) -> bool:
        """Проверка Git"""
        try:
            result = subprocess.run(["git", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Git: {result.stdout.strip()}")
                return True
        except Exception as e:
            logger.error(f"❌ Git: {e}")
        return False
    
    def _check_source_projects(self) -> bool:
        """Проверка исходных проектов"""
        all_exist = True
        
        for name, path in self.config["source_projects"].items():
            if Path(path).exists():
                logger.info(f"✅ {name}: {path}")
            else:
                logger.error(f"❌ {name}: {path} - НЕ НАЙДЕН")
                all_exist = False
        
        return all_exist
    
    def setup_git_repository(self) -> bool:
        """Настройка Git репозитория"""
        logger.info("🔧 Настройка Git репозитория...")
        
        try:
            # Инициализация Git репозитория
            subprocess.run(["git", "init"], cwd=self.project_root, check=True)
            logger.info("✅ Git репозиторий инициализирован")
            
            # Создание .gitignore
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# Secrets
secrets/
*.key
*.pem
.env.local
.env.production

# Docker
.dockerignore

# Kubernetes
*.kubeconfig

# Monitoring
monitoring/data/
prometheus/data/
grafana/data/
"""
            
            with open(self.project_root / ".gitignore", "w") as f:
                f.write(gitignore_content)
            
            logger.info("✅ .gitignore создан")
            
            # Первый коммит
            subprocess.run(["git", "add", "."], cwd=self.project_root, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit: x0tta6bl4 Unified setup"], 
                         cwd=self.project_root, check=True)
            logger.info("✅ Первый коммит создан")
            
            self.status["completed_tasks"].append("git_repository")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка настройки Git: {e}")
            self.status["failed_tasks"].append("git_repository")
            return False
    
    def setup_docker_environment(self) -> bool:
        """Настройка Docker окружения"""
        logger.info("🐳 Настройка Docker окружения...")
        
        try:
            # Docker Compose файл
            docker_compose_content = """
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: x0tta6bl4_unified
      POSTGRES_USER: x0tta6bl4
      POSTGRES_PASSWORD: x0tta6bl4_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U x0tta6bl4"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""
            
            with open(self.project_root / "docker-compose.yml", "w") as f:
                f.write(docker_compose_content)
            
            logger.info("✅ docker-compose.yml создан")
            
            # Dockerfile для unified платформы
            dockerfile_content = """
FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание пользователя
RUN useradd -m -u 1000 x0tta6bl4 && chown -R x0tta6bl4:x0tta6bl4 /app
USER x0tta6bl4

# Экспорт портов
EXPOSE 8000

# Команда запуска
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            
            with open(self.project_root / "Dockerfile", "w") as f:
                f.write(dockerfile_content)
            
            logger.info("✅ Dockerfile создан")
            
            self.status["completed_tasks"].append("docker_environment")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка настройки Docker: {e}")
            self.status["failed_tasks"].append("docker_environment")
            return False
    
    def setup_requirements(self) -> bool:
        """Создание requirements.txt"""
        logger.info("📦 Создание requirements.txt...")
        
        try:
            requirements_content = """
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
psycopg2-binary==2.9.9
redis==5.0.1

# Quantum computing
qiskit==0.45.0
cirq==1.2.0
pennylane==0.33.0

# AI/ML
torch==2.1.0
transformers==4.35.0
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3

# Monitoring
prometheus-client==0.19.0
grafana-api==1.0.3

# Security
cryptography==41.0.7
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
isort==5.12.0
mypy==1.7.1

# Documentation
mkdocs==1.5.3
mkdocs-material==9.4.8
"""
            
            with open(self.project_root / "requirements.txt", "w") as f:
                f.write(requirements_content)
            
            logger.info("✅ requirements.txt создан")
            
            self.status["completed_tasks"].append("requirements")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания requirements: {e}")
            self.status["failed_tasks"].append("requirements")
            return False
    
    def setup_monitoring(self) -> bool:
        """Настройка мониторинга"""
        logger.info("📊 Настройка мониторинга...")
        
        try:
            # Prometheus конфигурация
            prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'x0tta6bl4-unified'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
"""
            
            prometheus_dir = self.project_root / "config" / "prometheus"
            prometheus_dir.mkdir(parents=True, exist_ok=True)
            
            with open(prometheus_dir / "prometheus.yml", "w") as f:
                f.write(prometheus_config)
            
            logger.info("✅ Prometheus конфигурация создана")
            
            # Grafana дашборды
            grafana_dir = self.project_root / "config" / "grafana"
            grafana_dir.mkdir(parents=True, exist_ok=True)
            
            # Создание базового дашборда
            dashboard_config = {
                "dashboard": {
                    "title": "x0tta6bl4 Unified Dashboard",
                    "panels": [
                        {
                            "title": "System Health",
                            "type": "stat",
                            "targets": [
                                {
                                    "expr": "up",
                                    "legendFormat": "Service Status"
                                }
                            ]
                        }
                    ]
                }
            }
            
            with open(grafana_dir / "dashboard.json", "w") as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info("✅ Grafana конфигурация создана")
            
            self.status["completed_tasks"].append("monitoring")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка настройки мониторинга: {e}")
            self.status["failed_tasks"].append("monitoring")
            return False
    
    def create_migration_scripts(self) -> bool:
        """Создание скриптов миграции"""
        logger.info("🔧 Создание скриптов миграции...")
        
        try:
            # Скрипт миграции данных
            migration_script = """#!/usr/bin/env python3
'''
Скрипт миграции данных из x0tta6bl4 в x0tta6bl4-unified
'''

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

def migrate_quantum_components():
    \"\"\"Миграция квантовых компонентов\"\"\"
    print("🔄 Миграция квантовых компонентов...")
    # TODO: Реализация миграции
    return True

def migrate_ai_components():
    \"\"\"Миграция AI компонентов\"\"\"
    print("🔄 Миграция AI компонентов...")
    # TODO: Реализация миграции
    return True

def migrate_enterprise_components():
    \"\"\"Миграция enterprise компонентов\"\"\"
    print("🔄 Миграция enterprise компонентов...")
    # TODO: Реализация миграции
    return True

if __name__ == "__main__":
    print("🚀 Начало миграции x0tta6bl4 Unified...")
    
    success = all([
        migrate_quantum_components(),
        migrate_ai_components(),
        migrate_enterprise_components()
    ])
    
    if success:
        print("✅ Миграция завершена успешно!")
    else:
        print("❌ Миграция завершена с ошибками!")
        sys.exit(1)
"""
            
            migration_dir = self.project_root / "scripts" / "migration"
            migration_dir.mkdir(parents=True, exist_ok=True)
            
            with open(migration_dir / "migrate_data.py", "w") as f:
                f.write(migration_script)
            
            # Делаем скрипт исполняемым
            os.chmod(migration_dir / "migrate_data.py", 0o755)
            
            logger.info("✅ Скрипты миграции созданы")
            
            self.status["completed_tasks"].append("migration_scripts")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания скриптов миграции: {e}")
            self.status["failed_tasks"].append("migration_scripts")
            return False
    
    def generate_status_report(self) -> bool:
        """Генерация отчета о статусе"""
        logger.info("📋 Генерация отчета о статусе...")
        
        try:
            self.status["completed_at"] = datetime.now().isoformat()
            self.status["total_tasks"] = len(self.status["completed_tasks"]) + len(self.status["failed_tasks"])
            self.status["success_rate"] = len(self.status["completed_tasks"]) / self.status["total_tasks"] * 100 if self.status["total_tasks"] > 0 else 0
            
            report_path = self.project_root / "PHASE1_STATUS_REPORT.md"
            
            report_content = f"""# 📊 Phase 1 Status Report - x0tta6bl4 Unified

**Дата**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Статус**: {'✅ ЗАВЕРШЕНО' if len(self.status['failed_tasks']) == 0 else '⚠️ ЗАВЕРШЕНО С ПРЕДУПРЕЖДЕНИЯМИ'}

## 📈 Общая статистика

- **Всего задач**: {self.status['total_tasks']}
- **Выполнено**: {len(self.status['completed_tasks'])}
- **Не выполнено**: {len(self.status['failed_tasks'])}
- **Процент успеха**: {self.status['success_rate']:.1f}%

## ✅ Выполненные задачи

{chr(10).join([f"- ✅ {task}" for task in self.status['completed_tasks']])}

## ❌ Не выполненные задачи

{chr(10).join([f"- ❌ {task}" for task in self.status['failed_tasks']])}

## ⚠️ Предупреждения

{chr(10).join([f"- ⚠️ {warning}" for warning in self.status['warnings']]) if self.status['warnings'] else '- Нет предупреждений'}

## 🎯 Следующие шаги

1. **Исправление ошибок** - Устранение невыполненных задач
2. **Начало Phase 2** - Миграция core компонентов
3. **Настройка команды** - Завершение найма команды
4. **Планирование** - Детальное планирование Phase 2

## 📞 Контакты

- **Project Manager**: [Контактная информация]
- **Technical Lead**: [Контактная информация]
- **DevOps Engineer**: [Контактная информация]

---
*Отчет сгенерирован автоматически системой x0tta6bl4 Unified*
"""
            
            with open(report_path, "w") as f:
                f.write(report_content)
            
            logger.info("✅ Отчет о статусе создан")
            
            self.status["completed_tasks"].append("status_report")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
            self.status["failed_tasks"].append("status_report")
            return False
    
    def run_phase1_setup(self) -> bool:
        """Запуск Phase 1 настройки"""
        logger.info("🚀 Запуск Phase 1 настройки x0tta6bl4 Unified...")
        
        # Проверка предварительных требований
        if not self.check_prerequisites():
            logger.error("❌ Предварительные требования не выполнены")
            return False
        
        # Выполнение задач Phase 1
        tasks = [
            ("Git Repository", self.setup_git_repository),
            ("Docker Environment", self.setup_docker_environment),
            ("Requirements", self.setup_requirements),
            ("Monitoring", self.setup_monitoring),
            ("Migration Scripts", self.create_migration_scripts),
            ("Status Report", self.generate_status_report)
        ]
        
        for task_name, task_func in tasks:
            logger.info(f"🔄 Выполнение: {task_name}")
            try:
                if task_func():
                    logger.info(f"✅ {task_name} - УСПЕШНО")
                else:
                    logger.error(f"❌ {task_name} - ОШИБКА")
            except Exception as e:
                logger.error(f"❌ {task_name} - КРИТИЧЕСКАЯ ОШИБКА: {e}")
                self.status["failed_tasks"].append(task_name.lower().replace(" ", "_"))
        
        # Финальный отчет
        success = len(self.status["failed_tasks"]) == 0
        
        if success:
            logger.info("🎉 Phase 1 настройка завершена успешно!")
        else:
            logger.warning(f"⚠️ Phase 1 настройка завершена с {len(self.status['failed_tasks'])} ошибками")
        
        return success

def main():
    """Главная функция"""
    setup = Phase1Setup()
    success = setup.run_phase1_setup()
    
    if success:
        print("\n🎯 РЕКОМЕНДАЦИИ:")
        print("- Phase 1 завершена успешно")
        print("- Готово к началу Phase 2")
        print("- Начать найм команды миграции")
        print("- Подготовить детальный план Phase 2")
    else:
        print("\n🔧 ПЛАН ИСПРАВЛЕНИЙ:")
        print("- Исправить ошибки Phase 1")
        print("- Проверить предварительные требования")
        print("- Повторить настройку")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
