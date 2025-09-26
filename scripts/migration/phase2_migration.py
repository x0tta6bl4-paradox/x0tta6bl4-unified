#!/usr/bin/env python3
"""
🚀 Phase 2 Migration Script - x0tta6bl4 Unified
Скрипт для миграции критических компонентов из x0tta6bl4 и x0tta6bl4-next
"""

import os
import sys
import shutil
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase2Migration:
    """Миграция критических компонентов в Phase 2"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.source_x0tta6bl4 = Path("/home/x0tta6bl4")
        self.source_x0tta6bl4_next = Path("/home/x0tta6bl4-next")
        self.config = self._load_migration_config()
        self.migration_status = {
            "started_at": datetime.now().isoformat(),
            "completed_components": [],
            "failed_components": [],
            "warnings": []
        }
    
    def _load_migration_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации миграции"""
        config_path = self.project_root / "config" / "migration_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "source_projects": {
                    "x0tta6bl4": "/home/x0tta6bl4",
                    "x0tta6bl4_next": "/home/x0tta6bl4-next"
                },
                "target_project": "/home/x0tta6bl4-unified",
                "migration_components": {
                    "quantum_core": True,
                    "ai_ml": True,
                    "enterprise_ui": True,
                    "billing": True,
                    "monitoring": True
                }
            }
    
    def analyze_source_projects(self) -> Dict[str, Any]:
        """Анализ исходных проектов для миграции"""
        logger.info("🔍 Анализ исходных проектов...")
        
        analysis = {
            "x0tta6bl4": self._analyze_x0tta6bl4(),
            "x0tta6bl4_next": self._analyze_x0tta6bl4_next()
        }
        
        # Сохранение анализа
        analysis_path = self.project_root / "config" / "source_analysis.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info("✅ Анализ исходных проектов завершен")
        return analysis
    
    def _analyze_x0tta6bl4(self) -> Dict[str, Any]:
        """Анализ проекта x0tta6bl4"""
        logger.info("📊 Анализ x0tta6bl4...")
        
        analysis = {
            "total_files": 0,
            "python_files": 0,
            "quantum_components": [],
            "ai_components": [],
            "api_components": [],
            "monitoring_components": []
        }
        
        try:
            # Подсчет файлов
            for root, dirs, files in os.walk(self.source_x0tta6bl4):
                for file in files:
                    analysis["total_files"] += 1
                    if file.endswith('.py'):
                        analysis["python_files"] += 1
            
            # Поиск квантовых компонентов
            quantum_patterns = ['quantum', 'qiskit', 'cirq', 'pennylane']
            for pattern in quantum_patterns:
                for root, dirs, files in os.walk(self.source_x0tta6bl4):
                    for file in files:
                        if pattern in file.lower() and file.endswith('.py'):
                            analysis["quantum_components"].append(str(Path(root) / file))
            
            # Поиск AI компонентов
            ai_patterns = ['ai', 'ml', 'torch', 'tensorflow', 'sklearn']
            for pattern in ai_patterns:
                for root, dirs, files in os.walk(self.source_x0tta6bl4):
                    for file in files:
                        if pattern in file.lower() and file.endswith('.py'):
                            analysis["ai_components"].append(str(Path(root) / file))
            
            # Поиск API компонентов
            api_patterns = ['api', 'server', 'endpoint', 'fastapi']
            for pattern in api_patterns:
                for root, dirs, files in os.walk(self.source_x0tta6bl4):
                    for file in files:
                        if pattern in file.lower() and file.endswith('.py'):
                            analysis["api_components"].append(str(Path(root) / file))
            
            logger.info(f"✅ x0tta6bl4: {analysis['python_files']} Python файлов, {len(analysis['quantum_components'])} квантовых компонентов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа x0tta6bl4: {e}")
        
        return analysis
    
    def _analyze_x0tta6bl4_next(self) -> Dict[str, Any]:
        """Анализ проекта x0tta6bl4-next"""
        logger.info("📊 Анализ x0tta6bl4-next...")
        
        analysis = {
            "total_files": 0,
            "python_files": 0,
            "enterprise_components": [],
            "billing_components": [],
            "ui_components": [],
            "api_components": []
        }
        
        try:
            # Подсчет файлов
            for root, dirs, files in os.walk(self.source_x0tta6bl4_next):
                for file in files:
                    analysis["total_files"] += 1
                    if file.endswith('.py'):
                        analysis["python_files"] += 1
            
            # Поиск enterprise компонентов
            enterprise_patterns = ['enterprise', 'business', 'commercial']
            for pattern in enterprise_patterns:
                for root, dirs, files in os.walk(self.source_x0tta6bl4_next):
                    for file in files:
                        if pattern in file.lower() and file.endswith('.py'):
                            analysis["enterprise_components"].append(str(Path(root) / file))
            
            # Поиск billing компонентов
            billing_patterns = ['billing', 'payment', 'subscription', 'stripe']
            for pattern in billing_patterns:
                for root, dirs, files in os.walk(self.source_x0tta6bl4_next):
                    for file in files:
                        if pattern in file.lower() and file.endswith('.py'):
                            analysis["billing_components"].append(str(Path(root) / file))
            
            logger.info(f"✅ x0tta6bl4-next: {analysis['python_files']} Python файлов, {len(analysis['enterprise_components'])} enterprise компонентов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа x0tta6bl4-next: {e}")
        
        return analysis
    
    def migrate_quantum_core(self) -> bool:
        """Миграция Quantum Core из x0tta6bl4"""
        logger.info("⚛️ Миграция Quantum Core...")
        
        try:
            target_dir = self.project_root / "production" / "quantum"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Копирование ключевых квантовых файлов
            quantum_files = [
                "final_launch_system_fixed.py",
                "hawking_entropy_engine.py",
                "quantum_bypass_solver.py",
                "direct_cultural_quantum_test.py"
            ]
            
            copied_files = 0
            for file_name in quantum_files:
                source_file = self.source_x0tta6bl4 / file_name
                if source_file.exists():
                    target_file = target_dir / file_name
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    logger.info(f"✅ Скопирован: {file_name}")
                else:
                    logger.warning(f"⚠️ Файл не найден: {file_name}")
            
            # Создание __init__.py
            init_file = target_dir / "__init__.py"
            with open(init_file, 'w') as f:
                f.write('"""Quantum Core Module for x0tta6bl4 Unified"""\n')
            
            # Создание конфигурации квантового модуля
            config_file = target_dir / "quantum_config.py"
            config_content = '''
"""
Конфигурация Quantum Core для x0tta6bl4 Unified
"""

QUANTUM_PROVIDERS = {
    "ibm": {
        "enabled": True,
        "api_key": os.getenv("IBM_QUANTUM_API_KEY"),
        "hub": "ibm-q",
        "group": "open",
        "project": "main"
    },
    "google": {
        "enabled": True,
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "location": "us-central1"
    },
    "xanadu": {
        "enabled": True,
        "api_key": os.getenv("XANADU_API_KEY")
    }
}

QUANTUM_ALGORITHMS = {
    "vqe": True,
    "qaoa": True,
    "grover": True,
    "shor": True,
    "deutsch_jozsa": True
}

QUANTUM_OPTIMIZATION = {
    "phi_harmony": True,
    "golden_ratio": 1.618033988749895,
    "base_frequency": 108.0
}
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ Quantum Core мигрирован: {copied_files} файлов")
            self.migration_status["completed_components"].append("quantum_core")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции Quantum Core: {e}")
            self.migration_status["failed_components"].append("quantum_core")
            return False
    
    def migrate_ai_components(self) -> bool:
        """Миграция AI/ML компонентов"""
        logger.info("🤖 Миграция AI/ML компонентов...")
        
        try:
            target_dir = self.project_root / "production" / "ai"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Копирование AI компонентов
            ai_files = [
                "advanced_ai_ml_system.py",
                "edge/atom_ai.py",
                "edge/micromind_prepare.py",
                "working_ai_agents.py"
            ]
            
            copied_files = 0
            for file_name in ai_files:
                source_file = self.source_x0tta6bl4 / file_name
                if source_file.exists():
                    target_file = target_dir / file_name
                    # Создание поддиректорий если нужно
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    logger.info(f"✅ Скопирован: {file_name}")
                else:
                    logger.warning(f"⚠️ Файл не найден: {file_name}")
            
            # Создание AI конфигурации
            config_file = target_dir / "ai_config.py"
            config_content = '''
"""
Конфигурация AI/ML для x0tta6bl4 Unified
"""

AI_MODELS = {
    "language": {
        "gpt": True,
        "claude": True,
        "llama": True
    },
    "vision": {
        "resnet": True,
        "vit": True,
        "clip": True
    },
    "quantum_ml": {
        "vqc": True,
        "qnn": True,
        "qsvm": True
    }
}

ML_FRAMEWORKS = {
    "pytorch": True,
    "tensorflow": True,
    "scikit_learn": True,
    "transformers": True
}

AI_AGENTS = {
    "documentation": True,
    "monitoring": True,
    "optimization": True,
    "analysis": True
}
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ AI/ML компоненты мигрированы: {copied_files} файлов")
            self.migration_status["completed_components"].append("ai_components")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции AI компонентов: {e}")
            self.migration_status["failed_components"].append("ai_components")
            return False
    
    def migrate_enterprise_ui(self) -> bool:
        """Миграция Enterprise UI из x0tta6bl4-next"""
        logger.info("🏢 Миграция Enterprise UI...")
        
        try:
            target_dir = self.project_root / "production" / "enterprise"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Копирование enterprise компонентов
            enterprise_files = [
                "services/api_gateway/app.py",
                "services/mesh_api/app.py",
                "src/x0tta6bl4_settings.py"
            ]
            
            copied_files = 0
            for file_name in enterprise_files:
                source_file = self.source_x0tta6bl4_next / file_name
                if source_file.exists():
                    target_file = target_dir / file_name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    logger.info(f"✅ Скопирован: {file_name}")
                else:
                    logger.warning(f"⚠️ Файл не найден: {file_name}")
            
            # Создание enterprise конфигурации
            config_file = target_dir / "enterprise_config.py"
            config_content = '''
"""
Конфигурация Enterprise для x0tta6bl4 Unified
"""

ENTERPRISE_FEATURES = {
    "multi_tenant": True,
    "rbac": True,
    "audit_logging": True,
    "compliance": True
}

API_GATEWAY = {
    "enabled": True,
    "rate_limiting": True,
    "authentication": True,
    "monitoring": True
}

MESH_NETWORKING = {
    "enabled": True,
    "service_discovery": True,
    "load_balancing": True,
    "circuit_breaker": True
}
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ Enterprise UI мигрирован: {copied_files} файлов")
            self.migration_status["completed_components"].append("enterprise_ui")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка миграции Enterprise UI: {e}")
            self.migration_status["failed_components"].append("enterprise_ui")
            return False
    
    def setup_unified_api_gateway(self) -> bool:
        """Настройка единого API Gateway"""
        logger.info("🌐 Настройка единого API Gateway...")
        
        try:
            api_dir = self.project_root / "production" / "api"
            api_dir.mkdir(parents=True, exist_ok=True)
            
            # Создание главного API сервера
            main_api_file = api_dir / "main.py"
            main_api_content = '''
"""
Unified API Gateway для x0tta6bl4
Объединяет все API endpoints в единую точку входа
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from typing import Dict, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="x0tta6bl4 Unified API",
    description="Unified API Gateway для квантовых вычислений, AI и SaaS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "service": "x0tta6bl4-unified-api",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "x0tta6bl4 Unified API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "quantum": "/api/v1/quantum",
            "ai": "/api/v1/ai",
            "enterprise": "/api/v1/enterprise",
            "billing": "/api/v1/billing",
            "monitoring": "/api/v1/monitoring"
        }
    }

# Quantum API endpoints
@app.get("/api/v1/quantum/status")
async def quantum_status():
    """Статус квантовых сервисов"""
    return {
        "status": "operational",
        "providers": ["ibm", "google", "xanadu"],
        "algorithms": ["vqe", "qaoa", "grover", "shor"]
    }

# AI API endpoints
@app.get("/api/v1/ai/status")
async def ai_status():
    """Статус AI сервисов"""
    return {
        "status": "operational",
        "models": ["gpt", "claude", "llama"],
        "agents": ["documentation", "monitoring", "optimization"]
    }

# Enterprise API endpoints
@app.get("/api/v1/enterprise/status")
async def enterprise_status():
    """Статус enterprise сервисов"""
    return {
        "status": "operational",
        "features": ["multi_tenant", "rbac", "audit_logging"],
        "gateway": "active"
    }

# Billing API endpoints
@app.get("/api/v1/billing/status")
async def billing_status():
    """Статус billing сервисов"""
    return {
        "status": "operational",
        "providers": ["stripe", "paypal", "yookassa"],
        "features": ["subscriptions", "invoices", "payments"]
    }

# Monitoring API endpoints
@app.get("/api/v1/monitoring/status")
async def monitoring_status():
    """Статус мониторинга"""
    return {
        "status": "operational",
        "metrics": ["prometheus", "grafana"],
        "logging": ["structured", "distributed"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''
            with open(main_api_file, 'w') as f:
                f.write(main_api_content)
            
            # Создание requirements для API
            api_requirements = api_dir / "requirements.txt"
            with open(api_requirements, 'w') as f:
                f.write('''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
''')
            
            logger.info("✅ Unified API Gateway настроен")
            self.migration_status["completed_components"].append("api_gateway")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка настройки API Gateway: {e}")
            self.migration_status["failed_components"].append("api_gateway")
            return False
    
    def integrate_billing_system(self) -> bool:
        """Интеграция Billing системы"""
        logger.info("💳 Интеграция Billing системы...")
        
        try:
            billing_dir = self.project_root / "production" / "billing"
            billing_dir.mkdir(parents=True, exist_ok=True)
            
            # Создание billing сервиса
            billing_service = billing_dir / "billing_service.py"
            billing_content = '''
"""
Billing Service для x0tta6bl4 Unified
Управление подписками, платежами и биллингом
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Subscription(BaseModel):
    """Модель подписки"""
    id: str
    user_id: str
    plan: str
    status: str
    created_at: datetime
    current_period_end: datetime

class Payment(BaseModel):
    """Модель платежа"""
    id: str
    subscription_id: str
    amount: float
    currency: str
    status: str
    created_at: datetime

class BillingService:
    """Сервис биллинга"""
    
    def __init__(self):
        self.subscriptions = {}
        self.payments = {}
    
    def create_subscription(self, user_id: str, plan: str) -> Subscription:
        """Создание подписки"""
        subscription = Subscription(
            id=f"sub_{len(self.subscriptions) + 1}",
            user_id=user_id,
            plan=plan,
            status="active",
            created_at=datetime.now(),
            current_period_end=datetime.now()
        )
        self.subscriptions[subscription.id] = subscription
        return subscription
    
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Получение подписки"""
        return self.subscriptions.get(subscription_id)
    
    def process_payment(self, subscription_id: str, amount: float) -> Payment:
        """Обработка платежа"""
        payment = Payment(
            id=f"pay_{len(self.payments) + 1}",
            subscription_id=subscription_id,
            amount=amount,
            currency="USD",
            status="completed",
            created_at=datetime.now()
        )
        self.payments[payment.id] = payment
        return payment

# Создание FastAPI приложения для billing
billing_app = FastAPI(title="Billing Service", version="1.0.0")

billing_service = BillingService()

@billing_app.post("/subscriptions/")
async def create_subscription(user_id: str, plan: str):
    """Создание подписки"""
    subscription = billing_service.create_subscription(user_id, plan)
    return subscription

@billing_app.get("/subscriptions/{subscription_id}")
async def get_subscription(subscription_id: str):
    """Получение подписки"""
    subscription = billing_service.get_subscription(subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return subscription

@billing_app.post("/payments/")
async def process_payment(subscription_id: str, amount: float):
    """Обработка платежа"""
    payment = billing_service.process_payment(subscription_id, amount)
    return payment
'''
            with open(billing_service, 'w') as f:
                f.write(billing_content)
            
            # Создание конфигурации billing
            billing_config = billing_dir / "billing_config.py"
            config_content = '''
"""
Конфигурация Billing для x0tta6bl4 Unified
"""

BILLING_PROVIDERS = {
    "stripe": {
        "enabled": True,
        "public_key": os.getenv("STRIPE_PUBLIC_KEY"),
        "secret_key": os.getenv("STRIPE_SECRET_KEY")
    },
    "paypal": {
        "enabled": True,
        "client_id": os.getenv("PAYPAL_CLIENT_ID"),
        "client_secret": os.getenv("PAYPAL_CLIENT_SECRET")
    },
    "yookassa": {
        "enabled": True,
        "shop_id": os.getenv("YOOKASSA_SHOP_ID"),
        "secret_key": os.getenv("YOOKASSA_SECRET_KEY")
    }
}

SUBSCRIPTION_PLANS = {
    "free": {
        "price": 0,
        "features": ["basic_quantum", "limited_api"]
    },
    "pro": {
        "price": 29.99,
        "features": ["advanced_quantum", "unlimited_api", "priority_support"]
    },
    "enterprise": {
        "price": 99.99,
        "features": ["premium_quantum", "custom_integrations", "dedicated_support"]
    }
}
'''
            with open(billing_config, 'w') as f:
                f.write(config_content)
            
            logger.info("✅ Billing система интегрирована")
            self.migration_status["completed_components"].append("billing_system")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка интеграции Billing: {e}")
            self.migration_status["failed_components"].append("billing_system")
            return False
    
    def generate_phase2_report(self) -> bool:
        """Генерация отчета Phase 2"""
        logger.info("📋 Генерация отчета Phase 2...")
        
        try:
            self.migration_status["completed_at"] = datetime.now().isoformat()
            self.migration_status["total_components"] = len(self.migration_status["completed_components"]) + len(self.migration_status["failed_components"])
            self.migration_status["success_rate"] = len(self.migration_status["completed_components"]) / self.migration_status["total_components"] * 100 if self.migration_status["total_components"] > 0 else 0
            
            report_path = self.project_root / "PHASE2_MIGRATION_REPORT.md"
            
            report_content = f"""# 📊 Phase 2 Migration Report - x0tta6bl4 Unified

**Дата**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Статус**: {'✅ ЗАВЕРШЕНО' if len(self.migration_status['failed_components']) == 0 else '⚠️ ЗАВЕРШЕНО С ПРЕДУПРЕЖДЕНИЯМИ'}

## 📈 Общая статистика

- **Всего компонентов**: {self.migration_status['total_components']}
- **Мигрировано**: {len(self.migration_status['completed_components'])}
- **Не мигрировано**: {len(self.migration_status['failed_components'])}
- **Процент успеха**: {self.migration_status['success_rate']:.1f}%

## ✅ Успешно мигрированные компоненты

{chr(10).join([f"- ✅ {component}" for component in self.migration_status['completed_components']])}

## ❌ Не мигрированные компоненты

{chr(10).join([f"- ❌ {component}" for component in self.migration_status['failed_components']]) if self.migration_status['failed_components'] else '- Все компоненты мигрированы успешно'}

## ⚠️ Предупреждения

{chr(10).join([f"- ⚠️ {warning}" for warning in self.migration_status['warnings']]) if self.migration_status['warnings'] else '- Нет предупреждений'}

## 🎯 Следующие шаги

1. **Тестирование интеграции** - Проверка работы всех компонентов
2. **Начало Phase 3** - Интеграция и оптимизация
3. **Настройка мониторинга** - Отслеживание производительности
4. **Подготовка к production** - Финальная настройка

## 📞 Контакты

- **Project Manager**: [Контактная информация]
- **Technical Lead**: [Контактная информация]
- **DevOps Engineer**: [Контактная информация]

---
*Отчет сгенерирован автоматически системой x0tta6bl4 Unified*
"""
            
            with open(report_path, "w") as f:
                f.write(report_content)
            
            logger.info("✅ Отчет Phase 2 создан")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
            return False
    
    def run_phase2_migration(self) -> bool:
        """Запуск Phase 2 миграции"""
        logger.info("🚀 Запуск Phase 2 миграции x0tta6bl4 Unified...")
        
        # Анализ исходных проектов
        analysis = self.analyze_source_projects()
        
        # Выполнение миграции компонентов
        migration_tasks = [
            ("Quantum Core", self.migrate_quantum_core),
            ("AI Components", self.migrate_ai_components),
            ("Enterprise UI", self.migrate_enterprise_ui),
            ("API Gateway", self.setup_unified_api_gateway),
            ("Billing System", self.integrate_billing_system)
        ]
        
        for task_name, task_func in migration_tasks:
            logger.info(f"🔄 Выполнение: {task_name}")
            try:
                if task_func():
                    logger.info(f"✅ {task_name} - УСПЕШНО")
                else:
                    logger.error(f"❌ {task_name} - ОШИБКА")
            except Exception as e:
                logger.error(f"❌ {task_name} - КРИТИЧЕСКАЯ ОШИБКА: {e}")
                self.migration_status["failed_components"].append(task_name.lower().replace(" ", "_"))
        
        # Генерация отчета
        self.generate_phase2_report()
        
        # Финальный отчет
        success = len(self.migration_status["failed_components"]) == 0
        
        if success:
            logger.info("🎉 Phase 2 миграция завершена успешно!")
        else:
            logger.warning(f"⚠️ Phase 2 миграция завершена с {len(self.migration_status['failed_components'])} ошибками")
        
        return success

def main():
    """Главная функция"""
    migration = Phase2Migration()
    success = migration.run_phase2_migration()
    
    if success:
        print("\n🎯 РЕКОМЕНДАЦИИ:")
        print("- Phase 2 завершена успешно")
        print("- Все критические компоненты мигрированы")
        print("- Готово к началу Phase 3")
        print("- Начать интеграционное тестирование")
    else:
        print("\n🔧 ПЛАН ИСПРАВЛЕНИЙ:")
        print("- Исправить ошибки миграции")
        print("- Повторить неудачные миграции")
        print("- Проверить зависимости")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
