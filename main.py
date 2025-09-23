#!/usr/bin/env python3
"""
🚀 x0tta6bl4 Unified Platform - Main Entry Point
Объединенная платформа квантовых вычислений, AI и SaaS
"""

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
from datetime import datetime

# Импорт компонентов
from production.quantum import QuantumCore
from production.ai import AICore
from production.enterprise import EnterpriseCore
from production.billing import BillingCore
from production.api import APIGateway

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class X0tta6bl4Unified:
    """Главный класс unified платформы x0tta6bl4"""
    
    def __init__(self):
        self.app = FastAPI(
            title="x0tta6bl4 Unified Platform",
            description="Unified platform for quantum computing, AI, and SaaS",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Инициализация компонентов
        self.quantum_core = QuantumCore()
        self.ai_core = AICore()
        self.enterprise_core = EnterpriseCore()
        self.billing_core = BillingCore()
        self.api_gateway = APIGateway()
        
        # Настройка middleware
        self._setup_middleware()
        
        # Настройка routes
        self._setup_routes()
        
        logger.info("✅ x0tta6bl4 Unified Platform инициализирована")
    
    def _setup_middleware(self):
        """Настройка middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Настройка маршрутов"""
        
        @self.app.get("/")
        async def root():
            """Корневой endpoint"""
            return {
                "message": "x0tta6bl4 Unified Platform",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "quantum": "active",
                    "ai": "active",
                    "enterprise": "active",
                    "billing": "active",
                    "api": "active"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Проверка здоровья системы"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": await self._check_components_health()
            }
        
        @self.app.get("/api/v1/quantum/status")
        async def quantum_status():
            """Статус квантовых сервисов"""
            return await self.quantum_core.get_status()
        
        @self.app.get("/api/v1/ai/status")
        async def ai_status():
            """Статус AI сервисов"""
            return await self.ai_core.get_status()
        
        @self.app.get("/api/v1/enterprise/status")
        async def enterprise_status():
            """Статус enterprise сервисов"""
            return await self.enterprise_core.get_status()
        
        @self.app.get("/api/v1/billing/status")
        async def billing_status():
            """Статус billing сервисов"""
            return await self.billing_core.get_status()
    
    async def _check_components_health(self) -> Dict[str, str]:
        """Проверка здоровья компонентов"""
        try:
            quantum_health = await self.quantum_core.health_check()
            ai_health = await self.ai_core.health_check()
            enterprise_health = await self.enterprise_core.health_check()
            billing_health = await self.billing_core.health_check()
            
            return {
                "quantum": "healthy" if quantum_health else "unhealthy",
                "ai": "healthy" if ai_health else "unhealthy",
                "enterprise": "healthy" if enterprise_health else "unhealthy",
                "billing": "healthy" if billing_health else "unhealthy"
            }
        except Exception as e:
            logger.error(f"Ошибка проверки здоровья компонентов: {e}")
            return {
                "quantum": "unknown",
                "ai": "unknown",
                "enterprise": "unknown",
                "billing": "unknown"
            }
    
    async def start(self):
        """Запуск unified платформы"""
        logger.info("🚀 Запуск x0tta6bl4 Unified Platform...")
        
        try:
            # Инициализация всех компонентов
            await self.quantum_core.initialize()
            await self.ai_core.initialize()
            await self.enterprise_core.initialize()
            await self.billing_core.initialize()
            
            logger.info("✅ Все компоненты инициализированы")
            logger.info("🌐 x0tta6bl4 Unified Platform готова к работе")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            raise
    
    async def stop(self):
        """Остановка unified платформы"""
        logger.info("🛑 Остановка x0tta6bl4 Unified Platform...")
        
        try:
            await self.quantum_core.shutdown()
            await self.ai_core.shutdown()
            await self.enterprise_core.shutdown()
            await self.billing_core.shutdown()
            
            logger.info("✅ Все компоненты остановлены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка остановки: {e}")

# Создание экземпляра unified платформы
unified_platform = X0tta6bl4Unified()

# FastAPI приложение
app = unified_platform.app

@app.on_event("startup")
async def startup_event():
    """Событие запуска"""
    await unified_platform.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Событие остановки"""
    await unified_platform.stop()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
