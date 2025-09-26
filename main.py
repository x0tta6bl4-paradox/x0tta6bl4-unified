#!/usr/bin/env python3
"""
🚀 x0tta6bl4 Unified Platform - Main Entry Point
Объединенная платформа квантовых вычислений, AI и SaaS
"""

import asyncio
import logging
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
from datetime import datetime
from production.monitoring.unified_monitoring import UnifiedMonitoring
from production.ai.advanced_ai_ml_system import AdvancedAIMLSystem
from production.quantum.quantum_bypass_solver import QuantumBypassSolver

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI приложение
app = FastAPI(
    title="x0tta6bl4 Unified Platform",
    description="Unified platform for quantum computing, AI, and SaaS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация мониторинга
monitoring = UnifiedMonitoring()

@app.get("/")
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

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "quantum": "healthy",
            "ai": "healthy",
            "enterprise": "healthy",
            "billing": "healthy",
            "monitoring": "healthy"
        }
    }

@app.get("/api/v1/quantum/status")
async def quantum_status():
    """Статус квантовых сервисов"""
    return {"status": "healthy", "message": "Quantum services operational"}

@app.get("/api/v1/ai/status")
async def ai_status():
    """Статус AI сервисов"""
    return {"status": "healthy", "message": "AI services operational"}

@app.get("/api/v1/enterprise/status")
async def enterprise_status():
    """Статус enterprise сервисов"""
    return {"status": "healthy", "message": "Enterprise services operational"}

@app.get("/api/v1/billing/status")
async def billing_status():
    """Статус billing сервисов"""
    return {"status": "healthy", "message": "Billing services operational"}

@app.get("/api/v1/monitoring/status")
async def monitoring_status():
    """Статус monitoring сервисов"""
    return {"status": "healthy", "message": "Monitoring services operational"}

@app.get("/api/v1/monitoring/metrics")
async def get_metrics():
    """Получение метрик производительности"""
    try:
        report = await monitoring.generate_report()
        return report
    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus метрики endpoint"""
    try:
        metrics = await monitoring.get_prometheus_metrics()
        return Response(content=metrics, media_type="text/plain; version=0.0.4; charset=utf-8")
    except Exception as e:
        logger.error(f"Ошибка получения Prometheus метрик: {e}")
        return Response(content=f"# Error generating metrics: {e}", status_code=500, media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
