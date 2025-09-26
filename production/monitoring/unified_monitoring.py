
"""
Unified Monitoring для x0tta6bl4
Объединенный мониторинг всех компонентов
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import json
import psutil
import time
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

class UnifiedMonitoring:
    """Unified мониторинг системы"""

    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.components = {}

        # Prometheus метрики (с обработкой дубликатов)
        try:
            self.cpu_usage = Gauge('x0tta6bl4_cpu_usage_percent', 'CPU usage percentage')
            self.memory_usage = Gauge('x0tta6bl4_memory_usage_percent', 'Memory usage percentage')
            self.disk_usage = Gauge('x0tta6bl4_disk_usage_percent', 'Disk usage percentage')

            # Метрики компонентов
            self.quantum_requests = Counter('x0tta6bl4_quantum_requests_total', 'Total quantum requests')
            self.ai_requests = Counter('x0tta6bl4_ai_requests_total', 'Total AI requests')
            self.enterprise_requests = Counter('x0tta6bl4_enterprise_requests_total', 'Total enterprise requests')
            self.billing_requests = Counter('x0tta6bl4_billing_requests_total', 'Total billing requests')

            # Метрики производительности
            self.quantum_error_rate = Gauge('x0tta6bl4_quantum_error_rate', 'Quantum error rate percentage')
            self.ai_model_accuracy = Gauge('x0tta6bl4_ai_model_accuracy', 'AI model accuracy percentage')
            self.enterprise_response_time = Histogram('x0tta6bl4_enterprise_response_time_seconds', 'Enterprise response time in seconds')
            self.billing_failed_payments = Counter('x0tta6bl4_billing_failed_payments_total', 'Total failed payments')
            self.billing_conversion_rate = Gauge('x0tta6bl4_billing_conversion_rate', 'Billing conversion rate percentage')

            # Бизнес метрики
            self.enterprise_active_users = Gauge('x0tta6bl4_enterprise_active_users', 'Number of active enterprise users')
            self.billing_active_disputes = Gauge('x0tta6bl4_billing_active_disputes', 'Number of active billing disputes')
            self.quantum_memory_bytes = Gauge('x0tta6bl4_quantum_memory_bytes', 'Quantum memory usage in bytes')
        except ValueError:
            # Метрики уже зарегистрированы, получаем существующие
            from prometheus_client import REGISTRY
            self.cpu_usage = REGISTRY._names_to_collectors['x0tta6bl4_cpu_usage_percent']
            self.memory_usage = REGISTRY._names_to_collectors['x0tta6bl4_memory_usage_percent']
            self.disk_usage = REGISTRY._names_to_collectors['x0tta6bl4_disk_usage_percent']
            self.quantum_requests = REGISTRY._names_to_collectors['x0tta6bl4_quantum_requests_total']
            self.ai_requests = REGISTRY._names_to_collectors['x0tta6bl4_ai_requests_total']
            self.enterprise_requests = REGISTRY._names_to_collectors['x0tta6bl4_enterprise_requests_total']
            self.billing_requests = REGISTRY._names_to_collectors['x0tta6bl4_billing_requests_total']
            self.quantum_error_rate = REGISTRY._names_to_collectors['x0tta6bl4_quantum_error_rate']
            self.ai_model_accuracy = REGISTRY._names_to_collectors['x0tta6bl4_ai_model_accuracy']
            self.enterprise_response_time = REGISTRY._names_to_collectors['x0tta6bl4_enterprise_response_time_seconds']
            self.billing_failed_payments = REGISTRY._names_to_collectors['x0tta6bl4_billing_failed_payments_total']
            self.billing_conversion_rate = REGISTRY._names_to_collectors['x0tta6bl4_billing_conversion_rate']
            self.enterprise_active_users = REGISTRY._names_to_collectors['x0tta6bl4_enterprise_active_users']
            self.billing_active_disputes = REGISTRY._names_to_collectors['x0tta6bl4_billing_active_disputes']
            self.quantum_memory_bytes = REGISTRY._names_to_collectors['x0tta6bl4_quantum_memory_bytes']
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Сбор метрик со всех компонентов"""
        try:
            # Сбор системных метрик
            cpu_usage = await self._get_cpu_usage()
            memory_usage = await self._get_memory_usage()
            disk_usage = await self._get_disk_usage()

            # Обновление Prometheus метрик
            self.cpu_usage.set(cpu_usage)
            self.memory_usage.set(memory_usage)
            self.disk_usage.set(disk_usage)

            # Сбор метрик компонентов
            components_metrics = await self._get_components_metrics()

            # Обновление Prometheus метрик компонентов
            self.quantum_requests._value.set(components_metrics.get("quantum", {}).get("requests", 0))
            self.ai_requests._value.set(components_metrics.get("ai", {}).get("requests", 0))
            self.enterprise_requests._value.set(components_metrics.get("enterprise", {}).get("requests", 0))
            self.billing_requests._value.set(components_metrics.get("billing", {}).get("requests", 0))

            # Обновление бизнес метрик (примеры значений)
            self.quantum_error_rate.set(2.5)
            self.ai_model_accuracy.set(95.0)
            self.billing_conversion_rate.set(85.0)
            self.enterprise_active_users.set(1250)
            self.billing_active_disputes.set(3)
            self.quantum_memory_bytes.set(150000000)

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage
                },
                "components": components_metrics
            }

            self.metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Ошибка сбора метрик: {e}")
            return {}
    
    async def _get_cpu_usage(self) -> float:
        """Получение использования CPU"""
        try:
            # Используем interval=0 для немедленного возврата, без блокировки
            return psutil.cpu_percent(interval=0)
        except Exception as e:
            logger.error(f"Ошибка получения CPU usage: {e}")
            return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Получение использования памяти"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.error(f"Ошибка получения memory usage: {e}")
            return 0.0
    
    async def _get_disk_usage(self) -> float:
        """Получение использования диска"""
        try:
            disk = psutil.disk_usage('/')
            return disk.percent
        except Exception as e:
            logger.error(f"Ошибка получения disk usage: {e}")
            return 0.0
    
    async def _get_components_metrics(self) -> Dict[str, Any]:
        """Получение метрик компонентов"""
        return {
            "quantum": {"status": "operational", "requests": 150},
            "ai": {"status": "operational", "requests": 89},
            "enterprise": {"status": "operational", "requests": 234},
            "billing": {"status": "operational", "requests": 67}
        }
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Проверка алертов"""
        alerts = []
        
        # Проверка CPU
        if self.metrics.get("system", {}).get("cpu_usage", 0) > 80:
            alerts.append({
                "type": "cpu_high",
                "message": "High CPU usage detected",
                "severity": "warning"
            })
        
        # Проверка памяти
        if self.metrics.get("system", {}).get("memory_usage", 0) > 90:
            alerts.append({
                "type": "memory_high",
                "message": "High memory usage detected",
                "severity": "critical"
            })
        
        self.alerts = alerts
        return alerts
    
    async def generate_report(self) -> Dict[str, Any]:
        """Генерация отчета мониторинга"""
        metrics = await self.collect_metrics()
        alerts = await self.check_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "alerts": alerts,
            "summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
                "system_health": "healthy" if len(alerts) == 0 else "degraded"
            }
        }

    async def get_prometheus_metrics(self) -> str:
        """Получение метрик в формате Prometheus"""
        try:
            # Обновляем метрики перед экспортом
            await self.collect_metrics()
            return generate_latest().decode('utf-8')
        except Exception as e:
            logger.error(f"Ошибка генерации Prometheus метрик: {e}")
            return ""

    async def update_component_metrics(self, component: str, metric_name: str, value: float):
        """Обновление метрик компонентов"""
        try:
            if component == "quantum" and metric_name == "requests":
                self.quantum_requests.inc(value)
            elif component == "ai" and metric_name == "requests":
                self.ai_requests.inc(value)
            elif component == "enterprise" and metric_name == "requests":
                self.enterprise_requests.inc(value)
            elif component == "billing" and metric_name == "requests":
                self.billing_requests.inc(value)
            elif component == "quantum" and metric_name == "error_rate":
                self.quantum_error_rate.set(value)
            elif component == "ai" and metric_name == "accuracy":
                self.ai_model_accuracy.set(value)
            elif component == "billing" and metric_name == "conversion_rate":
                self.billing_conversion_rate.set(value)
            elif component == "enterprise" and metric_name == "active_users":
                self.enterprise_active_users.set(value)
            elif component == "billing" and metric_name == "active_disputes":
                self.billing_active_disputes.set(value)
            elif component == "quantum" and metric_name == "memory_bytes":
                self.quantum_memory_bytes.set(value)
        except Exception as e:
            logger.error(f"Ошибка обновления метрики {component}.{metric_name}: {e}")
