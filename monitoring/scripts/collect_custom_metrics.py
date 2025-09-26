#!/usr/bin/env python3
"""
Custom Metrics Collector for x0tta6bl4 Unified Platform
Сбор и экспорт кастомных метрик в Prometheus
"""

import asyncio
import logging
import time
import psutil
import requests
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Системные метрики
SYSTEM_CPU_USAGE = Gauge('x0tta6bl4_system_cpu_usage_percent', 'System CPU usage percentage')
SYSTEM_MEMORY_USAGE = Gauge('x0tta6bl4_system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_DISK_USAGE = Gauge('x0tta6bl4_system_disk_usage_percent', 'System disk usage percentage')
SYSTEM_NETWORK_RX = Counter('x0tta6bl4_system_network_rx_bytes_total', 'Network received bytes total')
SYSTEM_NETWORK_TX = Counter('x0tta6bl4_system_network_tx_bytes_total', 'Network transmitted bytes total')

# Метрики платформы
PLATFORM_UPTIME = Gauge('x0tta6bl4_platform_uptime_seconds', 'Platform uptime in seconds')
PLATFORM_REQUESTS_TOTAL = Counter('x0tta6bl4_platform_requests_total', 'Total platform requests', ['method', 'endpoint'])
PLATFORM_RESPONSE_TIME = Histogram('x0tta6bl4_platform_response_time_seconds', 'Platform response time', ['method', 'endpoint'])
PLATFORM_ACTIVE_CONNECTIONS = Gauge('x0tta6bl4_platform_active_connections', 'Active platform connections')

# Кастомные бизнес метрики
BUSINESS_REVENUE_TOTAL = Counter('x0tta6bl4_business_revenue_total_usd', 'Total business revenue in USD')
BUSINESS_USERS_ACTIVE = Gauge('x0tta6bl4_business_active_users', 'Number of active business users')
BUSINESS_TRANSACTIONS_TOTAL = Counter('x0tta6bl4_business_transactions_total', 'Total business transactions')

class CustomMetricsCollector:
    """Сборщик кастомных метрик"""

    def __init__(self, platform_url="http://localhost:8000"):
        self.platform_url = platform_url
        self.start_time = time.time()
        self.last_net_io = psutil.net_io_counters()

    async def collect_system_metrics(self):
        """Сбор системных метрик"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_DISK_USAGE.set(disk.percent)

            # Network I/O
            net_io = psutil.net_io_counters()
            SYSTEM_NETWORK_RX.inc(net_io.bytes_recv - self.last_net_io.bytes_recv)
            SYSTEM_NETWORK_TX.inc(net_io.bytes_sent - self.last_net_io.bytes_sent)
            self.last_net_io = net_io

            logger.debug(f"✅ Собранны системные метрики: CPU {cpu_percent}%, Memory {memory.percent}%")

        except Exception as e:
            logger.error(f"❌ Ошибка сбора системных метрик: {e}")

    async def collect_platform_metrics(self):
        """Сбор метрик платформы"""
        try:
            # Uptime
            uptime = time.time() - self.start_time
            PLATFORM_UPTIME.set(uptime)

            # Health check
            health_response = requests.get(f"{self.platform_url}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                PLATFORM_ACTIVE_CONNECTIONS.set(len(health_data.get('components', {})))

                # Имитация запросов (в реальности брать из логов/метрик платформы)
                PLATFORM_REQUESTS_TOTAL.labels(method='GET', endpoint='/health').inc()
                PLATFORM_REQUESTS_TOTAL.labels(method='GET', endpoint='/').inc()

            logger.debug("✅ Собранны метрики платформы")

        except Exception as e:
            logger.error(f"❌ Ошибка сбора метрик платформы: {e}")

    async def collect_business_metrics(self):
        """Сбор бизнес метрик"""
        try:
            # Имитация бизнес метрик (в реальности из БД/внешних систем)
            import random

            # Активные пользователи
            active_users = random.randint(100, 1000)
            BUSINESS_USERS_ACTIVE.set(active_users)

            # Транзакции
            transactions = random.randint(1, 10)
            BUSINESS_TRANSACTIONS_TOTAL.inc(transactions)

            # Выручка
            revenue = random.uniform(10, 1000)
            BUSINESS_REVENUE_TOTAL.inc(revenue)

            logger.debug(f"✅ Собранны бизнес метрики: {active_users} активных пользователей")

        except Exception as e:
            logger.error(f"❌ Ошибка сбора бизнес метрик: {e}")

    async def simulate_api_calls(self):
        """Имитация API вызовов для измерения производительности"""
        try:
            endpoints = ['/', '/health', '/api/v1/quantum/status', '/api/v1/ai/status']

            for endpoint in endpoints:
                with PLATFORM_RESPONSE_TIME.labels(method='GET', endpoint=endpoint).time():
                    # Имитация задержки API
                    import random
                    delay = random.uniform(0.01, 0.5)
                    await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"❌ Ошибка имитации API вызовов: {e}")

async def main():
    """Главная функция для запуска сборщика метрик"""
    logger.info("🚀 Запуск сборщика кастомных метрик x0tta6bl4")

    collector = CustomMetricsCollector()

    # Запуск HTTP сервера для Prometheus
    start_http_server(8000)
    logger.info("📊 HTTP сервер метрик запущен на порту 8000")

    while True:
        await collector.collect_system_metrics()
        await collector.collect_platform_metrics()
        await collector.collect_business_metrics()
        await collector.simulate_api_calls()

        await asyncio.sleep(15)  # Сбор метрик каждые 15 секунд

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())