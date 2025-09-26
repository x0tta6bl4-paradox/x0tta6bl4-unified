#!/usr/bin/env python3
"""
Billing Component Metrics for x0tta6bl4
Экспорт метрик billing компонента в Prometheus
"""

import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import asyncio
import logging

logger = logging.getLogger(__name__)

# Метрики для billing компонента
BILLING_ACTIVE_SUBSCRIPTIONS = Gauge('x0tta6bl4_billing_active_subscriptions', 'Number of active subscriptions')
BILLING_PAYMENT_REQUESTS = Counter('x0tta6bl4_billing_payment_requests_total', 'Total payment requests')
BILLING_PAYMENT_PROCESSING_TIME = Histogram('x0tta6bl4_billing_payment_processing_seconds', 'Payment processing time')
BILLING_FAILED_PAYMENTS = Counter('x0tta6bl4_billing_failed_payments_total', 'Failed payment attempts')
BILLING_REVENUE_TOTAL = Counter('x0tta6bl4_billing_revenue_total_usd', 'Total revenue in USD')
BILLING_REFUND_REQUESTS = Counter('x0tta6bl4_billing_refund_requests_total', 'Total refund requests')
BILLING_INVOICE_GENERATION_TIME = Histogram('x0tta6bl4_billing_invoice_generation_seconds', 'Invoice generation time')
BILLING_PAYMENT_GATEWAY_LATENCY = Histogram('x0tta6bl4_billing_gateway_latency_seconds', 'Payment gateway latency')
BILLING_DISPUTE_COUNT = Gauge('x0tta6bl4_billing_active_disputes', 'Number of active payment disputes')
BILLING_CONVERSION_RATE = Gauge('x0tta6bl4_billing_conversion_rate', 'Payment conversion rate percentage')

class BillingMetricsCollector:
    """Сборщик метрик billing компонента"""

    def __init__(self):
        self.active_subscriptions = 1000
        self.conversion_rate = 0.85
        self.active_disputes = 5

    async def collect_metrics(self):
        """Сбор метрик с имитацией реальных данных"""
        try:
            # Имитация активных подписок
            self.active_subscriptions = random.randint(500, 2000)
            BILLING_ACTIVE_SUBSCRIPTIONS.set(self.active_subscriptions)

            # Имитация конверсии платежей
            self.conversion_rate = random.uniform(0.75, 0.95)
            BILLING_CONVERSION_RATE.set(self.conversion_rate * 100)

            # Имитация активных споров
            self.active_disputes = random.randint(0, 20)
            BILLING_DISPUTE_COUNT.set(self.active_disputes)

            # Имитация платежных запросов
            payment_requests = random.randint(10, 100)
            BILLING_PAYMENT_REQUESTS.inc(payment_requests)

            # Имитация неудачных платежей
            failed_payments = random.randint(0, 5)
            BILLING_FAILED_PAYMENTS.inc(failed_payments)

            # Имитация выручки (случайные суммы)
            revenue = random.uniform(100, 10000)
            BILLING_REVENUE_TOTAL.inc(revenue)

            # Имитация запросов на возврат
            if random.random() < 0.05:  # 5% шанс
                refund_requests = random.randint(1, 3)
                BILLING_REFUND_REQUESTS.inc(refund_requests)

            logger.info(f"✅ Собранны метрики billing: {self.active_subscriptions} активных подписок")

        except Exception as e:
            logger.error(f"❌ Ошибка сбора метрик billing: {e}")

    async def simulate_payment_processing(self):
        """Имитация обработки платежа"""
        with BILLING_PAYMENT_PROCESSING_TIME.time():
            processing_time = random.uniform(0.5, 10.0)  # 0.5s - 10s
            await asyncio.sleep(processing_time)

    async def simulate_invoice_generation(self):
        """Имитация генерации счета"""
        with BILLING_INVOICE_GENERATION_TIME.time():
            generation_time = random.uniform(0.1, 2.0)  # 100ms - 2s
            await asyncio.sleep(generation_time)

    async def simulate_gateway_call(self):
        """Имитация вызова платежного шлюза"""
        with BILLING_PAYMENT_GATEWAY_LATENCY.time():
            latency = random.uniform(0.05, 1.0)  # 50ms - 1s
            await asyncio.sleep(latency)

async def main():
    """Главная функция для запуска экспортера метрик billing"""
    logger.info("🚀 Запуск экспортера метрик billing x0tta6bl4")

    collector = BillingMetricsCollector()

    # Запуск HTTP сервера для Prometheus
    start_http_server(8004)
    logger.info("📊 HTTP сервер метрик запущен на порту 8004")

    while True:
        await collector.collect_metrics()
        await collector.simulate_payment_processing()
        await collector.simulate_invoice_generation()
        await collector.simulate_gateway_call()
        await asyncio.sleep(20)  # Сбор метрик каждые 20 секунд

if __name__ == "__main__":
    asyncio.run(main())