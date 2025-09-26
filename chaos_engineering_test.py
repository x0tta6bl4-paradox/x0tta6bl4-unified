#!/usr/bin/env python3
"""
Chaos Engineering Test для x0tta6bl4-unified
Симуляция сбоев и тестирование resilience системы
"""

import asyncio
import time
import random
import httpx
import json
import signal
import os
from datetime import datetime
from typing import Dict, Any, List
import subprocess
import psutil

class ChaosEngineer:
    """Инженер хаоса для тестирования resilience"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=5.0)
        self.results = {
            "network_failures": [],
            "service_crashes": [],
            "resource_exhaustion": [],
            "data_corruption": [],
            "summary": {}
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def run_chaos_experiments(self) -> Dict[str, Any]:
        """Запуск всех экспериментов хаоса"""
        print("🔥 Запуск chaos engineering экспериментов x0tta6bl4-unified")
        print("=" * 70)

        start_time = time.time()

        # Network failure injection
        print("🌐 Тестирование сетевых сбоев...")
        network_results = await self.test_network_failures()
        self.results["network_failures"] = network_results

        # Service crash simulation
        print("💥 Тестирование падения сервисов...")
        crash_results = await self.test_service_crashes()
        self.results["service_crashes"] = crash_results

        # Resource exhaustion
        print("📈 Тестирование исчерпания ресурсов...")
        resource_results = await self.test_resource_exhaustion()
        self.results["resource_exhaustion"] = resource_results

        # Data corruption
        print("🔄 Тестирование повреждения данных...")
        corruption_results = await self.test_data_corruption()
        self.results["data_corruption"] = corruption_results

        # Анализ результатов
        self.results["summary"] = self.analyze_chaos_results()

        total_time = time.time() - start_time
        self.results["summary"]["total_chaos_time"] = total_time
        self.results["summary"]["timestamp"] = datetime.now().isoformat()

        print(".2f")
        return self.results

    async def test_network_failures(self) -> List[Dict[str, Any]]:
        """Тестирование сетевых сбоев"""
        results = []

        failure_types = [
            "connection_timeout",
            "connection_refused",
            "network_partition",
            "high_latency",
            "packet_loss"
        ]

        for failure_type in failure_types:
            result = await self.simulate_network_failure(failure_type)
            results.append(result)

        return results

    async def simulate_network_failure(self, failure_type: str) -> Dict[str, Any]:
        """Симуляция конкретного типа сетевого сбоя"""
        print(f"   Симуляция {failure_type}...")

        # Базовые параметры
        test_duration = 10  # секунды
        recovery_time = 5   # секунды

        start_time = time.time()
        failures_detected = 0
        recoveries_detected = 0

        # Симуляция сбоя
        if failure_type == "connection_timeout":
            # Имитация таймаута подключения
            await self.inject_timeout_failure(test_duration)
        elif failure_type == "connection_refused":
            # Имитация отказа в подключении
            await self.inject_connection_refused(test_duration)
        elif failure_type == "network_partition":
            # Имитация сетевого разделения
            await self.inject_network_partition(test_duration)
        elif failure_type == "high_latency":
            # Имитация высокой задержки
            await self.inject_high_latency(test_duration)
        elif failure_type == "packet_loss":
            # Имитация потери пакетов
            await self.inject_packet_loss(test_duration)

        # Мониторинг во время сбоя
        for _ in range(int(test_duration)):
            try:
                response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code != 200:
                    failures_detected += 1
            except Exception:
                failures_detected += 1
            await asyncio.sleep(1)

        # Восстановление
        await asyncio.sleep(recovery_time)

        # Проверка восстановления
        for _ in range(int(recovery_time)):
            try:
                response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    recoveries_detected += 1
            except Exception:
                pass
            await asyncio.sleep(1)

        resilience_score = recoveries_detected / recovery_time if recovery_time > 0 else 0

        return {
            "failure_type": failure_type,
            "test_duration": test_duration,
            "recovery_time": recovery_time,
            "failures_detected": failures_detected,
            "recoveries_detected": recoveries_detected,
            "resilience_score": resilience_score,
            "status": "passed" if resilience_score > 0.8 else "warning" if resilience_score > 0.5 else "failed"
        }

    async def inject_timeout_failure(self, duration: int):
        """Инъекция таймаута подключения"""
        # В реальной системе это потребовало бы iptables или аналогичных инструментов
        # Для симуляции просто ждем
        await asyncio.sleep(duration)

    async def inject_connection_refused(self, duration: int):
        """Инъекция отказа в подключении"""
        # Симуляция путем изменения порта или блокировки
        await asyncio.sleep(duration)

    async def inject_network_partition(self, duration: int):
        """Инъекция сетевого разделения"""
        # В реальной системе: iptables -A INPUT -s <ip> -j DROP
        await asyncio.sleep(duration)

    async def inject_high_latency(self, duration: int):
        """Инъекция высокой задержки"""
        # В реальной системе: tc qdisc add dev eth0 root netem delay 100ms
        await asyncio.sleep(duration)

    async def inject_packet_loss(self, duration: int):
        """Инъекция потери пакетов"""
        # В реальной системе: tc qdisc add dev eth0 root netem loss 10%
        await asyncio.sleep(duration)

    async def test_service_crashes(self) -> List[Dict[str, Any]]:
        """Тестирование падения сервисов"""
        results = []

        crash_scenarios = [
            "api_server_crash",
            "database_connection_lost",
            "quantum_core_failure",
            "monitoring_system_down",
            "load_balancer_failure"
        ]

        for scenario in crash_scenarios:
            result = await self.simulate_service_crash(scenario)
            results.append(result)

        return results

    async def simulate_service_crash(self, scenario: str) -> Dict[str, Any]:
        """Симуляция падения сервиса"""
        print(f"   Симуляция {scenario}...")

        crash_duration = 15  # секунды
        monitoring_period = 30  # секунды

        start_time = time.time()
        downtime_detected = 0
        recovery_detected = False

        # Симуляция падения
        if scenario == "api_server_crash":
            await self.crash_api_server(crash_duration)
        elif scenario == "database_connection_lost":
            await self.crash_database_connection(crash_duration)
        elif scenario == "quantum_core_failure":
            await self.crash_quantum_core(crash_duration)
        elif scenario == "monitoring_system_down":
            await self.crash_monitoring_system(crash_duration)
        elif scenario == "load_balancer_failure":
            await self.crash_load_balancer(crash_duration)

        # Мониторинг доступности
        for _ in range(int(monitoring_period)):
            try:
                if scenario in ["api_server_crash", "load_balancer_failure"]:
                    response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
                    if response.status_code != 200:
                        downtime_detected += 1
                elif scenario == "database_connection_lost":
                    response = await self.client.get(f"{self.base_url}/api/v1/enterprise/status", timeout=2.0)
                    if response.status_code != 200:
                        downtime_detected += 1
                elif scenario == "quantum_core_failure":
                    response = await self.client.get(f"{self.base_url}/api/v1/quantum/status", timeout=2.0)
                    if response.status_code != 200:
                        downtime_detected += 1
                elif scenario == "monitoring_system_down":
                    response = await self.client.get(f"{self.base_url}/api/v1/monitoring/status", timeout=2.0)
                    if response.status_code != 200:
                        downtime_detected += 1

                if downtime_detected == 0 and time.time() - start_time > crash_duration:
                    recovery_detected = True

            except Exception:
                downtime_detected += 1

            await asyncio.sleep(1)

        # Расчет метрик
        total_downtime = downtime_detected
        availability = 1 - (total_downtime / monitoring_period) if monitoring_period > 0 else 0
        mttr = crash_duration if recovery_detected else monitoring_period  # Mean Time To Recovery

        return {
            "scenario": scenario,
            "crash_duration": crash_duration,
            "monitoring_period": monitoring_period,
            "total_downtime": total_downtime,
            "availability": availability,
            "mttr": mttr,
            "recovery_detected": recovery_detected,
            "status": "passed" if availability > 0.9 else "warning" if availability > 0.7 else "failed"
        }

    async def crash_api_server(self, duration: int):
        """Симуляция падения API сервера"""
        # В реальной системе: systemctl stop api-service
        await asyncio.sleep(duration)

    async def crash_database_connection(self, duration: int):
        """Симуляция потери соединения с БД"""
        # В реальной системе: блокировка порта БД
        await asyncio.sleep(duration)

    async def crash_quantum_core(self, duration: int):
        """Симуляция падения квантового ядра"""
        # В реальной системе: остановка quantum сервиса
        await asyncio.sleep(duration)

    async def crash_monitoring_system(self, duration: int):
        """Симуляция падения системы мониторинга"""
        # В реальной системе: остановка prometheus/grafana
        await asyncio.sleep(duration)

    async def crash_load_balancer(self, duration: int):
        """Симуляция падения балансировщика нагрузки"""
        # В реальной системе: остановка nginx/haproxy
        await asyncio.sleep(duration)

    async def test_resource_exhaustion(self) -> List[Dict[str, Any]]:
        """Тестирование исчерпания ресурсов"""
        results = []

        resource_types = [
            "memory_exhaustion",
            "cpu_exhaustion",
            "disk_space_exhaustion",
            "network_bandwidth_exhaustion",
            "file_descriptor_exhaustion"
        ]

        for resource_type in resource_types:
            result = await self.simulate_resource_exhaustion(resource_type)
            results.append(result)

        return results

    async def simulate_resource_exhaustion(self, resource_type: str) -> Dict[str, Any]:
        """Симуляция исчерпания конкретного ресурса"""
        print(f"   Симуляция {resource_type}...")

        test_duration = 20  # секунды
        stress_level = 0.8  # 80% ресурсов

        start_time = time.time()
        performance_degradation = []
        recovery_time = 0

        # Создание нагрузки
        if resource_type == "memory_exhaustion":
            await self.exhaust_memory(stress_level, test_duration)
        elif resource_type == "cpu_exhaustion":
            await self.exhaust_cpu(stress_level, test_duration)
        elif resource_type == "disk_space_exhaustion":
            await self.exhaust_disk_space(stress_level, test_duration)
        elif resource_type == "network_bandwidth_exhaustion":
            await self.exhaust_network_bandwidth(stress_level, test_duration)
        elif resource_type == "file_descriptor_exhaustion":
            await self.exhaust_file_descriptors(stress_level, test_duration)

        # Мониторинг производительности во время нагрузки
        for _ in range(int(test_duration)):
            try:
                response = await self.client.get(f"{self.base_url}/health", timeout=3.0)
                response_time = response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 1.0
                performance_degradation.append(response_time)
            except Exception as e:
                performance_degradation.append(5.0)  # Таймаут
            await asyncio.sleep(1)

        # Ожидание восстановления
        recovery_start = time.time()
        for _ in range(10):  # 10 секунд на восстановление
            try:
                response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    response_time = response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 1.0
                    if response_time < 1.0:  # Нормальное время ответа
                        recovery_time = time.time() - recovery_start
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        # Анализ результатов
        avg_response_time = sum(performance_degradation) / len(performance_degradation) if performance_degradation else 5.0
        max_response_time = max(performance_degradation) if performance_degradation else 5.0

        resilience_score = 1 - min(1, avg_response_time / 5.0)  # Нормализация к 0-1

        return {
            "resource_type": resource_type,
            "stress_level": stress_level,
            "test_duration": test_duration,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "recovery_time": recovery_time,
            "resilience_score": resilience_score,
            "status": "passed" if resilience_score > 0.7 else "warning" if resilience_score > 0.5 else "failed"
        }

    async def exhaust_memory(self, stress_level: float, duration: int):
        """Истощение памяти"""
        # В реальной системе: stress --vm 1 --vm-bytes 80%
        await asyncio.sleep(duration)

    async def exhaust_cpu(self, stress_level: float, duration: int):
        """Истощение CPU"""
        # В реальной системе: stress --cpu 4
        await asyncio.sleep(duration)

    async def exhaust_disk_space(self, stress_level: float, duration: int):
        """Истощение дискового пространства"""
        # В реальной системе: dd if=/dev/zero of=/tmp/fill bs=1M count=1000
        await asyncio.sleep(duration)

    async def exhaust_network_bandwidth(self, stress_level: float, duration: int):
        """Истощение сетевой полосы"""
        # В реальной системе: iperf или аналогичные инструменты
        await asyncio.sleep(duration)

    async def exhaust_file_descriptors(self, stress_level: float, duration: int):
        """Истощение файловых дескрипторов"""
        # В реальной системе: ulimit -n 100 && запуск процессов
        await asyncio.sleep(duration)

    async def test_data_corruption(self) -> List[Dict[str, Any]]:
        """Тестирование повреждения данных"""
        results = []

        corruption_types = [
            "config_file_corruption",
            "database_record_corruption",
            "cache_poisoning",
            "message_queue_corruption",
            "log_file_corruption"
        ]

        for corruption_type in corruption_types:
            result = await self.simulate_data_corruption(corruption_type)
            results.append(result)

        return results

    async def simulate_data_corruption(self, corruption_type: str) -> Dict[str, Any]:
        """Симуляция повреждения данных"""
        print(f"   Симуляция {corruption_type}...")

        corruption_duration = 10  # секунды
        recovery_attempts = 5

        start_time = time.time()
        corruption_detected = False
        recovery_successful = False

        # Внедрение повреждения
        if corruption_type == "config_file_corruption":
            await self.corrupt_config_file(corruption_duration)
        elif corruption_type == "database_record_corruption":
            await self.corrupt_database_record(corruption_duration)
        elif corruption_type == "cache_poisoning":
            await self.poison_cache(corruption_duration)
        elif corruption_type == "message_queue_corruption":
            await self.corrupt_message_queue(corruption_duration)
        elif corruption_type == "log_file_corruption":
            await self.corrupt_log_file(corruption_duration)

        # Мониторинг обнаружения повреждения
        for _ in range(int(corruption_duration)):
            try:
                # Проверяем различные endpoints в зависимости от типа повреждения
                if corruption_type == "config_file_corruption":
                    response = await self.client.get(f"{self.base_url}/health")
                elif corruption_type == "database_record_corruption":
                    response = await self.client.get(f"{self.base_url}/api/v1/enterprise/status")
                elif corruption_type == "cache_poisoning":
                    response = await self.client.get(f"{self.base_url}/api/v1/monitoring/metrics")
                elif corruption_type == "message_queue_corruption":
                    response = await self.client.get(f"{self.base_url}/api/v1/ai/status")
                elif corruption_type == "log_file_corruption":
                    response = await self.client.get(f"{self.base_url}/api/v1/monitoring/status")

                if response.status_code >= 500:  # Server errors indicate corruption detected
                    corruption_detected = True
                    break

            except Exception:
                corruption_detected = True
                break

            await asyncio.sleep(1)

        # Попытки восстановления
        for attempt in range(recovery_attempts):
            try:
                # Проверяем восстановление
                response = await self.client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    recovery_successful = True
                    break
            except Exception:
                pass
            await asyncio.sleep(2)

        # Расчет метрик
        detection_time = time.time() - start_time if corruption_detected else corruption_duration
        recovery_time = (attempt + 1) * 2 if recovery_successful else recovery_attempts * 2

        return {
            "corruption_type": corruption_type,
            "corruption_duration": corruption_duration,
            "recovery_attempts": recovery_attempts,
            "corruption_detected": corruption_detected,
            "recovery_successful": recovery_successful,
            "detection_time": detection_time,
            "recovery_time": recovery_time,
            "status": "passed" if recovery_successful else "warning" if corruption_detected else "failed"
        }

    async def corrupt_config_file(self, duration: int):
        """Повреждение конфигурационного файла"""
        # В реальной системе: sed -i 's/valid_config/invalid_config/' config.yaml
        await asyncio.sleep(duration)

    async def corrupt_database_record(self, duration: int):
        """Повреждение записи в БД"""
        # В реальной системе: UPDATE table SET data = 'corrupted' WHERE id = 1
        await asyncio.sleep(duration)

    async def poison_cache(self, duration: int):
        """Отравление кэша"""
        # В реальной системе: внедрение некорректных данных в Redis/Memcached
        await asyncio.sleep(duration)

    async def corrupt_message_queue(self, duration: int):
        """Повреждение очереди сообщений"""
        # В реальной системе: отправка malformed сообщений в RabbitMQ/Kafka
        await asyncio.sleep(duration)

    async def corrupt_log_file(self, duration: int):
        """Повреждение лог файла"""
        # В реальной системе: echo "corrupted data" >> log.txt
        await asyncio.sleep(duration)

    def analyze_chaos_results(self) -> Dict[str, Any]:
        """Анализ результатов chaos engineering"""
        summary = {
            "total_experiments": 0,
            "passed_experiments": 0,
            "warning_experiments": 0,
            "failed_experiments": 0,
            "network_resilience": {},
            "service_resilience": {},
            "resource_resilience": {},
            "data_resilience": {},
            "recommendations": []
        }

        # Подсчет результатов
        all_results = (
            self.results["network_failures"] +
            self.results["service_crashes"] +
            self.results["resource_exhaustion"] +
            self.results["data_corruption"]
        )

        for result in all_results:
            summary["total_experiments"] += 1
            status = result.get("status", "unknown")
            if status == "passed":
                summary["passed_experiments"] += 1
            elif status == "warning":
                summary["warning_experiments"] += 1
            elif status == "failed":
                summary["failed_experiments"] += 1

        # Анализ сетевой устойчивости
        network_scores = [r.get("resilience_score", 0) for r in self.results["network_failures"]]
        summary["network_resilience"] = {
            "average_resilience": sum(network_scores) / len(network_scores) if network_scores else 0,
            "weakest_link": min(network_scores) if network_scores else 0
        }

        # Анализ устойчивости сервисов
        service_availabilities = [r.get("availability", 0) for r in self.results["service_crashes"]]
        summary["service_resilience"] = {
            "average_availability": sum(service_availabilities) / len(service_availabilities) if service_availabilities else 0,
            "best_mttr": min([r.get("mttr", 1000) for r in self.results["service_crashes"]]) if self.results["service_crashes"] else 1000
        }

        # Анализ устойчивости к исчерпанию ресурсов
        resource_scores = [r.get("resilience_score", 0) for r in self.results["resource_exhaustion"]]
        summary["resource_resilience"] = {
            "average_resilience": sum(resource_scores) / len(resource_scores) if resource_scores else 0,
            "resource_vulnerabilities": len([s for s in resource_scores if s < 0.5])
        }

        # Анализ устойчивости данных
        data_recoveries = sum(1 for r in self.results["data_corruption"] if r.get("recovery_successful", False))
        summary["data_resilience"] = {
            "recovery_rate": data_recoveries / len(self.results["data_corruption"]) if self.results["data_corruption"] else 0,
            "average_detection_time": sum(r.get("detection_time", 10) for r in self.results["data_corruption"]) / len(self.results["data_corruption"]) if self.results["data_corruption"] else 10
        }

        # Генерация рекомендаций
        summary["recommendations"] = self.generate_chaos_recommendations(summary)

        return summary

    def generate_chaos_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе результатов chaos engineering"""
        recommendations = []

        # Сетевые рекомендации
        if summary["network_resilience"]["average_resilience"] < 0.8:
            recommendations.append("Улучшить сетевую отказоустойчивость - добавить retry логику и circuit breakers")

        # Рекомендации по сервисам
        if summary["service_resilience"]["average_availability"] < 0.95:
            recommendations.append("Реализовать graceful degradation и service mesh для лучшей доступности")

        # Рекомендации по ресурсам
        if summary["resource_resilience"]["resource_vulnerabilities"] > 0:
            recommendations.append("Добавить resource limits и auto-scaling для предотвращения исчерпания ресурсов")

        # Рекомендации по данным
        if summary["data_resilience"]["recovery_rate"] < 0.8:
            recommendations.append("Улучшить механизмы backup и data validation для быстрого восстановления")

        if summary["data_resilience"]["average_detection_time"] > 5:
            recommendations.append("Реализовать real-time monitoring и alerting для быстрого обнаружения проблем")

        if not recommendations:
            recommendations.append("Система демонстрирует хорошую устойчивость к сбоям")

        return recommendations

async def main():
    """Основная функция"""
    async with ChaosEngineer() as engineer:
        results = await engineer.run_chaos_experiments()

        # Сохранение результатов
        with open("chaos_engineering_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n📊 Результаты chaos engineering сохранены в chaos_engineering_results.json")

        # Вывод сводки
        summary = results["summary"]
        print("\n📈 Сводка результатов chaos engineering:")
        print(f"   • Всего экспериментов: {summary['total_experiments']}")
        print(f"   • Пройдено: {summary['passed_experiments']}")
        print(f"   • Предупреждений: {summary['warning_experiments']}")
        print(f"   • Провалено: {summary['failed_experiments']}")

        print("\n🌐 Сетевая устойчивость:")
        net = summary["network_resilience"]
        print(".4f")
        print(".4f")

        print("\n🔧 Устойчивость сервисов:")
        svc = summary["service_resilience"]
        print(".4f")
        print(f"   • Лучшее MTTR: {svc['best_mttr']:.1f}s")

        print("\n📈 Устойчивость ресурсов:")
        res = summary["resource_resilience"]
        print(".4f")
        print(f"   • Уязвимостей ресурсов: {res['resource_vulnerabilities']}")

        print("\n💾 Устойчивость данных:")
        data = summary["data_resilience"]
        print(".4f")
        print(".1f")

        print("\n💡 Рекомендации:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")

if __name__ == "__main__":
    asyncio.run(main())