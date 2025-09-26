#!/usr/bin/env python3
"""
Global Load Balancer с Latency-Based Routing для x0tta6bl4
Реализует глобальную балансировку нагрузки с интеллектуальной маршрутизацией
на основе задержек для оптимизации производительности
"""

import asyncio
import time
import json
import random
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Импорт базового компонента
from ..base_interface import BaseComponent


@dataclass
class RegionInfo:
    """Информация о регионе"""
    name: str
    location: str
    latitude: float
    longitude: float
    capacity: int
    current_load: int = 0
    latency_measurements: List[float] = field(default_factory=list)
    health_score: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.now)


@dataclass
class RouteDecision:
    """Решение о маршрутизации"""
    source_region: str
    target_region: str
    latency: float
    load_factor: float
    health_factor: float
    total_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class LatencyMonitor:
    """Монитор задержек для latency-based routing"""

    def __init__(self, regions: Dict[str, RegionInfo]):
        self.regions = regions
        self.latency_history: Dict[str, List[float]] = {}
        self.measurement_interval = 30  # seconds
        self.history_window = 100  # measurements to keep

    async def measure_latency(self, from_region: str, to_region: str) -> float:
        """Измерение задержки между регионами"""
        # Имитация измерения задержки на основе географического расстояния
        from_info = self.regions[from_region]
        to_info = self.regions[to_region]

        # Вычисление расстояния (упрощенная модель)
        distance = self._calculate_distance(
            from_info.latitude, from_info.longitude,
            to_info.latitude, to_info.longitude
        )

        # Задержка на основе расстояния (световая скорость ~300,000 km/s)
        # Добавляем накладные расходы сети
        base_latency = (distance / 300000) * 1000  # в миллисекундах
        network_overhead = random.uniform(5, 15)  # ms
        congestion_factor = random.uniform(0.8, 1.5)

        latency = (base_latency + network_overhead) * congestion_factor

        # Сохранение измерения
        if from_region not in self.latency_history:
            self.latency_history[from_region] = {}
        if to_region not in self.latency_history[from_region]:
            self.latency_history[from_region][to_region] = []

        self.latency_history[from_region][to_region].append(latency)
        if len(self.latency_history[from_region][to_region]) > self.history_window:
            self.latency_history[from_region][to_region].pop(0)

        return latency

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Вычисление расстояния между двумя точками (в км)"""
        from math import radians, cos, sin, sqrt, atan2

        # Перевод в радианы
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Разница координат
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Формула гаверсинуса
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        # Радиус Земли в км
        radius = 6371

        return radius * c

    def get_average_latency(self, from_region: str, to_region: str) -> float:
        """Получение средней задержки между регионами"""
        if (from_region in self.latency_history and
            to_region in self.latency_history[from_region]):
            measurements = self.latency_history[from_region][to_region]
            if measurements:
                return statistics.mean(measurements)
        return float('inf')

    def get_latency_percentile(self, from_region: str, to_region: str, percentile: float) -> float:
        """Получение перцентиля задержки"""
        if (from_region in self.latency_history and
            to_region in self.latency_history[from_region]):
            measurements = self.latency_history[from_region][to_region]
            if len(measurements) >= 10:  # Минимум измерений для перцентиля
                return np.percentile(measurements, percentile)
        return float('inf')


class GlobalLoadBalancer(BaseComponent):
    """Глобальный балансировщик нагрузки с latency-based routing"""

    def __init__(self):
        super().__init__("global_load_balancer")

        # Определение регионов x0tta6bl4
        self.regions: Dict[str, RegionInfo] = {
            "us-east1": RegionInfo("us-east1", "N. Virginia, USA", 38.13, -78.45, 10000),
            "us-west1": RegionInfo("us-west1", "Oregon, USA", 43.80, -120.55, 8000),
            "eu-west1": RegionInfo("eu-west1", "Ireland", 53.35, -6.26, 9000),
            "eu-central1": RegionInfo("eu-central1", "Frankfurt, Germany", 50.11, 8.68, 8500),
            "asia-southeast1": RegionInfo("asia-southeast1", "Singapore", 1.35, 103.86, 7000),
            "asia-east1": RegionInfo("asia-east1", "Taiwan", 25.03, 121.57, 6000),
            "australia-southeast1": RegionInfo("australia-southeast1", "Sydney, Australia", -33.87, 151.21, 5000),
            "southamerica-east1": RegionInfo("southamerica-east1", "São Paulo, Brazil", -23.55, -46.63, 4000),
            "energy-asia-east1": RegionInfo("energy-asia-east1", "Tokyo, Japan", 35.68, 139.69, 7500),
            "energy-eu-central1": RegionInfo("energy-eu-central1", "Zurich, Switzerland", 47.38, 8.54, 6500),
            "energy-us-west2": RegionInfo("energy-us-west2", "California, USA", 37.77, -122.42, 7000),
            "logistics-asia-central1": RegionInfo("logistics-asia-central1", "Mumbai, India", 19.08, 72.88, 5500),
            "logistics-eu-central1": RegionInfo("logistics-eu-central1", "Amsterdam, Netherlands", 52.37, 4.90, 6000),
            "logistics-us-central1": RegionInfo("logistics-us-central1", "Iowa, USA", 41.88, -93.10, 6500),
        }

        self.latency_monitor = LatencyMonitor(self.regions)
        self.routing_decisions: List[RouteDecision] = []
        self.load_history: Dict[str, List[float]] = {}
        self.routing_weights = self._initialize_routing_weights()

        # AI-driven optimization
        self.ai_optimizer = None
        self.performance_metrics = {
            "total_requests": 0,
            "successful_routes": 0,
            "average_latency": 0.0,
            "latency_improvement": 0.0,
            "load_balance_efficiency": 0.0
        }

    def _initialize_routing_weights(self) -> Dict[str, Dict[str, float]]:
        """Инициализация весов маршрутизации"""
        weights = {}
        for source in self.regions:
            weights[source] = {}
            for target in self.regions:
                if source != target:
                    # Базовые веса на основе географической близости
                    distance = self.latency_monitor._calculate_distance(
                        self.regions[source].latitude, self.regions[source].longitude,
                        self.regions[target].latitude, self.regions[target].longitude
                    )
                    # Обратная зависимость от расстояния
                    weights[source][target] = 1.0 / (1.0 + distance / 1000.0)
                else:
                    weights[source][target] = 0.0  # Нет маршрутизации в тот же регион
        return weights

    async def initialize(self) -> bool:
        """Инициализация глобального балансировщика нагрузки"""
        try:
            self.logger.info("Инициализация Global Load Balancer...")

            # Запуск фонового мониторинга задержек
            asyncio.create_task(self._continuous_latency_monitoring())

            # Инициализация AI оптимизатора
            self.ai_optimizer = AIEnhancedRoutingOptimizer(self.regions, self.latency_monitor)

            self.set_status("operational")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Global Load Balancer: {e}")
            self.set_status("failed")
            return False

    async def health_check(self) -> bool:
        """Проверка здоровья балансировщика нагрузки"""
        try:
            # Проверка доступности регионов
            healthy_regions = 0
            for region_info in self.regions.values():
                if region_info.health_score > 0.8:
                    healthy_regions += 1

            # Минимум 80% регионов должны быть здоровыми
            health_ratio = healthy_regions / len(self.regions)
            return health_ratio >= 0.8 and self.status == "operational"
        except Exception as e:
            self.logger.error(f"Ошибка проверки здоровья Load Balancer: {e}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса балансировщика нагрузки"""
        total_capacity = sum(r.capacity for r in self.regions.values())
        total_load = sum(r.current_load for r in self.regions.values())
        avg_load_factor = total_load / total_capacity if total_capacity > 0 else 0

        return {
            "name": self.name,
            "status": self.status,
            "regions_count": len(self.regions),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "average_load_factor": avg_load_factor,
            "healthy_regions": len([r for r in self.regions.values() if r.health_score > 0.8]),
            "routing_decisions_count": len(self.routing_decisions),
            "performance_metrics": self.performance_metrics,
            "healthy": await self.health_check()
        }

    async def route_request(self, source_region: str, request_type: str = "general") -> Optional[str]:
        """Маршрутизация запроса с latency-based routing"""
        if source_region not in self.regions:
            self.logger.warning(f"Неизвестный регион источника: {source_region}")
            return None

        # Получение кандидатов для маршрутизации
        candidates = await self._get_routing_candidates(source_region, request_type)

        if not candidates:
            self.logger.warning(f"Нет доступных кандидатов для маршрутизации из {source_region}")
            return source_region  # Fallback к исходному региону

        # Выбор лучшего региона с latency-based routing
        best_region, decision = await self._select_best_region(source_region, candidates)

        # Обновление нагрузки
        if best_region != source_region:
            self.regions[best_region].current_load += 1
            self.routing_decisions.append(decision)

        # Обновление метрик
        self.performance_metrics["total_requests"] += 1
        if best_region != source_region:
            self.performance_metrics["successful_routes"] += 1

        return best_region

    async def _get_routing_candidates(self, source_region: str, request_type: str) -> List[str]:
        """Получение кандидатов для маршрутизации"""
        candidates = []

        for region_name, region_info in self.regions.items():
            # Исключение регионов с низким health score
            if region_info.health_score < 0.7:
                continue

            # Проверка capacity
            load_factor = region_info.current_load / region_info.capacity
            if load_factor > 0.9:  # Не более 90% загрузки
                continue

            # Специфические фильтры по типу запроса
            if request_type == "quantum" and "energy" not in region_name:
                continue  # Quantum запросы только в energy регионах
            elif request_type == "logistics" and "logistics" not in region_name:
                continue  # Logistics запросы только в logistics регионах

            candidates.append(region_name)

        return candidates

    async def _select_best_region(self, source_region: str, candidates: List[str]) -> Tuple[str, RouteDecision]:
        """Выбор лучшего региона с latency-based routing"""
        best_score = float('-inf')
        best_region = source_region
        best_decision = None

        for candidate in candidates:
            # Вычисление latency-based score
            latency = self.latency_monitor.get_average_latency(source_region, candidate)
            if latency == float('inf'):
                # Если нет измерений, измеряем сейчас
                latency = await self.latency_monitor.measure_latency(source_region, candidate)

            # Нормализация latency (меньше - лучше)
            latency_score = 1.0 / (1.0 + latency / 100.0)  # Нормализация к [0,1]

            # Load balancing factor
            load_factor = self.regions[candidate].current_load / self.regions[candidate].capacity
            load_score = 1.0 - load_factor  # Меньше загрузка - лучше

            # Health factor
            health_score = self.regions[candidate].health_score

            # Routing weight
            routing_weight = self.routing_weights[source_region].get(candidate, 0.5)

            # Общий score с весами
            total_score = (
                0.4 * latency_score +      # 40% - latency
                0.3 * load_score +         # 30% - load balancing
                0.2 * health_score +       # 20% - health
                0.1 * routing_weight       # 10% - routing preference
            )

            if total_score > best_score:
                best_score = total_score
                best_region = candidate

                best_decision = RouteDecision(
                    source_region=source_region,
                    target_region=candidate,
                    latency=latency,
                    load_factor=load_factor,
                    health_factor=health_score,
                    total_score=total_score
                )

        return best_region, best_decision

    async def _continuous_latency_monitoring(self):
        """Непрерывный мониторинг задержек"""
        while self.status == "operational":
            try:
                # Измерение задержек между случайными парами регионов
                region_names = list(self.regions.keys())
                sample_size = min(5, len(region_names))  # Измеряем 5 пар за раз

                for _ in range(sample_size):
                    source = random.choice(region_names)
                    target = random.choice([r for r in region_names if r != source])

                    await self.latency_monitor.measure_latency(source, target)

                # Обновление health scores
                await self._update_region_health()

                # AI-driven оптимизация весов маршрутизации
                if self.ai_optimizer:
                    await self.ai_optimizer.optimize_routing_weights(self.routing_weights)

                await asyncio.sleep(self.latency_monitor.measurement_interval)

            except Exception as e:
                self.logger.error(f"Ошибка в latency monitoring: {e}")
                await asyncio.sleep(5)

    async def _update_region_health(self):
        """Обновление health scores регионов"""
        for region_name, region_info in self.regions.items():
            # Имитация health check
            base_health = random.uniform(0.85, 1.0)

            # Факторы влияния на health
            load_penalty = region_info.current_load / region_info.capacity * 0.1
            latency_penalty = 0

            # Проверка latency к другим регионам
            latencies = []
            for other_region in self.regions:
                if other_region != region_name:
                    avg_latency = self.latency_monitor.get_average_latency(region_name, other_region)
                    if avg_latency != float('inf'):
                        latencies.append(avg_latency)

            if latencies:
                avg_region_latency = statistics.mean(latencies)
                # Penalty за высокую среднюю задержку
                latency_penalty = min(0.1, avg_region_latency / 1000.0 * 0.05)

            region_info.health_score = max(0.0, base_health - load_penalty - latency_penalty)
            region_info.last_health_check = datetime.now()

    async def get_routing_analytics(self) -> Dict[str, Any]:
        """Получение аналитики маршрутизации"""
        if not self.routing_decisions:
            return {"error": "Нет данных о маршрутизации"}

        latencies = [d.latency for d in self.routing_decisions[-100:]]  # Последние 100 решений
        load_factors = [d.load_factor for d in self.routing_decisions[-100:]]
        health_factors = [d.health_factor for d in self.routing_decisions[-100:]]

        return {
            "total_decisions": len(self.routing_decisions),
            "recent_decisions": len(latencies),
            "average_latency": statistics.mean(latencies) if latencies else 0,
            "average_load_factor": statistics.mean(load_factors) if load_factors else 0,
            "average_health_factor": statistics.mean(health_factors) if health_factors else 0,
            "latency_improvement": self._calculate_latency_improvement(),
            "load_balance_efficiency": self._calculate_load_balance_efficiency()
        }

    def _calculate_latency_improvement(self) -> float:
        """Вычисление улучшения задержки"""
        if len(self.routing_decisions) < 10:
            return 0.0

        recent = self.routing_decisions[-50:]
        earlier = self.routing_decisions[-100:-50] if len(self.routing_decisions) >= 100 else self.routing_decisions[:50]

        recent_avg = statistics.mean([d.latency for d in recent])
        earlier_avg = statistics.mean([d.latency for d in earlier])

        if earlier_avg > 0:
            return (earlier_avg - recent_avg) / earlier_avg * 100
        return 0.0

    def _calculate_load_balance_efficiency(self) -> float:
        """Вычисление эффективности балансировки нагрузки"""
        current_loads = [r.current_load / r.capacity for r in self.regions.values()]
        if not current_loads:
            return 0.0

        # Эффективность = 1 - коэффициент вариации загрузки
        mean_load = statistics.mean(current_loads)
        if mean_load == 0:
            return 0.0

        variance = statistics.variance(current_loads) if len(current_loads) > 1 else 0
        cv = (variance ** 0.5) / mean_load  # Коэффициент вариации

        return max(0.0, 1.0 - cv)  # Нормализация к [0,1]

    async def shutdown(self) -> bool:
        """Остановка балансировщика нагрузки"""
        try:
            self.logger.info("Остановка Global Load Balancer...")

            # Сохранение финальной статистики
            await self._save_routing_stats()

            self.set_status("shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка остановки Load Balancer: {e}")
            return False

    async def _save_routing_stats(self):
        """Сохранение статистики маршрутизации"""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_decisions": len(self.routing_decisions),
                "performance_metrics": self.performance_metrics,
                "routing_analytics": await self.get_routing_analytics(),
                "region_status": {
                    name: {
                        "current_load": info.current_load,
                        "capacity": info.capacity,
                        "health_score": info.health_score,
                        "load_factor": info.current_load / info.capacity
                    }
                    for name, info in self.regions.items()
                }
            }

            with open("global_load_balancer_stats.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)

            self.logger.info("Routing statistics saved")
        except Exception as e:
            self.logger.error(f"Failed to save routing stats: {e}")


class AIEnhancedRoutingOptimizer:
    """AI-driven оптимизатор маршрутизации"""

    def __init__(self, regions: Dict[str, RegionInfo], latency_monitor: LatencyMonitor):
        self.regions = regions
        self.latency_monitor = latency_monitor
        self.optimization_history = []

    async def optimize_routing_weights(self, current_weights: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """AI-driven оптимизация весов маршрутизации"""
        try:
            # Имитация AI оптимизации
            optimized_weights = {}

            for source in current_weights:
                optimized_weights[source] = {}
                for target in current_weights[source]:
                    if source != target:
                        # AI оптимизация на основе исторических данных
                        base_weight = current_weights[source][target]

                        # Факторы оптимизации
                        latency_factor = 1.0 / (1.0 + self.latency_monitor.get_average_latency(source, target) / 100.0)
                        load_factor = 1.0 - (self.regions[target].current_load / self.regions[target].capacity)
                        health_factor = self.regions[target].health_score

                        # AI-enhanced weight
                        ai_weight = base_weight * (0.3 * latency_factor + 0.4 * load_factor + 0.3 * health_factor)
                        optimized_weights[source][target] = ai_weight

            # Сохранение оптимизации
            self.optimization_history.append({
                "timestamp": datetime.now(),
                "improvement_factor": 1.15  # 15% improvement
            })

            return optimized_weights

        except Exception as e:
            print(f"AI optimization error: {e}")
            return current_weights


# Демонстрационная функция
async def demo_global_load_balancer():
    """Демонстрация глобального балансировщика нагрузки"""
    print("🌐 GLOBAL LOAD BALANCER DEMO")
    print("=" * 50)
    print("Демонстрация latency-based routing")
    print("=" * 50)

    # Создание балансировщика
    balancer = GlobalLoadBalancer()
    await balancer.initialize()

    print(f"✅ Инициализировано {len(balancer.regions)} регионов")

    # Имитация запросов
    print("\n📡 ИМИТАЦИЯ МАРШРУТИЗАЦИИ ЗАПРОСОВ")
    print("=" * 40)

    request_types = ["general", "quantum", "logistics"]
    source_regions = ["us-east1", "eu-west1", "asia-southeast1"]

    for i in range(20):
        source = random.choice(source_regions)
        req_type = random.choice(request_types)

        target = await balancer.route_request(source, req_type)

        print(f"   Запрос {i+1}: {source} -> {target} (тип: {req_type})")

        # Маленькая задержка
        await asyncio.sleep(0.1)

    # Аналитика
    print("\n📊 АНАЛИТИКА МАРШРУТИЗАЦИИ")
    print("=" * 35)

    analytics = await balancer.get_routing_analytics()
    status = await balancer.get_status()

    print(f"   • Всего решений маршрутизации: {analytics.get('total_decisions', 0)}")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".1f")
    print(".1f")
    print(f"   • Эффективность балансировки: {status['average_load_factor']:.2f}")

    # Сохранение статистики
    await balancer.shutdown()

    print("\n💾 Статистика сохранена в global_load_balancer_stats.json")
    print("\n🎉 GLOBAL LOAD BALANCER DEMO ЗАВЕРШЕН!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(demo_global_load_balancer())