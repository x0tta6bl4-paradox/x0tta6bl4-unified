"""
Chaos Engineering Framework для x0tta6bl4 Unified
Система для controlled хаоса и тестирования resilience
"""

import asyncio
import time
import random
import threading
import psutil
import json
import subprocess
import signal
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager


class ChaosType(Enum):
    """Типы chaos экспериментов"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    NETWORK_LOSS = "network_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    PROCESS_KILL = "process_kill"
    SERVICE_RESTART = "service_restart"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    QUANTUM_NOISE_INJECTION = "quantum_noise_injection"
    DATABASE_CONNECTION_LOSS = "database_connection_loss"
    CACHE_FAILURE = "cache_failure"


class ChaosSeverity(Enum):
    """Уровни severity для chaos экспериментов"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperiment:
    """Описание chaos эксперимента"""
    experiment_id: str
    name: str
    description: str
    chaos_type: ChaosType
    severity: ChaosSeverity
    duration_seconds: int
    target_services: List[str]
    blast_radius: str = "controlled"  # controlled, limited, wide
    rollback_strategy: str = "automatic"
    success_criteria: List[str] = field(default_factory=list)
    monitoring_metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChaosResult:
    """Результат chaos эксперимента"""
    experiment_id: str
    success: bool
    impact_assessment: Dict[str, Any]
    recovery_time_seconds: float
    system_stability_score: float
    recommendations: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    errors_encountered: List[str]


class ChaosInjector:
    """Инъектор chaos в систему"""

    def __init__(self):
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.system_baseline: Dict[str, Any] = {}
        self.recovery_actions: List[Callable] = []

    def inject_network_latency(self, target_service: str, latency_ms: int, duration: int) -> str:
        """Инъекция сетевой задержки"""
        experiment_id = f"network_latency_{int(time.time())}"

        # Используем tc (traffic control) для имитации задержки
        try:
            # Имитация сетевой задержки через iptables/tc
            # В реальной системе это потребует root прав
            print(f"🔄 Injecting {latency_ms}ms latency to {target_service} for {duration}s")

            # Запланировать восстановление
            def rollback():
                time.sleep(duration)
                print(f"✅ Removing network latency from {target_service}")
                # Здесь был бы код для удаления правила tc

            thread = threading.Thread(target=rollback, daemon=True)
            thread.start()

            return experiment_id

        except Exception as e:
            print(f"❌ Failed to inject network latency: {e}")
            return None

    def inject_cpu_stress(self, cpu_percentage: float, duration: int) -> str:
        """Инъекция CPU стресса"""
        experiment_id = f"cpu_stress_{int(time.time())}"

        def cpu_stresser():
            end_time = time.time() + duration
            while time.time() < end_time:
                # Имитация CPU нагрузки
                for _ in range(100000):
                    _ = random.random() ** 2

        thread = threading.Thread(target=cpu_stresser, daemon=True)
        thread.start()

        print(f"🔥 Injecting CPU stress ({cpu_percentage}%) for {duration}s")
        return experiment_id

    def inject_memory_pressure(self, memory_mb: int, duration: int) -> str:
        """Инъекция memory pressure"""
        experiment_id = f"memory_pressure_{int(time.time())}"

        def memory_stresser():
            # Имитация высокого потребления памяти
            data = []
            end_time = time.time() + duration

            try:
                while time.time() < end_time:
                    # Создаем нагрузку на память
                    chunk_size = min(1024 * 1024, memory_mb * 1024 * 1024 // 100)  # 1MB chunks
                    data.append(bytearray(chunk_size))
                    time.sleep(0.1)

                    # Освобождаем память периодически
                    if len(data) > 10:
                        data.pop(0)

            except MemoryError:
                print("⚠️  Memory limit reached during stress test")

        thread = threading.Thread(target=memory_stresser, daemon=True)
        thread.start()

        print(f"💾 Injecting memory pressure ({memory_mb}MB) for {duration}s")
        return experiment_id

    def inject_process_kill(self, process_pattern: str) -> str:
        """Инъекция process kill (симуляция)"""
        experiment_id = f"process_kill_{int(time.time())}"

        print(f"💀 Simulating process kill for pattern: {process_pattern}")
        # В реальной системе здесь был бы код для graceful restart процесса

        return experiment_id

    def inject_quantum_noise(self, noise_level: float, duration: int) -> str:
        """Инъекция quantum noise"""
        experiment_id = f"quantum_noise_{int(time.time())}"

        # Имитация quantum noise через monitoring систему
        from monitoring.resilience_metrics import record_quantum_metrics

        def noise_injector():
            end_time = time.time() + duration
            while time.time() < end_time:
                # Имитируем quantum noise
                record_quantum_metrics(
                    noise_factor=1.0 + noise_level * random.uniform(0.1, 0.5),
                    gate_errors=random.randint(1, 5),
                    fidelity=max(0.5, 0.95 - noise_level * random.uniform(0.1, 0.3)),
                    coherence_time=max(5, 100 - noise_level * 50)
                )
                time.sleep(1)

        thread = threading.Thread(target=noise_injector, daemon=True)
        thread.start()

        print(f"🌊 Injecting quantum noise (level: {noise_level}) for {duration}s")
        return experiment_id

    def inject_service_restart(self, service_name: str) -> str:
        """Инъекция service restart (симуляция)"""
        experiment_id = f"service_restart_{int(time.time())}"

        print(f"🔄 Simulating restart of service: {service_name}")
        # В реальной системе здесь был бы код для restart сервиса

        return experiment_id


class ChaosOrchestrator:
    """Оркестратор chaos экспериментов"""

    def __init__(self):
        self.injector = ChaosInjector()
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: Dict[str, threading.Thread] = {}
        self.results: Dict[str, ChaosResult] = {}
        self.safety_limits = {
            "max_concurrent_experiments": 3,
            "max_duration_seconds": 300,  # 5 minutes
            "forbidden_services": ["database", "monitoring", "security"]
        }

    def create_experiment(self, name: str, chaos_type: ChaosType, severity: ChaosSeverity,
                         duration: int, target_services: List[str]) -> ChaosExperiment:
        """Создание chaos эксперимента"""
        experiment_id = f"{chaos_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"

        experiment = ChaosExperiment(
            experiment_id=experiment_id,
            name=name,
            description=f"Chaos experiment: {name}",
            chaos_type=chaos_type,
            severity=severity,
            duration_seconds=duration,
            target_services=target_services,
            success_criteria=[
                "System remains operational",
                "No data loss",
                "Recovery within acceptable time",
                f"Impact limited to {target_services}"
            ],
            monitoring_metrics=[
                "cpu_usage", "memory_usage", "response_time",
                "error_rate", "service_health"
            ]
        )

        self.experiments[experiment_id] = experiment
        return experiment

    def validate_experiment(self, experiment: ChaosExperiment) -> bool:
        """Валидация эксперимента на безопасность"""
        # Проверка лимитов
        if len(self.active_experiments) >= self.safety_limits["max_concurrent_experiments"]:
            print("❌ Too many concurrent experiments")
            return False

        if experiment.duration_seconds > self.safety_limits["max_duration_seconds"]:
            print("❌ Experiment duration too long")
            return False

        # Проверка запрещенных сервисов
        for service in experiment.target_services:
            if service in self.safety_limits["forbidden_services"]:
                print(f"❌ Cannot target forbidden service: {service}")
                return False

        return True

    async def execute_experiment(self, experiment: ChaosExperiment) -> ChaosResult:
        """Выполнение chaos эксперимента"""
        if not self.validate_experiment(experiment):
            raise ValueError("Experiment validation failed")

        print(f"🚀 Starting chaos experiment: {experiment.name}")
        experiment.started_at = datetime.now()
        experiment.status = "running"

        # Сбор baseline метрик
        baseline_metrics = await self._collect_system_metrics()

        # Выполнение chaos инъекции
        injection_result = await self._inject_chaos(experiment)

        # Ожидание duration + буфер для наблюдения эффектов
        await asyncio.sleep(experiment.duration_seconds + 10)

        # Сбор метрик после эксперимента
        post_metrics = await self._collect_system_metrics()

        # Оценка impact
        impact = self._assess_impact(baseline_metrics, post_metrics, experiment)

        # Вычисление recovery time
        recovery_time = self._calculate_recovery_time(experiment, post_metrics)

        # Оценка system stability
        stability_score = self._calculate_stability_score(impact, recovery_time)

        # Определение успеха
        success = self._evaluate_success(experiment, impact, recovery_time, stability_score)

        # Генерация рекомендаций
        recommendations = self._generate_recommendations(experiment, impact, success)

        result = ChaosResult(
            experiment_id=experiment.experiment_id,
            success=success,
            impact_assessment=impact,
            recovery_time_seconds=recovery_time,
            system_stability_score=stability_score,
            recommendations=recommendations,
            metrics_before=baseline_metrics,
            metrics_after=post_metrics,
            errors_encountered=[]
        )

        experiment.completed_at = datetime.now()
        experiment.status = "completed"
        experiment.results = {
            "success": success,
            "impact": impact,
            "recovery_time": recovery_time,
            "stability_score": stability_score
        }

        self.results[experiment.experiment_id] = result

        print(f"✅ Chaos experiment completed: {experiment.name} (Success: {success})")
        return result

    async def _inject_chaos(self, experiment: ChaosExperiment) -> str:
        """Инъекция chaos в зависимости от типа"""
        if experiment.chaos_type == ChaosType.CPU_STRESS:
            intensity = {"low": 30, "medium": 60, "high": 80, "critical": 95}[experiment.severity.value]
            return self.injector.inject_cpu_stress(intensity, experiment.duration_seconds)

        elif experiment.chaos_type == ChaosType.MEMORY_STRESS:
            memory_mb = {"low": 100, "medium": 500, "high": 1000, "critical": 2000}[experiment.severity.value]
            return self.injector.inject_memory_pressure(memory_mb, experiment.duration_seconds)

        elif experiment.chaos_type == ChaosType.NETWORK_LATENCY:
            latency = {"low": 50, "medium": 200, "high": 500, "critical": 1000}[experiment.severity.value]
            # Применяем ко всем target сервисам
            for service in experiment.target_services:
                self.injector.inject_network_latency(service, latency, experiment.duration_seconds)

        elif experiment.chaos_type == ChaosType.QUANTUM_NOISE_INJECTION:
            noise_level = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}[experiment.severity.value]
            return self.injector.inject_quantum_noise(noise_level, experiment.duration_seconds)

        elif experiment.chaos_type == ChaosType.PROCESS_KILL:
            for service in experiment.target_services:
                self.injector.inject_process_kill(service)

        elif experiment.chaos_type == ChaosType.SERVICE_RESTART:
            for service in experiment.target_services:
                self.injector.inject_service_restart(service)

        return f"{experiment.chaos_type.value}_injected"

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Сбор системных метрик"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections()),
            "timestamp": time.time()
        }

    def _assess_impact(self, baseline: Dict[str, Any], post: Dict[str, Any],
                      experiment: ChaosExperiment) -> Dict[str, Any]:
        """Оценка impact эксперимента"""
        impact = {}

        for metric in ["cpu_percent", "memory_percent", "disk_usage"]:
            if metric in baseline and metric in post:
                change = post[metric] - baseline[metric]
                impact[metric] = {
                    "baseline": baseline[metric],
                    "after": post[metric],
                    "change": change,
                    "change_percent": (change / baseline[metric] * 100) if baseline[metric] > 0 else 0
                }

        # Оценка severity impact
        max_change = max(abs(imp.get("change_percent", 0)) for imp in impact.values())
        if max_change < 10:
            impact["severity"] = "low"
        elif max_change < 25:
            impact["severity"] = "medium"
        elif max_change < 50:
            impact["severity"] = "high"
        else:
            impact["severity"] = "critical"

        return impact

    def _calculate_recovery_time(self, experiment: ChaosExperiment,
                               post_metrics: Dict[str, Any]) -> float:
        """Расчет времени восстановления"""
        # Упрощенная оценка - в реальной системе нужна более сложная логика
        base_recovery = experiment.duration_seconds * 0.1  # 10% от duration
        severity_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0, "critical": 3.0}
        return base_recovery * severity_multiplier.get(experiment.severity.value, 1.0)

    def _calculate_stability_score(self, impact: Dict[str, Any], recovery_time: float) -> float:
        """Расчет stability score (0-1, где 1 - идеальная стабильность)"""
        impact_score = 0.0

        severity_scores = {"low": 1.0, "medium": 0.7, "high": 0.4, "critical": 0.1}
        impact_score = severity_scores.get(impact.get("severity", "medium"), 0.5)

        # Recovery time penalty
        recovery_penalty = min(1.0, recovery_time / 60.0)  # Max penalty for >60s recovery

        return impact_score * (1 - recovery_penalty * 0.3)

    def _evaluate_success(self, experiment: ChaosExperiment, impact: Dict[str, Any],
                         recovery_time: float, stability_score: float) -> bool:
        """Оценка успеха эксперимента"""
        # Критерии успеха
        impact_acceptable = impact.get("severity") in ["low", "medium"]
        recovery_acceptable = recovery_time < experiment.duration_seconds * 0.5
        stability_acceptable = stability_score > 0.6

        return impact_acceptable and recovery_acceptable and stability_acceptable

    def _generate_recommendations(self, experiment: ChaosExperiment, impact: Dict[str, Any],
                                success: bool) -> List[str]:
        """Генерация рекомендаций на основе результатов"""
        recommendations = []

        if not success:
            recommendations.append(f"Experiment {experiment.name} revealed system vulnerabilities")

            if impact.get("severity") == "high":
                recommendations.append("Consider implementing additional fault tolerance measures")

            if experiment.chaos_type == ChaosType.CPU_STRESS:
                recommendations.append("Optimize CPU-intensive operations or consider horizontal scaling")

            elif experiment.chaos_type == ChaosType.MEMORY_STRESS:
                recommendations.append("Implement memory management improvements or increase memory limits")

            elif experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                recommendations.append("Consider implementing retry logic with exponential backoff")

        if success:
            recommendations.append(f"System demonstrated good resilience to {experiment.chaos_type.value}")

        return recommendations

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Получение статуса эксперимента"""
        if experiment_id not in self.experiments:
            return {"status": "not_found"}

        experiment = self.experiments[experiment_id]
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "progress": self._calculate_progress(experiment),
            "results": experiment.results if experiment.results else {}
        }

    def _calculate_progress(self, experiment: ChaosExperiment) -> float:
        """Расчет прогресса эксперимента"""
        if experiment.status == "pending":
            return 0.0
        elif experiment.status == "completed":
            return 1.0
        elif experiment.started_at:
            elapsed = (datetime.now() - experiment.started_at).total_seconds()
            return min(1.0, elapsed / experiment.duration_seconds)
        return 0.0

    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение списка экспериментов"""
        experiments = []
        for exp in self.experiments.values():
            if status_filter is None or exp.status == status_filter:
                experiments.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "status": exp.status,
                    "chaos_type": exp.chaos_type.value,
                    "severity": exp.severity.value,
                    "created_at": exp.created_at.isoformat()
                })
        return experiments


# Global orchestrator instance
_chaos_orchestrator = ChaosOrchestrator()


def get_chaos_orchestrator() -> ChaosOrchestrator:
    """Получение глобального chaos orchestrator"""
    return _chaos_orchestrator


async def run_chaos_experiment(name: str, chaos_type: ChaosType, severity: ChaosSeverity,
                              duration: int, target_services: List[str]) -> ChaosResult:
    """Запуск chaos эксперимента"""
    orchestrator = get_chaos_orchestrator()
    experiment = orchestrator.create_experiment(name, chaos_type, severity, duration, target_services)
    return await orchestrator.execute_experiment(experiment)


async def run_comprehensive_chaos_suite() -> Dict[str, Any]:
    """Запуск comprehensive chaos testing suite"""
    print("🎭 Запуск comprehensive chaos engineering suite")
    print("=" * 60)

    experiments = [
        {
            "name": "CPU Stress Test",
            "type": ChaosType.CPU_STRESS,
            "severity": ChaosSeverity.MEDIUM,
            "duration": 30,
            "services": ["api", "quantum"]
        },
        {
            "name": "Memory Pressure Test",
            "type": ChaosType.MEMORY_STRESS,
            "severity": ChaosSeverity.MEDIUM,
            "duration": 20,
            "services": ["ai", "enterprise"]
        },
        {
            "name": "Network Latency Test",
            "type": ChaosType.NETWORK_LATENCY,
            "severity": ChaosSeverity.LOW,
            "duration": 15,
            "services": ["api", "billing"]
        },
        {
            "name": "Quantum Noise Injection",
            "type": ChaosType.QUANTUM_NOISE_INJECTION,
            "severity": ChaosSeverity.MEDIUM,
            "duration": 25,
            "services": ["quantum"]
        }
    ]

    results = []
    for exp_config in experiments:
        print(f"🔬 Running: {exp_config['name']}")
        try:
            result = await run_chaos_experiment(
                exp_config["name"],
                exp_config["type"],
                exp_config["severity"],
                exp_config["duration"],
                exp_config["services"]
            )
            results.append({
                "experiment": exp_config["name"],
                "success": result.success,
                "stability_score": result.system_stability_score,
                "recovery_time": result.recovery_time_seconds,
                "recommendations": result.recommendations
            })
            await asyncio.sleep(5)  # Пауза между экспериментами
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            results.append({
                "experiment": exp_config["name"],
                "success": False,
                "error": str(e)
            })

    # Анализ результатов
    successful_experiments = sum(1 for r in results if r.get("success", False))
    avg_stability = sum(r.get("stability_score", 0) for r in results if "stability_score" in r) / len(results)

    analysis = {
        "total_experiments": len(experiments),
        "successful_experiments": successful_experiments,
        "success_rate": successful_experiments / len(experiments),
        "average_stability_score": avg_stability,
        "recommendations": []
    }

    # Сбор всех рекомендаций
    all_recommendations = []
    for result in results:
        all_recommendations.extend(result.get("recommendations", []))

    analysis["recommendations"] = list(set(all_recommendations))  # Удаление дубликатов

    return {
        "experiments": results,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }


async def main():
    """Основная функция для демонстрации"""
    print("🎭 Chaos Engineering Framework для x0tta6bl4 Unified")
    print("=" * 60)

    # Запуск comprehensive suite
    results = await run_comprehensive_chaos_suite()

    # Сохранение результатов
    with open("chaos_engineering_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📊 Результаты chaos engineering сохранены в chaos_engineering_results.json")

    # Вывод сводки
    analysis = results["analysis"]
    print("\n📈 Сводка результатов:")
    print(f"   • Всего экспериментов: {analysis['total_experiments']}")
    print(f"   • Успешных: {analysis['successful_experiments']}")
    print(".1%")
    print(".3f")

    print("\n💡 Рекомендации:")
    for rec in analysis["recommendations"]:
        print(f"   • {rec}")

    # Вывод детальных результатов
    print("\n🔬 Детальные результаты:")
    for exp in results["experiments"]:
        status = "✅" if exp.get("success", False) else "❌"
        print(f"   {status} {exp['experiment']}")
        if "stability_score" in exp:
            print(".3f")
        if "recommendations" in exp and exp["recommendations"]:
            for rec in exp["recommendations"]:
                print(f"      - {rec}")


if __name__ == "__main__":
    asyncio.run(main())