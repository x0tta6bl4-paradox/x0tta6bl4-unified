#!/usr/bin/env python3
"""
Load Testing Script для x0tta6bl4-unified
Использует httpx для нагрузочного тестирования API endpoints
Добавлена quantum noise simulation для realistic testing
"""

import asyncio
import time
import httpx
import statistics
from typing import List, Dict, Any
from datetime import datetime
import json
import numpy as np
import random

class QuantumNoiseSimulator:
    """Симулятор квантового шума для realistic testing"""

    def __init__(self):
        self.t1_relaxation_rate = 0.01  # T1 relaxation rate
        self.t2_dephasing_rate = 0.005  # T2 dephasing rate
        self.gate_error_rate = 0.001    # Base gate error rate
        self.entanglement_decay_rate = 0.02  # Entanglement degradation

    def apply_t1_relaxation(self, coherence: float, time_step: float) -> float:
        """Применение T1 relaxation"""
        decay = 1 - self.t1_relaxation_rate * time_step
        return max(0, coherence * decay)

    def apply_t2_dephasing(self, coherence: float, time_step: float) -> float:
        """Применение T2 dephasing"""
        decay = 1 - self.t2_dephasing_rate * time_step
        return max(0, coherence * decay)

    def apply_gate_error(self, success_probability: float) -> bool:
        """Симуляция ошибки гейта"""
        error_probability = self.gate_error_rate * (1 + random.uniform(-0.5, 0.5))
        return random.random() > error_probability

    def apply_entanglement_degradation(self, fidelity: float, time_step: float) -> float:
        """Симуляция деградации перепутывания"""
        degradation = self.entanglement_decay_rate * time_step
        noise = np.random.normal(0, 0.01)
        return max(0, fidelity - degradation + noise)

    def simulate_quantum_noise(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Применение полного набора quantum noise к запросу"""
        # Имитация quantum processing time с шумом
        processing_time = request_data.get("processing_time", 0.1)
        coherence = 1.0

        # Применение различных типов шума
        coherence = self.apply_t1_relaxation(coherence, processing_time)
        coherence = self.apply_t2_dephasing(coherence, processing_time)

        # Gate operations simulation
        gate_success = self.apply_gate_error(0.99)
        if not gate_success:
            coherence *= 0.8  # Gate error penalty

        # Entanglement if applicable
        if "entanglement_required" in request_data:
            fidelity = request_data.get("initial_fidelity", 0.98)
            fidelity = self.apply_entanglement_degradation(fidelity, processing_time)
            request_data["final_fidelity"] = fidelity

        # Добавление quantum noise к response time
        quantum_noise_factor = 1 + (1 - coherence) * 0.5  # Up to 50% slowdown
        request_data["quantum_noise_factor"] = quantum_noise_factor

        return request_data


class LoadTester:
    """Класс для нагрузочного тестирования с quantum noise simulation"""

    def __init__(self, base_url: str = "http://localhost:8000", enable_quantum_noise: bool = True):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.quantum_noise = QuantumNoiseSimulator() if enable_quantum_noise else None
        self.quantum_metrics = {
            "coherence_history": [],
            "gate_errors": 0,
            "entanglement_degradations": 0,
            "total_quantum_noise_factor": 0
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def test_endpoint(self, endpoint: str, num_requests: int = 100, concurrency: int = 10) -> Dict[str, Any]:
        """Тестирование конкретного endpoint'а с quantum noise simulation"""
        print(f"Тестирование {endpoint} с {num_requests} запросами, concurrency={concurrency}")

        async def single_request():
            start_time = time.time()

            # Подготовка данных для quantum noise simulation
            request_data = {
                "processing_time": random.uniform(0.05, 0.2),  # Simulated quantum processing time
                "endpoint": endpoint
            }

            # Добавление entanglement для quantum endpoints
            if "quantum" in endpoint:
                request_data["entanglement_required"] = True
                request_data["initial_fidelity"] = 0.98

            # Применение quantum noise если включено
            if self.quantum_noise:
                request_data = self.quantum_noise.simulate_quantum_noise(request_data)
                quantum_noise_factor = request_data.get("quantum_noise_factor", 1.0)

                # Обновление метрик
                coherence = 1.0  # Simplified coherence tracking
                self.quantum_metrics["coherence_history"].append(coherence)
                self.quantum_metrics["total_quantum_noise_factor"] += quantum_noise_factor

                if not self.quantum_noise.apply_gate_error(0.99):
                    self.quantum_metrics["gate_errors"] += 1

                if "final_fidelity" in request_data and request_data["final_fidelity"] < 0.9:
                    self.quantum_metrics["entanglement_degradations"] += 1

            try:
                # Имитация задержки от quantum noise
                if self.quantum_noise and quantum_noise_factor > 1.0:
                    await asyncio.sleep((quantum_noise_factor - 1.0) * 0.01)

                response = await self.client.get(f"{self.base_url}{endpoint}")
                response_time = time.time() - start_time

                # Применение quantum noise к response time
                if self.quantum_noise:
                    response_time *= quantum_noise_factor

                result = {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200,
                    "quantum_noise_applied": self.quantum_noise is not None
                }

                if self.quantum_noise and "final_fidelity" in request_data:
                    result["quantum_fidelity"] = request_data["final_fidelity"]

                return result

            except Exception as e:
                response_time = time.time() - start_time
                if self.quantum_noise:
                    response_time *= request_data.get("quantum_noise_factor", 1.0)

                return {
                    "status_code": None,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e),
                    "quantum_noise_applied": self.quantum_noise is not None
                }

        # Выполнение запросов с контролем concurrency
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def limited_request():
            async with semaphore:
                return await single_request()

        start_time = time.time()
        tasks = [limited_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Анализ результатов
        response_times = [r["response_time"] for r in results]
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        # Quantum-specific анализ
        quantum_fidelities = [r.get("quantum_fidelity", 1.0) for r in results if "quantum_fidelity" in r]
        quantum_noise_applied = any(r.get("quantum_noise_applied", False) for r in results)

        analysis = {
            "endpoint": endpoint,
            "total_requests": num_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / num_requests * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": np.percentile(response_times, 95),  # 95th percentile
            "p99_response_time": np.percentile(response_times, 99),  # 99th percentile
            "latency_under_100ms": len([t for t in response_times if t < 0.1]) / num_requests * 100,
            "timestamp": datetime.now().isoformat(),
            "quantum_noise_enabled": quantum_noise_applied
        }

        # Добавление quantum метрик если применимо
        if quantum_fidelities:
            analysis.update({
                "avg_quantum_fidelity": statistics.mean(quantum_fidelities),
                "min_quantum_fidelity": min(quantum_fidelities),
                "max_quantum_fidelity": max(quantum_fidelities),
                "fidelity_above_90": len([f for f in quantum_fidelities if f > 0.9]) / len(quantum_fidelities) * 100
            })

        return analysis

    async def run_full_test(self) -> Dict[str, Any]:
        """Полное тестирование всех endpoints"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/quantum/status",
            "/api/v1/ai/status",
            "/api/v1/enterprise/status",
            "/api/v1/billing/status",
            "/api/v1/monitoring/status",
            "/api/v1/monitoring/metrics"
        ]

        results = {}
        for endpoint in endpoints:
            try:
                result = await self.test_endpoint(endpoint, num_requests=50, concurrency=5)
                results[endpoint] = result
                print(f"✅ {endpoint}: {result.get('success_rate', 0):.2f}% success")
            except Exception as e:
                print(f"Ошибка тестирования {endpoint}: {e}")
                results[endpoint] = {"error": str(e)}

        # Добавление quantum метрик в общий результат
        quantum_summary = {}
        if self.quantum_noise:
            total_requests = sum(result.get("total_requests", 0) for result in results.values() if "total_requests" in result)
            quantum_summary = {
                "total_gate_errors": self.quantum_metrics["gate_errors"],
                "total_entanglement_degradations": self.quantum_metrics["entanglement_degradations"],
                "avg_quantum_noise_factor": self.quantum_metrics["total_quantum_noise_factor"] / max(total_requests, 1),
                "coherence_samples": len(self.quantum_metrics["coherence_history"])
            }

        return {
            "test_summary": {
                "total_endpoints": len(endpoints),
                "timestamp": datetime.now().isoformat(),
                "test_duration": "N/A",
                "quantum_noise_enabled": self.quantum_noise is not None
            },
            "quantum_summary": quantum_summary,
            "results": results
        }

async def main():
    """Основная функция"""
    print("🚀 Начинаем нагрузочное тестирование x0tta6bl4-unified с quantum noise simulation")
    print("=" * 60)

    async with LoadTester(enable_quantum_noise=True) as tester:
        results = await tester.run_full_test()

        # Сохранение результатов
        with open("load_test_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n📊 Результаты сохранены в load_test_results.json")

        # Вывод сводки
        print("\n📈 Сводка результатов:")
        for endpoint, result in results["results"].items():
            if "error" not in result:
                print(f"\n{endpoint}:")
                print(f"   • Success rate: {result['success_rate']:.2f}%")
                print(f"   • Avg response time: {result['avg_response_time']:.2f}s")
                print(f"   • Requests/sec: {result['requests_per_second']:.2f}")
                print(f"   • P95 latency: {result['p95_response_time']:.2f}s")
                if result["latency_under_100ms"] < 95:
                    print("⚠️  Низкий процент запросов <100ms")
                else:
                    print("✅ Хорошая производительность")

                # Вывод quantum метрик если применимо
                if result.get("quantum_noise_enabled"):
                    if "avg_quantum_fidelity" in result:
                        print(f"   • Avg quantum fidelity: {result['avg_quantum_fidelity']:.4f}")
                        print(f"   • Fidelity >90%: {result['fidelity_above_90']:.1f}%")

        # Вывод quantum summary
        if results.get("quantum_summary"):
            qs = results["quantum_summary"]
            print("\n🔬 Quantum Noise Summary:")
            print(f"   • Gate errors: {qs.get('total_gate_errors', 0)}")
            print(f"   • Entanglement degradations: {qs.get('total_entanglement_degradations', 0)}")
            print(f"   • Avg quantum noise factor: {qs.get('avg_quantum_noise_factor', 0):.3f}")
            print(f"   • Coherence samples: {qs.get('coherence_samples', 0)}")

if __name__ == "__main__":
    asyncio.run(main())