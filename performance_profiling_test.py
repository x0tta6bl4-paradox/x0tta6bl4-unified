#!/usr/bin/env python3
"""
Performance Profiling Test для x0tta6bl4-unified
Memory и CPU profiling системы
"""

import asyncio
import time
import psutil
import tracemalloc
import cProfile
import pstats
import io
import json
from datetime import datetime
from typing import Dict, Any, List
import gc
import threading
import httpx

class PerformanceProfiler:
    """Профилировщик производительности"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.results = {
            "memory_profiling": [],
            "cpu_profiling": [],
            "memory_leak_tests": [],
            "performance_benchmarks": [],
            "summary": {}
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def run_performance_profiling(self) -> Dict[str, Any]:
        """Запуск полного профилирования производительности"""
        print("🔬 Запуск performance profiling x0tta6bl4-unified")
        print("=" * 60)

        start_time = time.time()

        # Memory profiling
        print("🧠 Memory profiling...")
        memory_results = await self.profile_memory_usage()
        self.results["memory_profiling"] = memory_results

        # CPU profiling
        print("⚡ CPU profiling...")
        cpu_results = await self.profile_cpu_usage()
        self.results["cpu_profiling"] = cpu_results

        # Memory leak detection
        print("💧 Memory leak detection...")
        leak_results = await self.detect_memory_leaks()
        self.results["memory_leak_tests"] = leak_results

        # Performance benchmarks
        print("🏃 Performance benchmarks...")
        benchmark_results = await self.run_performance_benchmarks()
        self.results["performance_benchmarks"] = benchmark_results

        # Анализ результатов
        self.results["summary"] = self.analyze_profiling_results()

        total_time = time.time() - start_time
        self.results["summary"]["total_profiling_time"] = total_time
        self.results["summary"]["timestamp"] = datetime.now().isoformat()

        print(".2f")
        return self.results

    async def profile_memory_usage(self) -> List[Dict[str, Any]]:
        """Профилирование использования памяти"""
        results = []

        # Включаем tracemalloc для детального отслеживания
        tracemalloc.start()

        scenarios = [
            "idle_system",
            "light_load",
            "medium_load",
            "heavy_load",
            "stress_test"
        ]

        for scenario in scenarios:
            result = await self.measure_memory_scenario(scenario)
            results.append(result)

        tracemalloc.stop()
        return results

    async def measure_memory_scenario(self, scenario: str) -> Dict[str, Any]:
        """Измерение использования памяти для конкретного сценария"""
        print(f"   Измерение памяти для {scenario}...")

        # Базовые измерения
        initial_memory = psutil.virtual_memory()
        process = psutil.Process()

        start_time = time.time()
        memory_samples = []

        # Создание нагрузки в зависимости от сценария
        if scenario == "idle_system":
            # Просто измерение в состоянии покоя
            duration = 5
            for _ in range(duration):
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                await asyncio.sleep(1)

        elif scenario == "light_load":
            # Легкая нагрузка - несколько одновременных запросов
            duration = 10
            tasks = []
            for _ in range(duration):
                tasks.append(self.make_light_request())
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.5)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "medium_load":
            # Средняя нагрузка
            duration = 15
            tasks = []
            for _ in range(duration):
                tasks.extend([self.make_medium_request() for _ in range(3)])
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.3)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "heavy_load":
            # Тяжелая нагрузка
            duration = 20
            tasks = []
            for _ in range(duration):
                tasks.extend([self.make_heavy_request() for _ in range(5)])
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.2)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "stress_test":
            # Стресс-тест
            duration = 30
            tasks = []
            for _ in range(duration):
                tasks.extend([self.make_stress_request() for _ in range(10)])
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.1)
            await asyncio.gather(*tasks, return_exceptions=True)

        # Финальные измерения
        final_memory = psutil.virtual_memory()
        end_time = time.time()

        # Статистика
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        max_memory = max(memory_samples) if memory_samples else 0
        min_memory = min(memory_samples) if memory_samples else 0
        memory_variance = sum((x - avg_memory) ** 2 for x in memory_samples) / len(memory_samples) if memory_samples else 0

        # Tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]  # Top 10 memory consumers

        return {
            "scenario": scenario,
            "duration": end_time - start_time,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "min_memory_mb": min_memory,
            "memory_variance": memory_variance,
            "system_memory_percent": final_memory.percent,
            "memory_samples": memory_samples[:20],  # Первые 20 сэмплов
            "top_memory_consumers": [
                {
                    "file": stat.traceback[0].filename if stat.traceback else "unknown",
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in top_stats
            ],
            "status": "passed" if avg_memory < 500 else "warning" if avg_memory < 1000 else "failed"
        }

    async def make_light_request(self):
        """Легкий запрос"""
        try:
            await self.client.get(f"{self.base_url}/health")
        except Exception:
            pass

    async def make_medium_request(self):
        """Средний запрос"""
        try:
            await asyncio.gather(
                self.client.get(f"{self.base_url}/health"),
                self.client.get(f"{self.base_url}/api/v1/quantum/status"),
                self.client.get(f"{self.base_url}/api/v1/ai/status")
            )
        except Exception:
            pass

    async def make_heavy_request(self):
        """Тяжелый запрос"""
        try:
            await asyncio.gather(
                self.client.get(f"{self.base_url}/api/v1/monitoring/metrics"),
                self.client.get(f"{self.base_url}/api/v1/enterprise/status"),
                self.client.get(f"{self.base_url}/api/v1/billing/status"),
                self.client.get(f"{self.base_url}/api/v1/monitoring/status")
            )
        except Exception:
            pass

    async def make_stress_request(self):
        """Стресс запрос"""
        try:
            # Множественные запросы к одному endpoint
            tasks = [self.client.get(f"{self.base_url}/api/v1/monitoring/metrics") for _ in range(5)]
            await asyncio.gather(*tasks)
        except Exception:
            pass

    async def profile_cpu_usage(self) -> List[Dict[str, Any]]:
        """Профилирование использования CPU"""
        results = []

        scenarios = [
            "idle_cpu",
            "light_cpu_load",
            "medium_cpu_load",
            "heavy_cpu_load",
            "cpu_stress_test"
        ]

        for scenario in scenarios:
            result = await self.measure_cpu_scenario(scenario)
            results.append(result)

        return results

    async def measure_cpu_scenario(self, scenario: str) -> Dict[str, Any]:
        """Измерение использования CPU для сценария"""
        print(f"   Измерение CPU для {scenario}...")

        process = psutil.Process()
        start_time = time.time()
        cpu_samples = []

        # Создание нагрузки
        if scenario == "idle_cpu":
            duration = 5
            for _ in range(duration):
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.9)

        elif scenario == "light_cpu_load":
            duration = 10
            tasks = []
            for _ in range(duration):
                tasks.append(self.cpu_intensive_task(0.1))
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.4)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "medium_cpu_load":
            duration = 15
            tasks = []
            for _ in range(duration):
                tasks.extend([self.cpu_intensive_task(0.2) for _ in range(2)])
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.3)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "heavy_cpu_load":
            duration = 20
            tasks = []
            for _ in range(duration):
                tasks.extend([self.cpu_intensive_task(0.3) for _ in range(3)])
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.2)
            await asyncio.gather(*tasks, return_exceptions=True)

        elif scenario == "cpu_stress_test":
            duration = 30
            tasks = []
            for _ in range(duration):
                tasks.extend([self.cpu_intensive_task(0.5) for _ in range(5)])
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(0.1)
            await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # Статистика CPU
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        min_cpu = min(cpu_samples) if cpu_samples else 0

        # CPU profiling с cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Выполнение тестового кода
        await self.cpu_intensive_task(0.1)

        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        profile_output = s.getvalue()

        return {
            "scenario": scenario,
            "duration": end_time - start_time,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "min_cpu_percent": min_cpu,
            "cpu_samples": cpu_samples[:20],  # Первые 20 сэмплов
            "cpu_profile": profile_output.split('\n')[:20],  # Первые 20 строк профиля
            "status": "passed" if avg_cpu < 80 else "warning" if avg_cpu < 95 else "failed"
        }

    async def cpu_intensive_task(self, duration: float):
        """CPU-интенсивная задача"""
        def cpu_work():
            # Вычислительно сложная операция
            result = 0
            for i in range(int(duration * 100000)):
                result += i ** 2
            return result

        # Выполнение в thread pool чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, cpu_work)

    async def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Обнаружение утечек памяти"""
        results = []

        leak_scenarios = [
            "short_lifecycle",
            "long_running_operations",
            "cyclic_references",
            "large_object_creation",
            "garbage_collection_test"
        ]

        for scenario in leak_scenarios:
            result = await self.test_memory_leak_scenario(scenario)
            results.append(result)

        return results

    async def test_memory_leak_scenario(self, scenario: str) -> Dict[str, Any]:
        """Тестирование сценария утечки памяти"""
        print(f"   Тестирование утечек для {scenario}...")

        tracemalloc.start()
        gc.collect()  # Очистка перед тестом

        initial_snapshot = tracemalloc.take_snapshot()

        # Выполнение сценария
        if scenario == "short_lifecycle":
            await self.short_lifecycle_test()
        elif scenario == "long_running_operations":
            await self.long_running_test()
        elif scenario == "cyclic_references":
            await self.cyclic_references_test()
        elif scenario == "large_object_creation":
            await self.large_objects_test()
        elif scenario == "garbage_collection_test":
            await self.gc_test()

        final_snapshot = tracemalloc.take_snapshot()
        gc.collect()  # Очистка после теста

        # Анализ разницы
        stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        total_growth = sum(stat.size_diff for stat in stats)

        # Крупнейшие утечки
        top_leaks = []
        for stat in stats[:5]:  # Top 5
            if stat.size_diff > 0:  # Только рост
                top_leaks.append({
                    "file": stat.traceback[0].filename if stat.traceback else "unknown",
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count_diff": stat.count_diff
                })

        tracemalloc.stop()

        return {
            "scenario": scenario,
            "total_memory_growth_mb": total_growth / 1024 / 1024,
            "top_memory_leaks": top_leaks,
            "leak_detected": total_growth > 10 * 1024 * 1024,  # > 10MB рост
            "status": "passed" if total_growth < 50 * 1024 * 1024 else "warning" if total_growth < 100 * 1024 * 1024 else "failed"
        }

    async def short_lifecycle_test(self):
        """Тест короткого жизненного цикла объектов"""
        objects = []
        for _ in range(1000):
            objects.append({"data": "x" * 1000})  # Создание объектов
        # Объекты должны быть очищены автоматически
        del objects

    async def long_running_test(self):
        """Тест длительных операций"""
        data = []
        for i in range(100):
            data.append({"id": i, "payload": "x" * 10000})
            await asyncio.sleep(0.01)  # Имитация работы
        # Проверка что память освобождается

    async def cyclic_references_test(self):
        """Тест циклических ссылок"""
        class Node:
            def __init__(self, value):
                self.value = value
                self.children = []

        # Создание циклических ссылок
        root = Node(1)
        child1 = Node(2)
        child2 = Node(3)
        root.children = [child1, child2]
        child1.children = [root]  # Циклическая ссылка
        child2.children = [root]  # Циклическая ссылка

        # Без gc.collect() эти объекты могут остаться

    async def large_objects_test(self):
        """Тест создания больших объектов"""
        large_objects = []
        for _ in range(50):
            large_objects.append(bytearray(1024 * 1024))  # 1MB каждый
        # Проверка очистки

    async def gc_test(self):
        """Тест сборки мусора"""
        # Создание объектов с циклическими ссылками
        objects = []
        for _ in range(100):
            a = {"ref": None}
            b = {"ref": a}
            a["ref"] = b
            objects.append((a, b))

        # Принудительная сборка мусора
        collected = gc.collect()
        return collected

    async def run_performance_benchmarks(self) -> List[Dict[str, Any]]:
        """Запуск performance benchmarks"""
        results = []

        benchmarks = [
            "api_response_time",
            "concurrent_requests",
            "memory_allocation_rate",
            "cpu_intensive_operations",
            "io_operations"
        ]

        for benchmark in benchmarks:
            result = await self.run_single_benchmark(benchmark)
            results.append(result)

        return results

    async def run_single_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """Запуск одного benchmark'а"""
        print(f"   Запуск benchmark {benchmark_name}...")

        start_time = time.time()
        metrics = {}

        if benchmark_name == "api_response_time":
            metrics = await self.benchmark_api_response_time()
        elif benchmark_name == "concurrent_requests":
            metrics = await self.benchmark_concurrent_requests()
        elif benchmark_name == "memory_allocation_rate":
            metrics = await self.benchmark_memory_allocation()
        elif benchmark_name == "cpu_intensive_operations":
            metrics = await self.benchmark_cpu_operations()
        elif benchmark_name == "io_operations":
            metrics = await self.benchmark_io_operations()

        end_time = time.time()

        return {
            "benchmark": benchmark_name,
            "duration": end_time - start_time,
            "metrics": metrics,
            "status": "passed"  # Benchmarks всегда проходят, просто измеряют
        }

    async def benchmark_api_response_time(self) -> Dict[str, Any]:
        """Benchmark времени ответа API"""
        response_times = []

        for _ in range(100):
            start = time.time()
            try:
                response = await self.client.get(f"{self.base_url}/health")
                response_time = time.time() - start
                response_times.append(response_time)
            except Exception:
                response_times.append(1.0)  # Таймаут

        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)]
        }

    async def benchmark_concurrent_requests(self) -> Dict[str, Any]:
        """Benchmark конкурентных запросов"""
        concurrency_levels = [1, 5, 10, 20, 50]

        results = {}
        for concurrency in concurrency_levels:
            start_time = time.time()

            async def single_request():
                try:
                    await self.client.get(f"{self.base_url}/health")
                except Exception:
                    pass

            tasks = [single_request() for _ in range(concurrency)]
            await asyncio.gather(*tasks)

            total_time = time.time() - start_time
            results[f"concurrency_{concurrency}"] = {
                "total_time": total_time,
                "requests_per_second": concurrency / total_time if total_time > 0 else 0
            }

        return results

    async def benchmark_memory_allocation(self) -> Dict[str, Any]:
        """Benchmark скорости аллокации памяти"""
        allocation_sizes = [1000, 10000, 100000, 1000000]

        results = {}
        for size in allocation_sizes:
            start_time = time.time()
            objects = []
            for _ in range(1000):
                objects.append(bytearray(size))
            allocation_time = time.time() - start_time

            results[f"size_{size}"] = {
                "allocation_time": allocation_time,
                "allocation_rate": 1000 / allocation_time if allocation_time > 0 else 0
            }

        return results

    async def benchmark_cpu_operations(self) -> Dict[str, Any]:
        """Benchmark CPU операций"""
        operation_counts = [1000, 10000, 100000]

        results = {}
        for count in operation_counts:
            start_time = time.time()
            result = 0
            for i in range(count):
                result += i ** 2
            operation_time = time.time() - start_time

            results[f"operations_{count}"] = {
                "execution_time": operation_time,
                "operations_per_second": count / operation_time if operation_time > 0 else 0
            }

        return results

    async def benchmark_io_operations(self) -> Dict[str, Any]:
        """Benchmark IO операций"""
        # В реальной системе это включало бы файловые операции, сетевые вызовы и т.д.
        # Для симуляции используем asyncio.sleep
        io_operations = [10, 50, 100, 500]

        results = {}
        for ops in io_operations:
            start_time = time.time()
            tasks = [asyncio.sleep(0.001) for _ in range(ops)]  # Имитация IO
            await asyncio.gather(*tasks)
            io_time = time.time() - start_time

            results[f"io_ops_{ops}"] = {
                "total_time": io_time,
                "ops_per_second": ops / io_time if io_time > 0 else 0
            }

        return results

    def analyze_profiling_results(self) -> Dict[str, Any]:
        """Анализ результатов профилирования"""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "warning_tests": 0,
            "failed_tests": 0,
            "memory_analysis": {},
            "cpu_analysis": {},
            "leak_analysis": {},
            "benchmark_analysis": {},
            "recommendations": []
        }

        # Подсчет результатов
        all_results = (
            self.results["memory_profiling"] +
            self.results["cpu_profiling"] +
            self.results["memory_leak_tests"] +
            self.results["performance_benchmarks"]
        )

        for result in all_results:
            summary["total_tests"] += 1
            status = result.get("status", "unknown")
            if status == "passed":
                summary["passed_tests"] += 1
            elif status == "warning":
                summary["warning_tests"] += 1
            elif status == "failed":
                summary["failed_tests"] += 1

        # Анализ памяти
        memory_results = self.results["memory_profiling"]
        if memory_results:
            avg_memories = [r["avg_memory_mb"] for r in memory_results]
            summary["memory_analysis"] = {
                "average_memory_usage": sum(avg_memories) / len(avg_memories),
                "peak_memory_usage": max(avg_memories),
                "memory_efficiency": "good" if max(avg_memories) < 500 else "concerning" if max(avg_memories) < 1000 else "critical"
            }

        # Анализ CPU
        cpu_results = self.results["cpu_profiling"]
        if cpu_results:
            avg_cpus = [r["avg_cpu_percent"] for r in cpu_results]
            summary["cpu_analysis"] = {
                "average_cpu_usage": sum(avg_cpus) / len(avg_cpus),
                "peak_cpu_usage": max(avg_cpus),
                "cpu_efficiency": "good" if max(avg_cpus) < 70 else "concerning" if max(avg_cpus) < 90 else "critical"
            }

        # Анализ утечек
        leak_results = self.results["memory_leak_tests"]
        leaks_detected = sum(1 for r in leak_results if r.get("leak_detected", False))
        summary["leak_analysis"] = {
            "leaks_detected": leaks_detected,
            "total_scenarios": len(leak_results),
            "leak_percentage": leaks_detected / len(leak_results) if leak_results else 0
        }

        # Анализ benchmarks
        benchmark_results = self.results["performance_benchmarks"]
        if benchmark_results:
            api_benchmark = next((r for r in benchmark_results if r["benchmark"] == "api_response_time"), None)
            if api_benchmark:
                summary["benchmark_analysis"]["api_performance"] = api_benchmark["metrics"]

        # Генерация рекомендаций
        summary["recommendations"] = self.generate_profiling_recommendations(summary)

        return summary

    def generate_profiling_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе профилирования"""
        recommendations = []

        # Рекомендации по памяти
        mem_analysis = summary.get("memory_analysis", {})
        if mem_analysis.get("memory_efficiency") == "critical":
            recommendations.append("Критическое использование памяти - оптимизировать структуры данных и алгоритмы")
        elif mem_analysis.get("memory_efficiency") == "concerning":
            recommendations.append("Высокое использование памяти - рассмотреть оптимизацию кэширования")

        # Рекомендации по CPU
        cpu_analysis = summary.get("cpu_analysis", {})
        if cpu_analysis.get("cpu_efficiency") == "critical":
            recommendations.append("Критическая загрузка CPU - оптимизировать вычислительные алгоритмы")
        elif cpu_analysis.get("cpu_efficiency") == "concerning":
            recommendations.append("Высокая загрузка CPU - рассмотреть асинхронную обработку")

        # Рекомендации по утечкам
        leak_analysis = summary.get("leak_analysis", {})
        if leak_analysis.get("leak_percentage", 0) > 0.5:
            recommendations.append("Обнаружены множественные утечки памяти - провести аудит кода")

        # Рекомендации по производительности
        benchmark_analysis = summary.get("benchmark_analysis", {})
        api_perf = benchmark_analysis.get("api_performance", {})
        if api_perf.get("avg_response_time", 0) > 0.5:
            recommendations.append("Медленное время ответа API - оптимизировать endpoints")

        if not recommendations:
            recommendations.append("Производительность системы находится в приемлемых пределах")

        return recommendations

async def main():
    """Основная функция"""
    async with PerformanceProfiler() as profiler:
        results = await profiler.run_performance_profiling()

        # Сохранение результатов
        with open("performance_profiling_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print("\n📊 Результаты performance profiling сохранены в performance_profiling_results.json")

        # Вывод сводки
        summary = results["summary"]
        print("\n📈 Сводка результатов performance profiling:")
        print(f"   • Всего тестов: {summary['total_tests']}")
        print(f"   • Пройдено: {summary['passed_tests']}")
        print(f"   • Предупреждений: {summary['warning_tests']}")
        print(f"   • Провалено: {summary['failed_tests']}")

        print("\n🧠 Анализ памяти:")
        mem = summary["memory_analysis"]
        print(".1f")
        print(".1f")
        print(f"   • Эффективность: {mem['memory_efficiency']}")

        print("\n⚡ Анализ CPU:")
        cpu = summary["cpu_analysis"]
        print(".1f")
        print(".1f")
        print(f"   • Эффективность: {cpu['cpu_efficiency']}")

        print("\n💧 Анализ утечек:")
        leak = summary["leak_analysis"]
        print(f"   • Утечек обнаружено: {leak['leaks_detected']}/{leak['total_scenarios']}")
        print(".1%")

        print("\n🏃 Анализ benchmarks:")
        bench = summary["benchmark_analysis"]
        if "api_performance" in bench:
            api = bench["api_performance"]
            print(".4f")

        print("\n💡 Рекомендации:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")

if __name__ == "__main__":
    asyncio.run(main())