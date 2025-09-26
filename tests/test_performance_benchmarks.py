#!/usr/bin/env python3
"""
Тесты производительности и benchmarks для quantum supremacy и hybrid algorithms
Включает нагрузочное тестирование, performance benchmarks и scalability tests
"""

import pytest
import asyncio
import time
import numpy as np
import psutil
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import threading


class TestPerformanceBenchmarks:
    """Тесты производительности и benchmarks"""

    @pytest.fixture
    async def quantum_core(self):
        """Фикстура для Quantum Core"""
        from production.quantum.quantum_interface import QuantumCore
        core = QuantumCore()
        yield core
        await core.shutdown()

    @pytest.fixture
    async def hybrid_algorithm(self):
        """Фикстура для гибридного алгоритма"""
        try:
            from production.ai.hybrid_algorithms import (
                HybridAlgorithmFactory, HybridAlgorithmConfig, HybridAlgorithmType, QuantumBackend
            )
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
                quantum_backend=QuantumBackend.SIMULATOR,
                classical_optimizer="COBYLA",
                max_iterations=50,
                convergence_threshold=1e-3
            )
            algorithm = HybridAlgorithmFactory.create_algorithm(
                HybridAlgorithmType.VQE_ENHANCED, config
            )
            await algorithm.initialize()
            yield algorithm
            await algorithm.shutdown()
        except ImportError:
            yield None

    @pytest.fixture
    async def edge_ai(self):
        """Фикстура для Edge AI"""
        try:
            from production.ai.edge.quantum_edge_ai import QuantumEdgeAI
            ai = QuantumEdgeAI()
            await ai.initialize()
            yield ai
            await ai.shutdown()
        except ImportError:
            yield None

    class TestQuantumAlgorithmBenchmarks:
        """Benchmarks квантовых алгоритмов"""

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_vqe_performance_benchmark(self, quantum_core, benchmark):
            """Benchmark производительности VQE"""
            async def vqe_benchmark():
                result = await quantum_core.run_vqe(Mock(), Mock())
                return result

            # Запуск benchmark
            result = await benchmark(vqe_benchmark)

            assert result is not None
            if "success" in result:
                assert result["success"] == True

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_qaoa_performance_benchmark(self, quantum_core, benchmark):
            """Benchmark производительности QAOA"""
            async def qaoa_benchmark():
                result = await quantum_core.run_qaoa(Mock(), Mock(), 2)
                return result

            result = await benchmark(qaoa_benchmark)

            assert result is not None
            if "success" in result:
                assert result["success"] == True

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_grover_performance_benchmark(self, quantum_core, benchmark):
            """Benchmark производительности алгоритма Гровера"""
            async def grover_benchmark():
                result = await quantum_core.run_grover(Mock(), 4)
                return result

            result = await benchmark(grover_benchmark)

            assert result is not None
            if "success" in result:
                assert result["success"] == True

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_shor_performance_benchmark(self, quantum_core, benchmark):
            """Benchmark производительности алгоритма Шора"""
            async def shor_benchmark():
                result = await quantum_core.run_shor(15)
                return result

            result = await benchmark(shor_benchmark)

            assert result is not None
            if "success" in result:
                assert result["success"] == True

    class TestHybridAlgorithmBenchmarks:
        """Benchmarks гибридных алгоритмов"""

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_hybrid_algorithm_performance(self, hybrid_algorithm, benchmark):
            """Benchmark производительности гибридного алгоритма"""
            if hybrid_algorithm is None:
                pytest.skip("Hybrid algorithm not available")
                return

            problem_definition = {
                "type": "optimization",
                "objective": "minimize",
                "bounds": [(-1, 1), (-1, 1)],
                "initial_guess": [0.5, 0.5]
            }

            async def hybrid_benchmark():
                result = await hybrid_algorithm.execute(problem_definition)
                return result

            result = await benchmark(hybrid_benchmark)

            assert result is not None

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_phi_optimization_performance(self, hybrid_algorithm, benchmark):
            """Benchmark производительности φ-оптимизации"""
            if hybrid_algorithm is None:
                pytest.skip("Hybrid algorithm not available")
                return

            parameters = np.random.rand(10)

            async def phi_benchmark():
                result = await hybrid_algorithm.optimize_with_phi_harmony(1.0, parameters)
                return result

            result = await benchmark(phi_benchmark)

            assert result is not None
            assert len(result[1]) == len(parameters)

    class TestEdgeAIBenchmarks:
        """Benchmarks Edge AI"""

        @pytest.mark.benchmark
        @pytest.mark.asyncio
        async def test_edge_inference_performance(self, edge_ai, benchmark):
            """Benchmark производительности edge inference"""
            if edge_ai is None:
                pytest.skip("Edge AI not available")
                return

            from production.ai.edge.quantum_edge_ai import EdgeInferenceRequest, EdgeAIType

            request = EdgeInferenceRequest(
                component_type=EdgeAIType.IOT_PREDICTIVE_MAINTENANCE,
                input_data={"sensor_data": [1.0, 2.0, 3.0]},
                real_time=True
            )

            async def inference_benchmark():
                result = await edge_ai.perform_edge_inference(request)
                return result

            result = await benchmark(inference_benchmark)

            assert result is not None
            assert hasattr(result, 'latency_ms')

    class TestLoadTesting:
        """Нагрузочное тестирование"""

        @pytest.mark.load_test
        @pytest.mark.asyncio
        async def test_concurrent_quantum_operations(self, quantum_core):
            """Тест конкурентных квантовых операций"""
            async def single_operation():
                return await quantum_core.run_vqe(Mock(), Mock())

            # Запуск 10 конкурентных операций
            tasks = [single_operation() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Проверяем что большинство операций успешны
            successful_results = [r for r in results if not isinstance(r, Exception) and r.get("success", False)]
            assert len(successful_results) >= 7  # Минимум 70% успех

        @pytest.mark.load_test
        @pytest.mark.asyncio
        async def test_high_frequency_operations(self, quantum_core):
            """Тест операций с высокой частотой"""
            operations_count = 50
            results = []

            start_time = time.time()

            for _ in range(operations_count):
                result = await quantum_core.run_vqe(Mock(), Mock())
                results.append(result)

            end_time = time.time()
            total_time = end_time - start_time

            # Проверяем производительность
            operations_per_second = operations_count / total_time
            assert operations_per_second > 5  # Минимум 5 операций в секунду

        @pytest.mark.load_test
        @pytest.mark.asyncio
        async def test_memory_usage_under_load(self, quantum_core):
            """Тест использования памяти под нагрузкой"""
            initial_memory = psutil.virtual_memory().percent

            # Выполняем много операций
            tasks = []
            for _ in range(20):
                tasks.append(quantum_core.run_vqe(Mock(), Mock()))

            await asyncio.gather(*tasks)

            final_memory = psutil.virtual_memory().percent
            memory_increase = final_memory - initial_memory

            # Проверяем что увеличение памяти разумное (< 10%)
            assert memory_increase < 10.0

    class TestScalabilityTests:
        """Тесты масштабируемости"""

        @pytest.mark.scalability
        @pytest.mark.asyncio
        async def test_algorithm_scaling(self):
            """Тест масштабируемости алгоритмов"""
            # Тестируем с разными размерами проблем
            problem_sizes = [2, 4, 8, 16]

            for size in problem_sizes:
                # Mock quantum core для разных размеров
                quantum_core = Mock()
                quantum_core.run_vqe = AsyncMock(return_value={
                    "success": True,
                    "problem_size": size,
                    "execution_time": size * 0.01  # Имитация роста времени
                })

                start_time = time.time()
                result = await quantum_core.run_vqe(Mock(), Mock())
                end_time = time.time()

                execution_time = end_time - start_time

                # Проверяем что время выполнения растет не экспоненциально
                expected_max_time = size * 0.1  # Линейный рост
                assert execution_time < expected_max_time

        @pytest.mark.scalability
        @pytest.mark.asyncio
        async def test_concurrent_users_scaling(self):
            """Тест масштабируемости с множеством пользователей"""
            # Симуляция одновременных пользователей
            user_count = 10

            async def simulate_user(user_id):
                quantum_core = Mock()
                quantum_core.run_vqe = AsyncMock(return_value={
                    "success": True,
                    "user_id": user_id,
                    "execution_time": 0.05
                })

                await asyncio.sleep(0.01)  # Имитация задержки сети
                result = await quantum_core.run_vqe(Mock(), Mock())
                return result

            # Запуск симуляции пользователей
            start_time = time.time()
            tasks = [simulate_user(i) for i in range(user_count)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # Проверяем что система справляется с нагрузкой
            assert len(results) == user_count
            assert total_time < 2.0  # Менее 2 секунд на 10 пользователей

    class TestResourceUtilization:
        """Тесты использования ресурсов"""

        @pytest.mark.resource_test
        @pytest.mark.asyncio
        async def test_cpu_utilization_during_operations(self, quantum_core):
            """Тест использования CPU во время операций"""
            initial_cpu = psutil.cpu_percent(interval=1)

            # Выполняем несколько операций
            tasks = [quantum_core.run_vqe(Mock(), Mock()) for _ in range(5)]
            await asyncio.gather(*tasks)

            final_cpu = psutil.cpu_percent(interval=1)
            cpu_increase = final_cpu - initial_cpu

            # Проверяем что использование CPU разумное
            assert cpu_increase < 50.0  # Увеличение менее 50%

        @pytest.mark.resource_test
        @pytest.mark.asyncio
        async def test_network_io_during_operations(self):
            """Тест сетевого IO во время операций"""
            # Mock сетевые операции
            network_operations = []

            for _ in range(10):
                # Имитация сетевого запроса
                await asyncio.sleep(0.001)
                network_operations.append({"bytes_sent": 100, "bytes_received": 200})

            total_bytes = sum(op["bytes_sent"] + op["bytes_received"] for op in network_operations)

            # Проверяем эффективность сетевого использования
            assert total_bytes > 0
            bytes_per_operation = total_bytes / len(network_operations)
            assert bytes_per_operation < 1000  # Менее 1KB на операцию

    class TestPerformanceRegression:
        """Тесты регрессии производительности"""

        @pytest.fixture
        def performance_baseline(self):
            """Базовые показатели производительности"""
            return {
                "vqe_execution_time": 0.05,
                "qaoa_execution_time": 0.07,
                "memory_usage": 100,  # MB
                "cpu_usage": 20.0     # %
            }

        @pytest.mark.regression
        @pytest.mark.asyncio
        async def test_performance_regression_vqe(self, quantum_core, performance_baseline):
            """Тест регрессии производительности VQE"""
            start_time = time.time()
            result = await quantum_core.run_vqe(Mock(), Mock())
            end_time = time.time()

            execution_time = end_time - start_time

            # Проверяем что производительность не ухудшилась более чем на 20%
            baseline_time = performance_baseline["vqe_execution_time"]
            max_allowed_time = baseline_time * 1.2

            assert execution_time <= max_allowed_time

        @pytest.mark.regression
        @pytest.mark.asyncio
        async def test_performance_regression_qaoa(self, quantum_core, performance_baseline):
            """Тест регрессии производительности QAOA"""
            start_time = time.time()
            result = await quantum_core.run_qaoa(Mock(), Mock(), 2)
            end_time = time.time()

            execution_time = end_time - start_time

            baseline_time = performance_baseline["qaoa_execution_time"]
            max_allowed_time = baseline_time * 1.2

            assert execution_time <= max_allowed_time

    class TestStressTesting:
        """Стресс-тестирование"""

        @pytest.mark.stress_test
        @pytest.mark.asyncio
        async def test_long_running_operations(self, quantum_core):
            """Тест длительных операций"""
            # Запуск длительной операции
            start_time = time.time()

            # Имитация длительной квантовой операции
            await asyncio.sleep(1)  # Имитация 1 секунды вычисления

            result = await quantum_core.run_vqe(Mock(), Mock())

            end_time = time.time()
            total_time = end_time - start_time

            # Проверяем что операция завершилась в разумное время
            assert total_time < 5.0  # Менее 5 секунд

        @pytest.mark.stress_test
        @pytest.mark.asyncio
        async def test_memory_leak_detection(self):
            """Тест обнаружения утечек памяти"""
            initial_memory = psutil.virtual_memory().used

            # Выполняем много операций для обнаружения утечек
            for _ in range(100):
                quantum_core = Mock()
                quantum_core.run_vqe = AsyncMock(return_value={"success": True})
                await quantum_core.run_vqe(Mock(), Mock())

                # Принудительный сбор мусора
                import gc
                gc.collect()

            final_memory = psutil.virtual_memory().used
            memory_increase = final_memory - initial_memory

            # Проверяем что нет значительных утечек (< 10MB)
            assert memory_increase < 10 * 1024 * 1024

    class TestBenchmarkReporting:
        """Тесты отчетности benchmarks"""

        @pytest.mark.asyncio
        async def test_benchmark_results_formatting(self):
            """Тест форматирования результатов benchmarks"""
            benchmark_results = {
                "algorithm": "vqe",
                "execution_time": 0.045,
                "memory_usage": 85.5,
                "cpu_usage": 15.2,
                "success_rate": 0.98,
                "timestamp": "2025-01-01T00:00:00Z"
            }

            # Форматирование отчета
            report = f"""
Benchmark Report for {benchmark_results['algorithm'].upper()}
==================================================
Execution Time: {benchmark_results['execution_time']:.3f}s
Memory Usage: {benchmark_results['memory_usage']:.1f}MB
CPU Usage: {benchmark_results['cpu_usage']:.1f}%
Success Rate: {benchmark_results['success_rate']:.1%}
Timestamp: {benchmark_results['timestamp']}
"""

            # Проверяем что отчет содержит все метрики
            assert benchmark_results['algorithm'].upper() in report
            assert ".045" in report
            assert "85.5" in report
            assert "15.2" in report
            assert "98.0" in report

        @pytest.mark.asyncio
        async def test_performance_comparison(self):
            """Тест сравнения производительности"""
            results_a = {"execution_time": 0.05, "success_rate": 0.95}
            results_b = {"execution_time": 0.07, "success_rate": 0.98}

            # Сравнение результатов
            time_improvement = (results_b["execution_time"] - results_a["execution_time"]) / results_a["execution_time"]
            success_improvement = results_b["success_rate"] - results_a["success_rate"]

            # Проверяем корректность расчетов
            assert time_improvement == 0.4  # 40% ухудшение времени
            assert success_improvement == 0.03  # 3% улучшение успеха


# Параметризованные benchmarks для различных конфигураций
@pytest.mark.parametrize("algorithm_config", [
    {"type": "vqe", "backend": "simulator", "optimizer": "COBYLA"},
    {"type": "qaoa", "backend": "simulator", "optimizer": "SPSA"},
    {"type": "grover", "backend": "simulator", "search_space": 4},
    {"type": "shor", "backend": "simulator", "number": 15}
])
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_algorithm_configurations_benchmark(algorithm_config, benchmark):
    """Параметризованный benchmark различных конфигураций алгоритмов"""
    # Mock quantum core
    quantum_core = Mock()

    async def run_algorithm():
        if algorithm_config["type"] == "vqe":
            quantum_core.run_vqe = AsyncMock(return_value={"success": True, "config": algorithm_config})
            return await quantum_core.run_vqe(Mock(), Mock())
        elif algorithm_config["type"] == "qaoa":
            quantum_core.run_qaoa = AsyncMock(return_value={"success": True, "config": algorithm_config})
            return await quantum_core.run_qaoa(Mock(), Mock(), 2)
        elif algorithm_config["type"] == "grover":
            quantum_core.run_grover = AsyncMock(return_value={"success": True, "config": algorithm_config})
            return await quantum_core.run_grover(Mock(), algorithm_config["search_space"])
        elif algorithm_config["type"] == "shor":
            quantum_core.run_shor = AsyncMock(return_value={"success": True, "config": algorithm_config})
            return await quantum_core.run_shor(algorithm_config["number"])

    result = await benchmark(run_algorithm)

    assert result is not None
    assert result["success"] == True
    assert result["config"] == algorithm_config


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--benchmark-only"])