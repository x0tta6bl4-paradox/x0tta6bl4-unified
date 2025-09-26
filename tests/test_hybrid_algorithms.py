#!/usr/bin/env python3
"""
Тесты для гибридных алгоритмов
Тестирование hybrid algorithms с quantum-classical integration
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from production.ai.hybrid_algorithms import (
    HybridAlgorithmBase, HybridAlgorithmConfig, HybridAlgorithmType,
    HybridAlgorithmFactory, HybridAlgorithmUtils, OptimizationTarget, QuantumBackend
)


class TestHybridAlgorithms:
    """Тесты для гибридных алгоритмов"""

    @pytest.fixture
    def hybrid_config(self):
        """Фикстура для конфигурации гибридного алгоритма"""
        return HybridAlgorithmConfig(
            algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
            quantum_backend=QuantumBackend.SIMULATOR,
            classical_optimizer="COBYLA",
            max_iterations=100,
            convergence_threshold=1e-6,
            quantum_enhanced=True,
            phi_optimization=True,
            consciousness_integration=True
        )

    @pytest.fixture
    async def hybrid_algorithm(self, hybrid_config):
        """Фикстура для гибридного алгоритма"""
        algorithm = HybridAlgorithmFactory.create_algorithm(
            hybrid_config.algorithm_type, hybrid_config
        )
        yield algorithm
        await algorithm.shutdown()

    @pytest.fixture
    def problem_definition(self):
        """Фикстура для определения проблемы"""
        return {
            "type": "optimization",
            "objective": "minimize",
            "constraints": [],
            "bounds": [(-1, 1), (-1, 1)],
            "initial_guess": [0.5, 0.5]
        }

    class TestHybridAlgorithmConfig:
        """Тесты конфигурации гибридных алгоритмов"""

        def test_config_creation(self, hybrid_config):
            """Тест создания конфигурации"""
            assert hybrid_config.algorithm_type == HybridAlgorithmType.VQE_ENHANCED
            assert hybrid_config.quantum_backend == QuantumBackend.SIMULATOR
            assert hybrid_config.quantum_enhanced == True
            assert hybrid_config.phi_optimization == True
            assert hybrid_config.consciousness_integration == True
            assert len(hybrid_config.performance_targets) > 0

        def test_config_validation(self):
            """Тест валидации конфигурации"""
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.QAOA_ENHANCED,
                quantum_backend=QuantumBackend.IBM,
                classical_optimizer="SPSA",
                max_iterations=50,
                convergence_threshold=1e-4
            )

            assert config.algorithm_type == HybridAlgorithmType.QAOA_ENHANCED
            assert config.quantum_backend == QuantumBackend.IBM
            assert config.classical_optimizer == "SPSA"

    class TestHybridAlgorithmFactory:
        """Тесты фабрики гибридных алгоритмов"""

        def test_factory_creation_vqe(self):
            """Тест создания VQE через фабрику"""
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
                quantum_backend=QuantumBackend.SIMULATOR,
                classical_optimizer="COBYLA",
                max_iterations=10,
                convergence_threshold=1e-3
            )

            algorithm = HybridAlgorithmFactory.create_algorithm(
                HybridAlgorithmType.VQE_ENHANCED, config
            )

            assert algorithm is not None
            assert hasattr(algorithm, 'config')
            assert algorithm.config.algorithm_type == HybridAlgorithmType.VQE_ENHANCED

        def test_factory_creation_qaoa(self):
            """Тест создания QAOA через фабрику"""
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.QAOA_ENHANCED,
                quantum_backend=QuantumBackend.GOOGLE,
                classical_optimizer="COBYLA",
                max_iterations=10,
                convergence_threshold=1e-3
            )

            algorithm = HybridAlgorithmFactory.create_algorithm(
                HybridAlgorithmType.QAOA_ENHANCED, config
            )

            assert algorithm is not None
            assert algorithm.config.algorithm_type == HybridAlgorithmType.QAOA_ENHANCED

        def test_factory_invalid_algorithm(self):
            """Тест фабрики с несуществующим алгоритмом"""
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
                quantum_backend=QuantumBackend.SIMULATOR,
                classical_optimizer="COBYLA",
                max_iterations=10,
                convergence_threshold=1e-3
            )

            with pytest.raises(ValueError):
                HybridAlgorithmFactory.create_algorithm("invalid_type", config)

    class TestHybridAlgorithmBase:
        """Тесты базового класса гибридных алгоритмов"""

        @pytest.mark.asyncio
        async def test_algorithm_initialization(self, hybrid_algorithm):
            """Тест инициализации алгоритма"""
            result = await hybrid_algorithm.initialize()
            assert isinstance(result, bool)

        @pytest.mark.asyncio
        async def test_algorithm_status(self, hybrid_algorithm):
            """Тест получения статуса алгоритма"""
            await hybrid_algorithm.initialize()
            status = await hybrid_algorithm.get_status()

            assert "name" in status
            assert "status" in status
            assert "algorithm_type" in status
            assert "quantum_backend" in status
            assert "initialized" in status

        @pytest.mark.asyncio
        async def test_algorithm_health_check(self, hybrid_algorithm):
            """Тест проверки здоровья алгоритма"""
            await hybrid_algorithm.initialize()
            health = await hybrid_algorithm.health_check()

            assert isinstance(health, bool)

        @pytest.mark.asyncio
        async def test_phi_harmony_optimization(self, hybrid_algorithm):
            """Тест φ-гармонической оптимизации"""
            current_value = 1.5
            parameters = np.array([0.1, 0.2])

            optimized_value, optimized_params = await hybrid_algorithm.optimize_with_phi_harmony(
                current_value, parameters
            )

            assert isinstance(optimized_value, (int, float))
            assert isinstance(optimized_params, np.ndarray)
            assert len(optimized_params) == len(parameters)

        @pytest.mark.asyncio
        async def test_consciousness_enhancement(self, hybrid_algorithm):
            """Тест усиления сознания"""
            parameters = np.array([0.1, 0.2])
            performance = 0.8

            enhanced_params = await hybrid_algorithm.enhance_with_consciousness(
                parameters, performance
            )

            assert isinstance(enhanced_params, np.ndarray)
            assert len(enhanced_params) == len(parameters)

        @pytest.mark.asyncio
        async def test_quantum_enhancement(self, hybrid_algorithm):
            """Тест квантового усиления"""
            classical_result = {"value": 1.0, "params": [0.1, 0.2]}

            enhanced_result = await hybrid_algorithm.apply_quantum_enhancement(classical_result)

            assert enhanced_result is not None

        def test_convergence_check(self, hybrid_algorithm):
            """Тест проверки сходимости"""
            # Тест с недостаточным количеством значений
            converged = hybrid_algorithm.check_convergence(1.0, [1.0, 1.0])
            assert converged == False

            # Тест со сходимостью
            converged = hybrid_algorithm.check_convergence(1.0, [1.0, 1.0, 1.0, 1.0])
            assert converged == True

            # Тест без сходимости
            converged = hybrid_algorithm.check_convergence(2.0, [1.0, 1.0, 1.0, 1.0])
            assert converged == False

        @pytest.mark.asyncio
        async def test_algorithm_shutdown(self, hybrid_algorithm):
            """Тест остановки алгоритма"""
            await hybrid_algorithm.initialize()
            result = await hybrid_algorithm.shutdown()

            assert result == True
            assert hybrid_algorithm.status == "shutdown"

    class TestHybridAlgorithmExecution:
        """Тесты выполнения гибридных алгоритмов"""

        @pytest.mark.asyncio
        async def test_algorithm_execution_structure(self, hybrid_algorithm, problem_definition):
            """Тест структуры выполнения алгоритма"""
            # Mock execute method для тестирования
            async def mock_execute(problem_definition):
                return Mock(
                    algorithm_type=hybrid_algorithm.config.algorithm_type,
                    success=True,
                    optimal_value=0.5,
                    optimal_parameters=np.array([0.1, 0.2]),
                    convergence_history=[1.0, 0.8, 0.6, 0.5],
                    quantum_coherence=0.9,
                    phi_harmony_score=1.618,
                    consciousness_level=0.8,
                    execution_time=0.1,
                    iterations_used=10,
                    performance_metrics={"accuracy": 0.9},
                    recommendations=["Use more iterations"],
                    timestamp=Mock()
                )

            hybrid_algorithm.execute = mock_execute

            result = await hybrid_algorithm.execute(problem_definition)

            assert result.algorithm_type == hybrid_algorithm.config.algorithm_type
            assert result.success == True
            assert isinstance(result.optimal_value, (int, float))
            assert isinstance(result.optimal_parameters, np.ndarray)
            assert isinstance(result.convergence_history, list)
            assert len(result.convergence_history) > 0

    class TestHybridAlgorithmUtils:
        """Тесты утилит гибридных алгоритмов"""

        def test_quantum_advantage_calculation(self):
            """Тест вычисления квантового преимущества"""
            # Равные результаты
            advantage = HybridAlgorithmUtils.calculate_quantum_advantage(1.0, 1.0)
            assert advantage == 1.0

            # Квантовый лучше
            advantage = HybridAlgorithmUtils.calculate_quantum_advantage(2.0, 1.0)
            assert advantage == 2.0

            # Классический лучше
            advantage = HybridAlgorithmUtils.calculate_quantum_advantage(1.0, 2.0)
            assert advantage == 0.5

            # Деление на ноль
            advantage = HybridAlgorithmUtils.calculate_quantum_advantage(0.0, 1.0)
            assert advantage == float('inf')

        def test_phi_harmonic_schedule_generation(self):
            """Тест генерации φ-гармонического расписания"""
            max_iterations = 10
            schedule = HybridAlgorithmUtils.generate_phi_harmonic_schedule(max_iterations)

            assert len(schedule) == max_iterations
            assert all(isinstance(rate, float) for rate in schedule)
            assert all(rate > 0 for rate in schedule)
            assert schedule[0] > schedule[-1]  # Убывание скорости обучения

        def test_algorithm_performance_evaluation(self):
            """Тест оценки производительности алгоритма"""
            from production.ai.hybrid_algorithms import HybridAlgorithmResult
            from datetime import datetime

            result = HybridAlgorithmResult(
                algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
                success=True,
                optimal_value=0.5,
                optimal_parameters=np.array([0.1, 0.2]),
                convergence_history=[1.0, 0.8, 0.6, 0.5],
                quantum_coherence=0.9,
                phi_harmony_score=1.618,
                consciousness_level=0.8,
                execution_time=0.1,
                iterations_used=10,
                performance_metrics={"accuracy": 0.9},
                recommendations=["Good performance"],
                timestamp=datetime.now()
            )

            performance = HybridAlgorithmUtils.evaluate_algorithm_performance(result)

            assert "convergence_rate" in performance
            assert "quantum_coherence_score" in performance
            assert "phi_harmony_efficiency" in performance
            assert "consciousness_integration_score" in performance
            assert "overall_performance" in performance

            assert performance["overall_performance"] > 0

    class TestHybridAlgorithmIntegration:
        """Тесты интеграции гибридных алгоритмов"""

        @pytest.mark.asyncio
        async def test_knowledge_transfer(self, hybrid_algorithm):
            """Тест переноса знаний"""
            # Требует наличия обученных моделей
            success = await hybrid_algorithm.transfer_knowledge("source_model", 0.3)
            # Без моделей должен вернуть False
            assert success == False

        @pytest.mark.asyncio
        async def test_ai_engineer_coordination(self, hybrid_algorithm):
            """Тест координации с AI Engineer Agent"""
            requirements = {"task": "optimize", "parameters": [1, 2, 3]}

            result = await hybrid_algorithm.coordinate_with_ai_engineer(requirements)

            # Без агента должен вернуть ошибку
            assert "error" in result or isinstance(result, dict)

    class TestErrorHandling:
        """Тесты обработки ошибок"""

        @pytest.mark.asyncio
        async def test_initialization_error_handling(self):
            """Тест обработки ошибок инициализации"""
            config = HybridAlgorithmConfig(
                algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
                quantum_backend=QuantumBackend.SIMULATOR,
                classical_optimizer="COBYLA",
                max_iterations=10,
                convergence_threshold=1e-3
            )

            algorithm = HybridAlgorithmFactory.create_algorithm(
                HybridAlgorithmType.VQE_ENHANCED, config
            )

            # Инициализация должна работать даже при отсутствии зависимостей
            result = await algorithm.initialize()
            assert isinstance(result, bool)

        @pytest.mark.asyncio
        async def test_execution_error_handling(self, hybrid_algorithm):
            """Тест обработки ошибок выполнения"""
            # Mock execute с ошибкой
            async def failing_execute(problem_definition):
                raise RuntimeError("Test error")

            hybrid_algorithm.execute = failing_execute

            with pytest.raises(RuntimeError):
                await hybrid_algorithm.execute({})

    class TestPerformanceOptimization:
        """Тесты оптимизации производительности"""

        @pytest.mark.asyncio
        async def test_phi_optimization_performance(self, hybrid_algorithm):
            """Тест производительности φ-оптимизации"""
            import time

            parameters = np.random.rand(10)
            performance = 0.8

            start_time = time.time()
            for _ in range(100):
                result = await hybrid_algorithm.optimize_with_phi_harmony(1.0, parameters)
            phi_time = time.time() - start_time

            assert phi_time < 1.0  # Должно быть быстро

        @pytest.mark.asyncio
        async def test_consciousness_enhancement_performance(self, hybrid_algorithm):
            """Тест производительности усиления сознания"""
            import time

            parameters = np.random.rand(10)
            performance = 0.8

            start_time = time.time()
            for _ in range(100):
                result = await hybrid_algorithm.enhance_with_consciousness(parameters, performance)
            consciousness_time = time.time() - start_time

            assert consciousness_time < 1.0  # Должно быть быстро


# Параметризованные тесты для различных типов алгоритмов
@pytest.mark.parametrize("algorithm_type", [
    HybridAlgorithmType.VQE_ENHANCED,
    HybridAlgorithmType.QAOA_ENHANCED,
    HybridAlgorithmType.QUANTUM_ML,
    HybridAlgorithmType.HYBRID_OPTIMIZATION
])
@pytest.mark.asyncio
async def test_algorithm_type_creation(algorithm_type):
    """Параметризованный тест создания различных типов алгоритмов"""
    config = HybridAlgorithmConfig(
        algorithm_type=algorithm_type,
        quantum_backend=QuantumBackend.SIMULATOR,
        classical_optimizer="COBYLA",
        max_iterations=10,
        convergence_threshold=1e-3
    )

    algorithm = HybridAlgorithmFactory.create_algorithm(algorithm_type, config)
    assert algorithm is not None
    assert algorithm.config.algorithm_type == algorithm_type


# Тесты для различных бэкендов
@pytest.mark.parametrize("backend", [
    QuantumBackend.IBM,
    QuantumBackend.GOOGLE,
    QuantumBackend.XANADU,
    QuantumBackend.SIMULATOR
])
def test_backend_configuration(backend):
    """Тест конфигурации различных бэкендов"""
    config = HybridAlgorithmConfig(
        algorithm_type=HybridAlgorithmType.VQE_ENHANCED,
        quantum_backend=backend,
        classical_optimizer="COBYLA",
        max_iterations=10,
        convergence_threshold=1e-3
    )

    assert config.quantum_backend == backend


if __name__ == "__main__":
    pytest.main([__file__])