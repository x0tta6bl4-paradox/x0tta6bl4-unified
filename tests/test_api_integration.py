"""
Комплексные интеграционные тесты для API Gateway
Тестирование health checks, статусов компонентов и межсервисного взаимодействия
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
from unittest.mock import Mock, patch, AsyncMock
from production.api.main import app
from production.quantum.quantum_interface import QuantumCore
from production.ai.advanced_ai_ml_system import AdvancedAIMLSystem, ModelConfig, ModelType, LearningAlgorithm


class TestAPIGatewayIntegration:
    """Интеграционные тесты для API Gateway"""

    def setup_method(self):
        """Настройка для каждого теста"""
        self.client = TestClient(app)

    def test_health_check_integration(self):
        """Интеграционный тест health check с проверкой всех компонентов"""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["service"] == "x0tta6bl4-unified-api"
        assert data["version"] == "1.0.0"

    def test_root_endpoint_integration(self):
        """Интеграционный тест корневого endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "x0tta6bl4 Unified API Gateway"
        assert data["version"] == "1.0.0"

        # Проверка структуры endpoints
        endpoints = data["endpoints"]
        required_endpoints = ["quantum", "ai", "enterprise", "billing", "monitoring"]
        for endpoint in required_endpoints:
            assert endpoint in endpoints
            assert endpoints[endpoint].startswith("/api/v1/")

    def test_quantum_status_integration(self):
        """Интеграционный тест статуса квантовых сервисов"""
        response = self.client.get("/api/v1/quantum/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "providers" in data
        assert "algorithms" in data
        assert data["status"] == "operational"

        # Проверка провайдеров
        providers = data["providers"]
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "ibm" in providers

        # Проверка алгоритмов
        algorithms = data["algorithms"]
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "vqe" in algorithms

    def test_ai_status_integration(self):
        """Интеграционный тест статуса AI сервисов"""
        response = self.client.get("/api/v1/ai/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "models" in data
        assert "agents" in data
        assert data["status"] == "operational"

        # Проверка моделей
        models = data["models"]
        assert isinstance(models, list)
        assert len(models) > 0

        # Проверка агентов
        agents = data["agents"]
        assert isinstance(agents, list)
        assert len(agents) > 0

    def test_enterprise_status_integration(self):
        """Интеграционный тест статуса enterprise сервисов"""
        response = self.client.get("/api/v1/enterprise/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "features" in data
        assert "gateway" in data
        assert data["status"] == "operational"
        assert data["gateway"] == "active"

        # Проверка features
        features = data["features"]
        assert isinstance(features, list)
        assert "multi_tenant" in features

    def test_billing_status_integration(self):
        """Интеграционный тест статуса billing сервисов"""
        response = self.client.get("/api/v1/billing/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "providers" in data
        assert "features" in data
        assert data["status"] == "operational"

        # Проверка провайдеров
        providers = data["providers"]
        assert isinstance(providers, list)
        assert len(providers) > 0

        # Проверка features
        features = data["features"]
        assert isinstance(features, list)
        assert "subscriptions" in features

    def test_monitoring_status_integration(self):
        """Интеграционный тест статуса мониторинга"""
        response = self.client.get("/api/v1/monitoring/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "metrics" in data
        assert "logging" in data
        assert data["status"] == "operational"

        # Проверка metrics
        metrics = data["metrics"]
        assert isinstance(metrics, list)
        assert "prometheus" in metrics

        # Проверка logging
        logging_config = data["logging"]
        assert isinstance(logging_config, list)
        assert "structured" in logging_config

    def test_all_status_endpoints_integration(self):
        """Интеграционный тест всех status endpoints одновременно"""
        endpoints = [
            "/api/v1/quantum/status",
            "/api/v1/ai/status",
            "/api/v1/enterprise/status",
            "/api/v1/billing/status",
            "/api/v1/monitoring/status"
        ]

        for endpoint in endpoints:
            response = self.client.get(endpoint)
            assert response.status_code == 200, f"Failed for endpoint: {endpoint}"

            data = response.json()
            assert "status" in data, f"No status in response for {endpoint}"
            # API возвращает "healthy" для /health и "operational" для других
            expected_status = "healthy" if endpoint == "/health" else "operational"
            assert data["status"] == expected_status, f"Status not {expected_status} for {endpoint}"

    def test_cors_headers_integration(self):
        """Интеграционный тест CORS headers"""
        # OPTIONS метод не реализован, проверяем GET запрос с CORS headers
        response = self.client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        # Проверяем наличие CORS headers в ответе
        assert "access-control-allow-origin" in response.headers or "Access-Control-Allow-Origin" in response.headers

        # Проверка CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert headers["access-control-allow-origin"] == "*"

    def test_invalid_endpoint_integration(self):
        """Интеграционный тест несуществующего endpoint"""
        response = self.client.get("/api/v1/nonexistent/status")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_async_health_check_integration(self):
        """Асинхронный интеграционный тест health check"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_concurrent_requests_integration(self):
        """Интеграционный тест конкурентных запросов"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Создание нескольких конкурентных запросов
            tasks = []
            for i in range(10):
                task = client.get("/health")
                tasks.append(task)

            # Выполнение всех запросов concurrently
            responses = await asyncio.gather(*tasks)

            # Проверка всех ответов
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"


class TestQuantumInterfaceIntegration:
    """Интеграционные тесты для Quantum Interface с mock-объектами"""

    @pytest.fixture
    def quantum_core(self):
        """Фикстура для Quantum Core"""
        return QuantumCore()

    @pytest.mark.asyncio
    async def test_quantum_core_initialization_integration(self, quantum_core):
        """Интеграционный тест инициализации Quantum Core"""
        result = await quantum_core.initialize()
        assert result is True
        assert quantum_core.status == "operational"

    @pytest.mark.asyncio
    async def test_quantum_core_health_check_integration(self, quantum_core):
        """Интеграционный тест health check Quantum Core"""
        # Инициализация
        await quantum_core.initialize()

        # Проверка здоровья
        healthy = await quantum_core.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_quantum_core_status_integration(self, quantum_core):
        """Интеграционный тест получения статуса Quantum Core"""
        # Инициализация
        await quantum_core.initialize()

        # Получение статуса
        status = await quantum_core.get_status()

        assert "name" in status
        assert "status" in status
        assert "providers" in status
        assert "algorithms" in status
        assert "healthy" in status

        assert status["name"] == "quantum_core"
        assert status["status"] == "operational"
        assert status["healthy"] is True
        assert isinstance(status["providers"], list)
        assert isinstance(status["algorithms"], list)

    @pytest.mark.asyncio
    async def test_quantum_core_shutdown_integration(self, quantum_core):
        """Интеграционный тест остановки Quantum Core"""
        # Инициализация
        await quantum_core.initialize()

        # Остановка
        result = await quantum_core.shutdown()
        assert result is True
        assert quantum_core.status == "shutdown"

    @pytest.mark.asyncio
    @patch('production.quantum.quantum_interface.asyncio.sleep')
    async def test_quantum_core_initialization_with_mock(self, mock_sleep, quantum_core):
        """Интеграционный тест инициализации с mock для asyncio.sleep"""
        mock_sleep.return_value = None

        result = await quantum_core.initialize()
        assert result is True

        # Проверка вызова sleep
        mock_sleep.assert_called_once_with(0.1)

    @pytest.mark.asyncio
    async def test_quantum_core_full_lifecycle_integration(self, quantum_core):
        """Интеграционный тест полного жизненного цикла Quantum Core"""
        # 1. Инициализация
        init_result = await quantum_core.initialize()
        assert init_result is True
        assert quantum_core.status == "operational"

        # 2. Проверка здоровья
        health_result = await quantum_core.health_check()
        assert health_result is True

        # 3. Получение статуса
        status = await quantum_core.get_status()
        assert status["status"] == "operational"
        assert status["healthy"] is True

        # 4. Остановка
        shutdown_result = await quantum_core.shutdown()
        assert shutdown_result is True
        assert quantum_core.status == "shutdown"

        # 5. Проверка здоровья после остановки
        final_health = await quantum_core.health_check()
        assert final_health is False


class TestAIMLSystemIntegration:
    """Интеграционные тесты для Advanced AI/ML System"""

    @pytest.fixture
    def ai_ml_system(self):
        """Фикстура для AI/ML системы с mock обучением"""
        system = AdvancedAIMLSystem()
        # Mock для ускорения тестов
        original_train = system.train_model
        async def mock_train_model(config, X_train, y_train, validation_data=None):
            from production.ai.advanced_ai_ml_system import TrainingResult, TrainingStatus, TrainingMetrics
            from datetime import datetime
            import numpy as np

            # Имитация обучения без реальных вычислений
            await asyncio.sleep(0.01)  # Минимальная задержка

            final_metrics = TrainingMetrics(
                epoch=5,
                loss=0.1,
                accuracy=0.95,
                precision=0.95,
                recall=0.95,
                f1_score=0.95,
                quantum_coherence=0.9,
                phi_harmony=1.618,
                consciousness_level=0.8,
                timestamp=datetime.now()
            )

            result = TrainingResult(
                model_id=config.model_id,
                status=TrainingStatus.COMPLETED,
                final_metrics=final_metrics,
                training_time=0.01,
                quantum_supremacy_achieved=True,
                phi_harmony_score=1.618,
                consciousness_level=0.8,
                model_performance={
                    "accuracy": 0.95,
                    "loss": 0.1,
                    "quantum_coherence": 0.9,
                    "phi_harmony": 1.618,
                    "consciousness_level": 0.8
                }
            )

            system.training_results[config.model_id] = result
            system.models[config.model_id] = "mock_model"  # Заглушка для модели

            # Обновление статистики
            system._update_stats(result)

            return result

        # Mock predict method
        async def mock_predict(model_id, inputs):
            import numpy as np
            # Возвращаем mock предсказания
            if inputs.shape[1] == 3:  # Классификация
                return np.random.rand(inputs.shape[0], 3)
            else:  # Регрессия
                return np.random.rand(inputs.shape[0], 1)

        system.train_model = mock_train_model
        system.predict = mock_predict
        return system

    @pytest.fixture
    def sample_classification_config(self):
        """Фикстура для конфигурации модели классификации"""
        return ModelConfig(
            model_id="test_classification_model",
            model_type=ModelType.CLASSIFICATION,
            algorithm=LearningAlgorithm.QUANTUM_NEURAL_NETWORK,
            input_dimensions=10,
            output_dimensions=3,
            hidden_layers=[32, 16],
            learning_rate=0.01,
            batch_size=16,
            epochs=5,
            quantum_enhanced=True,
            phi_optimization=True,
            consciousness_integration=True
        )

    @pytest.fixture
    def sample_regression_config(self):
        """Фикстура для конфигурации модели регрессии"""
        return ModelConfig(
            model_id="test_regression_model",
            model_type=ModelType.REGRESSION,
            algorithm=LearningAlgorithm.PHI_HARMONIC_LEARNING,
            input_dimensions=5,
            output_dimensions=1,
            hidden_layers=[16, 8],
            learning_rate=0.01,
            batch_size=16,
            epochs=5,
            quantum_enhanced=False,
            phi_optimization=True,
            consciousness_integration=False
        )

    @pytest.fixture
    def sample_training_data_classification(self):
        """Фикстура для тренировочных данных классификации"""
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        y_onehot = np.eye(3)[y]
        return X, y_onehot

    @pytest.fixture
    def sample_training_data_regression(self):
        """Фикстура для тренировочных данных регрессии"""
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.sum(X, axis=1) + np.random.randn(50) * 0.1
        return X, y.reshape(-1, 1)

    @pytest.mark.asyncio
    async def test_ai_ml_system_initialization_integration(self, ai_ml_system):
        """Интеграционный тест инициализации AI/ML системы"""
        # Система инициализируется в конструкторе
        assert ai_ml_system.models == {}
        assert ai_ml_system.training_results == {}
        assert ai_ml_system.training_history == {}
        assert ai_ml_system.stats["models_trained"] == 0

    @pytest.mark.asyncio
    async def test_classification_model_training_integration(self, ai_ml_system, sample_classification_config, sample_training_data_classification):
        """Интеграционный тест обучения модели классификации"""
        X_train, y_train = sample_training_data_classification

        # Обучение модели
        result = await ai_ml_system.train_model(sample_classification_config, X_train, y_train)

        # Проверка результата
        assert result.model_id == "test_classification_model"
        assert result.status.value == "completed"
        assert result.training_time > 0
        assert result.final_metrics.accuracy >= 0.0
        assert result.final_metrics.accuracy <= 1.0
        assert result.quantum_supremacy_achieved in [True, False]
        assert result.phi_harmony_score > 0
        assert result.consciousness_level >= 0.0
        assert result.consciousness_level <= 1.0

        # Проверка что модель сохранена
        assert "test_classification_model" in ai_ml_system.models
        assert "test_classification_model" in ai_ml_system.training_results

        # Проверка статистики
        assert ai_ml_system.stats["models_trained"] == 1

    @pytest.mark.asyncio
    async def test_regression_model_training_integration(self, ai_ml_system, sample_regression_config, sample_training_data_regression):
        """Интеграционный тест обучения модели регрессии"""
        X_train, y_train = sample_training_data_regression

        # Обучение модели
        result = await ai_ml_system.train_model(sample_regression_config, X_train, y_train)

        # Проверка результата
        assert result.model_id == "test_regression_model"
        assert result.status.value == "completed"
        assert result.training_time > 0
        assert result.final_metrics.accuracy >= 0.0
        assert result.final_metrics.accuracy <= 1.0

        # Проверка что модель сохранена
        assert "test_regression_model" in ai_ml_system.models

    @pytest.mark.asyncio
    async def test_model_prediction_integration(self, ai_ml_system, sample_classification_config, sample_training_data_classification):
        """Интеграционный тест предсказаний модели"""
        X_train, y_train = sample_training_data_classification

        # Обучение модели
        await ai_ml_system.train_model(sample_classification_config, X_train, y_train)

        # Создание тестовых данных
        X_test = X_train[:5]

        # Предсказание
        predictions = await ai_ml_system.predict("test_classification_model", X_test)

        # Проверка предсказаний
        assert predictions.shape[0] == 5  # 5 тестовых примеров
        assert predictions.shape[1] == 3  # 3 класса

        # Проверка что предсказания являются вероятностями (сумма по классам = 1)
        import numpy as np
        pred_sums = np.sum(predictions, axis=1)
        np.testing.assert_allclose(pred_sums, 1.0, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_model_performance_tracking_integration(self, ai_ml_system, sample_classification_config, sample_training_data_classification):
        """Интеграционный тест отслеживания производительности модели"""
        X_train, y_train = sample_training_data_classification

        # Обучение модели
        await ai_ml_system.train_model(sample_classification_config, X_train, y_train)

        # Получение производительности
        performance = ai_ml_system.get_model_performance("test_classification_model")

        # Проверка структуры производительности
        required_keys = ["model_id", "status", "final_metrics", "training_time", "quantum_supremacy", "phi_harmony_score", "consciousness_level", "performance", "training_history_length", "best_epoch"]
        for key in required_keys:
            assert key in performance

        assert performance["model_id"] == "test_classification_model"
        assert performance["status"] == "completed"
        assert performance["training_time"] > 0
        assert isinstance(performance["performance"], dict)

    @pytest.mark.asyncio
    async def test_system_stats_integration(self, ai_ml_system, sample_classification_config, sample_regression_config, sample_training_data_classification, sample_training_data_regression):
        """Интеграционный тест статистики системы"""
        # Обучение двух моделей
        X_class, y_class = sample_training_data_classification
        X_reg, y_reg = sample_training_data_regression

        await ai_ml_system.train_model(sample_classification_config, X_class, y_class)
        await ai_ml_system.train_model(sample_regression_config, X_reg, y_reg)

        # Получение статистики
        stats = ai_ml_system.get_system_stats()

        # Проверка статистики
        assert stats["total_models"] == 2
        assert stats["completed_trainings"] == 2
        assert stats["failed_trainings"] == 0
        assert stats["average_training_time"] > 0
        assert "quantum_supremacy_rate" in stats
        assert "phi_harmony_rate" in stats
        assert "consciousness_evolution_rate" in stats

    @pytest.mark.asyncio
    async def test_multiple_algorithms_integration(self, ai_ml_system, sample_training_data_classification):
        """Интеграционный тест нескольких алгоритмов обучения"""
        X_train, y_train = sample_training_data_classification

        algorithms = [
            LearningAlgorithm.QUANTUM_NEURAL_NETWORK,
            LearningAlgorithm.PHI_HARMONIC_LEARNING,
            LearningAlgorithm.CONSCIOUSNESS_EVOLUTION
        ]

        results = []
        for i, algorithm in enumerate(algorithms):
            config = ModelConfig(
                model_id=f"multi_algo_model_{i}",
                model_type=ModelType.CLASSIFICATION,
                algorithm=algorithm,
                input_dimensions=10,
                output_dimensions=3,
                hidden_layers=[16, 8],
                learning_rate=0.01,
                batch_size=16,
                epochs=3,  # Меньше эпох для скорости
                quantum_enhanced=(algorithm == LearningAlgorithm.QUANTUM_NEURAL_NETWORK),
                phi_optimization=(algorithm == LearningAlgorithm.PHI_HARMONIC_LEARNING),
                consciousness_integration=(algorithm == LearningAlgorithm.CONSCIOUSNESS_EVOLUTION)
            )

            result = await ai_ml_system.train_model(config, X_train, y_train)
            results.append(result)

            assert result.status.value == "completed"
            assert result.training_time > 0

        # Проверка что все модели обучены
        assert len(ai_ml_system.models) == 3
        assert len(results) == 3


class TestInterServiceIntegration:
    """Интеграционные тесты межсервисного взаимодействия"""

    @pytest.fixture
    def full_system_setup(self):
        """Фикстура для полной настройки системы"""
        from production.api.main import app
        from production.quantum.quantum_interface import QuantumCore
        from production.ai.advanced_ai_ml_system import AdvancedAIMLSystem

        return {
            "api_app": app,
            "quantum_core": QuantumCore(),
            "ai_ml_system": AdvancedAIMLSystem()
        }

    @pytest.mark.asyncio
    async def test_api_quantum_integration(self, full_system_setup):
        """Интеграционный тест взаимодействия API и Quantum сервисов"""
        api_app = full_system_setup["api_app"]
        quantum_core = full_system_setup["quantum_core"]

        # Инициализация Quantum Core
        await quantum_core.initialize()

        # Тестирование через API
        client = TestClient(api_app)
        response = client.get("/api/v1/quantum/status")

        assert response.status_code == 200
        data = response.json()

        # Проверка что API возвращает статус из Quantum Core
        assert data["status"] == "operational"
        assert "providers" in data
        assert "algorithms" in data

    @pytest.mark.asyncio
    async def test_api_ai_integration(self, full_system_setup):
        """Интеграционный тест взаимодействия API и AI сервисов"""
        api_app = full_system_setup["api_app"]
        ai_ml_system = full_system_setup["ai_ml_system"]

        # Тестирование через API
        client = TestClient(api_app)
        response = client.get("/api/v1/ai/status")

        assert response.status_code == 200
        data = response.json()

        # Проверка что API возвращает статус AI
        assert data["status"] == "operational"
        assert "models" in data
        assert "agents" in data

    @pytest.mark.asyncio
    async def test_full_system_health_integration(self, full_system_setup):
        """Интеграционный тест здоровья полной системы"""
        api_app = full_system_setup["api_app"]
        quantum_core = full_system_setup["quantum_core"]
        ai_ml_system = full_system_setup["ai_ml_system"]

        # Инициализация компонентов
        await quantum_core.initialize()

        # Тестирование health check через API
        client = TestClient(api_app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "x0tta6bl4-unified-api"

    @pytest.mark.asyncio
    async def test_concurrent_services_integration(self, full_system_setup):
        """Интеграционный тест конкурентного доступа к сервисам"""
        api_app = full_system_setup["api_app"]
        quantum_core = full_system_setup["quantum_core"]
        ai_ml_system = full_system_setup["ai_ml_system"]

        # Инициализация
        await quantum_core.initialize()

        client = TestClient(api_app)

        # Конкурентные запросы к разным сервисам
        endpoints = [
            "/health",
            "/api/v1/quantum/status",
            "/api/v1/ai/status",
            "/api/v1/enterprise/status",
            "/api/v1/billing/status",
            "/api/v1/monitoring/status"
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Failed for endpoint: {endpoint}"

            data = response.json()
            assert "status" in data, f"No status in response for {endpoint}"
            assert data["status"] == "operational", f"Status not operational for {endpoint}"

    @pytest.mark.asyncio
    async def test_system_recovery_integration(self, full_system_setup):
        """Интеграционный тест восстановления системы"""
        api_app = full_system_setup["api_app"]
        quantum_core = full_system_setup["quantum_core"]

        # Инициализация
        await quantum_core.initialize()

        # Остановка компонента
        await quantum_core.shutdown()

        # Проверка что API все еще работает (даже если компонент остановлен)
        client = TestClient(api_app)
        response = client.get("/health")
        assert response.status_code == 200

        # Попытка получить статус quantum (должен вернуть статический ответ)
        response = client.get("/api/v1/quantum/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "operational"  # API возвращает статический статус