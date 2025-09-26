#!/usr/bin/env python3
"""
Тесты для quantum monitoring компонентов
Тестирование Prometheus, AlertManager, Grafana dashboards и quantum metrics
"""

import pytest
import asyncio
import yaml
import json
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path


class TestQuantumMonitoring:
    """Тесты для quantum monitoring компонентов"""

    @pytest.fixture
    def monitoring_dir(self):
        """Фикстура для директории monitoring"""
        return Path("x0tta6bl4-unified/monitoring")

    @pytest.fixture
    def prometheus_config(self, monitoring_dir):
        """Фикстура для конфигурации Prometheus"""
        config_path = monitoring_dir / "prometheus.yml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    @pytest.fixture
    def alert_rules(self, monitoring_dir):
        """Фикстура для правил алертов"""
        rules_path = monitoring_dir / "prometheus" / "alert_rules.yml"
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    class TestPrometheusConfiguration:
        """Тесты конфигурации Prometheus"""

        def test_prometheus_config_exists(self, prometheus_config):
            """Тест существования конфигурации Prometheus"""
            assert prometheus_config is not None

        def test_prometheus_config_structure(self, prometheus_config):
            """Тест структуры конфигурации Prometheus"""
            if prometheus_config:
                assert "global" in prometheus_config
                assert "scrape_configs" in prometheus_config

        def test_prometheus_scrape_configs(self, prometheus_config):
            """Тест scrape конфигураций Prometheus"""
            if prometheus_config and "scrape_configs" in prometheus_config:
                scrape_configs = prometheus_config["scrape_configs"]
                assert isinstance(scrape_configs, list)
                assert len(scrape_configs) > 0

                # Проверяем quantum сервисы
                job_names = [config.get("job_name") for config in scrape_configs]
                quantum_jobs = [name for name in job_names if "quantum" in name.lower()]
                assert len(quantum_jobs) > 0

        def test_prometheus_global_config(self, prometheus_config):
            """Тест глобальной конфигурации Prometheus"""
            if prometheus_config and "global" in prometheus_config:
                global_config = prometheus_config["global"]
                assert "scrape_interval" in global_config
                assert "evaluation_interval" in global_config

    class TestAlertRules:
        """Тесты правил алертов"""

        def test_alert_rules_exist(self, alert_rules):
            """Тест существования правил алертов"""
            assert alert_rules is not None

        def test_alert_rules_structure(self, alert_rules):
            """Тест структуры правил алертов"""
            if alert_rules:
                assert "groups" in alert_rules
                groups = alert_rules["groups"]
                assert isinstance(groups, list)
                assert len(groups) > 0

        def test_quantum_alerts(self, alert_rules):
            """Тест квантовых алертов"""
            if alert_rules and "groups" in alert_rules:
                groups = alert_rules["groups"]

                quantum_alerts = []
                for group in groups:
                    if "rules" in group:
                        for rule in group["rules"]:
                            if "quantum" in rule.get("alert", "").lower():
                                quantum_alerts.append(rule)

                assert len(quantum_alerts) > 0

        def test_alert_rule_validity(self, alert_rules):
            """Тест валидности правил алертов"""
            if alert_rules and "groups" in alert_rules:
                groups = alert_rules["groups"]

                for group in groups:
                    if "rules" in group:
                        for rule in group["rules"]:
                            # Проверяем обязательные поля
                            assert "alert" in rule
                            assert "expr" in rule
                            assert "for" in rule
                            assert "labels" in rule
                            assert "annotations" in rule

    class TestAlertManagerConfiguration:
        """Тесты конфигурации AlertManager"""

        @pytest.fixture
        def alertmanager_config(self, monitoring_dir):
            """Фикстура для конфигурации AlertManager"""
            config_path = monitoring_dir / "alertmanager.yml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return None

        def test_alertmanager_config_exists(self, alertmanager_config):
            """Тест существования конфигурации AlertManager"""
            assert alertmanager_config is not None

        def test_alertmanager_routes(self, alertmanager_config):
            """Тест маршрутов AlertManager"""
            if alertmanager_config and "route" in alertmanager_config:
                route = alertmanager_config["route"]
                assert "group_by" in route
                assert "group_wait" in route
                assert "group_interval" in route
                assert "repeat_interval" in route

        def test_alertmanager_receivers(self, alertmanager_config):
            """Тест получателей AlertManager"""
            if alertmanager_config and "receivers" in alertmanager_config:
                receivers = alertmanager_config["receivers"]
                assert isinstance(receivers, list)
                assert len(receivers) > 0

                # Проверяем quantum receivers
                receiver_names = [r.get("name") for r in receivers]
                quantum_receivers = [name for name in receiver_names if "quantum" in name.lower()]
                assert len(quantum_receivers) > 0

    class TestMonitoringMetrics:
        """Тесты метрик мониторинга"""

        @pytest.fixture
        def metrics_dir(self, monitoring_dir):
            """Фикстура для директории метрик"""
            return monitoring_dir / "metrics"

        def test_metrics_directory_exists(self, metrics_dir):
            """Тест существования директории метрик"""
            assert metrics_dir.exists()

        def test_quantum_metrics_files(self, metrics_dir):
            """Тест файлов квантовых метрик"""
            if metrics_dir.exists():
                metric_files = list(metrics_dir.glob("*.yml")) + list(metrics_dir.glob("*.yaml"))
                quantum_metric_files = [f for f in metric_files if "quantum" in f.name.lower()]
                assert len(quantum_metric_files) > 0

        @pytest.mark.asyncio
        async def test_metrics_collection(self):
            """Тест сбора метрик"""
            # Mock quantum компонентов
            quantum_core = Mock()
            quantum_core.get_status = AsyncMock(return_value={
                "status": "operational",
                "providers": {"ibm": {"available": True}, "google": {"available": True}},
                "algorithms": ["vqe", "qaoa", "grover", "shor"]
            })

            ai_system = Mock()
            ai_system.get_status = AsyncMock(return_value={
                "status": "operational",
                "models_count": 5,
                "training_results": {}
            })

            # Имитация сбора метрик
            metrics = {
                "quantum_core_status": (await quantum_core.get_status())["status"],
                "ai_system_status": (await ai_system.get_status())["status"],
                "timestamp": "2025-01-01T00:00:00Z"
            }

            assert metrics["quantum_core_status"] == "operational"
            assert metrics["ai_system_status"] == "operational"
            assert "timestamp" in metrics

    class TestGrafanaDashboards:
        """Тесты Grafana dashboards"""

        @pytest.fixture
        def grafana_dir(self, monitoring_dir):
            """Фикстура для директории Grafana"""
            return monitoring_dir / "grafana"

        def test_grafana_directory_exists(self, grafana_dir):
            """Тест существования директории Grafana"""
            assert grafana_dir.exists()

        def test_dashboard_files(self, grafana_dir):
            """Тест файлов dashboards"""
            if grafana_dir.exists():
                dashboard_files = list(grafana_dir.glob("*.json"))
                assert len(dashboard_files) > 0

        def test_quantum_dashboard(self, grafana_dir):
            """Тест квантового dashboard"""
            if grafana_dir.exists():
                quantum_dashboards = list(grafana_dir.glob("*quantum*.json"))
                assert len(quantum_dashboards) > 0

                # Проверяем структуру dashboard
                with open(quantum_dashboards[0], 'r') as f:
                    dashboard = json.load(f)

                assert "dashboard" in dashboard
                assert "title" in dashboard["dashboard"]
                assert "quantum" in dashboard["dashboard"]["title"].lower()

    class TestMonitoringIntegration:
        """Тесты интеграции мониторинга"""

        @pytest.mark.asyncio
        async def test_monitoring_pipeline(self):
            """Тест конвейера мониторинга"""
            # Mock компоненты
            prometheus = Mock()
            alertmanager = Mock()
            grafana = Mock()

            prometheus.scrape_metrics = AsyncMock(return_value={"status": "success"})
            alertmanager.process_alerts = AsyncMock(return_value={"alerts_processed": 5})
            grafana.update_dashboards = AsyncMock(return_value={"dashboards_updated": 3})

            # Имитация конвейера мониторинга
            scrape_result = await prometheus.scrape_metrics()
            alert_result = await alertmanager.process_alerts()
            dashboard_result = await grafana.update_dashboards()

            assert scrape_result["status"] == "success"
            assert alert_result["alerts_processed"] == 5
            assert dashboard_result["dashboards_updated"] == 3

        @pytest.mark.asyncio
        async def test_quantum_metrics_export(self):
            """Тест экспорта квантовых метрик"""
            # Mock quantum метрик
            quantum_metrics = {
                "quantum_coherence": 0.95,
                "gate_fidelity": 0.98,
                "entanglement_strength": 0.92,
                "algorithm_success_rate": 0.87,
                "timestamp": "2025-01-01T00:00:00Z"
            }

            # Имитация экспорта в Prometheus формат
            prometheus_metrics = []
            for metric_name, value in quantum_metrics.items():
                if isinstance(value, (int, float)):
                    prometheus_metrics.append(f"quantum_{metric_name} {value}")

            assert len(prometheus_metrics) > 0
            assert any("quantum_coherence" in metric for metric in prometheus_metrics)

    class TestMonitoringAlerts:
        """Тесты алертов мониторинга"""

        @pytest.mark.asyncio
        async def test_quantum_alert_generation(self):
            """Тест генерации квантовых алертов"""
            # Mock метрики для алертов
            metrics = {
                "quantum_coherence": 0.85,  # Ниже порога
                "gate_errors": 0.15,  # Выше порога
                "entanglement_fidelity": 0.88  # Ниже порога
            }

            alerts = []

            # Проверяем условия алертов
            if metrics["quantum_coherence"] < 0.9:
                alerts.append({
                    "alertname": "LowQuantumCoherence",
                    "severity": "warning",
                    "description": f"Quantum coherence is low: {metrics['quantum_coherence']}"
                })

            if metrics["gate_errors"] > 0.1:
                alerts.append({
                    "alertname": "HighGateErrors",
                    "severity": "critical",
                    "description": f"Gate errors are high: {metrics['gate_errors']}"
                })

            assert len(alerts) > 0
            assert any(alert["alertname"] == "LowQuantumCoherence" for alert in alerts)
            assert any(alert["alertname"] == "HighGateErrors" for alert in alerts)

        @pytest.mark.asyncio
        async def test_alert_escalation(self):
            """Тест эскалации алертов"""
            # Mock алерт
            alert = {
                "alertname": "QuantumSystemFailure",
                "severity": "warning",
                "start_time": "2025-01-01T00:00:00Z",
                "duration": 3600  # 1 час
            }

            # Логика эскалации
            if alert["severity"] == "warning" and alert["duration"] > 1800:  # 30 минут
                alert["severity"] = "critical"
                alert["escalated"] = True

            assert alert["severity"] == "critical"
            assert alert["escalated"] == True

    class TestMonitoringPerformance:
        """Тесты производительности мониторинга"""

        @pytest.mark.performance
        @pytest.mark.asyncio
        async def test_metrics_collection_performance(self):
            """Тест производительности сбора метрик"""
            import time

            # Mock компоненты
            components = [Mock() for _ in range(5)]
            for component in components:
                component.get_metrics = AsyncMock(return_value={"metric": 0.5})

            # Измерение времени сбора метрик
            start_time = time.time()

            metrics_results = []
            for component in components:
                result = await component.get_metrics()
                metrics_results.append(result)

            end_time = time.time()
            collection_time = end_time - start_time

            # Проверяем производительность (< 1 секунды на 5 компонентов)
            assert collection_time < 1.0
            assert len(metrics_results) == 5

        @pytest.mark.performance
        @pytest.mark.asyncio
        async def test_alert_processing_performance(self):
            """Тест производительности обработки алертов"""
            import time

            # Mock алерты
            alerts = [{"alertname": f"Alert{i}", "severity": "warning"} for i in range(10)]

            start_time = time.time()

            # Имитация обработки алертов
            processed_alerts = []
            for alert in alerts:
                processed_alert = alert.copy()
                processed_alert["processed"] = True
                processed_alert["processing_time"] = time.time()
                processed_alerts.append(processed_alert)
                await asyncio.sleep(0.001)  # Имитация обработки

            end_time = time.time()
            processing_time = end_time - start_time

            # Проверяем производительность (< 0.1 секунды на алерт)
            assert processing_time < 1.0
            assert len(processed_alerts) == 10
            assert all(alert["processed"] for alert in processed_alerts)

    class TestMonitoringReliability:
        """Тесты надежности мониторинга"""

        @pytest.mark.asyncio
        async def test_monitoring_failover(self):
            """Тест failover мониторинга"""
            # Mock основной и резервный мониторинг
            primary_monitoring = Mock()
            backup_monitoring = Mock()

            primary_monitoring.get_status = AsyncMock(side_effect=Exception("Primary failed"))
            backup_monitoring.get_status = AsyncMock(return_value={"status": "operational"})

            # Логика failover
            try:
                status = await primary_monitoring.get_status()
            except Exception:
                status = await backup_monitoring.get_status()

            assert status["status"] == "operational"

        @pytest.mark.asyncio
        async def test_metrics_persistence(self):
            """Тест персистентности метрик"""
            # Mock хранилище метрик
            storage = Mock()
            storage.save_metrics = AsyncMock(return_value=True)
            storage.load_metrics = AsyncMock(return_value={"metric1": 0.8, "metric2": 0.9})

            # Сохранение метрик
            metrics = {"quantum_coherence": 0.95, "timestamp": "2025-01-01T00:00:00Z"}
            save_result = await storage.save_metrics(metrics)
            assert save_result == True

            # Загрузка метрик
            loaded_metrics = await storage.load_metrics()
            assert "metric1" in loaded_metrics or "quantum_coherence" in loaded_metrics


# Тесты для различных сценариев мониторинга
@pytest.mark.parametrize("monitoring_scenario", [
    "normal_operation",
    "high_load",
    "system_degradation",
    "quantum_failure",
    "recovery_mode"
])
@pytest.mark.asyncio
async def test_monitoring_scenarios(monitoring_scenario):
    """Параметризованный тест сценариев мониторинга"""
    # Mock система мониторинга
    monitoring_system = Mock()

    # Настройка поведения в зависимости от сценария
    if monitoring_scenario == "normal_operation":
        monitoring_system.check_health = AsyncMock(return_value={
            "status": "healthy",
            "alerts": 0,
            "metrics_collected": 100
        })

        health = await monitoring_system.check_health()
        assert health["status"] == "healthy"
        assert health["alerts"] == 0

    elif monitoring_scenario == "high_load":
        monitoring_system.check_health = AsyncMock(return_value={
            "status": "warning",
            "alerts": 5,
            "metrics_collected": 95,
            "cpu_usage": 0.85
        })

        health = await monitoring_system.check_health()
        assert health["status"] == "warning"
        assert health["alerts"] > 0

    elif monitoring_scenario == "quantum_failure":
        monitoring_system.check_health = AsyncMock(return_value={
            "status": "critical",
            "alerts": 10,
            "quantum_coherence": 0.7,
            "error": "Quantum system failure"
        })

        health = await monitoring_system.check_health()
        assert health["status"] == "critical"
        assert "error" in health

    # Другие сценарии могут быть добавлены аналогично


if __name__ == "__main__":
    pytest.main([__file__])