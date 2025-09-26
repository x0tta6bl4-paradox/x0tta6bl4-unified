"""
Client-Specific Monitoring Dashboards для x0tta6bl4 Enterprise

Этот модуль предоставляет автоматизированную генерацию и управление monitoring dashboards
для Fortune 500 клиентов с поддержкой quantum metrics, SLA monitoring и enterprise alerting.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DashboardPanel:
    """Панель dashboard"""
    title: str
    type: str  # 'graph', 'singlestat', 'table', etc.
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]
    options: Optional[Dict[str, Any]] = None


@dataclass
class DashboardConfig:
    """Конфигурация dashboard"""
    client_id: str
    client_name: str
    panels: List[DashboardPanel]
    tags: List[str]
    refresh: str = "30s"
    time_range: Dict[str, str] = None

    def __post_init__(self):
        if self.time_range is None:
            self.time_range = {
                "from": "now-1h",
                "to": "now"
            }


class ClientMonitoringDashboard:
    """
    Генератор client-specific monitoring dashboards
    """

    def __init__(self):
        self.base_dashboard_template = self._load_base_template()

    def _load_base_template(self) -> Dict[str, Any]:
        """Загружает базовый шаблон dashboard"""
        return {
            "dashboard": {
                "id": None,
                "title": "",
                "tags": [],
                "timezone": "UTC",
                "panels": [],
                "time": {"from": "now-1h", "to": "now"},
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 0,
                "links": []
            }
        }

    def create_fortune500_dashboard(self, client_id: str, client_name: str) -> Dict[str, Any]:
        """
        Создает полный dashboard для Fortune 500 клиента

        Args:
            client_id: ID клиента
            client_name: Название клиента

        Returns:
            JSON конфигурация dashboard
        """
        config = DashboardConfig(
            client_id=client_id,
            client_name=client_name,
            panels=[],
            tags=["x0tta6bl4", "fortune500", client_id]
        )

        # Добавляем стандартные панели
        config.panels.extend(self._create_system_health_panels(client_id))
        config.panels.extend(self._create_quantum_performance_panels(client_id))
        config.panels.extend(self._create_ai_ml_performance_panels(client_id))
        config.panels.extend(self._create_sla_compliance_panels(client_id))
        config.panels.extend(self._create_security_monitoring_panels(client_id))
        config.panels.extend(self._create_business_metrics_panels(client_id))

        return self._generate_dashboard_json(config)

    def _create_system_health_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели системного здоровья"""
        return [
            DashboardPanel(
                title="Service Uptime",
                type="stat",
                targets=[{
                    "expr": f"up{{client=\"{client_id}\"}}",
                    "legendFormat": "Uptime"
                }],
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 0},
                options={
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto"
                }
            ),
            DashboardPanel(
                title="System Resources",
                type="timeseries",
                targets=[
                    {
                        "expr": f"100 - (avg by(instance) (irate(node_cpu_seconds_total{{mode=\"idle\",client=\"{client_id}\"}}[5m])) * 100)",
                        "legendFormat": "CPU Usage %"
                    },
                    {
                        "expr": f"(1 - (node_memory_MemAvailable_bytes{{client=\"{client_id}\"}} / node_memory_MemTotal_bytes{{client=\"{client_id}\"}})) * 100",
                        "legendFormat": "Memory Usage %"
                    },
                    {
                        "expr": f"(1 - (node_filesystem_avail_bytes{{client=\"{client_id}\"}} / node_filesystem_size_bytes{{client=\"{client_id}\"}})) * 100",
                        "legendFormat": "Disk Usage %"
                    }
                ],
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 0}
            )
        ]

    def _create_quantum_performance_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели quantum performance"""
        return [
            DashboardPanel(
                title="Quantum Fidelity",
                type="gauge",
                targets=[{
                    "expr": f"quantum_fidelity{{client=\"{client_id}\"}}",
                    "legendFormat": "Fidelity"
                }],
                grid_pos={"h": 8, "w": 8, "x": 0, "y": 8},
                options={
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto"
                }
            ),
            DashboardPanel(
                title="Quantum Gate Errors",
                type="timeseries",
                targets=[{
                    "expr": f"quantum_gate_errors_total{{client=\"{client_id}\"}}",
                    "legendFormat": "Gate Errors"
                }],
                grid_pos={"h": 8, "w": 8, "x": 8, "y": 8}
            ),
            DashboardPanel(
                title="Entanglement Degradation",
                type="timeseries",
                targets=[{
                    "expr": f"quantum_entanglement_degradation{{client=\"{client_id}\"}}",
                    "legendFormat": "Entanglement Loss"
                }],
                grid_pos={"h": 8, "w": 8, "x": 16, "y": 8}
            )
        ]

    def _create_ai_ml_performance_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели AI/ML performance"""
        return [
            DashboardPanel(
                title="AI Model Accuracy",
                type="gauge",
                targets=[{
                    "expr": f"ai_model_accuracy{{client=\"{client_id}\"}}",
                    "legendFormat": "Accuracy"
                }],
                grid_pos={"h": 8, "w": 8, "x": 0, "y": 16}
            ),
            DashboardPanel(
                title="AI Inference Latency",
                type="timeseries",
                targets=[{
                    "expr": f"ai_inference_latency_seconds{{client=\"{client_id}\"}}",
                    "legendFormat": "Latency (s)"
                }],
                grid_pos={"h": 8, "w": 8, "x": 8, "y": 16}
            ),
            DashboardPanel(
                title="ML Training Progress",
                type="bargauge",
                targets=[{
                    "expr": f"ml_training_progress{{client=\"{client_id}\"}}",
                    "legendFormat": "Training Progress"
                }],
                grid_pos={"h": 8, "w": 8, "x": 16, "y": 16}
            )
        ]

    def _create_sla_compliance_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели SLA compliance"""
        return [
            DashboardPanel(
                title="SLA Uptime (99.99%)",
                type="stat",
                targets=[{
                    "expr": f"(1 - (rate(http_requests_total{{status=~\"5..\",client=\"{client_id}\"}}[30d]) / rate(http_requests_total{{client=\"{client_id}\"}}[30d]))) * 100",
                    "legendFormat": "Uptime %"
                }],
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 24},
                options={
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    },
                    "orientation": "auto",
                    "textMode": "auto",
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto"
                }
            ),
            DashboardPanel(
                title="Response Time SLA",
                type="timeseries",
                targets=[{
                    "expr": f"http_request_duration_seconds{{quantile=\"0.95\",client=\"{client_id}\"}}",
                    "legendFormat": "P95 Response Time"
                }],
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 24}
            )
        ]

    def _create_security_monitoring_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели security monitoring"""
        return [
            DashboardPanel(
                title="Security Incidents",
                type="stat",
                targets=[{
                    "expr": f"security_incidents_total{{client=\"{client_id}\"}}",
                    "legendFormat": "Incidents"
                }],
                grid_pos={"h": 8, "w": 8, "x": 0, "y": 32}
            ),
            DashboardPanel(
                title="Failed Authentication Attempts",
                type="timeseries",
                targets=[{
                    "expr": f"auth_failures_total{{client=\"{client_id}\"}}",
                    "legendFormat": "Failed Auth"
                }],
                grid_pos={"h": 8, "w": 8, "x": 8, "y": 32}
            ),
            DashboardPanel(
                title="Data Encryption Status",
                type="table",
                targets=[{
                    "expr": f"encryption_status{{client=\"{client_id}\"}}",
                    "legendFormat": "Encryption Status"
                }],
                grid_pos={"h": 8, "w": 8, "x": 16, "y": 32}
            )
        ]

    def _create_business_metrics_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели business metrics"""
        return [
            DashboardPanel(
                title="API Requests per Minute",
                type="timeseries",
                targets=[{
                    "expr": f"rate(http_requests_total{{client=\"{client_id}\"}}[5m]) * 60",
                    "legendFormat": "Requests/min"
                }],
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 40}
            ),
            DashboardPanel(
                title="Data Processing Volume",
                type="timeseries",
                targets=[{
                    "expr": f"rate(data_processed_bytes_total{{client=\"{client_id}\"}}[5m])",
                    "legendFormat": "Data Volume (bytes/sec)"
                }],
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 40}
            )
        ]

    def _generate_dashboard_json(self, config: DashboardConfig) -> Dict[str, Any]:
        """Генерирует JSON конфигурацию dashboard"""
        dashboard = self.base_dashboard_template.copy()

        dashboard["dashboard"]["title"] = f"x0tta6bl4 - {config.client_name} Enterprise Dashboard"
        dashboard["dashboard"]["tags"] = config.tags
        dashboard["dashboard"]["refresh"] = config.refresh
        dashboard["dashboard"]["time"] = config.time_range

        # Добавляем переменные шаблона
        dashboard["dashboard"]["templating"]["list"] = [
            {
                "type": "query",
                "name": "client",
                "query": f"label_values(client)",
                "current": {
                    "selected": True,
                    "text": config.client_id,
                    "value": config.client_id
                },
                "label": "Client",
                "hide": 0,
                "includeAll": False,
                "multi": False,
                "options": []
            }
        ]

        # Конвертируем панели в формат Grafana
        grafana_panels = []
        for i, panel in enumerate(config.panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": panel.type,
                "gridPos": panel.grid_pos,
                "targets": panel.targets
            }

            if panel.options:
                grafana_panel.update(panel.options)

            # Добавляем специфичные настройки для типов панелей
            if panel.type == "timeseries":
                grafana_panel["fieldConfig"] = {
                    "defaults": {
                        "custom": {
                            "drawStyle": "line",
                            "lineInterpolation": "linear",
                            "barAlignment": 0,
                            "lineWidth": 1,
                            "fillOpacity": 10,
                            "gradientMode": "none",
                            "spanNulls": False,
                            "showPoints": "never",
                            "pointSize": 5,
                            "stacking": {
                                "mode": "none",
                                "group": "A"
                            },
                            "axisPlacement": "auto",
                            "axisLabel": "",
                            "scaleDistribution": {
                                "type": "linear",
                                "log": 2
                            },
                            "hideFrom": {
                                "vis": False,
                                "legend": False,
                                "tooltip": False
                            },
                            "thresholdsStyle": {
                                "mode": "off"
                            }
                        },
                        "color": {
                            "mode": "palette-classic"
                        },
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {
                                    "color": "green",
                                    "value": None
                                },
                                {
                                    "color": "red",
                                    "value": 80
                                }
                            ]
                        },
                        "unit": "none"
                    },
                    "overrides": []
                }

            grafana_panels.append(grafana_panel)

        dashboard["dashboard"]["panels"] = grafana_panels

        return dashboard

    def create_alert_dashboard(self, client_id: str, client_name: str) -> Dict[str, Any]:
        """
        Создает dashboard для monitoring alerts

        Args:
            client_id: ID клиента
            client_name: Название клиента

        Returns:
            JSON конфигурация alert dashboard
        """
        config = DashboardConfig(
            client_id=client_id,
            client_name=client_name,
            panels=[],
            tags=["x0tta6bl4", "alerts", "fortune500", client_id]
        )

        # Alert panels
        config.panels.extend(self._create_alert_panels(client_id))

        return self._generate_dashboard_json(config)

    def _create_alert_panels(self, client_id: str) -> List[DashboardPanel]:
        """Создает панели для alerts"""
        return [
            DashboardPanel(
                title="Active Alerts",
                type="table",
                targets=[{
                    "expr": f"ALERTS{{client=\"{client_id}\"}}",
                    "legendFormat": "Active Alerts"
                }],
                grid_pos={"h": 12, "w": 24, "x": 0, "y": 0}
            ),
            DashboardPanel(
                title="Alert Rate (Last 24h)",
                type="timeseries",
                targets=[{
                    "expr": f"rate(alerts_total{{client=\"{client_id}\"}}[5m]) * 300",
                    "legendFormat": "Alerts/hour"
                }],
                grid_pos={"h": 8, "w": 12, "x": 0, "y": 12}
            ),
            DashboardPanel(
                title="Alert Resolution Time",
                type="histogram",
                targets=[{
                    "expr": f"alert_resolution_time_seconds{{client=\"{client_id}\"}}",
                    "legendFormat": "Resolution Time"
                }],
                grid_pos={"h": 8, "w": 12, "x": 12, "y": 12}
            )
        ]

    def export_dashboard_to_file(self, dashboard_json: Dict[str, Any], output_path: str):
        """Экспортирует dashboard в JSON файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Dashboard exported to {output_path}")

    def create_client_dashboard_package(self, client_id: str, client_name: str, output_dir: str):
        """
        Создает полный пакет dashboards для клиента

        Args:
            client_id: ID клиента
            client_name: Название клиента
            output_dir: Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Main dashboard
        main_dashboard = self.create_fortune500_dashboard(client_id, client_name)
        self.export_dashboard_to_file(
            main_dashboard,
            os.path.join(output_dir, f"{client_id}_main_dashboard.json")
        )

        # Alert dashboard
        alert_dashboard = self.create_alert_dashboard(client_id, client_name)
        self.export_dashboard_to_file(
            alert_dashboard,
            os.path.join(output_dir, f"{client_id}_alert_dashboard.json")
        )

        # Prometheus alerting rules
        alerting_rules = self._generate_prometheus_alerting_rules(client_id, client_name)
        with open(os.path.join(output_dir, f"{client_id}_alerting_rules.yml"), 'w') as f:
            f.write(alerting_rules)

        logger.info(f"Dashboard package created for client {client_name} in {output_dir}")

    def _generate_prometheus_alerting_rules(self, client_id: str, client_name: str) -> str:
        """Генерирует Prometheus alerting rules для клиента"""
        return f"""groups:
  - name: x0tta6bl4_{client_id}_alerts
    rules:
    # System Health Alerts
    - alert: {client_name}ServiceDown
      expr: up{{client="{client_id}"}} == 0
      for: 5m
      labels:
        severity: critical
        client: {client_id}
        service: x0tta6bl4
      annotations:
        summary: "Service {client_name} is down"
        description: "x0tta6bl4 service for client {client_name} has been down for more than 5 minutes"

    - alert: {client_name}HighCPUUsage
      expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{{mode="idle",client="{client_id}"}}[5m])) * 100) > 90
      for: 10m
      labels:
        severity: warning
        client: {client_id}
        type: system
      annotations:
        summary: "High CPU usage on {client_name}"
        description: "CPU usage is above 90% for more than 10 minutes"

    - alert: {client_name}HighMemoryUsage
      expr: (1 - (node_memory_MemAvailable_bytes{{client="{client_id}"}} / node_memory_MemTotal_bytes{{client="{client_id}"}})) * 100 > 90
      for: 10m
      labels:
        severity: warning
        client: {client_id}
        type: system
      annotations:
        summary: "High memory usage on {client_name}"
        description: "Memory usage is above 90% for more than 10 minutes"

    # Quantum Performance Alerts
    - alert: {client_name}QuantumFidelityDegraded
      expr: quantum_fidelity{{client="{client_id}"}} < 0.95
      for: 5m
      labels:
        severity: critical
        client: {client_id}
        type: quantum
      annotations:
        summary: "Quantum fidelity degraded for {client_name}"
        description: "Quantum fidelity has dropped below 95% for more than 5 minutes"

    - alert: {client_name}HighQuantumGateErrors
      expr: rate(quantum_gate_errors_total{{client="{client_id}"}}[5m]) > 0.01
      for: 5m
      labels:
        severity: warning
        client: {client_id}
        type: quantum
      annotations:
        summary: "High quantum gate error rate for {client_name}"
        description: "Quantum gate error rate is above 1% for more than 5 minutes"

    # AI/ML Performance Alerts
    - alert: {client_name}AIModelAccuracyDegraded
      expr: ai_model_accuracy{{client="{client_id}"}} < 0.90
      for: 10m
      labels:
        severity: warning
        client: {client_id}
        type: ai
      annotations:
        summary: "AI model accuracy degraded for {client_name}"
        description: "AI model accuracy has dropped below 90% for more than 10 minutes"

    - alert: {client_name}HighAILatency
      expr: ai_inference_latency_seconds{{client="{client_id}"}} > 1.0
      for: 5m
      labels:
        severity: warning
        client: {client_id}
        type: ai
      annotations:
        summary: "High AI inference latency for {client_name}"
        description: "AI inference latency is above 1 second for more than 5 minutes"

    # SLA Compliance Alerts
    - alert: {client_name}SLABreach
      expr: (1 - (rate(http_requests_total{{status=~\"5..\",client=\"{client_id}\"}}[30d]) / rate(http_requests_total{{client=\"{client_id}\"}}[30d]))) * 100 < 99.99
      for: 1h
      labels:
        severity: critical
        client: {client_id}
        type: sla
      annotations:
        summary: "SLA breach for {client_name}"
        description: "Monthly uptime SLA of 99.99% has been breached"

    - alert: {client_name}HighErrorRate
      expr: rate(http_requests_total{{status=~\"5..\",client=\"{client_id}\"}}[5m]) / rate(http_requests_total{{client=\"{client_id}\"}}[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
        client: {client_id}
        type: api
      annotations:
        summary: "High error rate for {client_name}"
        description: "HTTP error rate is above 5% for more than 5 minutes"

    # Security Alerts
    - alert: {client_name}SecurityIncident
      expr: increase(security_incidents_total{{client="{client_id}"}}[5m]) > 0
      labels:
        severity: critical
        client: {client_id}
        type: security
      annotations:
        summary: "Security incident detected for {client_name}"
        description: "A security incident has been detected"

    - alert: {client_name}FailedAuthentications
      expr: rate(auth_failures_total{{client="{client_id}"}}[5m]) > 10
      for: 5m
      labels:
        severity: warning
        client: {client_id}
        type: security
      annotations:
        summary: "High rate of failed authentications for {client_name}"
        description: "Failed authentication rate is above 10 per minute for more than 5 minutes"
"""


# Utility functions
def create_client_monitoring_setup(client_id: str, client_name: str, output_dir: str = "./monitoring"):
    """
    Создает полный monitoring setup для клиента

    Args:
        client_id: ID клиента
        client_name: Название клиента
        output_dir: Директория для сохранения
    """
    dashboard_generator = ClientMonitoringDashboard()
    dashboard_generator.create_client_dashboard_package(client_id, client_name, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate client monitoring dashboards")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--client-name", required=True, help="Client name")
    parser.add_argument("--output-dir", default="./monitoring", help="Output directory")

    args = parser.parse_args()

    create_client_monitoring_setup(args.client_id, args.client_name, args.output_dir)

    print(f"✅ Monitoring dashboards generated for {args.client_name}")
    print(f"📁 Files saved in {args.output_dir}")