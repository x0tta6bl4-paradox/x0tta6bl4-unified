"""
Resilience Metrics Collector для x0tta6bl4 Unified
Сбор и экспорт метрик resilience в Prometheus формате
"""

import time
import threading
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
import psutil
import random


class ResilienceMetricsCollector:
    """Коллектор метрик resilience для real-time мониторинга"""

    def __init__(self):
        self.metrics = {
            # Service health metrics
            "service_requests_total": 0,
            "service_failures_total": 0,
            "service_recovery_time_seconds": 0,
            "service_health_status": {},  # per service

            # Circuit breaker metrics
            "circuit_breaker_state": {},  # per service

            # Retry metrics
            "retry_attempts_total": 0,
            "retry_success_total": 0,

            # Quantum-specific metrics
            "quantum_noise_factor": 1.0,
            "quantum_gate_errors_total": 0,
            "quantum_entanglement_fidelity": 1.0,
            "quantum_coherence_time_seconds": 100.0,

            # Chaos engineering metrics
            "chaos_test_active": 0,
            "chaos_test_name": "",

            # Resilience score
            "resilience_score": 1.0,

            # Recovery metrics
            "recovery_in_progress": 0,
            "recovery_attempts_total": 0,
            "recovery_success_total": 0,

            # Istio service mesh metrics
            "istio_requests_total": 0,
            "istio_request_duration_milliseconds": 0,
            "istio_tcp_connections_opened_total": 0,
            "istio_tcp_connections_closed_total": 0,
            "istio_circuit_breaker_open": 0,
            "istio_mtls_connection_total": 0,
            "istio_proxy_up": 1,
            "istio_pilot_up": 1,
            "istio_mixer_up": 1,
        }

        self.service_states = {
            "quantum": 1,
            "ai": 1,
            "enterprise": 1,
            "billing": 1,
            "monitoring": 1
        }

        self.circuit_breakers = {
            "quantum": "closed",
            "ai": "closed",
            "enterprise": "closed",
            "billing": "closed",
            "monitoring": "closed"
        }

        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_service_request(self, service: str, success: bool = True):
        """Запись запроса к сервису"""
        with self._lock:
            self.metrics["service_requests_total"] += 1
            if not success:
                self.metrics["service_failures_total"] += 1
                self.service_states[service] = 0  # Mark as unhealthy

    def record_retry_attempt(self, success: bool = False):
        """Запись попытки retry"""
        with self._lock:
            self.metrics["retry_attempts_total"] += 1
            if success:
                self.metrics["retry_success_total"] += 1

    def update_circuit_breaker(self, service: str, state: str):
        """Обновление состояния circuit breaker"""
        with self._lock:
            self.circuit_breakers[service] = state
            self.metrics["circuit_breaker_state"][service] = 1 if state == "open" else 0

    def record_quantum_metrics(self, noise_factor: float = None, gate_errors: int = None,
                              fidelity: float = None, coherence_time: float = None):
        """Запись quantum-specific метрик"""
        with self._lock:
            if noise_factor is not None:
                self.metrics["quantum_noise_factor"] = noise_factor
            if gate_errors is not None:
                self.metrics["quantum_gate_errors_total"] += gate_errors
            if fidelity is not None:
                self.metrics["quantum_entanglement_fidelity"] = fidelity
            if coherence_time is not None:
                self.metrics["quantum_coherence_time_seconds"] = coherence_time

    def start_recovery(self, service: str):
        """Начало процесса восстановления"""
        with self._lock:
            self.metrics["recovery_in_progress"] = 1
            self.metrics["recovery_attempts_total"] += 1

    def end_recovery(self, service: str, success: bool = True):
        """Завершение процесса восстановления"""
        with self._lock:
            self.metrics["recovery_in_progress"] = 0
            if success:
                self.metrics["recovery_success_total"] += 1
                self.service_states[service] = 1  # Mark as healthy

    def record_recovery_time(self, recovery_time_seconds: float):
        """Запись времени восстановления"""
        with self._lock:
            self.metrics["service_recovery_time_seconds"] = recovery_time_seconds

    def start_chaos_test(self, test_name: str):
        """Начало chaos engineering теста"""
        with self._lock:
            self.metrics["chaos_test_active"] = 1
            self.metrics["chaos_test_name"] = test_name

    def end_chaos_test(self):
        """Завершение chaos engineering теста"""
        with self._lock:
            self.metrics["chaos_test_active"] = 0
            self.metrics["chaos_test_name"] = ""

    def calculate_resilience_score(self) -> float:
        """Расчет общего resilience score"""
        with self._lock:
            total_requests = self.metrics["service_requests_total"]
            total_failures = self.metrics["service_failures_total"]

            if total_requests == 0:
                return 1.0

            failure_rate = total_failures / total_requests

            # Resilience score based on multiple factors
            health_score = sum(self.service_states.values()) / len(self.service_states)
            failure_score = max(0, 1 - failure_rate * 10)  # Penalize high failure rates
            retry_score = min(1.0, self.metrics["retry_success_total"] / max(1, self.metrics["retry_attempts_total"]))

            resilience_score = (health_score * 0.4 + failure_score * 0.4 + retry_score * 0.2)

            self.metrics["resilience_score"] = resilience_score
            return resilience_score

    def get_prometheus_metrics(self) -> str:
        """Экспорт метрик в Prometheus формате"""
        with self._lock:
            lines = []
            timestamp = int(time.time() * 1000)

            # Service metrics
            lines.append(f'# HELP x0tta6bl4_service_requests_total Total number of service requests')
            lines.append(f'# TYPE x0tta6bl4_service_requests_total counter')
            lines.append(f'x0tta6bl4_service_requests_total {self.metrics["service_requests_total"]} {timestamp}')

            lines.append(f'# HELP x0tta6bl4_service_failures_total Total number of service failures')
            lines.append(f'# TYPE x0tta6bl4_service_failures_total counter')
            lines.append(f'x0tta6bl4_service_failures_total {self.metrics["service_failures_total"]} {timestamp}')

            # Service health status per service
            lines.append(f'# HELP x0tta6bl4_service_health_status Service health status (1=healthy, 0=unhealthy)')
            lines.append(f'# TYPE x0tta6bl4_service_health_status gauge')
            for service, status in self.service_states.items():
                lines.append(f'x0tta6bl4_service_health_status{{service="{service}"}} {status} {timestamp}')

            # Circuit breaker states
            lines.append(f'# HELP x0tta6bl4_circuit_breaker_state Circuit breaker state (1=open, 0=closed)')
            lines.append(f'# TYPE x0tta6bl4_circuit_breaker_state gauge')
            for service, state in self.circuit_breakers.items():
                state_value = 1 if state == "open" else 0
                lines.append(f'x0tta6bl4_circuit_breaker_state{{service="{service}",state="{state}"}} {state_value} {timestamp}')

            # Retry metrics
            lines.append(f'# HELP x0tta6bl4_retry_attempts_total Total number of retry attempts')
            lines.append(f'# TYPE x0tta6bl4_retry_attempts_total counter')
            lines.append(f'x0tta6bl4_retry_attempts_total {self.metrics["retry_attempts_total"]} {timestamp}')

            # Quantum metrics
            lines.append(f'# HELP x0tta6bl4_quantum_noise_factor Current quantum noise factor')
            lines.append(f'# TYPE x0tta6bl4_quantum_noise_factor gauge')
            lines.append(f'x0tta6bl4_quantum_noise_factor {self.metrics["quantum_noise_factor"]} {timestamp}')

            lines.append(f'# HELP x0tta6bl4_quantum_gate_errors_total Total quantum gate errors')
            lines.append(f'# TYPE x0tta6bl4_quantum_gate_errors_total counter')
            lines.append(f'x0tta6bl4_quantum_gate_errors_total {self.metrics["quantum_gate_errors_total"]} {timestamp}')

            lines.append(f'# HELP x0tta6bl4_quantum_entanglement_fidelity Current entanglement fidelity')
            lines.append(f'# TYPE x0tta6bl4_quantum_entanglement_fidelity gauge')
            lines.append(f'x0tta6bl4_quantum_entanglement_fidelity {self.metrics["quantum_entanglement_fidelity"]} {timestamp}')

            lines.append(f'# HELP x0tta6bl4_quantum_coherence_time_seconds Current coherence time in seconds')
            lines.append(f'# TYPE x0tta6bl4_quantum_coherence_time_seconds gauge')
            lines.append(f'x0tta6bl4_quantum_coherence_time_seconds {self.metrics["quantum_coherence_time_seconds"]} {timestamp}')

            # Chaos engineering metrics
            lines.append(f'# HELP x0tta6bl4_chaos_test_active Whether chaos test is currently active')
            lines.append(f'# TYPE x0tta6bl4_chaos_test_active gauge')
            lines.append(f'x0tta6bl4_chaos_test_active {self.metrics["chaos_test_active"]} {timestamp}')

            # Resilience score
            lines.append(f'# HELP x0tta6bl4_resilience_score Overall system resilience score (0-1)')
            lines.append(f'# TYPE x0tta6bl4_resilience_score gauge')
            lines.append(f'x0tta6bl4_resilience_score {self.calculate_resilience_score()} {timestamp}')

            # Recovery metrics
            lines.append(f'# HELP x0tta6bl4_recovery_in_progress Whether recovery is currently in progress')
            lines.append(f'# TYPE x0tta6bl4_recovery_in_progress gauge')
            lines.append(f'x0tta6bl4_recovery_in_progress {self.metrics["recovery_in_progress"]} {timestamp}')

            lines.append(f'# HELP x0tta6bl4_service_recovery_time_seconds Time taken for last service recovery')
            lines.append(f'# TYPE x0tta6bl4_service_recovery_time_seconds gauge')
            lines.append(f'x0tta6bl4_service_recovery_time_seconds {self.metrics["service_recovery_time_seconds"]} {timestamp}')

            return '\n'.join(lines)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик для отладки"""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "total_requests": self.metrics["service_requests_total"],
                "total_failures": self.metrics["service_failures_total"],
                "failure_rate": (self.metrics["service_failures_total"] /
                               max(1, self.metrics["service_requests_total"])),
                "service_states": self.service_states.copy(),
                "circuit_breakers": self.circuit_breakers.copy(),
                "quantum_metrics": {
                    "noise_factor": self.metrics["quantum_noise_factor"],
                    "gate_errors": self.metrics["quantum_gate_errors_total"],
                    "fidelity": self.metrics["quantum_entanglement_fidelity"],
                    "coherence_time": self.metrics["quantum_coherence_time_seconds"]
                },
                "resilience_score": self.calculate_resilience_score(),
                "chaos_active": self.metrics["chaos_test_active"] == 1
            }


# Global instance for application-wide metrics collection
_resilience_collector = ResilienceMetricsCollector()


def get_resilience_collector() -> ResilienceMetricsCollector:
    """Получение глобального коллектора метрик resilience"""
    return _resilience_collector


def record_service_request(service: str, success: bool = True):
    """Глобальная функция для записи запросов к сервисам"""
    _resilience_collector.record_service_request(service, success)


def record_retry_attempt(success: bool = False):
    """Глобальная функция для записи retry attempts"""
    _resilience_collector.record_retry_attempt(success)


def update_circuit_breaker(service: str, state: str):
    """Глобальная функция для обновления circuit breaker"""
    _resilience_collector.update_circuit_breaker(service, state)


def record_quantum_metrics(noise_factor: float = None, gate_errors: int = None,
                          fidelity: float = None, coherence_time: float = None):
    """Глобальная функция для записи quantum метрик"""
    _resilience_collector.record_quantum_metrics(noise_factor, gate_errors, fidelity, coherence_time)


def start_recovery(service: str):
    """Глобальная функция для начала восстановления"""
    _resilience_collector.start_recovery(service)


def end_recovery(service: str, success: bool = True):
    """Глобальная функция для завершения восстановления"""
    _resilience_collector.end_recovery(service, success)


def start_chaos_test(test_name: str):
    """Глобальная функция для начала chaos теста"""
    _resilience_collector.start_chaos_test(test_name)


def end_chaos_test():
    """Глобальная функция для завершения chaos теста"""
    _resilience_collector.end_chaos_test()


# Background thread for periodic metrics updates
def _metrics_updater():
    """Фоновая задача для периодического обновления метрик"""
    while True:
        try:
            # Simulate some realistic metric updates
            if random.random() < 0.05:  # 5% chance of simulated failure
                service = random.choice(list(_resilience_collector.service_states.keys()))
                _resilience_collector.record_service_request(service, success=False)

            # Update quantum metrics occasionally
            if random.random() < 0.1:  # 10% chance
                noise_factor = 1.0 + random.uniform(-0.1, 0.2)
                _resilience_collector.record_quantum_metrics(noise_factor=noise_factor)

            time.sleep(5)  # Update every 5 seconds

        except Exception as e:
            print(f"Metrics updater error: {e}")
            time.sleep(10)


# Start background metrics updater
_updater_thread = threading.Thread(target=_metrics_updater, daemon=True)
_updater_thread.start()


if __name__ == "__main__":
    # Demo usage
    collector = get_resilience_collector()

    # Simulate some activity
    for i in range(20):
        service = random.choice(["quantum", "ai", "enterprise", "billing"])
        success = random.random() > 0.1  # 90% success rate
        record_service_request(service, success)

        if not success and random.random() < 0.5:  # 50% of failures trigger retry
            record_retry_attempt(success=random.random() > 0.3)  # 70% retry success

        time.sleep(0.1)

    # Record some quantum metrics
    record_quantum_metrics(noise_factor=1.2, gate_errors=2, fidelity=0.95, coherence_time=85.0)

    # Print metrics summary
    print(json.dumps(collector.get_metrics_summary(), indent=2))

    # Print Prometheus format
    print("\nPrometheus metrics:")
    print(collector.get_prometheus_metrics())