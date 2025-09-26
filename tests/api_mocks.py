"""
Enhanced API Mocks with realistic responses, error scenarios, and performance characteristics
"""

import random
import time
import json
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock


class RealisticAPIMock:
    """Base class for realistic API mocks with performance and error simulation"""

    def __init__(self, base_latency: float = 0.01, error_rate: float = 0.05):
        self.base_latency = base_latency
        self.error_rate = error_rate

    async def simulate_latency(self, multiplier: float = 1.0):
        """Simulate realistic API latency"""
        latency = self.base_latency * multiplier * random.uniform(0.8, 1.5)
        await asyncio.sleep(latency)

    def should_error(self) -> bool:
        """Determine if this request should result in an error"""
        return random.random() < self.error_rate


class QuantumAPIMock(RealisticAPIMock):
    """Realistic Quantum API mock responses"""

    def __init__(self):
        super().__init__(base_latency=0.05, error_rate=0.08)  # Quantum APIs are slower and less reliable

    async def get_status_response(self) -> Dict[str, Any]:
        """Mock quantum status endpoint"""
        await self.simulate_latency(1.2)

        if self.should_error():
            return {
                "status_code": 503,
                "error": "Quantum hardware coherence lost",
                "retry_after": random.randint(5, 30)
            }

        return {
            "status_code": 200,
            "response": {
                "name": "quantum_core",
                "status": "operational",
                "active_provider": random.choice(["ibm", "google", "xanadu"]),
                "providers": {
                    "ibm": {
                        "available": random.random() < 0.9,
                        "healthy": random.random() < 0.85,
                        "queue_depth": random.randint(0, 50)
                    },
                    "google": {
                        "available": random.random() < 0.8,
                        "healthy": random.random() < 0.75,
                        "coherence_time": random.uniform(10, 100)
                    },
                    "xanadu": {
                        "available": random.random() < 0.7,
                        "healthy": random.random() < 0.8,
                        "photon_efficiency": random.uniform(0.1, 0.4)
                    }
                },
                "algorithms": ["vqe", "qaoa", "grover", "shor"],
                "system_health": random.uniform(0.7, 0.98),
                "quantum_advantage": random.uniform(1.5, 5.0)
            }
        }

    async def run_algorithm_response(self, algorithm: str) -> Dict[str, Any]:
        """Mock quantum algorithm execution"""
        algorithm_latencies = {
            "vqe": 2.0,
            "qaoa": 3.5,
            "grover": 1.2,
            "shor": 8.0
        }

        await self.simulate_latency(algorithm_latencies.get(algorithm, 1.0))

        if self.should_error():
            errors = [
                "Gate error rate exceeded threshold",
                "Quantum decoherence detected",
                "Hardware calibration failed",
                "Qubit connectivity lost"
            ]
            return {
                "status_code": 500,
                "error": random.choice(errors),
                "algorithm": algorithm
            }

        # Realistic algorithm results
        if algorithm == "vqe":
            result = {
                "algorithm": "vqe",
                "eigenvalue": -1.8 + random.gauss(0, 0.2),
                "optimal_parameters": [random.gauss(0.5, 0.3) for _ in range(4)],
                "convergence_iterations": random.randint(50, 200),
                "quantum_circuit_depth": random.randint(20, 100)
            }
        elif algorithm == "qaoa":
            result = {
                "algorithm": "qaoa",
                "eigenvalue": -2.1 + random.gauss(0, 0.3),
                "layers": random.randint(2, 6),
                "parameter_count": random.randint(8, 24),
                "approximation_ratio": random.uniform(0.8, 0.98)
            }
        elif algorithm == "grover":
            result = {
                "algorithm": "grover",
                "found_solution": random.random() < 0.75,
                "iterations": random.randint(5, 15),
                "oracle_calls": random.randint(10, 30),
                "success_probability": random.uniform(0.6, 0.95)
            }
        else:  # shor
            result = {
                "algorithm": "shor",
                "factors_found": random.random() < 0.6,
                "qubits_used": random.randint(10, 50),
                "period_found": random.randint(2, 100),
                "classical_verification": random.random() < 0.8
            }

        return {
            "status_code": 200,
            "response": result,
            "execution_time": random.uniform(0.1, 10.0),
            "resource_usage": {
                "qubits": random.randint(2, 50),
                "gates": random.randint(100, 10000),
                "coherence_time": random.uniform(1, 100)
            }
        }


class AIAPIMock(RealisticAPIMock):
    """Realistic AI API mock responses"""

    def __init__(self):
        super().__init__(base_latency=0.02, error_rate=0.03)

    async def get_status_response(self) -> Dict[str, Any]:
        """Mock AI status endpoint"""
        await self.simulate_latency(0.8)

        if self.should_error():
            return {
                "status_code": 503,
                "error": "AI consciousness synchronization failed"
            }

        return {
            "status_code": 200,
            "response": {
                "name": "ai_system",
                "status": "operational",
                "models": {
                    "quantum_neural_net": {
                        "accuracy": random.uniform(0.85, 0.98),
                        "consciousness_level": random.uniform(0.7, 0.95),
                        "training_status": "completed"
                    },
                    "phi_harmonic_learner": {
                        "phi_ratio": random.uniform(1.5, 2.1),
                        "harmonic_convergence": random.uniform(0.8, 0.97),
                        "resonance_score": random.uniform(0.75, 0.95)
                    }
                },
                "agents": {
                    "ai_engineer": {
                        "status": "active",
                        "tasks_completed": random.randint(10, 100),
                        "success_rate": random.uniform(0.8, 0.98)
                    }
                },
                "system_metrics": {
                    "inference_latency": random.uniform(0.01, 0.1),
                    "throughput": random.uniform(50, 200),
                    "memory_usage": random.uniform(0.3, 0.8)
                }
            }
        }

    async def predict_response(self, model_id: str, input_size: int) -> Dict[str, Any]:
        """Mock AI prediction endpoint"""
        await self.simulate_latency(1.5)

        if self.should_error():
            return {
                "status_code": 500,
                "error": "Model inference failed - gradient explosion detected"
            }

        # Realistic prediction results based on model type
        if "classification" in model_id:
            predictions = []
            for _ in range(input_size):
                # Multi-class probabilities
                probs = [random.random() for _ in range(3)]
                total = sum(probs)
                normalized = [p/total for p in probs]
                predictions.append(normalized)
        elif "regression" in model_id:
            predictions = [[random.gauss(0.5, 0.2)] for _ in range(input_size)]
        else:
            predictions = [[random.uniform(0, 1) for _ in range(10)] for _ in range(input_size)]

        return {
            "status_code": 200,
            "response": {
                "model_id": model_id,
                "predictions": predictions,
                "confidence_scores": [random.uniform(0.6, 0.98) for _ in range(input_size)],
                "inference_time": random.uniform(0.01, 0.2),
                "quantum_enhanced": random.random() < 0.8,
                "consciousness_influence": random.uniform(0.1, 0.4)
            }
        }


class BillingAPIMock(RealisticAPIMock):
    """Realistic Billing API mock responses"""

    def __init__(self):
        super().__init__(base_latency=0.03, error_rate=0.02)

    async def get_status_response(self) -> Dict[str, Any]:
        """Mock billing status endpoint"""
        await self.simulate_latency(1.0)

        if self.should_error():
            return {
                "status_code": 503,
                "error": "Billing database connection timeout"
            }

        return {
            "status_code": 200,
            "response": {
                "name": "billing_system",
                "status": "operational",
                "providers": {
                    "stripe": {
                        "connected": random.random() < 0.95,
                        "healthy": random.random() < 0.9,
                        "pending_charges": random.randint(0, 100)
                    },
                    "paypal": {
                        "connected": random.random() < 0.9,
                        "healthy": random.random() < 0.85,
                        "api_calls_today": random.randint(100, 1000)
                    }
                },
                "features": ["subscription_management", "usage_tracking", "payment_processing"],
                "metrics": {
                    "monthly_revenue": random.uniform(50000, 200000),
                    "active_subscriptions": random.randint(500, 2000),
                    "failed_payments": random.randint(0, 50)
                }
            }
        }


class MonitoringAPIMock(RealisticAPIMock):
    """Realistic Monitoring API mock responses"""

    def __init__(self):
        super().__init__(base_latency=0.01, error_rate=0.01)

    async def get_metrics_response(self) -> Dict[str, Any]:
        """Mock monitoring metrics endpoint"""
        await self.simulate_latency(0.5)

        if self.should_error():
            return {
                "status_code": 503,
                "error": "Metrics collection service unavailable"
            }

        return {
            "status_code": 200,
            "response": {
                "timestamp": time.time(),
                "system_metrics": {
                    "cpu_usage": random.uniform(10, 80),
                    "memory_usage": random.uniform(20, 90),
                    "disk_usage": random.uniform(30, 85),
                    "network_io": random.uniform(50, 500)
                },
                "quantum_metrics": {
                    "coherence_time": random.uniform(5, 50),
                    "gate_fidelity": random.uniform(0.85, 0.98),
                    "error_rate": random.uniform(0.001, 0.01),
                    "qubit_count": random.randint(10, 100)
                },
                "ai_metrics": {
                    "inference_latency": random.uniform(0.01, 0.2),
                    "model_accuracy": random.uniform(0.8, 0.98),
                    "consciousness_level": random.uniform(0.6, 0.95),
                    "training_progress": random.uniform(0, 1)
                },
                "business_metrics": {
                    "active_users": random.randint(1000, 10000),
                    "api_calls": random.randint(10000, 100000),
                    "revenue": random.uniform(10000, 100000)
                }
            }
        }


# Factory function for creating API mocks
def create_api_mock(service: str) -> RealisticAPIMock:
    """Factory function to create appropriate API mock"""
    mocks = {
        "quantum": QuantumAPIMock,
        "ai": AIAPIMock,
        "billing": BillingAPIMock,
        "monitoring": MonitoringAPIMock
    }

    mock_class = mocks.get(service)
    if mock_class:
        return mock_class()
    else:
        return RealisticAPIMock()  # Default mock


# Convenience functions for common API mocking scenarios
async def mock_successful_response(service: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Generate a successful API response"""
    mock = create_api_mock(service)

    if service == "quantum" and "status" in endpoint:
        return await mock.get_status_response()
    elif service == "ai" and "status" in endpoint:
        return await mock.get_status_response()
    elif service == "billing" and "status" in endpoint:
        return await mock.get_status_response()
    elif service == "monitoring" and "metrics" in endpoint:
        return await mock.get_metrics_response()
    else:
        await mock.simulate_latency()
        return {
            "status_code": 200,
            "response": {"message": f"Mock {service} {endpoint} response"}
        }


async def mock_error_response(service: str, error_type: str = "random") -> Dict[str, Any]:
    """Generate an error API response"""
    mock = create_api_mock(service)
    await mock.simulate_latency(2.0)  # Errors take longer

    error_messages = {
        "quantum": ["Quantum coherence lost", "Gate error threshold exceeded", "Hardware failure"],
        "ai": ["Model inference failed", "Consciousness overload", "Training divergence"],
        "billing": ["Payment processing failed", "Database timeout", "Rate limit exceeded"],
        "monitoring": ["Metrics collection failed", "Service unavailable", "Data corruption"]
    }

    messages = error_messages.get(service, ["Service error"])
    error_msg = random.choice(messages) if error_type == "random" else error_type

    return {
        "status_code": random.choice([500, 502, 503, 504]),
        "error": error_msg,
        "service": service
    }


async def mock_degraded_response(service: str, degradation_level: float = 0.5) -> Dict[str, Any]:
    """Generate a degraded API response (slow but successful)"""
    mock = create_api_mock(service)
    await mock.simulate_latency(3.0 * degradation_level)  # Much slower

    return {
        "status_code": 206,  # Partial content - degraded
        "response": {
            "message": f"{service} service operating in degraded mode",
            "degradation_level": degradation_level,
            "estimated_recovery": f"{random.randint(5, 60)} minutes"
        }
    }