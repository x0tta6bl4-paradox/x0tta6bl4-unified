"""
Enhanced Performance Mocks with realistic latency, throughput, and resource usage simulation
"""

import random
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class PerformanceMetrics:
    """Realistic performance metrics"""
    latency_ms: float
    throughput_ops: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_io_mbps: float
    disk_io_iops: float
    error_rate_percent: float
    timestamp: float


class ResourceUsageSimulator:
    """Simulates realistic resource usage patterns"""

    def __init__(self):
        self.baseline_cpu = random.uniform(5, 15)
        self.baseline_memory = random.uniform(100, 300)
        self.baseline_network = random.uniform(10, 50)
        self.baseline_disk = random.uniform(100, 500)

        # Resource usage patterns
        self.cpu_pattern = self._generate_usage_pattern("cpu")
        self.memory_pattern = self._generate_usage_pattern("memory")
        self.network_pattern = self._generate_usage_pattern("network")
        self.disk_pattern = self._generate_usage_pattern("disk")

    def _generate_usage_pattern(self, resource_type: str) -> Callable[[float], float]:
        """Generate realistic usage pattern for a resource type"""
        if resource_type == "cpu":
            # CPU: sinusoidal with spikes
            return lambda t: 10 + 20 * math.sin(t * 0.1) + random.gauss(0, 5)
        elif resource_type == "memory":
            # Memory: gradual growth with occasional drops
            return lambda t: 200 + 50 * math.sin(t * 0.05) + random.gauss(0, 20)
        elif resource_type == "network":
            # Network: bursty traffic
            return lambda t: 30 + 40 * abs(math.sin(t * 0.2)) + random.gauss(0, 10)
        else:  # disk
            # Disk: steady with occasional bursts
            return lambda t: 300 + 100 * math.sin(t * 0.02) + random.gauss(0, 30)

    def get_current_usage(self, load_factor: float = 1.0) -> Dict[str, float]:
        """Get current resource usage under given load"""
        t = time.time()

        return {
            "cpu_percent": min(100, max(0, self.baseline_cpu + load_factor * self.cpu_pattern(t))),
            "memory_mb": max(50, self.baseline_memory + load_factor * self.memory_pattern(t)),
            "network_mbps": max(0, self.baseline_network + load_factor * self.network_pattern(t)),
            "disk_iops": max(0, self.baseline_disk + load_factor * self.disk_pattern(t))
        }


class LatencySimulator:
    """Simulates realistic latency patterns"""

    def __init__(self):
        self.base_latencies = {
            "quantum_compute": random.uniform(0.1, 0.5),
            "ai_inference": random.uniform(0.02, 0.2),
            "api_call": random.uniform(0.01, 0.1),
            "database_query": random.uniform(0.005, 0.05),
            "cache_lookup": random.uniform(0.001, 0.01),
            "network_request": random.uniform(0.05, 0.3)
        }

        self.latency_history = deque(maxlen=100)
        self.jitter_factor = random.uniform(0.1, 0.3)

    async def simulate_latency(self, operation_type: str, load_factor: float = 1.0) -> float:
        """Simulate latency for a specific operation type"""
        base_latency = self.base_latencies.get(operation_type, 0.01)

        # Add load-dependent latency
        load_latency = base_latency * (1 + load_factor * random.uniform(0.5, 2.0))

        # Add jitter
        jitter = load_latency * self.jitter_factor * random.gauss(0, 1)

        # Add occasional spikes (1% chance)
        if random.random() < 0.01:
            spike_multiplier = random.uniform(5, 20)
            load_latency *= spike_multiplier

        total_latency = max(0.001, load_latency + jitter)

        # Record in history
        self.latency_history.append({
            "operation": operation_type,
            "latency": total_latency,
            "timestamp": time.time()
        })

        # Actually wait for the simulated latency
        await asyncio.sleep(total_latency)

        return total_latency

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics"""
        if not self.latency_history:
            return {"error": "No latency data available"}

        latencies = [entry["latency"] for entry in self.latency_history]

        return {
            "mean_latency": sum(latencies) / len(latencies),
            "median_latency": sorted(latencies)[len(latencies) // 2],
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "sample_count": len(latencies)
        }


class ThroughputSimulator:
    """Simulates realistic throughput patterns"""

    def __init__(self):
        self.baseline_throughput = {
            "quantum_ops": random.uniform(10, 50),
            "ai_inferences": random.uniform(100, 500),
            "api_requests": random.uniform(1000, 5000),
            "database_queries": random.uniform(5000, 20000),
            "cache_operations": random.uniform(10000, 50000)
        }

        self.throughput_history = deque(maxlen=100)
        self.contention_factor = random.uniform(0.7, 0.9)

    async def simulate_throughput(self, operation_type: str, duration: float = 1.0,
                                load_factor: float = 1.0) -> Dict[str, Any]:
        """Simulate throughput for operations over a time period"""
        baseline = self.baseline_throughput.get(operation_type, 100)

        # Calculate effective throughput under load
        effective_throughput = baseline * self.contention_factor / (1 + load_factor)

        # Add variability
        variability = random.uniform(0.8, 1.2)
        effective_throughput *= variability

        # Simulate operations
        operations_completed = 0
        start_time = time.time()

        while time.time() - start_time < duration:
            # Batch processing simulation
            batch_size = min(100, int(effective_throughput * 0.1))
            operations_completed += batch_size

            # Simulate processing time
            processing_time = batch_size / effective_throughput
            await asyncio.sleep(min(processing_time, 0.1))

            # Occasional slowdowns
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.01, 0.05))

        actual_duration = time.time() - start_time
        actual_throughput = operations_completed / actual_duration

        result = {
            "operation_type": operation_type,
            "operations_completed": operations_completed,
            "duration": actual_duration,
            "throughput_ops_per_sec": actual_throughput,
            "efficiency": actual_throughput / baseline
        }

        self.throughput_history.append({
            **result,
            "timestamp": time.time()
        })

        return result

    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get throughput statistics"""
        if not self.throughput_history:
            return {"error": "No throughput data available"}

        throughputs = [entry["throughput_ops_per_sec"] for entry in self.throughput_history]

        return {
            "mean_throughput": sum(throughputs) / len(throughputs),
            "median_throughput": sorted(throughputs)[len(throughputs) // 2],
            "p95_throughput": sorted(throughputs)[int(len(throughputs) * 0.95)],
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
            "sample_count": len(throughputs)
        }


class PerformanceProfile:
    """Represents different performance profiles"""

    PROFILES = {
        "idle": {"load_factor": 0.1, "description": "System at idle"},
        "normal": {"load_factor": 0.5, "description": "Normal operating load"},
        "high": {"load_factor": 0.8, "description": "High load conditions"},
        "overload": {"load_factor": 1.5, "description": "System overload"},
        "stress": {"load_factor": 2.0, "description": "Stress testing conditions"}
    }

    def __init__(self, profile_name: str = "normal"):
        self.profile_name = profile_name
        self.load_factor = self.PROFILES[profile_name]["load_factor"]
        self.description = self.PROFILES[profile_name]["description"]


class ComprehensivePerformanceMock:
    """Comprehensive performance mock combining all aspects"""

    def __init__(self, profile: str = "normal"):
        self.profile = PerformanceProfile(profile)
        self.resource_simulator = ResourceUsageSimulator()
        self.latency_simulator = LatencySimulator()
        self.throughput_simulator = ThroughputSimulator()

        # Performance degradation tracking
        self.degradation_factors = {
            "memory_pressure": 1.0,
            "cpu_contention": 1.0,
            "network_saturation": 1.0,
            "disk_contention": 1.0
        }

    async def simulate_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Simulate a complete operation with all performance aspects"""
        start_time = time.time()

        # Get current resource usage
        resources = self.resource_simulator.get_current_usage(self.profile.load_factor)

        # Apply degradation factors
        effective_load = self.profile.load_factor
        for factor in self.degradation_factors.values():
            effective_load *= factor

        # Simulate latency
        latency = await self.latency_simulator.simulate_latency(operation_type, effective_load)

        # Simulate some processing (throughput)
        if "duration" in kwargs:
            throughput_result = await self.throughput_simulator.simulate_throughput(
                operation_type, kwargs["duration"], effective_load
            )
        else:
            throughput_result = None

        # Update degradation based on resource usage
        self._update_degradation_factors(resources)

        total_time = time.time() - start_time

        return {
            "operation_type": operation_type,
            "total_time": total_time,
            "latency": latency,
            "resources_used": resources,
            "throughput_result": throughput_result,
            "performance_profile": self.profile.profile_name,
            "degradation_factors": self.degradation_factors.copy(),
            "success": random.random() > (0.05 * effective_load)  # Higher load = higher failure rate
        }

    def _update_degradation_factors(self, resources: Dict[str, float]):
        """Update degradation factors based on resource usage"""
        # Memory pressure
        if resources["memory_mb"] > 800:
            self.degradation_factors["memory_pressure"] = 1.5
        elif resources["memory_mb"] > 600:
            self.degradation_factors["memory_pressure"] = 1.2
        else:
            self.degradation_factors["memory_pressure"] = 1.0

        # CPU contention
        if resources["cpu_percent"] > 90:
            self.degradation_factors["cpu_contention"] = 2.0
        elif resources["cpu_percent"] > 70:
            self.degradation_factors["cpu_contention"] = 1.3
        else:
            self.degradation_factors["cpu_contention"] = 1.0

        # Network saturation
        if resources["network_mbps"] > 800:
            self.degradation_factors["network_saturation"] = 1.8
        elif resources["network_mbps"] > 500:
            self.degradation_factors["network_saturation"] = 1.2
        else:
            self.degradation_factors["network_saturation"] = 1.0

        # Disk contention
        if resources["disk_iops"] > 5000:
            self.degradation_factors["disk_contention"] = 1.6
        elif resources["disk_iops"] > 3000:
            self.degradation_factors["disk_contention"] = 1.1
        else:
            self.degradation_factors["disk_contention"] = 1.0

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        resources = self.resource_simulator.get_current_usage(self.profile.load_factor)

        return PerformanceMetrics(
            latency_ms=self.latency_simulator.get_latency_stats().get("mean_latency", 0) * 1000,
            throughput_ops=self.throughput_simulator.get_throughput_stats().get("mean_throughput", 0),
            cpu_usage_percent=resources["cpu_percent"],
            memory_usage_mb=resources["memory_mb"],
            network_io_mbps=resources["network_mbps"],
            disk_io_iops=resources["disk_iops"],
            error_rate_percent=5.0 * self.profile.load_factor,  # Error rate increases with load
            timestamp=time.time()
        )

    async def simulate_load_test(self, duration: float = 60.0, concurrent_users: int = 10) -> Dict[str, Any]:
        """Simulate a comprehensive load test"""
        start_time = time.time()
        results = []

        # Create concurrent operations
        async def user_session(user_id: int):
            session_results = []
            session_start = time.time()

            while time.time() - session_start < duration:
                # Random operation mix
                operations = ["api_call", "database_query", "ai_inference", "quantum_compute"]
                operation = random.choice(operations)

                result = await self.simulate_operation(operation)
                session_results.append(result)

                # Think time between operations
                await asyncio.sleep(random.uniform(0.01, 0.1))

            return {
                "user_id": user_id,
                "operations_completed": len(session_results),
                "session_duration": time.time() - session_start,
                "results": session_results
            }

        # Run concurrent sessions
        tasks = [user_session(i) for i in range(concurrent_users)]
        session_results = await asyncio.gather(*tasks)

        total_operations = sum(r["operations_completed"] for r in session_results)
        total_duration = time.time() - start_time

        return {
            "test_duration": total_duration,
            "concurrent_users": concurrent_users,
            "total_operations": total_operations,
            "overall_throughput": total_operations / total_duration,
            "session_results": session_results,
            "final_performance_metrics": self.get_performance_metrics().__dict__,
            "resource_usage_summary": self.resource_simulator.get_current_usage(self.profile.load_factor)
        }


# Factory functions
def create_performance_mock(profile: str = "normal") -> ComprehensivePerformanceMock:
    """Create a comprehensive performance mock"""
    return ComprehensivePerformanceMock(profile)


async def simulate_realistic_workload(operation_types: List[str], duration: float = 30.0) -> Dict[str, Any]:
    """Simulate a realistic workload mix"""
    mock = create_performance_mock("normal")

    results = []
    start_time = time.time()

    while time.time() - start_time < duration:
        operation = random.choice(operation_types)
        result = await mock.simulate_operation(operation)
        results.append(result)

        # Brief pause between operations
        await asyncio.sleep(random.uniform(0.005, 0.02))

    return {
        "workload_duration": duration,
        "operations_performed": len(results),
        "operation_types": operation_types,
        "results": results,
        "performance_summary": mock.get_performance_metrics().__dict__
    }


async def benchmark_operation_scaling(operation_type: str, user_counts: List[int]) -> Dict[str, Any]:
    """Benchmark how an operation scales with user count"""
    results = {}

    for users in user_counts:
        mock = create_performance_mock("normal")
        load_test_result = await mock.simulate_load_test(duration=10.0, concurrent_users=users)

        results[users] = {
            "throughput": load_test_result["overall_throughput"],
            "latency_stats": mock.latency_simulator.get_latency_stats(),
            "resource_usage": load_test_result["resource_usage_summary"],
            "performance_metrics": load_test_result["final_performance_metrics"]
        }

    return {
        "operation_type": operation_type,
        "scaling_results": results,
        "user_counts_tested": user_counts
    }