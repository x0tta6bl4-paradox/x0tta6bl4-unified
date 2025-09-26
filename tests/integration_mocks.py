"""
Enhanced Integration Mocks with realistic cross-component interactions and failure propagation
"""

import random
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass


@dataclass
class ComponentState:
    """Represents the state of a system component"""
    name: str
    health: float  # 0.0 to 1.0
    dependencies: List[str]
    dependents: List[str]
    failure_probability: float
    recovery_time: float
    last_failure: Optional[float] = None


class SystemStateManager:
    """Manages the state of interconnected system components"""

    def __init__(self):
        self.components = {}
        self.failure_cascade_probability = 0.3
        self.recovery_probability = 0.7

        # Initialize component states
        self._initialize_components()

    def _initialize_components(self):
        """Initialize component dependency graph"""
        self.components = {
            "quantum_core": ComponentState(
                name="quantum_core",
                health=random.uniform(0.8, 1.0),
                dependencies=[],
                dependents=["ai_engine", "quantum_optimizer"],
                failure_probability=0.15,
                recovery_time=random.uniform(30, 120)
            ),
            "ai_engine": ComponentState(
                name="ai_engine",
                health=random.uniform(0.7, 0.95),
                dependencies=["quantum_core"],
                dependents=["api_gateway", "edge_processor"],
                failure_probability=0.10,
                recovery_time=random.uniform(20, 90)
            ),
            "api_gateway": ComponentState(
                name="api_gateway",
                health=random.uniform(0.9, 1.0),
                dependencies=["ai_engine", "billing_service"],
                dependents=["load_balancer"],
                failure_probability=0.05,
                recovery_time=random.uniform(10, 60)
            ),
            "billing_service": ComponentState(
                name="billing_service",
                health=random.uniform(0.85, 0.98),
                dependencies=[],
                dependents=["api_gateway", "monitoring"],
                failure_probability=0.08,
                recovery_time=random.uniform(15, 75)
            ),
            "monitoring": ComponentState(
                name="monitoring",
                health=random.uniform(0.8, 0.97),
                dependencies=["billing_service"],
                dependents=[],
                failure_probability=0.06,
                recovery_time=random.uniform(12, 45)
            ),
            "quantum_optimizer": ComponentState(
                name="quantum_optimizer",
                health=random.uniform(0.75, 0.92),
                dependencies=["quantum_core"],
                dependents=["ai_engine"],
                failure_probability=0.12,
                recovery_time=random.uniform(25, 100)
            ),
            "edge_processor": ComponentState(
                name="edge_processor",
                health=random.uniform(0.7, 0.9),
                dependencies=["ai_engine"],
                dependents=["api_gateway"],
                failure_probability=0.18,
                recovery_time=random.uniform(35, 140)
            ),
            "load_balancer": ComponentState(
                name="load_balancer",
                health=random.uniform(0.88, 0.98),
                dependencies=["api_gateway"],
                dependents=[],
                failure_probability=0.04,
                recovery_time=random.uniform(8, 40)
            )
        }

    def propagate_failure(self, failed_component: str):
        """Propagate failure through the dependency graph"""
        if failed_component not in self.components:
            return

        failed_state = self.components[failed_component]
        failed_state.health = random.uniform(0.0, 0.3)  # Severely degraded
        failed_state.last_failure = time.time()

        # Propagate to dependents
        for dependent in failed_state.dependents:
            if random.random() < self.failure_cascade_probability:
                self.propagate_failure(dependent)

        # Check if dependencies are still healthy
        for dependency in failed_state.dependencies:
            dep_state = self.components[dependency]
            if dep_state.health < 0.5:
                # Dependency failure affects this component
                failed_state.health = min(failed_state.health, dep_state.health * 0.8)

    def attempt_recovery(self, component_name: str):
        """Attempt to recover a failed component"""
        if component_name not in self.components:
            return False

        component = self.components[component_name]

        # Check if enough time has passed for recovery
        if component.last_failure and (time.time() - component.last_failure) < component.recovery_time:
            return False

        # Attempt recovery
        if random.random() < self.recovery_probability:
            component.health = random.uniform(0.7, 1.0)
            component.last_failure = None

            # Check if we can recover dependents
            for dependent in component.dependents:
                dep_state = self.components[dependent]
                if dep_state.health < 0.5 and random.random() < 0.6:
                    self.attempt_recovery(dependent)

            return True

        return False

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_health = sum(c.health for c in self.components.values()) / len(self.components)
        critical_components = [name for name, c in self.components.items() if c.health < 0.5]

        return {
            "overall_health": total_health,
            "critical_components": critical_components,
            "component_states": {name: c.health for name, c in self.components.items()},
            "failure_count": len([c for c in self.components.values() if c.last_failure is not None])
        }


class CrossComponentInteractionMock:
    """Mock for realistic cross-component interactions"""

    def __init__(self):
        self.state_manager = SystemStateManager()
        self.interaction_history = []
        self.latency_multiplier = 1.0

    async def simulate_interaction(self, source: str, target: str, operation: str) -> Dict[str, Any]:
        """Simulate an interaction between two components"""
        await asyncio.sleep(random.uniform(0.01, 0.05) * self.latency_multiplier)

        source_health = self.state_manager.components.get(source, ComponentState("", 1.0, [], [], 0, 0)).health
        target_health = self.state_manager.components.get(target, ComponentState("", 1.0, [], [], 0, 0)).health

        # Calculate interaction success probability
        success_prob = (source_health + target_health) / 2

        # Record interaction
        interaction = {
            "timestamp": time.time(),
            "source": source,
            "target": target,
            "operation": operation,
            "source_health": source_health,
            "target_health": target_health
        }

        if random.random() < success_prob:
            # Successful interaction
            result = {
                "status": "success",
                "latency": random.uniform(0.01, 0.1),
                "data_transfered": random.randint(100, 10000),
                "processing_time": random.uniform(0.005, 0.05)
            }

            # Occasionally trigger cascading effects
            if random.random() < 0.1 and target_health < 0.8:
                self.state_manager.attempt_recovery(target)

        else:
            # Failed interaction
            result = {
                "status": "failed",
                "error": self._generate_error_message(source, target, operation),
                "retry_after": random.randint(1, 10)
            }

            # Trigger failure propagation
            if random.random() < 0.3:
                self.state_manager.propagate_failure(target)

        interaction["result"] = result
        self.interaction_history.append(interaction)

        return result

    def _generate_error_message(self, source: str, target: str, operation: str) -> str:
        """Generate realistic error messages"""
        error_templates = [
            f"{source} failed to communicate with {target} during {operation}",
            f"Timeout in {operation} from {source} to {target}",
            f"Resource exhaustion in {target} during {operation}",
            f"Protocol mismatch between {source} and {target}",
            f"Authentication failure in {operation} request",
            f"Rate limit exceeded for {operation} to {target}",
            f"Network partition affecting {source}->{target} communication"
        ]
        return random.choice(error_templates)

    async def get_component_status(self, component: str) -> Dict[str, Any]:
        """Get status of a specific component"""
        await asyncio.sleep(random.uniform(0.005, 0.02))

        if component not in self.state_manager.components:
            return {"status": "unknown", "error": f"Component {component} not found"}

        state = self.state_manager.components[component]

        # Simulate status check affecting health slightly
        if random.random() < 0.05:
            state.health = max(0.1, state.health - random.uniform(0.01, 0.05))

        return {
            "status": "operational" if state.health > 0.7 else "degraded" if state.health > 0.3 else "failed",
            "health": state.health,
            "dependencies": state.dependencies,
            "dependents": state.dependents,
            "last_failure": state.last_failure,
            "uptime": time.time() - (state.last_failure or time.time())
        }

    async def trigger_system_failure(self, component: str, severity: float = 0.5):
        """Trigger a system failure starting from a specific component"""
        await asyncio.sleep(random.uniform(0.01, 0.03))

        if component in self.state_manager.components:
            self.state_manager.components[component].health = severity
            self.state_manager.propagate_failure(component)

        return {"triggered": True, "affected_components": self.state_manager.get_system_health()["critical_components"]}

    async def simulate_load_scenario(self, duration: float = 10.0, intensity: float = 1.0):
        """Simulate a load scenario affecting multiple components"""
        start_time = time.time()
        interactions = []

        while time.time() - start_time < duration:
            # Random interaction
            source = random.choice(list(self.state_manager.components.keys()))
            target = random.choice(list(self.state_manager.components.keys()))

            if source != target:
                result = await self.simulate_interaction(source, target, f"load_operation_{random.randint(1,100)}")
                interactions.append(result)

                # Adjust latency based on load
                self.latency_multiplier = 1.0 + (intensity - 1.0) * random.uniform(0.5, 2.0)

            await asyncio.sleep(random.uniform(0.01, 0.05))

        return {
            "duration": duration,
            "interactions": len(interactions),
            "success_rate": len([i for i in interactions if i["status"] == "success"]) / len(interactions),
            "final_system_health": self.state_manager.get_system_health()
        }


class FailurePropagationMock:
    """Mock for realistic failure propagation scenarios"""

    def __init__(self):
        self.failure_scenarios = {
            "quantum_cascade": {
                "trigger": "quantum_core",
                "affected": ["ai_engine", "quantum_optimizer", "edge_processor"],
                "recovery_order": ["quantum_core", "quantum_optimizer", "ai_engine", "edge_processor"]
            },
            "network_partition": {
                "trigger": "api_gateway",
                "affected": ["load_balancer", "ai_engine", "billing_service"],
                "recovery_order": ["api_gateway", "billing_service", "ai_engine", "load_balancer"]
            },
            "resource_exhaustion": {
                "trigger": "monitoring",
                "affected": ["billing_service", "ai_engine", "quantum_optimizer"],
                "recovery_order": ["monitoring", "billing_service", "quantum_optimizer", "ai_engine"]
            },
            "security_breach": {
                "trigger": "billing_service",
                "affected": ["api_gateway", "monitoring", "ai_engine"],
                "recovery_order": ["billing_service", "monitoring", "api_gateway", "ai_engine"]
            }
        }

    async def simulate_failure_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Simulate a complete failure scenario"""
        if scenario_name not in self.failure_scenarios:
            return {"error": f"Unknown scenario: {scenario_name}"}

        scenario = self.failure_scenarios[scenario_name]
        results = {"scenario": scenario_name, "phases": []}

        # Failure phase
        results["phases"].append({
            "phase": "failure",
            "trigger": scenario["trigger"],
            "affected": scenario["affected"],
            "timestamp": time.time()
        })

        await asyncio.sleep(random.uniform(0.1, 0.5))  # Failure propagation time

        # Recovery phase
        recovery_results = []
        for component in scenario["recovery_order"]:
            success = random.random() < 0.8  # 80% recovery success rate
            recovery_results.append({
                "component": component,
                "recovered": success,
                "recovery_time": random.uniform(5, 30) if success else None
            })
            if success:
                await asyncio.sleep(random.uniform(0.05, 0.2))  # Recovery time

        results["phases"].append({
            "phase": "recovery",
            "results": recovery_results,
            "timestamp": time.time()
        })

        return results

    async def cascading_failure_test(self, initial_failure: str, max_depth: int = 3) -> Dict[str, Any]:
        """Test cascading failure propagation"""
        affected = [initial_failure]
        propagation_path = [initial_failure]

        for depth in range(max_depth):
            if not affected:
                break

            current_failures = affected.copy()
            affected = []

            for component in current_failures:
                # Find dependents that might be affected
                dependents = self._get_component_dependents(component)
                for dependent in dependents:
                    if random.random() < 0.4:  # 40% chance of cascade
                        if dependent not in propagation_path:
                            affected.append(dependent)
                            propagation_path.append(dependent)

            await asyncio.sleep(random.uniform(0.02, 0.08))  # Propagation delay

        return {
            "initial_failure": initial_failure,
            "propagation_path": propagation_path,
            "max_depth_reached": len(propagation_path) > max_depth,
            "total_affected": len(propagation_path)
        }

    def _get_component_dependents(self, component: str) -> List[str]:
        """Get components that depend on the given component"""
        # Simplified dependency mapping
        dependencies = {
            "quantum_core": ["ai_engine", "quantum_optimizer"],
            "ai_engine": ["api_gateway", "edge_processor"],
            "api_gateway": ["load_balancer"],
            "billing_service": ["api_gateway", "monitoring"],
            "monitoring": [],
            "quantum_optimizer": ["ai_engine"],
            "edge_processor": ["api_gateway"],
            "load_balancer": []
        }
        return dependencies.get(component, [])


# Factory functions for creating integration mocks
def create_cross_component_mock() -> CrossComponentInteractionMock:
    """Create a cross-component interaction mock"""
    return CrossComponentInteractionMock()


def create_failure_propagation_mock() -> FailurePropagationMock:
    """Create a failure propagation mock"""
    return FailurePropagationMock()


async def simulate_system_wide_failure(scenario: str = "random") -> Dict[str, Any]:
    """Simulate a system-wide failure scenario"""
    scenarios = ["quantum_cascade", "network_partition", "resource_exhaustion", "security_breach"]
    if scenario == "random":
        scenario = random.choice(scenarios)

    mock = create_failure_propagation_mock()
    return await mock.simulate_failure_scenario(scenario)


async def test_component_interdependencies() -> Dict[str, Any]:
    """Test component interdependencies under load"""
    mock = create_cross_component_mock()

    # Simulate high-load scenario
    results = await mock.simulate_load_scenario(duration=5.0, intensity=2.0)

    # Analyze interdependencies
    interactions = mock.interaction_history
    dependency_failures = {}

    for interaction in interactions:
        if interaction["result"]["status"] == "failed":
            key = f"{interaction['source']}->{interaction['target']}"
            dependency_failures[key] = dependency_failures.get(key, 0) + 1

    return {
        "load_test_results": results,
        "dependency_failure_analysis": dependency_failures,
        "most_failure_prone_dependency": max(dependency_failures.items(), key=lambda x: x[1]) if dependency_failures else None
    }