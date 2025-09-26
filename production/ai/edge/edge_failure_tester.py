"""
Edge Computing Failure Tests для x0tta6bl4 Unified
Тестирование edge AI компонентов на устойчивость к сбоям
"""

import asyncio
import time
import random
import threading
import psutil
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, patch
import socket
import subprocess


class EdgeFailureSimulator:
    """Симулятор сбоев для edge computing"""

    def __init__(self):
        self.active_failures = set()
        self.failure_history = []
        self.recovery_times = {}

    def simulate_network_failure(self, duration: float = 5.0) -> str:
        """Симуляция сетевого сбоя"""
        failure_id = f"network_{int(time.time())}"
        self.active_failures.add(failure_id)

        def recover():
            time.sleep(duration)
            self.active_failures.discard(failure_id)
            self.recovery_times[failure_id] = time.time()

        thread = threading.Thread(target=recover, daemon=True)
        thread.start()

        self.failure_history.append({
            "type": "network",
            "id": failure_id,
            "start_time": time.time(),
            "duration": duration
        })

        return failure_id

    def simulate_power_failure(self, duration: float = 2.0) -> str:
        """Симуляция сбоя питания"""
        failure_id = f"power_{int(time.time())}"
        self.active_failures.add(failure_id)

        def recover():
            time.sleep(duration)
            self.active_failures.discard(failure_id)
            self.recovery_times[failure_id] = time.time()

        thread = threading.Thread(target=recover, daemon=True)
        thread.start()

        self.failure_history.append({
            "type": "power",
            "id": failure_id,
            "start_time": time.time(),
            "duration": duration
        })

        return failure_id

    def simulate_memory_pressure(self, pressure_level: float = 0.8) -> str:
        """Симуляция давления на память"""
        failure_id = f"memory_{int(time.time())}"
        self.active_failures.add(failure_id)

        # Имитация высокого потребления памяти
        memory_info = psutil.virtual_memory()
        target_usage = memory_info.total * pressure_level

        self.failure_history.append({
            "type": "memory",
            "id": failure_id,
            "start_time": time.time(),
            "target_usage": target_usage,
            "pressure_level": pressure_level
        })

        return failure_id

    def simulate_cpu_overload(self, overload_factor: float = 2.0) -> str:
        """Симуляция перегрузки CPU"""
        failure_id = f"cpu_{int(time.time())}"
        self.active_failures.add(failure_id)

        def cpu_stress():
            end_time = time.time() + 10  # 10 seconds of stress
            while time.time() < end_time and failure_id in self.active_failures:
                # Имитация CPU нагрузки
                for _ in range(10000):
                    _ = random.random() ** 2
            self.active_failures.discard(failure_id)
            self.recovery_times[failure_id] = time.time()

        thread = threading.Thread(target=cpu_stress, daemon=True)
        thread.start()

        self.failure_history.append({
            "type": "cpu",
            "id": failure_id,
            "start_time": time.time(),
            "overload_factor": overload_factor
        })

        return failure_id

    def simulate_disk_failure(self, duration: float = 3.0) -> str:
        """Симуляция сбоя диска"""
        failure_id = f"disk_{int(time.time())}"
        self.active_failures.add(failure_id)

        def recover():
            time.sleep(duration)
            self.active_failures.discard(failure_id)
            self.recovery_times[failure_id] = time.time()

        thread = threading.Thread(target=recover, daemon=True)
        thread.start()

        self.failure_history.append({
            "type": "disk",
            "id": failure_id,
            "start_time": time.time(),
            "duration": duration
        })

        return failure_id

    def is_failure_active(self, failure_type: str = None) -> bool:
        """Проверка активных сбоев"""
        if failure_type:
            return any(f.startswith(f"{failure_type}_") for f in self.active_failures)
        return len(self.active_failures) > 0

    def get_active_failures(self) -> List[str]:
        """Получение списка активных сбоев"""
        return list(self.active_failures)

    def get_failure_stats(self) -> Dict[str, Any]:
        """Получение статистики сбоев"""
        failure_types = {}
        for failure in self.failure_history:
            f_type = failure["type"]
            if f_type not in failure_types:
                failure_types[f_type] = 0
            failure_types[f_type] += 1

        return {
            "total_failures": len(self.failure_history),
            "active_failures": len(self.active_failures),
            "failure_types": failure_types,
            "recovery_times": self.recovery_times
        }


class EdgeDeviceSimulator:
    """Симулятор edge устройства"""

    def __init__(self, device_id: str, capabilities: Dict[str, Any]):
        self.device_id = device_id
        self.capabilities = capabilities
        self.status = "online"
        self.last_heartbeat = time.time()
        self.inference_queue = asyncio.Queue()
        self.failure_simulator = EdgeFailureSimulator()

    async def simulate_inference(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """Симуляция AI inference на edge устройстве"""
        if self.status != "online":
            raise Exception(f"Device {self.device_id} is {self.status}")

        # Проверка на активные сбои
        if self.failure_simulator.is_failure_active("network"):
            raise Exception("Network failure - cannot reach device")

        if self.failure_simulator.is_failure_active("power"):
            self.status = "offline"
            raise Exception("Power failure - device offline")

        if self.failure_simulator.is_failure_active("memory"):
            raise Exception("Memory pressure - inference failed")

        if self.failure_simulator.is_failure_active("cpu"):
            # Имитация замедления из-за CPU перегрузки
            await asyncio.sleep(random.uniform(0.5, 2.0))

        if self.failure_simulator.is_failure_active("disk"):
            raise Exception("Disk failure - cannot load model")

        # Имитация нормального inference
        processing_time = random.uniform(0.1, 0.5)
        await asyncio.sleep(processing_time)

        # Имитация результата
        confidence = random.uniform(0.7, 0.95)
        result = {
            "device_id": self.device_id,
            "inference_result": f"prediction_{random.randint(0, 9)}",
            "confidence": confidence,
            "processing_time": processing_time,
            "timestamp": time.time()
        }

        return result

    def trigger_failure(self, failure_type: str, **kwargs) -> str:
        """Триггер сбоя на устройстве"""
        if failure_type == "network":
            return self.failure_simulator.simulate_network_failure(**kwargs)
        elif failure_type == "power":
            return self.failure_simulator.simulate_power_failure(**kwargs)
        elif failure_type == "memory":
            return self.failure_simulator.simulate_memory_pressure(**kwargs)
        elif failure_type == "cpu":
            return self.failure_simulator.simulate_cpu_overload(**kwargs)
        elif failure_type == "disk":
            return self.failure_simulator.simulate_disk_failure(**kwargs)
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса устройства"""
        return {
            "device_id": self.device_id,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "capabilities": self.capabilities,
            "active_failures": self.failure_simulator.get_active_failures(),
            "failure_stats": self.failure_simulator.get_failure_stats()
        }


class EdgeFailureTester:
    """Тестер edge computing сбоев"""

    def __init__(self):
        self.devices: Dict[str, EdgeDeviceSimulator] = {}
        self.test_results: List[Dict[str, Any]] = []
        self.failure_scenarios = [
            "network_partition",
            "power_outage",
            "memory_exhaustion",
            "cpu_overload",
            "disk_failure",
            "multiple_failures",
            "cascading_failures"
        ]

    def add_device(self, device_id: str, capabilities: Dict[str, Any] = None):
        """Добавление edge устройства"""
        if capabilities is None:
            capabilities = {
                "cpu_cores": random.randint(2, 8),
                "memory_gb": random.randint(1, 16),
                "storage_gb": random.randint(16, 256),
                "supported_models": ["classification", "detection", "segmentation"]
            }

        device = EdgeDeviceSimulator(device_id, capabilities)
        self.devices[device_id] = device

    async def test_single_device_failure(self, device_id: str, failure_type: str) -> Dict[str, Any]:
        """Тестирование сбоя одного устройства"""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")

        device = self.devices[device_id]
        start_time = time.time()

        # Триггер сбоя
        failure_id = device.trigger_failure(failure_type)

        # Ожидание активации сбоя
        await asyncio.sleep(0.1)

        # Попытка inference во время сбоя
        success_count = 0
        failure_count = 0
        inference_times = []

        for i in range(10):
            try:
                start_inference = time.time()
                result = await device.simulate_inference({"input": f"test_{i}"})
                inference_time = time.time() - start_inference
                inference_times.append(inference_time)
                success_count += 1
            except Exception as e:
                failure_count += 1

            await asyncio.sleep(0.1)

        # Ожидание восстановления
        await asyncio.sleep(6)  # Ждем дольше чем типичные сбои

        # Тест после восстановления
        recovery_success = 0
        for i in range(5):
            try:
                result = await device.simulate_inference({"input": f"recovery_{i}"})
                recovery_success += 1
            except Exception:
                pass

        test_duration = time.time() - start_time

        result = {
            "test_type": "single_device_failure",
            "device_id": device_id,
            "failure_type": failure_type,
            "failure_id": failure_id,
            "during_failure": {
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / (success_count + failure_count),
                "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0
            },
            "after_recovery": {
                "recovery_success": recovery_success,
                "recovery_rate": recovery_success / 5
            },
            "test_duration": test_duration,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_network_partition(self) -> Dict[str, Any]:
        """Тестирование сетевого разделения"""
        start_time = time.time()

        # Выбираем подмножество устройств
        affected_devices = list(self.devices.keys())[:len(self.devices)//2]

        # Симулируем network partition
        failure_ids = []
        for device_id in affected_devices:
            failure_id = self.devices[device_id].trigger_failure("network", duration=8.0)
            failure_ids.append(failure_id)

        await asyncio.sleep(0.5)  # Ожидание активации

        # Тест inference во время partition
        partition_results = {}
        for device_id, device in self.devices.items():
            try:
                result = await device.simulate_inference({"input": "partition_test"})
                partition_results[device_id] = "success"
            except Exception as e:
                partition_results[device_id] = "failed"

        # Ожидание восстановления
        await asyncio.sleep(10)

        # Тест после восстановления
        recovery_results = {}
        for device_id, device in self.devices.items():
            try:
                result = await device.simulate_inference({"input": "recovery_test"})
                recovery_results[device_id] = "success"
            except Exception as e:
                recovery_results[device_id] = "failed"

        test_duration = time.time() - start_time

        result = {
            "test_type": "network_partition",
            "affected_devices": affected_devices,
            "failure_ids": failure_ids,
            "during_partition": partition_results,
            "after_recovery": recovery_results,
            "partition_success_rate": sum(1 for r in partition_results.values() if r == "success") / len(partition_results),
            "recovery_success_rate": sum(1 for r in recovery_results.values() if r == "success") / len(recovery_results),
            "test_duration": test_duration,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def test_cascading_failures(self) -> Dict[str, Any]:
        """Тестирование каскадных сбоев"""
        start_time = time.time()

        # Начинаем с одного сбоя
        initial_device = list(self.devices.keys())[0]
        failure_chain = [initial_device]

        # Каскад: сбой одного устройства вызывает сбой зависимых
        for i, device_id in enumerate(list(self.devices.keys())[1:], 1):
            if random.random() < 0.7:  # 70% шанс распространения
                failure_chain.append(device_id)
                self.devices[device_id].trigger_failure("power", duration=5.0 + i)
                await asyncio.sleep(0.2)  # Задержка распространения

        # Тест во время каскада
        cascade_results = {}
        for device_id, device in self.devices.items():
            try:
                result = await device.simulate_inference({"input": "cascade_test"})
                cascade_results[device_id] = "success"
            except Exception as e:
                cascade_results[device_id] = "failed"

        # Ожидание восстановления
        await asyncio.sleep(12)

        # Финальный тест
        final_results = {}
        for device_id, device in self.devices.items():
            try:
                result = await device.simulate_inference({"input": "final_test"})
                final_results[device_id] = "success"
            except Exception as e:
                final_results[device_id] = "failed"

        test_duration = time.time() - start_time

        result = {
            "test_type": "cascading_failures",
            "failure_chain": failure_chain,
            "during_cascade": cascade_results,
            "after_recovery": final_results,
            "cascade_failure_rate": sum(1 for r in cascade_results.values() if r == "failed") / len(cascade_results),
            "final_success_rate": sum(1 for r in final_results.values() if r == "success") / len(final_results),
            "cascade_length": len(failure_chain),
            "test_duration": test_duration,
            "timestamp": datetime.now().isoformat()
        }

        self.test_results.append(result)
        return result

    async def run_comprehensive_edge_tests(self) -> Dict[str, Any]:
        """Запуск комплексного тестирования edge сбоев"""
        print("🔌 Запуск comprehensive edge failure testing")
        print("=" * 60)

        # Создание устройств
        device_configs = [
            ("edge_001", {"location": "factory_floor", "model": "raspberry_pi_4"}),
            ("edge_002", {"location": "warehouse", "model": "jetson_nano"}),
            ("edge_003", {"location": "retail_store", "model": "intel_nuc"}),
            ("edge_004", {"location": "office", "model": "mac_mini"}),
            ("edge_005", {"location": "vehicle", "model": "raspberry_pi_5"}),
        ]

        for device_id, capabilities in device_configs:
            self.add_device(device_id, capabilities)

        print(f"📱 Создано {len(self.devices)} edge устройств")

        # Запуск тестов
        test_results = []

        # Тесты одиночных сбоев
        failure_types = ["network", "power", "memory", "cpu", "disk"]
        for device_id in list(self.devices.keys())[:3]:  # Тест на 3 устройствах
            for failure_type in failure_types:
                print(f"🔥 Тестирование {failure_type} сбоя на {device_id}")
                result = await self.test_single_device_failure(device_id, failure_type)
                test_results.append(result)
                await asyncio.sleep(0.5)

        # Тест сетевого разделения
        print("🌐 Тестирование network partition")
        partition_result = await self.test_network_partition()
        test_results.append(partition_result)
        await asyncio.sleep(1)

        # Тест каскадных сбоев
        print("⛓️  Тестирование cascading failures")
        cascade_result = await self.test_cascading_failures()
        test_results.append(cascade_result)

        # Анализ результатов
        analysis = self._analyze_edge_test_results(test_results)

        return {
            "test_summary": {
                "total_tests": len(test_results),
                "devices_tested": len(self.devices),
                "timestamp": datetime.now().isoformat(),
                "test_duration": sum(r.get("test_duration", 0) for r in test_results)
            },
            "results": test_results,
            "analysis": analysis
        }

    def _analyze_edge_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ результатов edge тестирования"""
        analysis = {
            "failure_type_effectiveness": {},
            "device_resilience": {},
            "recovery_performance": {},
            "recommendations": []
        }

        # Анализ по типам сбоев
        failure_analysis = {}
        for result in results:
            if result["test_type"] == "single_device_failure":
                f_type = result["failure_type"]
                if f_type not in failure_analysis:
                    failure_analysis[f_type] = []

                success_rate = result["during_failure"]["success_rate"]
                recovery_rate = result["after_recovery"]["recovery_rate"]
                failure_analysis[f_type].append({
                    "success_rate": success_rate,
                    "recovery_rate": recovery_rate
                })

        # Средние значения по типам сбоев
        for f_type, measurements in failure_analysis.items():
            avg_success = sum(m["success_rate"] for m in measurements) / len(measurements)
            avg_recovery = sum(m["recovery_rate"] for m in measurements) / len(measurements)
            analysis["failure_type_effectiveness"][f_type] = {
                "avg_success_rate": avg_success,
                "avg_recovery_rate": avg_recovery,
                "measurements": len(measurements)
            }

        # Анализ resilience устройств
        device_analysis = {}
        for result in results:
            if result["test_type"] == "single_device_failure":
                device_id = result["device_id"]
                if device_id not in device_analysis:
                    device_analysis[device_id] = []

                device_analysis[device_id].append(result["after_recovery"]["recovery_rate"])

        for device_id, recovery_rates in device_analysis.items():
            avg_recovery = sum(recovery_rates) / len(recovery_rates)
            analysis["device_resilience"][device_id] = {
                "avg_recovery_rate": avg_recovery,
                "tests_count": len(recovery_rates)
            }

        # Генерация рекомендаций
        analysis["recommendations"] = self._generate_edge_recommendations(analysis)

        return analysis

    def _generate_edge_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций для edge resilience"""
        recommendations = []

        # Анализ типов сбоев
        failure_effectiveness = analysis["failure_type_effectiveness"]
        if failure_effectiveness:
            worst_failure = min(failure_effectiveness.items(),
                              key=lambda x: x[1]["avg_success_rate"])
            if worst_failure[1]["avg_success_rate"] < 0.5:
                recommendations.append(f"Улучшить защиту от {worst_failure[0]} сбоев - низкая устойчивость")

        # Анализ устройств
        device_resilience = analysis["device_resilience"]
        if device_resilience:
            weakest_device = min(device_resilience.items(),
                               key=lambda x: x[1]["avg_recovery_rate"])
            if weakest_device[1]["avg_recovery_rate"] < 0.8:
                recommendations.append(f"Улучшить {weakest_device[0]} - низкая скорость восстановления")

        if not recommendations:
            recommendations.append("Edge устройства демонстрируют хорошую устойчивость к сбоям")

        return recommendations


async def main():
    """Основная функция для демонстрации"""
    tester = EdgeFailureTester()

    results = await tester.run_comprehensive_edge_tests()

    # Сохранение результатов
    with open("edge_failure_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n📊 Результаты edge failure testing сохранены в edge_failure_test_results.json")

    # Вывод сводки
    summary = results["analysis"]
    print("\n📈 Сводка результатов:")

    if "failure_type_effectiveness" in summary:
        print("\n🔥 Эффективность типов сбоев:")
        for f_type, stats in summary["failure_type_effectiveness"].items():
            print(".1%")

    if "device_resilience" in summary:
        print("\n📱 Resilience устройств:")
        for device_id, stats in summary["device_resilience"].items():
            print(".1%")

    print("\n💡 Рекомендации:")
    for rec in summary["recommendations"]:
        print(f"   • {rec}")


if __name__ == "__main__":
    asyncio.run(main())