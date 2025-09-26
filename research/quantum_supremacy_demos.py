"""
Демонстрации квантового превосходства для алгоритмов Shor, Grover, QAOA и VQE
Координирует выполнение всех демонстраций с измерением производительности
"""

import asyncio
import time
import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

# Импорты компонентов
from production.quantum.quantum_interface import QuantumCore
from research.research_engineer_agent import ResearchEngineerAgent


@dataclass
class QuantumMetrics:
    """Метрики квантовых вычислений"""
    coherence_time: float
    entanglement_fidelity: float
    gate_error_rate: float
    readout_error: float
    t1_time: float
    t2_time: float


@dataclass
class SupremacyResult:
    """Результат демонстрации квантового превосходства"""
    algorithm: str
    problem_size: int
    quantum_time: float
    classical_time: float
    speedup_factor: float
    quantum_metrics: QuantumMetrics
    success_rate: float
    provider: str
    timestamp: float
    metadata: Dict[str, Any]


class QuantumSupremacyDemos:
    """Координатор демонстраций квантового превосходства"""

    def __init__(self):
        self.quantum_core = QuantumCore()
        self.research_agent = ResearchEngineerAgent()
        self.results: List[SupremacyResult] = []
        self.logger = None

    async def initialize(self) -> bool:
        """Инициализация демонстраций"""
        try:
            print("Инициализация демонстраций квантового превосходства...")

            # Инициализация компонентов
            quantum_ok = await self.quantum_core.initialize()
            research_ok = await self.research_agent.initialize()

            if quantum_ok and research_ok:
                # Регистрация агентов для координации
                await self.research_agent.register_collaborator("quantum_engineer", self.quantum_core)

                print("Демонстрации успешно инициализированы")
                return True
            else:
                print("Ошибка инициализации компонентов")
                return False

        except Exception as e:
            print(f"Ошибка инициализации демонстраций: {e}")
            return False

    async def run_all_demos(self) -> Dict[str, Any]:
        """Запуск всех демонстраций"""
        try:
            print("Запуск демонстраций квантового превосходства...")

            results = {}

            # Демонстрация Shor
            print("Запуск демонстрации алгоритма Шора...")
            shor_result = await self.run_shor_demo()
            results["shor"] = shor_result

            # Демонстрация Grover
            print("Запуск демонстрации алгоритма Гровера...")
            grover_result = await self.run_grover_demo()
            results["grover"] = grover_result

            # Демонстрация QAOA
            print("Запуск демонстрации QAOA...")
            qaoa_result = await self.run_qaoa_demo()
            results["qaoa"] = qaoa_result

            # Демонстрация VQE
            print("Запуск демонстрации VQE...")
            vqe_result = await self.run_vqe_demo()
            results["vqe"] = vqe_result

            # Анализ результатов
            analysis = await self.analyze_all_results(results)

            # Сохранение результатов
            await self.save_results(results, analysis)

            return {
                "status": "completed",
                "results": results,
                "analysis": analysis,
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Ошибка выполнения демонстраций: {e}")
            return {"error": str(e)}

    async def run_shor_demo(self) -> Dict[str, Any]:
        """Демонстрация алгоритма Шора для факторизации"""
        try:
            from research.demo_shor_algorithm import ShorDemo
            demo = ShorDemo(self.quantum_core, self.research_agent)
            return await demo.run()
        except Exception as e:
            print(f"Ошибка демонстрации Shor: {e}")
            return {"error": str(e)}

    async def run_grover_demo(self) -> Dict[str, Any]:
        """Демонстрация алгоритма Гровера для поиска"""
        try:
            from research.demo_grover_algorithm import GroverDemo
            demo = GroverDemo(self.quantum_core, self.research_agent)
            return await demo.run()
        except Exception as e:
            print(f"Ошибка демонстрации Grover: {e}")
            return {"error": str(e)}

    async def run_qaoa_demo(self) -> Dict[str, Any]:
        """Демонстрация QAOA для оптимизации"""
        try:
            from research.demo_qaoa_algorithm import QAOADemo
            demo = QAOADemo(self.quantum_core, self.research_agent)
            return await demo.run()
        except Exception as e:
            print(f"Ошибка демонстрации QAOA: {e}")
            return {"error": str(e)}

    async def run_vqe_demo(self) -> Dict[str, Any]:
        """Демонстрация VQE для моделирования молекул"""
        try:
            from research.demo_vqe_algorithm import VQEDemo
            demo = VQEDemo(self.quantum_core, self.research_agent)
            return await demo.run()
        except Exception as e:
            print(f"Ошибка демонстрации VQE: {e}")
            return {"error": str(e)}

    async def analyze_all_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ всех результатов демонстраций"""
        try:
            print("Анализ результатов демонстраций...")

            # Сбор всех результатов для анализа
            all_results = []
            for algo, result in results.items():
                if "error" not in result:
                    all_results.append(result)

            if not all_results:
                return {"error": "Нет успешных результатов для анализа"}

            # Расчет общих метрик
            speedups = [r.get("speedup_factor", 1) for r in all_results]
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)

            # Анализ по алгоритмам
            algorithm_analysis = {}
            for algo, result in results.items():
                if "error" not in result:
                    algorithm_analysis[algo] = {
                        "speedup": result.get("speedup_factor", 1),
                        "problem_size": result.get("problem_size", 0),
                        "success_rate": result.get("success_rate", 0),
                        "quantum_advantage": result.get("speedup_factor", 1) > 1.1
                    }

            # Определение лучшего алгоритма
            best_algorithm = max(algorithm_analysis.items(),
                               key=lambda x: x[1]["speedup"]) if algorithm_analysis else None

            analysis = {
                "total_demos": len(results),
                "successful_demos": len([r for r in results.values() if "error" not in r]),
                "average_speedup": avg_speedup,
                "max_speedup": max_speedup,
                "min_speedup": min_speedup,
                "algorithm_analysis": algorithm_analysis,
                "best_algorithm": best_algorithm[0] if best_algorithm else None,
                "quantum_supremacy_achieved": avg_speedup > 2.0,
                "timestamp": time.time()
            }

            print(f"Анализ завершен. Среднее ускорение: {avg_speedup:.2f}x")
            return analysis

        except Exception as e:
            print(f"Ошибка анализа результатов: {e}")
            return {"error": str(e)}

    async def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Сохранение результатов демонстраций"""
        try:
            os.makedirs("research/data", exist_ok=True)

            # Сохранение детальных результатов
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research/data/supremacy_demos_{timestamp}.json"

            data = {
                "timestamp": time.time(),
                "results": results,
                "analysis": analysis,
                "metadata": {
                    "framework_version": "x0tta6bl4-unified",
                    "quantum_providers": ["ibm", "google", "xanadu"],
                    "algorithms_tested": ["shor", "grover", "qaoa", "vqe"]
                }
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            print(f"Результаты сохранены в {filename}")

            # Отправка результатов в Research Agent для дальнейшего анализа
            await self.research_agent.analyze_research_results({
                "experiment_id": f"supremacy_demos_{timestamp}",
                "algorithm": "quantum_supremacy_suite",
                "results": results,
                "analysis": analysis,
                "metadata": data["metadata"]
            })

        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")

    async def generate_report(self) -> Dict[str, Any]:
        """Генерация итогового отчета"""
        try:
            # Загрузка последних результатов
            results_files = [f for f in os.listdir("research/data")
                           if f.startswith("supremacy_demos_") and f.endswith(".json")]
            if not results_files:
                return {"error": "Нет результатов для отчета"}

            latest_file = max(results_files)
            with open(f"research/data/{latest_file}", "r") as f:
                data = json.load(f)

            # Генерация отчета через Research Agent
            report_data = {
                "title": "Quantum Supremacy Demonstrations Report",
                "results": data["results"],
                "analysis": data["analysis"],
                "timestamp": data["timestamp"],
                "algorithms": ["Shor", "Grover", "QAOA", "VQE"],
                "key_findings": [
                    f"Average speedup: {data['analysis'].get('average_speedup', 0):.2f}x",
                    f"Max speedup: {data['analysis'].get('max_speedup', 0):.2f}x",
                    f"Successful demos: {data['analysis'].get('successful_demos', 0)}/4",
                    f"Quantum supremacy: {'Achieved' if data['analysis'].get('quantum_supremacy_achieved', False) else 'Not achieved'}"
                ]
            }

            # Генерация технического отчета
            technical_report = await self.research_agent.generate_documentation(
                "technical", report_data
            )

            # Генерация научной публикации
            scientific_paper = await self.research_agent.create_scientific_publications(
                "Quantum Supremacy Demonstrations", report_data
            )

            # Генерация презентации
            presentation = await self.research_agent.generate_presentations(
                "conference", report_data
            )

            return {
                "technical_report": technical_report,
                "scientific_paper": scientific_paper,
                "presentation": presentation,
                "timestamp": time.time()
            }

        except Exception as e:
            print(f"Ошибка генерации отчета: {e}")
            return {"error": str(e)}


async def main():
    """Главная функция для запуска демонстраций"""
    demos = QuantumSupremacyDemos()

    # Инициализация
    if not await demos.initialize():
        print("Не удалось инициализировать демонстрации")
        return

    # Запуск демонстраций
    results = await demos.run_all_demos()
    print(f"Демонстрации завершены: {results.get('status', 'error')}")

    # Генерация отчета
    report = await demos.generate_report()
    print("Отчеты сгенерированы")

    return results


if __name__ == "__main__":
    asyncio.run(main())