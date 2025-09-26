#!/usr/bin/env python3
"""
Активация AI агентов команды миграции x0tta6bl4-unified
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Добавляем пути для импорта
sys.path.append('/home/x0tta6bl4/production')
sys.path.append('/home/x0tta6bl4-next/src')
sys.path.append('/home/x0tta6bl4')

# Импорт компонентов
try:
    from advanced_ai_service import AdvancedAIService
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

try:
    from quantum.integration.ultimate_quantum_integration import QuantumCoreIntegration
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

class SimpleMigrationAgent:
    """Упрощенный агент миграции"""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role
        self.active = False
        self.ai_service = None
        self.quantum_core = None

    async def initialize(self):
        """Инициализация агента"""
        try:
            if AI_AVAILABLE:
                self.ai_service = AdvancedAIService()
            if QUANTUM_AVAILABLE:
                self.quantum_core = QuantumCoreIntegration()
                await self.quantum_core.initialize_quantum_core()

            self.active = True
            return True
        except Exception as e:
            print(f"Ошибка инициализации {self.agent_id}: {e}")
            return False

    def get_status(self):
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "active": self.active,
            "ai_available": self.ai_service is not None,
            "quantum_available": self.quantum_core is not None
        }

class MigrationTeam:
    """Команда миграции"""

    def __init__(self):
        self.agents = {}
        self._create_agents()

    def _create_agents(self):
        """Создание агентов команд"""
        roles = [
            ("project_manager", "Project Manager"),
            ("senior_backend_dev", "Senior Backend Developer"),
            ("senior_fullstack_dev", "Senior Full-Stack Developer"),
            ("devops_engineer", "DevOps Engineer"),
            ("qa_engineer", "QA Engineer"),
            ("security_engineer", "Security Engineer"),
            ("database_engineer", "Database Engineer"),
            ("technical_writer", "Technical Writer")
        ]

        for agent_id, role in roles:
            self.agents[agent_id] = SimpleMigrationAgent(agent_id, role)

    async def activate_team(self):
        """Активация команды"""
        print("🚀 Активация команды миграции x0tta6bl4-unified...")

        activation_results = {}
        successful_activations = 0

        for agent_id, agent in self.agents.items():
            success = await agent.initialize()
            activation_results[agent_id] = {
                "success": success,
                "role": agent.role
            }
            if success:
                successful_activations += 1
                print(f"✅ Агент {agent_id} ({agent.role}) активирован")
            else:
                print(f"❌ Агент {agent_id} ({agent.role}) не активирован")

        result = {
            "activation_results": activation_results,
            "successful_activations": successful_activations,
            "total_agents": len(self.agents),
            "team_ready": successful_activations >= 6,
            "ai_components_available": AI_AVAILABLE,
            "quantum_core_available": QUANTUM_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }

        print(f"🎉 Команда активирована: {successful_activations}/{len(self.agents)} агентов")
        print(f"AI компоненты: {'✅' if AI_AVAILABLE else '❌'}")
        print(f"Квантовое ядро: {'✅' if QUANTUM_AVAILABLE else '❌'}")

        return result

    def get_team_status(self):
        """Статус команды"""
        agent_statuses = {}
        active_agents = 0

        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            agent_statuses[agent_id] = status
            if status["active"]:
                active_agents += 1

        return {
            "team_overview": {
                "total_agents": len(self.agents),
                "active_agents": active_agents,
                "team_readiness": "high" if active_agents >= 6 else "medium"
            },
            "agent_statuses": agent_statuses,
            "capabilities": {
                "ai_enhancement": "active" if AI_AVAILABLE else "limited",
                "quantum_integration": "active" if QUANTUM_AVAILABLE else "limited"
            },
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Главная функция активации"""
    print("🎯 Начинаем активацию AI агентов команды миграции x0tta6bl4-unified")
    print("=" * 70)

    team = MigrationTeam()

    # Активация команды
    activation_result = await team.activate_team()

    # Сохранение результатов
    with open('migration_team_activation_report.json', 'w', encoding='utf-8') as f:
        json.dump(activation_result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("ОТЧЕТ ОБ АКТИВАЦИИ КОМАНДЫ МИГРАЦИИ:")
    print("=" * 70)

    status = team.get_team_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    # Сохранение статуса
    with open('migration_team_status_report.json', 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    print("\n📄 Отчеты сохранены:")
    print("- migration_team_activation_report.json")
    print("- migration_team_status_report.json")

    if activation_result["team_ready"]:
        print("\n🎉 КОМАНДА МИГРАЦИИ ГОТОВА К РАБОТЕ!")
    else:
        print("\n⚠️ КОМАНДА МИГРАЦИИ НУЖДАЕТСЯ В ДОПОЛНИТЕЛЬНОЙ НАСТРОЙКЕ")

if __name__ == "__main__":
    asyncio.run(main())