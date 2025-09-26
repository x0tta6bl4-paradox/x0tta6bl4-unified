"""
Базовый агент для x0tta6bl4-unified
Наследуется от BaseComponent с дополнительными возможностями агентов
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Базовый класс для всех агентов x0tta6bl4-unified"""

    def __init__(self, name: str, agent_type: str = "generic"):
        self.name = name
        self.agent_type = agent_type
        self.status = "initialized"
        self.logger = logging.getLogger(f"x0tta6bl4.agents.{name}")
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.capabilities: Dict[str, Any] = {}
        self.coordination_peers: Dict[str, 'BaseAgent'] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """Инициализация агента"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Проверка здоровья агента"""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса агента"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Остановка агента"""
        pass

    def set_status(self, status: str):
        """Установка статуса агента"""
        self.status = status
        self.last_active = datetime.now()
        self.logger.info(f"Статус агента {self.name} изменен на: {status}")

    def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья"""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "healthy": self.status == "operational",
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "capabilities": self.capabilities,
            "coordination_peers": list(self.coordination_peers.keys())
        }

    async def coordinate_with_peer(self, peer_name: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Координация с другим агентом"""
        if peer_name not in self.coordination_peers:
            return {"error": f"Peer {peer_name} not found"}

        peer = self.coordination_peers[peer_name]
        try:
            # Имитация координации - в реальности здесь будет более сложная логика
            response = await peer.receive_coordination_message(self.name, message)
            return response
        except Exception as e:
            self.logger.error(f"Ошибка координации с {peer_name}: {e}")
            return {"error": str(e)}

    async def receive_coordination_message(self, from_agent: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Получение сообщения координации от другого агента"""
        self.logger.info(f"Получено сообщение координации от {from_agent}: {message}")
        # Базовая обработка - переопределяется в наследниках
        return {"status": "received", "from": from_agent, "message": message}

    def register_capability(self, capability_name: str, capability_info: Dict[str, Any]):
        """Регистрация возможности агента"""
        self.capabilities[capability_name] = capability_info
        self.logger.info(f"Зарегистрирована возможность: {capability_name}")

    def add_coordination_peer(self, peer_name: str, peer_agent: 'BaseAgent'):
        """Добавление агента для координации"""
        self.coordination_peers[peer_name] = peer_agent
        self.logger.info(f"Добавлен peer для координации: {peer_name}")

    async def execute_task(self, task_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение задачи агентом"""
        self.last_active = datetime.now()
        self.logger.info(f"Выполнение задачи: {task_name}")

        try:
            # Базовая реализация - переопределяется в наследниках
            result = await self._execute_task_internal(task_name, parameters)
            return {
                "task_name": task_name,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи {task_name}: {e}")
            return {
                "task_name": task_name,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @abstractmethod
    async def _execute_task_internal(self, task_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Внутренняя реализация выполнения задачи"""
        pass