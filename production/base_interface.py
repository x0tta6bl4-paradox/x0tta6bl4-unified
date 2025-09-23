
"""
Базовый интерфейс для всех компонентов x0tta6bl4 Unified
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """Базовый класс для всех компонентов"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = "initialized"
        self.logger = logging.getLogger(f"x0tta6bl4.{name}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Инициализация компонента"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Проверка здоровья компонента"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса компонента"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Остановка компонента"""
        pass
    
    def set_status(self, status: str):
        """Установка статуса компонента"""
        self.status = status
        self.logger.info(f"Статус {self.name} изменен на: {status}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья"""
        return {
            "name": self.name,
            "status": self.status,
            "healthy": self.status == "operational"
        }
