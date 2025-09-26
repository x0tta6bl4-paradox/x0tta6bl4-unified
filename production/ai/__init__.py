"""
AI Core Module
"""

from ..base_interface import BaseComponent
from .advanced_ai_ml_system import AdvancedAIMLSystem

class AICore(BaseComponent):
    """AI Core компонент с продвинутой AI/ML системой"""

    def __init__(self):
        super().__init__("ai_core")
        self.models = ["gpt", "claude", "llama"]
        self.agents = ["documentation", "monitoring", "optimization"]
        self.advanced_ai_ml_system = AdvancedAIMLSystem()

    async def initialize(self) -> bool:
        """Инициализация AI core"""
        try:
            self.logger.info("Инициализация AI Core...")

            # Инициализация продвинутой AI/ML системы
            ai_ml_init = await self.advanced_ai_ml_system.initialize()
            if not ai_ml_init:
                self.logger.warning("Advanced AI/ML System не инициализирована")

            self.set_status("operational")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка инициализации AI Core: {e}")
            return False

    async def health_check(self) -> bool:
        """Проверка здоровья AI core"""
        try:
            # Проверка здоровья продвинутой системы
            ai_ml_healthy = await self.advanced_ai_ml_system.health_check()
            return self.status == "operational" and ai_ml_healthy
        except Exception as e:
            self.logger.error(f"Ошибка проверки здоровья AI Core: {e}")
            return False

    async def get_status(self) -> dict:
        """Получение статуса AI core"""
        ai_ml_status = await self.advanced_ai_ml_system.get_status()

        return {
            "name": self.name,
            "status": self.status,
            "models": self.models,
            "agents": self.agents,
            "advanced_ai_ml_system": ai_ml_status,
            "healthy": await self.health_check()
        }

    async def shutdown(self) -> bool:
        """Остановка AI core"""
        try:
            self.logger.info("Остановка AI Core...")

            # Остановка продвинутой AI/ML системы
            await self.advanced_ai_ml_system.shutdown()

            self.set_status("shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка остановки AI Core: {e}")
            return False

__all__ = ['AICore']