"""
Интеграционные тесты для Base Interface
"""

import pytest
from production.base_interface import BaseComponent


class TestBaseComponent:
    """Тесты для базового компонента"""

    def test_base_component_initialization(self):
        """Тест инициализации базового компонента"""
        component = BaseComponent("test_component")

        assert component.name == "test_component"
        assert component.status == "initialized"
        assert hasattr(component, 'logger')

    def test_base_component_status_management(self):
        """Тест управления статусом компонента"""
        component = BaseComponent("test_component")

        # Проверка начального статуса
        assert component.status == "initialized"

        # Изменение статуса
        component.set_status("operational")
        assert component.status == "operational"

        component.set_status("shutdown")
        assert component.status == "shutdown"

    def test_base_component_health_status(self):
        """Тест получения статуса здоровья"""
        component = BaseComponent("test_component")

        # По умолчанию компонент не operational
        health_status = component.get_health_status()
        assert health_status["name"] == "test_component"
        assert health_status["status"] == "initialized"
        assert health_status["healthy"] == False

        # После установки operational статуса
        component.set_status("operational")
        health_status = component.get_health_status()
        assert health_status["status"] == "operational"
        assert health_status["healthy"] == True

    def test_abstract_methods_not_implemented(self):
        """Тест что абстрактные методы не реализованы в базовом классе"""
        component = BaseComponent("test_component")

        # Эти методы должны быть переопределены в подклассах
        with pytest.raises(NotImplementedError):
            component.initialize()

        with pytest.raises(NotImplementedError):
            component.health_check()

        with pytest.raises(NotImplementedError):
            component.get_status()

        with pytest.raises(NotImplementedError):
            component.shutdown()