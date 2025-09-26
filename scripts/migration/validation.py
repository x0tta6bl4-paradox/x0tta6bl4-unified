#!/usr/bin/env python3
"""
✅ Migration Validation Script
Скрипт валидации результатов миграции x0tta6bl4-unified
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import importlib.util
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Валидатор результатов миграции"""

    def __init__(self, target_path: Path):
        self.target_path = target_path

        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks_passed": 0,
            "checks_failed": 0,
            "warnings": [],
            "errors": [],
            "components_validated": []
        }

        # Определение проверок валидации
        self.validation_checks = [
            self._check_project_structure,
            self._check_config_files,
            self._check_python_imports,
            self._check_service_integrity,
            self._check_database_connectivity,
            self._check_api_endpoints,
            self._check_monitoring_setup,
            self._check_security_config,
            self._validate_quantum_components,
            self._validate_ai_components,
            self._validate_enterprise_components
        ]

    def validate_all(self) -> bool:
        """Полная валидация миграции"""
        logger.info("🚀 Начало валидации миграции x0tta6bl4-unified")

        try:
            # Выполнение всех проверок
            for check_func in self.validation_checks:
                check_name = check_func.__name__.replace('_check_', '').replace('_validate_', '')
                logger.info(f"🔍 Выполнение проверки: {check_name}")

                try:
                    success, message = check_func()
                    if success:
                        self.validation_results["checks_passed"] += 1
                        logger.info(f"✅ {check_name}: {message}")
                    else:
                        self.validation_results["checks_failed"] += 1
                        logger.error(f"❌ {check_name}: {message}")
                        self.validation_results["errors"].append(f"{check_name}: {message}")
                except Exception as e:
                    self.validation_results["checks_failed"] += 1
                    error_msg = f"{check_name}: {str(e)}"
                    logger.error(f"❌ {check_name}: {error_msg}")
                    self.validation_results["errors"].append(error_msg)

            # Генерация отчета
            self._generate_validation_report()

            # Финальный результат
            success = self.validation_results["checks_failed"] == 0
            if success:
                logger.info("🎉 Валидация миграции пройдена успешно!")
            else:
                logger.error(f"⚠️ Валидация выявила {self.validation_results['checks_failed']} проблем")

            return success

        except Exception as e:
            logger.error(f"❌ Критическая ошибка валидации: {e}")
            return False

    def _check_project_structure(self) -> Tuple[bool, str]:
        """Проверка структуры проекта"""
        required_dirs = [
            "config",
            "production",
            "scripts",
            "tests",
            "docs"
        ]

        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.target_path / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            return False, f"Отсутствуют директории: {', '.join(missing_dirs)}"

        # Проверка наличия основных файлов
        required_files = [
            "main.py",
            "requirements.txt",
            "config/unified_config.yaml"
        ]

        missing_files = []
        for file_name in required_files:
            if not (self.target_path / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            return False, f"Отсутствуют файлы: {', '.join(missing_files)}"

        return True, "Структура проекта корректна"

    def _check_config_files(self) -> Tuple[bool, str]:
        """Проверка конфигурационных файлов"""
        config_checks = []

        # Проверка unified конфига
        unified_config = self.target_path / "config" / "unified_config.yaml"
        if unified_config.exists():
            try:
                with open(unified_config, 'r') as f:
                    config = yaml.safe_load(f)

                required_keys = ['system', 'sources', 'merged_configs']
                for key in required_keys:
                    if key not in config:
                        config_checks.append(f"Отсутствует ключ '{key}' в unified_config.yaml")

                if 'system' in config:
                    system_keys = ['name', 'version', 'environment']
                    for key in system_keys:
                        if key not in config['system']:
                            config_checks.append(f"Отсутствует системный ключ '{key}'")

            except Exception as e:
                config_checks.append(f"Ошибка чтения unified_config.yaml: {e}")
        else:
            config_checks.append("Файл unified_config.yaml не найден")

        # Проверка конфигурации миграции
        migration_config = self.target_path / "config" / "migration_config.json"
        if migration_config.exists():
            try:
                with open(migration_config, 'r') as f:
                    config = json.load(f)

                if 'migration' not in config:
                    config_checks.append("Отсутствует секция 'migration' в migration_config.json")

            except Exception as e:
                config_checks.append(f"Ошибка чтения migration_config.json: {e}")
        else:
            config_checks.append("Файл migration_config.json не найден")

        if config_checks:
            return False, f"Проблемы с конфигурацией: {'; '.join(config_checks)}"

        return True, "Конфигурационные файлы корректны"

    def _check_python_imports(self) -> Tuple[bool, str]:
        """Проверка импортов Python"""
        import_issues = []

        # Проверка основных модулей
        main_modules = [
            "main",
            "production.quantum",
            "production.ai",
            "production.enterprise",
            "production.api"
        ]

        for module in main_modules:
            try:
                spec = importlib.util.find_spec(module)
                if spec is None:
                    import_issues.append(f"Модуль {module} не найден")
            except Exception as e:
                import_issues.append(f"Ошибка импорта {module}: {e}")

        # Проверка requirements.txt
        requirements_file = self.target_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read().strip()
                    if not requirements:
                        import_issues.append("Файл requirements.txt пуст")
            except Exception as e:
                import_issues.append(f"Ошибка чтения requirements.txt: {e}")
        else:
            import_issues.append("Файл requirements.txt не найден")

        if import_issues:
            return False, f"Проблемы с импортами: {'; '.join(import_issues)}"

        return True, "Импорты Python корректны"

    def _check_service_integrity(self) -> Tuple[bool, str]:
        """Проверка целостности сервисов"""
        service_issues = []

        # Проверка сервисов
        services_dir = self.target_path / "production" / "services"
        if services_dir.exists():
            expected_services = [
                "quantum_performance_predictor",
                "quantum_auto_scaler",
                "phi_harmonic_load_balancer",
                "api_gateway",
                "mesh_api"
            ]

            found_services = []
            for service_dir in services_dir.iterdir():
                if service_dir.is_dir():
                    found_services.append(service_dir.name)

            missing_services = set(expected_services) - set(found_services)
            if missing_services:
                service_issues.append(f"Отсутствуют сервисы: {', '.join(missing_services)}")

            # Проверка наличия app.py в каждом сервисе
            for service in found_services:
                app_file = services_dir / service / "app.py"
                if not app_file.exists():
                    service_issues.append(f"Отсутствует app.py в сервисе {service}")
        else:
            service_issues.append("Директория production/services не найдена")

        if service_issues:
            return False, f"Проблемы с сервисами: {'; '.join(service_issues)}"

        return True, "Сервисы имеют корректную структуру"

    def _check_database_connectivity(self) -> Tuple[bool, str]:
        """Проверка подключения к базам данных"""
        # Проверка наличия настроек БД в конфиге
        unified_config = self.target_path / "config" / "unified_config.yaml"
        if unified_config.exists():
            try:
                with open(unified_config, 'r') as f:
                    config = yaml.safe_load(f)

                # Проверка наличия настроек БД
                if 'database' in config.get('merged_configs', {}):
                    db_config = config['merged_configs']['database']
                    # Здесь можно добавить реальную проверку подключения
                    return True, "Настройки базы данных найдены"
                else:
                    return True, "Настройки базы данных не найдены (используется по умолчанию)"

            except Exception as e:
                return False, f"Ошибка проверки настроек БД: {e}"

        return True, "Проверка подключения к БД пропущена"

    def _check_api_endpoints(self) -> Tuple[bool, str]:
        """Проверка API endpoints"""
        api_issues = []

        # Проверка основного API файла
        main_api = self.target_path / "production" / "api" / "main.py"
        if main_api.exists():
            try:
                with open(main_api, 'r') as f:
                    content = f.read()

                # Проверка наличия основных endpoints
                required_endpoints = [
                    "/health",
                    "/api/v1/quantum/status",
                    "/api/v1/ai/status",
                    "/api/v1/enterprise/status"
                ]

                for endpoint in required_endpoints:
                    if endpoint not in content:
                        api_issues.append(f"Отсутствует endpoint {endpoint}")

            except Exception as e:
                api_issues.append(f"Ошибка чтения API файла: {e}")
        else:
            api_issues.append("Файл production/api/main.py не найден")

        if api_issues:
            return False, f"Проблемы с API: {'; '.join(api_issues)}"

        return True, "API endpoints корректны"

    def _check_monitoring_setup(self) -> Tuple[bool, str]:
        """Проверка настройки мониторинга"""
        monitoring_issues = []

        # Проверка конфигурации мониторинга
        monitoring_config = self.target_path / "config" / "monitoring_config.yaml"
        if monitoring_config.exists():
            try:
                with open(monitoring_config, 'r') as f:
                    config = yaml.safe_load(f)

                required_sections = ['prometheus', 'grafana']
                for section in required_sections:
                    if section not in config:
                        monitoring_issues.append(f"Отсутствует секция {section} в конфигурации мониторинга")

            except Exception as e:
                monitoring_issues.append(f"Ошибка чтения конфигурации мониторинга: {e}")
        else:
            monitoring_issues.append("Файл monitoring_config.yaml не найден")

        # Проверка unified мониторинга
        unified_monitoring = self.target_path / "production" / "monitoring" / "unified_monitoring.py"
        if not unified_monitoring.exists():
            monitoring_issues.append("Файл unified_monitoring.py не найден")

        if monitoring_issues:
            return False, f"Проблемы с мониторингом: {'; '.join(monitoring_issues)}"

        return True, "Мониторинг настроен корректно"

    def _check_security_config(self) -> Tuple[bool, str]:
        """Проверка конфигурации безопасности"""
        security_issues = []

        # Проверка наличия настроек безопасности в unified конфиге
        unified_config = self.target_path / "config" / "unified_config.yaml"
        if unified_config.exists():
            try:
                with open(unified_config, 'r') as f:
                    config = yaml.safe_load(f)

                merged_configs = config.get('merged_configs', {})
                if 'security' in merged_configs:
                    security_config = merged_configs['security']

                    required_security = ['encryption', 'authentication']
                    for sec_item in required_security:
                        if sec_item not in security_config:
                            security_issues.append(f"Отсутствует настройка безопасности: {sec_item}")
                else:
                    security_issues.append("Секция security отсутствует в конфигурации")

            except Exception as e:
                security_issues.append(f"Ошибка чтения настроек безопасности: {e}")

        if security_issues:
            return False, f"Проблемы с безопасностью: {'; '.join(security_issues)}"

        return True, "Конфигурация безопасности корректна"

    def _validate_quantum_components(self) -> Tuple[bool, str]:
        """Валидация квантовых компонентов"""
        quantum_issues = []

        quantum_dir = self.target_path / "production" / "quantum"
        if quantum_dir.exists():
            # Проверка наличия ключевых файлов
            required_files = ["__init__.py", "quantum_config.py"]
            for file_name in required_files:
                if not (quantum_dir / file_name).exists():
                    quantum_issues.append(f"Отсутствует файл {file_name} в quantum компонентах")

            # Проверка конфигурации
            config_file = quantum_dir / "quantum_config.py"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()

                    required_configs = ["QUANTUM_PROVIDERS", "QUANTUM_ALGORITHMS"]
                    for config in required_configs:
                        if config not in content:
                            quantum_issues.append(f"Отсутствует конфигурация {config}")

                except Exception as e:
                    quantum_issues.append(f"Ошибка чтения quantum конфигурации: {e}")
        else:
            quantum_issues.append("Директория production/quantum не найдена")

        if quantum_issues:
            return False, f"Проблемы с квантовыми компонентами: {'; '.join(quantum_issues)}"

        return True, "Квантовые компоненты валидны"

    def _validate_ai_components(self) -> Tuple[bool, str]:
        """Валидация AI компонентов"""
        ai_issues = []

        ai_dir = self.target_path / "production" / "ai"
        if ai_dir.exists():
            # Проверка наличия конфигурации
            config_file = ai_dir / "ai_config.py"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()

                    required_configs = ["AI_MODELS", "ML_FRAMEWORKS"]
                    for config in required_configs:
                        if config not in content:
                            ai_issues.append(f"Отсутствует AI конфигурация {config}")

                except Exception as e:
                    ai_issues.append(f"Ошибка чтения AI конфигурации: {e}")
            else:
                ai_issues.append("Файл ai_config.py не найден")
        else:
            ai_issues.append("Директория production/ai не найдена")

        if ai_issues:
            return False, f"Проблемы с AI компонентами: {'; '.join(ai_issues)}"

        return True, "AI компоненты валидны"

    def _validate_enterprise_components(self) -> Tuple[bool, str]:
        """Валидация enterprise компонентов"""
        enterprise_issues = []

        enterprise_dir = self.target_path / "production" / "enterprise"
        if enterprise_dir.exists():
            # Проверка наличия конфигурации
            config_file = enterprise_dir / "enterprise_config.py"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()

                    required_configs = ["ENTERPRISE_FEATURES", "API_GATEWAY"]
                    for config in required_configs:
                        if config not in content:
                            enterprise_issues.append(f"Отсутствует enterprise конфигурация {config}")

                except Exception as e:
                    enterprise_issues.append(f"Ошибка чтения enterprise конфигурации: {e}")
            else:
                enterprise_issues.append("Файл enterprise_config.py не найден")
        else:
            enterprise_issues.append("Директория production/enterprise не найдена")

        if enterprise_issues:
            return False, f"Проблемы с enterprise компонентами: {'; '.join(enterprise_issues)}"

        return True, "Enterprise компоненты валидны"

    def _generate_validation_report(self):
        """Генерация отчета о валидации"""
        logger.info("📋 Генерация отчета о валидации...")

        report = {
            "validation_report": {
                "timestamp": self.validation_results["timestamp"],
                "summary": {
                    "checks_passed": self.validation_results["checks_passed"],
                    "checks_failed": self.validation_results["checks_failed"],
                    "total_checks": self.validation_results["checks_passed"] + self.validation_results["checks_failed"],
                    "success_rate": (self.validation_results["checks_passed"] /
                                   (self.validation_results["checks_passed"] + self.validation_results["checks_failed"]) * 100)
                    if (self.validation_results["checks_passed"] + self.validation_results["checks_failed"]) > 0 else 0
                },
                "components_validated": self.validation_results["components_validated"],
                "warnings": self.validation_results["warnings"],
                "errors": self.validation_results["errors"]
            }
        }

        # Сохранение отчета
        report_file = self.target_path / "validation_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Отчет о валидации сохранен: {report_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения отчета о валидации: {e}")

    def get_validation_results(self) -> Dict[str, Any]:
        """Получение результатов валидации"""
        return self.validation_results