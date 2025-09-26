#!/usr/bin/env python3
"""
📊 Data Migration Script for x0tta6bl4-next
Скрипт миграции данных из проекта x0tta6bl4-next в x0tta6bl4-unified
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class X0tta6bl4NextDataMigrator:
    """Мигратор данных из x0tta6bl4-next"""

    def __init__(self, source_path: Path, target_path: Path, dry_run: bool = False):
        self.source_path = source_path
        self.target_path = target_path
        self.dry_run = dry_run

        self.migration_stats = {
            "services_migrated": 0,
            "configs_migrated": 0,
            "k8s_manifests_migrated": 0,
            "tests_migrated": 0,
            "docs_migrated": 0,
            "errors": []
        }

        # Определение ключевых компонентов для миграции
        self.migration_targets = {
            "services": {
                "quantum_performance_predictor": "services/quantum_performance_predictor/",
                "quantum_auto_scaler": "services/quantum_auto_scaler/",
                "phi_harmonic_load_balancer": "services/phi_harmonic_load_balancer/",
                "api_gateway": "services/api_gateway/",
                "mesh_api": "services/mesh_api/",
                "task_controller": "services/task_controller/",
                "analytics": "services/analytics/",
                "billing": "services/billing/"
            },
            "config_files": [
                "config/x0tta6bl4_config.yaml",
                "pyproject.toml",
                "requirements.txt",
                "requirements-dev.txt"
            ],
            "k8s_manifests": [
                "k8s/keda/",
                "k8s/deployments/",
                "k8s/services/"
            ],
            "source_code": [
                "src/x0tta6bl4_settings.py",
                "src/"
            ],
            "tests": [
                "tests/"
            ]
        }

    def migrate(self) -> bool:
        """Основной метод миграции"""
        logger.info("🚀 Начало миграции данных из x0tta6bl4-next")

        try:
            # Создание структуры директорий
            self._create_target_structure()

            # Миграция сервисов
            if not self._migrate_services():
                return False

            # Миграция конфигурационных файлов
            if not self._migrate_config_files():
                return False

            # Миграция Kubernetes манифестов
            if not self._migrate_k8s_manifests():
                return False

            # Миграция исходного кода
            if not self._migrate_source_code():
                return False

            # Миграция тестов
            if not self._migrate_tests():
                return False

            # Миграция документации
            if not self._migrate_documentation():
                return False

            logger.info("✅ Миграция данных из x0tta6bl4-next завершена успешно")
            logger.info(f"📊 Статистика: {self.migration_stats}")
            return True

        except Exception as e:
            logger.error(f"❌ Критическая ошибка миграции: {e}")
            self.migration_stats["errors"].append(str(e))
            return False

    def _create_target_structure(self):
        """Создание структуры директорий в целевом проекте"""
        logger.info("📁 Создание структуры директорий для x0tta6bl4-next...")

        directories = [
            "production/services",
            "config/x0tta6bl4_next",
            "k8s",
            "src",
            "tests",
            "docs/x0tta6bl4_next"
        ]

        for dir_path in directories:
            full_path = self.target_path / dir_path
            if self.dry_run:
                logger.info(f"📋 [DRY RUN] Будет создана директория: {full_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Создана директория: {full_path}")

    def _migrate_services(self) -> bool:
        """Миграция сервисов"""
        logger.info("🔧 Миграция сервисов...")

        try:
            target_services_dir = self.target_path / "production" / "services"

            for service_name, service_path in self.migration_targets["services"].items():
                source_service_dir = self.source_path / service_path

                if source_service_dir.exists():
                    target_service_dir = target_services_dir / service_name

                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будет мигрирован сервис: {service_name}")
                    else:
                        # Копирование всего сервиса
                        if target_service_dir.exists():
                            shutil.rmtree(target_service_dir)
                        shutil.copytree(source_service_dir, target_service_dir)

                        # Создание __init__.py если отсутствует
                        init_file = target_service_dir / "__init__.py"
                        if not init_file.exists():
                            init_file.write_text(f'"""Service: {service_name}"""\n')

                        self.migration_stats["services_migrated"] += 1
                        logger.info(f"✅ Мигрирован сервис: {service_name}")

                        # Специальная обработка для каждого сервиса
                        self._process_service_specifics(service_name, target_service_dir)
                else:
                    logger.warning(f"⚠️ Сервис не найден: {service_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции сервисов: {e}")
            self.migration_stats["errors"].append(f"services_migration: {str(e)}")
            return False

    def _process_service_specifics(self, service_name: str, service_dir: Path):
        """Обработка специфичных особенностей сервисов"""
        try:
            if service_name == "quantum_performance_predictor":
                self._adapt_quantum_performance_service(service_dir)
            elif service_name == "quantum_auto_scaler":
                self._adapt_quantum_auto_scaler_service(service_dir)
            elif service_name == "phi_harmonic_load_balancer":
                self._adapt_phi_harmonic_service(service_dir)
            elif service_name == "api_gateway":
                self._adapt_api_gateway_service(service_dir)
            elif service_name == "mesh_api":
                self._adapt_mesh_api_service(service_dir)

        except Exception as e:
            logger.warning(f"⚠️ Ошибка обработки специфики сервиса {service_name}: {e}")

    def _adapt_quantum_performance_service(self, service_dir: Path):
        """Адаптация сервиса quantum_performance_predictor"""
        app_file = service_dir / "app.py"
        if app_file.exists():
            # Добавление интеграции с unified системой
            with open(app_file, 'a') as f:
                f.write('\n\n# Integration with x0tta6bl4-unified\n')
                f.write('from production.monitoring.unified_monitoring import UnifiedMonitoring\n')
                f.write('monitoring = UnifiedMonitoring()\n')

    def _adapt_quantum_auto_scaler_service(self, service_dir: Path):
        """Адаптация сервиса quantum_auto_scaler"""
        app_file = service_dir / "app.py"
        if app_file.exists():
            with open(app_file, 'a') as f:
                f.write('\n\n# Auto-scaling integration\n')
                f.write('from production.scaling.unified_scaler import UnifiedScaler\n')
                f.write('scaler = UnifiedScaler()\n')

    def _adapt_phi_harmonic_service(self, service_dir: Path):
        """Адаптация сервиса phi_harmonic_load_balancer"""
        app_file = service_dir / "app.py"
        if app_file.exists():
            with open(app_file, 'a') as f:
                f.write('\n\n# Phi harmonic integration\n')
                f.write('PHI_CONSTANT = 1.618033988749895\n')
                f.write('SACRED_FREQUENCY = 108\n')

    def _adapt_api_gateway_service(self, service_dir: Path):
        """Адаптация API Gateway"""
        app_file = service_dir / "app.py"
        if app_file.exists():
            with open(app_file, 'a') as f:
                f.write('\n\n# Unified API Gateway integration\n')
                f.write('from production.api.main import app as unified_app\n')

    def _adapt_mesh_api_service(self, service_dir: Path):
        """Адаптация Mesh API"""
        app_file = service_dir / "app.py"
        if app_file.exists():
            with open(app_file, 'a') as f:
                f.write('\n\n# Mesh networking integration\n')
                f.write('from production.networking.mesh_network import MeshNetwork\n')
                f.write('mesh = MeshNetwork()\n')

    def _migrate_config_files(self) -> bool:
        """Миграция конфигурационных файлов"""
        logger.info("⚙️ Миграция конфигурационных файлов...")

        try:
            target_config_dir = self.target_path / "config" / "x0tta6bl4_next"

            for config_file in self.migration_targets["config_files"]:
                source_file = self.source_path / config_file

                if source_file.exists():
                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будет мигрирован конфиг: {config_file}")
                    else:
                        target_file = target_config_dir / source_file.name
                        shutil.copy2(source_file, target_file)

                        # Адаптация конфигурации
                        self._adapt_config_file(target_file)

                        self.migration_stats["configs_migrated"] += 1
                        logger.info(f"✅ Мигрирован конфиг: {config_file}")
                else:
                    logger.warning(f"⚠️ Конфиг файл не найден: {config_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции конфигов: {e}")
            self.migration_stats["errors"].append(f"config_migration: {str(e)}")
            return False

    def _adapt_config_file(self, config_file: Path):
        """Адаптация конфигурационного файла"""
        try:
            if config_file.suffix == '.yaml':
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # Адаптация портов для избежания конфликтов
                if 'ports' in config:
                    config['ports'] = self._adapt_ports(config['ports'])

                # Добавление unified-specific настроек
                config['unified_integration'] = {
                    'enabled': True,
                    'unified_api_url': 'http://localhost:8000',
                    'monitoring_enabled': True
                }

                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            elif config_file.name == 'requirements.txt':
                # Добавление зависимостей unified системы
                with open(config_file, 'a') as f:
                    f.write('\n# x0tta6bl4-unified dependencies\n')
                    f.write('fastapi==0.104.1\n')
                    f.write('uvicorn[standard]==0.24.0\n')
                    f.write('pydantic==2.5.0\n')

        except Exception as e:
            logger.warning(f"⚠️ Ошибка адаптации конфига {config_file}: {e}")

    def _adapt_ports(self, ports: Dict[str, int]) -> Dict[str, int]:
        """Адаптация портов для избежания конфликтов"""
        # Смещение портов x0tta6bl4-next на +1000
        adapted_ports = {}
        for service, port in ports.items():
            adapted_ports[service] = port + 1000
        return adapted_ports

    def _migrate_k8s_manifests(self) -> bool:
        """Миграция Kubernetes манифестов"""
        logger.info("☸️ Миграция Kubernetes манифестов...")

        try:
            target_k8s_dir = self.target_path / "k8s" / "x0tta6bl4_next"

            for k8s_path in self.migration_targets["k8s_manifests"]:
                source_k8s_dir = self.source_path / k8s_path

                if source_k8s_dir.exists():
                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будут мигрированы K8s манифесты: {k8s_path}")
                    else:
                        target_manifest_dir = target_k8s_dir / k8s_path.split('/')[-1]
                        if target_manifest_dir.exists():
                            shutil.rmtree(target_manifest_dir)
                        shutil.copytree(source_k8s_dir, target_manifest_dir)

                        # Адаптация манифестов
                        self._adapt_k8s_manifests(target_manifest_dir)

                        self.migration_stats["k8s_manifests_migrated"] += 1
                        logger.info(f"✅ Мигрированы K8s манифесты: {k8s_path}")
                else:
                    logger.warning(f"⚠️ K8s директория не найдена: {k8s_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции K8s манифестов: {e}")
            self.migration_stats["errors"].append(f"k8s_migration: {str(e)}")
            return False

    def _adapt_k8s_manifests(self, manifests_dir: Path):
        """Адаптация Kubernetes манифестов"""
        try:
            for yaml_file in manifests_dir.rglob("*.yaml"):
                if yaml_file.is_file():
                    with open(yaml_file, 'r') as f:
                        content = f.read()

                    # Замена namespace
                    content = content.replace('namespace: default', 'namespace: x0tta6bl4-unified')
                    content = content.replace('namespace: x0tta6bl4-next', 'namespace: x0tta6bl4-unified')

                    # Добавление unified labels
                    if 'metadata:' in content and 'labels:' not in content:
                        content = content.replace('metadata:', 'metadata:\n  labels:\n    app: x0tta6bl4-unified\n    component: x0tta6bl4-next', 1)

                    with open(yaml_file, 'w') as f:
                        f.write(content)

        except Exception as e:
            logger.warning(f"⚠️ Ошибка адаптации K8s манифестов: {e}")

    def _migrate_source_code(self) -> bool:
        """Миграция исходного кода"""
        logger.info("💻 Миграция исходного кода...")

        try:
            target_src_dir = self.target_path / "src" / "x0tta6bl4_next"

            for src_path in self.migration_targets["source_code"]:
                source_src_dir = self.source_path / src_path

                if source_src_dir.exists():
                    if source_src_dir.is_file():
                        # Копирование отдельного файла
                        target_file = target_src_dir / source_src_dir.name
                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет скопирован файл: {src_path}")
                        else:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_src_dir, target_file)
                            logger.info(f"✅ Скопирован файл: {src_path}")
                    else:
                        # Копирование директории
                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет скопирована директория: {src_path}")
                        else:
                            target_subdir = target_src_dir / source_src_dir.name
                            if target_subdir.exists():
                                shutil.rmtree(target_subdir)
                            shutil.copytree(source_src_dir, target_subdir)
                            logger.info(f"✅ Скопирована директория: {src_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции исходного кода: {e}")
            self.migration_stats["errors"].append(f"source_code_migration: {str(e)}")
            return False

    def _migrate_tests(self) -> bool:
        """Миграция тестов"""
        logger.info("🧪 Миграция тестов...")

        try:
            target_tests_dir = self.target_path / "tests" / "x0tta6bl4_next"

            for test_path in self.migration_targets["tests"]:
                source_test_dir = self.source_path / test_path

                if source_test_dir.exists():
                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будут мигрированы тесты: {test_path}")
                    else:
                        if target_tests_dir.exists():
                            shutil.rmtree(target_tests_dir)
                        shutil.copytree(source_test_dir, target_tests_dir)

                        self.migration_stats["tests_migrated"] += 1
                        logger.info(f"✅ Мигрированы тесты: {test_path}")
                else:
                    logger.warning(f"⚠️ Директория тестов не найдена: {test_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции тестов: {e}")
            self.migration_stats["errors"].append(f"tests_migration: {str(e)}")
            return False

    def _migrate_documentation(self) -> bool:
        """Миграция документации"""
        logger.info("📚 Миграция документации...")

        try:
            target_docs_dir = self.target_path / "docs" / "x0tta6bl4_next"

            # Копирование основных документов
            docs_to_copy = [
                "README.md",
                "CHANGELOG.md",
                "CONTRIBUTING.md",
                "docs/"
            ]

            for doc_path in docs_to_copy:
                source_doc = self.source_path / doc_path

                if source_doc.exists():
                    if source_doc.is_file():
                        target_file = target_docs_dir / source_doc.name
                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет скопирована документация: {doc_path}")
                        else:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_doc, target_file)
                            self.migration_stats["docs_migrated"] += 1
                            logger.info(f"✅ Скопирована документация: {doc_path}")
                    else:
                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет скопирована директория документации: {doc_path}")
                        else:
                            target_subdir = target_docs_dir / source_doc.name
                            if target_subdir.exists():
                                shutil.rmtree(target_subdir)
                            shutil.copytree(source_doc, target_subdir)
                            logger.info(f"✅ Скопирована директория документации: {doc_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции документации: {e}")
            self.migration_stats["errors"].append(f"documentation_migration: {str(e)}")
            return False

    def get_migration_report(self) -> Dict[str, Any]:
        """Получение отчета о миграции"""
        return {
            "source_project": "x0tta6bl4-next",
            "target_project": "x0tta6bl4-unified",
            "timestamp": datetime.now().isoformat(),
            "stats": self.migration_stats,
            "dry_run": self.dry_run
        }