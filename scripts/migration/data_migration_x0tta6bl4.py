#!/usr/bin/env python3
"""
📊 Data Migration Script for x0tta6bl4
Скрипт миграции данных из оригинального проекта x0tta6bl4 в x0tta6bl4-unified
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set
import sqlite3
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class X0tta6bl4DataMigrator:
    """Мигратор данных из x0tta6bl4"""

    def __init__(self, source_path: Path, target_path: Path, dry_run: bool = False):
        self.source_path = source_path
        self.target_path = target_path
        self.dry_run = dry_run

        self.migration_stats = {
            "files_copied": 0,
            "files_skipped": 0,
            "databases_migrated": 0,
            "configs_migrated": 0,
            "errors": []
        }

        # Определение ключевых файлов и директорий для миграции
        self.migration_targets = {
            "quantum_components": [
                "final_launch_system_fixed.py",
                "hawking_entropy_engine.py",
                "quantum_bypass_solver.py",
                "direct_cultural_quantum_test.py",
                "enhanced_quantum_system.py",
                "quantum_api_server.py",
                "quantum_chemistry_applications.py",
                "quantum_neural_network.py",
                "quantum_tetrahedron_cocreation.py"
            ],
            "ai_components": [
                "advanced_ai_ml_system.py",
                "edge/atom_ai.py",
                "edge/micromind_prepare.py",
                "working_ai_agents.py",
                "ai_ml_enhancement_system.py",
                "ai_models_improvement.py"
            ],
            "enterprise_components": [
                "commercialization_preparation_system.py",
                "enterprise_dashboard_architecture.md",
                "launch_commercial_services.py",
                "saas_platform.py"
            ],
            "api_components": [
                "api_implementation.py",
                "api_local.py",
                "start_api.py",
                "start_main_api.sh"
            ],
            "monitoring_components": [
                "system_monitor.py",
                "phi_metrics_monitor.py",
                "logging_enhancer.py"
            ],
            "config_files": [
                "*.yaml",
                "*.yml",
                "*.json",
                "*.conf",
                "*.cfg"
            ],
            "database_files": [
                "*.db",
                "*.sqlite",
                "*.sqlite3"
            ]
        }

    def migrate(self) -> bool:
        """Основной метод миграции"""
        logger.info("🚀 Начало миграции данных из x0tta6bl4")

        try:
            # Создание структуры директорий
            self._create_target_structure()

            # Миграция квантовых компонентов
            if not self._migrate_quantum_components():
                return False

            # Миграция AI компонентов
            if not self._migrate_ai_components():
                return False

            # Миграция enterprise компонентов
            if not self._migrate_enterprise_components():
                return False

            # Миграция API компонентов
            if not self._migrate_api_components():
                return False

            # Миграция компонентов мониторинга
            if not self._migrate_monitoring_components():
                return False

            # Миграция конфигурационных файлов
            if not self._migrate_config_files():
                return False

            # Миграция баз данных
            if not self._migrate_databases():
                return False

            # Миграция документации
            if not self._migrate_documentation():
                return False

            logger.info("✅ Миграция данных из x0tta6bl4 завершена успешно")
            logger.info(f"📊 Статистика: {self.migration_stats}")
            return True

        except Exception as e:
            logger.error(f"❌ Критическая ошибка миграции: {e}")
            self.migration_stats["errors"].append(str(e))
            return False

    def _create_target_structure(self):
        """Создание структуры директорий в целевом проекте"""
        logger.info("📁 Создание структуры директорий...")

        directories = [
            "production/quantum",
            "production/ai",
            "production/enterprise",
            "production/api",
            "production/monitoring",
            "config",
            "data",
            "docs",
            "scripts/migration/backups"
        ]

        for dir_path in directories:
            full_path = self.target_path / dir_path
            if self.dry_run:
                logger.info(f"📋 [DRY RUN] Будет создана директория: {full_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Создана директория: {full_path}")

    def _migrate_quantum_components(self) -> bool:
        """Миграция квантовых компонентов"""
        logger.info("⚛️ Миграция квантовых компонентов...")

        target_dir = self.target_path / "production" / "quantum"
        source_files = self.migration_targets["quantum_components"]

        return self._copy_files_by_list(source_files, target_dir, "quantum")

    def _migrate_ai_components(self) -> bool:
        """Миграция AI компонентов"""
        logger.info("🤖 Миграция AI компонентов...")

        target_dir = self.target_path / "production" / "ai"
        source_files = self.migration_targets["ai_components"]

        return self._copy_files_by_list(source_files, target_dir, "ai")

    def _migrate_enterprise_components(self) -> bool:
        """Миграция enterprise компонентов"""
        logger.info("🏢 Миграция enterprise компонентов...")

        target_dir = self.target_path / "production" / "enterprise"
        source_files = self.migration_targets["enterprise_components"]

        return self._copy_files_by_list(source_files, target_dir, "enterprise")

    def _migrate_api_components(self) -> bool:
        """Миграция API компонентов"""
        logger.info("🔌 Миграция API компонентов...")

        target_dir = self.target_path / "production" / "api"
        source_files = self.migration_targets["api_components"]

        return self._copy_files_by_list(source_files, target_dir, "api")

    def _migrate_monitoring_components(self) -> bool:
        """Миграция компонентов мониторинга"""
        logger.info("📊 Миграция компонентов мониторинга...")

        target_dir = self.target_path / "production" / "monitoring"
        source_files = self.migration_targets["monitoring_components"]

        return self._copy_files_by_list(source_files, target_dir, "monitoring")

    def _migrate_config_files(self) -> bool:
        """Миграция конфигурационных файлов"""
        logger.info("⚙️ Миграция конфигурационных файлов...")

        try:
            target_dir = self.target_path / "config" / "legacy_x0tta6bl4"
            target_dir.mkdir(parents=True, exist_ok=True)

            config_patterns = self.migration_targets["config_files"]

            for pattern in config_patterns:
                for file_path in self.source_path.rglob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.source_path)
                        target_file = target_dir / relative_path

                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет скопирован конфиг: {file_path} -> {target_file}")
                        else:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, target_file)
                            self.migration_stats["configs_migrated"] += 1
                            logger.info(f"✅ Скопирован конфиг: {relative_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции конфигов: {e}")
            self.migration_stats["errors"].append(f"config_migration: {str(e)}")
            return False

    def _migrate_databases(self) -> bool:
        """Миграция баз данных"""
        logger.info("🗄️ Миграция баз данных...")

        try:
            target_dir = self.target_path / "data" / "legacy_databases"
            target_dir.mkdir(parents=True, exist_ok=True)

            db_patterns = self.migration_targets["database_files"]

            for pattern in db_patterns:
                for file_path in self.source_path.rglob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.source_path)
                        target_file = target_dir / relative_path

                        if self.dry_run:
                            logger.info(f"📋 [DRY RUN] Будет мигрирована БД: {file_path} -> {target_file}")
                        else:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, target_file)
                            self.migration_stats["databases_migrated"] += 1
                            logger.info(f"✅ Скопирована БД: {relative_path}")

                            # Попытка миграции схемы БД
                            self._migrate_database_schema(file_path, target_file)

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции БД: {e}")
            self.migration_stats["errors"].append(f"database_migration: {str(e)}")
            return False

    def _migrate_database_schema(self, source_db: Path, target_db: Path):
        """Миграция схемы базы данных"""
        try:
            # Попытка подключения к SQLite БД
            if source_db.suffix in ['.db', '.sqlite', '.sqlite3']:
                conn = sqlite3.connect(str(target_db))
                cursor = conn.cursor()

                # Получение списка таблиц
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                logger.info(f"📋 Найдено таблиц в БД {source_db.name}: {len(tables)}")

                # Здесь можно добавить логику трансформации схемы
                # для совместимости с unified системой

                conn.close()

        except Exception as e:
            logger.warning(f"⚠️ Не удалось проанализировать схему БД {source_db}: {e}")

    def _migrate_documentation(self) -> bool:
        """Миграция документации"""
        logger.info("📚 Миграция документации...")

        try:
            target_dir = self.target_path / "docs" / "legacy_x0tta6bl4"
            target_dir.mkdir(parents=True, exist_ok=True)

            # Копирование Markdown файлов
            for md_file in self.source_path.rglob("*.md"):
                if md_file.is_file():
                    relative_path = md_file.relative_to(self.source_path)
                    target_file = target_dir / relative_path

                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будет скопирована документация: {md_file} -> {target_file}")
                    else:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(md_file, target_file)
                        logger.info(f"✅ Скопирована документация: {relative_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции документации: {e}")
            self.migration_stats["errors"].append(f"documentation_migration: {str(e)}")
            return False

    def _copy_files_by_list(self, file_list: List[str], target_dir: Path, component_type: str) -> bool:
        """Копирование файлов по списку"""
        try:
            for file_name in file_list:
                source_file = self.source_path / file_name

                if source_file.exists():
                    if self.dry_run:
                        logger.info(f"📋 [DRY RUN] Будет скопирован {component_type} файл: {file_name}")
                    else:
                        target_file = target_dir / file_name
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_file, target_file)
                        self.migration_stats["files_copied"] += 1
                        logger.info(f"✅ Скопирован {component_type} файл: {file_name}")
                else:
                    self.migration_stats["files_skipped"] += 1
                    logger.warning(f"⚠️ Файл не найден: {file_name}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка копирования {component_type} файлов: {e}")
            self.migration_stats["errors"].append(f"{component_type}_copy: {str(e)}")
            return False

    def get_migration_report(self) -> Dict[str, Any]:
        """Получение отчета о миграции"""
        return {
            "source_project": "x0tta6bl4",
            "target_project": "x0tta6bl4-unified",
            "timestamp": datetime.now().isoformat(),
            "stats": self.migration_stats,
            "dry_run": self.dry_run
        }