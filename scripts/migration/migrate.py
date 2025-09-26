#!/usr/bin/env python3
"""
🚀 Main Migration Script for x0tta6bl4 Unified
Основной скрипт миграции для объединения x0tta6bl4 и x0tta6bl4-next в x0tta6bl4-unified

Usage:
    python migrate.py [--dry-run] [--rollback] [--validate-only]

Options:
    --dry-run       Показать план миграции без выполнения
    --rollback      Выполнить откат миграции
    --validate-only Проверить только валидность после миграции
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import shutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationOrchestrator:
    """Оркестратор миграции x0tta6bl4 Unified"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent.parent
        self.source_x0tta6bl4 = Path("/home/x0tta6bl4")
        self.source_x0tta6bl4_next = Path("/home/x0tta6bl4-next")
        self.target_unified = Path("/home/x0tta6bl4-unified")

        self.migration_plan = {
            "started_at": datetime.now().isoformat(),
            "phases": [
                "pre_migration_checks",
                "data_migration_x0tta6bl4",
                "data_migration_x0tta6bl4_next",
                "config_migration",
                "service_migration",
                "validation",
                "cleanup"
            ],
            "status": "initialized",
            "completed_phases": [],
            "failed_phases": [],
            "warnings": [],
            "errors": []
        }

        # Импорт миграционных модулей
        self.data_migrator_x0tta6bl4 = None
        self.data_migrator_x0tta6bl4_next = None
        self.config_migrator = None
        self.validator = None
        self.rollback_handler = None

    def run_migration(self) -> bool:
        """Запуск полной миграции"""
        logger.info("🚀 Начало миграции x0tta6bl4 Unified")

        try:
            # Phase 1: Предварительные проверки
            if not self._run_phase("pre_migration_checks", self._pre_migration_checks):
                return False

            # Phase 2: Миграция данных из x0tta6bl4
            if not self._run_phase("data_migration_x0tta6bl4", self._migrate_x0tta6bl4_data):
                return False

            # Phase 3: Миграция данных из x0tta6bl4-next
            if not self._run_phase("data_migration_x0tta6bl4_next", self._migrate_x0tta6bl4_next_data):
                return False

            # Phase 4: Миграция конфигураций
            if not self._run_phase("config_migration", self._migrate_configurations):
                return False

            # Phase 5: Миграция сервисов
            if not self._run_phase("service_migration", self._migrate_services):
                return False

            # Phase 6: Валидация
            if not self._run_phase("validation", self._validate_migration):
                return False

            # Phase 7: Очистка
            if not self._run_phase("cleanup", self._cleanup):
                return False

            self.migration_plan["status"] = "completed"
            self.migration_plan["completed_at"] = datetime.now().isoformat()

            logger.info("✅ Миграция завершена успешно!")
            return True

        except Exception as e:
            logger.error(f"❌ Критическая ошибка миграции: {e}")
            self.migration_plan["status"] = "failed"
            self.migration_plan["error"] = str(e)
            return False

    def run_rollback(self) -> bool:
        """Запуск отката миграции"""
        logger.info("🔄 Начало отката миграции")

        try:
            if not self.rollback_handler:
                self.rollback_handler = RollbackHandler(self.migration_plan)

            success = self.rollback_handler.rollback()
            if success:
                logger.info("✅ Откат миграции завершен успешно")
            else:
                logger.error("❌ Откат миграции завершен с ошибками")

            return success

        except Exception as e:
            logger.error(f"❌ Ошибка отката: {e}")
            return False

    def run_validation_only(self) -> bool:
        """Запуск только валидации"""
        logger.info("🔍 Запуск валидации миграции")

        try:
            if not self.validator:
                self.validator = MigrationValidator(self.target_unified)

            success = self.validator.validate_all()
            if success:
                logger.info("✅ Валидация пройдена успешно")
            else:
                logger.error("❌ Валидация выявилa ошибки")

            return success

        except Exception as e:
            logger.error(f"❌ Ошибка валидации: {e}")
            return False

    def _run_phase(self, phase_name: str, phase_func) -> bool:
        """Выполнение фазы миграции"""
        logger.info(f"🔄 Выполнение фазы: {phase_name}")

        try:
            if self.dry_run:
                logger.info(f"📋 [DRY RUN] Фаза {phase_name} будет выполнена")
                self.migration_plan["completed_phases"].append(phase_name)
                return True

            success = phase_func()
            if success:
                logger.info(f"✅ Фаза {phase_name} выполнена успешно")
                self.migration_plan["completed_phases"].append(phase_name)
            else:
                logger.error(f"❌ Фаза {phase_name} завершилась с ошибкой")
                self.migration_plan["failed_phases"].append(phase_name)

            return success

        except Exception as e:
            logger.error(f"❌ Критическая ошибка в фазе {phase_name}: {e}")
            self.migration_plan["failed_phases"].append(phase_name)
            self.migration_plan["errors"].append(f"{phase_name}: {str(e)}")
            return False

    def _pre_migration_checks(self) -> bool:
        """Предварительные проверки перед миграцией"""
        logger.info("🔍 Выполнение предварительных проверок...")

        checks = [
            self._check_source_projects_exist,
            self._check_target_directory_ready,
            self._check_dependencies,
            self._backup_existing_data
        ]

        for check_func in checks:
            if not check_func():
                return False

        logger.info("✅ Предварительные проверки пройдены")
        return True

    def _check_source_projects_exist(self) -> bool:
        """Проверка существования исходных проектов"""
        sources = [
            ("x0tta6bl4", self.source_x0tta6bl4),
            ("x0tta6bl4-next", self.source_x0tta6bl4_next)
        ]

        for name, path in sources:
            if not path.exists():
                logger.error(f"❌ Исходный проект {name} не найден: {path}")
                return False
            logger.info(f"✅ Исходный проект {name} найден: {path}")

        return True

    def _check_target_directory_ready(self) -> bool:
        """Проверка готовности целевого каталога"""
        if not self.target_unified.exists():
            logger.error(f"❌ Целевой каталог не существует: {self.target_unified}")
            return False

        # Проверка прав на запись
        try:
            test_file = self.target_unified / ".migration_test"
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✅ Целевой каталог готов к миграции")
            return True
        except Exception as e:
            logger.error(f"❌ Нет прав на запись в целевой каталог: {e}")
            return False

    def _check_dependencies(self) -> bool:
        """Проверка зависимостей"""
        required_commands = ["python3", "pip", "git"]

        for cmd in required_commands:
            if not shutil.which(cmd):
                logger.error(f"❌ Команда не найдена: {cmd}")
                return False

        logger.info("✅ Зависимости проверены")
        return True

    def _backup_existing_data(self) -> bool:
        """Создание резервной копии существующих данных"""
        try:
            backup_dir = self.target_unified.parent / f"x0tta6bl4-unified-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if self.target_unified.exists():
                if self.dry_run:
                    logger.info(f"📋 [DRY RUN] Будет создана резервная копия: {backup_dir}")
                else:
                    shutil.copytree(self.target_unified, backup_dir, dirs_exist_ok=True)
                    logger.info(f"✅ Резервная копия создана: {backup_dir}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return False

    def _migrate_x0tta6bl4_data(self) -> bool:
        """Миграция данных из x0tta6bl4"""
        logger.info("📊 Миграция данных из x0tta6bl4...")

        if not self.data_migrator_x0tta6bl4:
            from .data_migration_x0tta6bl4 import X0tta6bl4DataMigrator
            self.data_migrator_x0tta6bl4 = X0tta6bl4DataMigrator(
                self.source_x0tta6bl4,
                self.target_unified,
                dry_run=self.dry_run
            )

        return self.data_migrator_x0tta6bl4.migrate()

    def _migrate_x0tta6bl4_next_data(self) -> bool:
        """Миграция данных из x0tta6bl4-next"""
        logger.info("📊 Миграция данных из x0tta6bl4-next...")

        if not self.data_migrator_x0tta6bl4_next:
            from .data_migration_x0tta6bl4_next import X0tta6bl4NextDataMigrator
            self.data_migrator_x0tta6bl4_next = X0tta6bl4NextDataMigrator(
                self.source_x0tta6bl4_next,
                self.target_unified,
                dry_run=self.dry_run
            )

        return self.data_migrator_x0tta6bl4_next.migrate()

    def _migrate_configurations(self) -> bool:
        """Миграция конфигураций"""
        logger.info("⚙️ Миграция конфигураций...")

        if not self.config_migrator:
            from .config_migration import ConfigurationMigrator
            self.config_migrator = ConfigurationMigrator(
                [self.source_x0tta6bl4, self.source_x0tta6bl4_next],
                self.target_unified,
                dry_run=self.dry_run
            )

        return self.config_migrator.migrate()

    def _migrate_services(self) -> bool:
        """Миграция сервисов"""
        logger.info("🔧 Миграция сервисов...")

        # Импорт и запуск миграции сервисов
        try:
            from .service_migration import ServiceMigrator
            service_migrator = ServiceMigrator(
                [self.source_x0tta6bl4, self.source_x0tta6bl4_next],
                self.target_unified,
                dry_run=self.dry_run
            )
            return service_migrator.migrate()
        except ImportError:
            logger.warning("⚠️ Модуль service_migration не найден, пропуск фазы")
            return True

    def _validate_migration(self) -> bool:
        """Валидация миграции"""
        logger.info("✅ Валидация результатов миграции...")

        if not self.validator:
            from .validation import MigrationValidator
            self.validator = MigrationValidator(self.target_unified)

        return self.validator.validate_all()

    def _cleanup(self) -> bool:
        """Очистка после миграции"""
        logger.info("🧹 Очистка после миграции...")

        try:
            # Сохранение отчета о миграции
            report_path = self.target_unified / "migration_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.migration_plan, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Отчет о миграции сохранен: {report_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка очистки: {e}")
            return False

    def generate_report(self) -> str:
        """Генерация отчета о миграции"""
        report = f"""
# 📊 Отчет о миграции x0tta6bl4 Unified

**Дата начала:** {self.migration_plan['started_at']}
**Дата окончания:** {self.migration_plan.get('completed_at', 'N/A')}
**Статус:** {self.migration_plan['status']}

## 📈 Статистика

- **Выполненные фазы:** {len(self.migration_plan['completed_phases'])}
- **Неудачные фазы:** {len(self.migration_plan['failed_phases'])}
- **Предупреждения:** {len(self.migration_plan['warnings'])}
- **Ошибки:** {len(self.migration_plan['errors'])}

## ✅ Выполненные фазы

{chr(10).join(f"- ✅ {phase}" for phase in self.migration_plan['completed_phases'])}

## ❌ Неудачные фазы

{chr(10).join(f"- ❌ {phase}" for phase in self.migration_plan['failed_phases']) if self.migration_plan['failed_phases'] else '- Все фазы выполнены успешно'}

## ⚠️ Предупреждения

{chr(10).join(f"- ⚠️ {warning}" for warning in self.migration_plan['warnings']) if self.migration_plan['warnings'] else '- Нет предупреждений'}

## 🚨 Ошибки

{chr(10).join(f"- 🚨 {error}" for error in self.migration_plan['errors']) if self.migration_plan['errors'] else '- Нет ошибок'}

---
*Отчет сгенерирован автоматически системой миграции x0tta6bl4 Unified*
"""

        return report

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Миграция x0tta6bl4 Unified")
    parser.add_argument("--dry-run", action="store_true", help="Показать план миграции без выполнения")
    parser.add_argument("--rollback", action="store_true", help="Выполнить откат миграции")
    parser.add_argument("--validate-only", action="store_true", help="Проверить только валидность")

    args = parser.parse_args()

    # Проверка конфликтующих опций
    if sum([args.rollback, args.validate_only]) > 1:
        logger.error("❌ Можно использовать только одну опцию: --rollback или --validate-only")
        sys.exit(1)

    orchestrator = MigrationOrchestrator(dry_run=args.dry_run)

    if args.rollback:
        success = orchestrator.run_rollback()
    elif args.validate_only:
        success = orchestrator.run_validation_only()
    else:
        success = orchestrator.run_migration()

    # Вывод отчета
    if not args.dry_run:
        report = orchestrator.generate_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()