#!/usr/bin/env python3
"""
🔄 Migration Rollback Script
Скрипт отката миграции x0tta6bl4-unified
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RollbackHandler:
    """Обработчик отката миграции"""

    def __init__(self, migration_plan: Optional[Dict[str, Any]] = None):
        self.migration_plan = migration_plan or self._load_migration_plan()
        self.rollback_results = {
            "timestamp": datetime.now().isoformat(),
            "steps_completed": 0,
            "steps_failed": 0,
            "errors": [],
            "warnings": []
        }

    def rollback(self) -> bool:
        """Выполнение отката миграции"""
        logger.info("🔄 Начало отката миграции x0tta6bl4-unified")

        if not self.migration_plan:
            logger.error("❌ План миграции не найден")
            return False

        try:
            # Проверка возможности отката
            if not self._can_rollback():
                logger.error("❌ Откат невозможен - отсутствуют необходимые условия")
                return False

            # Выполнение шагов отката
            rollback_steps = [
                self._stop_services,
                self._restore_backup,
                self._cleanup_unified_files,
                self._restore_original_configs,
                self._restart_services
            ]

            for step_func in rollback_steps:
                step_name = step_func.__name__.replace('_', ' ')
                logger.info(f"🔄 Выполнение шага отката: {step_name}")

                try:
                    success = step_func()
                    if success:
                        self.rollback_results["steps_completed"] += 1
                        logger.info(f"✅ Шаг {step_name} выполнен успешно")
                    else:
                        self.rollback_results["steps_failed"] += 1
                        logger.error(f"❌ Шаг {step_name} завершен с ошибкой")
                        # Продолжаем с другими шагами
                except Exception as e:
                    self.rollback_results["steps_failed"] += 1
                    error_msg = f"{step_name}: {str(e)}"
                    logger.error(f"❌ Критическая ошибка в шаге {step_name}: {e}")
                    self.rollback_results["errors"].append(error_msg)

            # Генерация отчета об откате
            self._generate_rollback_report()

            # Финальный результат
            success = self.rollback_results["steps_failed"] == 0
            if success:
                logger.info("🎉 Откат миграции завершен успешно!")
            else:
                logger.error(f"⚠️ Откат завершен с {self.rollback_results['steps_failed']} ошибками")

            return success

        except Exception as e:
            logger.error(f"❌ Критическая ошибка отката: {e}")
            return False

    def _can_rollback(self) -> bool:
        """Проверка возможности отката"""
        checks = []

        # Проверка наличия плана миграции
        if not self.migration_plan:
            checks.append("План миграции не найден")

        # Проверка наличия резервной копии
        backup_dir = self._find_backup_directory()
        if not backup_dir or not backup_dir.exists():
            checks.append("Резервная копия не найдена")

        # Проверка статуса миграции
        if self.migration_plan.get("status") != "completed":
            checks.append("Миграция не была завершена успешно")

        if checks:
            logger.error(f"❌ Невозможно выполнить откат: {'; '.join(checks)}")
            return False

        logger.info("✅ Условия для отката выполнены")
        return True

    def _find_backup_directory(self) -> Optional[Path]:
        """Поиск директории с резервной копией"""
        project_root = Path(__file__).parent.parent.parent

        # Поиск в стандартном месте
        backup_base = project_root.parent
        backup_pattern = "x0tta6bl4-unified-backup-*"

        for backup_dir in backup_base.glob(backup_pattern):
            if backup_dir.is_dir():
                return backup_dir

        return None

    def _load_migration_plan(self) -> Optional[Dict[str, Any]]:
        """Загрузка плана миграции"""
        project_root = Path(__file__).parent.parent.parent
        migration_report = project_root / "migration_report.json"

        if migration_report.exists():
            try:
                with open(migration_report, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Ошибка чтения плана миграции: {e}")

        return None

    def _stop_services(self) -> bool:
        """Остановка сервисов"""
        logger.info("🛑 Остановка сервисов...")

        try:
            # Остановка Docker сервисов
            result = subprocess.run(
                ["docker-compose", "down"],
                cwd=Path(__file__).parent.parent.parent,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"⚠️ Предупреждение при остановке Docker: {result.stderr}")

            # Здесь можно добавить остановку других сервисов
            # TODO: Добавить остановку специфичных сервисов x0tta6bl4

            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Таймаут при остановке сервисов")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка остановки сервисов: {e}")
            return False

    def _restore_backup(self) -> bool:
        """Восстановление из резервной копии"""
        logger.info("📦 Восстановление из резервной копии...")

        try:
            project_root = Path(__file__).parent.parent.parent
            backup_dir = self._find_backup_directory()

            if not backup_dir:
                logger.error("❌ Резервная копия не найдена")
                return False

            # Создание резервной копии текущего состояния перед откатом
            current_backup = project_root.parent / f"x0tta6bl4-unified-pre-rollback-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if project_root.exists():
                shutil.copytree(project_root, current_backup, dirs_exist_ok=True)
                logger.info(f"✅ Создана резервная копия текущего состояния: {current_backup}")

            # Удаление текущего содержимого
            for item in project_root.iterdir():
                if item.name.startswith('.git'):
                    continue  # Сохраняем Git репозиторий
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)

            # Восстановление из резервной копии
            for item in backup_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, project_root / item.name)
                else:
                    shutil.copytree(item, project_root / item.name, dirs_exist_ok=True)

            logger.info(f"✅ Восстановлено из резервной копии: {backup_dir}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка восстановления из резервной копии: {e}")
            return False

    def _cleanup_unified_files(self) -> bool:
        """Очистка файлов unified системы"""
        logger.info("🧹 Очистка файлов unified системы...")

        try:
            project_root = Path(__file__).parent.parent.parent

            # Удаление специфичных файлов unified
            unified_files = [
                "main.py",
                "config/unified_config.yaml",
                "config/migration_config.json",
                "migration_report.json",
                "validation_report.json"
            ]

            for file_path in unified_files:
                full_path = project_root / file_path
                if full_path.exists():
                    if full_path.is_file():
                        full_path.unlink()
                        logger.info(f"🗑️ Удален файл: {file_path}")
                    else:
                        shutil.rmtree(full_path)
                        logger.info(f"🗑️ Удалена директория: {file_path}")

            # Удаление production директории (созданной при миграции)
            production_dir = project_root / "production"
            if production_dir.exists():
                shutil.rmtree(production_dir)
                logger.info("🗑️ Удалена директория production")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка очистки unified файлов: {e}")
            return False

    def _restore_original_configs(self) -> bool:
        """Восстановление оригинальных конфигураций"""
        logger.info("⚙️ Восстановление оригинальных конфигураций...")

        try:
            project_root = Path(__file__).parent.parent.parent

            # Восстановление из legacy директорий
            legacy_dirs = [
                ("config/legacy", "config"),
                ("docs/legacy_x0tta6bl4", "docs"),
                ("data/legacy_databases", "data")
            ]

            for legacy_dir, target_dir in legacy_dirs:
                legacy_path = project_root / legacy_dir
                target_path = project_root / target_dir

                if legacy_path.exists():
                    # Перемещение содержимого обратно
                    for item in legacy_path.iterdir():
                        dest = target_path / item.name
                        if item.is_file():
                            shutil.copy2(item, dest)
                        else:
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)

                    # Удаление legacy директории
                    shutil.rmtree(legacy_path)
                    logger.info(f"✅ Восстановлена конфигурация из {legacy_dir}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка восстановления конфигураций: {e}")
            return False

    def _restart_services(self) -> bool:
        """Перезапуск сервисов"""
        logger.info("🚀 Перезапуск сервисов...")

        try:
            project_root = Path(__file__).parent.parent.parent

            # Перезапуск Docker сервисов
            result = subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logger.warning(f"⚠️ Предупреждение при перезапуске Docker: {result.stderr}")
                return False

            logger.info("✅ Сервисы перезапущены")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ Таймаут при перезапуске сервисов")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка перезапуска сервисов: {e}")
            return False

    def _generate_rollback_report(self):
        """Генерация отчета об откате"""
        logger.info("📋 Генерация отчета об откате...")

        report = {
            "rollback_report": {
                "timestamp": self.rollback_results["timestamp"],
                "migration_plan": self.migration_plan.get("started_at", "unknown") if self.migration_plan else "unknown",
                "summary": {
                    "steps_completed": self.rollback_results["steps_completed"],
                    "steps_failed": self.rollback_results["steps_failed"],
                    "total_steps": self.rollback_results["steps_completed"] + self.rollback_results["steps_failed"],
                    "success_rate": (self.rollback_results["steps_completed"] /
                                   (self.rollback_results["steps_completed"] + self.rollback_results["steps_failed"]) * 100)
                    if (self.rollback_results["steps_completed"] + self.rollback_results["steps_failed"]) > 0 else 0
                },
                "errors": self.rollback_results["errors"],
                "warnings": self.rollback_results["warnings"]
            }
        }

        # Сохранение отчета
        project_root = Path(__file__).parent.parent.parent
        report_file = project_root / "rollback_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Отчет об откате сохранен: {report_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения отчета об откате: {e}")

    def get_rollback_results(self) -> Dict[str, Any]:
        """Получение результатов отката"""
        return self.rollback_results

def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Откат миграции x0tta6bl4 Unified")
    parser.add_argument("--force", action="store_true", help="Принудительный откат без дополнительных проверок")
    parser.add_argument("--migration-report", type=str, help="Путь к файлу отчета о миграции")

    args = parser.parse_args()

    # Загрузка плана миграции
    migration_plan = None
    if args.migration_report:
        try:
            with open(args.migration_report, 'r') as f:
                migration_plan = json.load(f)
        except Exception as e:
            logger.error(f"❌ Ошибка чтения отчета о миграции: {e}")
            sys.exit(1)

    rollback_handler = RollbackHandler(migration_plan)

    if args.force:
        logger.warning("⚠️ Выполняется принудительный откат!")

    success = rollback_handler.rollback()

    if success:
        print("\n🎯 РЕКОМЕНДАЦИИ ПОСЛЕ ОТКАТА:")
        print("- Проверьте работоспособность системы")
        print("- Восстановите данные из резервных копий при необходимости")
        print("- Проверьте конфигурационные файлы")
    else:
        print("\n🔧 ПРОБЛЕМЫ ПРИ ОТКАТЕ:")
        print("- Некоторые шаги отката завершились с ошибками")
        print("- Проверьте логи для получения подробной информации")
        print("- Возможно потребуется ручное вмешательство")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()