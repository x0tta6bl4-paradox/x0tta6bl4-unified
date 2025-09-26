#!/usr/bin/env python3
"""
⚙️ Configuration Migration Script
Скрипт миграции и объединения конфигураций из x0tta6bl4 и x0tta6bl4-next
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import shutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationMigrator:
    """Мигратор конфигураций"""

    def __init__(self, source_paths: List[Path], target_path: Path, dry_run: bool = False):
        self.source_paths = source_paths
        self.target_path = target_path
        self.dry_run = dry_run

        self.migration_stats = {
            "configs_processed": 0,
            "configs_merged": 0,
            "conflicts_resolved": 0,
            "unified_config_created": False,
            "errors": []
        }

        # Приоритеты источников (x0tta6bl4-next имеет больший приоритет)
        self.source_priorities = {
            "x0tta6bl4-next": 2,
            "x0tta6bl4": 1
        }

    def migrate(self) -> bool:
        """Основной метод миграции конфигураций"""
        logger.info("🚀 Начало миграции конфигураций")

        try:
            # Создание директории конфигураций
            self._create_config_structure()

            # Сбор всех конфигураций
            all_configs = self._collect_all_configs()

            # Объединение конфигураций
            unified_config = self._merge_configurations(all_configs)

            # Создание unified конфигурации
            if not self._create_unified_config(unified_config):
                return False

            # Миграция специфичных конфигураций
            if not self._migrate_specific_configs(all_configs):
                return False

            # Создание конфигурации миграции
            if not self._create_migration_config():
                return False

            logger.info("✅ Миграция конфигураций завершена успешно")
            logger.info(f"📊 Статистика: {self.migration_stats}")
            return True

        except Exception as e:
            logger.error(f"❌ Критическая ошибка миграции конфигураций: {e}")
            self.migration_stats["errors"].append(str(e))
            return False

    def _create_config_structure(self):
        """Создание структуры директорий конфигураций"""
        logger.info("📁 Создание структуры конфигураций...")

        directories = [
            "config",
            "config/legacy",
            "config/services",
            "config/environments"
        ]

        for dir_path in directories:
            full_path = self.target_path / dir_path
            if self.dry_run:
                logger.info(f"📋 [DRY RUN] Будет создана директория: {full_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Создана директория: {full_path}")

    def _collect_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Сбор всех конфигураций из исходных проектов"""
        logger.info("🔍 Сбор конфигураций из исходных проектов...")

        all_configs = {}

        for source_path in self.source_paths:
            if not source_path.exists():
                logger.warning(f"⚠️ Исходный путь не существует: {source_path}")
                continue

            project_name = source_path.name
            all_configs[project_name] = {}

            # Поиск YAML файлов
            for yaml_file in source_path.rglob("*.yaml"):
                if yaml_file.is_file():
                    try:
                        with open(yaml_file, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)

                        if config_data:
                            relative_path = yaml_file.relative_to(source_path)
                            config_key = str(relative_path).replace('/', '.').replace('.yaml', '')
                            all_configs[project_name][config_key] = {
                                'data': config_data,
                                'file': yaml_file,
                                'priority': self.source_priorities.get(project_name, 0)
                            }
                            self.migration_stats["configs_processed"] += 1

                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка чтения конфига {yaml_file}: {e}")

            # Поиск JSON файлов
            for json_file in source_path.rglob("*.json"):
                if json_file.is_file():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)

                        relative_path = json_file.relative_to(source_path)
                        config_key = str(relative_path).replace('/', '.').replace('.json', '')
                        all_configs[project_name][config_key] = {
                            'data': config_data,
                            'file': json_file,
                            'priority': self.source_priorities.get(project_name, 0)
                        }
                        self.migration_stats["configs_processed"] += 1

                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка чтения конфига {json_file}: {e}")

            logger.info(f"📊 Собрано конфигураций из {project_name}: {len(all_configs[project_name])}")

        return all_configs

    def _merge_configurations(self, all_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение конфигураций из разных источников"""
        logger.info("🔀 Объединение конфигураций...")

        unified_config = {
            'system': {
                'name': 'x0tta6bl4-unified',
                'version': '1.0.0',
                'environment': 'production',
                'sacred_frequency': 108,
                'phi_constant': 1.618033988749895,
                'created_at': datetime.now().isoformat()
            },
            'sources': {},
            'merged_configs': {}
        }

        # Группировка конфигураций по ключам
        config_groups = {}

        for project_name, configs in all_configs.items():
            unified_config['sources'][project_name] = {
                'path': str(self.source_paths[0] if project_name in str(self.source_paths[0]) else self.source_paths[1]),
                'configs_count': len(configs),
                'priority': self.source_priorities.get(project_name, 0)
            }

            for config_key, config_info in configs.items():
                if config_key not in config_groups:
                    config_groups[config_key] = []
                config_groups[config_key].append({
                    'project': project_name,
                    'data': config_info['data'],
                    'priority': config_info['priority'],
                    'file': config_info['file']
                })

        # Объединение каждой группы конфигураций
        for config_key, config_list in config_groups.items():
            if len(config_list) == 1:
                # Только один источник
                unified_config['merged_configs'][config_key] = config_list[0]['data']
            else:
                # Несколько источников - разрешение конфликтов
                merged_data = self._resolve_config_conflicts(config_key, config_list)
                unified_config['merged_configs'][config_key] = merged_data
                self.migration_stats["configs_merged"] += 1

        logger.info(f"🔀 Объединено конфигураций: {len(unified_config['merged_configs'])}")
        return unified_config

    def _resolve_config_conflicts(self, config_key: str, config_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Разрешение конфликтов конфигураций"""
        logger.info(f"⚖️ Разрешение конфликтов для {config_key}...")

        # Сортировка по приоритету (высший приоритет первый)
        sorted_configs = sorted(config_list, key=lambda x: x['priority'], reverse=True)

        # Использование конфигурации с высшим приоритетом как базу
        base_config = sorted_configs[0]['data'].copy()

        # Специфическая логика объединения для разных типов конфигураций
        if config_key == 'config.x0tta6bl4_config':
            base_config = self._merge_x0tta6bl4_configs(sorted_configs)
        elif 'ports' in base_config:
            base_config = self._merge_port_configs(sorted_configs)
        elif 'agents' in base_config:
            base_config = self._merge_agent_configs(sorted_configs)
        elif 'quantum' in base_config:
            base_config = self._merge_quantum_configs(sorted_configs)
        else:
            # Общее объединение
            base_config = self._merge_generic_configs(sorted_configs)

        self.migration_stats["conflicts_resolved"] += 1
        return base_config

    def _merge_x0tta6bl4_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение основных конфигураций x0tta6bl4"""
        base_config = configs[0]['data'].copy()

        for config in configs[1:]:
            # Объединение портов
            if 'ports' in config['data']:
                base_config['ports'].update(config['data']['ports'])

            # Объединение агентов
            if 'agents' in config['data']:
                if 'agents' not in base_config:
                    base_config['agents'] = {}
                self._deep_merge(base_config['agents'], config['data']['agents'])

            # Объединение квантовых настроек
            if 'quantum' in config['data']:
                if 'quantum' not in base_config:
                    base_config['quantum'] = {}
                self._deep_merge(base_config['quantum'], config['data']['quantum'])

        return base_config

    def _merge_port_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение конфигураций портов"""
        base_config = configs[0]['data'].copy()

        # Сбор всех портов
        all_ports = {}
        for config in configs:
            if 'ports' in config['data']:
                all_ports.update(config['data']['ports'])

        # Разрешение конфликтов портов
        resolved_ports = {}
        used_ports = set()

        for service, port in all_ports.items():
            if port not in used_ports:
                resolved_ports[service] = port
                used_ports.add(port)
            else:
                # Поиск свободного порта
                new_port = port
                while new_port in used_ports:
                    new_port += 1
                resolved_ports[service] = new_port
                used_ports.add(new_port)
                logger.info(f"🔧 Разрешен конфликт порта для {service}: {port} -> {new_port}")

        base_config['ports'] = resolved_ports
        return base_config

    def _merge_agent_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение конфигураций агентов"""
        base_config = configs[0]['data'].copy()

        for config in configs[1:]:
            if 'agents' in config['data']:
                if 'agents' not in base_config:
                    base_config['agents'] = {}
                self._deep_merge(base_config['agents'], config['data']['agents'])

        return base_config

    def _merge_quantum_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение квантовых конфигураций"""
        base_config = configs[0]['data'].copy()

        for config in configs[1:]:
            if 'quantum' in config['data']:
                if 'quantum' not in base_config:
                    base_config['quantum'] = {}
                self._deep_merge(base_config['quantum'], config['data']['quantum'])

        return base_config

    def _merge_generic_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Общее объединение конфигураций"""
        base_config = configs[0]['data'].copy()

        for config in configs[1:]:
            self._deep_merge(base_config, config['data'])

        return base_config

    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Глубокое объединение словарей"""
        for key, value in update_dict.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    self._deep_merge(base_dict[key], value)
                elif isinstance(base_dict[key], list) and isinstance(value, list):
                    base_dict[key].extend(value)
                else:
                    # Конфликт - сохраняем значение с высшим приоритетом
                    pass
            else:
                base_dict[key] = value

    def _create_unified_config(self, unified_config: Dict[str, Any]) -> bool:
        """Создание unified конфигурационного файла"""
        logger.info("📝 Создание unified конфигурации...")

        try:
            config_file = self.target_path / "config" / "unified_config.yaml"

            if self.dry_run:
                logger.info(f"📋 [DRY RUN] Будет создан unified конфиг: {config_file}")
            else:
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(unified_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

                self.migration_stats["unified_config_created"] = True
                logger.info(f"✅ Создан unified конфиг: {config_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка создания unified конфига: {e}")
            self.migration_stats["errors"].append(f"unified_config_creation: {str(e)}")
            return False

    def _migrate_specific_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Миграция специфичных конфигураций"""
        logger.info("🔧 Миграция специфичных конфигураций...")

        try:
            # Миграция конфигураций сервисов
            if not self._migrate_service_configs(all_configs):
                return False

            # Миграция конфигураций окружений
            if not self._migrate_environment_configs(all_configs):
                return False

            # Миграция конфигураций мониторинга
            if not self._migrate_monitoring_configs(all_configs):
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка миграции специфичных конфигов: {e}")
            self.migration_stats["errors"].append(f"specific_configs_migration: {str(e)}")
            return False

    def _migrate_service_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Миграция конфигураций сервисов"""
        services_config = {
            'services': {},
            'ports': {},
            'dependencies': {}
        }

        # Сбор информации о сервисах из всех источников
        for project_name, configs in all_configs.items():
            for config_key, config_info in configs.items():
                if 'services' in config_info['data']:
                    services_config['services'].update(config_info['data']['services'])
                if 'ports' in config_info['data']:
                    services_config['ports'].update(config_info['data']['ports'])

        # Сохранение конфигурации сервисов
        services_file = self.target_path / "config" / "services" / "services_config.yaml"
        if not self.dry_run:
            services_file.parent.mkdir(parents=True, exist_ok=True)
            with open(services_file, 'w', encoding='utf-8') as f:
                yaml.dump(services_config, f, default_flow_style=False, allow_unicode=True)

        logger.info("✅ Мигрированы конфигурации сервисов")
        return True

    def _migrate_environment_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Миграция конфигураций окружений"""
        environments = ['development', 'staging', 'production']

        for env in environments:
            env_config = {
                'environment': env,
                'services': {},
                'monitoring': {},
                'security': {}
            }

            # Сбор настроек для каждого окружения
            for project_name, configs in all_configs.items():
                for config_key, config_info in configs.items():
                    if env in config_key.lower():
                        env_config.update(config_info['data'])

            # Сохранение конфигурации окружения
            env_file = self.target_path / "config" / "environments" / f"{env}_config.yaml"
            if not self.dry_run:
                env_file.parent.mkdir(parents=True, exist_ok=True)
                with open(env_file, 'w', encoding='utf-8') as f:
                    yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)

        logger.info("✅ Мигрированы конфигурации окружений")
        return True

    def _migrate_monitoring_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> bool:
        """Миграция конфигураций мониторинга"""
        monitoring_config = {
            'prometheus': {},
            'grafana': {},
            'alertmanager': {},
            'metrics': {},
            'dashboards': []
        }

        # Сбор настроек мониторинга
        for project_name, configs in all_configs.items():
            for config_key, config_info in configs.items():
                if 'monitoring' in config_info['data']:
                    self._deep_merge(monitoring_config, config_info['data']['monitoring'])

        # Сохранение конфигурации мониторинга
        monitoring_file = self.target_path / "config" / "monitoring_config.yaml"
        if not self.dry_run:
            with open(monitoring_file, 'w', encoding='utf-8') as f:
                yaml.dump(monitoring_config, f, default_flow_style=False, allow_unicode=True)

        logger.info("✅ Мигрированы конфигурации мониторинга")
        return True

    def _create_migration_config(self) -> bool:
        """Создание конфигурации миграции"""
        logger.info("📋 Создание конфигурации миграции...")

        try:
            migration_config = {
                'migration': {
                    'version': '1.0.0',
                    'timestamp': datetime.now().isoformat(),
                    'sources': list(self.source_priorities.keys()),
                    'target': 'x0tta6bl4-unified',
                    'stats': self.migration_stats
                },
                'backup': {
                    'enabled': True,
                    'retention_days': 30,
                    'location': 'scripts/migration/backups'
                },
                'rollback': {
                    'enabled': True,
                    'backup_required': True
                }
            }

            config_file = self.target_path / "config" / "migration_config.json"
            if not self.dry_run:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(migration_config, f, indent=2, ensure_ascii=False)

            logger.info("✅ Создана конфигурация миграции")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка создания конфигурации миграции: {e}")
            self.migration_stats["errors"].append(f"migration_config_creation: {str(e)}")
            return False

    def get_migration_report(self) -> Dict[str, Any]:
        """Получение отчета о миграции"""
        return {
            "operation": "configuration_migration",
            "timestamp": datetime.now().isoformat(),
            "stats": self.migration_stats,
            "dry_run": self.dry_run
        }