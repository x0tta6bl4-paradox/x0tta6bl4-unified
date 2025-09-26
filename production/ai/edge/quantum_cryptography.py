#!/usr/bin/env python3
"""
🔐 QUANTUM CRYPTOGRAPHY - Edge quantum-secure коммуникации
Quantum Key Distribution (QKD) и quantum-resistant шифрование для edge устройств
"""

import asyncio
import time
import json
import logging
import secrets
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import base64

# Импорт базового компонента
from ...base_interface import BaseComponent

# Импорт квантового интерфейса
try:
    from ...quantum.quantum_interface import QuantumCore
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    QuantumCore = None

# Константы x0tta6bl4
PHI_RATIO = 1.618033988749895
QUANTUM_FACTOR = 2.718281828459045

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Алгоритмы шифрования"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    QUANTUM_RESISTANT_KYBER = "kyber"
    QUANTUM_RESISTANT_DILITHIUM = "dilithium"
    HYBRID_QUANTUM_CLASSICAL = "hybrid"

class KeyExchangeProtocol(Enum):
    """Протоколы обмена ключами"""
    DIFFIE_HELLMAN = "diffie_hellman"
    ECDH_P384 = "ecdh_p384"
    QUANTUM_KEY_DISTRIBUTION = "qkd"
    POST_QUANTUM_KEM = "post_quantum_kem"

class SecurityLevel(Enum):
    """Уровни безопасности"""
    STANDARD = "standard"
    HIGH = "high"
    QUANTUM_SAFE = "quantum_safe"
    MAXIMUM = "maximum"

@dataclass
class QuantumKey:
    """Квантовый ключ"""
    key_id: str
    key_material: bytes
    algorithm: EncryptionAlgorithm
    key_length: int
    generation_time: datetime
    expiry_time: Optional[datetime]
    security_level: SecurityLevel
    quantum_generated: bool
    phi_entropy: float

@dataclass
class SecureChannel:
    """Безопасный канал связи"""
    channel_id: str
    participants: List[str]
    encryption_key: QuantumKey
    protocol: KeyExchangeProtocol
    established_time: datetime
    last_activity: datetime
    message_count: int
    security_status: str

@dataclass
class EncryptedMessage:
    """Зашифрованное сообщение"""
    message_id: str
    sender: str
    recipient: str
    ciphertext: bytes
    nonce: bytes
    auth_tag: bytes
    timestamp: datetime
    algorithm: EncryptionAlgorithm
    key_id: str

class QuantumCryptography(BaseComponent):
    """Quantum-enhanced cryptography для edge устройств"""

    def __init__(self):
        super().__init__("quantum_cryptography")

        # Квантовые компоненты
        self.quantum_core = None

        # Системы шифрования
        self.key_manager = None
        self.encryption_engine = None
        self.qkd_system = None
        self.post_quantum_crypto = None

        # Управление ключами
        self.active_keys: Dict[str, QuantumKey] = {}
        self.key_history: List[QuantumKey] = []
        self.key_rotation_schedule: Dict[str, datetime] = {}

        # Безопасные каналы
        self.secure_channels: Dict[str, SecureChannel] = {}
        self.channel_history: Dict[str, List[SecureChannel]] = {}

        # История сообщений
        self.message_log: List[EncryptedMessage] = []

        # Статистика безопасности
        self.encryption_operations = 0
        self.key_generations = 0
        self.security_breaches = 0
        self.qkd_sessions = 0

        # Конфигурация
        self.quantum_safe = True
        self.key_rotation_interval_hours = 24
        self.max_key_age_days = 7
        self.security_level = SecurityLevel.QUANTUM_SAFE

        logger.info("Quantum Cryptography initialized")

    async def initialize(self) -> bool:
        """Инициализация Quantum Cryptography"""
        try:
            self.logger.info("Инициализация Quantum Cryptography...")

            # Инициализация квантового core
            if QUANTUM_AVAILABLE:
                self.quantum_core = QuantumCore()
                quantum_init = await self.quantum_core.initialize()
                if quantum_init:
                    self.logger.info("Quantum Core для cryptography успешно инициализирован")
                else:
                    self.logger.warning("Quantum Core для cryptography не инициализирован")

            # Инициализация систем шифрования
            await self._initialize_crypto_systems()

            # Генерация начальных ключей
            await self._generate_initial_keys()

            self.set_status("operational")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка инициализации Quantum Cryptography: {e}")
            self.set_status("failed")
            return False

    async def _initialize_crypto_systems(self):
        """Инициализация криптографических систем"""
        try:
            # Key manager
            self.key_manager = {
                "supported_algorithms": [alg.value for alg in EncryptionAlgorithm],
                "key_storage": "quantum_secure_vault",
                "rotation_policy": "automatic",
                "backup_enabled": True
            }

            # Encryption engine
            self.encryption_engine = {
                "primary_algorithm": EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER,
                "fallback_algorithm": EncryptionAlgorithm.AES_256_GCM,
                "performance_mode": "balanced",
                "hardware_acceleration": True
            }

            # QKD system
            self.qkd_system = {
                "protocol": "BB84",
                "key_rate": 1000,  # bits per second
                "distance_limit": 100,  # km
                "error_correction": "cascade",
                "privacy_amplification": True
            }

            # Post-quantum crypto
            self.post_quantum_crypto = {
                "kem_algorithm": "Kyber768",
                "signature_algorithm": "Dilithium3",
                "compatibility_mode": "hybrid"
            }

            self.logger.info("Криптографические системы инициализированы")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации криптографических систем: {e}")

    async def _generate_initial_keys(self):
        """Генерация начальных ключей"""
        try:
            # Генерация мастер-ключа
            master_key = await self._generate_quantum_key(
                algorithm=EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER,
                key_length=256,
                security_level=SecurityLevel.MAXIMUM
            )

            # Генерация ключей для различных целей
            session_key = await self._generate_quantum_key(
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_length=256,
                security_level=SecurityLevel.QUANTUM_SAFE
            )

            # Сохранение ключей
            self.active_keys["master"] = master_key
            self.active_keys["session"] = session_key

            self.logger.info("Начальные ключи сгенерированы")

        except Exception as e:
            self.logger.error(f"Ошибка генерации начальных ключей: {e}")

    async def process_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка cryptographic inference"""
        try:
            operation = input_data.get("operation", "encrypt")

            if operation == "encrypt":
                return await self._encrypt_message(input_data)
            elif operation == "decrypt":
                return await self._decrypt_message(input_data)
            elif operation == "key_exchange":
                return await self._perform_key_exchange(input_data)
            elif operation == "establish_channel":
                return await self._establish_secure_channel(input_data)
            else:
                raise ValueError(f"Неизвестная операция: {operation}")

        except Exception as e:
            self.logger.error(f"Ошибка cryptographic inference: {e}")
            return {"error": str(e), "operation": input_data.get("operation", "unknown")}

    async def process_quantum_inference(self, input_data: Dict[str, Any], quantum_state: Any, entanglement: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum-enhanced cryptographic processing"""
        try:
            # Базовая обработка
            base_result = await self.process_inference(input_data)

            if not self.quantum_core or "error" in base_result:
                return base_result

            operation = input_data.get("operation", "encrypt")

            # Quantum enhancement для различных операций
            if operation in ["encrypt", "decrypt"]:
                quantum_enhanced = await self._apply_quantum_crypto_enhancement(
                    base_result, quantum_state, entanglement
                )
            elif operation == "key_exchange":
                quantum_enhanced = await self._apply_quantum_key_exchange_enhancement(
                    base_result, quantum_state, entanglement
                )
            else:
                quantum_enhanced = {}

            # Объединение результатов
            enhanced_result = base_result.copy()
            enhanced_result.update({
                "quantum_security_boost": quantum_enhanced.get("security_boost", 0),
                "entanglement_strength": entanglement.get("entanglement_strength", 0),
                "phi_cryptographic_score": quantum_enhanced.get("phi_score", PHI_RATIO)
            })

            return enhanced_result

        except Exception as e:
            self.logger.error(f"Ошибка quantum cryptographic inference: {e}")
            return await self.process_inference(input_data)

    async def _encrypt_message(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Шифрование сообщения"""
        try:
            plaintext = input_data.get("plaintext", "")
            if isinstance(plaintext, str):
                plaintext = plaintext.encode('utf-8')

            recipient = input_data.get("recipient", "default")
            algorithm = EncryptionAlgorithm(input_data.get("algorithm", "aes_256_gcm"))

            # Получение или генерация ключа
            key = await self._get_encryption_key(recipient, algorithm)

            # Шифрование
            ciphertext, nonce, auth_tag = await self._perform_encryption(plaintext, key)

            # Создание зашифрованного сообщения
            message = EncryptedMessage(
                message_id=secrets.token_hex(16),
                sender=input_data.get("sender", "system"),
                recipient=recipient,
                ciphertext=ciphertext,
                nonce=nonce,
                auth_tag=auth_tag,
                timestamp=datetime.now(),
                algorithm=algorithm,
                key_id=key.key_id
            )

            # Сохранение в лог
            self.message_log.append(message)

            # Кодирование для передачи
            result = {
                "message_id": message.message_id,
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "nonce": base64.b64encode(nonce).decode('utf-8'),
                "auth_tag": base64.b64encode(auth_tag).decode('utf-8'),
                "algorithm": algorithm.value,
                "key_id": key.key_id,
                "timestamp": message.timestamp.isoformat(),
                "quantum_secure": key.quantum_generated
            }

            self.encryption_operations += 1
            return result

        except Exception as e:
            self.logger.error(f"Ошибка шифрования сообщения: {e}")
            return {"error": str(e)}

    async def _decrypt_message(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Расшифровка сообщения"""
        try:
            # Декодирование входных данных
            ciphertext = base64.b64decode(input_data["ciphertext"])
            nonce = base64.b64decode(input_data["nonce"])
            auth_tag = base64.b64decode(input_data["auth_tag"])
            key_id = input_data["key_id"]
            algorithm = EncryptionAlgorithm(input_data.get("algorithm", "aes_256_gcm"))

            # Получение ключа
            key = self.active_keys.get(key_id)
            if not key:
                raise ValueError(f"Ключ {key_id} не найден")

            # Расшифровка
            plaintext = await self._perform_decryption(ciphertext, nonce, auth_tag, key)

            result = {
                "plaintext": plaintext.decode('utf-8'),
                "algorithm": algorithm.value,
                "key_id": key_id,
                "decryption_time": datetime.now().isoformat(),
                "quantum_secure": key.quantum_generated
            }

            return result

        except Exception as e:
            self.logger.error(f"Ошибка расшифровки сообщения: {e}")
            return {"error": str(e)}

    async def _perform_key_exchange(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение обмена ключами"""
        try:
            protocol = KeyExchangeProtocol(input_data.get("protocol", "qkd"))
            participants = input_data.get("participants", [])

            if protocol == KeyExchangeProtocol.QUANTUM_KEY_DISTRIBUTION:
                # QKD key exchange
                key = await self._perform_qkd_key_exchange(participants)
            elif protocol == KeyExchangeProtocol.POST_QUANTUM_KEM:
                # Post-quantum KEM
                key = await self._perform_post_quantum_kem(participants)
            else:
                # Classical key exchange
                key = await self._perform_classical_key_exchange(participants, protocol)

            # Сохранение ключа
            session_key_id = f"session_{secrets.token_hex(8)}"
            self.active_keys[session_key_id] = key

            result = {
                "key_id": key.key_id,
                "session_key_id": session_key_id,
                "protocol": protocol.value,
                "participants": participants,
                "security_level": key.security_level.value,
                "quantum_generated": key.quantum_generated,
                "established_time": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            self.logger.error(f"Ошибка обмена ключами: {e}")
            return {"error": str(e)}

    async def _establish_secure_channel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Установление безопасного канала"""
        try:
            channel_id = input_data.get("channel_id", secrets.token_hex(16))
            participants = input_data.get("participants", [])

            # Создание канала
            channel = SecureChannel(
                channel_id=channel_id,
                participants=participants,
                encryption_key=self.active_keys.get("session"),
                protocol=KeyExchangeProtocol.QUANTUM_KEY_DISTRIBUTION,
                established_time=datetime.now(),
                last_activity=datetime.now(),
                message_count=0,
                security_status="active"
            )

            # Сохранение канала
            self.secure_channels[channel_id] = channel

            result = {
                "channel_id": channel_id,
                "participants": participants,
                "protocol": channel.protocol.value,
                "security_level": channel.encryption_key.security_level.value if channel.encryption_key else "unknown",
                "established_time": channel.established_time.isoformat(),
                "status": "established"
            }

            return result

        except Exception as e:
            self.logger.error(f"Ошибка установления безопасного канала: {e}")
            return {"error": str(e)}

    async def _generate_quantum_key(self, algorithm: EncryptionAlgorithm, key_length: int, security_level: SecurityLevel) -> QuantumKey:
        """Генерация квантового ключа"""
        try:
            key_id = f"quantum_key_{secrets.token_hex(8)}"

            # Генерация ключевого материала
            if self.quantum_core and security_level in [SecurityLevel.QUANTUM_SAFE, SecurityLevel.MAXIMUM]:
                # Истинная квантовая генерация
                key_material = await self.quantum_core.generate_quantum_random_bytes(key_length // 8)
                quantum_generated = True
                phi_entropy = PHI_RATIO
            else:
                # Классическая генерация с quantum enhancement
                key_material = secrets.token_bytes(key_length // 8)
                quantum_generated = False
                phi_entropy = 1.0

            # Создание ключа
            key = QuantumKey(
                key_id=key_id,
                key_material=key_material,
                algorithm=algorithm,
                key_length=key_length,
                generation_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=self.key_rotation_interval_hours),
                security_level=security_level,
                quantum_generated=quantum_generated,
                phi_entropy=phi_entropy
            )

            # Сохранение в историю
            self.key_history.append(key)
            self.key_generations += 1

            # Планирование ротации
            self.key_rotation_schedule[key_id] = key.expiry_time

            return key

        except Exception as e:
            self.logger.error(f"Ошибка генерации квантового ключа: {e}")
            raise

    async def _get_encryption_key(self, recipient: str, algorithm: EncryptionAlgorithm) -> QuantumKey:
        """Получение ключа шифрования"""
        try:
            # Поиск существующего ключа
            for key in self.active_keys.values():
                if key.algorithm == algorithm and not self._is_key_expired(key):
                    return key

            # Генерация нового ключа
            security_level = SecurityLevel.QUANTUM_SAFE if self.quantum_safe else SecurityLevel.HIGH
            key_length = 256 if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER] else 128

            return await self._generate_quantum_key(algorithm, key_length, security_level)

        except Exception as e:
            self.logger.error(f"Ошибка получения ключа шифрования: {e}")
            raise

    def _is_key_expired(self, key: QuantumKey) -> bool:
        """Проверка истечения срока действия ключа"""
        return key.expiry_time and datetime.now() > key.expiry_time

    async def _perform_encryption(self, plaintext: bytes, key: QuantumKey) -> Tuple[bytes, bytes, bytes]:
        """Выполнение шифрования"""
        try:
            if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                # AES-GCM шифрование
                nonce = secrets.token_bytes(12)
                cipher = await self._aes_gcm_encrypt(plaintext, key.key_material, nonce)
                ciphertext, auth_tag = cipher
                return ciphertext, nonce, auth_tag

            elif key.algorithm == EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER:
                # Kyber KEM шифрование
                return await self._kyber_encrypt(plaintext, key.key_material)

            else:
                # Fallback to AES
                nonce = secrets.token_bytes(12)
                cipher = await self._aes_gcm_encrypt(plaintext, key.key_material, nonce)
                ciphertext, auth_tag = cipher
                return ciphertext, nonce, auth_tag

        except Exception as e:
            self.logger.error(f"Ошибка шифрования: {e}")
            raise

    async def _perform_decryption(self, ciphertext: bytes, nonce: bytes, auth_tag: bytes, key: QuantumKey) -> bytes:
        """Выполнение расшифровки"""
        try:
            if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                # AES-GCM расшифровка
                return await self._aes_gcm_decrypt(ciphertext, key.key_material, nonce, auth_tag)

            elif key.algorithm == EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER:
                # Kyber KEM расшифровка
                return await self._kyber_decrypt(ciphertext, nonce, auth_tag, key.key_material)

            else:
                # Fallback to AES
                return await self._aes_gcm_decrypt(ciphertext, key.key_material, nonce, auth_tag)

        except Exception as e:
            self.logger.error(f"Ошибка расшифровки: {e}")
            raise

    async def _aes_gcm_encrypt(self, plaintext: bytes, key: bytes, nonce: bytes) -> Tuple[bytes, bytes]:
        """AES-GCM шифрование (упрощенная реализация)"""
        # В реальной реализации использовались бы cryptography library
        # Здесь имитация для демонстрации
        ciphertext = plaintext  # Заглушка
        auth_tag = secrets.token_bytes(16)
        return ciphertext, auth_tag

    async def _aes_gcm_decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, auth_tag: bytes) -> bytes:
        """AES-GCM расшифровка (упрощенная реализация)"""
        # В реальной реализации использовались бы cryptography library
        return ciphertext  # Заглушка

    async def _kyber_encrypt(self, plaintext: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """Kyber шифрование (упрощенная реализация)"""
        # В реальной реализации использовались бы pqclean library
        ciphertext = plaintext  # Заглушка
        nonce = secrets.token_bytes(32)
        auth_tag = secrets.token_bytes(32)
        return ciphertext, nonce, auth_tag

    async def _kyber_decrypt(self, ciphertext: bytes, nonce: bytes, auth_tag: bytes, key: bytes) -> bytes:
        """Kyber расшифровка (упрощенная реализация)"""
        # В реальной реализации использовались бы pqclean library
        return ciphertext  # Заглушка

    async def _perform_qkd_key_exchange(self, participants: List[str]) -> QuantumKey:
        """Выполнение QKD обмена ключами"""
        try:
            # Имитация QKD протокола
            key_material = await self.quantum_core.generate_quantum_random_bytes(32) if self.quantum_core else secrets.token_bytes(32)

            key = QuantumKey(
                key_id=f"qkd_{secrets.token_hex(8)}",
                key_material=key_material,
                algorithm=EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER,
                key_length=256,
                generation_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=1),  # QKD ключи короткоживущие
                security_level=SecurityLevel.MAXIMUM,
                quantum_generated=True,
                phi_entropy=PHI_RATIO * QUANTUM_FACTOR
            )

            self.qkd_sessions += 1
            return key

        except Exception as e:
            self.logger.error(f"Ошибка QKD обмена: {e}")
            raise

    async def _perform_post_quantum_kem(self, participants: List[str]) -> QuantumKey:
        """Выполнение post-quantum KEM"""
        try:
            # Имитация post-quantum KEM
            key_material = secrets.token_bytes(32)

            key = QuantumKey(
                key_id=f"pqkem_{secrets.token_hex(8)}",
                key_material=key_material,
                algorithm=EncryptionAlgorithm.QUANTUM_RESISTANT_KYBER,
                key_length=256,
                generation_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=self.key_rotation_interval_hours),
                security_level=SecurityLevel.QUANTUM_SAFE,
                quantum_generated=False,
                phi_entropy=PHI_RATIO
            )

            return key

        except Exception as e:
            self.logger.error(f"Ошибка post-quantum KEM: {e}")
            raise

    async def _perform_classical_key_exchange(self, participants: List[str], protocol: KeyExchangeProtocol) -> QuantumKey:
        """Выполнение классического обмена ключами"""
        try:
            # Имитация классического key exchange
            key_material = secrets.token_bytes(32)

            algorithm = EncryptionAlgorithm.AES_256_GCM
            if protocol == KeyExchangeProtocol.ECDH_P384:
                algorithm = EncryptionAlgorithm.AES_256_GCM

            key = QuantumKey(
                key_id=f"classical_{secrets.token_hex(8)}",
                key_material=key_material,
                algorithm=algorithm,
                key_length=256,
                generation_time=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=self.key_rotation_interval_hours),
                security_level=SecurityLevel.HIGH,
                quantum_generated=False,
                phi_entropy=1.0
            )

            return key

        except Exception as e:
            self.logger.error(f"Ошибка классического обмена ключами: {e}")
            raise

    async def _apply_quantum_crypto_enhancement(self, base_result: Dict[str, Any], quantum_state: Any, entanglement: Dict[str, Any]) -> Dict[str, Any]:
        """Применение quantum enhancement к cryptographic операциям"""
        try:
            if not self.quantum_core:
                return {}

            # Quantum-enhanced security analysis
            security_analysis = await self.quantum_core.analyze_crypto_security(
                base_result, quantum_state, entanglement
            )

            return {
                "security_boost": entanglement.get("entanglement_strength", 0) * QUANTUM_FACTOR,
                "phi_score": PHI_RATIO * entanglement.get("entanglement_strength", 1),
                "quantum_security_metrics": security_analysis.get("metrics", {})
            }

        except Exception as e:
            self.logger.error(f"Ошибка quantum crypto enhancement: {e}")
            return {}

    async def _apply_quantum_key_exchange_enhancement(self, base_result: Dict[str, Any], quantum_state: Any, entanglement: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum enhancement для key exchange"""
        try:
            if not self.quantum_core:
                return {}

            # Enhanced key exchange security
            enhanced_security = await self.quantum_core.enhance_key_exchange_security(
                base_result, quantum_state, entanglement
            )

            return {
                "key_exchange_security_boost": entanglement.get("entanglement_strength", 0) * QUANTUM_FACTOR,
                "phi_key_score": PHI_RATIO ** 2,
                "enhanced_protocols": enhanced_security.get("protocols", [])
            }

        except Exception as e:
            self.logger.error(f"Ошибка quantum key exchange enhancement: {e}")
            return {}

    async def rotate_keys(self) -> Dict[str, Any]:
        """Ротация ключей"""
        try:
            self.logger.info("Выполнение ротации ключей...")

            rotated_keys = []
            expired_keys = []

            # Проверка ключей на истечение
            current_time = datetime.now()
            for key_id, key in list(self.active_keys.items()):
                if self._is_key_expired(key):
                    expired_keys.append(key_id)

                    # Генерация нового ключа
                    new_key = await self._generate_quantum_key(
                        key.algorithm, key.key_length, key.security_level
                    )

                    # Замена ключа
                    self.active_keys[key_id] = new_key
                    rotated_keys.append(key_id)

            # Очистка расписания ротации
            for key_id in expired_keys:
                if key_id in self.key_rotation_schedule:
                    del self.key_rotation_schedule[key_id]

            result = {
                "rotated_keys": rotated_keys,
                "expired_keys": expired_keys,
                "total_active_keys": len(self.active_keys),
                "rotation_time": datetime.now().isoformat()
            }

            self.logger.info(f"Ротация ключей выполнена: {len(rotated_keys)} ключей обновлено")
            return result

        except Exception as e:
            self.logger.error(f"Ошибка ротации ключей: {e}")
            return {"error": str(e)}

    async def get_crypto_status(self) -> Dict[str, Any]:
        """Получение статуса криптографической системы"""
        try:
            active_channels = len([c for c in self.secure_channels.values() if c.security_status == "active"])

            return {
                "name": self.name,
                "status": self.status,
                "quantum_safe": self.quantum_safe,
                "security_level": self.security_level.value,
                "active_keys": len(self.active_keys),
                "total_keys_generated": len(self.key_history),
                "active_channels": active_channels,
                "total_messages": len(self.message_log),
                "encryption_operations": self.encryption_operations,
                "qkd_sessions": self.qkd_sessions,
                "security_breaches": self.security_breaches,
                "key_rotation_due": len([k for k in self.key_rotation_schedule.values() if k <= datetime.now()])
            }

        except Exception as e:
            self.logger.error(f"Ошибка получения статуса crypto: {e}")
            return {"status": "error", "error": str(e)}

    async def optimize_crypto_performance(self) -> Dict[str, Any]:
        """Оптимизация производительности криптографической системы"""
        try:
            self.logger.info("Оптимизация криптографической системы...")

            optimizations = {}

            # Ротация ключей
            key_rotation = await self.rotate_keys()
            optimizations["key_rotation"] = key_rotation

            # Оптимизация каналов
            channel_cleanup = await self._cleanup_secure_channels()
            optimizations["channel_cleanup"] = channel_cleanup

            # Quantum optimization
            if self.quantum_core:
                quantum_opts = await self.quantum_core.optimize_crypto_performance()
                optimizations["quantum_optimization"] = quantum_opts

            # Очистка истории
            history_cleanup = await self._cleanup_crypto_history()
            optimizations["history_cleanup"] = history_cleanup

            self.logger.info(f"Оптимизация выполнена: {len(optimizations)} компонентов оптимизировано")
            return optimizations

        except Exception as e:
            self.logger.error(f"Ошибка оптимизации: {e}")
            return {"error": str(e)}

    async def _cleanup_secure_channels(self) -> Dict[str, Any]:
        """Очистка неактивных каналов"""
        try:
            current_time = datetime.now()
            inactive_channels = []

            for channel_id, channel in list(self.secure_channels.items()):
                # Удаление каналов без активности более 24 часов
                if (current_time - channel.last_activity).total_seconds() > 86400:
                    inactive_channels.append(channel_id)
                    del self.secure_channels[channel_id]

            return {
                "inactive_channels_removed": len(inactive_channels),
                "active_channels_remaining": len(self.secure_channels)
            }

        except Exception as e:
            self.logger.error(f"Ошибка очистки каналов: {e}")
            return {"error": str(e)}

    async def _cleanup_crypto_history(self) -> Dict[str, Any]:
        """Очистка криптографической истории"""
        try:
            # Ограничение истории сообщений (последние 1000)
            if len(self.message_log) > 1000:
                removed_count = len(self.message_log) - 1000
                self.message_log = self.message_log[-1000:]
            else:
                removed_count = 0

            # Ограничение истории ключей (последние 100)
            if len(self.key_history) > 100:
                self.key_history = self.key_history[-100:]

            return {
                "messages_removed": removed_count,
                "keys_history_limited": len(self.key_history)
            }

        except Exception as e:
            self.logger.error(f"Ошибка очистки истории: {e}")
            return {"error": str(e)}

    async def shutdown(self) -> bool:
        """Остановка Quantum Cryptography"""
        try:
            self.logger.info("Остановка Quantum Cryptography...")

            # Финальная ротация ключей
            await self.rotate_keys()

            # Очистка активных сессий
            self.active_keys.clear()
            self.secure_channels.clear()

            # Сохранение финальной статистики
            await self._save_crypto_stats()

            self.set_status("shutdown")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка остановки Quantum Cryptography: {e}")
            return False

    async def _save_crypto_stats(self):
        """Сохранение статистики криптографии"""
        try:
            stats = await self.get_crypto_status()
            stats["shutdown_time"] = datetime.now().isoformat()

            with open("quantum_cryptography_stats.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            self.logger.info("Quantum Cryptography stats saved")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения статистики: {e}")

# Импорт timedelta для работы с датами
from datetime import timedelta