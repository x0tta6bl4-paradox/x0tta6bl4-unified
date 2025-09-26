"""
Encryption module для x0tta6bl4 API
AES-256 encryption для sensitive data at rest и in transit
"""

import os
import base64
import json
from typing import Any, Dict, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import logging
from .security import security_config, log_security_event

logger = logging.getLogger(__name__)

class DataEncryption:
    """Класс для шифрования данных"""

    def __init__(self, key: Optional[str] = None):
        self.key = key or security_config.encryption_key
        if len(self.key) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes")

        self.backend = default_backend()

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(password.encode())

    def encrypt_data(self, data: Any, password: Optional[str] = None) -> str:
        """Зашифровать данные"""
        try:
            # Convert data to JSON string if not already
            if not isinstance(data, str):
                data = json.dumps(data, default=str)

            # Generate salt and IV
            salt = os.urandom(16)
            iv = os.urandom(16)

            # Use password-derived key or default key
            if password:
                key = self._derive_key(password, salt)
            else:
                key = self.key.encode()

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()

            # Pad data
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(data.encode()) + padder.finalize()

            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            # Combine salt + iv + encrypted_data
            if password:
                combined = salt + iv + encrypted_data
            else:
                combined = iv + encrypted_data

            # Base64 encode
            encrypted_b64 = base64.b64encode(combined).decode()

            log_security_event("data_encrypted", {
                "data_type": type(data).__name__,
                "used_password": password is not None
            })

            return encrypted_b64

        except Exception as e:
            log_security_event("encryption_failed", {
                "error": str(e),
                "data_type": type(data).__name__
            }, "error")
            raise

    def decrypt_data(self, encrypted_data: str, password: Optional[str] = None) -> Any:
        """Расшифровать данные"""
        try:
            # Base64 decode
            combined = base64.b64decode(encrypted_data)

            if password:
                # Extract salt, iv, encrypted_data
                salt = combined[:16]
                iv = combined[16:32]
                encrypted = combined[32:]

                # Derive key from password
                key = self._derive_key(password, salt)
            else:
                # Extract iv, encrypted_data
                iv = combined[:16]
                encrypted = combined[16:]
                key = self.key.encode()

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()

            # Decrypt
            padded_data = decryptor.update(encrypted) + decryptor.finalize()

            # Unpad
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            data = unpadder.update(padded_data) + unpadder.finalize()

            # Try to parse as JSON
            try:
                return json.loads(data.decode())
            except json.JSONDecodeError:
                return data.decode()

        except Exception as e:
            log_security_event("decryption_failed", {
                "error": str(e),
                "used_password": password is not None
            }, "error")
            raise

class SecureStorage:
    """Безопасное хранение данных"""

    def __init__(self, encryption: Optional[DataEncryption] = None):
        self.encryption = encryption or DataEncryption()

    def store_secure_data(self, key: str, data: Any, password: Optional[str] = None) -> bool:
        """Безопасно сохранить данные"""
        try:
            encrypted = self.encryption.encrypt_data(data, password)

            # In production, store in database or secure file
            # For demo, we'll use a simple dict
            if not hasattr(self, '_storage'):
                self._storage = {}

            self._storage[key] = encrypted

            log_security_event("secure_data_stored", {
                "key": key,
                "data_type": type(data).__name__
            })

            return True

        except Exception as e:
            log_security_event("secure_storage_failed", {
                "key": key,
                "error": str(e)
            }, "error")
            return False

    def retrieve_secure_data(self, key: str, password: Optional[str] = None) -> Optional[Any]:
        """Получить безопасные данные"""
        try:
            if not hasattr(self, '_storage') or key not in self._storage:
                return None

            encrypted = self._storage[key]
            return self.encryption.decrypt_data(encrypted, password)

        except Exception as e:
            log_security_event("secure_retrieval_failed", {
                "key": key,
                "error": str(e)
            }, "error")
            return None

class APIEncryption:
    """Encryption для API responses"""

    def __init__(self):
        self.encryption = DataEncryption()

    def encrypt_response(self, data: Dict[str, Any], client_key: Optional[str] = None) -> Dict[str, Any]:
        """Зашифровать API response"""
        if not client_key:
            return data  # No encryption if no client key

        try:
            encrypted_data = self.encryption.encrypt_data(data, client_key)
            return {
                "encrypted": True,
                "data": encrypted_data
            }
        except Exception as e:
            logger.error(f"Failed to encrypt response: {e}")
            return data

    def decrypt_request(self, encrypted_data: str, client_key: str) -> Optional[Dict[str, Any]]:
        """Расшифровать API request"""
        try:
            return self.encryption.decrypt_data(encrypted_data, client_key)
        except Exception as e:
            logger.error(f"Failed to decrypt request: {e}")
            return None

class QuantumKeyDistribution:
    """Quantum Key Distribution simulation для secure key exchange"""

    def __init__(self):
        self.shared_keys: Dict[str, str] = {}

    def generate_quantum_key(self, client_id: str) -> str:
        """Generate quantum-resistant key"""
        # In real implementation, this would use quantum key distribution
        # For demo, generate strong classical key
        key = base64.b64encode(os.urandom(32)).decode()
        self.shared_keys[client_id] = key

        log_security_event("quantum_key_generated", {
            "client_id": client_id
        })

        return key

    def get_shared_key(self, client_id: str) -> Optional[str]:
        """Get shared key for client"""
        return self.shared_keys.get(client_id)

    def revoke_key(self, client_id: str) -> bool:
        """Revoke shared key"""
        if client_id in self.shared_keys:
            del self.shared_keys[client_id]
            log_security_event("quantum_key_revoked", {
                "client_id": client_id
            })
            return True
        return False

# Global instances
data_encryption = DataEncryption()
secure_storage = SecureStorage(data_encryption)
api_encryption = APIEncryption()
qkd = QuantumKeyDistribution()

def encrypt_sensitive_data(data: Any) -> str:
    """Утилита для шифрования sensitive data"""
    return data_encryption.encrypt_data(data)

def decrypt_sensitive_data(encrypted: str) -> Any:
    """Утилита для расшифровки sensitive data"""
    return data_encryption.decrypt_data(encrypted)

def generate_secure_token() -> str:
    """Generate cryptographically secure token"""
    return base64.b64encode(os.urandom(32)).decode()

def hash_data(data: str) -> str:
    """Hash data with SHA-256"""
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(data.encode())
    return base64.b64encode(digest.finalize()).decode()