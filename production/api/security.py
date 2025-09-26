"""
Security module для x0tta6bl4 API
Обеспечивает базовые security функции, headers, audit logging
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware для добавления security headers"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для audit logging всех запросов"""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()

        # Логируем входящий запрос
        audit_data = {
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
        }

        try:
            response = await call_next(request)

            # Логируем ответ
            end_time = datetime.utcnow()
            audit_data.update({
                "status_code": response.status_code,
                "response_time": (end_time - start_time).total_seconds(),
                "success": True
            })

            logger.info(f"AUDIT: {json.dumps(audit_data, default=str)}")

            return response

        except Exception as e:
            # Логируем ошибки
            end_time = datetime.utcnow()
            audit_data.update({
                "status_code": 500,
                "response_time": (end_time - start_time).total_seconds(),
                "success": False,
                "error": str(e)
            })

            logger.error(f"AUDIT ERROR: {json.dumps(audit_data, default=str)}")
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Получить IP адрес клиента"""
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

class SecurityConfig:
    """Конфигурация security"""

    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration = int(os.getenv("JWT_EXPIRATION", "3600"))  # 1 hour

        # Rate limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

        # Encryption
        self.encryption_key = os.getenv("ENCRYPTION_KEY", "your-32-byte-encryption-key-12345678901234567890123456789012")

        # CORS
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        self.allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Глобальная конфигурация
security_config = SecurityConfig()

def get_security_config() -> SecurityConfig:
    """Получить конфигурацию security"""
    return security_config

def log_security_event(event_type: str, details: Dict[str, Any], level: str = "info"):
    """Логировать security событие"""
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }

    if level == "error":
        logger.error(f"SECURITY EVENT: {json.dumps(log_data, default=str)}")
    elif level == "warning":
        logger.warning(f"SECURITY EVENT: {json.dumps(log_data, default=str)}")
    else:
        logger.info(f"SECURITY EVENT: {json.dumps(log_data, default=str)}")

def create_error_response(status_code: int, message: str, details: Optional[Dict] = None) -> JSONResponse:
    """Создать standardized error response"""
    error_data = {
        "error": {
            "code": status_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    if details:
        error_data["error"]["details"] = details

    # Логируем security error
    log_security_event("api_error", {
        "status_code": status_code,
        "message": message,
        "details": details
    }, "warning")

    return JSONResponse(status_code=status_code, content=error_data)