
"""
Unified API Gateway для x0tta6bl4
Объединяет все API endpoints в единую точку входа
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import security modules
from .security import (
    SecurityHeadersMiddleware,
    AuditLoggingMiddleware,
    security_config,
    create_error_response
)
from .auth import (
    get_current_active_user,
    require_permissions,
    Permission,
    User,
    authenticate_user,
    login_user,
    UserLogin,
    TokenResponse
)
from .validation import (
    validate_request_data,
    check_rate_limit_middleware,
    SecureUserInput,
    SecurePasswordInput
)
from .encryption import api_encryption

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Создание FastAPI приложения
app = FastAPI(
    title="x0tta6bl4 Unified API",
    description="Secure Unified API Gateway для квантовых вычислений, AI и SaaS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(SlowAPIMiddleware)

# HTTPS enforcement (only in production)
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# CORS middleware with restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=security_config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware with restrictions
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=security_config.allowed_hosts
)

# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, user_credentials: UserLogin):
    """Аутентификация пользователя"""
    # Validate input
    validate_request_data(user_credentials.dict())

    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    return login_user(user)

@app.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def simple_login(request: Request, user_credentials: UserLogin):
    """Простой endpoint для входа"""
    # Validate input
    validate_request_data(user_credentials.dict())

    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    return login_user(user)

@app.post("/auth/register")
@limiter.limit("3/minute")
async def register(request: Request, user_data: SecureUserInput, password: SecurePasswordInput):
    """Регистрация нового пользователя"""
    # Validate input
    validate_request_data(user_data.dict())
    validate_request_data({"password": password.password})

    from .auth import create_user, UserCreate, UserRole

    try:
        user = create_user(UserCreate(
            email=user_data.email,
            username=user_data.username,
            password=password.password,
            role=UserRole.USER
        ))
        return {"message": "User created successfully", "user_id": user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Обновить access token"""
    from .auth import refresh_access_token
    new_tokens = refresh_access_token(refresh_token)
    if not new_tokens:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return new_tokens

# Protected endpoints
@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Получить информацию о текущем пользователе"""
    return {
        "user": current_user.dict(),
        "permissions": current_user.role.value
    }

# Health check endpoint (public)
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "service": "x0tta6bl4-unified-api",
        "version": "1.0.0",
        "security": "enabled"
    }

# Root endpoint
@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "x0tta6bl4 Unified API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "quantum": "/api/v1/quantum",
            "ai": "/api/v1/ai",
            "enterprise": "/api/v1/enterprise",
            "billing": "/api/v1/billing",
            "monitoring": "/api/v1/monitoring"
        }
    }

# Quantum API endpoints (protected)
@app.get("/api/v1/quantum/status")
@limiter.limit("30/minute")
async def quantum_status(
    request: Request,
    current_user: User = Depends(require_permissions([Permission.READ_QUANTUM]))
):
    """Статус квантовых сервисов"""
    response_data = {
        "status": "operational",
        "providers": ["ibm", "google", "xanadu"],
        "algorithms": ["vqe", "qaoa", "grover", "shor"],
        "user": current_user.username,
        "role": current_user.role.value
    }

    # Encrypt response if client requests it
    if request.headers.get("X-Encrypt-Response") == "true":
        return api_encryption.encrypt_response(response_data)

    return response_data

# AI API endpoints (protected)
@app.get("/api/v1/ai/status")
@limiter.limit("30/minute")
async def ai_status(
    request: Request,
    current_user: User = Depends(require_permissions([Permission.READ_AI]))
):
    """Статус AI сервисов"""
    response_data = {
        "status": "operational",
        "models": ["gpt", "claude", "llama"],
        "agents": ["documentation", "monitoring", "optimization"],
        "user": current_user.username
    }

    if request.headers.get("X-Encrypt-Response") == "true":
        return api_encryption.encrypt_response(response_data)

    return response_data

# Enterprise API endpoints (protected)
@app.get("/api/v1/enterprise/status")
@limiter.limit("20/minute")
async def enterprise_status(
    request: Request,
    current_user: User = Depends(require_permissions([Permission.ADMIN_ACCESS]))
):
    """Статус enterprise сервисов"""
    response_data = {
        "status": "operational",
        "features": ["multi_tenant", "rbac", "audit_logging"],
        "gateway": "active",
        "user": current_user.username,
        "role": current_user.role.value
    }

    if request.headers.get("X-Encrypt-Response") == "true":
        return api_encryption.encrypt_response(response_data)

    return response_data

# Billing API endpoints (protected)
@app.get("/api/v1/billing/status")
@limiter.limit("20/minute")
async def billing_status(
    request: Request,
    current_user: User = Depends(require_permissions([Permission.READ_BILLING]))
):
    """Статус billing сервисов"""
    response_data = {
        "status": "operational",
        "providers": ["stripe", "paypal", "yookassa"],
        "features": ["subscriptions", "invoices", "payments"],
        "user": current_user.username
    }

    if request.headers.get("X-Encrypt-Response") == "true":
        return api_encryption.encrypt_response(response_data)

    return response_data

# Monitoring API endpoints (protected)
@app.get("/api/v1/monitoring/status")
@limiter.limit("15/minute")
async def monitoring_status(
    request: Request,
    current_user: User = Depends(require_permissions([Permission.READ_MONITORING]))
):
    """Статус мониторинга"""
    response_data = {
        "status": "operational",
        "metrics": ["prometheus", "grafana"],
        "logging": ["structured", "distributed"],
        "user": current_user.username
    }

    if request.headers.get("X-Encrypt-Response") == "true":
        return api_encryption.encrypt_response(response_data)

    return response_data

if __name__ == "__main__":
    # SSL configuration for production
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")

    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", "8000")),
        "reload": os.getenv("ENVIRONMENT") != "production",
        "log_level": "info"
    }

    # Add SSL if certificates are provided
    if ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        })
        logger.info("SSL/TLS encryption enabled")

    uvicorn.run(**uvicorn_config)
