"""
Authentication и Authorization module для x0tta6bl4 API
JWT tokens, RBAC, user management
"""

import jwt
import bcrypt
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from enum import Enum
import logging
from .security import security_config, log_security_event, create_error_response

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

class UserRole(str, Enum):
    """Роли пользователей"""
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class Permission(str, Enum):
    """Разрешения"""
    READ_QUANTUM = "read:quantum"
    WRITE_QUANTUM = "write:quantum"
    READ_AI = "read:ai"
    WRITE_AI = "write:ai"
    READ_BILLING = "read:billing"
    WRITE_BILLING = "write:billing"
    READ_MONITORING = "read:monitoring"
    ADMIN_ACCESS = "admin:access"

# Role-based permissions
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_QUANTUM, Permission.WRITE_QUANTUM,
        Permission.READ_AI, Permission.WRITE_AI,
        Permission.READ_BILLING, Permission.WRITE_BILLING,
        Permission.READ_MONITORING, Permission.ADMIN_ACCESS
    ],
    UserRole.ENTERPRISE: [
        Permission.READ_QUANTUM, Permission.WRITE_QUANTUM,
        Permission.READ_AI, Permission.WRITE_AI,
        Permission.READ_BILLING, Permission.WRITE_BILLING,
        Permission.READ_MONITORING
    ],
    UserRole.PREMIUM: [
        Permission.READ_QUANTUM, Permission.WRITE_QUANTUM,
        Permission.READ_AI, Permission.WRITE_AI,
        Permission.READ_BILLING
    ],
    UserRole.USER: [
        Permission.READ_QUANTUM, Permission.READ_AI
    ]
}

class User(BaseModel):
    """Модель пользователя"""
    id: str
    email: EmailStr
    username: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = datetime.utcnow()
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    """Модель для создания пользователя"""
    email: EmailStr
    username: str
    password: str
    role: UserRole = UserRole.USER

class UserLogin(BaseModel):
    """Модель для входа"""
    username: str
    password: str

class TokenData(BaseModel):
    """Данные токена"""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]

class TokenResponse(BaseModel):
    """Ответ с токенами"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

# In-memory user storage (в production использовать базу данных)
_users_db: Dict[str, Dict[str, Any]] = {}

def hash_password(password: str) -> str:
    """Хэшировать пароль"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Проверить пароль"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_user(user_data: UserCreate) -> User:
    """Создать нового пользователя"""
    if user_data.username in _users_db:
        raise HTTPException(status_code=400, detail="Username already exists")

    user_dict = {
        "id": user_data.username,  # simplified
        "email": user_data.email,
        "username": user_data.username,
        "password_hash": hash_password(user_data.password),
        "role": user_data.role,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": None
    }

    _users_db[user_data.username] = user_dict

    log_security_event("user_created", {
        "username": user_data.username,
        "role": user_data.role.value
    })

    return User(**{k: v for k, v in user_dict.items() if k != "password_hash"})

def get_user(username: str) -> Optional[User]:
    """Получить пользователя по username"""
    user_data = _users_db.get(username)
    if not user_data:
        return None

    return User(**{k: v for k, v in user_data.items() if k != "password_hash"})

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Аутентифицировать пользователя"""
    user_data = _users_db.get(username)
    if not user_data:
        return None

    if not verify_password(password, user_data["password_hash"]):
        log_security_event("authentication_failed", {
            "username": username,
            "reason": "invalid_password"
        }, "warning")
        return None

    if not user_data["is_active"]:
        log_security_event("authentication_failed", {
            "username": username,
            "reason": "user_inactive"
        }, "warning")
        return None

    # Update last login
    user_data["last_login"] = datetime.utcnow()
    _users_db[username] = user_data

    log_security_event("authentication_success", {
        "username": username
    })

    return User(**{k: v for k, v in user_data.items() if k != "password_hash"})

def create_access_token(data: Dict[str, Any]) -> str:
    """Создать access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=security_config.jwt_expiration)
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, security_config.jwt_secret, algorithm=security_config.jwt_algorithm)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Создать refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # 7 days
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, security_config.jwt_secret, algorithm=security_config.jwt_algorithm)

def verify_token(token: str) -> Optional[TokenData]:
    """Проверить токен"""
    try:
        payload = jwt.decode(token, security_config.jwt_secret, algorithms=[security_config.jwt_algorithm])
        user_id = payload.get("user_id")
        username = payload.get("username")
        role = payload.get("role")

        if not all([user_id, username, role]):
            return None

        permissions = ROLE_PERMISSIONS.get(UserRole(role), [])

        return TokenData(
            user_id=user_id,
            username=username,
            role=UserRole(role),
            permissions=permissions
        )
    except jwt.ExpiredSignatureError:
        log_security_event("token_expired", {"token_type": "unknown"}, "warning")
        return None
    except jwt.InvalidTokenError:
        log_security_event("token_invalid", {"token_type": "unknown"}, "warning")
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Получить текущего пользователя из токена"""
    token_data = verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user(token_data.username)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Получить активного пользователя"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def check_permissions(required_permissions: List[Permission], token_data: TokenData) -> bool:
    """Проверить разрешения"""
    user_permissions = set(token_data.permissions)
    required = set(required_permissions)
    return required.issubset(user_permissions)

def require_permissions(required_permissions: List[Permission]):
    """Dependency для проверки разрешений"""
    def permission_checker(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token_data = verify_token(credentials.credentials)
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid token")

        if not check_permissions(required_permissions, token_data):
            log_security_event("permission_denied", {
                "username": token_data.username,
                "required_permissions": [p.value for p in required_permissions],
                "user_permissions": [p.value for p in token_data.permissions]
            }, "warning")

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )

        return token_data

    return permission_checker

def require_role(required_role: UserRole):
    """Dependency для проверки роли"""
    def role_checker(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token_data = verify_token(credentials.credentials)
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid token")

        if token_data.role != required_role:
            log_security_event("role_denied", {
                "username": token_data.username,
                "required_role": required_role.value,
                "user_role": token_data.role.value
            }, "warning")

            raise HTTPException(
                status_code=403,
                detail="Insufficient role"
            )

        return token_data

    return role_checker

def login_user(user: User) -> TokenResponse:
    """Войти пользователя и создать токены"""
    token_data = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role.value
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=security_config.jwt_expiration
    )

def refresh_access_token(refresh_token: str) -> Optional[TokenResponse]:
    """Обновить access token"""
    token_data = verify_token(refresh_token)
    if not token_data or token_data.get("type") != "refresh":
        return None

    user = get_user(token_data.username)
    if not user:
        return None

    return login_user(user)

# Initialize default admin user
def init_default_users():
    """Инициализировать пользователей по умолчанию"""
    if "admin" not in _users_db:
        create_user(UserCreate(
            email="admin@x0tta6bl4.com",
            username="admin",
            password="admin123!",  # Change in production!
            role=UserRole.ADMIN
        ))

    if "user" not in _users_db:
        create_user(UserCreate(
            email="user@x0tta6bl4.com",
            username="user",
            password="user123!",
            role=UserRole.USER
        ))

# Initialize on import
init_default_users()