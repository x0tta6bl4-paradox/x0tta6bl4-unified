"""
Input validation и sanitization module для x0tta6bl4 API
SQL injection, XSS, CSRF protection
"""

import re
import html
import bleach
from typing import Any, Dict, List, Optional, Union
from fastapi import HTTPException, Request
from pydantic import BaseModel, validator, Field
import logging
from .security import log_security_event

logger = logging.getLogger(__name__)

class InputValidator:
    """Класс для валидации и sanitization input данных"""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r';\s*--',  # SQL comment
        r';\s*/\*',  # SQL comment block start
        r'union\s+select',  # UNION SELECT
        r'/\*.*\*/',  # SQL comment blocks
        r'--.*',  # SQL line comments
        r';\s*drop\s+table',  # DROP TABLE
        r';\s*delete\s+from',  # DELETE FROM
        r';\s*update\s+.*set',  # UPDATE SET
        r';\s*insert\s+into',  # INSERT INTO
        r'exec\s*\(',  # EXEC function
        r'xp_cmdshell',  # XP_CMDSHELL
        r'sp_executesql',  # SP_EXECUTESQL
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'on\w+\s*=',  # Event handlers
        r'<iframe[^>]*>.*?</iframe>',  # Iframe tags
        r'<object[^>]*>.*?</object>',  # Object tags
        r'<embed[^>]*>.*?</embed>',  # Embed tags
        r'vbscript:',  # VBScript
        r'data:text/html',  # Data URLs
        r'expression\s*\(',  # CSS expressions
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'%2e%2e%2f',  # URL encoded ../
        r'%2e%2e/',  # URL encoded ..
        r'\.\.',  # Double dots
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r';\s*rm\s+',  # Remove command
        r';\s*wget\s+',  # Wget command
        r';\s*curl\s+',  # Curl command
        r';\s*nc\s+',  # Netcat command
        r'\|\s*cat\s+',  # Pipe to cat
        r'&&\s*rm\s+',  # Logical AND with rm
        r'\|\s*rm\s+',  # Pipe to rm
    ]

    @staticmethod
    def sanitize_sql_input(input_str: str) -> str:
        """Sanitize input to prevent SQL injection"""
        if not isinstance(input_str, str):
            return str(input_str)

        # Remove or escape dangerous characters
        sanitized = input_str.replace("'", "''")  # Escape single quotes
        sanitized = sanitized.replace('"', '""')  # Escape double quotes
        sanitized = sanitized.replace(';', '')  # Remove semicolons
        sanitized = sanitized.replace('--', '')  # Remove SQL comments
        sanitized = sanitized.replace('/*', '')  # Remove comment starts
        sanitized = sanitized.replace('*/', '')  # Remove comment ends

        return sanitized

    @staticmethod
    def check_sql_injection(input_str: str) -> bool:
        """Check for SQL injection patterns"""
        if not isinstance(input_str, str):
            return False

        input_lower = input_str.lower()
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                log_security_event("sql_injection_attempt", {
                    "pattern": pattern,
                    "input_length": len(input_str)
                }, "warning")
                return True
        return False

    @staticmethod
    def sanitize_html_input(input_str: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        if not isinstance(input_str, str):
            return str(input_str)

        # Use bleach for comprehensive HTML sanitization
        allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        allowed_attributes = {}

        sanitized = bleach.clean(input_str, tags=allowed_tags, attributes=allowed_attributes, strip=True)
        return html.escape(sanitized)

    @staticmethod
    def check_xss(input_str: str) -> bool:
        """Check for XSS patterns"""
        if not isinstance(input_str, str):
            return False

        input_lower = input_str.lower()
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                log_security_event("xss_attempt", {
                    "pattern": pattern,
                    "input_length": len(input_str)
                }, "warning")
                return True
        return False

    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize file paths to prevent directory traversal"""
        if not isinstance(path, str):
            return str(path)

        # Remove dangerous path components
        sanitized = path.replace('../', '')
        sanitized = sanitized.replace('..\\', '')
        sanitized = sanitized.replace('./', '')
        sanitized = sanitized.replace('.\\', '')

        # Remove URL encoded versions
        sanitized = sanitized.replace('%2e%2e%2f', '')
        sanitized = sanitized.replace('%2e%2e/', '')
        sanitized = sanitized.replace('%2e%2e%5c', '')

        return sanitized

    @staticmethod
    def check_path_traversal(path: str) -> bool:
        """Check for path traversal attempts"""
        if not isinstance(path, str):
            return False

        for pattern in InputValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path):
                log_security_event("path_traversal_attempt", {
                    "pattern": pattern,
                    "path": path
                }, "warning")
                return True
        return False

    @staticmethod
    def check_command_injection(input_str: str) -> bool:
        """Check for command injection patterns"""
        if not isinstance(input_str, str):
            return False

        input_lower = input_str.lower()
        for pattern in InputValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, input_lower):
                log_security_event("command_injection_attempt", {
                    "pattern": pattern,
                    "input_length": len(input_str)
                }, "warning")
                return True
        return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format"""
        # Allow alphanumeric, underscore, dash, min 3 chars, max 50
        username_pattern = r'^[a-zA-Z0-9_-]{3,50}$'
        return bool(re.match(username_pattern, username))

    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password strength"""
        checks = {
            'length': len(password) >= 8,
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'lowercase': bool(re.search(r'[a-z]', password)),
            'digit': bool(re.search(r'[0-9]', password)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        checks['strong'] = all(checks.values())
        return checks

    @staticmethod
    def sanitize_input(input_data: Any) -> Any:
        """Comprehensive input sanitization"""
        if isinstance(input_data, str):
            # Check for attacks first
            if InputValidator.check_sql_injection(input_data):
                raise HTTPException(status_code=400, detail="SQL injection attempt detected")
            if InputValidator.check_xss(input_data):
                raise HTTPException(status_code=400, detail="XSS attempt detected")
            if InputValidator.check_command_injection(input_data):
                raise HTTPException(status_code=400, detail="Command injection attempt detected")

            # Sanitize
            sanitized = InputValidator.sanitize_sql_input(input_data)
            sanitized = InputValidator.sanitize_html_input(sanitized)
            return sanitized

        elif isinstance(input_data, dict):
            return {k: InputValidator.sanitize_input(v) for k, v in input_data.items()}

        elif isinstance(input_data, list):
            return [InputValidator.sanitize_input(item) for item in input_data]

        else:
            return input_data

class CSRFProtection:
    """CSRF protection utilities"""

    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token"""
        import secrets
        return secrets.token_urlsafe(32)

    @staticmethod
    def validate_csrf_token(request: Request, token: str) -> bool:
        """Validate CSRF token"""
        session_token = request.session.get('csrf_token') if hasattr(request, 'session') else None
        header_token = request.headers.get('X-CSRF-Token')

        if not session_token or not header_token:
            return False

        return secrets.compare_digest(session_token, token) and secrets.compare_digest(header_token, token)

class RateLimitValidator:
    """Rate limiting validation"""

    def __init__(self):
        self.requests = {}

    def check_rate_limit(self, client_ip: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
        """Check if client exceeded rate limit"""
        import time
        current_time = time.time()

        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < window_seconds
        ]

        # Check limit
        if len(self.requests[client_ip]) >= max_requests:
            log_security_event("rate_limit_exceeded", {
                "client_ip": client_ip,
                "request_count": len(self.requests[client_ip])
            }, "warning")
            return False

        # Add current request
        self.requests[client_ip].append(current_time)
        return True

# Pydantic models with validation
class SecureUserInput(BaseModel):
    """Secure user input model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    @validator('username')
    def validate_username(cls, v):
        if not InputValidator.validate_username(v):
            raise ValueError('Invalid username format')
        return v

    @validator('email')
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError('Invalid email format')
        return v

class SecurePasswordInput(BaseModel):
    """Secure password input model"""
    password: str = Field(..., min_length=8)

    @validator('password')
    def validate_password(cls, v):
        checks = InputValidator.validate_password(v)
        if not checks['strong']:
            missing = [k for k, v in checks.items() if k != 'strong' and not v]
            raise ValueError(f'Password too weak. Missing: {", ".join(missing)}')
        return v

class SecureQueryInput(BaseModel):
    """Secure query input model"""
    query: str = Field(..., max_length=1000)

    @validator('query')
    def validate_query(cls, v):
        if InputValidator.check_sql_injection(v):
            raise ValueError('Invalid query: SQL injection detected')
        if InputValidator.check_xss(v):
            raise ValueError('Invalid query: XSS detected')
        return InputValidator.sanitize_input(v)

# Global instances
input_validator = InputValidator()
csrf_protection = CSRFProtection()
rate_limit_validator = RateLimitValidator()

def validate_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize request data"""
    try:
        return input_validator.sanitize_input(data)
    except Exception as e:
        log_security_event("input_validation_failed", {
            "error": str(e),
            "data_keys": list(data.keys()) if isinstance(data, dict) else None
        }, "error")
        raise HTTPException(status_code=400, detail="Invalid input data")

def check_rate_limit_middleware(client_ip: str) -> bool:
    """Middleware function for rate limiting"""
    return rate_limit_validator.check_rate_limit(client_ip)