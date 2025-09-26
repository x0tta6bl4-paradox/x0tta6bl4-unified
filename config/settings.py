"""
x0tta6bl4-unified Configuration Settings
Central configuration file for the unified platform
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Environment detection
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Environment configurations
ENV_CONFIGS = {
    'development': {
        'debug': True,
        'host': 'localhost',
        'port': 8000,
        'database_url': 'sqlite:///dev.db',
        'redis_url': 'redis://localhost:6379/0',
        'log_level': 'DEBUG',
    },
    'staging': {
        'debug': False,
        'host': '0.0.0.0',
        'port': 8000,
        'database_url': os.getenv('DATABASE_URL', 'postgresql://user:pass@staging-db:5432/staging'),
        'redis_url': os.getenv('REDIS_URL', 'redis://staging-redis:6379/0'),
        'log_level': 'INFO',
    },
    'production': {
        'debug': False,
        'host': '0.0.0.0',
        'port': 8000,
        'database_url': os.getenv('DATABASE_URL'),
        'redis_url': os.getenv('REDIS_URL'),
        'log_level': 'WARNING',
    }
}

# Current environment config
CURRENT_CONFIG = ENV_CONFIGS.get(ENVIRONMENT, ENV_CONFIGS['development'])

# Component configurations
COMPONENT_CONFIGS = {
    'quantum': {
        'enabled': True,
        'max_qubits': 1000,
        'optimization_level': 2,
        'backend': 'qiskit',
    },
    'ai': {
        'enabled': True,
        'model_provider': 'openai',
        'max_tokens': 4096,
        'temperature': 0.7,
    },
    'enterprise': {
        'enabled': True,
        'max_users': 10000,
        'features': ['analytics', 'reporting', 'api_access'],
    },
    'billing': {
        'enabled': True,
        'currency': 'USD',
        'tax_rate': 0.08,
        'payment_provider': 'stripe',
    }
}

# API Keys and secrets (loaded from environment)
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'stripe': os.getenv('STRIPE_SECRET_KEY'),
    'database': os.getenv('DATABASE_PASSWORD'),
}

# Security settings
SECURITY = {
    'secret_key': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'jwt_expiry': 3600,  # seconds
    'cors_origins': ['http://localhost:3000', 'https://app.x0tta6bl4.com'],
}

# Monitoring and metrics
MONITORING = {
    'prometheus_enabled': True,
    'metrics_port': 9090,
    'alert_webhook': os.getenv('ALERT_WEBHOOK_URL'),
}

# Feature flags
FEATURE_FLAGS = {
    'quantum_computing': True,
    'ai_assistance': True,
    'enterprise_features': True,
    'billing_integration': True,
}