
"""
Конфигурация Billing для x0tta6bl4 Unified
"""

import os

BILLING_PROVIDERS = {
    "stripe": {
        "enabled": True,
        "public_key": os.getenv("STRIPE_PUBLIC_KEY"),
        "secret_key": os.getenv("STRIPE_SECRET_KEY")
    },
    "paypal": {
        "enabled": True,
        "client_id": os.getenv("PAYPAL_CLIENT_ID"),
        "client_secret": os.getenv("PAYPAL_CLIENT_SECRET")
    },
    "yookassa": {
        "enabled": True,
        "shop_id": os.getenv("YOOKASSA_SHOP_ID"),
        "secret_key": os.getenv("YOOKASSA_SECRET_KEY")
    }
}

SUBSCRIPTION_PLANS = {
    "free": {
        "price": 0,
        "features": ["basic_quantum", "limited_api"]
    },
    "pro": {
        "price": 29.99,
        "features": ["advanced_quantum", "unlimited_api", "priority_support"]
    },
    "enterprise": {
        "price": 99.99,
        "features": ["premium_quantum", "custom_integrations", "dedicated_support"]
    }
}
