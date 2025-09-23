
"""
Конфигурация Billing для x0tta6bl4 Unified
"""

BILLING_PROVIDERS = {
    "stripe": {
        "enabled": True,
        "public_key": "pk_test_...",
        "secret_key": "sk_test_..."
    },
    "paypal": {
        "enabled": True,
        "client_id": "your_paypal_client_id",
        "client_secret": "your_paypal_client_secret"
    },
    "yookassa": {
        "enabled": True,
        "shop_id": "your_shop_id",
        "secret_key": "your_secret_key"
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
