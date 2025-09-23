"""Shared configuration loader for the unified x0tta6bl4 platform."""

from __future__ import annotations

import functools
import pathlib
from dataclasses import dataclass
from typing import Any, Dict

import yaml

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "x0tta6bl4_config.yaml"


class ConfigurationError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


@functools.lru_cache(maxsize=1)
def _load_raw_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise ConfigurationError(f"Configuration file not found: {_CONFIG_PATH}")
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return data


def get_raw_config() -> Dict[str, Any]:
    """Return the raw dict version of the system configuration."""

    return dict(_load_raw_config())


@dataclass(frozen=True)
class PortsConfig:
    mesh_api: int
    main_api: int
    cultural_api: int
    dashboard_wrapper: int
    phi_harmonic_load_balancer: int


@dataclass(frozen=True)
class AgentsConfig:
    enabled: bool
    concurrent_limit: int
    timeout_seconds: int
    health_check_interval: int


@dataclass(frozen=True)
class PricingTierConfig:
    name: str
    monthly_price: float
    max_qubits: int
    max_shots_per_month: int
    features: List[str]


@dataclass(frozen=True)
class BillingConfig:
    enabled: bool
    currency: str
    pricing_tiers: Dict[str, PricingTierConfig]
    usage_pricing: Dict[str, float]
    discounts: Dict[str, float]


@dataclass(frozen=True)
class SystemConfig:
    name: str
    version: str
    environment: str
    phi_constant: float
    sacred_frequency: int


@dataclass(frozen=True)
class Settings:
    system: SystemConfig
    ports: PortsConfig
    agents: AgentsConfig
    billing: BillingConfig
    raw: Dict[str, Any]


def load_settings() -> Settings:
    """Parse and return a strongly typed subset of the configuration."""

    config = _load_raw_config()

    try:
        system = config["system"]
        ports = config["ports"]
        agents = config["agents"]
        billing = config["billing"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Missing required section: {exc}") from exc

    system_cfg = SystemConfig(
        name=str(system.get("name", "x0tta6bl4")),
        version=str(system.get("version", "0.0.0")),
        environment=str(system.get("environment", "development")),
        phi_constant=float(system.get("phi_constant", 1.6180339887)),
        sacred_frequency=int(system.get("sacred_frequency", 108)),
    )

    ports_cfg = PortsConfig(
        mesh_api=int(ports.get("mesh_api", 8200)),
        main_api=int(ports.get("main_api", 8080)),
        cultural_api=int(ports.get("cultural_api", 8100)),
        dashboard_wrapper=int(ports.get("dashboard_wrapper", 8123)),
        phi_harmonic_load_balancer=int(ports.get("phi_harmonic_load_balancer", 8310)),
    )

    agents_cfg = AgentsConfig(
        enabled=bool(agents.get("enabled", True)),
        concurrent_limit=int(agents.get("concurrent_limit", 8)),
        timeout_seconds=int(agents.get("timeout_seconds", 300)),
        health_check_interval=int(agents.get("health_check_interval", 30)),
    )

    # Load billing config
    pricing_tiers = {}
    for tier_name, tier_data in billing.get("pricing_tiers", {}).items():
        pricing_tiers[tier_name] = PricingTierConfig(
            name=str(tier_data.get("name", tier_name)),
            monthly_price=float(tier_data.get("monthly_price", 0.0)),
            max_qubits=int(tier_data.get("max_qubits", 0)),
            max_shots_per_month=int(tier_data.get("max_shots_per_month", 0)),
            features=list(tier_data.get("features", [])),
        )

    billing_cfg = BillingConfig(
        enabled=bool(billing.get("enabled", True)),
        currency=str(billing.get("currency", "USD")),
        pricing_tiers=pricing_tiers,
        usage_pricing=dict(billing.get("usage_pricing", {})),
        discounts=dict(billing.get("discounts", {})),
    )

    return Settings(
        system=system_cfg,
        ports=ports_cfg,
        agents=agents_cfg,
        billing=billing_cfg,
        raw=config,
    )


__all__ = ["ConfigurationError", "Settings", "load_settings", "get_raw_config"]
