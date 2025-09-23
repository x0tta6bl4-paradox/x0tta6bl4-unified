"""Minimal API gateway for the unified x0tta6bl4 platform."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.x0tta6bl4_settings import load_settings
from src.x0tta6bl4_security import SecurityHeadersMiddleware, RateLimitMiddleware, SecurityMonitor


DEFAULT_SERVICE_PORTS: Dict[str, int] = {
    "mesh_api": 8200,
    "agents_gateway": 8301,
    "quantum_service": 8302,
    "task_controller": 8303,
    "ml_pipeline": 8304,
    "analytics": 8305,
}


class UnifiedRequest(BaseModel):
    """Description of a proxied request to another service."""

    service: str = Field(..., description="Registered service name")
    endpoint: str = Field(..., description="Target endpoint, e.g. /health")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    timeout: Optional[float] = Field(
        None, description="Optional request timeout in seconds"
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="Extra headers forwarded to the downstream service"
    )
    payload: Optional[Dict[str, Any]] = Field(
        None, description="JSON payload forwarded to the downstream service"
    )


class ServiceStatus(BaseModel):
    name: str
    status: Literal["healthy", "unreachable", "error"]
    url: str
    response_time_ms: Optional[float] = None
    detail: Optional[str] = None


settings = load_settings()

# Initialize security components
security_monitor = SecurityMonitor()

app = FastAPI(
    title="x0tta6bl4 API Gateway",
    description="Lightweight proxy that fronts platform microservices",
    version=settings.system.version,
)

# Security middleware (applied in order of priority)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS middleware with restricted origins for security
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)


def _base_url() -> str:
    host = os.getenv("X0TTA6BL4_SERVICE_HOST", "http://localhost")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host.rstrip("/")


def _service_url(name: str) -> str:
    env_key = f"X0TTA6BL4_SERVICE_{name.upper()}_URL"
    if env := os.getenv(env_key):
        return env.rstrip("/")
    port = DEFAULT_SERVICE_PORTS.get(name)
    if port is None:
        raise HTTPException(status_code=400, detail=f"Unknown service: {name}")
    return f"{_base_url()}:{port}"


def _normalise_path(endpoint: str) -> str:
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


async def _proxy(request: UnifiedRequest) -> Dict[str, Any]:
    service_url = _service_url(request.service)
    url = f"{service_url}{_normalise_path(request.endpoint)}"
    timeout = request.timeout or float(os.getenv("X0TTA6BL4_GATEWAY_TIMEOUT", "10"))

    async with httpx.AsyncClient(timeout=timeout) as client:
        method = request.method
        try:
            response = await client.request(
                method,
                url,
                headers=request.headers,
                json=request.payload if method != "GET" else None,
                params=request.payload if method == "GET" else None,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:  # pragma: no cover - network issues
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return {"status": "ok", "raw": response.text}


@app.get("/")
def index() -> Dict[str, Any]:
    return {
        "service": "x0tta6bl4-api-gateway",
        "version": settings.system.version,
        "environment": settings.system.environment,
        "available_services": sorted(DEFAULT_SERVICE_PORTS.keys()),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "api-gateway"}


@app.get("/services/status", response_model=Dict[str, ServiceStatus])
async def services_status() -> Dict[str, ServiceStatus]:
    statuses: Dict[str, ServiceStatus] = {}

    async def check(name: str) -> None:
        url = _service_url(name)
        start = asyncio.get_event_loop().time()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{url}/health")
                elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
                if response.status_code == 200:
                    statuses[name] = ServiceStatus(
                        name=name,
                        status="healthy",
                        url=url,
                        response_time_ms=elapsed_ms,
                    )
                else:
                    statuses[name] = ServiceStatus(
                        name=name,
                        status="error",
                        url=url,
                        response_time_ms=elapsed_ms,
                        detail=f"Status code {response.status_code}",
                    )
        except httpx.RequestError as exc:
            statuses[name] = ServiceStatus(
                name=name,
                status="unreachable",
                url=url,
                detail=str(exc),
            )

    await asyncio.gather(*(check(name) for name in DEFAULT_SERVICE_PORTS))
    return statuses


@app.post("/unified/request")
async def unified_request(request: UnifiedRequest, req: Request) -> Dict[str, Any]:
    start_time = asyncio.get_event_loop().time()

    try:
        result = await _proxy(request)

        # Record successful request
        duration = asyncio.get_event_loop().time() - start_time
        security_monitor.record_request_duration(duration, "/unified/request", "POST")

        return result

    except Exception as e:
        # Record failed request
        duration = asyncio.get_event_loop().time() - start_time
        security_monitor.record_request_duration(duration, "/unified/request", "POST")

        # Log security violation if it's a potential attack
        client_ip = req.client.host if req.client else "unknown"
        security_monitor.log_security_violation(
            "api_gateway_error",
            "medium",
            {"error": str(e), "service": request.service},
            client_ip
        )

        raise


@app.middleware("http")
async def add_gateway_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Gateway-Service"] = "x0tta6bl4"
    response.headers["X-Gateway-Environment"] = settings.system.environment
    return response


__all__ = ["app"]
