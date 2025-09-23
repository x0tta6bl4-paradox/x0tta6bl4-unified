"""Simplified mesh API orchestrator for x0tta6bl4 services."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.x0tta6bl4_settings import load_settings


DEFAULT_BACKENDS: Dict[str, int] = {
    "agents_gateway": 8301,
    "quantum_service": 8302,
    "task_controller": 8303,
}

settings = load_settings()
app = FastAPI(
    title="x0tta6bl4 Mesh API",
    description="Routes requests to internal platform backends",
    version=settings.system.version,
)


def _base_url() -> str:
    host = os.getenv("X0TTA6BL4_SERVICE_HOST", "http://localhost")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host.rstrip("/")


def _backend_url(name: str) -> str:
    env_key = f"X0TTA6BL4_BACKEND_{name.upper()}_URL"
    if env := os.getenv(env_key):
        return env.rstrip("/")
    port = DEFAULT_BACKENDS.get(name)
    if port is None:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {name}")
    return f"{_base_url()}:{port}"


class ProcessRequest(BaseModel):
    agents: List[str] = Field(..., description="Agent identifiers to invoke")
    payload: Dict[str, Any] = Field(..., description="Payload forwarded to agents")
    parallel: bool = Field(True, description="Whether to fan-out in parallel")


class QuantumRequest(BaseModel):
    algorithm: str = Field("grover", description="Quantum algorithm name")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    task_type: str = Field(..., description="Type of task to schedule")
    data: Dict[str, Any] = Field(..., description="Task payload")
    priority: int = Field(1, ge=1, le=10)


@app.get("/")
def index() -> Dict[str, Any]:
    return {
        "service": "x0tta6bl4-mesh-api",
        "version": settings.system.version,
        "environment": settings.system.environment,
        "backends": sorted(DEFAULT_BACKENDS.keys()),
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "mesh-api"}


async def _post(name: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("X0TTA6BL4_MESH_FAKE_MODE") == "1":
        return {
            "service": name,
            "endpoint": endpoint,
            "payload": payload,
            "status": "simulated",
        }
    url = f"{_backend_url(name)}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/agents/process")
async def process_agents(request: ProcessRequest) -> Dict[str, Any]:
    async def call(agent: str) -> Dict[str, Any]:
        return await _post(
            "agents_gateway",
            "/agents/process",
            payload={"agent": agent, "payload": request.payload},
        )

    if request.parallel:
        responses = await asyncio.gather(*(call(agent) for agent in request.agents))
    else:
        responses = []
        for agent in request.agents:
            responses.append(await call(agent))

    return {
        "agents": request.agents,
        "results": dict(zip(request.agents, responses)),
    }


@app.post("/quantum/run")
async def run_quantum(request: QuantumRequest) -> Dict[str, Any]:
    return await _post(
        "quantum_service",
        "/quantum/run",
        payload={"algorithm": request.algorithm, "parameters": request.parameters},
    )


@app.post("/tasks/schedule")
async def schedule_task(request: TaskRequest) -> Dict[str, Any]:
    return await _post(
        "task_controller",
        "/tasks/schedule",
        payload={
            "task_type": request.task_type,
            "priority": request.priority,
            "data": request.data,
        },
    )


__all__ = ["app"]
