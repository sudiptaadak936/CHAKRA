from pydantic import BaseModel
from typing import Dict, Literal


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    service: str = "chakra-backend"


class DependencyStatus(BaseModel):
    status: Literal["connected", "disconnected", "error"]
    details: str
    latency_ms: float


class DependenciesHealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str = "chakra-backend"
    dependencies: Dict[str, DependencyStatus]
    
    