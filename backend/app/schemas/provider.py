from typing import Optional, Literal
from pydantic import BaseModel


ProviderConfigStatus = Literal["configured", "not_configured"]
ProviderHealthStatus = Literal[
    "healthy",
    "authentication_failed",
    "rate_limited",
    "unavailable",
    "not_checked",
    "error",
]


class ProviderStatus(BaseModel):
    """Status of a single external provider.

    IMPORTANT: This model must never include the raw API key value.
    The `config_status` field indicates presence/absence only.
    """

    provider: str
    config_status: ProviderConfigStatus
    health_status: ProviderHealthStatus
    latency_ms: Optional[float] = None
    details: str
    quota_pricing_status: str = "not_programmatically_available"


class AllProvidersHealthResponse(BaseModel):
    """Aggregated response for GET /health/providers."""

    service: str = "chakra-backend"
    providers: dict[str, ProviderStatus]
