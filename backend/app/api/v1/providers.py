"""Provider health endpoints.

GET /health/providers — returns per-provider configuration and health status.
API keys are NEVER included in any response.
"""
from fastapi import APIRouter
from app.schemas.provider import AllProvidersHealthResponse
from app.services.provider_service import ProviderService

router = APIRouter(tags=["Providers"])


@router.get("/health/providers", response_model=AllProvidersHealthResponse)
async def get_providers_health() -> AllProvidersHealthResponse:
    """Check all configured external providers.

    Returns configuration status (configured/not_configured) and health
    status for each provider. API keys are never included in the response.
    """
    return await ProviderService.check_all()
