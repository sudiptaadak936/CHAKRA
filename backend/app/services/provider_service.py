"""Provider health service — aggregates all provider health checks."""
import asyncio
import logging
from typing import Dict

from app.providers.registry import get_all_providers
from app.schemas.provider import ProviderStatus, AllProvidersHealthResponse

logger = logging.getLogger(__name__)


class ProviderService:
    @staticmethod
    async def check_all() -> AllProvidersHealthResponse:
        """Run all provider health checks concurrently.

        Returns a response containing per-provider statuses.
        API keys are never included in any returned value.
        """
        providers = get_all_providers()
        tasks = [p.check_health() for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        statuses: Dict[str, ProviderStatus] = {}
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                logger.error(
                    "[%s] Unexpected exception during health check: %s",
                    provider.provider_name,
                    type(result).__name__,
                )
                statuses[provider.provider_name] = ProviderStatus(
                    provider=provider.provider_name,
                    config_status=provider.config_status,
                    health_status="error",
                    details=f"Unexpected exception: {type(result).__name__}",
                )
            else:
                statuses[provider.provider_name] = result

        return AllProvidersHealthResponse(providers=statuses)
