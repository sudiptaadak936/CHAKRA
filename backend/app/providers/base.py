"""Base class for all provider clients."""
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.schemas.provider import ProviderStatus

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10.0


class BaseProviderClient(ABC):
    """Abstract base for provider health-check clients.

    Subclasses must implement `provider_name` and `_probe()`.
    The API key is stored internally and must never be included in
    any returned `ProviderStatus` or log message.
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key  # private; must not appear in any output

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (no key data)."""
        ...

    @property
    def config_status(self) -> str:
        return "configured" if self._api_key else "not_configured"

    async def check_health(self) -> ProviderStatus:
        """Run a minimal safe probe against the provider API.

        Returns a ProviderStatus with the API key value redacted.
        """
        if not self._api_key:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="not_configured",
                health_status="not_checked",
                details="API key not configured.",
            )

        start = time.perf_counter()
        try:
            status = await self._probe()
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            status.latency_ms = elapsed
            return status
        except httpx.TimeoutException:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.warning("[%s] Health check timed out.", self.provider_name)
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="unavailable",
                latency_ms=elapsed,
                details="Request timed out.",
            )
        except httpx.NetworkError as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.warning("[%s] Network error: %s", self.provider_name, type(exc).__name__)
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="unavailable",
                latency_ms=elapsed,
                details=f"Network error: {type(exc).__name__}",
            )
        except Exception as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.error("[%s] Unexpected error during health check: %s", self.provider_name, type(exc).__name__)
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="error",
                latency_ms=elapsed,
                details=f"Unexpected error: {type(exc).__name__}",
            )

    @abstractmethod
    async def _probe(self) -> ProviderStatus:
        """Perform the actual API probe. May raise httpx exceptions."""
        ...

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=TIMEOUT_SECONDS)

    def _classify_http_status(self, http_status: int) -> str:
        """Map HTTP response code to ProviderHealthStatus."""
        if http_status == 200:
            return "healthy"
        if http_status in (401, 403):
            return "authentication_failed"
        if http_status == 429:
            return "rate_limited"
        if http_status >= 500:
            return "unavailable"
        return "error"
