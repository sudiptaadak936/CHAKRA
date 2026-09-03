"""ChainAbuse provider client.

Probe endpoint: GET https://api.chainabuse.com/v0/reports?address=0x0000000000000000000000000000000000000000
Uses HTTP Basic Authentication with API key as username and empty password.

Quota Safety & Caching:
To prevent aggressive health-check polling from exhausting ChainAbuse API quota,
responses are cached for 300 seconds (5 minutes). Within the cache window,
subsequent health checks return the cached status with zero network requests.
"""
import asyncio
import time
from typing import Optional

import httpx

from app.providers.base import BaseProviderClient
from app.schemas.provider import ProviderStatus

_PROBE_URL = "https://api.chainabuse.com/v0/reports"
_PROBE_PARAMS = {"address": "0x0000000000000000000000000000000000000000"}

# Cache TTL: 300 seconds (5 minutes)
CACHE_TTL_SECONDS = 300.0


class ChainAbuseClient(BaseProviderClient):
    provider_name = "chainabuse"

    # Class-level cache to persist across client instantiations
    _cache_status: Optional[ProviderStatus] = None
    _cache_timestamp: float = 0.0
    _lock = asyncio.Lock()

    async def check_health(self) -> ProviderStatus:
        """Check health with caching/throttling to preserve API quota."""
        if not self._api_key:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="not_configured",
                health_status="not_checked",
                details="API key not configured.",
            )

        now = time.monotonic()
        if self._cache_status is not None and (now - self._cache_timestamp) < CACHE_TTL_SECONDS:
            cached = self._cache_status.model_copy()
            if "(cached)" not in cached.details:
                cached.details = f"{cached.details} (cached)"
            return cached

        async with self._lock:
            # Re-check after acquiring lock (double-checked locking)
            now = time.monotonic()
            if self._cache_status is not None and (now - self._cache_timestamp) < CACHE_TTL_SECONDS:
                cached = self._cache_status.model_copy()
                if "(cached)" not in cached.details:
                    cached.details = f"{cached.details} (cached)"
                return cached

            status = await super().check_health()
            self.__class__._cache_status = status.model_copy()
            self.__class__._cache_timestamp = time.monotonic()
            return status

    async def _probe(self) -> ProviderStatus:
        # ChainAbuse v0 uses HTTP Basic Auth (api_key as username, empty password)
        async with self._make_client() as client:
            resp = await client.get(
                _PROBE_URL,
                auth=(self._api_key, ""),
                params=_PROBE_PARAMS,
            )

        if resp.status_code == 200:
            try:
                data = resp.json()
                count = data.get("count", 0)
                return ProviderStatus(
                    provider=self.provider_name,
                    config_status="configured",
                    health_status="healthy",
                    details=f"ChainAbuse API active (response count={count}).",
                )
            except Exception:
                return ProviderStatus(
                    provider=self.provider_name,
                    config_status="configured",
                    health_status="healthy",
                    details="ChainAbuse API responded HTTP 200.",
                )

        # 400 with 'Missing address or domain' also indicates valid credentials
        if resp.status_code == 400:
            try:
                data = resp.json()
                reason = str(data.get("reason", ""))
                if "address" in reason.lower() or "domain" in reason.lower():
                    return ProviderStatus(
                        provider=self.provider_name,
                        config_status="configured",
                        health_status="healthy",
                        details="ChainAbuse API authenticated (schema validated).",
                    )
            except Exception:
                pass

        if resp.status_code in (401, 403):
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="authentication_failed",
                details=f"ChainAbuse rejected the API key (HTTP {resp.status_code}).",
            )

        if resp.status_code == 429:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="rate_limited",
                details="ChainAbuse rate limit reached (429).",
            )

        if resp.status_code >= 500:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="unavailable",
                details=f"ChainAbuse server error {resp.status_code}.",
            )

        return ProviderStatus(
            provider=self.provider_name,
            config_status="configured",
            health_status="error",
            details=f"Unexpected HTTP {resp.status_code} from ChainAbuse.",
        )
