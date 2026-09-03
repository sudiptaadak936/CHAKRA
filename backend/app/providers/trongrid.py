"""TronGrid provider client.

Probe endpoint: POST https://api.trongrid.io/wallet/getnowblock
Returns the latest block — a read-only, zero-cost call that validates
the API key and connectivity in one step.
"""
import httpx

from app.providers.base import BaseProviderClient
from app.schemas.provider import ProviderStatus

_PROBE_URL = "https://api.trongrid.io/wallet/getnowblock"


class TronGridClient(BaseProviderClient):
    provider_name = "trongrid"

    async def _probe(self) -> ProviderStatus:
        headers = {"TRON-PRO-API-KEY": self._api_key}
        async with self._make_client() as client:
            resp = await client.post(_PROBE_URL, headers=headers, json={})

        health = self._classify_http_status(resp.status_code)
        if health == "authentication_failed":
            detail = "TronGrid rejected the API key (401/403)."
        elif health == "rate_limited":
            detail = "TronGrid rate limit reached (429)."
        elif health == "unavailable":
            detail = f"TronGrid returned server error {resp.status_code}."
        elif health == "healthy":
            detail = "TronGrid getnowblock succeeded."
        else:
            detail = f"Unexpected HTTP {resp.status_code} from TronGrid."

        return ProviderStatus(
            provider=self.provider_name,
            config_status="configured",
            health_status=health,
            details=detail,
        )
