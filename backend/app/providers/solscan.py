"""Solscan provider client.

Probe endpoint: GET https://pro-api.solscan.io/v2.0/token/list?sortBy=holder&direction=desc&page=1&page_size=1
This endpoint lists top tokens — a read-only call on Solscan Pro API.

Solscan is optional in CHAKRA; Helius + public Solana RPC are the primary
and sufficient infrastructure for Solana ingestion.
"""
import httpx

from app.providers.base import BaseProviderClient
from app.schemas.provider import ProviderStatus

_PROBE_URL = "https://pro-api.solscan.io/v2.0/token/list?sortBy=holder&direction=desc&page=1&page_size=1"


class SolscanClient(BaseProviderClient):
    provider_name = "solscan"

    async def _probe(self) -> ProviderStatus:
        headers = {"token": self._api_key}
        async with self._make_client() as client:
            resp = await client.get(_PROBE_URL, headers=headers)

        health = self._classify_http_status(resp.status_code)

        if resp.status_code == 401:
            try:
                body = resp.json()
                err_msg = str(body.get("errors", {}).get("message", "")).lower()
                if "upgrade" in err_msg or "plan" in err_msg or "level" in err_msg:
                    detail = (
                        "Solscan is optional; key requires paid Pro tier for v2.0 API. "
                        "Helius + public Solana RPC remain active and sufficient."
                    )
                else:
                    detail = "Solscan rejected the API key (401 Unauthorized)."
            except Exception:
                detail = "Solscan rejected the API key (401 Unauthorized)."
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="authentication_failed",
                details=detail,
            )
        elif health == "rate_limited":
            detail = "Solscan rate limit reached (429)."
        elif health == "unavailable":
            detail = f"Solscan server error {resp.status_code}."
        elif health == "healthy":
            detail = "Solscan probe succeeded."
        else:
            detail = f"Unexpected HTTP {resp.status_code} from Solscan."
            health = "error"

        return ProviderStatus(
            provider=self.provider_name,
            config_status="configured",
            health_status=health,
            details=detail,
        )
