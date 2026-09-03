"""Helius provider client (Solana RPC + enhanced APIs).

Probe endpoint: POST https://mainnet.helius-rpc.com/?api-key=KEY
with a minimal getHealth JSON-RPC call (no cost, no data written).
"""
import httpx

from app.providers.base import BaseProviderClient
from app.schemas.provider import ProviderStatus

_BASE_URL = "https://mainnet.helius-rpc.com/"

_PROBE_PAYLOAD = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getHealth",
    "params": [],
}


class HeliusClient(BaseProviderClient):
    provider_name = "helius"

    async def _probe(self) -> ProviderStatus:
        url = f"{_BASE_URL}?api-key={self._api_key}"
        async with self._make_client() as client:
            resp = await client.post(url, json=_PROBE_PAYLOAD)

        if resp.status_code == 401 or resp.status_code == 403:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="authentication_failed",
                details=f"Helius rejected the API key (HTTP {resp.status_code}).",
            )
        if resp.status_code == 429:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="rate_limited",
                details="Helius rate limit reached (429).",
            )
        if resp.status_code >= 500:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="unavailable",
                details=f"Helius server error {resp.status_code}.",
            )

        try:
            data = resp.json()
        except Exception:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="error",
                details="Non-JSON response from Helius.",
            )

        rpc_result = data.get("result", "")
        if rpc_result == "ok":
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="healthy",
                details="Helius RPC getHealth returned ok.",
            )

        # Some auth failures surface in the JSON-RPC error field
        rpc_error = data.get("error", {})
        if isinstance(rpc_error, dict):
            code = rpc_error.get("code", 0)
            if code in (-32005, -32600, 403):
                return ProviderStatus(
                    provider=self.provider_name,
                    config_status="configured",
                    health_status="authentication_failed",
                    details=f"Helius RPC auth error: {rpc_error.get('message', '')}",
                )

        return ProviderStatus(
            provider=self.provider_name,
            config_status="configured",
            health_status="healthy",
            details=f"Helius probe succeeded (result={rpc_result!r}).",
        )
