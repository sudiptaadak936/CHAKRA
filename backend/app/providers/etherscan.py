"""Etherscan provider client.

Probe endpoint: GET https://api.etherscan.io/v2/api?chainid=1&module=stats&action=ethsupply&apikey=KEY
Uses Etherscan API V2 (V1 endpoints are deprecated).
This is a read-only, zero-cost query for total ETH supply.
"""
import httpx

from app.providers.base import BaseProviderClient
from app.schemas.provider import ProviderStatus

_PROBE_URL = "https://api.etherscan.io/v2/api"


class EtherscanClient(BaseProviderClient):
    provider_name = "etherscan"

    async def _probe(self) -> ProviderStatus:
        params = {
            "chainid": "1",
            "module": "stats",
            "action": "ethsupply",
            "apikey": self._api_key,
        }
        async with self._make_client() as client:
            resp = await client.get(_PROBE_URL, params=params)

        if resp.status_code != 200:
            health = self._classify_http_status(resp.status_code)
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status=health,
                details=f"HTTP {resp.status_code} from Etherscan.",
            )

        try:
            data = resp.json()
        except Exception:
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="error",
                details="Non-JSON response from Etherscan.",
            )

        status_val = str(data.get("status", ""))
        msg = str(data.get("message", ""))
        result = str(data.get("result", ""))

        if status_val == "1" or msg.upper() == "OK":
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="healthy",
                details="Etherscan API V2 probe succeeded.",
            )

        # Handle application-level error codes from Etherscan
        if status_val == "0":
            if "invalid" in result.lower() or "apikey" in result.lower():
                return ProviderStatus(
                    provider=self.provider_name,
                    config_status="configured",
                    health_status="authentication_failed",
                    details="Etherscan rejected the API key.",
                )
            if "rate limit" in result.lower() or "max rate" in result.lower():
                return ProviderStatus(
                    provider=self.provider_name,
                    config_status="configured",
                    health_status="rate_limited",
                    details="Etherscan rate limit reached.",
                )
            return ProviderStatus(
                provider=self.provider_name,
                config_status="configured",
                health_status="error",
                details=f"Etherscan reported error: {result}",
            )

        return ProviderStatus(
            provider=self.provider_name,
            config_status="configured",
            health_status="healthy",
            details="Etherscan probe succeeded.",
        )
