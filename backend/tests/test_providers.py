"""
Tests for provider configuration, health checks, and secret redaction.

All tests are unit tests — no real API calls, no real keys required.
Live provider smoke tests are opt-in via CHAKRA_LIVE_PROVIDER_TESTS=1.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.provider import ProviderStatus, AllProvidersHealthResponse


# ---------------------------------------------------------------------------
# Part 1 — Config: API key loading and not_configured handling
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_missing_key_yields_not_configured(self, monkeypatch):
        """All five keys default to None when not set."""
        for var in [
            "TRONGRID_API_KEY", "ETHERSCAN_API_KEY", "HELIUS_API_KEY",
            "SOLSCAN_API_KEY", "CHAINABUSE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CHAKRA_DISABLE_ENV_FILE", "1")

        # Re-import settings with clean environment
        from importlib import reload
        import app.core.config as cfg_mod
        reload(cfg_mod)
        s = cfg_mod.Settings()
        assert s.TRONGRID_API_KEY is None
        assert s.ETHERSCAN_API_KEY is None
        assert s.HELIUS_API_KEY is None
        assert s.SOLSCAN_API_KEY is None
        assert s.CHAINABUSE_API_KEY is None

    def test_bitcoinabuse_not_required(self):
        """BitcoinAbuse must NOT be a required config field."""
        from importlib import reload
        import app.core.config as cfg_mod
        reload(cfg_mod)
        s = cfg_mod.Settings()
        assert not hasattr(s, "BITCOINABUSE_API_KEY"), (
            "BitcoinAbuse key must not be required"
        )

    def test_key_present_when_set(self, monkeypatch):
        """A non-empty env var is surfaced as a non-None value in settings."""
        monkeypatch.setenv("TRONGRID_API_KEY", "test-key-value")
        from importlib import reload
        import app.core.config as cfg_mod
        reload(cfg_mod)
        s = cfg_mod.Settings()
        assert s.TRONGRID_API_KEY is not None
        assert len(s.TRONGRID_API_KEY) > 0


# ---------------------------------------------------------------------------
# Part 2 — Provider base: not_configured short-circuit
# ---------------------------------------------------------------------------

class TestBaseProviderNotConfigured:
    @pytest.mark.asyncio
    async def test_missing_key_returns_not_checked(self):
        from app.providers.trongrid import TronGridClient
        client = TronGridClient(api_key=None)
        status = await client.check_health()
        assert status.config_status == "not_configured"
        assert status.health_status == "not_checked"
        assert status.provider == "trongrid"

    @pytest.mark.asyncio
    async def test_missing_key_etherscan(self):
        from app.providers.etherscan import EtherscanClient
        status = await EtherscanClient(None).check_health()
        assert status.config_status == "not_configured"

    @pytest.mark.asyncio
    async def test_missing_key_helius(self):
        from app.providers.helius import HeliusClient
        status = await HeliusClient(None).check_health()
        assert status.config_status == "not_configured"

    @pytest.mark.asyncio
    async def test_missing_key_solscan(self):
        from app.providers.solscan import SolscanClient
        status = await SolscanClient(None).check_health()
        assert status.config_status == "not_configured"

    @pytest.mark.asyncio
    async def test_missing_key_chainabuse(self):
        from app.providers.chainabuse import ChainAbuseClient
        status = await ChainAbuseClient(None).check_health()
        assert status.config_status == "not_configured"


# ---------------------------------------------------------------------------
# Part 3 — Secret redaction: key never appears in ProviderStatus
# ---------------------------------------------------------------------------

class TestSecretRedaction:
    @pytest.mark.asyncio
    async def test_api_key_not_in_status_fields(self):
        """The raw API key value must not appear in any ProviderStatus field."""
        FAKE_KEY = "super-secret-fake-key-xyz123"
        from app.providers.trongrid import TronGridClient
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
            client = TronGridClient(api_key=FAKE_KEY)
            status = await client.check_health()

        # Check all string fields in the status
        status_dict = status.model_dump()
        for field_name, value in status_dict.items():
            if isinstance(value, str):
                assert FAKE_KEY not in value, (
                    f"API key leaked into ProviderStatus.{field_name}!"
                )

    @pytest.mark.asyncio
    async def test_api_key_not_in_all_providers_response(self):
        """AllProvidersHealthResponse must not contain raw key values."""
        FAKE_TRONGRID = "trongrid-secret-abc"
        FAKE_ETHERSCAN = "etherscan-secret-def"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"status": "1", "result": "100000"})

        with patch.dict(os.environ, {
            "TRONGRID_API_KEY": FAKE_TRONGRID,
            "ETHERSCAN_API_KEY": FAKE_ETHERSCAN,
        }):
            from importlib import reload
            import app.core.config as cfg_mod
            reload(cfg_mod)
            from app.providers.registry import get_all_providers

            with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)), \
                 patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
                from app.services.provider_service import ProviderService
                result = await ProviderService.check_all()

        result_json = result.model_dump_json()
        for secret in [FAKE_TRONGRID, FAKE_ETHERSCAN]:
            assert secret not in result_json, f"Secret '{secret[:8]}...' leaked into response!"


# ---------------------------------------------------------------------------
# Part 4 — Status classification
# ---------------------------------------------------------------------------

class TestStatusClassification:
    def _make_client(self):
        from app.providers.trongrid import TronGridClient
        return TronGridClient(api_key="dummy")

    def test_200_is_healthy(self):
        assert self._make_client()._classify_http_status(200) == "healthy"

    def test_401_is_auth_failed(self):
        assert self._make_client()._classify_http_status(401) == "authentication_failed"

    def test_403_is_auth_failed(self):
        assert self._make_client()._classify_http_status(403) == "authentication_failed"

    def test_429_is_rate_limited(self):
        assert self._make_client()._classify_http_status(429) == "rate_limited"

    def test_500_is_unavailable(self):
        assert self._make_client()._classify_http_status(500) == "unavailable"

    def test_503_is_unavailable(self):
        assert self._make_client()._classify_http_status(503) == "unavailable"

    def test_404_is_error(self):
        assert self._make_client()._classify_http_status(404) == "error"

    @pytest.mark.asyncio
    async def test_timeout_maps_to_unavailable(self):
        import httpx
        from app.providers.trongrid import TronGridClient
        client = TronGridClient(api_key="dummy")
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
            status = await client.check_health()
        assert status.health_status == "unavailable"
        assert "timed out" in status.details.lower()

    @pytest.mark.asyncio
    async def test_network_error_maps_to_unavailable(self):
        import httpx
        from app.providers.trongrid import TronGridClient
        client = TronGridClient(api_key="dummy")
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.NetworkError("connect failed"))):
            status = await client.check_health()
        assert status.health_status == "unavailable"

    @pytest.mark.asyncio
    async def test_auth_failure_401(self):
        from app.providers.trongrid import TronGridClient
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        client = TronGridClient(api_key="dummy")
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()
        assert status.health_status == "authentication_failed"

    @pytest.mark.asyncio
    async def test_rate_limit_429(self):
        from app.providers.trongrid import TronGridClient
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        client = TronGridClient(api_key="dummy")
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()
        assert status.health_status == "rate_limited"


# ---------------------------------------------------------------------------
# Part 5 — Etherscan application-level auth detection
# ---------------------------------------------------------------------------

class TestEtherscanAppLevelAuth:
    @pytest.mark.asyncio
    async def test_etherscan_notok_is_auth_failure(self):
        from app.providers.etherscan import EtherscanClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "status": "0",
            "message": "NOTOK",
            "result": "Invalid API Key",
        })
        client = EtherscanClient(api_key="bad-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()
        assert status.health_status == "authentication_failed"

    @pytest.mark.asyncio
    async def test_etherscan_ok_is_healthy(self):
        from app.providers.etherscan import EtherscanClient
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={
            "status": "1",
            "message": "OK",
            "result": "123456789",
        })
        client = EtherscanClient(api_key="good-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()
        assert status.health_status == "healthy"


# ---------------------------------------------------------------------------
# Part 6 — ChainAbuse Authentication and Quota Safety Tests
# ---------------------------------------------------------------------------

class TestChainAbuseAuthAndQuota:
    @pytest.mark.asyncio
    async def test_chainabuse_zero_count_is_healthy(self):
        """count=0 with empty reports list indicates valid credentials and must be healthy."""
        from app.providers.chainabuse import ChainAbuseClient
        # Reset cache for testing
        ChainAbuseClient._cache_status = None
        ChainAbuseClient._cache_timestamp = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"count": 0, "reports": []})

        client = ChainAbuseClient(api_key="test-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()

        assert status.health_status == "healthy"
        assert "response count=0" in status.details

    @pytest.mark.asyncio
    async def test_chainabuse_missing_param_is_healthy(self):
        """HTTP 400 with 'Missing address or domain' confirms valid credentials."""
        from app.providers.chainabuse import ChainAbuseClient
        ChainAbuseClient._cache_status = None
        ChainAbuseClient._cache_timestamp = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json = MagicMock(return_value={"reason": "Missing address or domain"})

        client = ChainAbuseClient(api_key="test-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()

        assert status.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_chainabuse_401_is_auth_failure(self):
        """HTTP 401 indicates invalid credentials."""
        from app.providers.chainabuse import ChainAbuseClient
        ChainAbuseClient._cache_status = None
        ChainAbuseClient._cache_timestamp = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        client = ChainAbuseClient(api_key="bad-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            status = await client.check_health()

        assert status.health_status == "authentication_failed"

    @pytest.mark.asyncio
    async def test_chainabuse_caching_protects_quota(self):
        """Subsequent check_health calls within cache TTL must not make new HTTP calls."""
        from app.providers.chainabuse import ChainAbuseClient
        ChainAbuseClient._cache_status = None
        ChainAbuseClient._cache_timestamp = 0.0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"count": 5, "reports": []})

        client1 = ChainAbuseClient(api_key="valid-key")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)) as mock_get:
            status1 = await client1.check_health()
            assert mock_get.call_count == 1
            assert status1.health_status == "healthy"

            # Second call with new client instance within TTL
            client2 = ChainAbuseClient(api_key="valid-key")
            status2 = await client2.check_health()
            # Must NOT make another HTTP call
            assert mock_get.call_count == 1
            assert status2.health_status == "healthy"
            assert "(cached)" in status2.details



# ---------------------------------------------------------------------------
# Part 6 — ProviderService aggregation
# ---------------------------------------------------------------------------

class TestProviderServiceAggregation:
    @pytest.mark.asyncio
    async def test_all_not_configured_returns_full_response(self, monkeypatch):
        """With no keys set, service still returns all 5 providers as not_configured."""
        for var in [
            "TRONGRID_API_KEY", "ETHERSCAN_API_KEY", "HELIUS_API_KEY",
            "SOLSCAN_API_KEY", "CHAINABUSE_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CHAKRA_DISABLE_ENV_FILE", "1")

        from importlib import reload
        import app.core.config as cfg_mod
        reload(cfg_mod)
        from app.services.provider_service import ProviderService
        result = await ProviderService.check_all()

        assert isinstance(result, AllProvidersHealthResponse)
        assert len(result.providers) == 5
        for name, status in result.providers.items():
            assert status.config_status == "not_configured"
            assert status.health_status == "not_checked"

    @pytest.mark.asyncio
    async def test_provider_exception_is_caught(self):
        """An exception inside a provider check is caught and returns 'error' status."""
        from app.providers.base import BaseProviderClient
        from app.schemas.provider import ProviderStatus

        class BrokenClient(BaseProviderClient):
            provider_name = "broken_test"
            async def _probe(self) -> ProviderStatus:
                raise RuntimeError("simulated crash")

        clients = [BrokenClient(api_key="x")]

        with patch("app.services.provider_service.get_all_providers", return_value=clients):
            from app.services.provider_service import ProviderService
            result = await ProviderService.check_all()

        assert "broken_test" in result.providers
        assert result.providers["broken_test"].health_status == "error"
