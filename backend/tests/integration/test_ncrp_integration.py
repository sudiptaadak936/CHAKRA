"""End-to-end integration tests for POST /intake/ncrp endpoint (Step 1F).

Verifies:
1. HTTP POST /intake/ncrp successfully accepts valid payloads
2. Architectural flow: HTTP POST -> Router -> NCRPAdapter -> Canonical CaseIntakeService
3. Stub honesty disclaimer in API response and schema documentation
4. Field validation failures return 422 Unprocessable Entity
5. Canonical intake errors propagate without being masked as fake success
6. Dependency override mechanism cleanly replaces canonical intake service in tests
"""
from datetime import datetime, timezone
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.case import CaseIntakeRequest, CaseIntakeResponse
from app.schemas.chain import Chain
from app.schemas.ncrp import NCRP_STUB_NOTICE
from app.services.case_service import CaseIntakeService, get_case_service


@pytest.mark.asyncio
async def test_post_intake_ncrp_success():
    """Verify HTTP POST /intake/ncrp returns 200, valid payload, and stub honesty notice."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "complaint_id": "NCRP-2026-004412",
            "wallet_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "chain": "evm",
            "reported_by": "Special Agent Miller",
            "narrative": "Unauthorized drain of USDC tokens via malicious permit signature.",
        }
        response = await client.post("/intake/ncrp", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["complaint_id"] == "NCRP-2026-004412"
        assert data["case_id"].startswith("case_")
        assert data["status"] == "registered"
        assert data["wallet_address"] == "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        assert data["chain"] == "evm"
        assert data["is_stub"] is True
        assert data["notice"] == NCRP_STUB_NOTICE
        # Ensure external complaint_id is distinctly not the internal case_id
        assert data["complaint_id"] != data["case_id"]


@pytest.mark.asyncio
async def test_post_intake_ncrp_field_validation_errors():
    """Verify invalid payloads return HTTP 422 with validation errors."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing narrative
        bad_payload_missing = {
            "complaint_id": "NCRP-123",
            "wallet_address": "0x123",
            "chain": "evm",
            "reported_by": "Officer",
        }
        res = await client.post("/intake/ncrp", json=bad_payload_missing)
        assert res.status_code == 422

        # Empty whitespace string
        bad_payload_whitespace = {
            "complaint_id": "   ",
            "wallet_address": "0x123",
            "chain": "evm",
            "reported_by": "Officer",
            "narrative": "valid",
        }
        res = await client.post("/intake/ncrp", json=bad_payload_whitespace)
        assert res.status_code == 422

        # Invalid chain
        bad_payload_chain = {
            "complaint_id": "NCRP-123",
            "wallet_address": "0x123",
            "chain": "dogecoin",
            "reported_by": "Officer",
            "narrative": "valid",
        }
        res = await client.post("/intake/ncrp", json=bad_payload_chain)
        assert res.status_code == 422


@pytest.mark.asyncio
async def test_canonical_delegation_via_dependency_override():
    """Prove POST /intake/ncrp delegates into the canonical CaseIntakeService.

    Uses FastAPI's dependency_overrides to inject a mock service and verify
    the canonical request was properly received with source='ncrp'.
    """
    received_requests = []

    class MockCanonicalService(CaseIntakeService):
        def submit_case(self, request: CaseIntakeRequest) -> CaseIntakeResponse:
            received_requests.append(request)
            return CaseIntakeResponse(
                case_id="case_mock_override_999",
                status="registered",
                wallet_address=request.wallet_address,
                chain=request.chain,
                created_at=datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc),
            )

    app.dependency_overrides[get_case_service] = lambda: MockCanonicalService()

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {
                "complaint_id": "NCRP-MOCK-TEST-1",
                "wallet_address": "TQn9Y2khEsLJW1ChVWFMSMeSTow5K3GW5Z",
                "chain": "tron",
                "reported_by": "Cyber Desk",
                "narrative": "USDT extortion claim.",
            }
            response = await client.post("/intake/ncrp", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["case_id"] == "case_mock_override_999"
            assert data["complaint_id"] == "NCRP-MOCK-TEST-1"
            assert data["is_stub"] is True
            assert data["notice"] == NCRP_STUB_NOTICE

            # Verify the mock received exactly one canonical request with correct mapping
            assert len(received_requests) == 1
            canonical_req = received_requests[0]
            assert canonical_req.source == "ncrp"
            assert canonical_req.external_reference_id == "NCRP-MOCK-TEST-1"
            assert canonical_req.wallet_address == "TQn9Y2khEsLJW1ChVWFMSMeSTow5K3GW5Z"
            assert canonical_req.chain == Chain.TRON
            assert canonical_req.reported_by == "Cyber Desk"
            assert canonical_req.narrative == "USDT extortion claim."
    finally:
        app.dependency_overrides.pop(get_case_service, None)


@pytest.mark.asyncio
async def test_canonical_error_propagation():
    """Verify that canonical intake failures propagate rather than being converted into fake successes."""
    class FailingCanonicalService(CaseIntakeService):
        def submit_case(self, request: CaseIntakeRequest) -> CaseIntakeResponse:
            raise RuntimeError("Canonical intake database unavailable")

    app.dependency_overrides[get_case_service] = lambda: FailingCanonicalService()

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {
                "complaint_id": "NCRP-FAIL-TEST",
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "chain": "evm",
                "reported_by": "Officer",
                "narrative": "details",
            }
            with pytest.raises(RuntimeError, match="Canonical intake database unavailable"):
                await client.post("/intake/ncrp", json=payload)
    finally:
        app.dependency_overrides.pop(get_case_service, None)
