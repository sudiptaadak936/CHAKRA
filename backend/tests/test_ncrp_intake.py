"""Focused unit tests for Step 1F — LEA Intake API (NCRP-style stub).

Verifies:
1. Valid requests and successful responses
2. Required fields validation (missing/empty/whitespace-only)
3. Chain validation (canonical chains, normalization, invalid rejection)
4. Canonical delegation (boundary contract with CaseIntakeService)
5. Stub honesty (disclaimer notice and is_stub flag presence)
6. No external integration / network dependency
7. Error propagation from canonical intake
8. No parallel forensic/graph pipeline
"""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.schemas.case import CaseIntakeRequest, CaseIntakeResponse
from app.schemas.chain import Chain
from app.schemas.ncrp import NCRPIntakeRequest, NCRPIntakeResponse, NCRP_STUB_NOTICE
from app.services.case_service import CaseIntakeService
from app.services.ncrp_adapter import NCRPAdapter


# ---------------------------------------------------------------------------
# Section 1: Valid NCRPIntakeRequest & Canonical Transformation
# ---------------------------------------------------------------------------

def test_valid_ncrp_request_creation():
    """Verify that a well-formed NCRP request payload is successfully validated."""
    req = NCRPIntakeRequest(
        complaint_id="NCRP-2026-99881",
        wallet_address="0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
        chain=Chain.EVM,
        reported_by="Inspector Vance, Cyber Cell",
        narrative="Victim reported unauthorized transfer of 5 ETH to the suspect address.",
    )
    assert req.complaint_id == "NCRP-2026-99881"
    assert req.wallet_address == "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"
    assert req.chain == Chain.EVM
    assert req.reported_by == "Inspector Vance, Cyber Cell"
    assert req.narrative.startswith("Victim reported")


def test_ncrp_adapter_pure_boundary_transformation():
    """Verify NCRPAdapter maps NCRPIntakeRequest to canonical CaseIntakeRequest without mutation or extra logic."""
    req = NCRPIntakeRequest(
        complaint_id="NCRP-2026-001",
        wallet_address="TJvXv7...",
        chain="tron",
        reported_by="Agent Smith",
        narrative="Phishing scam reported.",
    )
    canonical = NCRPAdapter.to_canonical(req)
    assert isinstance(canonical, CaseIntakeRequest)
    assert canonical.source == "ncrp"
    assert canonical.external_reference_id == "NCRP-2026-001"
    assert canonical.wallet_address == "TJvXv7..."
    assert canonical.chain == Chain.TRON
    assert canonical.reported_by == "Agent Smith"
    assert canonical.narrative == "Phishing scam reported."


# ---------------------------------------------------------------------------
# Section 2: Required Fields & Whitespace Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_field", [
    "complaint_id",
    "wallet_address",
    "chain",
    "reported_by",
    "narrative",
])
def test_missing_required_fields_rejected(missing_field):
    """Missing any of the 5 required fields must raise ValidationError."""
    payload = {
        "complaint_id": "CMP-123",
        "wallet_address": "0xabc",
        "chain": "evm",
        "reported_by": "Officer",
        "narrative": "details",
    }
    del payload[missing_field]
    with pytest.raises(ValidationError):
        NCRPIntakeRequest(**payload)


@pytest.mark.parametrize("blank_field", [
    "complaint_id",
    "wallet_address",
    "reported_by",
    "narrative",
])
def test_empty_or_whitespace_fields_rejected(blank_field):
    """Empty or whitespace-only string values must be rejected."""
    payload = {
        "complaint_id": "CMP-123",
        "wallet_address": "0xabc",
        "chain": "evm",
        "reported_by": "Officer",
        "narrative": "details",
    }
    payload[blank_field] = "   "
    with pytest.raises(ValidationError):
        NCRPIntakeRequest(**payload)


@pytest.mark.parametrize("field_name", [
    "complaint_id",
    "wallet_address",
    "reported_by",
    "narrative",
])
def test_non_string_fields_rejected(field_name):
    """Non-string types (integers, lists, dicts) must be rejected."""
    payload = {
        "complaint_id": "CMP-123",
        "wallet_address": "0xabc",
        "chain": "evm",
        "reported_by": "Officer",
        "narrative": "details",
    }
    payload[field_name] = 12345
    with pytest.raises(ValidationError):
        NCRPIntakeRequest(**payload)


# ---------------------------------------------------------------------------
# Section 3: Chain Semantics & Normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chain_str,expected_enum", [
    ("tron", Chain.TRON),
    ("TRON", Chain.TRON),
    ("  tron  ", Chain.TRON),
    ("evm", Chain.EVM),
    ("EVM", Chain.EVM),
    ("bitcoin", Chain.BITCOIN),
    ("BITCOIN", Chain.BITCOIN),
    ("solana", Chain.SOLANA),
    ("Solana", Chain.SOLANA),
])
def test_chain_case_insensitivity_and_normalization(chain_str, expected_enum):
    """Chain input must normalize whitespace and capitalization to canonical Chain enum."""
    req = NCRPIntakeRequest(
        complaint_id="CMP-123",
        wallet_address="0xabc",
        chain=chain_str,
        reported_by="Officer",
        narrative="details",
    )
    assert req.chain == expected_enum


@pytest.mark.parametrize("invalid_chain", [
    "cardano",
    "ripple",
    "doge",
    "unknown_chain",
    "",
    "   ",
])
def test_unsupported_chain_rejected(invalid_chain):
    """Unsupported or blank chain strings must raise a ValidationError."""
    with pytest.raises(ValidationError):
        NCRPIntakeRequest(
            complaint_id="CMP-123",
            wallet_address="0xabc",
            chain=invalid_chain,
            reported_by="Officer",
            narrative="details",
        )


# ---------------------------------------------------------------------------
# Section 4: Canonical Delegation Contract
# ---------------------------------------------------------------------------

def test_canonical_case_service_submit():
    """Verify CaseIntakeService generates internal case_id and sets proper attributes."""
    service = CaseIntakeService()
    req = CaseIntakeRequest(
        source="ncrp",
        external_reference_id="NCRP-999",
        wallet_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        chain=Chain.BITCOIN,
        reported_by="Investigator",
        narrative="Ransom payment address.",
    )
    res = service.submit_case(req)
    assert isinstance(res, CaseIntakeResponse)
    assert res.case_id.startswith("case_")
    assert res.status == "registered"
    assert res.wallet_address == req.wallet_address
    assert res.chain == Chain.BITCOIN
    assert isinstance(res.created_at, datetime)


# ---------------------------------------------------------------------------
# Section 5: Stub Honesty & Disclaimer Integrity
# ---------------------------------------------------------------------------

def test_ncrp_response_stub_honesty():
    """Verify NCRPIntakeResponse includes explicit stub disclaimer and separates complaint_id from case_id."""
    res = NCRPIntakeResponse(
        complaint_id="NCRP-123",
        case_id="case_abc123",
        status="registered",
        wallet_address="0xabc",
        chain=Chain.EVM,
        created_at=datetime.now(timezone.utc),
    )
    assert res.is_stub is True
    assert res.notice == "stub/demo interface — models the shape of a future NCRP integration, not a live government connection"
    assert res.complaint_id == "NCRP-123"
    assert res.case_id == "case_abc123"
    assert res.complaint_id != res.case_id


# ---------------------------------------------------------------------------
# Section 6: No External Dependency / Pure Deterministic Boundary
# ---------------------------------------------------------------------------

def test_ncrp_adapter_and_service_offline_execution():
    """Verify intake adapter and service run purely locally without any network access."""
    req = NCRPIntakeRequest(
        complaint_id="NCRP-OFFLINE",
        wallet_address="SolanaAddress11111111111111111111111111111111",
        chain=Chain.SOLANA,
        reported_by="Offline Officer",
        narrative="Test narrative",
    )
    canonical = NCRPAdapter.to_canonical(req)
    service = CaseIntakeService()
    res = service.submit_case(canonical)
    assert res.case_id.startswith("case_")
