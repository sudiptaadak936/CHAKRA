"""
Tests for FIX 3: KYCPreparationService — local prototype, no external calls.

Invariants tested:
- UNKNOWN registration does NOT produce UNREGISTERED_VASP_PATH
- NOT_REGISTERED produces UNREGISTERED_VASP_PATH
- REGISTERED produces REGISTERED_VASP_PATH
- UNKNOWN returns BLOCKED, never PREPARED
- No network calls are made
"""
import json
from pathlib import Path

import pytest

from app.attribution.engine_models import (
    AttributionConfidence,
    AttributionProvenance,
    AttributionDecision,
)
from app.attribution.fiu_registry import FIURegistryRepository, FIURegistrationStatus
from app.attribution.kyc import KYCPreparationService
from app.attribution.kyc_models import KYCRequestPath, KYCRequestStatus
from app.attribution.reid import FIUReIDBrancher
from app.attribution.reid_models import ReIDBranchDecision
from app.schemas.chain import Chain


@pytest.fixture()
def fiu_registry(tmp_path: Path) -> FIURegistryRepository:
    data = {
        "_metadata": {
            "data_classification": "DEMONSTRATION_ONLY",
            "disclaimer": "TEST",
            "source": "TEST",
            "schema_version": "1.0.0",
        },
        "vasps": [
            {
                "re_id": "FIU-TEST-001",
                "vasp_name": "RegisteredVASP",
                "registration_status": "REGISTERED",
                "registered_jurisdiction": "IN",
                "registration_date": "2023-01-01",
                "compliance_officer_contact": "test@example.invalid",
                "notes": "TEST",
            },
            {
                "re_id": None,
                "vasp_name": "UnregisteredVASP",
                "registration_status": "NOT_REGISTERED",
                "registered_jurisdiction": None,
                "registration_date": None,
                "compliance_officer_contact": None,
                "notes": "TEST",
            },
        ],
    }
    path = tmp_path / "demo_fiu_registry.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return FIURegistryRepository(registry_path=path)


def _make_reid_request(confidence, vasp_name, registry=None):
    """Helper to build a ReIdentificationRequest for testing."""
    provenance = AttributionProvenance(
        direct_observations=[],
        inferred_observations=[],
        cluster_id=None,
        corroborating_evidence_count=0,
        authority_class=None,
    )
    decision = AttributionDecision(
        address="test-address-001",
        chain=Chain.BITCOIN,
        confidence=confidence,
        vasp_name=vasp_name,
        provenance=provenance,
        explanation="test",
    )
    return FIUReIDBrancher.evaluate(decision=decision, fiu_registry=registry)


class TestKYCPreparationService:
    def test_registered_vasp_produces_registered_path(self, fiu_registry):
        request = _make_reid_request(
            AttributionConfidence.HIGH_CONFIDENCE, "RegisteredVASP", registry=fiu_registry
        )
        assert request.branch_decision == ReIDBranchDecision.REID_REQUIRED
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.kyc_status == KYCRequestStatus.PREPARED
        assert result.kyc_path == KYCRequestPath.REGISTERED_VASP_PATH
        assert result.packet is not None
        assert result.packet.re_id == "FIU-TEST-001"

    def test_not_registered_vasp_produces_unregistered_path(self, fiu_registry):
        request = _make_reid_request(
            AttributionConfidence.HIGH_CONFIDENCE, "UnregisteredVASP", registry=fiu_registry
        )
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.kyc_status == KYCRequestStatus.PREPARED
        assert result.kyc_path == KYCRequestPath.UNREGISTERED_VASP_PATH
        assert result.packet is not None
        assert result.packet.re_id is None

    def test_unknown_registration_never_produces_unregistered_path(self, fiu_registry):
        """INVARIANT: UNKNOWN must NEVER trigger UNREGISTERED_VASP_PATH."""
        request = _make_reid_request(
            AttributionConfidence.HIGH_CONFIDENCE, "SomeCompletelyUnknownVASP", registry=fiu_registry
        )
        assert request.registration_status == FIURegistrationStatus.UNKNOWN.value
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.kyc_path != KYCRequestPath.UNREGISTERED_VASP_PATH
        assert result.kyc_status == KYCRequestStatus.BLOCKED
        assert result.kyc_path == KYCRequestPath.UNKNOWN_REGISTRATION

    def test_confirmed_attribution_not_required(self, fiu_registry):
        request = _make_reid_request(
            AttributionConfidence.CONFIRMED, "RegisteredVASP", registry=fiu_registry
        )
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.kyc_status == KYCRequestStatus.NOT_REQUIRED

    def test_unknown_confidence_not_required(self, fiu_registry):
        request = _make_reid_request(
            AttributionConfidence.UNKNOWN, None, registry=None
        )
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.kyc_status == KYCRequestStatus.NOT_REQUIRED

    def test_none_request_raises(self):
        with pytest.raises(ValueError):
            KYCPreparationService.prepare(request=None)

    def test_no_network_calls(self, fiu_registry, monkeypatch):
        """No external network calls must be made."""
        import urllib.request
        def fail(*args, **kwargs):
            raise AssertionError("Network call detected during KYC preparation!")
        monkeypatch.setattr(urllib.request, "urlopen", fail)
        request = _make_reid_request(
            AttributionConfidence.HIGH_CONFIDENCE, "RegisteredVASP", registry=fiu_registry
        )
        KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)

    def test_packet_has_demonstration_classification(self, fiu_registry):
        request = _make_reid_request(
            AttributionConfidence.HIGH_CONFIDENCE, "RegisteredVASP", registry=fiu_registry
        )
        result = KYCPreparationService.prepare(request=request, fiu_registry=fiu_registry)
        assert result.packet is not None
        assert "DEMONSTRATION" in result.reason.upper()
