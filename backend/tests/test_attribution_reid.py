import pytest
from app.schemas.chain import Chain
from app.attribution.models import VASPAddressRecord
from app.attribution.authority import AuthorityClass
from app.attribution.engine_models import (
    AttributionDecision,
    AttributionConfidence,
    AttributionProvenance
)
from app.attribution.reid_models import (
    ReIDStatus,
    ReIDBranchDecision,
    ReIdentificationRequest
)
from app.attribution.reid import FIUReIDBrancher

def create_mock_decision(
    confidence: AttributionConfidence,
    vasp_name: str = "TestVASP",
    evidence_id: str = "EVID-123",
    cluster_id: str = None
) -> AttributionDecision:
    obs = VASPAddressRecord(
        address="0xTestAddress",
        chain=Chain.EVM,
        vasp_name=vasp_name,
        service_type="exchange",
        source="test_source",
        evidence_id=evidence_id
    )
    prov = AttributionProvenance(
        direct_observations=[obs] if confidence == AttributionConfidence.CONFIRMED else [],
        inferred_observations=[obs] if confidence != AttributionConfidence.CONFIRMED and confidence != AttributionConfidence.UNKNOWN else [],
        cluster_id=cluster_id,
        corroborating_evidence_count=1,
        authority_class=AuthorityClass.AUTHORITATIVE if confidence == AttributionConfidence.CONFIRMED else AuthorityClass.NON_AUTHORITATIVE
    )
    return AttributionDecision(
        address="0xTestAddress",
        chain=Chain.EVM,
        confidence=confidence,
        vasp_name=vasp_name if confidence != AttributionConfidence.UNKNOWN else None,
        service_type="exchange" if confidence != AttributionConfidence.UNKNOWN else None,
        candidate_attributions=[obs] if confidence != AttributionConfidence.UNKNOWN else [],
        provenance=prov,
        explanation="Test explanation"
    )

def test_reid_branch_confirmed():
    # Test 1 — CONFIRMED -> NO_REID_REQUIRED
    decision = create_mock_decision(AttributionConfidence.CONFIRMED)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.NO_REID_REQUIRED
    assert req.status == ReIDStatus.REQUEST_PREPARED

def test_reid_branch_high_confidence():
    # Test 2 — HIGH_CONFIDENCE -> REID_REQUIRED
    decision = create_mock_decision(AttributionConfidence.HIGH_CONFIDENCE)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.REID_REQUIRED
    assert req.status == ReIDStatus.REQUEST_PREPARED

def test_reid_branch_probable_inferred():
    # Test 3 — PROBABLE_INFERRED -> REID_REQUIRED
    decision = create_mock_decision(AttributionConfidence.PROBABLE_INFERRED)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.REID_REQUIRED
    assert req.status == ReIDStatus.REQUEST_PREPARED

def test_reid_branch_unknown():
    # Test 4 — UNKNOWN -> REID_NOT_JUSTIFIED, no fabricated target
    decision = create_mock_decision(AttributionConfidence.UNKNOWN)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.REID_NOT_JUSTIFIED
    assert req.attributed_vasp is None
    assert req.status == ReIDStatus.REQUEST_PREPARED

def test_reid_branch_missing_vasp():
    # Test 5 — Missing VASP -> fail closed
    decision = AttributionDecision(
        address="0x123",
        chain=Chain.EVM,
        confidence=AttributionConfidence.PROBABLE_INFERRED,
        vasp_name=None, # Missing VASP
        service_type=None,
        provenance=AttributionProvenance(),
        explanation="Missing VASP candidate"
    )
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.REID_NOT_JUSTIFIED
    assert req.attributed_vasp is None

def test_reid_branch_provenance_preservation():
    # Test 6 — Provenance preservation
    decision = create_mock_decision(AttributionConfidence.PROBABLE_INFERRED, evidence_id="EVID-777", cluster_id="CLUSTER-99")
    req = FIUReIDBrancher.evaluate(decision)
    assert "EVID-777" in req.evidence_ids
    assert "CLUSTER-99" in req.evidence_ids
    assert req.provenance == decision.provenance

def test_reid_branch_no_authority_escalation():
    # Test 7 — No authority escalation
    decision = create_mock_decision(AttributionConfidence.PROBABLE_INFERRED)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.attribution_confidence == AttributionConfidence.PROBABLE_INFERRED
    # Confidence level remains unchanged in the request model
    assert req.branch_decision == ReIDBranchDecision.REID_REQUIRED

def test_reid_branch_determinism():
    # Test 8 — Determinism
    decision1 = create_mock_decision(AttributionConfidence.PROBABLE_INFERRED)
    decision2 = create_mock_decision(AttributionConfidence.PROBABLE_INFERRED)
    req1 = FIUReIDBrancher.evaluate(decision1)
    req2 = FIUReIDBrancher.evaluate(decision2)
    assert req1.model_dump() == req2.model_dump()

def test_reid_branch_synthetic_demo_evidence():
    # Test 9 — Synthetic/demo evidence
    # Demo evidence should map to UNKNOWN or NON_AUTHORITATIVE. 
    # Let's create an UNKNOWN decision because demo evidence cannot independently establish authoritative attribution.
    decision = create_mock_decision(AttributionConfidence.UNKNOWN)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.branch_decision == ReIDBranchDecision.REID_NOT_JUSTIFIED
    assert req.status == ReIDStatus.REQUEST_PREPARED
    # Ensure it doesn't masquerade as a submitted or received FIU response
    assert req.status != ReIDStatus.REQUEST_SUBMITTED
    assert req.status != ReIDStatus.RESPONSE_RECEIVED

def test_reid_branch_request_status():
    # Test 10 — Request status
    decision = create_mock_decision(AttributionConfidence.HIGH_CONFIDENCE)
    req = FIUReIDBrancher.evaluate(decision)
    assert req.status == ReIDStatus.REQUEST_PREPARED
    # It must never claim SUBMITTED or RESPONSE_RECEIVED out of the box
    assert req.status != ReIDStatus.REQUEST_SUBMITTED
    assert req.status != ReIDStatus.RESPONSE_RECEIVED

def test_reid_branch_missing_decision():
    # Test 15 Invalid Inputs - safely handles missing decision
    with pytest.raises(ValueError):
        FIUReIDBrancher.evaluate(None)
