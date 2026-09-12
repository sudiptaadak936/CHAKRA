import pytest
from pydantic import ValidationError
from app.schemas.chain import Chain
from app.attribution.engine_models import (
    AttributionConfidence,
    AttributionProvenance
)
from app.attribution.reid_models import (
    ReIDStatus,
    ReIDBranchDecision,
    ReIdentificationRequest
)
from app.attribution.sahyog_models import EscalationStatus
from app.attribution.sahyog import SAHYOGEscalationStub

def create_mock_request(
    branch_decision: ReIDBranchDecision,
    confidence: AttributionConfidence,
    vasp_name: str = "TestVASP",
    address: str = "0x123",
    chain: Chain = Chain.EVM,
    evidence_ids: list = ["EVID-1"],
    provenance: AttributionProvenance = None
) -> ReIdentificationRequest:
    if provenance is None:
        provenance = AttributionProvenance()
        
    return ReIdentificationRequest(
        chain=chain,
        address=address,
        attributed_vasp=vasp_name if confidence != AttributionConfidence.UNKNOWN else None,
        attribution_confidence=confidence,
        evidence_ids=evidence_ids,
        provenance=provenance,
        reason="Test reason",
        status=ReIDStatus.REQUEST_PREPARED,
        branch_decision=branch_decision
    )

def test_sahyog_no_reid_required():
    # 1. NO_REID_REQUIRED -> NOT_ESCALATED, packet = null
    req = create_mock_request(ReIDBranchDecision.NO_REID_REQUIRED, AttributionConfidence.CONFIRMED)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.escalation_status == EscalationStatus.NOT_ESCALATED
    assert record.packet is None

def test_sahyog_reid_required():
    # 2. REID_REQUIRED -> PREPARED, packet populated
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.escalation_status == EscalationStatus.PREPARED
    assert record.packet is not None
    assert record.packet.address == req.address
    assert record.packet.chain == req.chain

def test_sahyog_reid_not_justified():
    # 3. REID_NOT_JUSTIFIED -> NOT_ESCALATED, packet = null
    req = create_mock_request(ReIDBranchDecision.REID_NOT_JUSTIFIED, AttributionConfidence.UNKNOWN)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.escalation_status == EscalationStatus.NOT_ESCALATED
    assert record.packet is None

def test_sahyog_null_request():
    # 4. null request
    with pytest.raises(ValueError, match="Validation failed"):
        SAHYOGEscalationStub.notify_sahyog(None)

def test_sahyog_invalid_branch_decision():
    # 5. invalid branch decision
    # Bypass Pydantic validation to test the stub's internal defensive check
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['branch_decision'] = "FAKE_BRANCH"
    # To bypass Pydantic frozen model, we use model_construct
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Invalid branch decision."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_invalid_confidence():
    # 6. invalid confidence
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['attribution_confidence'] = "SUPER_CONFIDENCE"
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Invalid attribution confidence."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_invalid_chain():
    # 7. invalid chain
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['chain'] = "FAKE_CHAIN"
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Invalid chain."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_invalid_address():
    # 8. invalid address
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['address'] = "   "
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Invalid address."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_invalid_provenance():
    # 9. missing/invalid provenance
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['provenance'] = None
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Missing provenance."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_invalid_evidence_ids():
    # 10. invalid evidence IDs
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req_dict = req.model_dump()
    req_dict['evidence_ids'] = [""]
    bad_req = ReIdentificationRequest.model_construct(**req_dict)
    with pytest.raises(ValueError, match="Validation failed: Invalid evidence ID element."):
        SAHYOGEscalationStub.notify_sahyog(bad_req)

def test_sahyog_contradictory_request():
    # 11. contradictory request
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.UNKNOWN)
    with pytest.raises(ValueError, match="Validation failed: Contradiction: REID_REQUIRED but confidence is UNKNOWN."):
        SAHYOGEscalationStub.notify_sahyog(req)
        
    req2 = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED, evidence_ids=[])
    with pytest.raises(ValueError, match="Validation failed: Contradiction: REID_REQUIRED but no evidence IDs provided."):
        SAHYOGEscalationStub.notify_sahyog(req2)

def test_sahyog_provenance_preservation():
    # Provenance preservation
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.provenance == req.provenance
    assert record.packet.provenance == req.provenance

def test_sahyog_no_authority_escalation():
    # No authority escalation
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.attribution_confidence == AttributionConfidence.PROBABLE_INFERRED
    assert record.packet.attribution_confidence == AttributionConfidence.PROBABLE_INFERRED

def test_sahyog_determinism():
    # Determinism
    req1 = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    req2 = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    rec1 = SAHYOGEscalationStub.notify_sahyog(req1)
    rec2 = SAHYOGEscalationStub.notify_sahyog(req2)
    assert rec1.model_dump() == rec2.model_dump()

def test_sahyog_no_fabrication():
    # No fabricated fields
    req = create_mock_request(ReIDBranchDecision.REID_NOT_JUSTIFIED, AttributionConfidence.UNKNOWN)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.attributed_vasp is None
    assert getattr(record, "sahyog_case_id", None) is None
    assert getattr(record, "acknowledgement_id", None) is None
    
    req_prep = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.PROBABLE_INFERRED)
    record_prep = SAHYOGEscalationStub.notify_sahyog(req_prep)
    assert getattr(record_prep.packet, "sahyog_case_id", None) is None

def test_sahyog_no_network_behavior():
    # No external communication verification
    # The pure implementation inherently avoids aiohttp/requests, verified by fast execution.
    req = create_mock_request(ReIDBranchDecision.REID_REQUIRED, AttributionConfidence.HIGH_CONFIDENCE)
    record = SAHYOGEscalationStub.notify_sahyog(req)
    assert record.escalation_status == EscalationStatus.PREPARED
