from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.chain import Chain
from app.attribution.engine_models import AttributionConfidence, AttributionProvenance
from app.attribution.reid_models import ReIDBranchDecision

class EscalationStatus(str, Enum):
    """
    Status of the SAHYOG-style escalation.
    Explicitly modeling local preparation only. No live submissions.
    """
    PREPARED = "PREPARED"
    NOT_ESCALATED = "NOT_ESCALATED"

class SahyogEscalationPacket(BaseModel):
    """
    Prepared SAHYOG-style escalation packet data.
    This is a local preparation model. It does not imply external submission.
    No government case IDs or responses are modeled here.
    """
    chain: Chain
    address: str = Field(..., min_length=1)
    attributed_vasp: Optional[str]
    attribution_confidence: AttributionConfidence
    branch_decision: ReIDBranchDecision
    evidence_ids: List[str]
    provenance: AttributionProvenance
    reason: str
    # Advisory recommended action for human investigator review.
    # This does NOT trigger any automated escalation or government submission.
    recommended_action: Optional[str] = None

    model_config = {"frozen": True}

class EscalationRecord(BaseModel):
    """
    The outcome of an escalation evaluation.
    """
    chain: Chain
    address: str = Field(..., min_length=1)
    attributed_vasp: Optional[str]
    attribution_confidence: AttributionConfidence
    branch_decision: ReIDBranchDecision
    evidence_ids: List[str]
    provenance: AttributionProvenance
    reason: str
    escalation_status: EscalationStatus
    packet: Optional[SahyogEscalationPacket] = None

    model_config = {"frozen": True}
