from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.chain import Chain
from app.attribution.engine_models import AttributionConfidence, AttributionProvenance

class ReIDStatus(str, Enum):
    REQUEST_PREPARED = "REQUEST_PREPARED"
    REQUEST_SUBMITTED = "REQUEST_SUBMITTED"
    RESPONSE_RECEIVED = "RESPONSE_RECEIVED"

class ReIDBranchDecision(str, Enum):
    NO_REID_REQUIRED = "NO_REID_REQUIRED"
    REID_REQUIRED = "REID_REQUIRED"
    REID_NOT_JUSTIFIED = "REID_NOT_JUSTIFIED"

class ReIdentificationRequest(BaseModel):
    chain: Chain
    address: str
    attributed_vasp: Optional[str]
    attribution_confidence: AttributionConfidence
    evidence_ids: List[str]
    provenance: AttributionProvenance
    reason: str
    status: ReIDStatus
    branch_decision: ReIDBranchDecision

    model_config = {"frozen": True}
