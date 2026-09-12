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
    # Registration status is a SEPARATE decision from branch_decision.
    # UNKNOWN means the VASP was not found in the demo registry.
    # UNKNOWN must NEVER be silently treated as NOT_REGISTERED.
    # Import is deferred via string annotation to avoid circular imports.
    registration_status: str = "UNKNOWN"

    model_config = {"frozen": True}

