"""
KYC request models — Step 4 prototype.

No external KYC API is called. This models the local preparation of a KYC
request packet that a human investigator would use for follow-up.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.chain import Chain
from app.attribution.engine_models import AttributionConfidence
from app.attribution.reid_models import ReIDBranchDecision
from app.attribution.fiu_registry import FIURegistrationStatus


class KYCRequestStatus(str, Enum):
    """
    Status of a KYC request preparation.

    PREPARED    — Packet assembled locally; awaits human investigator action.
    NOT_REQUIRED — KYC was not triggered (e.g., CONFIRMED attribution, UNKNOWN).
    BLOCKED     — Inputs were insufficient to produce a defensible KYC packet.
    """
    PREPARED = "PREPARED"
    NOT_REQUIRED = "NOT_REQUIRED"
    BLOCKED = "BLOCKED"


class KYCRequestPath(str, Enum):
    """
    Which KYC pathway applies to this VASP.

    REGISTERED_VASP_PATH   — VASP is REGISTERED with FIU-IND (in demo data).
                             Standard KYC workflow applies.
    UNREGISTERED_VASP_PATH — VASP is explicitly NOT_REGISTERED.
                             Escalation path; human review mandatory.
    UNKNOWN_REGISTRATION   — Registration status could not be determined.
                             Missing data; do NOT treat as registered or unregistered.
    """
    REGISTERED_VASP_PATH = "REGISTERED_VASP_PATH"
    UNREGISTERED_VASP_PATH = "UNREGISTERED_VASP_PATH"
    UNKNOWN_REGISTRATION = "UNKNOWN_REGISTRATION"


class KYCRequestPacket(BaseModel):
    """
    Locally prepared KYC request packet.

    This does NOT represent a submitted KYC request to any real entity.
    It is a prototype data structure for human-investigator handoff.
    """
    chain: Chain
    address: str = Field(..., min_length=1)
    attributed_vasp: Optional[str]
    attribution_confidence: AttributionConfidence
    branch_decision: ReIDBranchDecision
    kyc_path: KYCRequestPath
    registration_status: str  # FIURegistrationStatus value
    re_id: Optional[str]
    evidence_ids: List[str]
    reason: str

    model_config = {"frozen": True}


class KYCRequestRecord(BaseModel):
    """
    Outcome of a KYC preparation evaluation.
    """
    chain: Chain
    address: str = Field(..., min_length=1)
    attributed_vasp: Optional[str]
    kyc_status: KYCRequestStatus
    kyc_path: KYCRequestPath
    packet: Optional[KYCRequestPacket] = None
    reason: str

    model_config = {"frozen": True}
