"""Schemas for Step 4.5A LEA Alerting Layer."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.chain import Chain


class AlertType(str, Enum):
    """Supported types of deterministic alerts."""
    SANCTIONED_ADDRESS_HIT = "SANCTIONED_ADDRESS_HIT"
    UNREGISTERED_VASP_ATTRIBUTION = "UNREGISTERED_VASP_ATTRIBUTION"
    RISK_SCORE_THRESHOLD_EXCEEDED = "RISK_SCORE_THRESHOLD_EXCEEDED"
    SUSPECT_REGISTRY_LINK = "SUSPECT_REGISTRY_LINK"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Deterministic alert status. Never auto-escalated."""
    PENDING_REVIEW = "PENDING_REVIEW"
    REVIEWED = "REVIEWED"
    DISMISSED = "DISMISSED"


class SanctionedAddressHit(BaseModel):
    """Input contract for the sanctioned-address hit trigger."""
    chain: Chain
    address: str
    source: str
    evidence_id: str
    reason: str


class UnregisteredVASPAttribution(BaseModel):
    """
    Input contract for an unregistered VASP attribution alert.

    IMPORTANT: This alert must ONLY be generated when registration_status
    is EXPLICITLY NOT_REGISTERED. It must NEVER be generated when status
    is UNKNOWN. UNKNOWN != NOT_REGISTERED.
    """
    chain: Chain
    address: str
    attributed_vasp: str
    attribution_confidence: str  # AttributionConfidence value string
    registration_status: str    # Must be NOT_REGISTERED (validated in service)
    evidence_ids: List[str]
    source: str
    reason: str


class RiskScoreThresholdExceeded(BaseModel):
    """
    Input contract for a risk-score threshold alert.

    NOTE: This is a FORWARD CONTRACT for Step 5 Risk Scoring.
    The risk_score_record field holds a dict of the RiskScoreRecord
    to avoid circular imports between alert and risk schemas.
    """
    chain: Chain
    address: str
    overall_score: float
    risk_level: str          # RiskLevel value string
    threshold_crossed: float
    risk_score_record_id: str  # deterministic_hash of the RiskScoreRecord
    evidence_ids: List[str]
    source: str
    reason: str


class SuspectRegistryLink(BaseModel):
    """
    Input contract for a suspect registry link alert.

    This is a FORWARD CONTRACT for Step 6 Suspect Registry.
    match_result must be MATCH (never UNKNOWN or NO_MATCH).
    """
    chain: Chain
    address: str
    registry_match_id: str
    match_result: str   # Must be MATCH (validated in service)
    evidence_ids: List[str]
    source: str
    reason: str


class AlertRecord(BaseModel):
    """Explicit domain model for an alert event."""
    alert_id: str
    alert_type: AlertType
    chain: Chain
    address: str
    severity: AlertSeverity
    reason: str
    evidence_ids: List[str]
    source: str
    created_at: datetime
    requires_human_review: bool = Field(default=True, frozen=True)
    status: AlertStatus = AlertStatus.PENDING_REVIEW

    model_config = {"frozen": True}
