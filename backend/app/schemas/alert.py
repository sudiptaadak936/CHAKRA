"""Schemas for Step 4.5A LEA Alerting Layer."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.chain import Chain


class AlertType(str, Enum):
    """Supported types of deterministic alerts."""
    SANCTIONED_ADDRESS_HIT = "SANCTIONED_ADDRESS_HIT"


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

