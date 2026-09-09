"""Schemas for Step 5 Risk Scoring."""
from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Conceptual bands for risk score."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class RiskComponentContribution(BaseModel):
    """A breakdown component of a risk score."""
    component_name: str
    score_contribution: float
    evidence_ids: List[str]
    detection_ids: List[str]
    reason: str


class RiskScoreRecord(BaseModel):
    """The final deterministic output of risk scoring."""
    target_address: str
    chain: str
    network: str
    overall_score: float
    risk_level: RiskLevel
    components: List[RiskComponentContribution]
    computed_at: datetime
    requires_human_review: bool = Field(default=True, frozen=True)
    deterministic_hash: str

    model_config = {"frozen": True}

