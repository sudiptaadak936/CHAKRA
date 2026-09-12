"""
Alerts API routes — Step 4.5A LEA Alerting Layer.

Read-only query endpoints for alert records.
Alert state transitions (approve/dismiss) are exposed for human investigators.
No alerts are auto-escalated. Human review is always required.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.alert import (
    AlertRecord,
    AlertSeverity,
    AlertStatus,
    AlertType,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class AlertTypeListResponse(BaseModel):
    """Lists all supported alert types."""
    alert_types: List[str]
    note: str


class AlertSummary(BaseModel):
    """Summary of a single alert record (read-only)."""
    alert_id: str
    alert_type: str
    chain: str
    address: str
    severity: str
    status: str
    reason: str
    source: str
    created_at: str
    requires_human_review: bool
    evidence_ids: List[str]


@router.get("/health")
async def alerts_health() -> Dict[str, str]:
    """Alerts service health check."""
    return {"status": "ok", "service": "alerts"}


@router.get("/types", response_model=AlertTypeListResponse)
async def list_alert_types() -> AlertTypeListResponse:
    """
    List all supported deterministic alert types.

    All alert types require human investigator review.
    No alert is auto-escalated to any government system.
    """
    return AlertTypeListResponse(
        alert_types=[t.value for t in AlertType],
        note=(
            "All alerts require human investigator review. "
            "No automated escalation to government systems. "
            "SAHYOG: NOT INTEGRATED — stub only."
        ),
    )


@router.get("/severities")
async def list_severities() -> Dict[str, Any]:
    """List all supported alert severity levels."""
    return {"severities": [s.value for s in AlertSeverity]}


@router.get("/statuses")
async def list_statuses() -> Dict[str, Any]:
    """List all supported alert status values."""
    return {
        "statuses": [s.value for s in AlertStatus],
        "note": "Transitions: PENDING_REVIEW -> REVIEWED or DISMISSED (human action only).",
    }


@router.get("/contracts")
async def alert_input_contracts() -> Dict[str, Any]:
    """
    Describe input contracts for each alert type.
    These contracts define what upstream services must provide to trigger each alert.
    """
    return {
        "SANCTIONED_ADDRESS_HIT": {
            "trigger": "OFAC SDN or equivalent list match.",
            "required_fields": ["chain", "address", "source", "evidence_id", "reason"],
            "invariants": ["address must be present", "evidence_id must be present"],
        },
        "UNREGISTERED_VASP_ATTRIBUTION": {
            "trigger": "Attribution resolves to a VASP that is explicitly NOT_REGISTERED with FIU.",
            "required_fields": [
                "chain", "address", "attributed_vasp",
                "attribution_confidence", "registration_status",
                "evidence_ids", "source", "reason",
            ],
            "invariants": [
                "registration_status MUST be NOT_REGISTERED",
                "MUST NOT fire when registration_status is UNKNOWN",
                "UNKNOWN != NOT_REGISTERED",
            ],
        },
        "RISK_SCORE_THRESHOLD_EXCEEDED": {
            "trigger": "Step 5 risk engine reports overall_score >= configured threshold.",
            "required_fields": [
                "chain", "address", "overall_score", "risk_level",
                "threshold_crossed", "risk_score_record_id",
                "evidence_ids", "source", "reason",
            ],
            "note": "FORWARD CONTRACT — Step 5 not yet implemented.",
        },
        "SUSPECT_REGISTRY_LINK": {
            "trigger": "Step 6 suspect registry returns MATCH for an address.",
            "required_fields": [
                "chain", "address", "registry_match_id", "match_result",
                "evidence_ids", "source", "reason",
            ],
            "invariants": [
                "match_result MUST be MATCH",
                "MUST NOT fire when match_result is UNKNOWN or NO_MATCH",
            ],
            "note": "FORWARD CONTRACT — Step 6 not yet implemented.",
        },
    }
