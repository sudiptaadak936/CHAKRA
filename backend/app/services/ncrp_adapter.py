"""NCRP to canonical CaseIntake adapter.

Pure boundary transformation:
NCRPIntakeRequest -> CaseIntakeRequest

Architectural boundaries:
- Pure data transformation
- NO graph logic
- NO persistence logic
- NO blockchain provider calls
- NO traversal or scoring
- NO clustering or typology logic
"""
from app.schemas.case import CaseIntakeRequest
from app.schemas.ncrp import NCRPIntakeRequest


class NCRPAdapter:
    """Thin adapter transforming NCRP requests to canonical CHAKRA case intake requests."""

    @staticmethod
    def to_canonical(request: NCRPIntakeRequest) -> CaseIntakeRequest:
        """Transform an NCRP complaint payload into the canonical CaseIntakeRequest."""
        return CaseIntakeRequest(
            source="ncrp",
            external_reference_id=request.complaint_id,
            wallet_address=request.wallet_address,
            chain=request.chain,
            reported_by=request.reported_by,
            narrative=request.narrative,
        )
