"""
Attribution API routes — Step 4.

Provides read-only access to attribution decisions, RE-ID branching,
and SAHYOG-style escalation packet preparation.

All endpoints are deterministic. No external calls are made.
No SAHYOG API is contacted. Human review is always required.
"""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.attribution.engine import AttributionDecisionEngine
from app.attribution.fiu_registry import FIURegistryRepository
from app.attribution.kyc import KYCPreparationService
from app.attribution.reid import FIUReIDBrancher
from app.attribution.sahyog import SAHYOGEscalationStub
from app.schemas.chain import Chain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/attribution", tags=["attribution"])

# Module-level demo registry instance (loaded once on import)
_demo_registry: Optional[FIURegistryRepository] = None


def _get_fiu_registry() -> FIURegistryRepository:
    """Lazy-load the demonstration FIU registry."""
    global _demo_registry
    if _demo_registry is None:
        _demo_registry = FIURegistryRepository()
    return _demo_registry


class AttributionRequest(BaseModel):
    """Input for an attribution evaluation."""
    address: str
    chain: str  # Chain enum value


class AttributionResponse(BaseModel):
    """Attribution decision result."""
    address: str
    chain: str
    confidence: str
    vasp_name: Optional[str]
    explanation: str
    branch_decision: str
    registration_status: str
    kyc_status: Optional[str] = None
    kyc_path: Optional[str] = None
    escalation_status: Optional[str] = None
    recommended_action: Optional[str] = None
    data_classification: str = "DEMONSTRATION_ONLY"


@router.get("/health")
async def attribution_health() -> Dict[str, str]:
    """Attribution service health check."""
    return {
        "status": "ok",
        "service": "attribution",
        "data_classification": "DEMONSTRATION_ONLY",
    }


@router.post("/evaluate", response_model=AttributionResponse)
async def evaluate_attribution(request: AttributionRequest) -> AttributionResponse:
    """
    Evaluate VASP attribution for a blockchain address.

    Returns:
    - Attribution confidence (CONFIRMED / HIGH_CONFIDENCE / PROBABLE_INFERRED / UNKNOWN)
    - RE-ID branch decision
    - FIU registration status (REGISTERED / NOT_REGISTERED / UNKNOWN)
    - KYC pathway
    - SAHYOG escalation status

    DEMONSTRATION DATA ONLY. No real attribution or government data.
    """
    try:
        chain = Chain(request.chain.lower())
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid chain: {request.chain!r}")

    if not request.address or not request.address.strip():
        raise HTTPException(status_code=422, detail="Address is required.")

    logger.info(
        "Attribution evaluate request: address=%r chain=%s (DEMONSTRATION_ONLY)",
        request.address,
        request.chain,
    )

    try:
        # Step 1: Attribution decision
        engine = AttributionDecisionEngine()
        registry = _get_fiu_registry()
        from app.attribution.registry import VASPAttributionRegistry
        vasp_registry = VASPAttributionRegistry()
        decision = engine.evaluate(address=request.address, chain=chain, registry=vasp_registry)

        # Step 2: RE-ID branching (confidence + registration as separate decisions)
        reid_request = FIUReIDBrancher.evaluate(decision=decision, fiu_registry=registry)

        # Step 3: KYC pathway
        kyc_record = KYCPreparationService.prepare(request=reid_request, fiu_registry=registry)

        # Step 4: SAHYOG escalation
        escalation_record = SAHYOGEscalationStub.notify_sahyog(
            request=reid_request,
            recommended_action="Human investigator review required." if reid_request.branch_decision.value == "REID_REQUIRED" else None,
        )

        return AttributionResponse(
            address=decision.address,
            chain=decision.chain.value,
            confidence=decision.confidence.value,
            vasp_name=decision.vasp_name,
            explanation=decision.explanation,
            branch_decision=reid_request.branch_decision.value,
            registration_status=reid_request.registration_status,
            kyc_status=kyc_record.kyc_status.value if kyc_record else None,
            kyc_path=kyc_record.kyc_path.value if kyc_record else None,
            escalation_status=escalation_record.escalation_status.value,
            recommended_action=(
                escalation_record.packet.recommended_action
                if escalation_record.packet
                else None
            ),
            data_classification="DEMONSTRATION_ONLY",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Attribution evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Attribution evaluation failed.")


@router.get("/registry/status")
async def registry_status() -> Dict[str, Any]:
    """
    Return the demonstration FIU registry load status.
    DEMONSTRATION DATA ONLY.
    """
    registry = _get_fiu_registry()
    return {
        "data_classification": registry.data_classification,
        "status": "loaded",
        "note": "DEMONSTRATION DATA ONLY — no real FIU-IND records",
    }
