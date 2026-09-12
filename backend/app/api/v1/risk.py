"""CHAKRA Step 5: Risk API routes."""
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.schemas.chain import Chain
from app.schemas.alert import SanctionedAddressHit
from app.forensics.typology_detector import TypologyDetection
from app.schemas.risk_scoring import RiskAssessmentResult
from app.core.database import db_manager
from app.risk.service import RiskService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/risk", tags=["risk"])

class RiskScoreRequest(BaseModel):
    """Input for Step 5 Risk Scoring."""
    target_address: str
    chain: str
    network: str = "mainnet"
    # Optional context from upstream
    sanctions_hits: List[SanctionedAddressHit] = []
    typologies: List[TypologyDetection] = []

def get_risk_service() -> RiskService:
    if not db_manager.pg_pool:
        raise HTTPException(status_code=503, detail="Database pool not initialized")
    # Base path for models relative to the backend root where FastAPI runs
    return RiskService(pool=db_manager.pg_pool, model_dir="models/")

@router.get("/health")
async def risk_health() -> Dict[str, str]:
    """Risk service health check."""
    return {"status": "ok", "service": "risk"}

@router.post("/score", response_model=RiskAssessmentResult)
async def score_risk(request: RiskScoreRequest, service: RiskService = Depends(get_risk_service)) -> RiskAssessmentResult:
    """Evaluate end-to-end risk for an address."""
    try:
        chain_enum = Chain(request.chain.lower())
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid chain: {request.chain}")

    if not request.target_address:
        raise HTTPException(status_code=422, detail="Target address is required")

    try:
        result = await service.evaluate_risk(
            target_address=request.target_address,
            chain=chain_enum.value,
            network=request.network,
            sanctions_hits=request.sanctions_hits,
            typologies=request.typologies
        )
        return result
    except Exception as e:
        logger.exception("Risk evaluation failed")
        raise HTTPException(status_code=500, detail="Risk evaluation failed")
