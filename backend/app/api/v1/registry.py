"""CHAKRA Step 6: Suspect Registry API routes."""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.schemas.chain import Chain
from app.attribution.suspect_registry import SuspectRegistryService
from app.core.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/registry", tags=["registry"])

def get_registry_service() -> SuspectRegistryService:
    if not db_manager.pg_pool:
        raise HTTPException(status_code=503, detail="Database pool not initialized")
    return SuspectRegistryService(pool=db_manager.pg_pool)

class RegistryAddRequest(BaseModel):
    address: str
    chain: str
    case_id: str
    risk_score: float
    fingerprint: List[float]

class SimilaritySearchRequest(BaseModel):
    fingerprint: List[float]
    limit: int = 5
    threshold: float = 0.8

@router.get("/lookup")
async def lookup_suspect(address: str, chain: str, service: SuspectRegistryService = Depends(get_registry_service)):
    """Exact match lookup for an address."""
    if not address or not chain:
        raise HTTPException(status_code=400, detail="address and chain required")
    try:
        res = await service.lookup(address, chain)
        return {
            "address": res.address,
            "chain": res.chain,
            "match_result": res.match_result,
            "match_id": res.match_id,
            "reason": res.reason
        }
    except Exception as e:
        logger.exception("Lookup failed")
        raise HTTPException(status_code=500, detail="Lookup failed")

@router.post("/add")
async def add_suspect(req: RegistryAddRequest, service: SuspectRegistryService = Depends(get_registry_service)):
    """Add a suspect and their graph fingerprint."""
    if len(req.fingerprint) != 6:
        raise HTTPException(status_code=400, detail="Fingerprint must be 6-dimensional")
        
    try:
        registry_id = await service.add_to_registry(
            req.address, req.chain, req.case_id, req.risk_score, req.fingerprint
        )
        if registry_id:
            return {"status": "success", "registry_id": registry_id}
        raise HTTPException(status_code=500, detail="Failed to insert")
    except Exception as e:
        logger.exception("Add failed")
        raise HTTPException(status_code=500, detail="Add failed")

@router.post("/similar")
async def find_similar(req: SimilaritySearchRequest, service: SuspectRegistryService = Depends(get_registry_service)):
    """Find structurally similar malicious addresses."""
    if len(req.fingerprint) != 6:
        raise HTTPException(status_code=400, detail="Fingerprint must be 6-dimensional")
    try:
        matches = await service.find_similar_cases(req.fingerprint, req.limit, req.threshold)
        return {"matches": matches}
    except Exception as e:
        logger.exception("Similarity search failed")
        raise HTTPException(status_code=500, detail="Similarity search failed")
