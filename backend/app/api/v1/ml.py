"""CHAKRA Step 5I: Address-Level ML Inference API Router.

POST /api/v1/ml/address/infer

CRITICAL INVARIANTS:
- Purely analytical and read-only.
- Does NOT trigger alerts.
- Does NOT assign legal ownership, culpability, or criminality.
- Does NOT mutate any canonical database record.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.forensics.address_ml_service import AddressMLInferenceService, get_address_ml_service
from app.schemas.ml_inference import (
    AddressMLInferenceRequest,
    AddressMLInferenceResult,
    ArtifactIntegrityError,
    FeatureContractError,
    MLInferenceError,
)

router = APIRouter(prefix="/ml/address", tags=["ML Inference"])


@router.post(
    "/infer",
    response_model=AddressMLInferenceResult,
    summary="Address-Level ML Probability Inference",
    description=(
        "Consumes frozen Step 5H.7 ML artifacts to compute calibrated reference probabilities "
        "for an address feature vector. Purely analytical; does NOT create alerts or assign criminality."
    ),
)
async def infer_address_ml(
    request: AddressMLInferenceRequest,
    service: AddressMLInferenceService = Depends(get_address_ml_service),
) -> AddressMLInferenceResult:
    """Execute address-level model inference and Platt sigmoid recalibration."""
    try:
        return service.infer(request)
    except FeatureContractError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature contract violation: {str(e)}",
        )
    except ArtifactIntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frozen artifact integrity failure: {str(e)}",
        )
    except MLInferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(e)}",
        )
