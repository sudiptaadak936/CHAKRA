"""Intake router providing the NCRP LEA intake stub endpoint (Step 1F).

POST /intake/ncrp

CRITICAL NOTICE:
stub/demo interface — models the shape of a future NCRP integration, not a live government connection
"""
from fastapi import APIRouter, Depends

from app.schemas.ncrp import NCRPIntakeRequest, NCRPIntakeResponse, NCRP_STUB_NOTICE
from app.services.case_service import CaseIntakeService, get_case_service
from app.services.ncrp_adapter import NCRPAdapter

router = APIRouter(prefix="/intake", tags=["Intake"])


@router.post(
    "/ncrp",
    response_model=NCRPIntakeResponse,
    summary="NCRP LEA Intake API (Stub)",
    description=(
        f"**NOTICE**: {NCRP_STUB_NOTICE}\n\n"
        "Accepts NCRP-style law enforcement complaint data, validates fields, "
        "and delegates into CHAKRA's canonical case intake workflow."
    ),
)
async def intake_ncrp(
    request: NCRPIntakeRequest,
    case_service: CaseIntakeService = Depends(get_case_service),
) -> NCRPIntakeResponse:
    """NCRP-style law enforcement agency intake endpoint (stub).

    Transforms the incoming NCRP request into the canonical case intake representation
    using NCRPAdapter, then delegates directly to CaseIntakeService.
    """
    canonical_request = NCRPAdapter.to_canonical(request)
    canonical_response = case_service.submit_case(canonical_request)

    return NCRPIntakeResponse(
        complaint_id=request.complaint_id,
        case_id=canonical_response.case_id,
        status=canonical_response.status,
        wallet_address=canonical_response.wallet_address,
        chain=canonical_response.chain,
        created_at=canonical_response.created_at,
        is_stub=True,
        notice=NCRP_STUB_NOTICE,
    )
