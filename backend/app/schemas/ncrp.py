"""Schemas for the NCRP-style LEA Intake API (Step 1F).

CRITICAL STUB NOTICE:
This interface is a:
    stub/demo interface — models the shape of a future NCRP integration, not a live government connection

It models the request/response shape of a law enforcement agency intake platform
such as NCRP, but does NOT connect to any live government systems, does NOT perform
government-side authentication, and does NOT fabricate government-side status.
"""
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator

from app.schemas.chain import Chain

NCRP_STUB_NOTICE: str = (
    "stub/demo interface — models the shape of a future NCRP integration, not a live government connection"
)


class NCRPIntakeRequest(BaseModel):
    """NCRP intake complaint payload.

    Authoritative fields required by NCRP intake:
    - complaint_id: External NCRP complaint reference number
    - wallet_address: Reported crypto wallet address
    - chain: Target blockchain (e.g. tron, evm, bitcoin, solana)
    - reported_by: Reporting officer, agency, or citizen
    - narrative: Description of fraudulent transaction or crime details
    """

    complaint_id: str = Field(..., description="External NCRP complaint tracking identifier")
    wallet_address: str = Field(..., description="Suspect or victim wallet address")
    chain: Chain = Field(..., description="Blockchain identifier (tron, evm, bitcoin, solana)")
    reported_by: str = Field(..., description="Reporting agency, officer, or victim")
    narrative: str = Field(..., description="Factual narrative of the incident")

    @field_validator("complaint_id", "wallet_address", "reported_by", "narrative", mode="before")
    @classmethod
    def validate_non_empty_string(cls, v: Any, info) -> str:
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"{info.field_name} cannot be empty or whitespace-only")
        return v_stripped

    @field_validator("chain", mode="before")
    @classmethod
    def normalize_chain(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_norm = v.strip().lower()
            return v_norm
        return v

    model_config = {"frozen": True, "extra": "forbid"}


class NCRPIntakeResponse(BaseModel):
    """Response returned by the NCRP intake endpoint.

    Distinguishes external complaint_id from CHAKRA-internal case_id.
    Explicitly bears the stub/demo disclaimer.
    """

    complaint_id: str = Field(..., description="The external NCRP complaint ID provided in the request")
    case_id: str = Field(..., description="CHAKRA internal case identifier (NOT a government case number)")
    status: str = Field("registered", description="CHAKRA internal intake status")
    wallet_address: str = Field(..., description="Reported wallet address")
    chain: Chain = Field(..., description="Canonical blockchain identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="CHAKRA ingestion timestamp"
    )
    is_stub: bool = Field(True, description="Always True; indicates this is a mock/demo interface")
    notice: str = Field(
        default=NCRP_STUB_NOTICE,
        description="Mandatory stub honesty notice"
    )

    model_config = {"frozen": True}
