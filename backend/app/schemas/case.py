"""Canonical CHAKRA case domain and intake schemas.

Defines the internal representation of a case ingested into CHAKRA.
Both manual/internal intake and third-party intake stubs (e.g. NCRP)
delegate into this canonical representation.
"""
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.chain import Chain


class CaseIntakeRequest(BaseModel):
    """Canonical internal case intake request in CHAKRA.

    Represents the authoritative payload required to register a case
    in CHAKRA's investigation workflow.
    """

    source: str = Field(..., min_length=1, description="Source system or intake channel (e.g. 'ncrp', 'manual', 'api')")
    external_reference_id: Optional[str] = Field(
        None,
        min_length=1,
        description="External reference or complaint ID from the source system (e.g. NCRP complaint_id)"
    )
    wallet_address: str = Field(..., min_length=1, description="Suspect or victim target wallet address")
    chain: Chain = Field(..., description="Canonical blockchain identifier")
    reported_by: str = Field(..., min_length=1, description="Reporting party or investigator identity")
    narrative: str = Field(..., min_length=1, description="Incident narrative or complaint details")

    @field_validator("source", "wallet_address", "reported_by", "narrative", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Value must be a valid string")
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Field cannot be empty or contain only whitespace")
        return v_stripped

    @field_validator("external_reference_id", mode="before")
    @classmethod
    def validate_optional_reference(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("External reference ID must be a string")
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("External reference ID cannot be whitespace-only if provided")
        return v_stripped

    model_config = {"frozen": True}


class CaseIntakeResponse(BaseModel):
    """Canonical response for registered cases in CHAKRA."""

    case_id: str = Field(..., min_length=1, description="CHAKRA-internal unique case identifier (e.g. case_...)")
    status: str = Field("registered", description="Internal intake status")
    wallet_address: str
    chain: Chain
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the case was registered in CHAKRA"
    )

    model_config = {"frozen": True}
