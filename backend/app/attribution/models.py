from typing import Dict, Any
from pydantic import BaseModel, Field
from app.schemas.chain import Chain

class VASPAddressRecord(BaseModel):
    """
    Domain model representing an attribution record for a specific address on a specific chain.
    """
    address: str = Field(..., min_length=1)
    chain: Chain
    vasp_name: str = Field(..., min_length=1)
    service_type: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    evidence_id: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}
