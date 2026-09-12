from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.chain import Chain
from app.attribution.models import VASPAddressRecord
from app.attribution.authority import AuthorityClass

class AttributionConfidence(str, Enum):
    CONFIRMED = "CONFIRMED"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    PROBABLE_INFERRED = "PROBABLE_INFERRED"
    UNKNOWN = "UNKNOWN"

class AttributionProvenance(BaseModel):
    """Provenance tracking for attribution decisions."""
    direct_observations: List[VASPAddressRecord] = Field(default_factory=list)
    inferred_observations: List[VASPAddressRecord] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    corroborating_evidence_count: int = 0
    authority_class: Optional[AuthorityClass] = None

    model_config = {"frozen": True}

class AttributionDecision(BaseModel):
    """Result of an attribution evaluation."""
    address: str
    chain: Chain
    confidence: AttributionConfidence
    vasp_name: Optional[str] = None
    service_type: Optional[str] = None
    candidate_attributions: List[VASPAddressRecord] = Field(default_factory=list)
    provenance: AttributionProvenance
    explanation: str

    model_config = {"frozen": True}
