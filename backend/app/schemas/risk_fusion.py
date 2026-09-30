"""CHAKRA Step 5J: Controlled Risk & ML Fusion Schemas.

Defines the composite schemas presenting:
1. Deterministic evidence-backed risk scoring (Step 5), and
2. Supervised address-level ML analytical probabilities (Step 5I).

CRITICAL INVARIANTS:
- Deterministic and ML channels remain strictly separated.
- No arbitrary mathematical weighting or composite score synthesis without an
  explicit, frozen fusion policy.
- When fusion policy is unconfigured, fusion_status must explicitly report NOT_CONFIGURED.
- Zero alert generation, zero thresholding, zero culpability assertions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.risk import RiskScoreRecord
from app.schemas.ml_inference import AddressMLInferenceResult


class FusionStatus(str, Enum):
    """Configuration and execution status of the mathematical fusion layer."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"
    FAILED = "FAILED"


class FusionLayerResult(BaseModel):
    """Detailed metadata for the mathematical fusion layer.

    If no formal frozen fusion policy exists, fusion_status is NOT_CONFIGURED,
    and composite scoring fields remain None.
    """
    fusion_status: FusionStatus = Field(
        default=FusionStatus.NOT_CONFIGURED,
        description="Indicates whether a validated mathematical fusion policy was executed.",
    )
    fusion_policy_version: Optional[str] = Field(
        default=None,
        description="Version string of the authorized fusion policy, if configured.",
    )
    fusion_score: Optional[float] = Field(
        default=None,
        description="Composite score if mathematically synthesized; None if unconfigured.",
    )
    fusion_methodology: Optional[str] = Field(
        default=None,
        description="Formal mathematical formula or decision logic executed.",
    )
    reason: str = Field(
        default=(
            "Mathematical fusion policy not specified in frozen repository contracts; "
            "deterministic risk evidence and ML analytical signals are presented independently."
        ),
        description="Explanation of fusion state.",
    )

    model_config = {"frozen": True}


class RiskMLFusionRecord(BaseModel):
    """The authoritative composite analytical result for a target address.

    Preserves the strict boundary between:
    - Deterministic risk policy (Step 5)
    - Calibrated ML probabilities (Step 5I)
    - Mathematical fusion state (Step 5J)
    """
    target_address: str = Field(..., description="Evaluated blockchain address.")
    chain: str = Field(..., description="Target chain.")
    network: str = Field(..., description="Target network.")

    deterministic_layer: RiskScoreRecord = Field(
        ...,
        description="Deterministic risk score and component evidence from Step 5.",
    )
    ml_layer: AddressMLInferenceResult = Field(
        ...,
        description="Calibrated model probabilities and artifact provenance from Step 5I.",
    )
    fusion_layer: FusionLayerResult = Field(
        default_factory=FusionLayerResult,
        description="Status of mathematical integration.",
    )

    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of composite analysis.",
    )
    disclaimer: str = Field(
        default=(
            "Composite forensic analytical result. Deterministic risk evidence is policy-bounded; "
            "ML probabilities are calibrated analytical signals only. Does NOT establish criminal "
            "intent, legal ownership, or liability."
        ),
        frozen=True,
        description="Mandatory semantic limitation disclosure.",
    )

    model_config = {"frozen": True}
