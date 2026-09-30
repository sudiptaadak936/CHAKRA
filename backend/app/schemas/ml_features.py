"""CHAKRA Step 5A: ML Feature Contract & Schemas.

Defines the immutable feature models, schema version, canonical feature ordering,
and availability metadata for machine learning feature extraction.

Strictly analytical and read-only.
Preserves exact monetary representations (Decimal/int).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Initial canonical feature schema version
FEATURE_SCHEMA_VERSION: str = "1.1.0"

# Ordered canonical list of available model features (13 features)
CANONICAL_FEATURE_NAMES: Tuple[str, ...] = (
    "in_degree",
    "out_degree",
    "total_tx_count",
    "total_received_native",
    "total_sent_native",
    "amount_retention_ratio",
    "time_active_seconds",
    "inter_hop_velocity_avg",
    "unique_counterparties",
    "typology_peel_chain_flag",
    "typology_rapid_hop_flag",
    "typology_fan_in_flag",
    "typology_fan_out_flag",
)

# Explicitly unsupported features in Step 5A
UNSUPPORTED_FEATURE_NAMES: Tuple[str, ...] = (
    "hop_distance_to_mixer",
    "cluster_size",
)


class FeatureStatus(str, Enum):
    """Availability status of a feature in the current CHAKRA environment."""
    AVAILABLE = "AVAILABLE"
    UNSUPPORTED = "UNSUPPORTED"


class FeatureAvailability(BaseModel):
    """Explicit availability declaration for an individual feature."""
    feature_name: str
    status: FeatureStatus
    reason: Optional[str] = None

    model_config = {"frozen": True}


def _safe_decimal_to_float(d: Decimal) -> float:
    """Safely convert Decimal to a finite Python float at the ML boundary.

    Guards against overflow or non-finite values.
    """
    f = float(d)
    if not math.isfinite(f):
        raise OverflowError(f"Monetary value {d} cannot be converted to finite float.")
    return f


class MLFeatureValues(BaseModel):
    """Exact, typed container for extracted ML feature values.

    Monetary values MUST remain exact (Decimal). Conversion to float
    occurs strictly at the `to_model_vector` boundary.
    """
    in_degree: int = Field(
        default=0, ge=0,
        description="Count of distinct incoming transaction-flow edges."
    )
    out_degree: int = Field(
        default=0, ge=0,
        description="Count of distinct outgoing transaction-flow edges."
    )
    total_tx_count: int = Field(
        default=0, ge=0,
        description="Count of distinct confirmed canonical transactions involving the address."
    )
    total_received_native: Decimal = Field(
        default=Decimal(0), ge=0,
        description="Exact cumulative amount received in atomic native units (satoshis/wei/sun/lamports)."
    )
    total_sent_native: Decimal = Field(
        default=Decimal(0), ge=0,
        description="Exact cumulative amount sent in atomic native units (satoshis/wei/sun/lamports)."
    )
    amount_retention_ratio: float = Field(
        default=0.0,
        description="Fraction of received funds retained: (received - sent) / received. 0.0 if received == 0."
    )
    time_active_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Elapsed seconds between max and min block timestamps. 0.0 if < 2 timestamps."
    )
    inter_hop_velocity_avg: float = Field(
        default=-1.0,
        description="Mean elapsed seconds between receipt and subsequent spend. -1.0 sentinel if unavailable."
    )
    unique_counterparties: int = Field(
        default=0, ge=0,
        description="Count of distinct counterparty addresses connected to this target."
    )
    typology_peel_chain_flag: int = Field(
        default=0, ge=0, le=1,
        description="1 if PEEL_CHAIN detected with confidence == 'observed', else 0."
    )
    typology_rapid_hop_flag: int = Field(
        default=0, ge=0, le=1,
        description="1 if RAPID_HOPPING detected with confidence == 'observed', else 0."
    )
    typology_fan_in_flag: int = Field(
        default=0, ge=0, le=1,
        description="1 if FAN_IN detected with confidence == 'observed', else 0."
    )
    typology_fan_out_flag: int = Field(
        default=0, ge=0, le=1,
        description="1 if FAN_OUT detected with confidence == 'observed', else 0."
    )

    model_config = {"frozen": True}

    def to_model_vector(self) -> List[float]:
        """Convert features into canonical ordered float vector for ML inference.

        Ordering strictly matches CANONICAL_FEATURE_NAMES.
        """
        return [
            float(self.in_degree),
            float(self.out_degree),
            float(self.total_tx_count),
            _safe_decimal_to_float(self.total_received_native),
            _safe_decimal_to_float(self.total_sent_native),
            float(self.amount_retention_ratio),
            float(self.time_active_seconds),
            float(self.inter_hop_velocity_avg),
            float(self.unique_counterparties),
            float(self.typology_peel_chain_flag),
            float(self.typology_rapid_hop_flag),
            float(self.typology_fan_in_flag),
            float(self.typology_fan_out_flag),
        ]


class MLFeatureRecord(BaseModel):
    """Immutable domain model for Step 5A ML Feature Vector.

    Captures extracted feature values, schema version, and explicit feature availability.
    """
    target_address: str = Field(..., min_length=1)
    chain: str = Field(..., min_length=1)
    network: str = Field(..., min_length=1)
    normalized_address: str = Field(..., min_length=1)
    feature_schema_version: str = Field(default=FEATURE_SCHEMA_VERSION, frozen=True)
    features: MLFeatureValues
    availability: Dict[str, FeatureAvailability] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}

    def to_model_vector(self) -> List[float]:
        """Return the canonical ordered float vector."""
        return self.features.to_model_vector()
