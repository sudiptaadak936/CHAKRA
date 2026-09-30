"""CHAKRA Step 5I: Address-Level ML Inference Schemas.

Defines the strongly typed domain models, schemas, and result containers for
address-level supervised ML inference consuming the frozen Step 5H.7 artifacts.

Strictly analytical and read-only.
Preserves non-punitive semantic terminology.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MLFeatureValues,
)


class InferenceStatus(str, Enum):
    """Execution status of the inference pipeline."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class MLInferenceError(Exception):
    """Base exception for all address-level ML inference failures."""
    pass


class ArtifactIntegrityError(MLInferenceError):
    """Raised when an artifact is missing, unreadable, or fails SHA-256 validation."""
    pass


class FeatureContractError(MLInferenceError):
    """Raised when an input feature vector violates Schema 1.1.0."""
    pass


class NonFiniteFeatureError(FeatureContractError):
    """Raised when a non-finite feature value is provided (outside allowed sentinel)."""
    pass


class PreprocessingError(MLInferenceError):
    """Raised when deterministic preprocessing fails."""
    pass


class ModelProvenanceRecord(BaseModel):
    """Cryptographic provenance record of the frozen artifacts used for inference."""
    freeze_identity: str = Field(
        default="STEP_5H_7_FREEZE",
        description="Frozen pipeline stage milestone.",
    )
    feature_schema_version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Canonical feature schema version.",
    )
    lr_model_hash: str = Field(
        ...,
        description="SHA-256 hash of frozen logistic_regression_final.joblib.",
    )
    xgb_model_hash: str = Field(
        ...,
        description="SHA-256 hash of frozen xgboost_final.json.",
    )
    preprocessor_hash: str = Field(
        ...,
        description="SHA-256 hash of frozen final_preprocessor_params.json.",
    )
    lr_calibrator_hash: str = Field(
        ...,
        description="SHA-256 hash of frozen lr_calibrator.joblib.",
    )
    xgb_calibrator_hash: str = Field(
        ...,
        description="SHA-256 hash of frozen xgb_calibrator.joblib.",
    )

    model_config = {"frozen": True}


class AddressMLInferenceRequest(BaseModel):
    """Input request for address-level ML inference.

    Accepts target metadata and either an `MLFeatureValues` instance or a raw
    ordered 13-feature vector.
    """
    target_address: str = Field(..., min_length=1, description="Target blockchain address.")
    chain: str = Field(..., min_length=1, description="Blockchain network family (e.g. bitcoin, evm).")
    network: str = Field(..., min_length=1, description="Network context (e.g. mainnet, testnet).")
    feature_schema_version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Must strictly match canonical schema 1.1.0.",
    )
    features: Optional[MLFeatureValues] = Field(
        default=None,
        description="Strongly-typed MLFeatureValues object.",
    )
    feature_vector: Optional[List[float]] = Field(
        default=None,
        description="Raw canonical 13-element feature vector in exact CANONICAL_FEATURE_NAMES order.",
    )

    @model_validator(mode="after")
    def validate_input_features(self) -> "AddressMLInferenceRequest":
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported feature schema '{self.feature_schema_version}'. Expected '{FEATURE_SCHEMA_VERSION}'."
            )
        if self.features is None and self.feature_vector is None:
            raise ValueError("Either 'features' or 'feature_vector' must be supplied.")
        if self.feature_vector is not None:
            if len(self.feature_vector) != len(CANONICAL_FEATURE_NAMES):
                raise ValueError(
                    f"Feature vector length {len(self.feature_vector)} != expected {len(CANONICAL_FEATURE_NAMES)}."
                )
            for i, val in enumerate(self.feature_vector):
                if val is None or not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite value '{val}' at index {i} ({CANONICAL_FEATURE_NAMES[i]})."
                    )
        return self

    def get_canonical_vector(self) -> List[float]:
        """Extract the canonical 13-element float vector."""
        if self.feature_vector is not None:
            return list(self.feature_vector)
        if self.features is not None:
            return self.features.to_model_vector()
        raise ValueError("No feature representation available.")

    model_config = {"frozen": True}


class AddressMLInferenceResult(BaseModel):
    """Authoritative, read-only result of address-level ML inference.

    Terminology is strictly non-punitive: probabilities represent calibrated
    analytical signals and reference scores, NEVER proof of wrongdoing, ownership,
    criminality, or intent.
    """
    target_address: str = Field(..., description="Evaluated blockchain address.")
    chain: str = Field(..., description="Target chain.")
    network: str = Field(..., description="Target network.")
    feature_schema_version: str = Field(
        default=FEATURE_SCHEMA_VERSION,
        description="Feature contract version.",
    )
    status: InferenceStatus = Field(
        default=InferenceStatus.SUCCESS,
        description="Status of the inference run.",
    )
    lr_raw_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw output probability from final Logistic Regression model.",
    )
    lr_calibrated_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Recalibrated reference probability via frozen LR Platt scaling.",
    )
    xgb_raw_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw output probability from final XGBoost model.",
    )
    xgb_calibrated_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Recalibrated reference probability via frozen XGB Platt scaling.",
    )
    provenance: ModelProvenanceRecord = Field(
        ...,
        description="Cryptographic artifact hash inventory.",
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of inference computation.",
    )
    disclaimer: str = Field(
        default=(
            "Calibrated reference probability for analytical research only. "
            "Does NOT constitute proof of illicit activity, criminal intent, or entity attribution."
        ),
        frozen=True,
        description="Mandatory semantic limitation disclosure.",
    )

    model_config = {"frozen": True}
