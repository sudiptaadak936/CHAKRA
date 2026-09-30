"""CHAKRA Step 5I: Frozen Address ML Preprocessor.

Applies the authoritative frozen preprocessing parameters (derived strictly from
development data in Step 5H.6.3 and locked in Step 5H.7) to inference feature vectors.

Guarantees:
- ZERO runtime fitting or parameter recomputation.
- Strict input validation against Schema 1.1.0 (13 canonical dimensions).
- Sentinel handling:
    - LR: Sentinel (-1.0) velocity is imputed using the frozen development median (24253.384615)
      and an explicit binary missingness indicator column is appended (14 output dimensions).
    - XGB: Sentinel (-1.0) velocity is mapped to np.nan (13 output dimensions).
- Monetary compression: log1p applied to native received and sent totals.
- Standard scaling using frozen development means and stds.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from app.schemas.ml_inference import (
    FeatureContractError,
    NonFiniteFeatureError,
    PreprocessingError,
)

IDX_IN_DEGREE = 0
IDX_OUT_DEGREE = 1
IDX_TOTAL_TX_COUNT = 2
IDX_TOTAL_RECEIVED_NATIVE = 3
IDX_TOTAL_SENT_NATIVE = 4
IDX_AMOUNT_RETENTION_RATIO = 5
IDX_TIME_ACTIVE_SECONDS = 6
IDX_INTER_HOP_VELOCITY_AVG = 7
IDX_UNIQUE_COUNTERPARTIES = 8
IDX_TYPOLOGY_PEEL_CHAIN = 9
IDX_TYPOLOGY_RAPID_HOP = 10
IDX_TYPOLOGY_FAN_IN = 11
IDX_TYPOLOGY_FAN_OUT = 12

SENTINEL_VELOCITY = -1.0


class FrozenAddressPreprocessor:
    """Inference preprocessor using immutable parameters locked in Step 5H.7."""

    def __init__(self, params: Dict[str, Any]):
        """Initialize with frozen parameter dictionary from final_preprocessor_params.json."""
        if not params.get("is_fitted", False):
            raise PreprocessingError("Preprocessor parameter dictionary indicates is_fitted=False.")

        schema_ver = params.get("schema_version")
        if schema_ver != FEATURE_SCHEMA_VERSION:
            raise PreprocessingError(
                f"Preprocessor schema version '{schema_ver}' != expected '{FEATURE_SCHEMA_VERSION}'"
            )

        self.velocity_median_train: float = float(params["velocity_median_train"])
        self.means: np.ndarray = np.array(params["means"], dtype=np.float64)
        self.stds: np.ndarray = np.array(params["stds"], dtype=np.float64)
        self.zero_variance_indices: List[int] = list(params.get("zero_variance_indices", []))

        if len(self.means) != len(CANONICAL_FEATURE_NAMES) or len(self.stds) != len(CANONICAL_FEATURE_NAMES):
            raise PreprocessingError(
                f"Preprocessor mean/std dimension ({len(self.means)}/{len(self.stds)}) "
                f"!= canonical feature count ({len(CANONICAL_FEATURE_NAMES)})."
            )

    def validate_feature_vector(self, vector: Sequence[float]) -> np.ndarray:
        """Validate feature vector conformity to Schema 1.1.0."""
        if len(vector) != len(CANONICAL_FEATURE_NAMES):
            raise FeatureContractError(
                f"Input vector length {len(vector)} != expected {len(CANONICAL_FEATURE_NAMES)}."
            )

        for i, val in enumerate(vector):
            if val is None or not math.isfinite(val):
                raise NonFiniteFeatureError(
                    f"Non-finite value '{val}' at feature {i} ({CANONICAL_FEATURE_NAMES[i]})."
                )

        return np.array(vector, dtype=np.float64).reshape(1, -1)

    def transform_for_linear(self, vector: Sequence[float]) -> np.ndarray:
        """Transform 13-feature vector for Logistic Regression (14-dim output).
        
        Steps:
        1. Check missingness indicator for inter_hop_velocity_avg == -1.0.
        2. Impute -1.0 sentinel with frozen development median (24253.384615).
        3. Log1p on native received and sent totals.
        4. Standardize using frozen development means and stds.
        5. Column-stack standardized vector with the missingness indicator.
        """
        X = self.validate_feature_vector(vector)

        # 1. Missingness indicator
        is_missing_velocity = (X[:, IDX_INTER_HOP_VELOCITY_AVG] == SENTINEL_VELOCITY).astype(np.float64)

        # 2. Impute sentinel with frozen median
        X_clean = X.copy()
        vel = X_clean[:, IDX_INTER_HOP_VELOCITY_AVG]
        vel[vel == SENTINEL_VELOCITY] = self.velocity_median_train
        X_clean[:, IDX_INTER_HOP_VELOCITY_AVG] = vel

        # 3. Log1p on native amounts
        X_clean[:, IDX_TOTAL_RECEIVED_NATIVE] = np.log1p(np.maximum(0.0, X_clean[:, IDX_TOTAL_RECEIVED_NATIVE]))
        X_clean[:, IDX_TOTAL_SENT_NATIVE] = np.log1p(np.maximum(0.0, X_clean[:, IDX_TOTAL_SENT_NATIVE]))

        # 4. Standard scale using frozen development parameters
        X_scaled = (X_clean - self.means) / self.stds

        # 5. Append missingness indicator (14-dimensional vector)
        X_out = np.column_stack([X_scaled, is_missing_velocity])
        return X_out

    def transform_for_trees(self, vector: Sequence[float]) -> np.ndarray:
        """Transform 13-feature vector for XGBoost (13-dim output).
        
        Steps:
        1. Log1p on native received and sent totals.
        2. Map inter_hop_velocity_avg == -1.0 sentinel to np.nan for native tree missing handling.
        """
        X = self.validate_feature_vector(vector).copy()

        # 1. Log1p on native amounts
        X[:, IDX_TOTAL_RECEIVED_NATIVE] = np.log1p(np.maximum(0.0, X[:, IDX_TOTAL_RECEIVED_NATIVE]))
        X[:, IDX_TOTAL_SENT_NATIVE] = np.log1p(np.maximum(0.0, X[:, IDX_TOTAL_SENT_NATIVE]))

        # 2. Sentinel -> NaN
        vel = X[:, IDX_INTER_HOP_VELOCITY_AVG]
        vel[vel == SENTINEL_VELOCITY] = np.nan
        X[:, IDX_INTER_HOP_VELOCITY_AVG] = vel

        return X
