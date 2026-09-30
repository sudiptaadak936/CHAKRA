"""CHAKRA Step 5H.5.1: Address-Level Model Contract and Preprocessing Specification.

Defines the mathematical, architectural, and preprocessing contracts for new
address-level supervised models prior to training (Step 5H.5.2).

Absolute Guarantees:
- NO model fitting, training, or weight optimization occurs in this module.
- Input feature vector strictly adheres to CANONICAL_FEATURE_NAMES (13 dimensions, Schema 1.1.0).
- Entity-grouped partitioning guarantees zero entity leakage across partitions.
- All preprocessing statistics (scaling, centering, imputation medians, class weights)
  are derived strictly from TRAINING data only (zero test/val leakage).
- Legacy 1.0.0 EVM records and invalidated provisional v3 records are rejected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Constants & Contracts
# ---------------------------------------------------------------------------

CANONICAL_FEATURE_COUNT = 13

# Indices matching CANONICAL_FEATURE_NAMES
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


# ---------------------------------------------------------------------------
# Record Validation & Lineage Guards
# ---------------------------------------------------------------------------

def is_eligible_training_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Determine whether an address record is eligible for 5H.5 candidate pools.
    
    Guards:
    - Banned: EVM records with schema != '1.1.0' (e.g. legacy 1.0.0 EVM).
    - Banned: Provisional v3 records.
    - Banned: Records without valid binary label (label not in {0, 1}).
    - Banned: Records without exactly 13 canonical features.
    - Allowed: Bitcoin records from legacy validated population.
    - Allowed: Schema 1.1.0 EVM records.
    """
    chain = record.get("chain", "").lower()
    schema = record.get("feature_schema_version", "")
    sample_id = record.get("sample_id", "")
    label = record.get("label")

    if label not in (0, 1):
        return False, f"Invalid or non-binary label: {label}"

    # Provisional v3 guard
    if "PROVISIONAL" in str(record.get("batch_id", "")).upper() or "provisional" in sample_id.lower():
        return False, "Invalidated provisional v3 sample rejected"

    # Schema & Chain guard
    if chain == "evm":
        if schema != FEATURE_SCHEMA_VERSION:
            return False, f"Legacy EVM schema '{schema}' rejected; requires '{FEATURE_SCHEMA_VERSION}'"
    elif chain == "bitcoin":
        # Legacy validated Bitcoin records are permitted
        pass
    else:
        return False, f"Unsupported chain '{chain}'"

    # Feature vector validation
    feat_vals = record.get("feature_values")
    if feat_vals is None or len(feat_vals) != CANONICAL_FEATURE_COUNT:
        return False, f"Feature vector length {len(feat_vals) if feat_vals else 0} != {CANONICAL_FEATURE_COUNT}"

    for v in feat_vals:
        if v is None or not math.isfinite(v):
            return False, f"Non-finite feature value encountered: {v}"

    return True, None


# ---------------------------------------------------------------------------
# Entity-Grouped Splitter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPartition:
    """A single partition of address-level dataset records."""
    name: str
    records: List[Dict[str, Any]]
    entity_ids: Set[str] = field(default_factory=set)

    def __post_init__(self):
        e_set = {r["entity_id"] for r in self.records if r.get("entity_id")}
        object.__setattr__(self, "entity_ids", e_set)

    @property
    def sample_count(self) -> int:
        return len(self.records)

    @property
    def entity_count(self) -> int:
        return len(self.entity_ids)

    @property
    def positive_sample_count(self) -> int:
        return sum(1 for r in self.records if r.get("label") == 1)

    @property
    def negative_sample_count(self) -> int:
        return sum(1 for r in self.records if r.get("label") == 0)

    @property
    def positive_entity_count(self) -> int:
        pos_entities = {r["entity_id"] for r in self.records if r.get("label") == 1 and r.get("entity_id")}
        return len(pos_entities)

    @property
    def negative_entity_count(self) -> int:
        neg_entities = {r["entity_id"] for r in self.records if r.get("label") == 0 and r.get("entity_id")}
        return len(neg_entities)


class AddressDatasetSplitter:
    """Deterministic entity-grouped dataset partitioning."""

    @staticmethod
    def split_chronological(
        records: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[DatasetPartition, DatasetPartition, DatasetPartition]:
        """Chronological entity-grouped split.
        
        Entities are ordered by their earliest temporal cutoff, then deterministically
        partitioned into Train, Validation, and Test.
        
        Guarantees:
        - 1 entity_id -> exactly 1 partition (zero entity leakage).
        - Deterministic ordering by (earliest_cutoff, entity_id).
        """
        # 1. Group records by entity_id
        entity_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            eid = r["entity_id"]
            entity_map.setdefault(eid, []).append(r)

        # 2. Determine earliest cutoff per entity
        entity_meta = []
        for eid, recs in entity_map.items():
            earliest_cutoff = min(
                r.get("observation_end") or r.get("observation_cutoff") or r.get("label_timestamp") or "9999"
                for r in recs
            )
            lbl = recs[0]["label"]
            entity_meta.append((earliest_cutoff, eid, lbl))

        entity_meta.sort(key=lambda x: (x[0], x[1]))

        # 3. Partition entity list
        n_entities = len(entity_meta)
        n_train = int(round(n_entities * train_ratio))
        n_val = int(round(n_entities * val_ratio))
        # Ensure at least 1 in val and test if possible
        if n_train + n_val >= n_entities:
            n_train = n_entities - 2
            n_val = 1

        train_cohort = entity_meta[:n_train]
        val_cohort = entity_meta[n_train:n_train + n_val]
        test_cohort = entity_meta[n_train + n_val:]

        train_recs: List[Dict[str, Any]] = []
        for _, eid, _ in train_cohort:
            train_recs.extend(entity_map[eid])

        val_recs: List[Dict[str, Any]] = []
        for _, eid, _ in val_cohort:
            val_recs.extend(entity_map[eid])

        test_recs: List[Dict[str, Any]] = []
        for _, eid, _ in test_cohort:
            test_recs.extend(entity_map[eid])

        # Leakage guard check
        s_train = {r["entity_id"] for r in train_recs}
        s_val = {r["entity_id"] for r in val_recs}
        s_test = {r["entity_id"] for r in test_recs}

        assert s_train.isdisjoint(s_val), f"Entity leakage Train-Val: {s_train & s_val}"
        assert s_train.isdisjoint(s_test), f"Entity leakage Train-Test: {s_train & s_test}"
        assert s_val.isdisjoint(s_test), f"Entity leakage Val-Test: {s_val & s_test}"

        return (
            DatasetPartition(name="train", records=train_recs),
            DatasetPartition(name="val", records=val_recs),
            DatasetPartition(name="test", records=test_recs),
        )


# ---------------------------------------------------------------------------
# Preprocessing Specification & Transformer
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorState:
    """Fitted preprocessing parameters derived strictly from Training data."""
    is_fitted: bool = False
    train_sample_count: int = 0
    zero_variance_indices: List[int] = field(default_factory=list)
    means: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    velocity_median_train: float = 0.0
    class_weights_balanced: Dict[int, float] = field(default_factory=dict)
    scale_pos_weight_xgboost: float = 1.0


class AddressModelPreprocessor:
    """Preprocesses canonical 13-feature vectors for tabular model training.
    
    Guarantees:
    - Fits scaling and statistics ONLY on provided training data.
    - Preserves canonical 13-feature ordering.
    - Deterministic transformations.
    - Supports tree-based (XGBoost) and linear (LogisticRegression) model spaces.
    """

    def __init__(self):
        self.state = PreprocessorState()

    def fit(self, train_records: List[Dict[str, Any]]) -> "AddressModelPreprocessor":
        """Fit preprocessor parameters on the Training partition only."""
        if not train_records:
            raise ValueError("Cannot fit on empty training records.")

        X_raw = np.array([r["feature_values"] for r in train_records], dtype=np.float64)
        y_train = np.array([r["label"] for r in train_records], dtype=np.int32)

        if X_raw.shape[1] != CANONICAL_FEATURE_COUNT:
            raise ValueError(f"Expected {CANONICAL_FEATURE_COUNT} features, got {X_raw.shape[1]}")

        n_samples = len(train_records)

        # 1. Detect zero-variance features on train
        variances = np.var(X_raw, axis=0)
        zero_var_idx = [i for i, v in enumerate(variances) if v == 0.0]

        # 2. Log1p-transform amounts before calculating scaling statistics
        X_interim = X_raw.copy()
        X_interim[:, IDX_TOTAL_RECEIVED_NATIVE] = np.log1p(np.maximum(0.0, X_interim[:, IDX_TOTAL_RECEIVED_NATIVE]))
        X_interim[:, IDX_TOTAL_SENT_NATIVE] = np.log1p(np.maximum(0.0, X_interim[:, IDX_TOTAL_SENT_NATIVE]))

        # 3. Sentinel velocity median calculation (excluding -1.0)
        vel_vals = X_interim[:, IDX_INTER_HOP_VELOCITY_AVG]
        valid_vel = vel_vals[vel_vals != SENTINEL_VELOCITY]
        if len(valid_vel) > 0:
            vel_med = float(np.median(valid_vel))
        else:
            vel_med = 0.0

        # Replace sentinel temporarily for interim mean/std calculation
        vel_interim = vel_vals.copy()
        vel_interim[vel_interim == SENTINEL_VELOCITY] = vel_med
        X_interim[:, IDX_INTER_HOP_VELOCITY_AVG] = vel_interim

        # 4. Fit Means and Stds strictly on Train
        means = np.mean(X_interim, axis=0)
        stds = np.std(X_interim, axis=0)
        # Avoid division by zero on constant features
        stds[stds == 0.0] = 1.0

        # 5. Calculate Training Class Weights
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        if n_pos > 0 and n_neg > 0:
            # Balanced: n / (n_classes * count)
            w_0 = float(n_samples / (2.0 * n_neg))
            w_1 = float(n_samples / (2.0 * n_pos))
            # XGBoost scale_pos_weight: ratio of negative to positive count
            scale_pos = float(n_neg / n_pos)
        else:
            w_0, w_1 = 1.0, 1.0
            scale_pos = 1.0

        self.state = PreprocessorState(
            is_fitted=True,
            train_sample_count=n_samples,
            zero_variance_indices=zero_var_idx,
            means=means,
            stds=stds,
            velocity_median_train=vel_med,
            class_weights_balanced={0: w_0, 1: w_1},
            scale_pos_weight_xgboost=scale_pos,
        )
        return self

    def transform_for_trees(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform for tree-based models (XGBoost).
        
        Policy:
        - log1p transform on native amount features to compress extreme magnitudes.
        - inter_hop_velocity_avg sentinel (-1.0) mapped to np.nan (native XGBoost missing value).
        - 13 canonical dimensions preserved.
        """
        if not self.state.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        X = np.array([r["feature_values"] for r in records], dtype=np.float64)
        y = np.array([r["label"] for r in records], dtype=np.int32)

        # Log1p on amounts
        X[:, IDX_TOTAL_RECEIVED_NATIVE] = np.log1p(np.maximum(0.0, X[:, IDX_TOTAL_RECEIVED_NATIVE]))
        X[:, IDX_TOTAL_SENT_NATIVE] = np.log1p(np.maximum(0.0, X[:, IDX_TOTAL_SENT_NATIVE]))

        # Sentinel -> NaN for tree missing value handling
        vel = X[:, IDX_INTER_HOP_VELOCITY_AVG]
        vel[vel == SENTINEL_VELOCITY] = np.nan
        X[:, IDX_INTER_HOP_VELOCITY_AVG] = vel

        return X, y

    def transform_for_linear(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Transform for linear models (Logistic Regression).
        
        Policy:
        - log1p transform on native amounts.
        - Missingness indicator column appended for inter_hop_velocity sentinel.
        - Sentinel imputed with training median.
        - Standard scaling (mean, std) applied strictly using training statistics.
        - Output dimension is 14 (13 standardized features + 1 missingness indicator).
        """
        if not self.state.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        X = np.array([r["feature_values"] for r in records], dtype=np.float64)
        y = np.array([r["label"] for r in records], dtype=np.int32)

        # 1. Missingness indicator for velocity sentinel
        is_missing_velocity = (X[:, IDX_INTER_HOP_VELOCITY_AVG] == SENTINEL_VELOCITY).astype(np.float64)

        # 2. Impute velocity sentinel with train median
        X_clean = X.copy()
        vel = X_clean[:, IDX_INTER_HOP_VELOCITY_AVG]
        vel[vel == SENTINEL_VELOCITY] = self.state.velocity_median_train
        X_clean[:, IDX_INTER_HOP_VELOCITY_AVG] = vel

        # 3. Log1p on amounts
        X_clean[:, IDX_TOTAL_RECEIVED_NATIVE] = np.log1p(np.maximum(0.0, X_clean[:, IDX_TOTAL_RECEIVED_NATIVE]))
        X_clean[:, IDX_TOTAL_SENT_NATIVE] = np.log1p(np.maximum(0.0, X_clean[:, IDX_TOTAL_SENT_NATIVE]))

        # 4. Standard scale using Train means and stds
        X_scaled = (X_clean - self.state.means) / self.state.stds

        # 5. Append missingness indicator
        X_out = np.column_stack([X_scaled, is_missing_velocity])

        feature_names = list(CANONICAL_FEATURE_NAMES) + ["inter_hop_velocity_is_sentinel"]
        return X_out, y, feature_names
