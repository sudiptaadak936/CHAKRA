"""CHAKRA Step 5H.5.1: Address-Level Model Contract & Preprocessing Test Suite.

Verifies:
- Exact 13-feature canonical order and count.
- Schema 1.1.0 enforcement for EVM records.
- Legacy 1.0.0 EVM and invalidated provisional v3 rejection.
- Zero entity leakage in entity-grouped partitioning.
- Train-only preprocessing fitting (zero val/test leakage).
- Sentinel handling (-1.0) for tree and linear spaces.
- Zero-variance typology feature handling.
- Class weight calculation on training partition.
- Predictive matrix excludes metadata (address, entity_id, chain, etc.).
"""
import copy
import math
import numpy as np
import pytest

from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from ml_training.address_preprocessing import (
    CANONICAL_FEATURE_COUNT,
    IDX_INTER_HOP_VELOCITY_AVG,
    IDX_TOTAL_RECEIVED_NATIVE,
    IDX_TOTAL_SENT_NATIVE,
    SENTINEL_VELOCITY,
    AddressDatasetSplitter,
    AddressModelPreprocessor,
    is_eligible_training_record,
)


@pytest.fixture
def mock_valid_evm_record():
    return {
        "sample_id": "sample_valid_01",
        "chain": "evm",
        "network": "ethereum-mainnet",
        "entity_id": "entity_01",
        "normalized_address": "0x1111111111111111111111111111111111111111",
        "label": 1,
        "feature_schema_version": "1.1.0",
        "feature_values": [
            10.0, 5.0, 15.0, 1e19, 5e18, 0.5, 3600.0, 120.0, 8.0, 0.0, 0.0, 0.0, 0.0
        ],
        "observation_end": "2023-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_valid_btc_record():
    return {
        "sample_id": "sample_valid_btc_01",
        "chain": "bitcoin",
        "network": "bitcoin-mainnet",
        "entity_id": "entity_btc_01",
        "normalized_address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "label": 0,
        "feature_schema_version": "1.0.0",
        "feature_values": [
            2.0, 1.0, 3.0, 100000000.0, 50000000.0, 0.5, 7200.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0
        ],
        "observation_end": "2022-01-01T00:00:00Z",
    }


def test_canonical_feature_names_order():
    """Verify that canonical feature names order exactly matches 13 features."""
    assert len(CANONICAL_FEATURE_NAMES) == 13
    assert CANONICAL_FEATURE_NAMES[0] == "in_degree"
    assert CANONICAL_FEATURE_NAMES[1] == "out_degree"
    assert CANONICAL_FEATURE_NAMES[2] == "total_tx_count"
    assert CANONICAL_FEATURE_NAMES[3] == "total_received_native"
    assert CANONICAL_FEATURE_NAMES[4] == "total_sent_native"
    assert CANONICAL_FEATURE_NAMES[5] == "amount_retention_ratio"
    assert CANONICAL_FEATURE_NAMES[6] == "time_active_seconds"
    assert CANONICAL_FEATURE_NAMES[7] == "inter_hop_velocity_avg"
    assert CANONICAL_FEATURE_NAMES[8] == "unique_counterparties"
    assert CANONICAL_FEATURE_NAMES[9] == "typology_peel_chain_flag"
    assert CANONICAL_FEATURE_NAMES[10] == "typology_rapid_hop_flag"
    assert CANONICAL_FEATURE_NAMES[11] == "typology_fan_in_flag"
    assert CANONICAL_FEATURE_NAMES[12] == "typology_fan_out_flag"


def test_schema_1_1_0_enforcement_and_legacy_rejection(mock_valid_evm_record, mock_valid_btc_record):
    """Verify that 1.1.0 EVM and 1.0.0 Bitcoin are accepted, but 1.0.0 EVM is rejected."""
    # 1. Valid 1.1.0 EVM passes
    ok, err = is_eligible_training_record(mock_valid_evm_record)
    assert ok is True
    assert err is None

    # 2. Valid legacy BTC passes
    ok, err = is_eligible_training_record(mock_valid_btc_record)
    assert ok is True

    # 3. Legacy 1.0.0 EVM rejected
    legacy_evm = copy.deepcopy(mock_valid_evm_record)
    legacy_evm["feature_schema_version"] = "1.0.0"
    ok, err = is_eligible_training_record(legacy_evm)
    assert ok is False
    assert "Legacy EVM schema '1.0.0' rejected" in err

    # 4. Provisional v3 rejected
    provisional_rec = copy.deepcopy(mock_valid_evm_record)
    provisional_rec["sample_id"] = "PROVISIONAL_v3_deadbeef"
    ok, err = is_eligible_training_record(provisional_rec)
    assert ok is False
    assert "Invalidated provisional v3 sample rejected" in err


def test_feature_vector_dimension_guard(mock_valid_evm_record):
    """Verify fail-closed rejection on invalid feature vector lengths or non-finite values."""
    rec = copy.deepcopy(mock_valid_evm_record)
    rec["feature_values"] = rec["feature_values"][:12]  # 12 instead of 13
    ok, err = is_eligible_training_record(rec)
    assert ok is False
    assert "length 12 != 13" in err

    rec2 = copy.deepcopy(mock_valid_evm_record)
    rec2["feature_values"][5] = float("nan")
    ok, err = is_eligible_training_record(rec2)
    assert ok is False
    assert "Non-finite feature value" in err


def test_entity_grouped_partitioning_zero_leakage():
    """Verify entity-grouped chronological partitioning preserves zero entity leakage."""
    # Create 10 entities with varying cutoff dates and sample counts
    records = []
    for i in range(10):
        eid = f"entity_{i:02d}"
        cutoff = f"202{i % 4}-0{1 + (i % 9):02d}-01T00:00:00Z"
        # Each entity has 1-3 addresses
        for j in range(1 + (i % 3)):
            records.append({
                "sample_id": f"s_{i}_{j}",
                "entity_id": eid,
                "label": 1 if i < 7 else 0,
                "observation_end": cutoff,
                "feature_values": [1.0] * 13,
            })

    train, val, test = AddressDatasetSplitter.split_chronological(records, 0.7, 0.15, 0.15)

    assert train.sample_count > 0
    assert val.sample_count > 0
    assert test.sample_count > 0

    # Ensure zero entity overlap
    assert train.entity_ids.isdisjoint(val.entity_ids)
    assert train.entity_ids.isdisjoint(test.entity_ids)
    assert val.entity_ids.isdisjoint(test.entity_ids)

    # Check entity counts sum to total entities
    assert train.entity_count + val.entity_count + test.entity_count == 10


def test_train_only_preprocessing_fitting():
    """Verify that preprocessor statistics are fit strictly on training data."""
    train_records = [
        {"feature_values": [10.0, 1.0, 11.0, 1e18, 5e17, 0.5, 100.0, 20.0, 2.0, 0.0, 0.0, 0.0, 0.0], "label": 1},
        {"feature_values": [20.0, 2.0, 22.0, 2e18, 1e18, 0.5, 200.0, 40.0, 4.0, 0.0, 0.0, 0.0, 0.0], "label": 0},
    ]
    test_records = [
        {"feature_values": [999.0, 99.0, 1098.0, 1e22, 5e21, 0.5, 9999.0, 999.0, 50.0, 0.0, 0.0, 0.0, 0.0], "label": 1},
    ]

    preprocessor = AddressModelPreprocessor()
    preprocessor.fit(train_records)

    # State reflects training sample count
    assert preprocessor.state.is_fitted is True
    assert preprocessor.state.train_sample_count == 2

    # Typology features should be detected as zero variance on train
    assert 9 in preprocessor.state.zero_variance_indices
    assert 10 in preprocessor.state.zero_variance_indices
    assert 11 in preprocessor.state.zero_variance_indices
    assert 12 in preprocessor.state.zero_variance_indices

    # Transform test without altering fitted state
    saved_means = copy.deepcopy(preprocessor.state.means)
    X_test, y_test, _ = preprocessor.transform_for_linear(test_records)

    np.testing.assert_array_equal(preprocessor.state.means, saved_means)
    assert X_test.shape[0] == 1


def test_sentinel_handling_tree_and_linear():
    """Verify sentinel (-1.0) handling: NaN for tree, indicator + median for linear."""
    train_records = [
        {"feature_values": [1.0, 1.0, 2.0, 100.0, 50.0, 0.5, 10.0, 100.0, 2.0, 0.0, 0.0, 0.0, 0.0], "label": 1},
        {"feature_values": [1.0, 1.0, 2.0, 100.0, 50.0, 0.5, 10.0, 200.0, 2.0, 0.0, 0.0, 0.0, 0.0], "label": 1},
        {"feature_values": [1.0, 1.0, 2.0, 100.0, 50.0, 0.5, 10.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0], "label": 0},
    ]

    preprocessor = AddressModelPreprocessor().fit(train_records)

    # Median of valid velocities [100.0, 200.0] should be 150.0
    assert preprocessor.state.velocity_median_train == 150.0

    # 1. Tree space: sentinel -> NaN
    X_tree, _ = preprocessor.transform_for_trees(train_records)
    assert not np.isnan(X_tree[0, IDX_INTER_HOP_VELOCITY_AVG])
    assert not np.isnan(X_tree[1, IDX_INTER_HOP_VELOCITY_AVG])
    assert np.isnan(X_tree[2, IDX_INTER_HOP_VELOCITY_AVG])

    # 2. Linear space: sentinel -> train median, indicator appended
    X_lin, _, feat_names = preprocessor.transform_for_linear(train_records)
    assert len(feat_names) == 14
    assert feat_names[-1] == "inter_hop_velocity_is_sentinel"
    # Indicator values
    assert X_lin[0, -1] == 0.0
    assert X_lin[1, -1] == 0.0
    assert X_lin[2, -1] == 1.0


def test_class_weights_calculation():
    """Verify class weight calculation on imbalanced training sets."""
    # 4 positives, 1 negative (ratio 4:1)
    train_records = [
        {"feature_values": [1.0] * 13, "label": 1},
        {"feature_values": [1.0] * 13, "label": 1},
        {"feature_values": [1.0] * 13, "label": 1},
        {"feature_values": [1.0] * 13, "label": 1},
        {"feature_values": [1.0] * 13, "label": 0},
    ]
    preprocessor = AddressModelPreprocessor().fit(train_records)

    # Balanced weight: N / (2 * count)
    # class 0: 5 / (2 * 1) = 2.5
    # class 1: 5 / (2 * 4) = 0.625
    weights = preprocessor.state.class_weights_balanced
    assert weights[0] == pytest.approx(2.5)
    assert weights[1] == pytest.approx(0.625)

    # XGBoost scale_pos_weight: neg / pos = 1 / 4 = 0.25
    assert preprocessor.state.scale_pos_weight_xgboost == pytest.approx(0.25)


def test_predictive_matrix_excludes_metadata():
    """Verify that metadata strings never leak into the predictive matrix."""
    records = [{
        "sample_id": "sample_secret_hash",
        "entity_id": "secret_entity_999",
        "normalized_address": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "chain": "evm",
        "network": "ethereum-mainnet",
        "label": 1,
        "feature_values": [float(i) for i in range(13)],
    }]
    preprocessor = AddressModelPreprocessor().fit(records)
    X_tree, y_tree = preprocessor.transform_for_trees(records)

    assert X_tree.shape == (1, 13)
    assert X_tree.dtype == np.float64

    X_lin, y_lin, names = preprocessor.transform_for_linear(records)
    assert X_lin.shape == (1, 14)
    assert "entity_id" not in names
    assert "normalized_address" not in names
    assert "chain" not in names
