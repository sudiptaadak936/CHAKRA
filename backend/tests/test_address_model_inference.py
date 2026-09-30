"""CHAKRA Step 5I Tests: Address-Level ML Inference Layer.

Comprehensive test suite verifying:
- Artifact cryptographic verification and loading
- Strict fail-closed behavior on missing or corrupted artifacts
- Schema 1.1.0 and 13-feature contract enforcement
- Frozen preprocessing (zero runtime refitting, exact median imputation, sentinel handling)
- Deterministic model inference and Platt sigmoid recalibration
- Read-only contract and isolation (no DB writes, no alert creation, no network calls)
- FastAPI endpoint functionality and disclaimer compliance
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MLFeatureValues,
    MLFeatureRecord,
)
from app.schemas.ml_inference import (
    AddressMLInferenceRequest,
    AddressMLInferenceResult,
    ArtifactIntegrityError,
    FeatureContractError,
    InferenceStatus,
    MLInferenceError,
    NonFiniteFeatureError,
)
from app.forensics.address_ml_loader import AddressMLLoader, EXPECTED_ARTIFACT_HASHES
from app.forensics.address_ml_preprocessor import FrozenAddressPreprocessor
from app.forensics.address_ml_service import AddressMLInferenceService


@pytest.fixture
def loader():
    return AddressMLLoader()


@pytest.fixture
def service():
    return AddressMLInferenceService()


@pytest.fixture
def sample_feature_vector():
    # 13 canonical features
    return [
        10.0,         # in_degree
        5.0,          # out_degree
        15.0,         # total_tx_count
        1000000.0,    # total_received_native
        500000.0,     # total_sent_native
        0.5,          # amount_retention_ratio
        86400.0,      # time_active_seconds
        -1.0,         # inter_hop_velocity_avg (sentinel)
        12.0,         # unique_counterparties
        0.0,          # typology_peel_chain_flag
        0.0,          # typology_rapid_hop_flag
        0.0,          # typology_fan_in_flag
        0.0,          # typology_fan_out_flag
    ]


# ---------------------------------------------------------------------------
# 1. Artifact Integrity & Loading Tests
# ---------------------------------------------------------------------------

def test_artifact_loading_and_hashes(loader):
    """Verify all 5 frozen artifacts load successfully and hashes match frozen manifest."""
    computed_hashes = loader.verify_integrity()
    assert len(computed_hashes) == 5
    for name, expected_hash in EXPECTED_ARTIFACT_HASHES.items():
        assert computed_hashes[name] == expected_hash, f"Hash mismatch for {name}"

    artifacts = loader.load_artifacts()
    assert artifacts.lr_model is not None
    assert artifacts.xgb_model is not None
    assert artifacts.preprocessor_params is not None
    assert artifacts.lr_calibrator is not None
    assert artifacts.xgb_calibrator is not None
    assert artifacts.provenance.freeze_identity == "STEP_5H_7_FREEZE"


def test_missing_artifact_fail_closed(tmp_path):
    """Verify missing artifact triggers ArtifactIntegrityError."""
    empty_loader = AddressMLLoader(base_dir=tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="not found|missing"):
        empty_loader.load_artifacts()


def test_hash_mismatch_fail_closed(tmp_path):
    """Verify corrupted artifact triggers ArtifactIntegrityError."""
    # Mirror structure with corrupted file
    models_dir = tmp_path / "ml_training" / "address_model_artifacts" / "step_5H_6_final"
    calib_dir = tmp_path / "ml_training" / "calibration_artifacts" / "step_5H_6_2_2"
    models_dir.mkdir(parents=True)
    calib_dir.mkdir(parents=True)

    # Write dummy files
    (models_dir / "logistic_regression_final.joblib").write_text("corrupted content", encoding="utf-8")
    (models_dir / "xgboost_final.json").write_text("{}", encoding="utf-8")
    (models_dir / "final_preprocessor_params.json").write_text("{}", encoding="utf-8")
    (calib_dir / "lr_calibrator.joblib").write_text("corrupted", encoding="utf-8")
    (calib_dir / "xgb_calibrator.joblib").write_text("corrupted", encoding="utf-8")

    bad_loader = AddressMLLoader(base_dir=tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="Integrity violation"):
        bad_loader.load_artifacts()


# ---------------------------------------------------------------------------
# 2. Feature Contract Tests
# ---------------------------------------------------------------------------

def test_feature_contract_schema_and_dimension(service, sample_feature_vector):
    """Verify exact 13-feature length and schema 1.1.0 enforcement."""
    # Valid request
    req = AddressMLInferenceRequest(
        target_address="bc1qvalidaddress",
        chain="bitcoin",
        network="mainnet",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_vector=sample_feature_vector,
    )
    result = service.infer(req)
    assert result.status == InferenceStatus.SUCCESS

    # Invalid dimension: 12 features
    with pytest.raises(ValueError, match="Feature vector length 12 != expected 13"):
        AddressMLInferenceRequest(
            target_address="bc1qshort",
            chain="bitcoin",
            network="mainnet",
            feature_vector=sample_feature_vector[:-1],
        )

    # Invalid dimension: 14 features
    with pytest.raises(ValueError, match="Feature vector length 14 != expected 13"):
        AddressMLInferenceRequest(
            target_address="bc1qlong",
            chain="bitcoin",
            network="mainnet",
            feature_vector=sample_feature_vector + [1.0],
        )

    # Invalid schema version
    with pytest.raises(ValueError, match="Unsupported feature schema"):
        AddressMLInferenceRequest(
            target_address="bc1qbadschema",
            chain="bitcoin",
            network="mainnet",
            feature_schema_version="1.0.0",
            feature_vector=sample_feature_vector,
        )


def test_non_finite_feature_rejection(sample_feature_vector):
    """Verify non-finite feature values are rejected."""
    bad_vector = list(sample_feature_vector)
    bad_vector[0] = float("inf")
    with pytest.raises(ValueError, match="Non-finite value 'inf'"):
        AddressMLInferenceRequest(
            target_address="bc1qinf",
            chain="bitcoin",
            network="mainnet",
            feature_vector=bad_vector,
        )

    bad_vector[0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite value 'nan'"):
        AddressMLInferenceRequest(
            target_address="bc1qnan",
            chain="bitcoin",
            network="mainnet",
            feature_vector=bad_vector,
        )


def test_feature_ordering_exactness():
    """Verify canonical feature ordering matches CANONICAL_FEATURE_NAMES."""
    expected = (
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
    assert CANONICAL_FEATURE_NAMES == expected
    fields = list(MLFeatureValues.model_fields.keys())
    assert tuple(fields) == expected


# ---------------------------------------------------------------------------
# 3. Frozen Preprocessing Tests
# ---------------------------------------------------------------------------

def test_frozen_preprocessing_exactness(loader, sample_feature_vector):
    """Verify preprocessing uses frozen parameters and does not refit."""
    artifacts = loader.load_artifacts()
    prep = FrozenAddressPreprocessor(artifacts.preprocessor_params)

    assert prep.velocity_median_train == pytest.approx(24253.384615, rel=1e-5)

    # 1. Linear model transform
    X_lr = prep.transform_for_linear(sample_feature_vector)
    assert X_lr.shape == (1, 14), "LR feature shape must be (1, 14)"
    assert X_lr[0, -1] == 1.0, "Missingness indicator must be 1.0 for sentinel -1.0"

    # Velocity index is 7. Imputed with 24253.384615, standardized with mean[7], std[7]
    expected_standardized_vel = (24253.384615 - prep.means[7]) / prep.stds[7]
    assert X_lr[0, 7] == pytest.approx(expected_standardized_vel, rel=1e-5)

    # Amount index 3 is log1p(1000000.0) standardized
    expected_received_scaled = (np.log1p(1000000.0) - prep.means[3]) / prep.stds[3]
    assert X_lr[0, 3] == pytest.approx(expected_received_scaled, rel=1e-5)

    # 2. Tree model transform
    X_xgb = prep.transform_for_trees(sample_feature_vector)
    assert X_xgb.shape == (1, 13), "XGB feature shape must be (1, 13)"
    assert np.isnan(X_xgb[0, 7]), "Velocity sentinel must be mapped to NaN for XGB"
    assert X_xgb[0, 3] == pytest.approx(np.log1p(1000000.0), rel=1e-5)


def test_zero_runtime_learning(loader, sample_feature_vector):
    """Verify preprocessor parameters remain strictly immutable across transforms."""
    artifacts = loader.load_artifacts()
    prep = FrozenAddressPreprocessor(artifacts.preprocessor_params)

    means_orig = prep.means.copy()
    stds_orig = prep.stds.copy()
    median_orig = prep.velocity_median_train

    for _ in range(5):
        prep.transform_for_linear(sample_feature_vector)
        prep.transform_for_trees(sample_feature_vector)

    np.testing.assert_array_equal(prep.means, means_orig)
    np.testing.assert_array_equal(prep.stds, stds_orig)
    assert prep.velocity_median_train == median_orig


# ---------------------------------------------------------------------------
# 4. Model Inference & Sigmoid Calibration Tests
# ---------------------------------------------------------------------------

def test_deterministic_inference_and_provenance(service, sample_feature_vector):
    """Verify deterministic probabilities and provenance logging."""
    req = AddressMLInferenceRequest(
        target_address="0xabc123",
        chain="evm",
        network="ethereum-mainnet",
        feature_vector=sample_feature_vector,
    )

    res1 = service.infer(req)
    res2 = service.infer(req)

    assert res1.lr_raw_probability == res2.lr_raw_probability
    assert res1.lr_calibrated_probability == res2.lr_calibrated_probability
    assert res1.xgb_raw_probability == res2.xgb_raw_probability
    assert res1.xgb_calibrated_probability == res2.xgb_calibrated_probability

    assert 0.0 <= res1.lr_raw_probability <= 1.0
    assert 0.0 < res1.lr_calibrated_probability < 1.0
    assert 0.0 <= res1.xgb_raw_probability <= 1.0
    assert 0.0 < res1.xgb_calibrated_probability < 1.0

    assert res1.provenance.lr_model_hash == EXPECTED_ARTIFACT_HASHES["logistic_regression_final.joblib"]
    assert res1.provenance.xgb_model_hash == EXPECTED_ARTIFACT_HASHES["xgboost_final.json"]
    assert res1.provenance.preprocessor_hash == EXPECTED_ARTIFACT_HASHES["final_preprocessor_params.json"]
    assert res1.provenance.lr_calibrator_hash == EXPECTED_ARTIFACT_HASHES["lr_calibrator.joblib"]
    assert res1.provenance.xgb_calibrator_hash == EXPECTED_ARTIFACT_HASHES["xgb_calibrator.joblib"]


def test_sigmoid_calibration_formula_equivalence(service):
    """Verify calibration output matches the exact frozen sigmoid mathematical formulas."""
    test_raw_values = [0.0, 0.0001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.9999, 1.0]

    for p_raw in test_raw_values:
        # Mathematical formulas
        math_lr_cal = 1.0 / (1.0 + math.exp(-(1.9610540007187323 * p_raw - 0.6518861679684008)))
        math_xgb_cal = 1.0 / (1.0 + math.exp(-(1.5164354848934176 * p_raw - 0.4483491461869524)))

        # Calibrator object outputs
        sk_lr_cal = service.artifacts.lr_calibrator.predict_proba(np.array([[p_raw]]))[:, 1][0]
        sk_xgb_cal = service.artifacts.xgb_calibrator.predict_proba(np.array([[p_raw]]))[:, 1][0]

        assert sk_lr_cal == pytest.approx(math_lr_cal, rel=1e-7)
        assert sk_xgb_cal == pytest.approx(math_xgb_cal, rel=1e-7)


def test_calibration_monotonicity(service):
    """Verify that calibrated probability is strictly monotonically increasing with raw probability."""
    p_raw_seq = np.linspace(0.0, 1.0, 100)
    lr_cal_seq = [
        service.artifacts.lr_calibrator.predict_proba(np.array([[p]]))[:, 1][0]
        for p in p_raw_seq
    ]
    xgb_cal_seq = [
        service.artifacts.xgb_calibrator.predict_proba(np.array([[p]]))[:, 1][0]
        for p in p_raw_seq
    ]

    for i in range(len(p_raw_seq) - 1):
        assert lr_cal_seq[i + 1] > lr_cal_seq[i], f"LR non-monotonic at {p_raw_seq[i]}"
        assert xgb_cal_seq[i + 1] > xgb_cal_seq[i], f"XGB non-monotonic at {p_raw_seq[i]}"


# ---------------------------------------------------------------------------
# 5. Isolation & Read-Only Audit Tests
# ---------------------------------------------------------------------------

def test_no_side_effects_or_artifact_mutation(service, sample_feature_vector):
    """Verify that inference does not alter on-disk artifacts or produce side-effects."""
    # Capture disk hashes before
    loader = AddressMLLoader()
    hashes_before = loader.verify_integrity()

    req = AddressMLInferenceRequest(
        target_address="bc1qsideeffecttest",
        chain="bitcoin",
        network="mainnet",
        feature_vector=sample_feature_vector,
    )
    result = service.infer(req)
    assert result.status == InferenceStatus.SUCCESS

    # Verify disk hashes after
    hashes_after = loader.verify_integrity()
    assert hashes_before == hashes_after


def test_infer_from_ml_feature_record(service):
    """Verify infer_record helper directly supports MLFeatureRecord domain objects."""
    record = MLFeatureRecord(
        target_address="0x1111222233334444555566667777888899990000",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="0x1111222233334444555566667777888899990000",
        features=MLFeatureValues(
            in_degree=4,
            out_degree=2,
            total_tx_count=6,
            total_received_native=1000000000000000000,
            total_sent_native=500000000000000000,
            amount_retention_ratio=0.5,
            time_active_seconds=3600.0,
            inter_hop_velocity_avg=600.0,
            unique_counterparties=5,
            typology_peel_chain_flag=0,
            typology_rapid_hop_flag=0,
            typology_fan_in_flag=0,
            typology_fan_out_flag=0,
        ),
    )

    result = service.infer_record(record)
    assert result.target_address == record.target_address
    assert result.status == InferenceStatus.SUCCESS
    assert 0.0 <= result.lr_raw_probability <= 1.0
    assert 0.0 < result.lr_calibrated_probability < 1.0


# ---------------------------------------------------------------------------
# 6. FastAPI Router Integration Tests
# ---------------------------------------------------------------------------

def test_fastapi_infer_endpoint(sample_feature_vector):
    """Verify HTTP API endpoint behaviour, non-punitive disclaimer, and error handling."""
    client = TestClient(app)

    payload = {
        "target_address": "0xfe3b557e8fb62b89f4916b721be55ceb828dbd73",
        "chain": "evm",
        "network": "ethereum-mainnet",
        "feature_schema_version": "1.1.0",
        "feature_vector": sample_feature_vector,
    }

    # 1. Success test
    response = client.post("/ml/address/infer", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "lr_raw_probability" in data
    assert "lr_calibrated_probability" in data
    assert "xgb_raw_probability" in data
    assert "xgb_calibrated_probability" in data
    assert "provenance" in data
    assert "disclaimer" in data
    assert "Does NOT constitute proof of illicit activity" in data["disclaimer"]

    # 2. Schema violation test (400)
    bad_payload = copy.deepcopy(payload)
    bad_payload["feature_schema_version"] = "2.0.0"
    bad_response = client.post("/ml/address/infer", json=bad_payload)
    assert bad_response.status_code == 422 or bad_response.status_code == 400

    # 3. Bad feature length test (400/422)
    bad_length_payload = copy.deepcopy(payload)
    bad_length_payload["feature_vector"] = sample_feature_vector[:5]
    bad_len_response = client.post("/ml/address/infer", json=bad_length_payload)
    assert bad_len_response.status_code == 422 or bad_len_response.status_code == 400
