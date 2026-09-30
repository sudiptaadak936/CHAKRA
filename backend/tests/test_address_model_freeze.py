"""Unit Test Suite for Step 5H.7: Final Address-Level Supervised ML Artifacts Freeze & Provenance Lock.

Verifies all 10 required reconciliation areas:
1. Final preprocessing artifact existence.
2. Exact LR velocity imputation value (24253.3846).
3. Sentinel handling (-1.0 missing sentinel).
4. Feature ordering (canonical 13 features).
5. Schema 1.1.0 and target ADDRESS.
6. Final model hashes (LR & XGBoost).
7. Calibrator hashes and immutability (LR & XGBoost).
8. Step 5H.5.2 historical baseline preservation byte-for-byte.
9. Training/test quarantine and zero leakage.
10. Freeze-manifest integrity and provenance lock.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np

from app.schemas.ml_features import CANONICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ml_training.address_preprocessing import (
    AddressModelPreprocessor,
    SENTINEL_VELOCITY,
    IDX_INTER_HOP_VELOCITY_AVG,
)
from ml_training.build_cross_sectional_oof import FROZEN_5H_6_FINAL_TEST_EIDS

BASE_DIR = Path("c:/Users/sudip/Documents/antigravity/Chakra")
FINAL_MODEL_DIR = BASE_DIR / "backend/ml_training/address_model_artifacts/step_5H_6_final"
CALIB_DIR = BASE_DIR / "backend/ml_training/calibration_artifacts/step_5H_6_2_2"
FREEZE_DIR = BASE_DIR / "backend/ml_training/address_model_artifacts/step_5H_7_freeze"
BASELINE_5H_5_2_DIR = BASE_DIR / "backend/ml_training/address_model_artifacts/step_5H_5_2"
REPORT_PATH = BASE_DIR / "backend/ml_training/STEP_5H_7_MODEL_FREEZE_REPORT.md"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def freeze_manifest():
    with open(FREEZE_DIR / "freeze_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def provenance_lock():
    with open(FREEZE_DIR / "provenance_lock.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def quarantine_verification():
    with open(FREEZE_DIR / "test_quarantine_verification.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def preprocessor_params():
    with open(FINAL_MODEL_DIR / "final_preprocessor_params.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_final_preprocessing_artifact_existence():
    """Verify existence of final preprocessing artifact and model artifacts."""
    assert (FINAL_MODEL_DIR / "final_preprocessor_params.json").exists()
    assert (FINAL_MODEL_DIR / "logistic_regression_final.joblib").exists()
    assert (FINAL_MODEL_DIR / "xgboost_final.json").exists()
    assert (FINAL_MODEL_DIR / "final_training_config.json").exists()
    assert (FINAL_MODEL_DIR / "final_dataset_manifest.json").exists()
    assert (FINAL_MODEL_DIR / "final_artifact_hashes.json").exists()


def test_2_exact_lr_velocity_imputation_value(preprocessor_params, freeze_manifest):
    """Verify that velocity median imputation value is authoritatively 24253.3846."""
    actual_val = float(preprocessor_params["velocity_median_train"])
    assert abs(actual_val - 24253.384615) < 1e-4
    assert round(actual_val, 4) == 24253.3846

    manifest_val = float(freeze_manifest["preprocessor"]["velocity_median_train"])
    assert abs(manifest_val - 24253.384615) < 1e-4

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "24253.3846" in report_text


def test_3_sentinel_handling():
    """Verify velocity missing sentinel is -1.0 and handled correctly."""
    assert SENTINEL_VELOCITY == -1.0
    assert IDX_INTER_HOP_VELOCITY_AVG == 7


def test_4_feature_ordering(freeze_manifest):
    """Verify exact 13-feature canonical ordering."""
    expected_order = [
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
    ]
    assert list(CANONICAL_FEATURE_NAMES) == expected_order
    assert freeze_manifest["feature_contract"]["canonical_features_exact_order"] == expected_order


def test_5_schema_1_1_0(freeze_manifest, preprocessor_params):
    """Verify Schema 1.1.0 and target ADDRESS."""
    assert FEATURE_SCHEMA_VERSION == "1.1.0"
    assert freeze_manifest["feature_contract"]["schema_version"] == "1.1.0"
    assert freeze_manifest["feature_contract"]["target"] == "ADDRESS"
    assert preprocessor_params["schema_version"] == "1.1.0"


def test_6_final_model_hashes(freeze_manifest):
    """Verify exact SHA-256 hashes for final models."""
    exp_lr = "569bbca10c392f06b262a5c0cb7ae5209a34ecc92a852ff93ed2e576b2f86bac"
    exp_xgb = "e320eb085b2d091bb202d03261cbc15683ede51070c152313cd211ba13575397"

    act_lr = sha256_file(FINAL_MODEL_DIR / "logistic_regression_final.joblib")
    act_xgb = sha256_file(FINAL_MODEL_DIR / "xgboost_final.json")

    assert act_lr == exp_lr
    assert act_xgb == exp_xgb
    assert freeze_manifest["final_models"]["logistic_regression"]["sha256"] == exp_lr
    assert freeze_manifest["final_models"]["xgboost"]["sha256"] == exp_xgb


def test_7_calibrator_hashes_and_immutability(freeze_manifest):
    """Verify exact SHA-256 hashes and parameters for calibrators."""
    exp_lr_cal = "1e78b493577a03884a4ddcaaf4488cf507f6448aaf16c9eb865f337f513a98df"
    exp_xgb_cal = "90d61a4a7cc91442e58104bb9e7fa9c57c0554df13a55b034fef4fe7ab00aad2"

    act_lr_cal = sha256_file(CALIB_DIR / "lr_calibrator.joblib")
    act_xgb_cal = sha256_file(CALIB_DIR / "xgb_calibrator.joblib")

    assert act_lr_cal == exp_lr_cal
    assert act_xgb_cal == exp_xgb_cal

    lr_params = freeze_manifest["calibrators"]["logistic_regression"]["parameters"]
    xgb_params = freeze_manifest["calibrators"]["xgboost"]["parameters"]

    assert abs(lr_params["slope_a"] - 1.9610540007187323) < 1e-9
    assert abs(lr_params["intercept_b"] - (-0.6518861679684008)) < 1e-9
    assert abs(xgb_params["slope_a"] - 1.5164354848934176) < 1e-9
    assert abs(xgb_params["intercept_b"] - (-0.4483491461869524)) < 1e-9


def test_8_5h_5_2_baseline_hashes():
    """Verify historical baseline models remain byte-for-byte unchanged."""
    with open(BASELINE_5H_5_2_DIR / "hashes.json", "r") as f:
        recorded = json.load(f)["artifact_hashes"]

    assert sha256_file(BASELINE_5H_5_2_DIR / "logistic_regression_baseline.joblib") == recorded["logistic_regression_model_sha256"]
    assert sha256_file(BASELINE_5H_5_2_DIR / "xgboost_baseline.json") == recorded["xgboost_model_sha256"]
    assert sha256_file(BASELINE_5H_5_2_DIR / "preprocessor_params.json") == recorded["preprocessor_params_sha256"]


def test_9_training_test_quarantine(freeze_manifest, quarantine_verification):
    """Verify 84 dev records / 44 entities and 9 final-test entities with zero overlap."""
    train = freeze_manifest["training_population"]
    assert train["total_records"] == 84
    assert train["total_entities"] == 44
    assert train["positive_records"] == 66
    assert train["negative_records"] == 18
    assert train["positive_entities"] == 26
    assert train["negative_entities"] == 18

    test_pop = freeze_manifest["final_test_population"]
    assert test_pop["total_records"] == 9
    assert test_pop["total_entities"] == 9
    assert test_pop["positive_records"] == 5
    assert test_pop["negative_records"] == 4
    assert set(test_pop["entity_ids"]) == set(FROZEN_5H_6_FINAL_TEST_EIDS)

    q = quarantine_verification["quarantine_checks"]
    assert q["test_entities_in_training_manifest"] is False
    assert q["test_records_in_training_dataset"] is False
    assert q["intersection_count"] == 0
    assert quarantine_verification["verdict"] == "STRICT_QUARANTINE_VERIFIED"


def test_10_freeze_manifest_integrity(freeze_manifest, provenance_lock):
    """Verify freeze-manifest integrity and provenance lock."""
    assert freeze_manifest["freeze_verdict"] == "STEP_5H_7_FROZEN"
    assert freeze_manifest["declarations"]["production_threshold_selected"] is False
    assert freeze_manifest["declarations"]["step_5i_inference_started"] is False

    sep = provenance_lock["architecture_separation"]
    assert sep["RESEARCH_MODELS"]["production_address_inference_use"] is False
    assert sep["BASELINE_MODELS"]["production_address_inference_use"] is False
    assert sep["TEST_EVALUATION"]["optimization_feedback"] is False
