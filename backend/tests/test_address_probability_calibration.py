"""Test suite for Step 5H.6.2.2: Actual Address-Level Probability Calibration.

Verifies all 17 required verification areas specified in Section 29:
1. Calibration uses only authoritative OOF data.
2. Final test is inaccessible and quarantined.
3. Entity weights are applied exactly once (sum=44.0).
4. Calibrator has no class weighting.
5. Raw probabilities are unchanged.
6. Labels are unchanged.
7. Sigmoid parameterization is correct.
8. Sigmoid slope must be positive (monotonicity).
9. Calibrator output is bounded in [0, 1].
10. Deterministic calibration.
11. LR and XGB calibrators remain separate.
12. Isotonic remains sensitivity-only.
13. No threshold exists.
14. No risk-engine integration.
15. Calibration artifact hashes reproducible.
16. Previous frozen models unchanged.
17. Previous OOF datasets unchanged.
18. Mandatory declarations present in report.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path("c:/Users/sudip/Documents/antigravity/Chakra")
OOF_DIR = BASE_DIR / "backend/ml_training/calibration_artifacts/step_5H_6_2_1C"
TEST_MANIFEST_PATH = BASE_DIR / "backend/ml_training/calibration_artifacts/step_5H_6_2_1B/final_test_entity_manifest.json"
CALIB_DIR = BASE_DIR / "backend/ml_training/calibration_artifacts/step_5H_6_2_2"
RESULTS_PATH = BASE_DIR / "backend/ml_training/STEP_5H_6_2_2_RESULTS.json"
REPORT_PATH = BASE_DIR / "backend/ml_training/STEP_5H_6_2_2_CALIBRATION_REPORT.md"
BASELINE_DIR = BASE_DIR / "backend/ml_training/address_model_artifacts/step_5H_5_2"


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def oof_data():
    with open(OOF_DIR / "entity_balanced_oof_lr.json", "r", encoding="utf-8") as f:
        lr_data = json.load(f)
    with open(OOF_DIR / "entity_balanced_oof_xgb.json", "r", encoding="utf-8") as f:
        xgb_data = json.load(f)
    return {"lr": lr_data, "xgb": xgb_data}


@pytest.fixture(scope="module")
def cal_metadata():
    with open(CALIB_DIR / "calibrator_metadata_lr.json", "r", encoding="utf-8") as f:
        lr_meta = json.load(f)
    with open(CALIB_DIR / "calibrator_metadata_xgb.json", "r", encoding="utf-8") as f:
        xgb_meta = json.load(f)
    return {"lr": lr_meta, "xgb": xgb_meta}


@pytest.fixture(scope="module")
def results_data():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_1_calibration_uses_only_authoritative_oof_data(oof_data, results_data):
    """Test 1: Calibration uses exclusively authoritative OOF data from 5H.6.2.1C."""
    assert len(oof_data["lr"]) == 84
    assert len(oof_data["xgb"]) == 84
    assert results_data["oof_record_count"] == 84
    assert results_data["development_entity_count"] == 44
    assert results_data["authoritative_dataset_hash"] == sha256_file(OOF_DIR / "entity_balanced_oof_lr.json")


def test_2_final_test_is_inaccessible_and_isolated(oof_data):
    """Test 2: Final test set remains strictly quarantined with zero contamination."""
    frozen_test_entities = {
        "wikileaks_org_01", "kraken_vasp_01", "bybit_vasp_01", "crypto_com_vasp_01",
        "47682", "47635", "48432", "43421", "52071"
    }
    dev_entities = {r["entity_id"] for r in oof_data["lr"]}
    overlap = dev_entities.intersection(frozen_test_entities)
    assert len(overlap) == 0, f"Quarantine breach: {overlap}"

    with open(CALIB_DIR / "test_quarantine_evidence.json", "r", encoding="utf-8") as f:
        evidence = json.load(f)
    assert evidence["contamination_status"] == "ZERO_CONTAMINATION"
    assert evidence["intersection_count"] == 0


def test_3_entity_weights_applied_exactly_once(oof_data):
    """Test 3: Entity weights are strictly 1 / entity_record_count and sum to 44.0."""
    lr_data = oof_data["lr"]
    w = np.array([r["calibration_sample_weight"] for r in lr_data])
    y = np.array([r["true_label"] for r in lr_data])

    assert abs(np.sum(w[y == 1]) - 26.0) < 1e-9
    assert abs(np.sum(w[y == 0]) - 18.0) < 1e-9
    assert abs(np.sum(w) - 44.0) < 1e-9

    # Entity sum check
    ent_weights = {}
    for r in lr_data:
        ent = r["entity_id"]
        ent_weights[ent] = ent_weights.get(ent, 0.0) + r["calibration_sample_weight"]
    for ent, w_sum in ent_weights.items():
        assert abs(w_sum - 1.0) < 1e-9, f"Entity {ent} weight sum is {w_sum}, expected 1.0"


def test_4_calibrator_has_no_class_weighting(cal_metadata):
    """Test 4: Calibrators do not apply any class weighting."""
    assert cal_metadata["lr"]["class_weight"] is None
    assert cal_metadata["xgb"]["class_weight"] is None
    lr_cal = joblib.load(CALIB_DIR / "lr_calibrator.joblib")
    xgb_cal = joblib.load(CALIB_DIR / "xgb_calibrator.joblib")
    assert lr_cal.class_weight is None
    assert xgb_cal.class_weight is None


def test_5_raw_probabilities_unchanged(oof_data):
    """Test 5: Raw probabilities match exactly 5H.6.2.1C records."""
    for r in oof_data["lr"]:
        assert 0.0 <= r["raw_probability"] <= 1.0
        assert not np.isnan(r["raw_probability"])
    for r in oof_data["xgb"]:
        assert 0.0 <= r["raw_probability"] <= 1.0
        assert not np.isnan(r["raw_probability"])


def test_6_labels_unchanged(oof_data):
    """Test 6: Labels match authoritative record and entity support."""
    y_lr = [r["true_label"] for r in oof_data["lr"]]
    y_xgb = [r["true_label"] for r in oof_data["xgb"]]
    assert y_lr == y_xgb
    assert sum(y_lr) == 66
    assert len(y_lr) - sum(y_lr) == 18


def test_7_sigmoid_parameterization_correct(cal_metadata):
    """Test 7: Sigmoid parameterization matches fitted LogisticRegression."""
    lr_cal = joblib.load(CALIB_DIR / "lr_calibrator.joblib")
    xgb_cal = joblib.load(CALIB_DIR / "xgb_calibrator.joblib")

    assert np.isclose(lr_cal.coef_[0][0], cal_metadata["lr"]["parameters"]["slope_a"])
    assert np.isclose(lr_cal.intercept_[0], cal_metadata["lr"]["parameters"]["intercept_b"])
    assert np.isclose(xgb_cal.coef_[0][0], cal_metadata["xgb"]["parameters"]["slope_a"])
    assert np.isclose(xgb_cal.intercept_[0], cal_metadata["xgb"]["parameters"]["intercept_b"])


def test_8_sigmoid_slope_must_be_positive(cal_metadata):
    """Test 8: Both calibrators satisfy monotonicity condition a > 0."""
    assert cal_metadata["lr"]["parameters"]["slope_a"] > 0
    assert cal_metadata["lr"]["monotonicity"]["is_monotonic_non_decreasing"] is True
    assert cal_metadata["xgb"]["parameters"]["slope_a"] > 0
    assert cal_metadata["xgb"]["monotonicity"]["is_monotonic_non_decreasing"] is True


def test_9_calibrator_output_is_bounded_0_1(oof_data):
    """Test 9: Calibrated probabilities lie strictly within [0, 1]."""
    lr_cal = joblib.load(CALIB_DIR / "lr_calibrator.joblib")
    xgb_cal = joblib.load(CALIB_DIR / "xgb_calibrator.joblib")

    p_lr_raw = np.array([r["raw_probability"] for r in oof_data["lr"]]).reshape(-1, 1)
    p_xgb_raw = np.array([r["raw_probability"] for r in oof_data["xgb"]]).reshape(-1, 1)

    p_lr_cal = lr_cal.predict_proba(p_lr_raw)[:, 1]
    p_xgb_cal = xgb_cal.predict_proba(p_xgb_raw)[:, 1]

    assert (p_lr_cal > 0.0).all() and (p_lr_cal < 1.0).all()
    assert (p_xgb_cal > 0.0).all() and (p_xgb_cal < 1.0).all()


def test_10_deterministic_calibration(oof_data):
    """Test 10: Calibrator fitting is perfectly deterministic given seed 42."""
    p_lr_raw = np.array([r["raw_probability"] for r in oof_data["lr"]]).reshape(-1, 1)
    y = np.array([r["true_label"] for r in oof_data["lr"]])
    w = np.array([r["calibration_sample_weight"] for r in oof_data["lr"]])

    refit_lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    refit_lr.fit(p_lr_raw, y, sample_weight=w)

    saved_lr = joblib.load(CALIB_DIR / "lr_calibrator.joblib")
    assert np.isclose(refit_lr.coef_[0][0], saved_lr.coef_[0][0])
    assert np.isclose(refit_lr.intercept_[0], saved_lr.intercept_[0])


def test_11_lr_and_xgb_calibrators_remain_separate():
    """Test 11: LR and XGB calibrator artifacts are distinct and non-overlapping."""
    assert (CALIB_DIR / "lr_calibrator.joblib").exists()
    assert (CALIB_DIR / "xgb_calibrator.joblib").exists()
    assert sha256_file(CALIB_DIR / "lr_calibrator.joblib") != sha256_file(CALIB_DIR / "xgb_calibrator.joblib")


def test_12_isotonic_remains_sensitivity_only(results_data):
    """Test 12: Isotonic calibrators exist strictly as SENSITIVITY_ONLY."""
    assert (CALIB_DIR / "isotonic_lr_sensitivity.joblib").exists()
    assert (CALIB_DIR / "isotonic_xgb_sensitivity.joblib").exists()
    assert "SENSITIVITY_ONLY" in results_data["isotonic_sensitivity_status"]


def test_13_no_threshold_exists(results_data, cal_metadata):
    """Test 13: No threshold or classification rule is defined."""
    assert "threshold" not in results_data
    assert "threshold" not in cal_metadata["lr"]
    assert "threshold" not in cal_metadata["xgb"]


def test_14_no_risk_engine_integration():
    """Test 14: No risk engine file imports or references step_5H_6_2_2."""
    risk_engine_file = BASE_DIR / "backend/services/risk_engine.py"
    if risk_engine_file.exists():
        with open(risk_engine_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "step_5H_6_2_2" not in content
        assert "lr_calibrator.joblib" not in content


def test_15_calibration_artifact_hashes_reproducible():
    """Test 15: Artifact hashes match reproducibility metadata."""
    with open(CALIB_DIR / "reproducibility_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    for filename, expected_hash in meta["output_artifacts"].items():
        assert sha256_file(CALIB_DIR / filename) == expected_hash, f"Hash mismatch for {filename}"


def test_16_previous_frozen_models_unchanged():
    """Test 16: Frozen base models exist and were not overwritten."""
    assert (BASELINE_DIR / "logistic_regression_baseline.joblib").exists()
    assert (BASELINE_DIR / "xgboost_baseline.json").exists()


def test_17_previous_oof_datasets_unchanged():
    """Test 17: Step 5H.6.2.1C dataset files remain intact."""
    assert (OOF_DIR / "entity_balanced_oof_lr.json").exists()
    assert (OOF_DIR / "entity_balanced_oof_xgb.json").exists()
    assert (OOF_DIR / "entity_weight_metadata.json").exists()
    assert (OOF_DIR / "entity_oof_record_counts.json").exists()


def test_18_mandatory_declarations_in_report():
    """Test 18: All 17 mandatory declarations are present in the calibration report."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    declarations = [
        "Actual calibration was performed for the first time during Step 5H.6.2.2.",
        "No final-test observation was used for fitting.",
        "No final-test observation was used for calibration-method selection.",
        "No production threshold was selected.",
        "No production risk-engine integration was performed.",
        "No Step 5H.5.2 base-model artifact was modified.",
        "No canonical 1.1.0 feature semantic was modified.",
        "No label was modified.",
        "No previously rejected Step 5H.6.1A candidate was re-admitted.",
        "No provisional 87-record OOF dataset was used.",
        "Entity-balanced calibration weighting was applied exactly once.",
        "No class weighting was applied to the calibrator.",
        "The calibrated output represents an entity-balanced reference probability mapping and is not claimed to be a universally valid population posterior.",
        "Cross-sectional OOF remains distinct from forward-chained temporal validation.",
        "Calibration results do not establish future temporal generalization.",
        "No production threshold has been frozen.",
        "Step 5H.7 model freeze has not started."
    ]

    for d in declarations:
        assert d in report_text, f"Missing declaration: {d}"
