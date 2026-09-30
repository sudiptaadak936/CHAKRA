"""CHAKRA Step 5J.2 Policy Contract Verification Tests.

Verifies:
- step_5J_2_INDEPENDENT_CHANNEL_POLICY.json exists and parses cleanly
- Contract version is present and matches '1.0.0'
- Policy status is 'FROZEN'
- Architecture is 'INDEPENDENT_CHANNELS'
- Numerical fusion is explicitly disabled (numerical_fusion_enabled = False)
- No LR/XGB averaging or ensemble (lr_xgb_ensemble_enabled = False, ensemble_weights = None)
- No probability-to-points mapping (probability_to_points_mapping = None)
- No operational ML threshold (operational_ml_threshold = None, risk_level_mapping_allowed = False)
- Deterministic channel remains authoritative (status = ACTIVE_AUTHORITATIVE)
- ML cannot modify deterministic score or level
- Sanctions override by ML is strictly False
- Provenance fields match Step 5H.7 frozen hashes and Schema 1.1.0
- Final test entities remain completely absent from policy inputs
- RiskMLFusionService returns NOT_CONFIGURED when fusion requested while disabled
- RiskMLFusionService fails closed on invalid/unsupported policy version
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.forensics.address_ml_loader import EXPECTED_ARTIFACT_HASHES
from app.forensics.risk_ml_fusion_service import RiskMLFusionService
from app.schemas.risk_fusion import FusionStatus
from ml_training.build_cross_sectional_oof import FROZEN_5H_6_FINAL_TEST_EIDS


POLICY_FILE = (
    Path(__file__).resolve().parents[1]
    / "ml_training"
    / "policy"
    / "step_5J_2_INDEPENDENT_CHANNEL_POLICY.json"
)
SPEC_DOC = (
    Path(__file__).resolve().parents[1]
    / "ml_training"
    / "STEP_5J_2_POLICY_SPECIFICATION.md"
)


@pytest.fixture
def policy_data() -> dict:
    assert POLICY_FILE.exists(), f"Policy file not found: {POLICY_FILE}"
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_policy_json_parses_and_has_required_top_level_keys(policy_data: dict):
    """Verify policy contract loads cleanly and possesses all required top-level keys."""
    assert policy_data["contract_version"] == "1.0.0"
    assert policy_data["policy_status"] == "FROZEN"
    assert policy_data["architecture"] == "INDEPENDENT_CHANNELS"
    assert policy_data["policy_name"] == "STEP_5J_2_INDEPENDENT_CHANNEL_POLICY"


def test_numerical_fusion_explicitly_disabled(policy_data: dict):
    """Verify numerical fusion is unconditionally disabled with explicit nulls."""
    assert policy_data["numerical_fusion_enabled"] is False
    assert policy_data["probability_to_points_mapping"] is None

    num_fusion = policy_data["numerical_fusion"]
    assert num_fusion["numerical_fusion_enabled"] is False
    assert num_fusion["fusion_status"] == "NOT_CONFIGURED"
    assert num_fusion["fusion_score"] is None
    assert num_fusion["fusion_policy_version"] is None
    assert num_fusion["probability_to_points_mapping"] is None


def test_no_model_combination_or_averaging(policy_data: dict):
    """Verify no LR/XGB averaging or ensemble weighting is authorized."""
    assert policy_data["lr_xgb_ensemble_enabled"] is False

    combo = policy_data["model_combination"]
    assert combo["lr_xgb_ensemble_enabled"] is False
    assert combo["ensemble_weights"] is None
    assert combo["combination_operator"] is None
    assert combo["selection_status"] == "NO_WINNER_SELECTED"


def test_threshold_policy_contract(policy_data: dict):
    """Verify no operational ML threshold is authorized and no risk band mapping is allowed."""
    assert policy_data["operational_ml_threshold"] is None

    thresh = policy_data["threshold_policy"]
    assert thresh["operational_ml_threshold"] is None
    assert thresh["risk_level_mapping_allowed"] is False
    assert thresh["diagnostic_eval_threshold"] == 0.5
    assert thresh["diagnostic_eval_scope"] == "HISTORICAL_EVALUATION_ONLY"


def test_deterministic_channel_immutability(policy_data: dict):
    """Verify deterministic channel is authoritative and cannot be altered by ML."""
    assert policy_data["ml_affects_deterministic_score"] is False
    assert policy_data["ml_affects_deterministic_level"] is False
    assert policy_data["sanctions_override_by_ml"] is False

    det = policy_data["deterministic_channel"]
    assert det["status"] == "ACTIVE_AUTHORITATIVE"
    assert det["scoring_semantics"] == "UNCHANGED"
    assert det["sanctions_behavior"] == "UNCHANGED"
    assert det["sanctions_hit_points"] == 100.0
    assert det["sanctions_override_by_ml"] is False
    assert det["ml_affects_deterministic_score"] is False
    assert det["ml_affects_deterministic_level"] is False
    assert det["ml_attenuates_deterministic_evidence"] is False
    assert det["score_range"] == [0.0, 100.0]


def test_sanctions_precedence_policy(policy_data: dict):
    """Verify sanctions policy has strict deterministic override."""
    sanc = policy_data["sanctions_policy"]
    assert sanc["sanctions_override_by_ml"] is False
    assert sanc["sanctions_precedence"] == "STRICT_DETERMINISTIC_OVERRIDE"
    assert sanc["deterministic_sanctions_points"] == 100.0


def test_provenance_cryptographic_integrity(policy_data: dict):
    """Verify policy provenance fields exactly match frozen Step 5H.7 hashes."""
    prov = policy_data["provenance_policy"]
    assert prov["deterministic_policy_version"] == "STEP_5_DETERMINISTIC_RISK_POLICY_v1.0.0"
    assert prov["ml_freeze_identity"] == "STEP_5H_7_FINAL_FREEZE"
    assert prov["feature_schema_version"] == "1.1.0"

    # Compare hashes against canonical EXPECTED_ARTIFACT_HASHES
    assert (
        prov["model_artifacts"]["logistic_regression"]["sha256"]
        == EXPECTED_ARTIFACT_HASHES["logistic_regression_final.joblib"]
    )
    assert (
        prov["model_artifacts"]["xgboost"]["sha256"]
        == EXPECTED_ARTIFACT_HASHES["xgboost_final.json"]
    )
    assert (
        prov["calibrator_artifacts"]["logistic_regression"]["sha256"]
        == EXPECTED_ARTIFACT_HASHES["lr_calibrator.joblib"]
    )
    assert (
        prov["calibrator_artifacts"]["xgboost"]["sha256"]
        == EXPECTED_ARTIFACT_HASHES["xgb_calibrator.joblib"]
    )
    assert (
        prov["preprocessor_artifact"]["sha256"]
        == EXPECTED_ARTIFACT_HASHES["final_preprocessor_params.json"]
    )


def test_final_test_entities_absent_from_policy(policy_data: dict):
    """Verify the 9 final-test entities are completely absent from policy contract."""
    policy_str = json.dumps(policy_data)
    for eid in FROZEN_5H_6_FINAL_TEST_EIDS:
        assert str(eid) not in policy_str


def test_fusion_service_behavior_under_disabled_policy():
    """Verify RiskMLFusionService returns NOT_CONFIGURED when fusion is disabled."""
    service = RiskMLFusionService(fusion_policy_version=None)
    res = service.evaluate_composite(
        target_address="0x1111111111111111111111111111111111111111",
        chain="evm",
        network="mainnet",
        feature_vector=[0.0] * 13,
    )
    assert res.fusion_layer.fusion_status == FusionStatus.NOT_CONFIGURED
    assert res.fusion_layer.fusion_score is None
    assert res.deterministic_layer.overall_score is not None
    assert res.ml_layer.lr_calibrated_probability is not None


def test_fusion_service_fails_closed_on_unsupported_version():
    """Verify RiskMLFusionService fails closed with FAILED on unrecognized policy versions."""
    service = RiskMLFusionService(fusion_policy_version="UNAUTHORIZED_FUSION_V99")
    res = service.evaluate_composite(
        target_address="0x1111111111111111111111111111111111111111",
        chain="evm",
        network="mainnet",
        feature_vector=[0.0] * 13,
    )
    assert res.fusion_layer.fusion_status == FusionStatus.FAILED
    assert res.fusion_layer.fusion_score is None
    assert "Unsupported or unauthorized fusion policy version" in res.fusion_layer.reason


def test_specification_document_content():
    """Verify STEP_5J_2_POLICY_SPECIFICATION.md exists and contains required declarations."""
    assert SPEC_DOC.exists()
    content = SPEC_DOC.read_text(encoding="utf-8")
    assert "This policy does not claim that ML is ineffective." in content
    assert (
        "This policy states that the current evidence does not identify a defensible "
        "numerical mapping between deterministic risk and ML signals." in content
    )
    assert "NUMERICAL ML/RISK FUSION DISABLED — INDEPENDENT CHANNEL ARCHITECTURE FROZEN" in content
    assert "NOT_CONFIGURED" in content
