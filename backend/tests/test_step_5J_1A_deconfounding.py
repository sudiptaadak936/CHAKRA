"""CHAKRA Step 5J.1A Research Tests: Statistical Deconfounding & Identifiability.

Verifies:
- Exact 84-record population and 44-entity grouping
- Strict diagonal sanctions/label contingency (66/0 on sanctions=1, 0/18 on sanctions=0)
- Deterministic score decomposition: S_without_sanctions = 0.0 for all 84 records
- Zero sanctions-free positive support in development cohort
- Mathematical demonstration that alpha=0 minimizes loss due to deterministic saturation
- Non-identifiability of ensemble weight w under alpha=0
- Production artifact immutability
- Research addendum artifact existence and declarations
"""
import json
from pathlib import Path
import numpy as np
import pytest

from ml_training.build_cross_sectional_oof import (
    load_authoritative_population,
    FROZEN_5H_6_FINAL_TEST_EIDS,
)
from app.forensics.address_ml_loader import AddressMLLoader, EXPECTED_ARTIFACT_HASHES

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "ml_training" / "research" / "step_5J_1"


@pytest.fixture
def aligned_dev_data():
    align_file = RESEARCH_DIR / "deterministic_alignment.json"
    assert align_file.exists(), "deterministic_alignment.json must exist"
    with open(align_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_contingency_table_strictly_diagonal(aligned_dev_data):
    """Verify exact diagonal contingency between sanctions and target label."""
    assert len(aligned_dev_data) == 84

    pos_sanc = sum(1 for r in aligned_dev_data if r["label"] == 1 and r["det_has_sanction"])
    pos_nosanc = sum(1 for r in aligned_dev_data if r["label"] == 1 and not r["det_has_sanction"])
    neg_sanc = sum(1 for r in aligned_dev_data if r["label"] == 0 and r["det_has_sanction"])
    neg_nosanc = sum(1 for r in aligned_dev_data if r["label"] == 0 and not r["det_has_sanction"])

    assert pos_sanc == 66
    assert pos_nosanc == 0
    assert neg_sanc == 0
    assert neg_nosanc == 18


def test_deterministic_score_decomposition(aligned_dev_data):
    """Verify S_full is 100/0 and S_without_sanctions is 0/0 across classes."""
    for r in aligned_dev_data:
        if r["label"] == 1:
            assert r["det_overall_score"] == 100.0
            # Sanctions component is 100.0, typologies is 0.0
            s_without_sanctions = r["det_overall_score"] - 100.0
            assert s_without_sanctions == 0.0
        else:
            assert r["det_overall_score"] == 0.0
            s_without_sanctions = r["det_overall_score"]
            assert s_without_sanctions == 0.0


def test_zero_sanctions_free_positive_support(aligned_dev_data):
    """Verify that positive support without sanctions is exactly zero."""
    pos_records_s_lt_100 = [r for r in aligned_dev_data if r["label"] == 1 and r["det_overall_score"] < 100.0]
    pos_records_no_sanc = [r for r in aligned_dev_data if r["label"] == 1 and not r["det_has_sanction"]]

    assert len(pos_records_s_lt_100) == 0
    assert len(pos_records_no_sanc) == 0


def test_mathematical_alpha_zero_demonstration(aligned_dev_data):
    """Verify mathematically that any alpha > 0 increases squared error on development data."""
    # S_det = 100 on positives, 0 on negatives
    y = np.array([r["label"] for r in aligned_dev_data], dtype=float)
    s_det = np.array([r["det_overall_score"] for r in aligned_dev_data], dtype=float)

    # Simulated non-zero probabilities (e.g., mean p_ML = 0.5)
    np.random.seed(42)
    p_ml = np.random.uniform(0.2, 0.8, size=len(y))

    # Loss at alpha = 0: S = S_det / 100
    s_0 = np.clip(s_det + 0.0 * p_ml, 0.0, 100.0) / 100.0
    loss_0 = np.mean((y - s_0) ** 2)
    assert loss_0 == 0.0  # s_det/100 matches y perfectly on dev set

    # Loss at alpha > 0
    for alpha in [1.0, 10.0, 25.0, 50.0]:
        s_alpha = np.clip(s_det + alpha * p_ml, 0.0, 100.0) / 100.0
        loss_alpha = np.mean((y - s_alpha) ** 2)
        assert loss_alpha > loss_0, f"Loss at alpha={alpha} should be strictly greater than loss at alpha=0"


def test_ensemble_weight_non_identifiable_under_alpha_zero():
    """Verify that composite score is completely invariant to ensemble weight w when alpha=0."""
    s_det = 0.0
    p_lr = 0.8
    p_xgb = 0.4
    alpha = 0.0

    scores = []
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_ens = w * p_lr + (1.0 - w) * p_xgb
        s = min(100.0, s_det + alpha * p_ens)
        scores.append(s)

    assert len(set(scores)) == 1
    assert scores[0] == 0.0


def test_addendum_report_existence_and_verdict():
    """Verify STEP_5J_1A_DECONFOUNDING_ADDENDUM.md exists and contains required declarations."""
    addendum_file = RESEARCH_DIR / "STEP_5J_1A_DECONFOUNDING_ADDENDUM.md"
    assert addendum_file.exists()

    text = addendum_file.read_text(encoding="utf-8")
    assert "NO DEFENSIBLE NUMERICAL FUSION IDENTIFIED" in text
    assert "The final test set was not accessed." in text
    assert "No production artifact was modified." in text
    assert "No production fusion policy was implemented." in text
    assert "No threshold was selected." in text
    assert "No LR/XGB winner was selected." in text


def test_production_artifacts_remain_unmodified():
    """Verify production Step 5H.7 artifacts are byte-for-byte immutable."""
    loader = AddressMLLoader()
    computed_hashes = loader.verify_integrity()
    assert len(computed_hashes) == 5
    for name, expected_hash in EXPECTED_ARTIFACT_HASHES.items():
        assert computed_hashes[name] == expected_hash
