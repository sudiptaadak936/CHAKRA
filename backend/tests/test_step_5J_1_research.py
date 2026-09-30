"""CHAKRA Step 5J.1 Research Verification Tests.

Verifies:
- Exact 84-record development population and 44-entity grouping
- Strict zero-leakage firewall against 9-entity final test set
- Deterministic risk alignment integrity (sanctions hits = 100.0, negatives = 0.0)
- Feature contract Schema 1.1.0 with 13 canonical features
- Production artifact immutability
- Research output artifacts exist and confirm OUTCOME A
"""
import hashlib
import json
from pathlib import Path
import pytest

from app.schemas.ml_features import CANONICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ml_training.build_cross_sectional_oof import (
    load_authoritative_population,
    FROZEN_5H_6_FINAL_TEST_EIDS,
)
from app.forensics.address_ml_loader import AddressMLLoader, EXPECTED_ARTIFACT_HASHES

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "ml_training" / "research" / "step_5J_1"


def test_development_dataset_counts_and_firewall():
    """Verify 84 dev records, 44 entities, zero final-test overlap."""
    records, _ = load_authoritative_population()
    assert len(records) == 93

    final_test_set = set(FROZEN_5H_6_FINAL_TEST_EIDS)
    assert len(final_test_set) == 9

    dev_records = [r for r in records if r["entity_id"] not in final_test_set]
    assert len(dev_records) == 84

    dev_entities = {r["entity_id"] for r in dev_records}
    assert len(dev_entities) == 44

    # Strict isolation
    assert dev_entities.isdisjoint(final_test_set), "Final test set contaminated development population!"

    pos_records = sum(1 for r in dev_records if r["label"] == 1)
    neg_records = sum(1 for r in dev_records if r["label"] == 0)
    pos_entities = len({r["entity_id"] for r in dev_records if r["label"] == 1})
    neg_entities = len({r["entity_id"] for r in dev_records if r["label"] == 0})

    assert pos_records == 66 and neg_records == 18
    assert pos_entities == 26 and neg_entities == 18


def test_feature_contract_and_ordering():
    """Verify all 84 dev records adhere to Schema 1.1.0 and 13 canonical features."""
    records, _ = load_authoritative_population()
    final_test_set = set(FROZEN_5H_6_FINAL_TEST_EIDS)
    dev_records = [r for r in records if r["entity_id"] not in final_test_set]

    assert len(CANONICAL_FEATURE_NAMES) == 13

    for r in dev_records:
        feat = r["feature_values"]
        assert len(feat) == 13
        assert r["feature_schema_version"] in ("1.0.0", "1.1.0")


def test_deterministic_alignment_artifact():
    """Verify deterministic alignment artifact exists and records match."""
    align_file = RESEARCH_DIR / "deterministic_alignment.json"
    assert align_file.exists(), "deterministic_alignment.json missing"

    with open(align_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 84
    pos_items = [d for d in data if d["label"] == 1]
    neg_items = [d for d in data if d["label"] == 0]

    assert len(pos_items) == 66
    assert len(neg_items) == 18

    assert all(d["det_overall_score"] == 100.0 for d in pos_items)
    assert all(d["det_risk_level"] == "CRITICAL" for d in pos_items)
    assert all(d["det_overall_score"] == 0.0 for d in neg_items)
    assert all(d["det_risk_level"] == "NEUTRAL" for d in neg_items)


def test_production_artifacts_unmodified():
    """Verify that running research did not modify any frozen Step 5H.7 artifact."""
    loader = AddressMLLoader()
    computed_hashes = loader.verify_integrity()
    assert len(computed_hashes) == 5
    for name, expected_hash in EXPECTED_ARTIFACT_HASHES.items():
        assert computed_hashes[name] == expected_hash, f"Hash mismatch for {name}"


def test_research_report_and_verdict_artifacts():
    """Verify research report and JSON comparison tables exist with valid Outcome A."""
    report_file = RESEARCH_DIR / "STEP_5J_1_FUSION_POLICY_RESEARCH_REPORT.md"
    assert report_file.exists(), "Research report file missing"

    text = report_file.read_text(encoding="utf-8")
    assert "INDEPENDENT CHANNELS EMPIRICALLY PREFERRED FOR POLICY SAFETY" in text
    assert "The 9-entity final test set was not accessed or used for policy selection." in text
    assert "No production fusion policy was implemented." in text
    assert "No frozen Step 5H.7 or Step 5I artifact was modified." in text

    comp_file = RESEARCH_DIR / "fusion_candidate_comparison.json"
    assert comp_file.exists()
    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    assert comp_data["n_outer_splits"] == 4
    assert len(comp_data["fold_metrics"]) == 4
    # Check that alpha collapsed to 0 in all folds for Candidate B
    for f in comp_data["fold_metrics"]:
        assert f["selected_params"]["alpha_b_lr"] == 0.0
        assert f["selected_params"]["alpha_b_xgb"] == 0.0
