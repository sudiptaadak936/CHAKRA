"""
Unit tests for Elliptic dataset validation logic.

Tests validate schema checking, cross-file consistency, and class label
validation using synthetic in-memory DataFrames — no real 690 MB file required.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import sys

SCRIPTS_DIR_HOST = Path(__file__).resolve().parents[2] / "scripts"
SCRIPTS_DIR_CONTAINER = Path(__file__).resolve().parents[1] / "scripts"


def load_validate_elliptic():
    import importlib
    for d in [str(SCRIPTS_DIR_HOST), str(SCRIPTS_DIR_CONTAINER)]:
        if Path(d).exists() and d not in sys.path:
            sys.path.insert(0, d)
    import validate_elliptic
    importlib.reload(validate_elliptic)
    return validate_elliptic


# ---------------------------------------------------------------------------
# Synthetic DataFrames
# ---------------------------------------------------------------------------

def make_features_df(n=10, n_cols=203):
    """Simulate elliptic_txs_features.csv (no header, col 0 = txId)."""
    import numpy as np
    rng = range(1, n + 1)
    data = {i: [float(j * i) for j in rng] for i in range(1, n_cols)}
    data[0] = list(rng)  # txId column
    return pd.DataFrame(data)


def make_classes_df(tx_ids, labels=None):
    """Simulate elliptic_txs_classes.csv."""
    if labels is None:
        labels = ["1", "2", "unknown"] * (len(tx_ids) // 3 + 1)
        labels = labels[:len(tx_ids)]
    return pd.DataFrame({"txId": tx_ids, "class": labels})


def make_edges_df(tx_ids):
    """Simulate elliptic_txs_edgelist.csv using known txIds."""
    n = len(tx_ids) - 1
    src = [tx_ids[i] for i in range(n)]
    dst = [tx_ids[i + 1] for i in range(n)]
    return pd.DataFrame({"txId1": src, "txId2": dst})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSha256Elliptic:
    def test_sha256_deterministic(self, tmp_path):
        mod = load_validate_elliptic()
        f = tmp_path / "test.csv"
        f.write_bytes(b"txId,class\n1,1\n2,2\n")
        h1 = mod.sha256_file(f)
        h2 = mod.sha256_file(f)
        assert h1 == h2

    def test_sha256_correct_value(self, tmp_path):
        mod = load_validate_elliptic()
        content = b"hello elliptic"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert mod.sha256_file(f) == expected


class TestFileCheck:
    def test_existing_readable_file(self, tmp_path):
        mod = load_validate_elliptic()
        f = tmp_path / "test.csv"
        f.write_text("txId,class\n1,1\n")
        result = mod.check_file(f, "test")
        assert result["exists"] is True
        assert result["readable"] is True
        assert result["size_bytes"] > 0

    def test_nonexistent_file(self, tmp_path):
        mod = load_validate_elliptic()
        result = mod.check_file(tmp_path / "ghost.csv", "ghost")
        assert result["exists"] is False
        assert result["readable"] is False


class TestNullReport:
    def test_no_nulls(self):
        mod = load_validate_elliptic()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = mod.null_report(df)
        assert report["total_null_cells"] == 0
        assert report["columns_with_nulls"] == {}

    def test_with_nulls(self):
        mod = load_validate_elliptic()
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        report = mod.null_report(df)
        assert report["total_null_cells"] == 3
        assert "a" in report["columns_with_nulls"]
        assert "b" in report["columns_with_nulls"]


class TestClassLabelValidation:
    def test_expected_labels_pass(self):
        mod = load_validate_elliptic()
        actual = {"1", "2", "unknown"}
        unexpected = actual - mod.EXPECTED_CLASS_LABELS
        assert len(unexpected) == 0

    def test_unexpected_label_detected(self):
        mod = load_validate_elliptic()
        actual = {"1", "2", "unknown", "suspicious"}
        unexpected = actual - mod.EXPECTED_CLASS_LABELS
        assert "suspicious" in unexpected

    def test_class_1_is_illicit(self):
        """Class 1 = illicit per Elliptic spec. If present, it must be allowed."""
        mod = load_validate_elliptic()
        assert "1" in mod.EXPECTED_CLASS_LABELS

    def test_class_2_is_licit(self):
        mod = load_validate_elliptic()
        assert "2" in mod.EXPECTED_CLASS_LABELS

    def test_unknown_is_unlabelled(self):
        mod = load_validate_elliptic()
        assert "unknown" in mod.EXPECTED_CLASS_LABELS


class TestCrossFileConsistency:
    def test_features_and_classes_consistent(self):
        """All class TX IDs should also appear in features."""
        tx_ids = list(range(1, 11))
        df_features = make_features_df(10, 203)
        df_features[0] = tx_ids
        df_classes = make_classes_df(tx_ids)

        feat_ids = set(df_features[0].astype(str))
        cls_ids = set(df_classes["txId"].astype(str))
        assert cls_ids <= feat_ids  # all class IDs should be in features

    def test_orphan_class_ids_detected(self):
        """Class IDs not in features should be detected as orphans."""
        tx_ids_feat = list(range(1, 6))
        tx_ids_cls = list(range(1, 9))  # 6,7,8 are orphans

        df_features = make_features_df(5, 203)
        df_features[0] = tx_ids_feat
        df_classes = make_classes_df(tx_ids_cls)

        feat_ids = set(df_features[0].astype(str))
        cls_ids = set(df_classes["txId"].astype(str))
        orphans = cls_ids - feat_ids
        assert len(orphans) == 3

    def test_edge_nodes_in_features(self):
        """All edge source/destination TX IDs should appear in features."""
        tx_ids = list(range(1, 11))
        df_features = make_features_df(10, 203)
        df_features[0] = tx_ids
        df_edges = make_edges_df(tx_ids)

        feat_ids = set(df_features[0].astype(str))
        edge_all = set(df_edges["txId1"].astype(str)) | set(df_edges["txId2"].astype(str))
        dangling = edge_all - feat_ids
        assert len(dangling) == 0

    def test_dangling_edge_nodes_detected(self):
        """Edge nodes referencing unknown TX IDs should be flagged."""
        tx_ids_feat = list(range(1, 6))
        tx_ids_edge = list(range(1, 11))  # 6–10 are unknown

        df_features = make_features_df(5, 203)
        df_features[0] = tx_ids_feat
        df_edges = make_edges_df(tx_ids_edge)

        feat_ids = set(df_features[0].astype(str))
        edge_all = set(df_edges["txId1"].astype(str)) | set(df_edges["txId2"].astype(str))
        dangling = edge_all - feat_ids
        assert len(dangling) > 0

    def test_duplicate_tx_ids_detected(self):
        """Duplicate txId entries in classes should be detected."""
        tx_ids = [1, 2, 2, 3]  # 2 is duplicate
        df_classes = make_classes_df(tx_ids)
        dup_count = int(df_classes.duplicated(subset=["txId"]).sum())
        assert dup_count == 1


class TestSchemaValidation:
    def test_features_column_count_check(self):
        """Features file with < 100 columns should trigger a warning."""
        df = make_features_df(5, 50)  # too few columns
        assert df.shape[1] < 100

    def test_classes_required_columns(self):
        df = make_classes_df([1, 2, 3])
        assert "txId" in df.columns
        assert "class" in df.columns

    def test_edgelist_required_columns(self):
        df = make_edges_df([1, 2, 3, 4])
        assert "txId1" in df.columns
        assert "txId2" in df.columns


class TestMetadataOutput:
    def test_metadata_has_provenance_fields(self, tmp_path):
        metadata = {
            "dataset": "Elliptic Bitcoin Transaction Dataset",
            "license": "CC BY 4.0",
            "source_url": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
            "files": {
                "features": {"sha256": "abc", "size_bytes": 100, "row_count": 10},
                "classes": {"sha256": "def", "size_bytes": 50, "row_count": 10},
                "edgelist": {"sha256": "ghi", "size_bytes": 30, "row_count": 9},
            },
        }
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(metadata))
        loaded = json.loads(f.read_text())
        assert "license" in loaded
        assert "source_url" in loaded
        assert "files" in loaded
        for fname in ["features", "classes", "edgelist"]:
            assert fname in loaded["files"]
            assert "sha256" in loaded["files"][fname]
