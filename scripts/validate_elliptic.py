#!/usr/bin/env python3
"""
CHAKRA Step 0.5 — Elliptic Bitcoin Transaction Dataset Validation Script

Validates the three Elliptic dataset files, verifies cross-file relationships,
checks schema and class labels, and writes validation metadata.

Expected files:
    data/elliptic/raw/elliptic_txs_features.csv   — node features (203 columns)
    data/elliptic/raw/elliptic_txs_classes.csv    — ground-truth labels
    data/elliptic/raw/elliptic_txs_edgelist.csv   — directed transaction graph

IMPORTANT:
    Raw files are never modified.
    No ML model is trained.
    No processed derivatives replicate the raw data in full.
    Processed output contains only aggregated statistics and metadata.

Elliptic Dataset Reference:
    Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph
    Neural Networks for Financial Forensics", KDD 2019.
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    License: CC BY 4.0 (attribution required)

Usage:
    python scripts/validate_elliptic.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "elliptic" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "elliptic" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "elliptic" / "metadata"

FILES = {
    "features": RAW_DIR / "elliptic_txs_features.csv",
    "classes": RAW_DIR / "elliptic_txs_classes.csv",
    "edgelist": RAW_DIR / "elliptic_txs_edgelist.csv",
}

# Expected class labels in elliptic_txs_classes.csv
EXPECTED_CLASS_LABELS = {"1", "2", "unknown"}

# The first column of features is the transaction ID
FEATURES_TX_ID_COL = 0   # index position (no header in features file)
CLASSES_TX_ID_COL = "txId"
CLASSES_LABEL_COL = "class"
EDGELIST_SRC_COL = "txId1"
EDGELIST_DST_COL = "txId2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, label: str) -> dict:
    """Return basic file metadata; does not load data."""
    result = {"file": path.name, "exists": False, "readable": False}
    if not path.exists():
        result["error"] = "File not found"
        return result
    result["exists"] = True
    result["size_bytes"] = path.stat().st_size
    try:
        # Just open and read one byte to confirm readability
        with open(path, "rb") as f:
            f.read(1)
        result["readable"] = True
    except OSError as exc:
        result["error"] = str(exc)
    return result


def null_report(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    cols_with_nulls = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    return {
        "total_null_cells": int(null_counts.sum()),
        "columns_with_nulls": cols_with_nulls,
    }


def main() -> int:
    print("=" * 60)
    print("  CHAKRA — Elliptic Dataset Validation")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    validation_errors = []
    validation_warnings = []
    file_meta = {}

    # ------------------------------------------------------------------
    # Step 1 — File existence and readability
    # ------------------------------------------------------------------
    print("\n[1] Checking file existence and readability...")
    for key, path in FILES.items():
        info = check_file(path, key)
        file_meta[key] = info
        status = "OK" if info["readable"] else "FAIL"
        size_str = f"{info.get('size_bytes', 0):,} bytes" if info.get("size_bytes") else "N/A"
        print(f"  [{status}] {path.name}  ({size_str})")
        if not info["readable"]:
            validation_errors.append(f"{key}: {info.get('error', 'not readable')}")

    if validation_errors:
        print("\n[FATAL] One or more required files are missing or unreadable.")
        print("Aborting.")
        return 1

    # ------------------------------------------------------------------
    # Step 2 — SHA-256 per file
    # ------------------------------------------------------------------
    print("\n[2] Computing SHA-256 hashes...")
    for key, path in FILES.items():
        digest = sha256_file(path)
        file_meta[key]["sha256"] = digest
        print(f"  {path.name}: {digest}")

    # ------------------------------------------------------------------
    # Step 3 — Load and validate features file
    # ------------------------------------------------------------------
    print("\n[3] Loading features file (this may take a moment for 690 MB)...")
    try:
        # The features file has no header row per the Elliptic dataset spec.
        df_features = pd.read_csv(FILES["features"], header=None)
        n_rows_feat, n_cols_feat = df_features.shape
        print(f"  Rows: {n_rows_feat:,}  |  Columns: {n_cols_feat}")
        print(f"  TX ID column (col 0): unique IDs = {df_features[0].nunique():,}")

        # Expected: 203 columns (1 txId + 1 time_step + 93 local + 72 aggregated + 36 aggregated)
        # Actual column count varies by release; we accept 166+ columns.
        if n_cols_feat < 100:
            validation_warnings.append(
                f"features: only {n_cols_feat} columns — expected 166–203 for Elliptic dataset."
            )

        null_feat = null_report(df_features)
        file_meta["features"].update({
            "row_count": n_rows_feat,
            "column_count": n_cols_feat,
            "unique_tx_ids": int(df_features[0].nunique()),
            "duplicate_rows": int(df_features.duplicated().sum()),
            "nulls": null_feat,
        })

        if null_feat["total_null_cells"] > 0:
            validation_warnings.append(
                f"features: {null_feat['total_null_cells']} null cells detected."
            )
    except Exception as exc:
        validation_errors.append(f"features: failed to load — {exc}")
        df_features = None
        print(f"  [ERROR] {exc}")

    # ------------------------------------------------------------------
    # Step 4 — Load and validate classes file
    # ------------------------------------------------------------------
    print("\n[4] Loading classes file...")
    try:
        df_classes = pd.read_csv(FILES["classes"])
        n_rows_cls = len(df_classes)
        print(f"  Rows: {n_rows_cls:,}  |  Columns: {list(df_classes.columns)}")

        # Validate column presence
        for req_col in [CLASSES_TX_ID_COL, CLASSES_LABEL_COL]:
            if req_col not in df_classes.columns:
                validation_errors.append(f"classes: missing required column '{req_col}'")

        actual_labels = set(df_classes[CLASSES_LABEL_COL].astype(str).unique()) if CLASSES_LABEL_COL in df_classes.columns else set()
        unexpected_labels = actual_labels - EXPECTED_CLASS_LABELS
        print(f"  Class labels found: {actual_labels}")
        if unexpected_labels:
            validation_warnings.append(f"classes: unexpected labels: {unexpected_labels}")

        label_counts = df_classes[CLASSES_LABEL_COL].value_counts().to_dict() if CLASSES_LABEL_COL in df_classes.columns else {}
        null_cls = null_report(df_classes)
        dup_cls = int(df_classes.duplicated(subset=[CLASSES_TX_ID_COL]).sum()) if CLASSES_TX_ID_COL in df_classes.columns else 0

        file_meta["classes"].update({
            "row_count": n_rows_cls,
            "columns": list(df_classes.columns),
            "unique_tx_ids": int(df_classes[CLASSES_TX_ID_COL].nunique()) if CLASSES_TX_ID_COL in df_classes.columns else None,
            "duplicate_tx_ids": dup_cls,
            "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "class_labels_found": sorted(actual_labels),
            "class_labels_expected": sorted(EXPECTED_CLASS_LABELS),
            "nulls": null_cls,
        })

        if dup_cls > 0:
            validation_warnings.append(f"classes: {dup_cls} duplicate txId entries.")
    except Exception as exc:
        validation_errors.append(f"classes: failed to load — {exc}")
        df_classes = None
        print(f"  [ERROR] {exc}")

    # ------------------------------------------------------------------
    # Step 5 — Load and validate edgelist file
    # ------------------------------------------------------------------
    print("\n[5] Loading edgelist file...")
    try:
        df_edges = pd.read_csv(FILES["edgelist"])
        n_rows_edge = len(df_edges)
        print(f"  Rows: {n_rows_edge:,}  |  Columns: {list(df_edges.columns)}")

        for req_col in [EDGELIST_SRC_COL, EDGELIST_DST_COL]:
            if req_col not in df_edges.columns:
                validation_errors.append(f"edgelist: missing required column '{req_col}'")

        null_edge = null_report(df_edges)
        dup_edge = int(df_edges.duplicated().sum())

        file_meta["edgelist"].update({
            "row_count": n_rows_edge,
            "columns": list(df_edges.columns),
            "duplicate_edges": dup_edge,
            "nulls": null_edge,
        })

        if dup_edge > 0:
            validation_warnings.append(f"edgelist: {dup_edge} duplicate edge rows.")
    except Exception as exc:
        validation_errors.append(f"edgelist: failed to load — {exc}")
        df_edges = None
        print(f"  [ERROR] {exc}")

    # ------------------------------------------------------------------
    # Step 6 — Cross-file transaction ID consistency
    # ------------------------------------------------------------------
    cross_file_validation = {}
    print("\n[6] Cross-file transaction ID consistency...")

    if df_features is not None and df_classes is not None:
        feat_ids = set(df_features[0].astype(str))
        cls_ids = set(df_classes[CLASSES_TX_ID_COL].astype(str))

        in_cls_not_feat = cls_ids - feat_ids
        in_feat_not_cls = feat_ids - cls_ids

        print(f"  Feature TX IDs: {len(feat_ids):,}")
        print(f"  Class TX IDs:   {len(cls_ids):,}")
        print(f"  In classes but not features: {len(in_cls_not_feat)}")
        print(f"  In features but not classes: {len(in_feat_not_cls)}")

        cross_file_validation["features_vs_classes"] = {
            "feature_tx_count": len(feat_ids),
            "class_tx_count": len(cls_ids),
            "in_classes_not_in_features": len(in_cls_not_feat),
            "in_features_not_in_classes": len(in_feat_not_cls),
        }

        if in_cls_not_feat:
            validation_warnings.append(
                f"{len(in_cls_not_feat)} TX IDs in classes not found in features."
            )

    if df_features is not None and df_edges is not None:
        feat_ids = set(df_features[0].astype(str))
        edge_src = set(df_edges[EDGELIST_SRC_COL].astype(str))
        edge_dst = set(df_edges[EDGELIST_DST_COL].astype(str))
        edge_all = edge_src | edge_dst

        dangling = edge_all - feat_ids
        print(f"  Edge TX IDs not in features: {len(dangling)}")

        cross_file_validation["edgelist_vs_features"] = {
            "unique_edge_nodes": len(edge_all),
            "edge_nodes_not_in_features": len(dangling),
        }

        if dangling:
            validation_warnings.append(
                f"{len(dangling)} edge TX IDs not present in features file."
            )

    # ------------------------------------------------------------------
    # Step 7 — Write metadata
    # ------------------------------------------------------------------
    print("\n[7] Writing metadata...")

    validation_report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not validation_errors else "fail",
        "errors": validation_errors,
        "warnings": validation_warnings,
        "files": file_meta,
        "cross_file_validation": cross_file_validation,
    }

    report_path = METADATA_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2, default=str)
    print(f"  Written: {report_path.relative_to(PROJECT_ROOT)}")

    metadata = {
        "dataset": "Elliptic Bitcoin Transaction Dataset",
        "authors": "Weber, M., Domeniconi, G., Chen, J., Weidele, D.K.I., Bellei, C., Robinson, T., Leiserson, C.E.",
        "paper": "Anti-Money Laundering in Bitcoin: Experimenting with Graph Neural Networks for Financial Forensics",
        "venue": "KDD 2019 Workshop on Anomaly Detection in Finance",
        "source_url": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "description": (
            "203 timestep Bitcoin transaction graph with ground-truth licit/illicit labels. "
            "Node features are anonymized. Class 1 = illicit, Class 2 = licit, unknown = unlabelled."
        ),
        "files": {k: {
            "filename": v.name,
            "sha256": file_meta[k].get("sha256"),
            "size_bytes": file_meta[k].get("size_bytes"),
            "row_count": file_meta[k].get("row_count"),
        } for k, v in FILES.items()},
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "processing_note": (
            "Validation only. No raw data was copied, modified, or redistributed. "
            "No ML model was trained."
        ),
    }

    meta_path = METADATA_DIR / "elliptic_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Written: {meta_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if validation_errors:
        print(f"  Elliptic Validation: FAIL ({len(validation_errors)} error(s))")
        for e in validation_errors:
            print(f"    [ERROR] {e}")
    else:
        print("  Elliptic Validation: PASS")

    if validation_warnings:
        print(f"  Warnings: {len(validation_warnings)}")
        for w in validation_warnings:
            print(f"    [WARN] {w}")
    print("=" * 60)

    return 0 if not validation_errors else 1


if __name__ == "__main__":
    sys.exit(main())
