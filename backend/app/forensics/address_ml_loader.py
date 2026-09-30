"""CHAKRA Step 5I: Frozen Address ML Artifact Loader.

Responsible for locating, cryptographically validating, and loading the frozen
Step 5H.7 address-level model and calibration artifacts.

Guarantees:
- Strict fail-closed integrity: missing or modified files raise ArtifactIntegrityError.
- Exact SHA-256 validation against frozen Step 5H.7 hashes.
- Never regenerates, recalibrates, retrains, or substitutes artifacts.
- Caches validated artifacts in memory for high-performance deterministic inference.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from xgboost import XGBClassifier

from app.schemas.ml_inference import (
    ArtifactIntegrityError,
    ModelProvenanceRecord,
)

logger = logging.getLogger(__name__)

# Canonical Step 5H.7 SHA-256 Hashes
EXPECTED_ARTIFACT_HASHES: Dict[str, str] = {
    "logistic_regression_final.joblib": "569bbca10c392f06b262a5c0cb7ae5209a34ecc92a852ff93ed2e576b2f86bac",
    "xgboost_final.json": "e320eb085b2d091bb202d03261cbc15683ede51070c152313cd211ba13575397",
    "final_preprocessor_params.json": "8b6a44e10e0c3b47f7c887f95cc12022aaa206db2c0d6a643cc0f72d5c60e3d5",
    "lr_calibrator.joblib": "1e78b493577a03884a4ddcaaf4488cf507f6448aaf16c9eb865f337f513a98df",
    "xgb_calibrator.joblib": "90d61a4a7cc91442e58104bb9e7fa9c57c0554df13a55b034fef4fe7ab00aad2",
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    if not filepath.exists() or not filepath.is_file():
        raise ArtifactIntegrityError(f"Artifact file not found: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LoadedAddressMLArtifacts:
    """Immutable container for loaded and validated Step 5H.7 ML artifacts."""
    lr_model: Any
    xgb_model: XGBClassifier
    preprocessor_params: Dict[str, Any]
    lr_calibrator: Any
    xgb_calibrator: Any
    provenance: ModelProvenanceRecord


class AddressMLLoader:
    """Production artifact loader with cryptographic verification."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize loader.
        
        If base_dir is None, resolves relative to backend root.
        """
        if base_dir is not None:
            self._backend_dir = Path(base_dir).resolve()
        else:
            # Resolves to `backend/` directory regardless of process CWD
            self._backend_dir = Path(__file__).resolve().parents[2]

        self._models_dir = self._backend_dir / "ml_training" / "address_model_artifacts" / "step_5H_6_final"
        self._calib_dir = self._backend_dir / "ml_training" / "calibration_artifacts" / "step_5H_6_2_2"

        self._cached_artifacts: Optional[LoadedAddressMLArtifacts] = None

    @property
    def lr_model_path(self) -> Path:
        return self._models_dir / "logistic_regression_final.joblib"

    @property
    def xgb_model_path(self) -> Path:
        return self._models_dir / "xgboost_final.json"

    @property
    def preprocessor_params_path(self) -> Path:
        return self._models_dir / "final_preprocessor_params.json"

    @property
    def lr_calibrator_path(self) -> Path:
        return self._calib_dir / "lr_calibrator.joblib"

    @property
    def xgb_calibrator_path(self) -> Path:
        return self._calib_dir / "xgb_calibrator.joblib"

    def verify_integrity(self) -> Dict[str, str]:
        """Compute and verify SHA-256 checksums of all 5 authoritative artifacts.
        
        Returns computed hashes on success, or raises ArtifactIntegrityError.
        """
        paths = {
            "logistic_regression_final.joblib": self.lr_model_path,
            "xgboost_final.json": self.xgb_model_path,
            "final_preprocessor_params.json": self.preprocessor_params_path,
            "lr_calibrator.joblib": self.lr_calibrator_path,
            "xgb_calibrator.joblib": self.xgb_calibrator_path,
        }

        computed_hashes: Dict[str, str] = {}
        for name, path in paths.items():
            if not path.exists():
                raise ArtifactIntegrityError(
                    f"Required frozen artifact '{name}' is missing at: {path}"
                )
            digest = compute_sha256(path)
            expected = EXPECTED_ARTIFACT_HASHES[name]
            if digest != expected:
                raise ArtifactIntegrityError(
                    f"Integrity violation for '{name}': expected {expected}, got {digest}"
                )
            computed_hashes[name] = digest

        return computed_hashes

    def load_artifacts(self, force_reload: bool = False) -> LoadedAddressMLArtifacts:
        """Load and return verified ML artifacts.
        
        Fails closed on any hash mismatch, missing artifact, or deserialization failure.
        """
        if self._cached_artifacts is not None and not force_reload:
            return self._cached_artifacts

        # 1. Verify cryptographic integrity before any loading
        hashes = self.verify_integrity()

        try:
            # 2. Load Logistic Regression model
            lr_model = joblib.load(self.lr_model_path)

            # 3. Load XGBoost model
            xgb_model = XGBClassifier()
            xgb_model.load_model(str(self.xgb_model_path))

            # 4. Load Preprocessor parameters
            with open(self.preprocessor_params_path, "r", encoding="utf-8") as f:
                prep_params = json.load(f)

            # 5. Load Calibrators
            lr_calibrator = joblib.load(self.lr_calibrator_path)
            xgb_calibrator = joblib.load(self.xgb_calibrator_path)

        except Exception as e:
            logger.error("Failed to deserialize verified artifacts: %s", str(e))
            raise ArtifactIntegrityError(f"Artifact deserialization error: {e}") from e

        provenance = ModelProvenanceRecord(
            freeze_identity="STEP_5H_7_FREEZE",
            feature_schema_version=prep_params.get("schema_version", "1.1.0"),
            lr_model_hash=hashes["logistic_regression_final.joblib"],
            xgb_model_hash=hashes["xgboost_final.json"],
            preprocessor_hash=hashes["final_preprocessor_params.json"],
            lr_calibrator_hash=hashes["lr_calibrator.joblib"],
            xgb_calibrator_hash=hashes["xgb_calibrator.joblib"],
        )

        artifacts = LoadedAddressMLArtifacts(
            lr_model=lr_model,
            xgb_model=xgb_model,
            preprocessor_params=prep_params,
            lr_calibrator=lr_calibrator,
            xgb_calibrator=xgb_calibrator,
            provenance=provenance,
        )

        self._cached_artifacts = artifacts
        logger.info("Successfully loaded and cryptographically verified Step 5H.7 ML artifacts.")
        return artifacts
