"""CHAKRA Step 5I: Production Address-Level ML Inference Service.

Consumes the frozen Step 5H.7 models and sigmoid recalibration artifacts to provide
deterministic, read-only, calibrated analytical signals.

Key Invariants:
- Purely analytical and read-only.
- Zero runtime learning or parameter fitting.
- Fails closed deterministically on invalid schema, bad dimensions, non-finite values,
  or artifact integrity violations.
- Never creates alerts, assigns culpability, or classifies criminality/intent.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Union

import numpy as np

from app.forensics.address_ml_loader import AddressMLLoader, LoadedAddressMLArtifacts
from app.forensics.address_ml_preprocessor import FrozenAddressPreprocessor
from app.schemas.ml_features import FEATURE_SCHEMA_VERSION, MLFeatureRecord, MLFeatureValues
from app.schemas.ml_inference import (
    AddressMLInferenceRequest,
    AddressMLInferenceResult,
    FeatureContractError,
    InferenceStatus,
    MLInferenceError,
)

logger = logging.getLogger(__name__)


class AddressMLInferenceService:
    """Production inference engine consuming frozen Step 5H.7 artifacts."""

    def __init__(self, loader: Optional[AddressMLLoader] = None):
        self._loader = loader or AddressMLLoader()
        self._artifacts: Optional[LoadedAddressMLArtifacts] = None
        self._preprocessor: Optional[FrozenAddressPreprocessor] = None

    def _ensure_initialized(self) -> None:
        """Ensure artifacts and preprocessor are loaded and verified."""
        if self._artifacts is None or self._preprocessor is None:
            self._artifacts = self._loader.load_artifacts()
            self._preprocessor = FrozenAddressPreprocessor(self._artifacts.preprocessor_params)

    @property
    def artifacts(self) -> LoadedAddressMLArtifacts:
        """Access loaded artifacts (ensuring initialization)."""
        self._ensure_initialized()
        return self._artifacts

    @property
    def provenance(self):
        """Access the cryptographic provenance of currently loaded artifacts."""
        self._ensure_initialized()
        return self._artifacts.provenance

    def infer(
        self,
        request: AddressMLInferenceRequest,
    ) -> AddressMLInferenceResult:
        """Execute deterministic inference for a validated request.
        
        Fails closed on any contract violation or computation error.
        """
        self._ensure_initialized()

        # 1. Feature contract validation
        if request.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise FeatureContractError(
                f"Feature schema version '{request.feature_schema_version}' != '{FEATURE_SCHEMA_VERSION}'"
            )

        vector = request.get_canonical_vector()

        # 2. Deterministic preprocessing
        X_lr = self._preprocessor.transform_for_linear(vector)
        X_xgb = self._preprocessor.transform_for_trees(vector)

        # 3. Model inference (Raw Probabilities)
        try:
            lr_raw_probs = self._artifacts.lr_model.predict_proba(X_lr)[:, 1]
            xgb_raw_probs = self._artifacts.xgb_model.predict_proba(X_xgb)[:, 1]
            lr_raw = float(lr_raw_probs[0])
            xgb_raw = float(xgb_raw_probs[0])
        except Exception as e:
            logger.error("Raw model inference failed: %s", str(e))
            raise MLInferenceError(f"Model prediction failure: {e}") from e

        if not (math.isfinite(lr_raw) and 0.0 <= lr_raw <= 1.0):
            raise MLInferenceError(f"LR produced non-finite/out-of-bounds raw probability: {lr_raw}")
        if not (math.isfinite(xgb_raw) and 0.0 <= xgb_raw <= 1.0):
            raise MLInferenceError(f"XGB produced non-finite/out-of-bounds raw probability: {xgb_raw}")

        # 4. Sigmoid Recalibration (Platt Scaling)
        try:
            lr_cal_probs = self._artifacts.lr_calibrator.predict_proba(np.array([[lr_raw]]))[:, 1]
            xgb_cal_probs = self._artifacts.xgb_calibrator.predict_proba(np.array([[xgb_raw]]))[:, 1]
            lr_cal = float(lr_cal_probs[0])
            xgb_cal = float(xgb_cal_probs[0])
        except Exception as e:
            logger.error("Calibrator execution failed: %s", str(e))
            raise MLInferenceError(f"Calibration transformation failure: {e}") from e

        if not (math.isfinite(lr_cal) and 0.0 < lr_cal < 1.0):
            raise MLInferenceError(f"LR calibrated probability out of bounds (0, 1): {lr_cal}")
        if not (math.isfinite(xgb_cal) and 0.0 < xgb_cal < 1.0):
            raise MLInferenceError(f"XGB calibrated probability out of bounds (0, 1): {xgb_cal}")

        # 5. Build authoritative result
        return AddressMLInferenceResult(
            target_address=request.target_address,
            chain=request.chain,
            network=request.network,
            feature_schema_version=request.feature_schema_version,
            status=InferenceStatus.SUCCESS,
            lr_raw_probability=lr_raw,
            lr_calibrated_probability=lr_cal,
            xgb_raw_probability=xgb_raw,
            xgb_calibrated_probability=xgb_cal,
            provenance=self._artifacts.provenance,
            computed_at=datetime.now(timezone.utc),
        )

    def infer_record(
        self,
        record: MLFeatureRecord,
    ) -> AddressMLInferenceResult:
        """Convenience method to run inference directly on an extracted MLFeatureRecord."""
        request = AddressMLInferenceRequest(
            target_address=record.target_address,
            chain=record.chain,
            network=record.network,
            feature_schema_version=record.feature_schema_version,
            features=record.features,
        )
        return self.infer(request)


# Global singleton instance for FastAPI dependency injection
_inference_service_instance: Optional[AddressMLInferenceService] = None


def get_address_ml_service() -> AddressMLInferenceService:
    """Dependency provider for AddressMLInferenceService."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = AddressMLInferenceService()
    return _inference_service_instance
