"""CHAKRA Step 5J: Controlled Risk & ML Fusion Service.

Integrates the frozen Step 5 deterministic risk engine and the frozen Step 5I
address-level ML inference service into a unified, auditable analytical view.

Key Invariants:
- Preserves the independent integrity of both deterministic and ML layers.
- Strict fail-closed error handling.
- When no explicit frozen mathematical fusion policy exists, reports fusion_status=NOT_CONFIGURED.
- Never invents ad-hoc weights, never implicitly averages LR and XGB, never synthesizes
  arbitrary risk thresholds.
- Purely analytical and read-only: zero database mutation, zero alert creation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from app.forensics.risk_engine import RiskEngine
from app.forensics.address_ml_service import AddressMLInferenceService, get_address_ml_service
from app.attribution.models import VASPAddressRecord
from app.forensics.models import ChangeAddressInference
from app.forensics.typology_detector import TypologyDetection
from app.schemas.alert import SanctionedAddressHit
from app.schemas.ml_features import FEATURE_SCHEMA_VERSION, MLFeatureRecord, MLFeatureValues
from app.schemas.ml_inference import AddressMLInferenceRequest, AddressMLInferenceResult
from app.schemas.risk import RiskScoreRecord
from app.schemas.risk_fusion import FusionLayerResult, FusionStatus, RiskMLFusionRecord

logger = logging.getLogger(__name__)


class RiskMLFusionService:
    """Orchestrates deterministic risk evaluation and ML inference into a composite view."""

    def __init__(
        self,
        ml_service: Optional[AddressMLInferenceService] = None,
        fusion_policy_version: Optional[str] = None,
    ):
        self._ml_service = ml_service or get_address_ml_service()
        self._fusion_policy_version = fusion_policy_version

    def evaluate_composite(
        self,
        target_address: str,
        chain: str,
        network: str,
        # Deterministic inputs
        sanctions_hits: Optional[List[SanctionedAddressHit]] = None,
        typologies: Optional[List[TypologyDetection]] = None,
        attributions: Optional[List[VASPAddressRecord]] = None,
        change_inferences: Optional[List[ChangeAddressInference]] = None,
        cluster_contagions: Optional[List[Dict[str, Any]]] = None,
        # ML inputs
        feature_vector: Optional[Sequence[float]] = None,
        features: Optional[MLFeatureValues] = None,
        feature_schema_version: str = FEATURE_SCHEMA_VERSION,
        computed_at: Optional[datetime] = None,
    ) -> RiskMLFusionRecord:
        """Evaluate deterministic risk and ML inference for a target address.

        Fails closed if either evaluation cannot be performed.
        """
        # 1. Input sanity checks
        if not target_address or not target_address.strip():
            raise ValueError("Target address is required.")
        if not chain or not chain.strip():
            raise ValueError("Chain is required.")
        if not network or not network.strip():
            raise ValueError("Network is required.")

        ts = computed_at or datetime.now(timezone.utc)

        # 2. Evaluate Deterministic Risk (Step 5 Engine)
        deterministic_result = RiskEngine.evaluate(
            target_address=target_address,
            chain=chain,
            network=network,
            sanctions_hits=sanctions_hits,
            typologies=typologies,
            attributions=attributions,
            change_inferences=change_inferences,
            cluster_contagions=cluster_contagions,
            computed_at=ts,
        )

        # 3. Evaluate ML Inference (Step 5I Service)
        ml_request = AddressMLInferenceRequest(
            target_address=target_address,
            chain=chain,
            network=network,
            feature_schema_version=feature_schema_version,
            features=features,
            feature_vector=list(feature_vector) if feature_vector is not None else None,
        )
        ml_result = self._ml_service.infer(ml_request)

        # 4. Assemble Fusion Layer
        fusion_layer = self._resolve_fusion_layer(deterministic_result, ml_result)

        # 5. Emit Composite Result
        return RiskMLFusionRecord(
            target_address=target_address,
            chain=chain,
            network=network,
            deterministic_layer=deterministic_result,
            ml_layer=ml_result,
            fusion_layer=fusion_layer,
            computed_at=ts,
        )

    def combine_results(
        self,
        deterministic_result: RiskScoreRecord,
        ml_result: AddressMLInferenceResult,
    ) -> RiskMLFusionRecord:
        """Combine pre-computed deterministic and ML results into a composite record.

        Validates that both records correspond to the exact same target address, chain, and network.
        """
        if deterministic_result.target_address != ml_result.target_address:
            raise ValueError(
                f"Address mismatch: deterministic '{deterministic_result.target_address}' "
                f"!= ML '{ml_result.target_address}'"
            )
        if deterministic_result.chain != ml_result.chain:
            raise ValueError(
                f"Chain mismatch: deterministic '{deterministic_result.chain}' != ML '{ml_result.chain}'"
            )
        if deterministic_result.network != ml_result.network:
            raise ValueError(
                f"Network mismatch: deterministic '{deterministic_result.network}' != ML '{ml_result.network}'"
            )

        fusion_layer = self._resolve_fusion_layer(deterministic_result, ml_result)

        return RiskMLFusionRecord(
            target_address=deterministic_result.target_address,
            chain=deterministic_result.chain,
            network=deterministic_result.network,
            deterministic_layer=deterministic_result,
            ml_layer=ml_result,
            fusion_layer=fusion_layer,
            computed_at=datetime.now(timezone.utc),
        )

    def _resolve_fusion_layer(
        self,
        deterministic: RiskScoreRecord,
        ml: AddressMLInferenceResult,
    ) -> FusionLayerResult:
        """Determine fusion outcome.

        If no formal frozen fusion policy exists, returns NOT_CONFIGURED.
        Refuses to invent ad-hoc weights or average models without an explicit policy contract.
        """
        if self._fusion_policy_version is None:
            return FusionLayerResult(
                fusion_status=FusionStatus.NOT_CONFIGURED,
                fusion_policy_version=None,
                fusion_score=None,
                fusion_methodology=None,
                reason=(
                    "Mathematical fusion policy not specified in frozen repository contracts; "
                    "deterministic risk evidence and ML analytical signals are presented independently."
                ),
            )

        # Fail closed on unsupported policy versions
        return FusionLayerResult(
            fusion_status=FusionStatus.FAILED,
            fusion_policy_version=self._fusion_policy_version,
            fusion_score=None,
            fusion_methodology=None,
            reason=f"Unsupported or unauthorized fusion policy version: '{self._fusion_policy_version}'.",
        )
