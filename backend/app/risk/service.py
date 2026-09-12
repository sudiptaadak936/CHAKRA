"""CHAKRA Step 5: Risk Fusion Service."""
from __future__ import annotations
import logging
import uuid
import json
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg

from app.schemas.chain import Chain
from app.schemas.alert import SanctionedAddressHit, RiskScoreThresholdExceeded
from app.schemas.risk_scoring import (
    RiskAssessmentResult, 
    ModelStatus, 
    DeterministicSignals, 
    SHAPExplanation, 
    GNNExplanation
)
from app.forensics.typology_detector import TypologyDetection

from app.risk.repository import RiskRepository
from app.risk.features import FeatureExtractor
from app.risk.signals import SignalAdapter

from app.risk.ml.xgboost_model import XGBoostRiskModel
from app.risk.ml.isolation_forest import IsolationForestModel
from app.risk.ml.graph_model import GraphRiskModel
from app.services.alert_service import AlertService

logger = logging.getLogger(__name__)

# Configurable prototype fusion coefficients
CONFIG = {
    "weights": {
        "rule_engine": 0.40,
        "xgboost": 0.40,
        "isolation_forest": 0.10,
        "graph_ml": 0.10
    },
    "alert_threshold": 80.0
}

class RiskService:
    """Orchestrates end-to-end Step 5 risk scoring."""

    def __init__(self, pool: asyncpg.Pool, model_dir: str = "models/"):
        self.pool = pool
        self.repo = RiskRepository(pool)
        self.feature_extractor = FeatureExtractor(self.repo)
        
        # Load ML wrappers
        self.xgb_model = XGBoostRiskModel(os.path.join(model_dir, "xgboost"))
        self.iso_model = IsolationForestModel(os.path.join(model_dir, "isolation_forest"))
        self.gnn_model = GraphRiskModel(os.path.join(model_dir, "graph_ml"))
        
        self.alert_service = AlertService()

    async def evaluate_risk(
        self,
        target_address: str,
        chain: str,
        network: str,
        sanctions_hits: List[SanctionedAddressHit],
        typologies: List[TypologyDetection]
    ) -> RiskAssessmentResult:
        """Evaluate and fuse all risk signals."""
        analysis_id = str(uuid.uuid4())
        
        # 1. Deterministic Signals
        signals = SignalAdapter.evaluate_signals(target_address, chain, network, sanctions_hits, typologies)
        
        # 2. Extract Features
        features = await self.feature_extractor.extract_features(chain, target_address)
        feature_dict = {
            "degree": features.degree.value if features.degree.available else None,
            "transaction_volume": features.transaction_volume.value if features.transaction_volume.available else None,
            "hop_distance_to_mixer": features.hop_distance_to_mixer.value if features.hop_distance_to_mixer.available else None,
            "time_since_first_activity": features.time_since_first_activity.value if features.time_since_first_activity.available else None,
            "cluster_size": features.cluster_size.value if features.cluster_size.available else None,
            "amount_retention_ratio": features.amount_retention_ratio.value if features.amount_retention_ratio.available else None,
            "inter_hop_velocity": features.inter_hop_velocity.value if features.inter_hop_velocity.available else None,
        }

        # 3. Run ML Models
        xgb_score, shap_exp = self.xgb_model.predict(feature_dict)
        iso_score = self.iso_model.predict(feature_dict)
        gnn_score, gnn_exp = self.gnn_model.predict(feature_dict)

        # 4. Fusion
        # Start with rule engine mapped to 0-100 scale
        rule_score = 100.0 if signals.rule_engine_signal.state == "TRUE" else 0.0
        
        components_present = 0.0
        fused_score = 0.0
        
        fused_score += rule_score * CONFIG["weights"]["rule_engine"]
        components_present += CONFIG["weights"]["rule_engine"]
        
        if xgb_score is not None:
            fused_score += xgb_score * CONFIG["weights"]["xgboost"]
            components_present += CONFIG["weights"]["xgboost"]
            
        if iso_score is not None:
            fused_score += iso_score * CONFIG["weights"]["isolation_forest"]
            components_present += CONFIG["weights"]["isolation_forest"]
            
        if gnn_score is not None:
            fused_score += gnn_score * CONFIG["weights"]["graph_ml"]
            components_present += CONFIG["weights"]["graph_ml"]

        # Normalize score if some components are missing (e.g. no models trained)
        if components_present > 0:
            final_score = fused_score / components_present
        else:
            final_score = rule_score
            
        final_score = max(0.0, min(100.0, round(final_score, 2)))

        category = "NEUTRAL"
        if final_score >= 80.0:
            category = "CRITICAL"
        elif final_score >= 60.0:
            category = "HIGH"
        elif final_score >= 35.0:
            category = "MEDIUM"
        elif final_score >= 10.0:
            category = "LOW"

        # Evidence collection
        ev_ids = []
        if signals.known_bad_address_hit.evidence_id: ev_ids.append(signals.known_bad_address_hit.evidence_id)
        if signals.mixer_signal.evidence_id: ev_ids.append(signals.mixer_signal.evidence_id)
        if signals.cross_chain_signal.evidence_id: ev_ids.append(signals.cross_chain_signal.evidence_id)
        if signals.temporal_signal.evidence_id: ev_ids.append(signals.temporal_signal.evidence_id)

        # Explainability mapping
        shap_explanation = None
        if shap_exp:
            shap_explanation = SHAPExplanation(**shap_exp)
            
        gnn_explanation = None
        if gnn_exp:
            gnn_explanation = GNNExplanation(**gnn_exp)

        result = RiskAssessmentResult(
            analysis_id=analysis_id,
            target_address=target_address,
            chain=chain,
            risk_score=final_score,
            risk_category=category,
            is_prototype_fusion=True,
            xgboost_score=xgb_score,
            isolation_forest_score=iso_score,
            graph_ml_score=gnn_score,
            signals=signals,
            features=features,
            model_status=ModelStatus(
                xgboost_trained=self.xgb_model.is_trained(),
                isolation_forest_trained=self.iso_model.is_trained(),
                graph_ml_trained=self.gnn_model.is_trained(),
                xgboost_version=self.xgb_model.metadata.get("version"),
                isolation_forest_version=self.iso_model.metadata.get("version"),
                graph_ml_version=self.gnn_model.metadata.get("version"),
            ),
            shap_explanation=shap_explanation,
            gnn_explanation=gnn_explanation,
            evidence_ids=list(set(ev_ids))
        )
        
        # 5. Alert Integration
        if final_score >= CONFIG["alert_threshold"]:
            alert_contract = RiskScoreThresholdExceeded(
                chain=Chain(chain.lower()),
                address=target_address,
                overall_score=final_score,
                risk_level=category,
                threshold_crossed=CONFIG["alert_threshold"],
                risk_score_record_id=analysis_id,
                evidence_ids=result.evidence_ids,
                source="step5_risk_engine",
                reason=f"Fused risk score {final_score} exceeded threshold."
            )
            # This triggers the Redis stream in background
            await self.alert_service.generate_risk_threshold_alert(alert_contract)
            
        # 6. Persistence
        await self._persist_result(result)
        
        return result

    async def _persist_result(self, result: RiskAssessmentResult):
        query = """
            INSERT INTO risk_assessments (
                analysis_id, target_address, chain, risk_score, risk_category,
                is_prototype_fusion, model_status, signal_status, features,
                explanations, evidence_ids, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    result.analysis_id,
                    result.target_address,
                    result.chain,
                    result.risk_score,
                    result.risk_category,
                    result.is_prototype_fusion,
                    result.model_status.model_dump_json(),
                    result.signals.model_dump_json(),
                    result.features.model_dump_json(),
                    json.dumps({
                        "shap": result.shap_explanation.model_dump() if result.shap_explanation else None,
                        "gnn": result.gnn_explanation.model_dump() if result.gnn_explanation else None
                    }),
                    json.dumps(result.evidence_ids),
                    result.timestamp
                )
        except asyncpg.UndefinedTableError:
            # During some testing environments, migration might not have run
            logger.warning("Table risk_assessments does not exist, skipping persistence.")
        except Exception as e:
            logger.error(f"Failed to persist risk result: {e}")

import os
