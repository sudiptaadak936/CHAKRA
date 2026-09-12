"""CHAKRA Step 5: Risk Scoring Schemas."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class FeatureAvailability(BaseModel):
    available: bool
    value: Optional[float] = None
    reason_if_missing: Optional[str] = None

class RiskFeatures(BaseModel):
    """Engineered features required by XGBoost and GNN."""
    degree: FeatureAvailability
    transaction_volume: FeatureAvailability
    hop_distance_to_mixer: FeatureAvailability
    time_since_first_activity: FeatureAvailability
    cluster_size: FeatureAvailability
    amount_retention_ratio: FeatureAvailability
    inter_hop_velocity: FeatureAvailability

class SignalState(BaseModel):
    state: str  # "TRUE", "FALSE", "UNKNOWN"
    evidence_id: Optional[str] = None

class DeterministicSignals(BaseModel):
    rule_engine_signal: SignalState
    known_bad_address_hit: SignalState
    mixer_signal: SignalState
    cross_chain_signal: SignalState
    temporal_signal: SignalState

class ModelStatus(BaseModel):
    xgboost_trained: bool
    isolation_forest_trained: bool
    graph_ml_trained: bool
    xgboost_version: Optional[str] = None
    isolation_forest_version: Optional[str] = None
    graph_ml_version: Optional[str] = None
    feature_schema_version: str = "1.0.0"

class SHAPExplanation(BaseModel):
    base_value: float
    feature_contributions: Dict[str, float]
    top_features: List[str]

class GNNExplanation(BaseModel):
    attention_weights_available: bool
    top_contributing_nodes: List[str]
    explanation_text: str

class RiskAssessmentResult(BaseModel):
    """Final fused Step 5 risk result schema."""
    analysis_id: str
    target_address: str
    chain: str
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_category: str
    is_prototype_fusion: bool = True
    
    # Model scores (if available)
    xgboost_score: Optional[float] = None
    isolation_forest_score: Optional[float] = None
    graph_ml_score: Optional[float] = None

    signals: DeterministicSignals
    features: RiskFeatures
    model_status: ModelStatus
    
    # Explainability
    shap_explanation: Optional[SHAPExplanation] = None
    gnn_explanation: Optional[GNNExplanation] = None
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_ids: List[str] = Field(default_factory=list)

    model_config = {"frozen": True}
