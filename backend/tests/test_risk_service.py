"""CHAKRA Step 5: Risk Service Unit Tests."""
import pytest
import os
import json
from unittest.mock import AsyncMock, patch

from app.schemas.chain import Chain
from app.risk.service import RiskService
from app.schemas.risk_scoring import RiskAssessmentResult

@pytest.mark.asyncio
@patch("app.risk.service.AlertService.generate_risk_threshold_alert", new_callable=AsyncMock)
@patch("app.risk.service.FeatureExtractor.extract_features")
@patch("app.risk.service.SignalAdapter.evaluate_signals")
async def test_risk_service_fusion_missing_ml(mock_signals, mock_features, mock_alert):
    """Test risk fusion without ML models available."""
    # Setup stubs
    from app.schemas.risk_scoring import DeterministicSignals, SignalState
    mock_signals.return_value = DeterministicSignals(
        rule_engine_signal=SignalState(state="TRUE"),
        known_bad_address_hit=SignalState(state="FALSE"),
        mixer_signal=SignalState(state="UNKNOWN"),
        cross_chain_signal=SignalState(state="UNKNOWN"),
        temporal_signal=SignalState(state="FALSE")
    )
    
    from app.schemas.risk_scoring import RiskFeatures, FeatureAvailability
    mock_features.return_value = RiskFeatures(
        degree=FeatureAvailability(available=True, value=5),
        transaction_volume=FeatureAvailability(available=True, value=100.0),
        hop_distance_to_mixer=FeatureAvailability(available=False),
        time_since_first_activity=FeatureAvailability(available=True, value=10.0),
        cluster_size=FeatureAvailability(available=False),
        amount_retention_ratio=FeatureAvailability(available=True, value=0.5),
        inter_hop_velocity=FeatureAvailability(available=False)
    )
    
    # Run service
    mock_pool = AsyncMock()
    service = RiskService(pool=mock_pool, model_dir="/non/existent/dir")
    
    # Disable persistence for unit test
    service._persist_result = AsyncMock()
    
    result = await service.evaluate_risk("addr1", "bitcoin", "mainnet", [], [])
    
    assert isinstance(result, RiskAssessmentResult)
    assert result.is_prototype_fusion is True
    
    # With missing ML, rule engine state="TRUE" translates to 100.0, which means high risk.
    # Because ML is missing, it falls back to the deterministic score.
    assert result.risk_score == 100.0
    assert result.risk_category == "CRITICAL"
    
    assert result.model_status.xgboost_trained is False
    assert result.model_status.isolation_forest_trained is False
    
    # Assert alert was triggered
    mock_alert.assert_called_once()
    args, _ = mock_alert.call_args
    alert = args[0]
    assert alert.address == "addr1"
    assert alert.overall_score == 100.0
    assert alert.risk_level == "CRITICAL"

@pytest.mark.asyncio
@patch("app.risk.service.FeatureExtractor.extract_features")
@patch("app.risk.service.SignalAdapter.evaluate_signals")
async def test_risk_service_fusion_ml_scores(mock_signals, mock_features):
    """Test risk fusion with simulated ML models available."""
    from app.schemas.risk_scoring import DeterministicSignals, SignalState
    mock_signals.return_value = DeterministicSignals(
        rule_engine_signal=SignalState(state="FALSE"),
        known_bad_address_hit=SignalState(state="FALSE"),
        mixer_signal=SignalState(state="FALSE"),
        cross_chain_signal=SignalState(state="FALSE"),
        temporal_signal=SignalState(state="FALSE")
    )
    
    from app.schemas.risk_scoring import RiskFeatures, FeatureAvailability
    mock_features.return_value = RiskFeatures(
        degree=FeatureAvailability(available=True, value=5),
        transaction_volume=FeatureAvailability(available=True, value=100.0),
        hop_distance_to_mixer=FeatureAvailability(available=False),
        time_since_first_activity=FeatureAvailability(available=True, value=10.0),
        cluster_size=FeatureAvailability(available=False),
        amount_retention_ratio=FeatureAvailability(available=True, value=0.5),
        inter_hop_velocity=FeatureAvailability(available=False)
    )
    
    mock_pool = AsyncMock()
    service = RiskService(pool=mock_pool)
    service._persist_result = AsyncMock()
    
    # Mock ML predictions
    service.xgb_model.is_trained = lambda: True
    service.xgb_model.predict = lambda f: (50.0, None)
    
    service.iso_model.is_trained = lambda: True
    service.iso_model.predict = lambda f: (20.0)
    
    service.gnn_model.is_trained = lambda: True
    service.gnn_model.predict = lambda f: (10.0, None)
    
    # Run
    result = await service.evaluate_risk("addr1", "bitcoin", "mainnet", [], [])
    
    # Rule=0 (weight 0.4), XGB=50 (weight 0.4), Iso=20 (weight 0.1), GNN=10 (weight 0.1)
    # Total = 0 + 20 + 2 + 1 = 23.0
    assert result.risk_score == 23.0
    assert result.risk_category == "LOW"
    assert result.model_status.xgboost_trained is True

