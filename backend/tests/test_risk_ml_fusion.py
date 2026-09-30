"""CHAKRA Step 5J Tests: Controlled Risk & ML Fusion Layer.

Verifies:
- Deterministic risk engine preservation (zero changes to deterministic scoring or hash).
- Address ML inference preservation (exact probabilities and artifact provenance).
- Fusion policy handling (explicit NOT_CONFIGURED status when unconfigured, no fabricated weights).
- No implicit averaging of LR and XGB.
- Strict input validation and fail-closed error handling.
- Read-only contract and zero-side-effect isolation.
- Threshold protection (no operational risk bands assigned to ML probabilities).
"""
import pytest
from app.forensics.risk_engine import RiskEngine
from app.forensics.address_ml_service import AddressMLInferenceService
from app.forensics.risk_ml_fusion_service import RiskMLFusionService
from app.forensics.address_ml_loader import AddressMLLoader
from app.schemas.alert import SanctionedAddressHit
from app.schemas.chain import Chain
from app.schemas.ml_features import FEATURE_SCHEMA_VERSION
from app.schemas.risk import RiskLevel
from app.schemas.risk_fusion import FusionStatus


@pytest.fixture
def fusion_service():
    return RiskMLFusionService()


@pytest.fixture
def sample_feature_vector():
    return [
        10.0,         # in_degree
        5.0,          # out_degree
        15.0,         # total_tx_count
        1000000.0,    # total_received_native
        500000.0,     # total_sent_native
        0.5,          # amount_retention_ratio
        86400.0,      # time_active_seconds
        -1.0,         # inter_hop_velocity_avg (sentinel)
        12.0,         # unique_counterparties
        0.0,          # typology_peel_chain_flag
        0.0,          # typology_rapid_hop_flag
        0.0,          # typology_fan_in_flag
        0.0,          # typology_fan_out_flag
    ]


def test_deterministic_risk_preservation(fusion_service, sample_feature_vector):
    """Verify that deterministic risk calculation in composite matches standalone RiskEngine."""
    address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    chain = "bitcoin"
    network = "mainnet"

    # Standalone evaluation
    standalone_det = RiskEngine.evaluate(
        target_address=address,
        chain=chain,
        network=network,
    )

    # Composite evaluation
    composite = fusion_service.evaluate_composite(
        target_address=address,
        chain=chain,
        network=network,
        feature_vector=sample_feature_vector,
    )

    # Must match exactly
    assert composite.deterministic_layer.overall_score == standalone_det.overall_score
    assert composite.deterministic_layer.risk_level == standalone_det.risk_level
    assert composite.deterministic_layer.deterministic_hash == standalone_det.deterministic_hash
    assert len(composite.deterministic_layer.components) == len(standalone_det.components)


def test_ml_inference_preservation(fusion_service, sample_feature_vector):
    """Verify that ML probabilities and provenance in composite match standalone AddressMLInferenceService."""
    address = "0x742d35cc6634c0532925a3b844bc454e4438f44e"
    chain = "evm"
    network = "ethereum-mainnet"

    ml_service = AddressMLInferenceService()
    standalone_ml = ml_service.infer(
        AddressMLInferenceService()._loader.load_artifacts() and
        __import__("app.schemas.ml_inference", fromlist=["AddressMLInferenceRequest"]).AddressMLInferenceRequest(
            target_address=address,
            chain=chain,
            network=network,
            feature_vector=sample_feature_vector,
        )
    )

    composite = fusion_service.evaluate_composite(
        target_address=address,
        chain=chain,
        network=network,
        feature_vector=sample_feature_vector,
    )

    assert composite.ml_layer.lr_raw_probability == standalone_ml.lr_raw_probability
    assert composite.ml_layer.lr_calibrated_probability == standalone_ml.lr_calibrated_probability
    assert composite.ml_layer.xgb_raw_probability == standalone_ml.xgb_raw_probability
    assert composite.ml_layer.xgb_calibrated_probability == standalone_ml.xgb_calibrated_probability
    assert composite.ml_layer.provenance == standalone_ml.provenance


def test_fusion_layer_not_configured(fusion_service, sample_feature_vector):
    """Verify that when no explicit fusion policy exists, fusion_status is NOT_CONFIGURED."""
    composite = fusion_service.evaluate_composite(
        target_address="0x1234567890123456789012345678901234567890",
        chain="evm",
        network="ethereum-mainnet",
        feature_vector=sample_feature_vector,
    )

    assert composite.fusion_layer.fusion_status == FusionStatus.NOT_CONFIGURED
    assert composite.fusion_layer.fusion_score is None
    assert composite.fusion_layer.fusion_policy_version is None
    assert "not specified in frozen repository contracts" in composite.fusion_layer.reason


def test_no_implicit_model_averaging(fusion_service, sample_feature_vector):
    """Verify that LR and XGB probabilities are not synthesized into an ad-hoc average."""
    composite = fusion_service.evaluate_composite(
        target_address="0x1234567890123456789012345678901234567890",
        chain="evm",
        network="ethereum-mainnet",
        feature_vector=sample_feature_vector,
    )

    # Probabilities must be accessible independently
    lr_cal = composite.ml_layer.lr_calibrated_probability
    xgb_cal = composite.ml_layer.xgb_calibrated_probability
    assert lr_cal != xgb_cal
    assert composite.fusion_layer.fusion_score is None


def test_combine_results_mismatch_fails_closed(fusion_service, sample_feature_vector):
    """Verify combining mismatched address results raises ValueError."""
    det = RiskEngine.evaluate(
        target_address="0x1111111111111111111111111111111111111111",
        chain="evm",
        network="mainnet",
    )
    from app.schemas.ml_inference import AddressMLInferenceRequest
    ml_service = AddressMLInferenceService()
    ml = ml_service.infer(
        AddressMLInferenceRequest(
            target_address="0x2222222222222222222222222222222222222222",
            chain="evm",
            network="mainnet",
            feature_vector=sample_feature_vector,
        )
    )

    with pytest.raises(ValueError, match="Address mismatch"):
        fusion_service.combine_results(det, ml)


def test_input_validation_fails_closed(fusion_service, sample_feature_vector):
    """Verify missing required target metadata fails closed."""
    with pytest.raises(ValueError, match="Target address is required"):
        fusion_service.evaluate_composite(
            target_address="",
            chain="evm",
            network="mainnet",
            feature_vector=sample_feature_vector,
        )

    with pytest.raises(ValueError, match="Chain is required"):
        fusion_service.evaluate_composite(
            target_address="0xabc",
            chain="",
            network="mainnet",
            feature_vector=sample_feature_vector,
        )


def test_threshold_protection_no_ml_bands(fusion_service, sample_feature_vector):
    """Verify ML probabilities are never categorized into risk levels."""
    composite = fusion_service.evaluate_composite(
        target_address="0x1234567890123456789012345678901234567890",
        chain="evm",
        network="ethereum-mainnet",
        feature_vector=sample_feature_vector,
    )

    # Deterministic layer has risk_level (e.g. NEUTRAL for 0 score)
    assert composite.deterministic_layer.risk_level == RiskLevel.NEUTRAL

    # ML layer has strictly numeric probabilities, no risk bands
    assert not hasattr(composite.ml_layer, "risk_level")
    assert not hasattr(composite.ml_layer, "severity")


def test_side_effect_isolation(fusion_service, sample_feature_vector):
    """Verify that composite evaluation does not mutate disk artifacts."""
    loader = AddressMLLoader()
    hashes_before = loader.verify_integrity()

    composite = fusion_service.evaluate_composite(
        target_address="0xsideeffectcheck",
        chain="evm",
        network="ethereum-mainnet",
        feature_vector=sample_feature_vector,
    )
    assert composite.fusion_layer.fusion_status == FusionStatus.NOT_CONFIGURED

    hashes_after = loader.verify_integrity()
    assert hashes_before == hashes_after
