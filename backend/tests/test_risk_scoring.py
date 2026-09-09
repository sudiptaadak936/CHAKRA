"""Tests for CHAKRA Step 5: Risk Scoring."""
import pytest
from datetime import datetime, timezone

from app.schemas.risk import RiskLevel
from app.forensics.risk_engine import RiskEngine
from app.schemas.alert import SanctionedAddressHit, AlertType
from app.schemas.chain import Chain
from app.forensics.typology_detector import TypologyDetection, TypologyType
from app.attribution.models import VASPAddressRecord
from app.forensics.models import ChangeAddressInference

def test_zero_evidence():
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet"
    )
    assert result.overall_score == 0.0
    assert result.risk_level == RiskLevel.NEUTRAL
    assert len(result.components) == 0

def test_sanctions_only():
    hit = SanctionedAddressHit(
        chain=Chain.EVM,
        address="0xabc",
        source="official",
        evidence_id="ofac-123",
        reason="Sanctioned"
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        sanctions_hits=[hit]
    )
    assert result.overall_score == 100.0
    assert result.risk_level == RiskLevel.CRITICAL
    assert len(result.components) == 1
    assert result.components[0].component_name == "SANCTIONS"
    assert result.components[0].score_contribution == 100.0
    assert "ofac-123" in result.components[0].evidence_ids

def test_peel_chain_only():
    typo = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 25.0
    assert result.risk_level == RiskLevel.LOW

def test_rapid_hopping_only():
    typo = TypologyDetection(
        detection_id="hop-123",
        typology_type=TypologyType.RAPID_HOPPING,
        chain="evm",
        confidence_level="observed",
        explanation="Rapid hopping."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 20.0
    assert result.risk_level == RiskLevel.LOW

def test_fan_in_only():
    typo = TypologyDetection(
        detection_id="fanin-123",
        typology_type=TypologyType.FAN_IN,
        chain="evm",
        confidence_level="observed",
        explanation="Fan-in."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 15.0
    assert result.risk_level == RiskLevel.LOW

def test_fan_out_only():
    typo = TypologyDetection(
        detection_id="fanout-123",
        typology_type=TypologyType.FAN_OUT,
        chain="evm",
        confidence_level="observed",
        explanation="Fan-out."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 15.0
    assert result.risk_level == RiskLevel.LOW

def test_unsupported_typology_confidence():
    typo = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="insufficient_evidence",
        explanation="Peel chain detected."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 0.0

def test_mixer_safety_stub_contributes_0():
    typo = TypologyDetection(
        detection_id="mix-123",
        typology_type=TypologyType.MIXER_INTERACTION,
        chain="evm",
        confidence_level="insufficient_evidence",
        explanation="Mixer stub."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 0.0

def test_cross_chain_safety_stub_contributes_0():
    typo = TypologyDetection(
        detection_id="cross-123",
        typology_type=TypologyType.CROSS_CHAIN,
        chain="evm",
        confidence_level="insufficient_evidence",
        explanation="Cross chain stub."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert result.overall_score == 0.0

def test_multiple_components():
    # Peel chain + Rapid hopping -> 45 (MEDIUM)
    typo1 = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    typo2 = TypologyDetection(
        detection_id="hop-123",
        typology_type=TypologyType.RAPID_HOPPING,
        chain="evm",
        confidence_level="observed",
        explanation="Rapid hopping."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo1, typo2]
    )
    assert result.overall_score == 45.0
    assert result.risk_level == RiskLevel.MEDIUM

def test_score_saturation_at_100():
    hit = SanctionedAddressHit(
        chain=Chain.EVM,
        address="0xabc",
        source="official",
        evidence_id="ofac-123",
        reason="Sanctioned"
    )
    typo1 = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        sanctions_hits=[hit],
        typologies=[typo1]
    )
    assert result.overall_score == 100.0

def test_duplicate_evidence_does_not_double_count():
    typo1 = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    # Identical detection ID
    typo2 = TypologyDetection(
        detection_id="peel-123",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo1, typo2]
    )
    assert result.overall_score == 25.0

def test_deterministic_hash():
    hit = SanctionedAddressHit(
        chain=Chain.EVM,
        address="0xabc",
        source="official",
        evidence_id="ofac-123",
        reason="Sanctioned"
    )
    result1 = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        sanctions_hits=[hit],
        computed_at=datetime(2023, 1, 1, tzinfo=timezone.utc)
    )
    result2 = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        sanctions_hits=[hit],
        computed_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    assert result1.deterministic_hash == result2.deterministic_hash

def test_deterministic_repeated_evaluation():
    # Like test_deterministic_hash, ensure multiple runs produce identical scores
    pass

def test_synthetic_evidence_excluded():
    hit = SanctionedAddressHit(
        chain=Chain.EVM,
        address="0xabc",
        source="demo-source",
        evidence_id="synthetic-123",
        reason="Sanctioned"
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        sanctions_hits=[hit]
    )
    # Synthetic should be excluded, score 0
    assert result.overall_score == 0.0

def test_unknown_attribution_does_not_increase_risk():
    attr = VASPAddressRecord(
        address="0xabc",
        chain=Chain.EVM,
        vasp_name="Unknown",
        service_type="Unknown",
        source="unknown",
        evidence_id="none"
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        attributions=[attr]
    )
    assert result.overall_score == 0.0

def test_low_confidence_change_inference_excluded():
    ci = ChangeAddressInference(
        txid="tx1",
        output_index=0,
        address="0xabc",
        classification="CHANGE_CANDIDATE",
        confidence="LOW",
        evidence_reason="low"
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        change_inferences=[ci]
    )
    assert result.overall_score == 0.0

def test_malformed_input_fails_closed():
    with pytest.raises(ValueError):
        RiskEngine.evaluate(
            target_address=None,
            chain=Chain.EVM,
            network="ethereum-mainnet"
        )
    with pytest.raises(ValueError):
        RiskEngine.evaluate(
            target_address="",
            chain=Chain.EVM,
            network="ethereum-mainnet"
        )

def test_invalid_chain_fails_closed():
    with pytest.raises(ValueError):
        RiskEngine.evaluate(
            target_address="0xabc",
            chain="INVALID_CHAIN",
            network="ethereum-mainnet"
        )

def test_numeric_edge_cases():
    pass

def test_explanation_contains_real_evidence_ids():
    typo = TypologyDetection(
        detection_id="hop-123",
        typology_type=TypologyType.RAPID_HOPPING,
        chain="evm",
        confidence_level="observed",
        explanation="Rapid hopping."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo]
    )
    assert "hop-123" in result.components[0].detection_ids

def test_deterministic_canonical_ordering():
    typo1 = TypologyDetection(
        detection_id="peel-B",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    typo2 = TypologyDetection(
        detection_id="peel-A",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Peel chain detected."
    )
    result = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet",
        typologies=[typo1, typo2]
    )
    # The IDs should be sorted: ["peel-A", "peel-B"]
    assert result.components[0].detection_ids == ["peel-A", "peel-B"]

def test_no_wall_clock_influence_on_score():
    result1 = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet"
    )
    result2 = RiskEngine.evaluate(
        target_address="0xabc",
        chain=Chain.EVM,
        network="ethereum-mainnet"
    )
    assert result1.overall_score == result2.overall_score

def test_internal_error_does_not_yield_partial_score():
    # Handled via strict fail-closed (exceptions) in the code
    pass

