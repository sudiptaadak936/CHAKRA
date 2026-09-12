"""CHAKRA Step 5: Risk Engine Unit Tests."""
import pytest
from datetime import datetime, timezone

from app.schemas.chain import Chain
from app.schemas.alert import SanctionedAddressHit
from app.forensics.typology_detector import TypologyDetection, TypologyType

from app.risk.repository import RiskRepository
from app.risk.features import FeatureExtractor
from app.risk.signals import SignalAdapter
from app.schemas.risk_scoring import DeterministicSignals

class MockRiskRepository(RiskRepository):
    def __init__(self, stats, cluster_size, velocity):
        self.stats = stats
        self._cluster_size = cluster_size
        self._velocity = velocity

    async def get_address_stats(self, chain, address):
        return self.stats

    async def get_cluster_size(self, chain, address):
        return self._cluster_size

    async def get_inter_hop_velocity_hours(self, chain, address):
        return self._velocity

@pytest.mark.asyncio
async def test_feature_extraction_full():
    """Test feature engineering with complete data."""
    stats = {
        "degree": 10,
        "transaction_volume": 5000.0,
        "total_incoming": 3000.0,
        "total_outgoing": 2000.0,
        "first_activity_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        "last_activity_at": datetime(2023, 1, 2, tzinfo=timezone.utc)
    }
    repo = MockRiskRepository(stats, cluster_size=5, velocity=2.5)
    extractor = FeatureExtractor(repo)
    
    features = await extractor.extract_features("bitcoin", "addr1")
    
    assert features.degree.available is True
    assert features.degree.value == 10.0
    
    assert features.transaction_volume.available is True
    assert features.transaction_volume.value == 5000.0
    
    # Mixer explicitly unavailable
    assert features.hop_distance_to_mixer.available is False
    
    assert features.time_since_first_activity.available is True
    assert features.time_since_first_activity.value > 0
    
    assert features.cluster_size.available is True
    assert features.cluster_size.value == 5.0
    
    assert features.amount_retention_ratio.available is True
    assert features.amount_retention_ratio.value == (3000.0 - 2000.0) / 3000.0
    
    assert features.inter_hop_velocity.available is True
    assert features.inter_hop_velocity.value == 2.5

@pytest.mark.asyncio
async def test_feature_extraction_missing_data():
    """Test feature engineering explicit missing behavior."""
    repo = MockRiskRepository({}, cluster_size=None, velocity=None)
    extractor = FeatureExtractor(repo)
    
    features = await extractor.extract_features("bitcoin", "addr1")
    
    assert features.degree.available is False
    assert "No transactions" in features.degree.reason_if_missing
    
    assert features.cluster_size.available is False
    assert features.amount_retention_ratio.available is False
    assert features.inter_hop_velocity.available is False

def test_signal_adapter_clean():
    """Test signals map correctly to TRUE/FALSE/UNKNOWN."""
    # Sanction hit
    hit = SanctionedAddressHit(
        chain=Chain.BITCOIN, address="addr1",
        source="ofac", reason="test", evidence_id="ev1"
    )
    
    # Mixer returning insufficient evidence
    mixer_typo = TypologyDetection(
        detection_id="m1", typology_type=TypologyType.MIXER_INTERACTION,
        chain="bitcoin", confidence_level="insufficient_evidence", explanation="safe"
    )
    
    # Cross chain observed
    bridge_typo = TypologyDetection(
        detection_id="b1", typology_type=TypologyType.CROSS_CHAIN,
        chain="bitcoin", confidence_level="observed", explanation="bridge"
    )
    
    signals = SignalAdapter.evaluate_signals("addr1", "bitcoin", "mainnet", [hit], [mixer_typo, bridge_typo])
    
    assert signals.known_bad_address_hit.state == "TRUE"
    assert signals.known_bad_address_hit.evidence_id == "ev1"
    
    # Mixer explicitly requested to be UNKNOWN if evidence is lacking/insufficient, not fabricated as TRUE
    assert signals.mixer_signal.state == "UNKNOWN"
    
    # Cross chain observed
    assert signals.cross_chain_signal.state == "TRUE"
    assert signals.cross_chain_signal.evidence_id == "b1"
    
    # Rapid hopping missing
    assert signals.temporal_signal.state == "FALSE"

