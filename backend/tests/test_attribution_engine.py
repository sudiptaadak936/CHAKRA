import pytest
from app.schemas.chain import Chain
from app.attribution.models import VASPAddressRecord
from app.attribution.engine_models import AttributionConfidence
from app.attribution.engine import AttributionDecisionEngine
from app.clustering.models import AddressCluster, ClusterMember

class MockRegistry:
    def __init__(self):
        self.records = []
        
    def add_record(self, record: VASPAddressRecord):
        self.records.append(record)
        
    async def lookup(self, address: str, chain: str):
        return [r for r in self.records if r.address == address and r.chain.value == chain]

@pytest.fixture
def registry():
    return MockRegistry()

@pytest.fixture
def engine(registry):
    return AttributionDecisionEngine(registry)

@pytest.mark.asyncio
async def test_1_confirmed_attribution(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0x1", chain=Chain.EVM, vasp_name="ExchangeA", service_type="exchange",
        source="official", evidence_id="1", metadata={}
    ))
    decision = await engine.evaluate("0x1", Chain.EVM)
    assert decision.confidence == AttributionConfidence.CONFIRMED
    assert decision.vasp_name == "ExchangeA"
    assert decision.explanation == "Direct VASP registry attribution supported by authoritative source evidence."

@pytest.mark.asyncio
async def test_2_high_confidence_attribution(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0x2", chain=Chain.EVM, vasp_name="ExchangeB", service_type="exchange",
        source="demo", evidence_id="2", metadata={}
    ))
    # Sibling in the cluster also attributed to ExchangeB, providing genuine cluster corroboration
    registry.add_record(VASPAddressRecord(
        address="0x3", chain=Chain.EVM, vasp_name="ExchangeB", service_type="exchange",
        source="demo", evidence_id="3", metadata={}
    ))
    cluster = AddressCluster(
        cluster_id="c-1", representative_composite_id="0x2_evm", chain=Chain.EVM.value, network="ethereum",
        members=[ClusterMember(cluster_id="c-1", composite_id="0x2_evm", chain="evm", network="ethereum", normalized_address="0x2"), 
                 ClusterMember(cluster_id="c-1", composite_id="0x3_evm", chain="evm", network="ethereum", normalized_address="0x3")]
    )
    decision = await engine.evaluate("0x2", Chain.EVM, cluster)
    assert decision.confidence == AttributionConfidence.HIGH_CONFIDENCE
    assert decision.vasp_name == "ExchangeB"
    assert decision.provenance.corroborating_evidence_count == 1

@pytest.mark.asyncio
async def test_3_probable_inferred_attribution(engine, registry):
    # No direct observation on 0x4, but observation on 0x5 in the same cluster
    registry.add_record(VASPAddressRecord(
        address="0x5", chain=Chain.EVM, vasp_name="ExchangeC", service_type="exchange",
        source="demo", evidence_id="3", metadata={}
    ))
    cluster = AddressCluster(
        cluster_id="c-2", representative_composite_id="0x4_evm", chain=Chain.EVM.value, network="ethereum",
        members=[ClusterMember(cluster_id="c-2", composite_id="0x4_evm", chain="evm", network="ethereum", normalized_address="0x4"), 
                 ClusterMember(cluster_id="c-2", composite_id="0x5_evm", chain="evm", network="ethereum", normalized_address="0x5")]
    )
    decision = await engine.evaluate("0x4", Chain.EVM, cluster)
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.vasp_name == "ExchangeC"

@pytest.mark.asyncio
async def test_4_unknown(engine, registry):
    decision = await engine.evaluate("0xunknown", Chain.EVM)
    assert decision.confidence == AttributionConfidence.UNKNOWN
    assert decision.vasp_name is None
    assert decision.explanation == "No sufficient attribution evidence was available."

@pytest.mark.asyncio
async def test_5_unknown_registry_address(engine, registry):
    decision = await engine.evaluate("0xempty", Chain.EVM)
    assert decision.confidence == AttributionConfidence.UNKNOWN
    # No RE-ID inference or NOT_REGISTERED state
    assert decision.vasp_name is None

@pytest.mark.asyncio
async def test_6_wrong_chain(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0x1", chain=Chain.TRON, vasp_name="ExchangeTron", service_type="exchange",
        source="official", evidence_id="1", metadata={}
    ))
    decision = await engine.evaluate("0x1", Chain.EVM)
    assert decision.confidence == AttributionConfidence.UNKNOWN
    assert decision.vasp_name is None

@pytest.mark.asyncio
async def test_7_multiple_candidates(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xmulti", chain=Chain.EVM, vasp_name="ExchangeA", service_type="exchange",
        source="official", evidence_id="1", metadata={}
    ))
    registry.add_record(VASPAddressRecord(
        address="0xmulti", chain=Chain.EVM, vasp_name="ExchangeB", service_type="exchange",
        source="official", evidence_id="2", metadata={}
    ))
    decision = await engine.evaluate("0xmulti", Chain.EVM)
    assert decision.confidence == AttributionConfidence.UNKNOWN
    assert decision.vasp_name is None
    assert len(decision.candidate_attributions) == 2

@pytest.mark.asyncio
async def test_8_provenance_preservation(engine, registry):
    obs = VASPAddressRecord(
        address="0xprov", chain=Chain.EVM, vasp_name="ExchangeP", service_type="exchange",
        source="official", evidence_id="ev_prov_123", metadata={}
    )
    registry.add_record(obs)
    decision = await engine.evaluate("0xprov", Chain.EVM)
    assert decision.provenance.direct_observations[0] == obs
    assert decision.candidate_attributions[0] == obs

@pytest.mark.asyncio
async def test_9_demo_provenance(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xdemo", chain=Chain.EVM, vasp_name="ExchangeD", service_type="exchange",
        source="demo_vasp_registry.json", evidence_id="demo_1", metadata={}
    ))
    decision = await engine.evaluate("0xdemo", Chain.EVM)
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED

@pytest.mark.asyncio
async def test_10_no_confidence_leakage(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xleak", chain=Chain.EVM, vasp_name="ExchangeL", service_type="exchange",
        source="demo", evidence_id="1", metadata={"confidence": "CONFIRMED"}
    ))
    decision = await engine.evaluate("0xleak", Chain.EVM)
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED

@pytest.mark.asyncio
async def test_11_determinism(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xdet", chain=Chain.EVM, vasp_name="ExchangeD", service_type="exchange",
        source="official", evidence_id="1", metadata={}
    ))
    dec1 = await engine.evaluate("0xdet", Chain.EVM)
    dec2 = await engine.evaluate("0xdet", Chain.EVM)
    assert dec1 == dec2

@pytest.mark.asyncio
async def test_12_step3_evidence_integration(engine, registry):
    cluster = AddressCluster(
        cluster_id="c-9", representative_composite_id="0x_step3_evm", chain=Chain.EVM.value, network="ethereum",
        members=[ClusterMember(cluster_id="c-9", composite_id="0x_step3_evm", chain="evm", network="ethereum", normalized_address="0x_step3")]
    )
    registry.add_record(VASPAddressRecord(
        address="0x_step3", chain=Chain.EVM, vasp_name="ExchangeS3", service_type="exchange",
        source="official", evidence_id="1", metadata={}
    ))
    decision = await engine.evaluate("0x_step3", Chain.EVM, cluster)
    assert decision.provenance.cluster_id == "c-9"

@pytest.mark.asyncio
async def test_cluster_size_alone_does_not_escalate_confidence(engine, registry):
    # Target address has a single non-authoritative observation
    registry.add_record(VASPAddressRecord(
        address="0xsolo", chain=Chain.EVM, vasp_name="ExchangeSolo", service_type="exchange",
        source="demo", evidence_id="solo_1", metadata={}
    ))
    # Cluster has multiple members, but siblings have NO attribution for ExchangeSolo
    cluster = AddressCluster(
        cluster_id="c-5", representative_composite_id="0xsolo_evm", chain=Chain.EVM.value, network="ethereum",
        members=[
            ClusterMember(cluster_id="c-5", composite_id="0xsolo_evm", chain="evm", network="ethereum", normalized_address="0xsolo"),
            ClusterMember(cluster_id="c-5", composite_id="0xother1_evm", chain="evm", network="ethereum", normalized_address="0xother1"),
            ClusterMember(cluster_id="c-5", composite_id="0xother2_evm", chain="evm", network="ethereum", normalized_address="0xother2"),
        ]
    )
    decision = await engine.evaluate("0xsolo", Chain.EVM, cluster)
    # member_count > 1 alone MUST NOT escalate to HIGH_CONFIDENCE
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.vasp_name == "ExchangeSolo"
    assert decision.provenance.corroborating_evidence_count == 0

@pytest.mark.asyncio
async def test_high_confidence_multiple_independent_sources(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xindep", chain=Chain.EVM, vasp_name="ExchangeMulti", service_type="exchange",
        source="source_alpha", evidence_id="alpha_1", metadata={}
    ))
    registry.add_record(VASPAddressRecord(
        address="0xindep", chain=Chain.EVM, vasp_name="ExchangeMulti", service_type="exchange",
        source="source_beta", evidence_id="beta_1", metadata={}
    ))
    decision = await engine.evaluate("0xindep", Chain.EVM)
    assert decision.confidence == AttributionConfidence.HIGH_CONFIDENCE
    assert decision.vasp_name == "ExchangeMulti"
    assert decision.provenance.corroborating_evidence_count == 1

@pytest.mark.asyncio
async def test_duplicate_source_does_not_count_as_independent(engine, registry):
    registry.add_record(VASPAddressRecord(
        address="0xdup", chain=Chain.EVM, vasp_name="ExchangeDup", service_type="exchange",
        source="same_source", evidence_id="ev1", metadata={}
    ))
    registry.add_record(VASPAddressRecord(
        address="0xdup", chain=Chain.EVM, vasp_name="ExchangeDup", service_type="exchange",
        source="same_source", evidence_id="ev2", metadata={}
    ))
    decision = await engine.evaluate("0xdup", Chain.EVM)
    # Identical sources must NOT escalate to HIGH_CONFIDENCE
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.vasp_name == "ExchangeDup"

@pytest.mark.asyncio
async def test_competing_candidates_stronger_evidence_wins(engine, registry):
    # Candidate A has authoritative source (CONFIRMED)
    registry.add_record(VASPAddressRecord(
        address="0xcomp", chain=Chain.EVM, vasp_name="ExchangeOfficial", service_type="exchange",
        source="official", evidence_id="ev_off", metadata={}
    ))
    # Candidate B has non-authoritative source (PROBABLE_INFERRED)
    registry.add_record(VASPAddressRecord(
        address="0xcomp", chain=Chain.EVM, vasp_name="ExchangeDemo", service_type="exchange",
        source="demo", evidence_id="ev_demo", metadata={}
    ))
    decision = await engine.evaluate("0xcomp", Chain.EVM)
    assert decision.confidence == AttributionConfidence.CONFIRMED
    assert decision.vasp_name == "ExchangeOfficial"
    # Competing candidate observations must be preserved
    assert len(decision.candidate_attributions) == 2


# ===========================================================================
# Step 4.2 Authority Contract Hardening Tests
# ===========================================================================

from app.attribution.authority import AuthorityPolicy, AuthorityClass

@pytest.mark.asyncio
async def test_authority_contract_recognized_authoritative(engine, registry):
    """Test 1: Recognized authoritative provenance establishes CONFIRMED."""
    for auth_source in ["official", "verified_institutional", "statutory_register", "regulatory_gazette"]:
        record = VASPAddressRecord(
            address=f"0xauth_{auth_source}", chain=Chain.EVM, vasp_name="VerifiedVASP", service_type="exchange",
            source=auth_source, evidence_id=f"ev_{auth_source}", metadata={}
        )
        assert AuthorityPolicy.classify_observation(record) == AuthorityClass.AUTHORITATIVE
        assert AuthorityPolicy.is_authoritative(record) is True
        
        registry.add_record(record)
        decision = await engine.evaluate(f"0xauth_{auth_source}", Chain.EVM)
        assert decision.confidence == AttributionConfidence.CONFIRMED
        assert decision.vasp_name == "VerifiedVASP"
        assert decision.provenance.authority_class == AuthorityClass.AUTHORITATIVE

@pytest.mark.asyncio
async def test_authority_contract_unknown_source(engine, registry):
    """Test 2: Unrecognized source cannot establish authority."""
    record = VASPAddressRecord(
        address="0xunk_src", chain=Chain.EVM, vasp_name="VASPUnk", service_type="exchange",
        source="random_forum_tag", evidence_id="ev_forum_1", metadata={}
    )
    assert AuthorityPolicy.classify_observation(record) == AuthorityClass.NON_AUTHORITATIVE
    assert AuthorityPolicy.is_authoritative(record) is False
    
    registry.add_record(record)
    decision = await engine.evaluate("0xunk_src", Chain.EVM)
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.provenance.authority_class == AuthorityClass.NON_AUTHORITATIVE

@pytest.mark.asyncio
async def test_authority_contract_demo_source(engine, registry):
    """Test 3: Demo/synthetic provenance cannot establish authority."""
    demo_records = [
        VASPAddressRecord(address="0xd1", chain=Chain.EVM, vasp_name="V1", service_type="exchange", source="demo_feed", evidence_id="ev_1", metadata={}),
        VASPAddressRecord(address="0xd2", chain=Chain.EVM, vasp_name="V2", service_type="exchange", source="synthetic_source", evidence_id="ev_2", metadata={}),
        VASPAddressRecord(address="0xd3", chain=Chain.EVM, vasp_name="V3", service_type="exchange", source="official", evidence_id="test_id_99", metadata={}),
        VASPAddressRecord(address="0xd4", chain=Chain.EVM, vasp_name="V4", service_type="exchange", source="official", evidence_id="demo_evidence", metadata={}),
    ]
    for rec in demo_records:
        assert AuthorityPolicy.classify_observation(rec) == AuthorityClass.DEMO
        assert AuthorityPolicy.is_authoritative(rec) is False
        registry.add_record(rec)
        decision = await engine.evaluate(rec.address, Chain.EVM)
        assert decision.confidence != AttributionConfidence.CONFIRMED

@pytest.mark.asyncio
async def test_authority_contract_metadata_spoofing(engine, registry):
    """Test 4: Metadata spoofing cannot manufacture authority."""
    record = VASPAddressRecord(
        address="0xspoof", chain=Chain.EVM, vasp_name="SpoofedVASP", service_type="exchange",
        source="public_tag", evidence_id="tag_100",
        metadata={
            "authority": "AUTHORITATIVE",
            "confidence": "CONFIRMED",
            "verified": True,
            "trust_level": "maximum",
            "source_authority": "government"
        }
    )
    assert AuthorityPolicy.classify_observation(record) == AuthorityClass.NON_AUTHORITATIVE
    assert AuthorityPolicy.is_authoritative(record) is False
    
    registry.add_record(record)
    decision = await engine.evaluate("0xspoof", Chain.EVM)
    # Must NOT produce CONFIRMED despite spoofed metadata
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.provenance.authority_class == AuthorityClass.NON_AUTHORITATIVE

@pytest.mark.asyncio
async def test_authority_contract_missing_or_invalid_evidence_id(engine, registry):
    """Test 5: Missing or placeholder evidence ID cannot satisfy authoritative contract."""
    for invalid_id in ["none", "null", "unknown", "0", "undefined", "n/a", "na"]:
        record = VASPAddressRecord(
            address=f"0xinv_{invalid_id}", chain=Chain.EVM, vasp_name="InvalidIdVASP", service_type="exchange",
            source="official", evidence_id=invalid_id, metadata={}
        )
        assert AuthorityPolicy.classify_observation(record) == AuthorityClass.UNKNOWN
        assert AuthorityPolicy.is_authoritative(record) is False
        
        registry.add_record(record)
        decision = await engine.evaluate(f"0xinv_{invalid_id}", Chain.EVM)
        assert decision.confidence != AttributionConfidence.CONFIRMED

@pytest.mark.asyncio
async def test_authority_contract_fake_official_strings(engine, registry):
    """Test 6: Strings like official-demo, unofficial, not-official cannot pass through loose matching."""
    for fake_source in ["official-demo", "unofficial", "not-official", "official_registry", "trusted", "certified"]:
        record = VASPAddressRecord(
            address=f"0xfake_{fake_source}", chain=Chain.EVM, vasp_name="FakeVASP", service_type="exchange",
            source=fake_source, evidence_id="ev_real_123", metadata={}
        )
        assert AuthorityPolicy.is_authoritative(record) is False
        registry.add_record(record)
        decision = await engine.evaluate(f"0xfake_{fake_source}", Chain.EVM)
        assert decision.confidence != AttributionConfidence.CONFIRMED

@pytest.mark.asyncio
async def test_authority_contract_cluster_semantics_preserved(engine, registry):
    """Test 7: member_count > 1 still cannot independently elevate authority or confidence."""
    record = VASPAddressRecord(
        address="0xcluster_target", chain=Chain.EVM, vasp_name="ClusterVASP", service_type="exchange",
        source="community_tag", evidence_id="ev_comm_1", metadata={}
    )
    registry.add_record(record)
    
    # Non-authoritative observation in a 10-member cluster with unlabelled siblings
    members = [
        ClusterMember(cluster_id="c-auth", composite_id=f"0xcluster_{i}_evm", chain="evm", network="ethereum", normalized_address=f"0xcluster_{i}")
        for i in range(10)
    ]
    members[0] = ClusterMember(cluster_id="c-auth", composite_id="0xcluster_target_evm", chain="evm", network="ethereum", normalized_address="0xcluster_target")
    
    cluster = AddressCluster(
        cluster_id="c-auth", representative_composite_id="0xcluster_target_evm", chain=Chain.EVM.value, network="ethereum",
        members=members
    )
    
    # Authority cannot be elevated
    assert AuthorityPolicy.is_authoritative(record) is False
    assert AuthorityPolicy.classify_observation(record) == AuthorityClass.NON_AUTHORITATIVE
    
    decision = await engine.evaluate("0xcluster_target", Chain.EVM, cluster)
    # Confidence must NOT be elevated to HIGH_CONFIDENCE or CONFIRMED
    assert decision.confidence == AttributionConfidence.PROBABLE_INFERRED
    assert decision.provenance.corroborating_evidence_count == 0


