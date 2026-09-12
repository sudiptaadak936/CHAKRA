import os
import pytest
import asyncpg
from app.attribution.models import VASPAddressRecord
from app.attribution.registry import VASPAttributionRegistry
from app.attribution.repository import VASPAttributionRepository
from app.schemas.chain import Chain

PG_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "chakra_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "your_actual_password")
PG_DB = os.getenv("POSTGRES_DB", "chakra_db")

@pytest.fixture
async def pg_pool():
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    user = os.environ.get("POSTGRES_USER", "chakra_user")
    password = os.environ.get("POSTGRES_PASSWORD", "your_actual_password")
    db = os.environ.get("POSTGRES_DB", "chakra_db")
    
    pool = await asyncpg.create_pool(
        host=host, port=port, user=user, password=password, database=db,
        min_size=1, max_size=5,
    )
    yield pool
    await pool.close()

@pytest.fixture
async def repo(pg_pool):
    repository = VASPAttributionRepository(pg_pool)
    await repository.init_schema()
    await repository._clear_all_for_testing()
    return repository

@pytest.fixture
async def registry(repo):
    return VASPAttributionRegistry(repo)

@pytest.mark.asyncio
async def test_exact_lookup(repo, registry):
    record = VASPAddressRecord(
        address="0x123",
        chain=Chain.EVM,
        vasp_name="Binance",
        service_type="exchange",
        source="tag",
        evidence_id="1",
        metadata={}
    )
    await repo.save_record(record)
    
    results = await registry.lookup("0x123", "evm")
    assert len(results) == 1
    assert results[0].address == "0x123"
    assert results[0].vasp_name == "Binance"

@pytest.mark.asyncio
async def test_unknown_address(registry):
    results = await registry.lookup("0xUNKNOWN", "evm")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_chain_isolation(repo, registry):
    record = VASPAddressRecord(
        address="0x123",
        chain=Chain.EVM,
        vasp_name="Binance",
        service_type="exchange",
        source="tag",
        evidence_id="1",
        metadata={}
    )
    await repo.save_record(record)
    
    results = await registry.lookup("0x123", "tron")
    assert len(results) == 0
    results_evm = await registry.lookup("0x123", "evm")
    assert len(results_evm) == 1

@pytest.mark.asyncio
async def test_provenance_preservation(repo, registry):
    record = VASPAddressRecord(
        address="TG3X",
        chain=Chain.TRON,
        vasp_name="Tether",
        service_type="issuer",
        source="manual",
        evidence_id="ev-123",
        metadata={"reviewer": "Alice"}
    )
    await repo.save_record(record)
    
    results = await registry.lookup("TG3X", "tron")
    assert len(results) == 1
    assert results[0].source == "manual"
    assert results[0].evidence_id == "ev-123"
    assert results[0].metadata["reviewer"] == "Alice"

@pytest.mark.asyncio
async def test_multiple_candidates(repo, registry):
    record1 = VASPAddressRecord(
        address="0xabc", chain=Chain.EVM, vasp_name="Binance",
        service_type="exchange", source="tag1", evidence_id="1", metadata={}
    )
    record2 = VASPAddressRecord(
        address="0xabc", chain=Chain.EVM, vasp_name="Huobi",
        service_type="exchange", source="tag2", evidence_id="2", metadata={}
    )
    await repo.save_record(record1)
    await repo.save_record(record2)
    
    results = await registry.lookup("0xabc", "evm")
    assert len(results) == 2
    names = {r.vasp_name for r in results}
    assert "Binance" in names
    assert "Huobi" in names

@pytest.mark.asyncio
async def test_normalization(repo, registry):
    record = VASPAddressRecord(
        address="0xABCdef", chain=Chain.EVM, vasp_name="Binance",
        service_type="exchange", source="tag", evidence_id="1", metadata={}
    )
    await repo.save_record(record)
    
    # lookup with mixed case, should find the lowercase stored one
    results = await registry.lookup("0xABCDEF", "evm")
    assert len(results) == 1
    assert results[0].address == "0xabcdef" # normalizes to lower for EVM

@pytest.mark.asyncio
async def test_duplicate_handling(repo, registry):
    record1 = VASPAddressRecord(
        address="0xabc", chain=Chain.EVM, vasp_name="Binance",
        service_type="exchange", source="tag", evidence_id="1", metadata={"v": 1}
    )
    record2 = VASPAddressRecord(
        address="0xabc", chain=Chain.EVM, vasp_name="Binance",
        service_type="exchange", source="tag", evidence_id="2", metadata={"v": 2}
    )
    # Both have same (address, chain, vasp_name, source)
    await repo.save_record(record1)
    await repo.save_record(record2) # Should update
    
    results = await registry.lookup("0xabc", "evm")
    assert len(results) == 1
    assert results[0].evidence_id == "2"
    assert results[0].metadata["v"] == 2

@pytest.mark.asyncio
async def test_malformed_record_handling():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        VASPAddressRecord(
            address="", # Invalid
            chain=Chain.EVM,
            vasp_name="Binance",
            service_type="exchange",
            source="tag",
            evidence_id="1"
        )

@pytest.mark.asyncio
async def test_bitcoin_normalization(repo, registry):
    record = VASPAddressRecord(
        address="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", chain=Chain.BITCOIN, vasp_name="Genesis",
        service_type="miner", source="tag", evidence_id="1", metadata={}
    )
    await repo.save_record(record)
    
    results = await registry.lookup("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "bitcoin")
    assert len(results) == 1
    assert results[0].address == "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" # Case preserved

@pytest.mark.asyncio
async def test_integration_with_step3_wallet(repo, registry):
    class MockClusterMember:
        def __init__(self, address):
            self.address = address
    class MockCluster:
        def __init__(self):
            self.chain = Chain.EVM
            self.members = [MockClusterMember("0xabc"), MockClusterMember("0xdef")]
            
    cluster = MockCluster()  
    # Create attribution for one address
    record = VASPAddressRecord(
        address="0xabc", chain=Chain.EVM, vasp_name="Binance",
        service_type="exchange", source="tag", evidence_id="1", metadata={}
    )
    await repo.save_record(record)
    
    # Simulate step 4 checking all addresses in a Step 3 cluster
    cluster_attributions = []
    for member in cluster.members:
        res = await registry.lookup(member.address, cluster.chain.value)
        cluster_attributions.extend(res)
        
    assert len(cluster_attributions) == 1
    assert cluster_attributions[0].vasp_name == "Binance"
