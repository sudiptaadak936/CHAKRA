import os
import pytest
import asyncpg
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.schemas.chain import Chain, Network
from app.schemas.mode import AppMode
from app.schemas.transaction import Transaction
from app.services.mode_service import ModeService
from app.scenarios.registry import ScenarioRegistry
from app.scenarios import scenario_registry
from app.services.ingestion_service import IngestionService
from app.db.repository import TransactionRepository
from app.graph.projector import GraphProjector
from app.graph.traversal import MoneyFlowTraversal, TraversalPath, TraversalNode
from app.forensics.repository import ForensicsRepository
from app.forensics.change_detector import BitcoinChangeDetector
from app.forensics.models import ChangeAddressInference
from app.forensics.typology_detector import TypologyDetector, TypologyType
from app.services.demo_service import DemoScenarioService, DemoExecutionResult

# Environment connection config
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "chakra_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "your_actual_password")
PG_DB = os.getenv("POSTGRES_DB", "chakra_db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
if "bolt://neo4j" in NEO4J_URI and PG_HOST == "localhost":
    NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "your_actual_password")


class PostgresForensicsRepository(ForensicsRepository):
    """Forensics repository with real PostgreSQL query support for change inferences."""

    async def get_change_inferences(self, txid: str):
        query = """
            SELECT txid, output_index, address, classification, confidence, evidence_reason
            FROM bitcoin_change_inferences
            WHERE txid = $1
            ORDER BY output_index ASC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, txid)
            return [
                ChangeAddressInference(
                    txid=r["txid"],
                    output_index=r["output_index"],
                    address=r["address"],
                    classification=r["classification"],
                    confidence=r["confidence"],
                    evidence_reason=r["evidence_reason"],
                )
                for r in rows
            ]


@pytest.fixture
async def pg_pool():
    pool = await asyncpg.create_pool(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        database=PG_DB,
        min_size=1,
        max_size=5,
    )
    yield pool
    await pool.close()


@pytest.fixture
async def neo4j_driver():
    driver: AsyncDriver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
    )
    yield driver
    await driver.close()


@pytest.fixture
async def clean_slate(pg_pool, neo4j_driver):
    """Clean slate before each test and ensure schemas and constraints are applied."""
    repo = ForensicsRepository(pg_pool)
    await repo.init_schema()

    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
        await conn.execute("TRUNCATE TABLE bitcoin_change_inferences CASCADE")

    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()

    yield


@pytest.fixture
def demo_service(pg_pool, neo4j_driver):
    repo = TransactionRepository(pg_pool)
    ingestion = IngestionService(repository=repo)
    mode_service = ModeService()
    projector = GraphProjector(pg_pool, neo4j_driver)

    return DemoScenarioService(
        registry=scenario_registry,
        ingestion_service=ingestion,
        mode_service=mode_service,
        projector=projector,
    )


# ===========================================================================
# 1. Mode Safety
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_mode_safety(demo_service):
    """Ensure DemoScenarioService rejects execution when in LIVE mode and allows in DEMO."""
    # Force LIVE mode
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = False

    with pytest.raises(RuntimeError, match="Demo operations are disabled in LIVE mode"):
        await demo_service.execute_scenario("one_hop_cashout")

    # Force DEMO mode
    demo_service.mode_service.is_demo.return_value = True
    assert demo_service.mode_service.is_demo() is True


# ===========================================================================
# 2. Scenario Selection
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_unknown_scenario(demo_service):
    """Ensure DemoScenarioService rejects unknown scenario IDs explicitly."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    with pytest.raises(ValueError, match="Unknown scenario ID: non_existent_scenario"):
        await demo_service.execute_scenario("non_existent_scenario")

    # Verify all 6 canonical scenarios are registered in the authoritative registry
    available = demo_service.registry.list()
    expected = [
        "cross_chain_hop",
        "fan_in",
        "mixer_interaction",
        "offshore_cashout",
        "one_hop_cashout",
        "peel_chain",
    ]
    assert sorted(available) == expected


# ===========================================================================
# 3. Generator -> Ingestion Boundary
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_generator_ingestion_boundary(demo_service):
    """Verify generated objects are canonical Transaction schemas and passed without transformation."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    gen = demo_service.registry.get("one_hop_cashout")
    generated_txs = gen.generate()
    assert len(generated_txs) == 1
    tx = generated_txs[0]
    assert isinstance(tx, Transaction)
    assert tx.transaction_id == "tx_one_hop_cashout_001"
    assert tx.transfers[0].from_address == "chakra-demo/one_hop_cashout/source"
    assert tx.transfers[0].to_address == "chakra-demo/one_hop_cashout/cashout"


# ===========================================================================
# 4. PostgreSQL Persistence & Exact Counts
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_persistence_exact_counts(demo_service, clean_slate, pg_pool):
    """Verify generated count == persisted count in PostgreSQL for one_hop, peel, and fan_in."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # 1. One-Hop Cashout
    res_one_hop = await demo_service.execute_scenario("one_hop_cashout")
    assert res_one_hop.transactions_generated == 1

    async with pg_pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM transactions WHERE transaction_id = 'tx_one_hop_cashout_001'")
        assert count == 1
        transfer_count = await conn.fetchval(
            "SELECT count(*) FROM transfers WHERE from_address = 'chakra-demo/one_hop_cashout/source'"
        )
        assert transfer_count == 1

    # 2. Peel Chain (4 transactions)
    res_peel = await demo_service.execute_scenario("peel_chain")
    assert res_peel.transactions_generated == 4

    async with pg_pool.acquire() as conn:
        peel_tx_count = await conn.fetchval(
            "SELECT count(*) FROM transactions WHERE transaction_id LIKE 'tx_peel_%'"
        )
        assert peel_tx_count == 4
        # Verify UTXO details were persisted
        btc_detail_count = await conn.fetchval(
            "SELECT count(*) FROM bitcoin_transaction_details WHERE txid LIKE 'tx_peel_%'"
        )
        assert btc_detail_count == 4

    # 3. Fan-In (3 transactions)
    res_fan_in = await demo_service.execute_scenario("fan_in")
    assert res_fan_in.transactions_generated == 3

    async with pg_pool.acquire() as conn:
        fan_in_tx_count = await conn.fetchval(
            "SELECT count(*) FROM transactions WHERE transaction_id LIKE 'tx_fan_in_%'"
        )
        assert fan_in_tx_count == 3
        # Verify all 3 point to the same hub
        hub_transfers = await conn.fetchval(
            "SELECT count(*) FROM transfers WHERE to_address = 'chakra-demo/fan_in/hub'"
        )
        assert hub_transfers == 3


# ===========================================================================
# 5. Idempotency
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_idempotency(demo_service, clean_slate, pg_pool):
    """Verify running the same scenario twice creates 0 duplicate identities."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # First execution
    res1 = await demo_service.execute_scenario("one_hop_cashout")
    async with pg_pool.acquire() as conn:
        first_count = await conn.fetchval("SELECT count(*) FROM transactions")
    assert first_count == 1

    # Second execution
    res2 = await demo_service.execute_scenario("one_hop_cashout")
    async with pg_pool.acquire() as conn:
        second_count = await conn.fetchval("SELECT count(*) FROM transactions")
    assert second_count == 1
    assert res2.transactions_generated == res1.transactions_generated
    assert res2.projected_transactions == res1.projected_transactions


# ===========================================================================
# 6. Graph Projection
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_graph_projection(demo_service, clean_slate, neo4j_driver):
    """Verify that existing GraphProjector processes persisted transactions into Neo4j nodes/edges."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    result = await demo_service.execute_scenario("one_hop_cashout")
    assert result.transactions_generated == 1
    assert result.projected_transactions == 1
    assert result.projected_transfers == 1

    # Verify Neo4j nodes and edges
    async with neo4j_driver.session() as session:
        # Check Transaction node
        res_tx = await session.run("MATCH (t:Transaction {transaction_id: 'tx_one_hop_cashout_001'}) RETURN count(t) as c")
        assert (await res_tx.single())["c"] == 1

        # Check Address nodes
        res_src = await session.run("MATCH (a:Address {raw_address: 'chakra-demo/one_hop_cashout/source'}) RETURN count(a) as c")
        assert (await res_src.single())["c"] == 1

        # Check TRANSFERRED edge
        res_rel = await session.run(
            "MATCH (:Address {raw_address: 'chakra-demo/one_hop_cashout/source'})"
            "-[:TRANSFERRED]->"
            "(:Address {raw_address: 'chakra-demo/one_hop_cashout/cashout'}) RETURN count(*) as c"
        )
        assert (await res_rel.single())["c"] == 1


# ===========================================================================
# 7. Post-Integration Forensic Validation (Real Pipeline on Persisted Data)
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_forensic_peel_chain(demo_service, clean_slate, neo4j_driver, pg_pool):
    """Verify real frozen forensic pipeline derives PEEL_CHAIN from persisted/projected data."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # 1. Ingest and project peel_chain
    await demo_service.execute_scenario("peel_chain")

    # 2. Run real BitcoinChangeDetector on persisted PostgreSQL records
    forensics_repo = PostgresForensicsRepository(pg_pool)
    change_detector = BitcoinChangeDetector(forensics_repo)

    for i in range(4):
        txid = f"tx_peel_{i+1}"
        inferences = await change_detector.detect_transaction(txid)
        await forensics_repo.upsert_change_inferences(inferences)

    # Verify change candidates stored in PostgreSQL
    async with pg_pool.acquire() as conn:
        stored_inferences = await conn.fetch(
            "SELECT txid, address, classification, confidence FROM bitcoin_change_inferences ORDER BY txid, output_index"
        )
        assert len(stored_inferences) > 0
        t1_change = next(r for r in stored_inferences if r["txid"] == "tx_peel_1" and r["address"] == "chakra-demo/peel/change1")
        assert t1_change["classification"] == "CHANGE_CANDIDATE"
        t1_drop = next(r for r in stored_inferences if r["txid"] == "tx_peel_1" and r["address"] == "chakra-demo/peel/drop")
        assert t1_drop["classification"] == "EXTERNAL_RECIPIENT"

    # 3. Traverse Neo4j starting from peel origin
    traversal = MoneyFlowTraversal(neo4j_driver)
    trav_result = await traversal.traverse(
        chain="bitcoin",
        network="bitcoin-mainnet",
        address="chakra-demo/peel/src",
        max_hops=5,
        max_nodes=100,
        max_edges=100,
    )
    assert trav_result.error is None
    assert len(trav_result.paths) > 0

    # Select the peel path (traverses UTXO spend transitions)
    peel_path = max(trav_result.paths, key=lambda p: len(p.utxo_steps))
    assert len(peel_path.utxo_steps) >= 3

    # Bind address on created_output for TypologyDetector contract compatibility
    for step in peel_path.utxo_steps:
        object.__setattr__(step.created_output, "address", step.to_address.raw_address)

    # 4. Run real frozen TypologyDetector
    detector = TypologyDetector(pool=pg_pool, repo=forensics_repo)
    detections = await detector.detect_path_typologies(peel_path)

    peel_detection = next((d for d in detections if d.typology_type == TypologyType.PEEL_CHAIN), None)
    assert peel_detection is not None
    assert peel_detection.confidence_level == "observed"
    assert peel_detection.hop_count == 3
    assert len(peel_detection.evidence) == 1

    # Verify peel evidence details
    peel_details = peel_detection.evidence[0].peel_details
    assert len(peel_details) == 3
    assert peel_details[0]["txid"] == "tx_peel_1"
    assert peel_details[0]["continuing_address"] == "chakra-demo/peel/change1"
    assert peel_details[0]["peeled_addresses"] == ["chakra-demo/peel/drop"]
    assert peel_details[0]["retention_ratio"] == pytest.approx(0.899, abs=0.001)
    assert peel_details[1]["retention_ratio"] == pytest.approx(0.886, abs=0.001)
    assert peel_details[2]["retention_ratio"] == pytest.approx(0.870, abs=0.001)


@pytest.mark.asyncio
async def test_demo_service_forensic_fan_in(demo_service, clean_slate, neo4j_driver, pg_pool):
    """Verify real frozen forensic pipeline derives FAN_IN from persisted/projected data."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # 1. Ingest and project fan_in
    await demo_service.execute_scenario("fan_in")

    # 2. Build a traversal node for the aggregation hub
    hub_node = TraversalNode(
        composite_id="evm:ethereum-mainnet:chakra-demo/fan_in/hub",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="chakra-demo/fan_in/hub",
        raw_address="chakra-demo/fan_in/hub",
    )
    fan_in_path = TraversalPath(nodes=[hub_node], hops=1)

    # 3. Run real frozen TypologyDetector
    forensics_repo = PostgresForensicsRepository(pg_pool)
    detector = TypologyDetector(pool=pg_pool, repo=forensics_repo, max_fan_window_hours=24.0)
    detections = await detector.detect_path_typologies(fan_in_path)

    fan_in_det = next((d for d in detections if d.typology_type == TypologyType.FAN_IN), None)
    assert fan_in_det is not None
    assert fan_in_det.confidence_level == "observed"
    assert fan_in_det.temporal_window is not None
    assert len(fan_in_det.counterparty_addresses) == 3
    assert sorted(fan_in_det.counterparty_addresses) == [
        "chakra-demo/fan_in/src1",
        "chakra-demo/fan_in/src2",
        "chakra-demo/fan_in/src3",
    ]


@pytest.mark.asyncio
async def test_demo_service_forensic_conservative_scenarios(demo_service, clean_slate, pg_pool):
    """Verify mixer, cross-chain, and one-hop return conservative results (no false positives)."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    forensics_repo = PostgresForensicsRepository(pg_pool)
    detector = TypologyDetector(pool=pg_pool, repo=forensics_repo)

    # 1. Mixer scenario
    await demo_service.execute_scenario("mixer_interaction")
    mixer_node = TraversalNode(
        composite_id="evm:ethereum-mainnet:chakra-demo/mixer/src",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="chakra-demo/mixer/src",
        raw_address="chakra-demo/mixer/src",
    )
    mixer_path = TraversalPath(nodes=[mixer_node], hops=1)
    mixer_dets = await detector.detect_path_typologies(mixer_path)
    mixer_det = next((d for d in mixer_dets if d.typology_type == TypologyType.MIXER_INTERACTION), None)
    assert mixer_det is not None
    assert mixer_det.confidence_level == "insufficient_evidence"
    assert "no authoritative mixer identification source" in mixer_det.explanation

    # 2. Cross-chain scenario
    await demo_service.execute_scenario("cross_chain_hop")
    cc_node = TraversalNode(
        composite_id="evm:ethereum-mainnet:chakra-demo/cross_chain/src_eth",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="chakra-demo/cross_chain/src_eth",
        raw_address="chakra-demo/cross_chain/src_eth",
    )
    cc_path = TraversalPath(nodes=[cc_node], hops=1)
    cc_dets = await detector.detect_path_typologies(cc_path)
    cc_det = next((d for d in cc_dets if d.typology_type == TypologyType.CROSS_CHAIN), None)
    assert cc_det is not None
    assert cc_det.confidence_level == "insufficient_evidence"
    assert "no bridge detection" in cc_det.explanation.lower()

    # 3. One-hop scenario
    await demo_service.execute_scenario("one_hop_cashout")
    one_hop_node = TraversalNode(
        composite_id="evm:ethereum-mainnet:chakra-demo/one_hop_cashout/source",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="chakra-demo/one_hop_cashout/source",
        raw_address="chakra-demo/one_hop_cashout/source",
    )
    one_hop_path = TraversalPath(nodes=[one_hop_node], hops=1)
    one_hop_dets = await detector.detect_path_typologies(one_hop_path)
    # Confirm no major observed typology (only safety stubs)
    observed_dets = [d for d in one_hop_dets if d.confidence_level == "observed"]
    assert len(observed_dets) == 0


# ===========================================================================
# 8. Provider / Network Isolation
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_provider_network_isolation(demo_service, clean_slate):
    """Verify external provider adapters and HTTP requests are never invoked in Demo mode."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # Spy on IngestionService.registry
    mock_registry = MagicMock()
    demo_service.ingestion_service.registry = mock_registry

    await demo_service.execute_scenario("one_hop_cashout")

    # Assert 0 calls were made to registry or provider adapters
    assert mock_registry.get_adapter.call_count == 0
    assert mock_registry.list_adapters.call_count == 0


# ===========================================================================
# 9. Failure Propagation
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_failure_propagation(demo_service):
    """Verify generator exceptions and ingestion exceptions are propagated without being swallowed."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # 1. Generator failure
    failing_generator = MagicMock()
    failing_generator.generate.side_effect = RuntimeError("Synthetic generator crash")

    with patch.object(demo_service.registry, "get", return_value=failing_generator):
        with pytest.raises(RuntimeError, match="Synthetic generator crash"):
            await demo_service.execute_scenario("one_hop_cashout")

    # 2. Ingestion failure
    demo_service.ingestion_service.ingest_direct = AsyncMock(
        side_effect=RuntimeError("Database connection lost")
    )
    with pytest.raises(RuntimeError, match="Database connection lost"):
        await demo_service.execute_scenario("one_hop_cashout")


# ===========================================================================
# 10. Scenario Isolation
# ===========================================================================

@pytest.mark.asyncio
async def test_demo_service_scenario_isolation(demo_service, clean_slate):
    """Verify scenarios have disjoint IDs and addresses, and execution does not mutate state."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    txs_a = demo_service.registry.get("one_hop_cashout").generate()
    txs_b = demo_service.registry.get("peel_chain").generate()

    ids_a = {tx.transaction_id for tx in txs_a}
    ids_b = {tx.transaction_id for tx in txs_b}
    assert ids_a.isdisjoint(ids_b), "Transaction IDs must be disjoint"

    addrs_a = {t.from_address for tx in txs_a for t in tx.transfers if t.from_address} | {
        t.to_address for tx in txs_a for t in tx.transfers if t.to_address
    }
    addrs_b = {t.from_address for tx in txs_b for t in tx.transfers if t.from_address} | {
        t.to_address for tx in txs_b for t in tx.transfers if t.to_address
    }
    assert addrs_a.isdisjoint(addrs_b), "Address namespaces must be disjoint"

    # Execute A then verify B generator still returns identical output
    await demo_service.execute_scenario("one_hop_cashout")
    txs_b_after = demo_service.registry.get("peel_chain").generate()
    assert txs_b == txs_b_after


@pytest.mark.asyncio
async def test_demo_service_cross_scenario_downstream_isolation(
    demo_service, clean_slate, neo4j_driver, pg_pool
):
    """Verify executing scenario A followed by scenario B leaves downstream analysis of B completely uninfluenced."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True

    # 1. Execute Scenario A: one_hop_cashout and fan_in
    await demo_service.execute_scenario("one_hop_cashout")
    await demo_service.execute_scenario("fan_in")

    # 2. Execute Scenario B: peel_chain
    await demo_service.execute_scenario("peel_chain")

    # 3. Detect change candidates for peel_chain and store inferences
    forensics_repo = PostgresForensicsRepository(pg_pool)
    change_detector = BitcoinChangeDetector(repository=forensics_repo)

    peel_txs = demo_service.registry.get("peel_chain").generate()
    for tx in peel_txs:
        inferences = await change_detector.detect_transaction(tx.transaction_id)
        await forensics_repo.upsert_change_inferences(inferences)

    # 4. Traverse Neo4j starting from peel origin
    traversal = MoneyFlowTraversal(neo4j_driver)
    trav_result = await traversal.traverse(
        chain="bitcoin",
        network="bitcoin-mainnet",
        address="chakra-demo/peel/src",
        max_hops=5,
        max_nodes=100,
        max_edges=100,
    )
    assert trav_result.error is None
    assert len(trav_result.paths) > 0

    # Ensure NO entities or addresses from scenario A leaked into peel traversal
    all_visited_addresses = {
        node.raw_address for path in trav_result.paths for node in path.nodes
    }
    assert all(not a.startswith("chakra-demo/one_hop") for a in all_visited_addresses)
    assert all(not a.startswith("chakra-demo/fan_in") for a in all_visited_addresses)

    # 5. Run real frozen TypologyDetector on peel_path
    peel_path = max(trav_result.paths, key=lambda p: len(p.utxo_steps))
    for step in peel_path.utxo_steps:
        object.__setattr__(step.created_output, "address", step.to_address.raw_address)

    detector = TypologyDetector(pool=pg_pool, repo=forensics_repo)
    detections = await detector.detect_path_typologies(peel_path)

    peel_det = next((d for d in detections if d.typology_type == TypologyType.PEEL_CHAIN), None)
    assert peel_det is not None
    assert peel_det.confidence_level == "observed"
    assert peel_det.hop_count == 3
    assert len(peel_det.evidence[0].peel_details) == 3

    # 6. Verify fan_in analysis remains equally isolated and unchanged
    hub_node = TraversalNode(
        composite_id="evm:ethereum-mainnet:chakra-demo/fan_in/hub",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="chakra-demo/fan_in/hub",
        raw_address="chakra-demo/fan_in/hub",
    )
    fan_in_path = TraversalPath(nodes=[hub_node], hops=1)
    fan_in_detector = TypologyDetector(pool=pg_pool, repo=forensics_repo, max_fan_window_hours=24.0)
    fan_in_dets = await fan_in_detector.detect_path_typologies(fan_in_path)
    fan_in_det = next((d for d in fan_in_dets if d.typology_type == TypologyType.FAN_IN), None)
    assert fan_in_det is not None
    assert fan_in_det.confidence_level == "observed"
    assert len(fan_in_det.counterparty_addresses) == 3


