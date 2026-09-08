"""CHAKRA Step 2B: Deterministic Money-Flow Traversal Test Suite.

Tests A through Q as specified in the Step 2B specification.

Prerequisite: Step 2A baseline of 178 tests must continue passing.
All tests are read-only with respect to Neo4j and PostgreSQL after setup.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import asyncpg
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.schemas.chain import Chain, Network
from app.schemas.transaction import (
    Transaction,
    Transfer,
    TransactionProvenance,
    AssetType,
    TransactionStatus,
    TransactionType,
)
from app.db.repository import TransactionRepository
from app.graph.projector import GraphProjector
from app.graph.traversal import (
    MoneyFlowTraversal,
    TraversalParameterError,
    MAX_HOPS_CEILING,
    MAX_NODES_CEILING,
    MAX_EDGES_CEILING,
)
from app.graph.models import make_address_composite_id, make_transfer_relationship_id

# ---------------------------------------------------------------------------
# Connection configuration (mirrors test_graph_projection.py)
# ---------------------------------------------------------------------------

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "chakra_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "your_actual_password")
PG_DB   = os.getenv("POSTGRES_DB",       "chakra_db")

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
if "bolt://neo4j" in NEO4J_URI and PG_HOST == "localhost":
    NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "your_actual_password")


# ---------------------------------------------------------------------------
# Fixtures (function-scoped — module scope causes event-loop conflicts)
# ---------------------------------------------------------------------------


@pytest.fixture
async def pg_pool():
    pool = await asyncpg.create_pool(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASS, database=PG_DB,
        min_size=1, max_size=5,
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
async def clean_env(pg_pool, neo4j_driver):
    """Wipe PostgreSQL transactions and all Neo4j data; re-apply constraints."""
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()
    repo     = TransactionRepository(pg_pool)
    traversal = MoneyFlowTraversal(neo4j_driver)
    return repo, projector, traversal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evm_transfer(
    tx_id: str,
    from_addr: str,
    to_addr: str,
    amount: int = 1_000_000_000_000_000_000,
    asset_type: AssetType = AssetType.NATIVE,
    asset_symbol: str = "ETH",
    asset_contract: str | None = None,
    network: Network = Network.ETH_MAINNET,
) -> Transaction:
    return Transaction(
        transaction_id=tx_id,
        chain=Chain.EVM,
        network=network,
        chain_id=1,
        block_number=100,
        block_hash="0xblockhash",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        from_address=from_addr,
        to_address=to_addr,
        native_value=amount,
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21000,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address=from_addr,
                to_address=to_addr,
                asset_type=asset_type,
                asset_symbol=asset_symbol,
                asset_contract=asset_contract,
                amount=amount,
                amount_unit="wei",
                decimals=18,
                chain=Chain.EVM,
                network=network,
            )
        ],
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.EVM,
            network=network,
            original_id=tx_id,
        ),
    )


# ===========================================================================
# A. Single-Hop Traversal
# ===========================================================================
@pytest.mark.asyncio
async def test_a_single_hop(clean_env):
    """A single TRANSFERRED edge A->B is returned as a one-hop path."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    tx = _evm_transfer("0xtx_a1", addr_a, addr_b)
    await repo.save_transaction(tx)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=1000,
    )

    assert result.error is None
    assert result.total_paths >= 1
    assert result.truncated is False
    # The path A->B must be present.
    path_nodes = [p.nodes for p in result.paths]
    found = any(
        len(p) == 2
        and p[0].normalized_address == addr_a.lower()
        and p[1].normalized_address == addr_b.lower()
        for p in path_nodes
    )
    assert found, f"Expected A->B path; got paths: {path_nodes}"
    for path in result.paths:
        assert path.hops == len(path.edges)
        assert len(path.nodes) == path.hops + 1


# ===========================================================================
# B. Multi-Hop Traversal
# ===========================================================================
@pytest.mark.asyncio
async def test_b_multi_hop(clean_env):
    """A->B->C is returned when max_hops >= 2."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_b1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_b2", addr_b, addr_c))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=2,
        max_nodes=100,
        max_edges=1000,
    )

    assert result.error is None
    assert result.truncated is False

    addrs_per_path = [tuple(n.normalized_address for n in p.nodes) for p in result.paths]
    # 1-hop path A->B
    assert (addr_a.lower(), addr_b.lower()) in addrs_per_path
    # 2-hop path A->B->C
    assert (addr_a.lower(), addr_b.lower(), addr_c.lower()) in addrs_per_path


# ===========================================================================
# C. max_hops Limit Enforced
# ===========================================================================
@pytest.mark.asyncio
async def test_c_max_hops_limit(clean_env):
    """When max_hops=1, paths do not extend past one hop even if graph is deeper."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_c1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_c2", addr_b, addr_c))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=1000,
    )

    assert result.error is None
    for path in result.paths:
        assert path.hops <= 1, f"Expected hops <= 1, got {path.hops}"

    addrs_per_path = [tuple(n.normalized_address for n in p.nodes) for p in result.paths]
    assert (addr_a.lower(), addr_b.lower(), addr_c.lower()) not in addrs_per_path


# ===========================================================================
# D. max_nodes Limit Enforced
# ===========================================================================
@pytest.mark.asyncio
async def test_d_max_nodes_limit(clean_env):
    """When max_nodes=2, traversal stops after admitting 2 distinct nodes."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_d1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_d2", addr_b, addr_c))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=2,
        max_edges=1000,
    )

    assert result.error is None
    assert result.truncated is True
    assert result.total_nodes_visited <= 2


# ===========================================================================
# E. max_edges Limit Enforced
# ===========================================================================
@pytest.mark.asyncio
async def test_e_max_edges_limit(clean_env):
    """When max_edges=1, traversal stops after visiting one TRANSFERRED edge."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_e1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_e2", addr_a, addr_c))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=1000,
        max_edges=1,
    )

    assert result.error is None
    assert result.truncated is True
    assert result.total_edges_visited <= 1


# ===========================================================================
# F. Cycle Detection (Simple-Path Semantics)
# ===========================================================================
@pytest.mark.asyncio
async def test_f_cycle_detection(clean_env):
    """A->B->A cycle: no path should contain the same address twice."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await repo.save_transaction(_evm_transfer("0xtx_f1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_f2", addr_b, addr_a))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=1000,
        max_edges=1000,
    )

    assert result.error is None
    for path in result.paths:
        seen = [n.composite_id for n in path.nodes]
        assert len(seen) == len(set(seen)), (
            f"Cycle detected in path: {seen}"
        )


# ===========================================================================
# G. Deterministic Ordering by transfer_id
# ===========================================================================
@pytest.mark.asyncio
async def test_g_deterministic_ordering(clean_env):
    """Repeated traversals return paths in the same order (deterministic by transfer_id)."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_g1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_g2", addr_a, addr_c))
    await projector.project_all()

    result1 = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=1000,
        max_edges=1000,
    )
    result2 = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=1000,
        max_edges=1000,
    )

    assert result1.error is None
    assert result2.error is None
    seq1 = [e.transfer_id for p in result1.paths for e in p.edges]
    seq2 = [e.transfer_id for p in result2.paths for e in p.edges]
    assert seq1 == seq2, f"Non-deterministic ordering: {seq1} vs {seq2}"


# ===========================================================================
# H. Cross-Chain Isolation
# ===========================================================================
@pytest.mark.asyncio
async def test_h_cross_chain_isolation(clean_env):
    """EVM-mainnet traversal must not return Tron addresses on the same path."""
    repo, projector, traversal = clean_env
    evm_a   = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    evm_b   = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    tron_a  = "TRX_addr_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    tron_b  = "TRX_addr_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    await repo.save_transaction(_evm_transfer("0xtx_h1", evm_a, evm_b, network=Network.ETH_MAINNET))

    # Tron transfer on a different chain
    tron_tx = Transaction(
        transaction_id="TRX_tx_h2",
        chain=Chain.TRON,
        network=Network.TRON_MAINNET,
        block_number=200,
        block_hash="TRX_block",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        from_address=tron_a,
        to_address=tron_b,
        native_value=1_000_000,
        native_value_unit="sun",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=1000,
        fee_asset="TRX",
        transfers=[
            Transfer(
                from_address=tron_a,
                to_address=tron_b,
                asset_type=AssetType.NATIVE,
                asset_symbol="TRX",
                amount=1_000_000,
                amount_unit="sun",
                decimals=6,
                chain=Chain.TRON,
                network=Network.TRON_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.TRON,
            network=Network.TRON_MAINNET,
            original_id="TRX_tx_h2",
        ),
    )
    await repo.save_transaction(tron_tx)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=evm_a,
        max_hops=5,
        max_nodes=1000,
        max_edges=1000,
    )

    assert result.error is None
    for path in result.paths:
        for node in path.nodes:
            assert node.chain == "evm", f"Non-EVM node in EVM traversal: {node}"
        for edge in path.edges:
            assert edge.chain == "evm", f"Non-EVM edge in EVM traversal: {edge}"


# ===========================================================================
# I. Large Integer Preservation (2^53+1 and beyond 2^63-1)
# ===========================================================================
@pytest.mark.asyncio
async def test_i_large_integer_preservation(clean_env):
    """amount_str is authoritative; amount may be None for beyond-int64 values."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    # 2^53 + 1: exceeds JavaScript/float safe integer range but fits in int64.
    beyond_float = (2 ** 53) + 1
    tx = _evm_transfer("0xtx_i1", addr_a, addr_b, amount=beyond_float)
    await repo.save_transaction(tx)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert len(result.paths) == 1
    edge = result.paths[0].edges[0]
    # amount_str must be exact
    assert edge.amount_str == str(beyond_float)
    # amount must be an int (fits in int64) or None
    if edge.amount is not None:
        assert edge.amount == beyond_float


# ===========================================================================
# J. Repeated Transfers Between Same Addresses
# ===========================================================================
@pytest.mark.asyncio
async def test_j_repeated_transfers(clean_env):
    """Multiple TRANSFERRED edges A->B from different transactions all appear."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    await repo.save_transaction(_evm_transfer("0xtx_j1", addr_a, addr_b, amount=1_000))
    await repo.save_transaction(_evm_transfer("0xtx_j2", addr_a, addr_b, amount=2_000))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    # Both edges from different transactions must be represented.
    assert result.total_edges_visited >= 2
    edge_tx_ids = {e.transaction_id for p in result.paths for e in p.edges}
    assert "0xtx_j1" in edge_tx_ids
    assert "0xtx_j2" in edge_tx_ids


# ===========================================================================
# K. Token Distinction (same addresses, different asset_symbol / asset_contract)
# ===========================================================================
@pytest.mark.asyncio
async def test_k_token_distinction(clean_env):
    """TRANSFERRED edges for different tokens between same A->B pair are distinct."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    contract_usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    contract_dai  = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

    await repo.save_transaction(_evm_transfer(
        "0xtx_k1", addr_a, addr_b,
        asset_type=AssetType.TOKEN, asset_symbol="USDC",
        asset_contract=contract_usdc,
    ))
    await repo.save_transaction(_evm_transfer(
        "0xtx_k2", addr_a, addr_b,
        asset_type=AssetType.TOKEN, asset_symbol="DAI",
        asset_contract=contract_dai,
    ))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    symbols = {e.asset_symbol for p in result.paths for e in p.edges}
    assert "USDC" in symbols
    assert "DAI"  in symbols
    contracts = {e.asset_contract for p in result.paths for e in p.edges}
    assert contract_usdc.lower() in {c.lower() if c else "" for c in contracts}
    assert contract_dai.lower()  in {c.lower() if c else "" for c in contracts}


# ===========================================================================
# L. No Fabricated Relationships
# ===========================================================================
@pytest.mark.asyncio
async def test_l_no_fabricated_relationships(clean_env):
    """Traversal returns zero paths when there are no outgoing TRANSFERRED edges."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    # Only inbound: B->A (so A has no outgoing edges).
    await repo.save_transaction(_evm_transfer("0xtx_l1", addr_b, addr_a))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=1000,
        max_edges=1000,
    )

    assert result.error is None
    assert result.total_paths == 0
    assert result.truncated is False


# ===========================================================================
# M. Bitcoin Boundary — explicit rejection
# ===========================================================================
@pytest.mark.asyncio
async def test_m_bitcoin_boundary(clean_env):
    """Bitcoin chain traversal is explicitly rejected with an informative error."""
    _, _, traversal = clean_env

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf Na",
        max_hops=3,
        max_nodes=100,
        max_edges=1000,
    )

    assert result.error is not None
    assert "bitcoin" in result.error.lower() or "Bitcoin" in result.error
    assert result.total_paths == 0


# ===========================================================================
# N. Start Address Not Found
# ===========================================================================
@pytest.mark.asyncio
async def test_n_start_not_found(clean_env):
    """Traversal of a non-existent start address returns an informative error."""
    _, _, traversal = clean_env

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address="0x0000000000000000000000000000000000000099",
        max_hops=3,
        max_nodes=100,
        max_edges=1000,
    )

    assert result.error is not None
    assert result.total_paths == 0


# ===========================================================================
# O. Zero Outgoing Transfers
# ===========================================================================
@pytest.mark.asyncio
async def test_o_zero_outgoing(clean_env):
    """A node with only incoming edges produces zero paths."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    # B receives from A but sends nothing.
    await repo.save_transaction(_evm_transfer("0xtx_o1", addr_a, addr_b))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_b,
        max_hops=5,
        max_nodes=1000,
        max_edges=1000,
    )

    assert result.error is None
    assert result.total_paths == 0
    assert result.truncated is False


# ===========================================================================
# P. Read-Only: No Side Effects on Neo4j
# ===========================================================================
@pytest.mark.asyncio
async def test_p_read_only(clean_env, neo4j_driver):
    """Traversal does not mutate the graph — node/relationship count unchanged."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    await repo.save_transaction(_evm_transfer("0xtx_p1", addr_a, addr_b))
    await projector.project_all()

    async with neo4j_driver.session() as session:
        r = await session.run("MATCH (n) RETURN count(n) AS cnt")
        before_nodes = (await r.single())["cnt"]
        r = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        before_rels = (await r.single())["cnt"]

    await traversal.traverse(
        chain=Chain.EVM.value,
        network=Network.ETH_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=1000,
        max_edges=1000,
    )

    async with neo4j_driver.session() as session:
        r = await session.run("MATCH (n) RETURN count(n) AS cnt")
        after_nodes = (await r.single())["cnt"]
        r = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        after_rels = (await r.single())["cnt"]

    assert before_nodes == after_nodes, "Traversal must not add/remove nodes"
    assert before_rels  == after_rels,  "Traversal must not add/remove relationships"


# ===========================================================================
# Q. Concurrent Execution
# ===========================================================================
@pytest.mark.asyncio
async def test_q_concurrent_execution(clean_env):
    """Multiple concurrent traversals return the same result as sequential runs."""
    repo, projector, traversal = clean_env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    await repo.save_transaction(_evm_transfer("0xtx_q1", addr_a, addr_b))
    await repo.save_transaction(_evm_transfer("0xtx_q2", addr_b, addr_c))
    await projector.project_all()

    async def _run():
        return await traversal.traverse(
            chain=Chain.EVM.value,
            network=Network.ETH_MAINNET.value,
            address=addr_a,
            max_hops=3,
            max_nodes=1000,
            max_edges=1000,
        )

    results = await asyncio.gather(*[_run() for _ in range(5)])

    ref = results[0]
    assert ref.error is None
    for r in results[1:]:
        assert r.error is None
        assert r.total_paths == ref.total_paths
        assert r.total_edges_visited == ref.total_edges_visited
        # Edge transfer_id sequences must be identical.
        seq_ref = [e.transfer_id for p in ref.paths for e in p.edges]
        seq_r   = [e.transfer_id for p in r.paths   for e in p.edges]
        assert seq_ref == seq_r, "Concurrent traversal produced non-deterministic results"


# ===========================================================================
# R (bonus). Parameter validation — out-of-bounds max_hops
# ===========================================================================
@pytest.mark.asyncio
async def test_r_invalid_params(clean_env):
    """Out-of-bounds traversal parameters return an informative error, not an exception."""
    _, _, traversal = clean_env

    # max_hops = 0 is invalid
    r1 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        max_hops=0, max_nodes=100, max_edges=100,
    )
    assert r1.error is not None
    assert "max_hops" in r1.error

    # max_hops exceeding ceiling is invalid
    r2 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        max_hops=MAX_HOPS_CEILING + 1, max_nodes=100, max_edges=100,
    )
    assert r2.error is not None
    assert "max_hops" in r2.error

    # max_nodes = 0 is invalid
    r3 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        max_hops=1, max_nodes=0, max_edges=100,
    )
    assert r3.error is not None
    assert "max_nodes" in r3.error

