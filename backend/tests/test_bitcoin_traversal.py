"""CHAKRA Step 2B: Bitcoin UTXO-Aware Deterministic Money-Flow Traversal Test Suite.

Dedicated tests covering requirements A through P:
A. Single Bitcoin spend: Address A -> SPENT_INPUT -> Tx1 -> CREATED_OUTPUT -> Address B
B. Multi-output transaction: A spends Tx1; Tx1 creates outputs B and C
C. Multi-input transaction: A and X both spend inputs into Tx1; ensure traversal does NOT fabricate A->X or X->A
D. Multi-input/multi-output: verify no artificial 1-to-1 pairing
E. Multi-hop Bitcoin: A -> Tx1 -> B -> Tx2 -> C
F. Cycle protection: graph with cycle returns simple paths without infinite recursion
G. Deterministic ordering across repeated runs
H. max_hops enforcement
I. max_nodes enforcement
J. max_edges enforcement
K. Exact large satoshi amount preservation
L. Addressless Bitcoin records (coinbase, OP_RETURN, unparsed)
M. Nonexistent Bitcoin start address
N. Zero outgoing/spend transitions (unspent outputs)
O. Concurrent traversal determinism
P. Read-only graph verification
14. Live Neo4j Community integration verification
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
import asyncpg
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.schemas.chain import Chain, Network
from app.schemas.transaction import (
    Transaction,
    Transfer,
    TransactionProvenance,
    BitcoinTransactionDetail,
    BitcoinVin,
    BitcoinVout,
    AssetType,
    TransactionStatus,
    TransactionType,
)
from app.db.repository import TransactionRepository
from app.graph.projector import GraphProjector
from app.graph.traversal import (
    MoneyFlowTraversal,
    TraversalResult,
    SpentInputEdge,
    TransactionNode,
    CreatedOutputEdge,
    UtxoSpendStep,
)

# ---------------------------------------------------------------------------
# Connection configuration
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
# Fixtures
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
async def btc_env(pg_pool, neo4j_driver):
    """Wipe PostgreSQL transactions and Neo4j graph; ensure constraints."""
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()
    repo = TransactionRepository(pg_pool)
    traversal = MoneyFlowTraversal(neo4j_driver)
    return repo, projector, traversal


# ---------------------------------------------------------------------------
# Helper to build canonical Bitcoin transactions
# ---------------------------------------------------------------------------


def _make_btc_tx(
    txid: str,
    vins: list[dict],
    vouts: list[dict],
    fee_sat: int = 1000,
    network: Network = Network.BTC_MAINNET,
) -> tuple[Transaction, BitcoinTransactionDetail]:
    total_in = sum(v.get("value_sat", 0) for v in vins)
    total_out = sum(v.get("value_sat", 0) for v in vouts)

    transfers = []
    for vin in vins:
        transfers.append(
            Transfer(
                from_address=vin.get("address"),
                to_address=None,
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=vin.get("value_sat", 0),
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=network,
            )
        )
    for vout in vouts:
        transfers.append(
            Transfer(
                from_address=None,
                to_address=vout.get("address"),
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=vout.get("value_sat", 0),
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=network,
            )
        )

    tx = Transaction(
        transaction_id=txid,
        chain=Chain.BITCOIN,
        network=network,
        native_value=total_out,
        native_value_unit="sat",
        fee=fee_sat,
        fee_asset="BTC",
        status=TransactionStatus.SUCCESS,
        transfers=transfers,
        provenance=TransactionProvenance(
            provider="mempool.space",
            chain=Chain.BITCOIN,
            network=network,
            original_id=txid,
        ),
    )

    detail_vins = [
        BitcoinVin(
            txid=v.get("txid", "prev_tx"),
            vout=v.get("vout", 0),
            address=v.get("address"),
            value_sat=v.get("value_sat", 0),
            coinbase=v.get("coinbase", False),
        )
        for v in vins
    ]

    detail_vouts = [
        BitcoinVout(
            n=v.get("n", i),
            address=v.get("address"),
            value_sat=v.get("value_sat", 0),
            script_type=v.get("script_type", "v0_p2wpkh"),
        )
        for i, v in enumerate(vouts)
    ]

    detail = BitcoinTransactionDetail(
        txid=txid,
        inputs=detail_vins,
        outputs=detail_vouts,
        total_input_sat=total_in,
        total_output_sat=total_out,
        fee_sat=fee_sat,
    )

    return tx, detail


# ===========================================================================
# A. Single Bitcoin Spend
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_a_single_spend(btc_env):
    """Address A -> SPENT_INPUT -> Tx1 -> CREATED_OUTPUT -> Address B."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_sender_a"
    addr_b = "bc1q_receiver_b"

    tx, detail = _make_btc_tx(
        "btc_tx_a1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_a", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 49000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 1
    path = result.paths[0]
    assert path.hops == 1
    assert len(path.nodes) == 2
    assert path.nodes[0].normalized_address == addr_a
    assert path.nodes[1].normalized_address == addr_b
    assert len(path.edges) == 0  # CRITICAL: No synthetic TRANSFERRED edges!
    assert len(path.utxo_steps) == 1

    step = path.utxo_steps[0]
    assert step.from_address.normalized_address == addr_a
    assert step.to_address.normalized_address == addr_b
    assert step.transaction.transaction_id == "btc_tx_a1"
    assert step.spent_input.value_sat == 50000
    assert step.spent_input.value_sat_str == "50000"
    assert step.created_output.value_sat == 49000
    assert step.created_output.value_sat_str == "49000"


# ===========================================================================
# B. Multi-Output Transaction
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_b_multi_output(btc_env):
    """A spends Tx1; Tx1 creates outputs B and C. Both deterministic branches represented."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_multi_a"
    addr_b = "bc1q_out_b"
    addr_c = "bc1q_out_c"

    tx, detail = _make_btc_tx(
        "btc_tx_b1",
        vins=[{"address": addr_a, "value_sat": 100000, "txid": "prev_b", "vout": 0}],
        vouts=[
            {"address": addr_b, "value_sat": 60000, "n": 0},
            {"address": addr_c, "value_sat": 38000, "n": 1},
        ],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 2
    destinations = {p.nodes[-1].normalized_address for p in result.paths}
    assert destinations == {addr_b, addr_c}
    for path in result.paths:
        assert path.hops == 1
        assert len(path.edges) == 0
        assert len(path.utxo_steps) == 1
        assert path.utxo_steps[0].transaction.transaction_id == "btc_tx_b1"


# ===========================================================================
# C. Multi-Input Transaction — No Fabricated Sender Relationship
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_c_multi_input_no_fabricated_sender(btc_env):
    """A and X both spend inputs into Tx1. Ensure traversal does NOT fabricate A->X or X->A."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_sender_a_c"
    addr_x = "bc1q_sender_x_c"
    addr_b = "bc1q_receiver_b_c"

    tx, detail = _make_btc_tx(
        "btc_tx_c1",
        vins=[
            {"address": addr_a, "value_sat": 50000, "txid": "prev_ca", "vout": 0},
            {"address": addr_x, "value_sat": 25000, "txid": "prev_cx", "vout": 1},
        ],
        vouts=[{"address": addr_b, "value_sat": 73000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    # Traverse starting from A
    result_a = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result_a.error is None
    assert result_a.total_paths == 1
    path_a = result_a.paths[0]
    addrs_in_path_a = [n.normalized_address for n in path_a.nodes]
    assert addr_a in addrs_in_path_a
    assert addr_b in addrs_in_path_a
    # CRITICAL: addr_x MUST NOT be in the path!
    assert addr_x not in addrs_in_path_a
    assert path_a.utxo_steps[0].from_address.normalized_address == addr_a

    # Traverse starting from X
    result_x = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_x,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )
    assert result_x.error is None
    assert result_x.total_paths == 1
    path_x = result_x.paths[0]
    addrs_in_path_x = [n.normalized_address for n in path_x.nodes]
    assert addr_x in addrs_in_path_x
    assert addr_b in addrs_in_path_x
    # CRITICAL: addr_a MUST NOT be in X's path!
    assert addr_a not in addrs_in_path_x


# ===========================================================================
# D. Multi-Input / Multi-Output — No Artificial 1-to-1 Pairing
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_d_multi_input_multi_output(btc_env):
    """Verify no artificial 1-to-1 pairing of inputs and outputs."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_in_a_d"
    addr_x = "bc1q_in_x_d"
    addr_b = "bc1q_out_b_d"
    addr_c = "bc1q_out_c_d"

    tx, detail = _make_btc_tx(
        "btc_tx_d1",
        vins=[
            {"address": addr_a, "value_sat": 50000, "txid": "prev_da", "vout": 0},
            {"address": addr_x, "value_sat": 30000, "txid": "prev_dx", "vout": 1},
        ],
        vouts=[
            {"address": addr_b, "value_sat": 45000, "n": 0},
            {"address": addr_c, "value_sat": 32000, "n": 1},
        ],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 2
    # In both paths, the spent input is A's input (50,000 sat), and outputs are B and C
    step_b = next(p.utxo_steps[0] for p in result.paths if p.nodes[-1].normalized_address == addr_b)
    step_c = next(p.utxo_steps[0] for p in result.paths if p.nodes[-1].normalized_address == addr_c)

    assert step_b.spent_input.value_sat == 50000
    assert step_b.created_output.value_sat == 45000
    assert step_c.spent_input.value_sat == 50000
    assert step_c.created_output.value_sat == 32000
    # No assertion that 50,000 sat went solely to B or C


# ===========================================================================
# E. Multi-Hop Bitcoin Traversal
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_e_multi_hop(btc_env):
    """A -> Tx1 -> B -> Tx2 -> C."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_hop_a"
    addr_b = "bc1q_hop_b"
    addr_c = "bc1q_hop_c"

    tx1, detail1 = _make_btc_tx(
        "btc_tx_e1",
        vins=[{"address": addr_a, "value_sat": 100000, "txid": "prev_ea", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 98000, "n": 0}],
    )
    tx2, detail2 = _make_btc_tx(
        "btc_tx_e2",
        vins=[{"address": addr_b, "value_sat": 98000, "txid": "btc_tx_e1", "vout": 0}],
        vouts=[{"address": addr_c, "value_sat": 95000, "n": 0}],
    )
    await repo.save_transaction(tx1, detail1)
    await repo.save_transaction(tx2, detail2)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=2,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 2
    paths_by_hop = {p.hops: p for p in result.paths}
    assert 1 in paths_by_hop
    assert 2 in paths_by_hop

    path_2hop = paths_by_hop[2]
    assert [n.normalized_address for n in path_2hop.nodes] == [addr_a, addr_b, addr_c]
    assert len(path_2hop.utxo_steps) == 2
    assert path_2hop.utxo_steps[0].transaction.transaction_id == "btc_tx_e1"
    assert path_2hop.utxo_steps[1].transaction.transaction_id == "btc_tx_e2"


# ===========================================================================
# F. Cycle Protection
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_f_cycle_protection(btc_env):
    """Cycle in graph: A -> Tx1 -> B -> Tx2 -> A. Traversal must terminate and return simple paths."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_cycle_a"
    addr_b = "bc1q_cycle_b"

    tx1, detail1 = _make_btc_tx(
        "btc_tx_f1",
        vins=[{"address": addr_a, "value_sat": 100000, "txid": "prev_fa", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 98000, "n": 0}],
    )
    tx2, detail2 = _make_btc_tx(
        "btc_tx_f2",
        vins=[{"address": addr_b, "value_sat": 98000, "txid": "btc_tx_f1", "vout": 0}],
        vouts=[{"address": addr_a, "value_sat": 95000, "n": 0}],
    )
    await repo.save_transaction(tx1, detail1)
    await repo.save_transaction(tx2, detail2)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    # Must terminate without error
    for path in result.paths:
        addr_cids = [n.composite_id for n in path.nodes]
        assert len(addr_cids) == len(set(addr_cids)), f"Cycle found in path: {addr_cids}"


# ===========================================================================
# G. Deterministic Ordering
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_g_deterministic_ordering(btc_env):
    """Repeated traversals produce identical path ordering and input/output IDs."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_det_a"
    addr_b = "bc1q_det_b"
    addr_c = "bc1q_det_c"

    tx, detail = _make_btc_tx(
        "btc_tx_g1",
        vins=[{"address": addr_a, "value_sat": 100000, "txid": "prev_ga", "vout": 0}],
        vouts=[
            {"address": addr_b, "value_sat": 40000, "n": 0},
            {"address": addr_c, "value_sat": 58000, "n": 1},
        ],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    r1 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=2, max_nodes=100, max_edges=100,
    )
    r2 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=2, max_nodes=100, max_edges=100,
    )

    assert r1.error is None
    assert r2.error is None
    seq1 = [s.created_output.output_id for p in r1.paths for s in p.utxo_steps]
    seq2 = [s.created_output.output_id for p in r2.paths for s in p.utxo_steps]
    assert seq1 == seq2


# ===========================================================================
# H. max_hops Enforcement
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_h_max_hops_enforcement(btc_env):
    """With max_hops=1, a 2-hop spend chain is truncated to 1-hop only."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_h_a"
    addr_b = "bc1q_h_b"
    addr_c = "bc1q_h_c"

    tx1, detail1 = _make_btc_tx(
        "btc_tx_h1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_ha", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 48000, "n": 0}],
    )
    tx2, detail2 = _make_btc_tx(
        "btc_tx_h2",
        vins=[{"address": addr_b, "value_sat": 48000, "txid": "btc_tx_h1", "vout": 0}],
        vouts=[{"address": addr_c, "value_sat": 46000, "n": 0}],
    )
    await repo.save_transaction(tx1, detail1)
    await repo.save_transaction(tx2, detail2)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert all(p.hops <= 1 for p in result.paths)
    assert not any(p.nodes[-1].normalized_address == addr_c for p in result.paths)


# ===========================================================================
# I. max_nodes Enforcement
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_i_max_nodes_enforcement(btc_env):
    """When max_nodes=2, traversal truncates immediately after reaching the limit."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_i_a"
    addr_b = "bc1q_i_b"

    tx, detail = _make_btc_tx(
        "btc_tx_i1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_ia", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 48000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    # Start node A (1) + Tx node (2) -> destination B would be 3 > max_nodes
    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=2,
        max_edges=100,
    )

    assert result.error is None
    assert result.truncated is True
    assert result.total_nodes_visited <= 2


# ===========================================================================
# J. max_edges Enforcement
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_j_max_edges_enforcement(btc_env):
    """When max_edges=1, traversal halts after 1 relationship visit."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_j_a"
    addr_b = "bc1q_j_b"

    tx, detail = _make_btc_tx(
        "btc_tx_j1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_ja", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 48000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=5,
        max_nodes=100,
        max_edges=1,
    )

    assert result.error is None
    assert result.truncated is True
    assert result.total_edges_visited <= 1


# ===========================================================================
# K. Exact Large Satoshi Amount Preservation
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_k_exact_large_satoshi(btc_env):
    """Preserve exact satoshi integer and string representations without float loss."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_large_a"
    addr_b = "bc1q_large_b"
    large_sat = 2_100_000_000_000_000  # 21 million BTC in satoshis

    tx, detail = _make_btc_tx(
        "btc_tx_k1",
        vins=[{"address": addr_a, "value_sat": large_sat, "txid": "prev_ka", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": large_sat - 1000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert len(result.paths) == 1
    step = result.paths[0].utxo_steps[0]
    assert step.spent_input.value_sat == large_sat
    assert step.spent_input.value_sat_str == str(large_sat)
    assert step.created_output.value_sat == large_sat - 1000
    assert step.created_output.value_sat_str == str(large_sat - 1000)


# ===========================================================================
# L. Addressless Bitcoin Records (coinbase and OP_RETURN)
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_l_addressless_records(btc_env):
    """Ensure coinbase inputs and OP_RETURN outputs do not crash or fabricate ownership."""
    repo, projector, traversal = btc_env
    addr_miner = "bc1q_miner_reward"

    # Coinbase transaction with OP_RETURN output
    tx, detail = _make_btc_tx(
        "btc_tx_coinbase",
        vins=[{"address": None, "value_sat": 312500000, "coinbase": True}],
        vouts=[
            {"address": addr_miner, "value_sat": 312500000, "n": 0},
            {"address": None, "value_sat": 0, "n": 1, "script_type": "op_return"},
        ],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    # Traverse starting from coinbase
    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address="(coinbase)",
        max_hops=1,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 2
    dests = {p.nodes[-1].normalized_address for p in result.paths}
    assert addr_miner in dests
    assert any("op_return" in d for d in dests)


# ===========================================================================
# M. Nonexistent Bitcoin Start Address
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_m_nonexistent_start_address(btc_env):
    """Nonexistent Bitcoin start address returns an informative error result."""
    _, _, traversal = btc_env

    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address="bc1q_totally_nonexistent_address_12345",
        max_hops=2,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is not None
    assert "Start address not found" in result.error
    assert result.total_paths == 0


# ===========================================================================
# N. Zero Outgoing / Spend Transitions
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_n_zero_outgoing(btc_env):
    """Address that only received an output and never spent it produces 0 paths."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_sender_n"
    addr_b = "bc1q_hodler_b"

    tx, detail = _make_btc_tx(
        "btc_tx_n1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_na", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 48000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    # Traverse starting from hodler B (no outgoing SPENT_INPUT)
    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_b,
        max_hops=5,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 0
    assert result.truncated is False


# ===========================================================================
# O. Concurrent Traversal Determinism
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_o_concurrent_traversal(btc_env):
    """Multiple concurrent Bitcoin traversals yield identical deterministic results."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_conc_a"
    addr_b = "bc1q_conc_b"
    addr_c = "bc1q_conc_c"

    tx1, detail1 = _make_btc_tx(
        "btc_tx_o1",
        vins=[{"address": addr_a, "value_sat": 100000, "txid": "prev_oa", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 98000, "n": 0}],
    )
    tx2, detail2 = _make_btc_tx(
        "btc_tx_o2",
        vins=[{"address": addr_b, "value_sat": 98000, "txid": "btc_tx_o1", "vout": 0}],
        vouts=[{"address": addr_c, "value_sat": 95000, "n": 0}],
    )
    await repo.save_transaction(tx1, detail1)
    await repo.save_transaction(tx2, detail2)
    await projector.project_all()

    async def _run():
        return await traversal.traverse(
            chain=Chain.BITCOIN.value,
            network=Network.BTC_MAINNET.value,
            address=addr_a,
            max_hops=2,
            max_nodes=100,
            max_edges=100,
        )

    results = await asyncio.gather(*[_run() for _ in range(5)])

    ref = results[0]
    assert ref.error is None
    for r in results[1:]:
        assert r.error is None
        assert r.total_paths == ref.total_paths
        ref_steps = [s.created_output.output_id for p in ref.paths for s in p.utxo_steps]
        r_steps   = [s.created_output.output_id for p in r.paths for s in p.utxo_steps]
        assert ref_steps == r_steps


# ===========================================================================
# P. Read-Only Graph Verification
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_p_read_only(btc_env, neo4j_driver):
    """Bitcoin traversal causes zero mutations on the Neo4j graph."""
    repo, projector, traversal = btc_env
    addr_a = "bc1q_ro_a"
    addr_b = "bc1q_ro_b"

    tx, detail = _make_btc_tx(
        "btc_tx_p1",
        vins=[{"address": addr_a, "value_sat": 50000, "txid": "prev_pa", "vout": 0}],
        vouts=[{"address": addr_b, "value_sat": 48000, "n": 0}],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        r = await session.run("MATCH (n) RETURN count(n) AS cnt")
        before_nodes = (await r.single())["cnt"]
        r = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        before_rels = (await r.single())["cnt"]

    await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_a,
        max_hops=2,
        max_nodes=100,
        max_edges=100,
    )

    async with neo4j_driver.session() as session:
        r = await session.run("MATCH (n) RETURN count(n) AS cnt")
        after_nodes = (await r.single())["cnt"]
        r = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        after_rels = (await r.single())["cnt"]

    assert before_nodes == after_nodes
    assert before_rels == after_rels


# ===========================================================================
# 14. Live Neo4j Community Integration Test
# ===========================================================================
@pytest.mark.asyncio
async def test_btc_live_neo4j_integration(btc_env):
    """Live Neo4j Community integration verifying actual Step 2A UTXO graph traversal."""
    repo, projector, traversal = btc_env
    addr_s = "bc1q_live_start"
    addr_m = "bc1q_live_mid"
    addr_e = "bc1q_live_end"

    tx1, detail1 = _make_btc_tx(
        "btc_live_tx1",
        vins=[{"address": addr_s, "value_sat": 200000, "txid": "prev_live", "vout": 0}],
        vouts=[{"address": addr_m, "value_sat": 195000, "n": 0}],
    )
    tx2, detail2 = _make_btc_tx(
        "btc_live_tx2",
        vins=[{"address": addr_m, "value_sat": 195000, "txid": "btc_live_tx1", "vout": 0}],
        vouts=[{"address": addr_e, "value_sat": 190000, "n": 0}],
    )
    await repo.save_transaction(tx1, detail1)
    await repo.save_transaction(tx2, detail2)

    # Actual Step 2A graph projection into live Neo4j Community
    summary = await projector.project_all()
    assert summary.total_transactions == 2
    assert summary.total_bitcoin_inputs == 2
    assert summary.total_bitcoin_outputs == 2
    assert summary.total_transfers == 0

    # Live traversal query against actual projected graph
    result = await traversal.traverse(
        chain=Chain.BITCOIN.value,
        network=Network.BTC_MAINNET.value,
        address=addr_s,
        max_hops=2,
        max_nodes=100,
        max_edges=100,
    )

    assert result.error is None
    assert result.total_paths == 2
    path_2 = next(p for p in result.paths if p.hops == 2)
    assert [n.normalized_address for n in path_2.nodes] == [addr_s, addr_m, addr_e]
    assert len(path_2.utxo_steps) == 2
    assert path_2.utxo_steps[0].spent_input.value_sat == 200000
    assert path_2.utxo_steps[1].created_output.value_sat == 190000
