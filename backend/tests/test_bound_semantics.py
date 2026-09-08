"""CHAKRA Step 2B: Targeted Bound Semantics Verification Tests.

Verifies:
1. Account-model branching (A -> B, A -> C) across max_nodes=1, 2, 3
2. Bitcoin branching (A -> Tx1 -> B, A -> Tx1 -> C) across max_nodes=1, 2, 3, 4
3. Diamond non-cycle behavior: A -> B -> D and A -> C -> D
4. Bitcoin max_edges accounting: SPENT_INPUT=1, CREATED_OUTPUT=1
"""
from __future__ import annotations

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
from app.graph.traversal import MoneyFlowTraversal

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
async def env(pg_pool, neo4j_driver):
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()
    repo = TransactionRepository(pg_pool)
    traversal = MoneyFlowTraversal(neo4j_driver)
    return repo, projector, traversal


def _evm_tx(tx_id: str, from_addr: str, to_addr: str) -> Transaction:
    return Transaction(
        transaction_id=tx_id,
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        chain_id=1,
        native_value=1000,
        native_value_unit="wei",
        status=TransactionStatus.SUCCESS,
        transfers=[
            Transfer(
                from_address=from_addr,
                to_address=to_addr,
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=1000,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id=tx_id,
        ),
    )


def _btc_tx(txid: str, vins: list[dict], vouts: list[dict]) -> tuple[Transaction, BitcoinTransactionDetail]:
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
                network=Network.BTC_MAINNET,
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
                network=Network.BTC_MAINNET,
            )
        )


    tx = Transaction(
        transaction_id=txid,
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=sum(v.get("value_sat", 0) for v in vouts),
        native_value_unit="sat",
        fee=1000,
        fee_asset="BTC",
        status=TransactionStatus.SUCCESS,
        transfers=transfers,
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
            original_id=txid,
        ),
    )

    detail = BitcoinTransactionDetail(
        txid=txid,
        inputs=[
            BitcoinVin(txid="prev", vout=0, address=v.get("address"), value_sat=v.get("value_sat", 0))
            for v in vins
        ],
        outputs=[
            BitcoinVout(n=v.get("n", i), address=v.get("address"), value_sat=v.get("value_sat", 0))
            for i, v in enumerate(vouts)
        ],
        total_input_sat=sum(v.get("value_sat", 0) for v in vins),
        total_output_sat=sum(v.get("value_sat", 0) for v in vouts),
        fee_sat=1000,
    )
    return tx, detail


# ===========================================================================
# 1. Account Branching Verification: A -> B, A -> C
# ===========================================================================
@pytest.mark.asyncio
async def test_account_branching_max_nodes_semantics(env):
    """Verify max_nodes=1, 2, 3 exact behavior for A -> B, A -> C."""
    repo, projector, traversal = env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"

    # Two transactions giving A -> B (tx 1) and A -> C (tx 2)
    await repo.save_transaction(_evm_tx("0x01_tx_b", addr_a, addr_b))
    await repo.save_transaction(_evm_tx("0x02_tx_c", addr_a, addr_c))
    await projector.project_all()

    # max_nodes=1: Start node A is 1. Evaluating B halts before admitting B.
    r1 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=1, max_edges=100,
    )
    assert r1.total_paths == 0
    assert r1.truncated is True
    assert r1.total_nodes_visited == 1
    assert r1.total_edges_visited == 1

    # max_nodes=2: A is admitted (1), B is admitted (2). Path A->B recorded.
    # Evaluating C halts before admitting C.
    r2 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=2, max_edges=100,
    )
    assert r2.total_paths == 1
    assert [n.normalized_address for n in r2.paths[0].nodes] == [addr_a, addr_b]
    assert r2.truncated is True
    assert r2.total_nodes_visited == 2
    assert r2.total_edges_visited == 2

    # max_nodes=3: A (1), B (2), C (3) all admitted. Both paths recorded.
    r3 = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=3, max_edges=100,
    )
    assert r3.total_paths == 2
    paths_addrs = [[n.normalized_address for n in p.nodes] for p in r3.paths]
    assert [addr_a, addr_b] in paths_addrs
    assert [addr_a, addr_c] in paths_addrs
    assert r3.truncated is False
    assert r3.total_nodes_visited == 3
    assert r3.total_edges_visited == 2


# ===========================================================================
# 2. Bitcoin Branching Verification: A -> Tx1 -> B, A -> Tx1 -> C
# ===========================================================================
@pytest.mark.asyncio
async def test_bitcoin_branching_max_nodes_semantics(env):
    """Verify max_nodes=1, 2, 3, 4 exact behavior for A -> Tx1 -> B and A -> Tx1 -> C."""
    repo, projector, traversal = env
    addr_a = "bc1q_a_branch"
    addr_b = "bc1q_b_branch"
    addr_c = "bc1q_c_branch"

    tx, detail = _btc_tx(
        "btc_tx_branch",
        vins=[{"address": addr_a, "value_sat": 50000}],
        vouts=[
            {"address": addr_b, "value_sat": 25000, "n": 0},
            {"address": addr_c, "value_sat": 24000, "n": 1},
        ],
    )
    await repo.save_transaction(tx, detail)
    await projector.project_all()

    # max_nodes=1: A is 1. Trying to admit Tx1 hits ceiling.
    r1 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=1, max_edges=100,
    )
    assert r1.total_paths == 0
    assert r1.truncated is True
    assert r1.total_nodes_visited == 1
    assert r1.total_edges_visited == 1  # SPENT_INPUT evaluated

    # max_nodes=2: A is 1, Tx1 is 2. Trying to admit B hits ceiling.
    r2 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=2, max_edges=100,
    )
    assert r2.total_paths == 0
    assert r2.truncated is True
    assert r2.total_nodes_visited == 2  # A + Tx1
    assert r2.total_edges_visited == 2  # SPENT_INPUT + CREATED_OUTPUT(B)

    # max_nodes=3: A (1), Tx1 (2), B (3) admitted. Path A->Tx1->B recorded.
    # Trying to admit C hits ceiling.
    r3 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=3, max_edges=100,
    )
    assert r3.total_paths == 1
    assert [n.normalized_address for n in r3.paths[0].nodes] == [addr_a, addr_b]
    assert r3.truncated is True
    assert r3.total_nodes_visited == 3  # A + Tx1 + B
    assert r3.total_edges_visited == 3  # SPENT_INPUT + CREATED_OUTPUT(B) + CREATED_OUTPUT(C)

    # max_nodes=4: A (1), Tx1 (2), B (3), C (4) all admitted. Both branches recorded.
    r4 = await traversal.traverse(
        chain=Chain.BITCOIN.value, network=Network.BTC_MAINNET.value,
        address=addr_a, max_hops=1, max_nodes=4, max_edges=100,
    )
    assert r4.total_paths == 2
    paths_addrs = [[n.normalized_address for n in p.nodes] for p in r4.paths]
    assert [addr_a, addr_b] in paths_addrs
    assert [addr_a, addr_c] in paths_addrs
    assert r4.truncated is False
    assert r4.total_nodes_visited == 4  # A + Tx1 + B + C
    assert r4.total_edges_visited == 3  # 1 SPENT_INPUT + 2 CREATED_OUTPUT


# ===========================================================================
# 3. Diamond Graph Non-Cycle Verification: A -> B -> D and A -> C -> D
# ===========================================================================
@pytest.mark.asyncio
async def test_diamond_graph_cycle_independence(env):
    """Verify a node visited in branch 1 is NOT treated as a cycle in branch 2."""
    repo, projector, traversal = env
    addr_a = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    addr_b = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    addr_c = "0xcccccccccccccccccccccccccccccccccccccccc"
    addr_d = "0xdddddddddddddddddddddddddddddddddddddddd"

    # Branch 1: A -> B -> D
    await repo.save_transaction(_evm_tx("0x01_a_b", addr_a, addr_b))
    await repo.save_transaction(_evm_tx("0x02_b_d", addr_b, addr_d))
    # Branch 2: A -> C -> D
    await repo.save_transaction(_evm_tx("0x03_a_c", addr_a, addr_c))
    await repo.save_transaction(_evm_tx("0x04_c_d", addr_c, addr_d))
    await projector.project_all()

    result = await traversal.traverse(
        chain=Chain.EVM.value, network=Network.ETH_MAINNET.value,
        address=addr_a, max_hops=2, max_nodes=100, max_edges=100,
    )

    assert result.error is None
    paths_addrs = [tuple(n.normalized_address for n in p.nodes) for p in result.paths]

    # Both 2-hop paths reaching D must be found
    assert (addr_a, addr_b, addr_d) in paths_addrs
    assert (addr_a, addr_c, addr_d) in paths_addrs
