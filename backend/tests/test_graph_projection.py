"""CHAKRA Step 2A: Neo4j Graph Projection Test Suite.

Comprehensive tests covering:
A. EVM Native Transfer
B. ERC-20 Transfer
C. TRON Native Transfer
D. TRC-20 Transfer
E. Solana Native Transfer
F. SPL Transfer
G. Bitcoin Multi-Input/Multi-Output Transaction
H. Cross-Chain Address Collision
I. Repeated Projection / Idempotency
J. Exact Large Integer Preservation
K. Token Contract/Mint Distinction
L. Transaction Identity
M. Batch Processing
N. Projection Failure Handling
O. Multiple Transfers in One Transaction
P. Repeated Transfers Between Same Addresses
Q. No Inferred Relationships
18. Live Neo4j Community Integration
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
from app.graph.models import (
    normalize_address,
    make_address_composite_id,
    make_transaction_composite_id,
    make_transfer_relationship_id,
    make_bitcoin_input_id,
    make_bitcoin_output_id,
    to_neo4j_numeric,
    BatchProjectionError,
)
from app.graph.projector import GraphProjector

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
async def setup_databases(pg_pool, neo4j_driver):
    """Clean slate before each test and ensure constraints are applied."""
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")

    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()

    repo = TransactionRepository(pg_pool)
    return repo, projector


# ===========================================================================
# A. EVM Native Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_a_evm_native_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="0xevm_native_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        chain_id=1,
        block_number=123456,
        block_hash="0xblockhash1",
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        from_address="0x1111111111111111111111111111111111111111",
        to_address="0x2222222222222222222222222222222222222222",
        native_value=1500000000000000000,  # 1.5 ETH in wei
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21000000000000,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address="0x1111111111111111111111111111111111111111",
                to_address="0x2222222222222222222222222222222222222222",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=1500000000000000000,
                amount_unit="wei",
                decimals=18,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="etherscan",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0xevm_native_1",
        ),
    )
    await repo.save_transaction(tx)

    summary = await projector.project_all()
    assert summary.total_transactions == 1
    assert summary.total_transfers == 1

    async with neo4j_driver.session() as session:
        # Verify Address identity
        from_comp = make_address_composite_id(
            "evm", Network.ETH_MAINNET.value, "0x1111111111111111111111111111111111111111"
        )
        to_comp = make_address_composite_id(
            "evm", Network.ETH_MAINNET.value, "0x2222222222222222222222222222222222222222"
        )

        res_from = await session.run(
            "MATCH (a:Address {composite_id: $cid}) RETURN a", cid=from_comp
        )
        rec_from = await res_from.single()
        assert rec_from is not None
        assert rec_from["a"]["chain"] == "evm"
        assert rec_from["a"]["network"] == Network.ETH_MAINNET.value
        assert rec_from["a"]["normalized_address"] == "0x1111111111111111111111111111111111111111"

        # Verify Transaction identity
        tx_comp = make_transaction_composite_id("evm", Network.ETH_MAINNET.value, "0xevm_native_1")
        res_tx = await session.run(
            "MATCH (t:Transaction {composite_id: $cid}) RETURN t", cid=tx_comp
        )
        rec_tx = await res_tx.single()
        assert rec_tx is not None
        assert rec_tx["t"]["transaction_id"] == "0xevm_native_1"
        assert rec_tx["t"]["chain"] == "evm"
        assert rec_tx["t"]["network"] == Network.ETH_MAINNET.value
        assert rec_tx["t"]["native_value"] == 1500000000000000000

        # Verify Transfer relationship & exact amount & correlation
        res_rel = await session.run(
            """
            MATCH (f:Address)-[r:TRANSFERRED]->(t:Address)
            RETURN r, f.composite_id AS f_id, t.composite_id AS t_id
            """
        )
        rec_rel = await res_rel.single()
        assert rec_rel is not None
        r = rec_rel["r"]
        assert rec_rel["f_id"] == from_comp
        assert rec_rel["t_id"] == to_comp
        assert r["amount"] == 1500000000000000000
        assert r["amount_str"] == "1500000000000000000"
        assert r["asset_symbol"] == "ETH"
        assert r["asset_type"] == "native"
        assert r["transaction_id"] == "0xevm_native_1"


# ===========================================================================
# B. ERC-20 Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_b_erc20_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    contract_addr = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    tx = Transaction(
        transaction_id="0xerc20_tx_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        transfers=[
            Transfer(
                from_address="0xSenderERC20",
                to_address="0xReceiverERC20",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDC",
                asset_contract=contract_addr,
                amount=50000000,  # 50 USDC (6 decimals)
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0xerc20_tx_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH ()-[r:TRANSFERRED {transaction_id: '0xerc20_tx_1'}]->() RETURN r"
        )
        rec = await res.single()
        assert rec is not None
        r = rec["r"]
        assert r["asset_type"] == "token"
        assert r["asset_symbol"] == "USDC"
        assert r["asset_contract"] == contract_addr
        assert r["amount"] == 50000000
        assert r["decimals"] == 6


# ===========================================================================
# C. TRON Native Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_c_tron_native_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="tron_tx_native_1",
        chain=Chain.TRON,
        network=Network.TRON_MAINNET,
        native_value=2500000,  # 2.5 TRX in sun
        native_value_unit="sun",
        transfers=[
            Transfer(
                from_address="TAddressFrom1111111111111111111",
                to_address="TAddressTo222222222222222222222",
                asset_type=AssetType.NATIVE,
                asset_symbol="TRX",
                amount=2500000,
                amount_unit="sun",
                decimals=6,
                chain=Chain.TRON,
                network=Network.TRON_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="trongrid", chain=Chain.TRON, network=Network.TRON_MAINNET, original_id="tron_tx_native_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH (f:Address)-[r:TRANSFERRED {chain: 'tron'}]->(t:Address) RETURN f, r, t"
        )
        rec = await res.single()
        assert rec is not None
        assert rec["f"]["chain"] == "tron"
        assert rec["f"]["network"] == "tron-mainnet"
        assert rec["f"]["normalized_address"] == "TAddressFrom1111111111111111111"
        assert rec["r"]["amount"] == 2500000
        assert rec["r"]["asset_symbol"] == "TRX"


# ===========================================================================
# D. TRC-20 Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_d_trc20_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    trc20_contract = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
    tx = Transaction(
        transaction_id="tron_trc20_tx_1",
        chain=Chain.TRON,
        network=Network.TRON_MAINNET,
        native_value=0,
        native_value_unit="sun",
        transfers=[
            Transfer(
                from_address="TSourceAddr1111111111111111111",
                to_address="TDestAddr222222222222222222222",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract=trc20_contract,
                amount=100000000,  # 100 USDT
                amount_unit="sun",
                decimals=6,
                chain=Chain.TRON,
                network=Network.TRON_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="trongrid", chain=Chain.TRON, network=Network.TRON_MAINNET, original_id="tron_trc20_tx_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH ()-[r:TRANSFERRED {asset_contract: $contract}]->() RETURN r",
            contract=trc20_contract,
        )
        rec = await res.single()
        assert rec is not None
        assert rec["r"]["asset_symbol"] == "USDT"
        assert rec["r"]["amount"] == 100000000
        assert rec["r"]["chain"] == "tron"


# ===========================================================================
# E. Solana Native Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_e_solana_native_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="sol_native_sig_1",
        chain=Chain.SOLANA,
        network=Network.SOL_MAINNET,
        native_value=5000000000,  # 5 SOL in lamports
        native_value_unit="lamports",
        transfers=[
            Transfer(
                from_address="SolSenderPubkey111111111111111111111111111",
                to_address="SolReceiverPubkey222222222222222222222222222",
                asset_type=AssetType.NATIVE,
                asset_symbol="SOL",
                amount=5000000000,
                amount_unit="lamports",
                decimals=9,
                chain=Chain.SOLANA,
                network=Network.SOL_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="helius", chain=Chain.SOLANA, network=Network.SOL_MAINNET, original_id="sol_native_sig_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH (f:Address)-[r:TRANSFERRED {chain: 'solana'}]->(t:Address) RETURN f, r, t"
        )
        rec = await res.single()
        assert rec is not None
        assert rec["f"]["chain"] == "solana"
        assert rec["r"]["amount"] == 5000000000
        assert rec["r"]["amount_unit"] == "lamports"


# ===========================================================================
# F. SPL Transfer
# ===========================================================================
@pytest.mark.asyncio
async def test_f_spl_transfer(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    tx = Transaction(
        transaction_id="sol_spl_sig_1",
        chain=Chain.SOLANA,
        network=Network.SOL_MAINNET,
        native_value=0,
        native_value_unit="lamports",
        transfers=[
            Transfer(
                from_address="SPLSenderPubkey111111111111111111111111111",
                to_address="SPLReceiverPubkey222222222222222222222222222",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDC",
                asset_contract=usdc_mint,
                amount=25000000,  # 25 USDC
                amount_unit="lamports",
                decimals=6,
                chain=Chain.SOLANA,
                network=Network.SOL_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="helius", chain=Chain.SOLANA, network=Network.SOL_MAINNET, original_id="sol_spl_sig_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH ()-[r:TRANSFERRED {asset_contract: $mint}]->() RETURN r",
            mint=usdc_mint,
        )
        rec = await res.single()
        assert rec is not None
        assert rec["r"]["asset_symbol"] == "USDC"
        assert rec["r"]["amount"] == 25000000
        assert rec["r"]["decimals"] == 6


# ===========================================================================
# G. Bitcoin Multi-Input/Multi-Output Transaction
# ===========================================================================
@pytest.mark.asyncio
async def test_g_bitcoin_multi_input_multi_output(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="btc_mimo_txid_1",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=75000,
        native_value_unit="sat",
        fee=2500,
        fee_asset="BTC",
        transfers=[
            Transfer(
                from_address="bc1q_in_1",
                to_address=None,
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=50000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            ),
            Transfer(
                from_address="bc1q_in_2",
                to_address=None,
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=27500,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            ),
            Transfer(
                from_address=None,
                to_address="bc1q_out_1",
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=60000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            ),
            Transfer(
                from_address=None,
                to_address="bc1q_out_change",
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=15000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            ),
        ],
        provenance=TransactionProvenance(
            provider="mempool.space",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
            original_id="btc_mimo_txid_1",
        ),
    )

    btc_detail = BitcoinTransactionDetail(
        txid="btc_mimo_txid_1",
        inputs=[
            BitcoinVin(txid="prev_tx_a", vout=0, address="bc1q_in_1", value_sat=50000, coinbase=False),
            BitcoinVin(txid="prev_tx_b", vout=1, address="bc1q_in_2", value_sat=27500, coinbase=False),
        ],
        outputs=[
            BitcoinVout(n=0, address="bc1q_out_1", value_sat=60000, script_type="v0_p2wpkh"),
            BitcoinVout(n=1, address="bc1q_out_change", value_sat=15000, script_type="v0_p2wpkh"),
        ],
        total_input_sat=77500,
        total_output_sat=75000,
        fee_sat=2500,
        version=2,
    )

    await repo.save_transaction(tx, btc_detail)
    summary = await projector.project_all()

    assert summary.total_transactions == 1
    assert summary.total_bitcoin_inputs == 2
    assert summary.total_bitcoin_outputs == 2
    # Crucially, zero artificial :TRANSFERRED relationships projected for Bitcoin
    assert summary.total_transfers == 0

    async with neo4j_driver.session() as session:
        # 1. Verify Transaction Node
        res_tx = await session.run(
            "MATCH (t:Transaction {transaction_id: 'btc_mimo_txid_1'}) RETURN t"
        )
        rec_tx = await res_tx.single()
        assert rec_tx is not None
        assert rec_tx["t"]["chain"] == "bitcoin"

        # 2. Verify Inputs: (:Address)-[:SPENT_INPUT]->(:Transaction)
        res_in = await session.run(
            """
            MATCH (a:Address)-[r:SPENT_INPUT]->(t:Transaction {transaction_id: 'btc_mimo_txid_1'})
            RETURN a.normalized_address AS addr, r.value_sat AS val, r.spent_txid AS stx, r.spent_vout AS svout
            ORDER BY val DESC
            """
        )
        records_in = [record async for record in res_in]
        assert len(records_in) == 2
        assert records_in[0]["addr"] == "bc1q_in_1"
        assert records_in[0]["val"] == 50000
        assert records_in[0]["stx"] == "prev_tx_a"
        assert records_in[1]["addr"] == "bc1q_in_2"
        assert records_in[1]["val"] == 27500

        # 3. Verify Outputs: (:Transaction)-[:CREATED_OUTPUT]->(:Address)
        res_out = await session.run(
            """
            MATCH (t:Transaction {transaction_id: 'btc_mimo_txid_1'})-[r:CREATED_OUTPUT]->(a:Address)
            RETURN a.normalized_address AS addr, r.value_sat AS val, r.n AS n, r.script_type AS st
            ORDER BY n ASC
            """
        )
        records_out = [record async for record in res_out]
        assert len(records_out) == 2
        assert records_out[0]["addr"] == "bc1q_out_1"
        assert records_out[0]["val"] == 60000
        assert records_out[0]["n"] == 0
        assert records_out[1]["addr"] == "bc1q_out_change"
        assert records_out[1]["val"] == 15000
        assert records_out[1]["n"] == 1

        # 4. Strict UTXO semantics: NO artificial sender -> receiver relationships exist!
        res_transferred = await session.run(
            "MATCH (a:Address)-[r:TRANSFERRED]->(b:Address) WHERE r.chain = 'bitcoin' RETURN r"
        )
        records_fake = [record async for record in res_transferred]
        assert len(records_fake) == 0, "Disallowed: Bitcoin must NOT infer artificial sender->receiver edges!"


# ===========================================================================
# H. Cross-Chain Address Collision
# ===========================================================================
@pytest.mark.asyncio
async def test_h_cross_chain_address_collision(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    shared_address_string = "0x9999999999999999999999999999999999999999"

    # Transaction 1: EVM
    tx_evm = Transaction(
        transaction_id="tx_collision_evm",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=1000,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address=shared_address_string,
                to_address="0xReceiverEVM",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=1000,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_collision_evm"
        ),
    )

    # Transaction 2: TRON with same hex-formatted address string
    tx_tron = Transaction(
        transaction_id="tx_collision_tron",
        chain=Chain.TRON,
        network=Network.TRON_MAINNET,
        native_value=2000,
        native_value_unit="sun",
        transfers=[
            Transfer(
                from_address=shared_address_string,
                to_address="0xReceiverTron",
                asset_type=AssetType.NATIVE,
                asset_symbol="TRX",
                amount=2000,
                amount_unit="sun",
                chain=Chain.TRON,
                network=Network.TRON_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="trongrid", chain=Chain.TRON, network=Network.TRON_MAINNET, original_id="tx_collision_tron"
        ),
    )

    await repo.save_transaction(tx_evm)
    await repo.save_transaction(tx_tron)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            """
            MATCH (a:Address)
            WHERE a.normalized_address = $addr
            RETURN a.chain AS chain, a.network AS network, a.composite_id AS cid
            """,
            addr=shared_address_string.lower(),
        )
        records = [record async for record in res]
        assert len(records) == 2, "Address collision across chains MUST produce two distinct nodes!"
        chains = {r["chain"] for r in records}
        assert chains == {"evm", "tron"}


# ===========================================================================
# I. Repeated Projection / Idempotency
# ===========================================================================
@pytest.mark.asyncio
async def test_i_repeated_projection_idempotency(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="tx_idempotency_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=1000000,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address="0xAlice",
                to_address="0xBob",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=1000000,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_idempotency_1"
        ),
    )
    await repo.save_transaction(tx)

    # First projection run
    summary1 = await projector.project_all()
    assert summary1.total_transactions == 1
    assert summary1.total_transfers == 1

    # Second projection run on the same dataset
    summary2 = await projector.project_all()
    assert summary2.total_transactions == 1
    assert summary2.total_transfers == 1

    async with neo4j_driver.session() as session:
        res_node_count = await session.run("MATCH (n) RETURN count(n) AS cnt")
        rec_node = await res_node_count.single()
        # Exactly 2 Address nodes (Alice, Bob) + 1 Transaction node = 3 nodes
        assert rec_node["cnt"] == 3

        res_rel_count = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        rec_rel = await res_rel_count.single()
        # Exactly 1 TRANSFERRED relationship
        assert rec_rel["cnt"] == 1


# ===========================================================================
# J. Exact Large Integer Preservation: Explicitly 2^53 + 1 and beyond signed 64-bit
# ===========================================================================
@pytest.mark.asyncio
async def test_j_exact_large_integer_preservation(setup_databases, neo4j_driver):
    """Verify exact large integer preservation.

    - Explicitly tests 2^53 + 1 (9,007,199,254,740,993) to expose IEEE 754 float64 precision loss.
      Since 2^53 + 1 <= 2^63 - 1, amount is representable as a Neo4j signed 64-bit integer.
    - Explicitly tests a 78-digit integer (beyond signed 64-bit int max).
      Neo4j signed 64-bit amount must be null (None) because it cannot be represented as an int64,
      while amount_str contains the exact canonical decimal string representation.
    """
    repo, projector = setup_databases

    # 1. 2^53 + 1: Exposes IEEE 754 float64 precision loss (9007199254740993 -> 9007199254740992.0 in float)
    float_breaking_int = (2 ** 53) + 1  # Exactly 9007199254740993
    # 2. 78-digit integer: Exposes 64-bit int overflow and large scale storage beyond signed 64-bit
    huge_78_digit_int = 999999999999999999999999999999999999999999999999999999999999999999999999999999

    tx = Transaction(
        transaction_id="tx_large_ints_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=float_breaking_int,
        native_value_unit="wei",
        fee=float_breaking_int,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address="0xRich1",
                to_address="0xRich2",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=float_breaking_int,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
            Transfer(
                from_address="0xRich1",
                to_address="0xRich2",
                asset_type=AssetType.TOKEN,
                asset_symbol="HUGE",
                amount=huge_78_digit_int,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_large_ints_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        # Verify Transaction node exact values
        res_tx = await session.run(
            "MATCH (t:Transaction {transaction_id: 'tx_large_ints_1'}) RETURN t"
        )
        rec_tx = await res_tx.single()
        assert rec_tx is not None
        assert rec_tx["t"]["native_value"] == float_breaking_int
        assert int(rec_tx["t"]["native_value_str"]) == float_breaking_int

        # Verify Transfer relationships exact values
        res_rel = await session.run(
            """
            MATCH ()-[r:TRANSFERRED {transaction_id: 'tx_large_ints_1'}]->()
            RETURN r.asset_symbol AS sym, r.amount AS amt, r.amount_str AS amt_str
            ORDER BY sym ASC
            """
        )
        records = [record async for record in res_rel]
        assert len(records) == 2

        # 1. Native ETH transfer: 2^53 + 1 fits in signed 64-bit int, exactly equal
        assert records[0]["sym"] == "ETH"
        assert records[0]["amt"] == float_breaking_int
        assert records[0]["amt_str"] == str(float_breaking_int)
        assert int(records[0]["amt_str"]) == float_breaking_int

        # 2. Huge token transfer: 78 digits exceeds signed 64-bit, amt is None, amt_str is exact
        assert records[1]["sym"] == "HUGE"
        assert records[1]["amt"] is None, "Values beyond signed 64-bit must have amount as null/None in Neo4j"
        assert records[1]["amt_str"] == str(huge_78_digit_int)
        assert int(records[1]["amt_str"]) == huge_78_digit_int


# ===========================================================================
# K. Token Contract/Mint Distinction
# ===========================================================================
@pytest.mark.asyncio
async def test_k_token_contract_mint_distinction(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    # Two distinct tokens with the exact same symbol "USDT" on Ethereum
    contract_a = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    contract_fake = "0x1111111111111111111111111111111111111111"

    tx = Transaction(
        transaction_id="tx_token_distinction_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address="0xUserA",
                to_address="0xUserB",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract=contract_a,
                amount=1000000,
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
            Transfer(
                from_address="0xUserA",
                to_address="0xUserB",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract=contract_fake,
                amount=5000000,
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_token_distinction_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            """
            MATCH ()-[r:TRANSFERRED {transaction_id: 'tx_token_distinction_1'}]->()
            RETURN r.asset_contract AS contract, r.amount AS amount
            ORDER BY amount ASC
            """
        )
        records = [record async for record in res]
        assert len(records) == 2
        assert records[0]["contract"] == contract_a
        assert records[0]["amount"] == 1000000
        assert records[1]["contract"] == contract_fake
        assert records[1]["amount"] == 5000000


# ===========================================================================
# L. Transaction Identity
# ===========================================================================
@pytest.mark.asyncio
async def test_l_transaction_identity(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="0xnative_tx_identifier_123",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=500,
        native_value_unit="wei",
        transfers=[],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0xnative_tx_identifier_123"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            """
            MATCH (t:Transaction)
            RETURN t.transaction_id AS txid, t.composite_id AS cid, t.chain AS chain, t.network AS net, t.pg_id AS pg_id
            """
        )
        rec = await res.single()
        assert rec is not None
        # Must use native blockchain transaction identifier, NOT random UUID
        assert rec["txid"] == "0xnative_tx_identifier_123"
        assert rec["cid"] == f"evm:{Network.ETH_MAINNET.value}:0xnative_tx_identifier_123"
        assert rec["chain"] == "evm"
        assert rec["net"] == Network.ETH_MAINNET.value
        assert rec["pg_id"] is not None


# ===========================================================================
# M. Batch Processing
# ===========================================================================
@pytest.mark.asyncio
async def test_m_batch_processing(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    # Insert 5 transactions into PostgreSQL
    for i in range(1, 6):
        tx = Transaction(
            transaction_id=f"tx_batch_{i}",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            native_value=i * 100,
            native_value_unit="wei",
            transfers=[
                Transfer(
                    from_address=f"0xSender{i}",
                    to_address=f"0xReceiver{i}",
                    asset_type=AssetType.NATIVE,
                    asset_symbol="ETH",
                    amount=i * 100,
                    amount_unit="wei",
                    chain=Chain.EVM,
                    network=Network.ETH_MAINNET,
                )
            ],
            provenance=TransactionProvenance(
                provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id=f"tx_batch_{i}"
            ),
        )
        await repo.save_transaction(tx)

    # Process in bounded batches of 2 (expect 3 batches: 2 + 2 + 1)
    summary = await projector.project_all(batch_size=2)
    assert summary.total_batches == 3
    assert summary.total_transactions == 5
    assert summary.total_transfers == 5
    assert len(summary.batch_results) == 3
    assert summary.batch_results[0].transactions_count == 2
    assert summary.batch_results[1].transactions_count == 2
    assert summary.batch_results[2].transactions_count == 1

    # Verify that all 5 transactions exist in Neo4j without any duplicates or omissions
    async with neo4j_driver.session() as session:
        res = await session.run("MATCH (t:Transaction) RETURN count(t) AS cnt")
        rec = await res.single()
        assert rec["cnt"] == 5

        res_rel = await session.run("MATCH ()-[r:TRANSFERRED]->() RETURN count(r) AS cnt")
        rec_rel = await res_rel.single()
        assert rec_rel["cnt"] == 5


# ===========================================================================
# N. Projection Failure Handling
# ===========================================================================
@pytest.mark.asyncio
async def test_n_projection_failure_handling(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    # Create a batch with valid and invalid data to force a failure
    tx_valid = {
        "id": 100,
        "transaction_id": "tx_fail_test",
        "chain": "evm",
        "network": "eth-mainnet",
        "block_number": 1,
        "block_hash": "0x1",
        "timestamp": None,
        "transaction_type": "transfer",
        "status": "success",
        "fee": 0,
        "fee_asset": "ETH",
        "native_value": 0,
        "native_value_unit": "wei",
        "from_address": "0x1",
        "to_address": "0x2",
    }

    # Pass invalid Cypher input or simulate unexpected failure inside transaction
    with pytest.raises(BatchProjectionError) as exc_info:
        # Pass broken transfer data that triggers an error during Neo4j execution
        broken_transfers = [{
            "id": "not_an_int_and_broken",
            "chain": "evm",
            "network": "eth-mainnet",
            "transaction_pk": 100,
            "transaction_id": "tx_fail_test",
            "from_address": "0x1",
            "to_address": "0x2",
            "asset_type": "native",
            "asset_symbol": "ETH",
            "asset_contract": None,
            "amount": "not-a-number",  # to_neo4j_numeric will fail on int conversion
            "amount_unit": "wei",
            "decimals": 18,
        }]
        await projector.project_batch_data(
            tx_rows=[tx_valid],
            transfer_rows=broken_transfers,
            btc_details={},
            btc_vins={},
            btc_vouts={},
            batch_index=42,
        )

    err = exc_info.value
    assert err.batch_index == 42
    assert err.start_tx_id == 100
    assert err.end_tx_id == 100

    # Verify that the failed batch was rolled back and no partial nodes were committed
    async with neo4j_driver.session() as session:
        res = await session.run(
            "MATCH (t:Transaction {transaction_id: 'tx_fail_test'}) RETURN t"
        )
        rec = await res.single()
        assert rec is None, "Failed batch must be rolled back completely!"


# ===========================================================================
# O. Multiple Transfers in One Transaction
# ===========================================================================
@pytest.mark.asyncio
async def test_o_multiple_transfers_in_one_transaction(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="tx_multi_transfers_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address="0xAlice",
                to_address="0xBob",
                asset_type=AssetType.TOKEN,
                asset_symbol="DAI",
                asset_contract="0x6B175474E89094C44Da98b954EedeAC495271d0F",
                amount=1000000000000000000,
                amount_unit="wei",
                decimals=18,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
            Transfer(
                from_address="0xAlice",
                to_address="0xCharlie",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDC",
                asset_contract="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                amount=50000000,
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
            Transfer(
                from_address="0xCharlie",
                to_address="0xDave",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                amount=75000000,
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_multi_transfers_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            """
            MATCH ()-[r:TRANSFERRED {transaction_id: 'tx_multi_transfers_1'}]->()
            RETURN r.asset_symbol AS sym, r.amount AS amount
            ORDER BY sym ASC
            """
        )
        records = [record async for record in res]
        assert len(records) == 3
        symbols = [r["sym"] for r in records]
        assert symbols == ["DAI", "USDC", "USDT"]


# ===========================================================================
# P. Repeated Transfers Between Same Addresses
# ===========================================================================
@pytest.mark.asyncio
async def test_p_repeated_transfers_between_same_addresses(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    # Two distinct transfers with the EXACT same endpoints and asset in the same transaction
    tx = Transaction(
        transaction_id="tx_repeated_endpoints_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address="0xAlice",
                to_address="0xBob",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                amount=100000000,  # 100 USDT
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
            Transfer(
                from_address="0xAlice",
                to_address="0xBob",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDT",
                asset_contract="0xdAC17F958D2ee523a2206206994597C13D831ec7",
                amount=250000000,  # 250 USDT
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            ),
        ],
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_repeated_endpoints_1"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        res = await session.run(
            """
            MATCH (f:Address {normalized_address: '0xalice'})-[r:TRANSFERRED]->(t:Address {normalized_address: '0xbob'})
            RETURN r.amount AS amount, r.transfer_id AS tid
            ORDER BY amount ASC
            """
        )
        records = [record async for record in res]
        # Mandatory requirement: must NOT be collapsed into one relationship!
        assert len(records) == 2
        assert records[0]["amount"] == 100000000
        assert records[1]["amount"] == 250000000
        assert records[0]["tid"] != records[1]["tid"]


# ===========================================================================
# Q. No Inferred Relationships
# ===========================================================================
@pytest.mark.asyncio
async def test_q_no_inferred_relationships(setup_databases, neo4j_driver):
    repo, projector = setup_databases

    # A contract call transaction where from and to addresses appear, but no asset transfer occurred
    tx = Transaction(
        transaction_id="tx_contract_call_no_transfer",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        from_address="0xCallerAddress",
        to_address="0xContractAddress",
        native_value=0,
        native_value_unit="wei",
        transaction_type=TransactionType.CONTRACT_CALL,
        status=TransactionStatus.SUCCESS,
        transfers=[],  # No asset movements
        provenance=TransactionProvenance(
            provider="etherscan", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="tx_contract_call_no_transfer"
        ),
    )
    await repo.save_transaction(tx)
    await projector.project_all()

    async with neo4j_driver.session() as session:
        # Addresses should exist
        res_addr = await session.run("MATCH (a:Address) RETURN count(a) AS cnt")
        rec_addr = await res_addr.single()
        assert rec_addr["cnt"] == 2

        # Transaction node should exist
        res_tx = await session.run("MATCH (t:Transaction) RETURN count(t) AS cnt")
        rec_tx = await res_tx.single()
        assert rec_tx["cnt"] == 1

        # No economic relationships inferred
        res_rel = await session.run("MATCH ()-[r:TRANSFERRED]->() RETURN count(r) AS cnt")
        rec_rel = await res_rel.single()
        assert rec_rel["cnt"] == 0, "Disallowed: Cannot infer economic transfers when canonical transfers[] is empty!"


# ===========================================================================
# R. Bitcoin Addressless Records (Coinbase and OP_RETURN)
# ===========================================================================
@pytest.mark.asyncio
async def test_r_bitcoin_addressless_records(setup_databases, neo4j_driver):
    """Verify that addressless Bitcoin inputs (Coinbase) and outputs (OP_RETURN)

    are not silently discarded and are projected into explicit nodes.
    """
    repo, projector = setup_databases

    tx = Transaction(
        transaction_id="btc_coinbase_opreturn_tx",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=5000000000,
        native_value_unit="sat",
        fee=0,
        fee_asset="BTC",
        transfers=[
            Transfer(
                from_address=None,
                to_address="bc1q_miner_reward",
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=5000000000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            )
        ],
        provenance=TransactionProvenance(
            provider="mempool.space",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
            original_id="btc_coinbase_opreturn_tx",
        ),
    )

    btc_detail = BitcoinTransactionDetail(
        txid="btc_coinbase_opreturn_tx",
        inputs=[
            # Coinbase input has address=None, coinbase=True
            BitcoinVin(txid=None, vout=None, address=None, value_sat=0, coinbase=True),
        ],
        outputs=[
            # Standard miner reward output
            BitcoinVout(n=0, address="bc1q_miner_reward", value_sat=5000000000, script_type="v0_p2wpkh"),
            # OP_RETURN data output (burn/null data) with address=None
            BitcoinVout(n=1, address=None, value_sat=0, script_type="op_return"),
        ],
        total_input_sat=0,
        total_output_sat=5000000000,
        fee_sat=0,
        version=2,
    )

    await repo.save_transaction(tx, btc_detail)
    summary = await projector.project_all()

    assert summary.total_transactions == 1
    assert summary.total_bitcoin_inputs == 1, "Coinbase input must not be discarded!"
    assert summary.total_bitcoin_outputs == 2, "OP_RETURN output must not be discarded!"

    async with neo4j_driver.session() as session:
        # Verify Coinbase input edge: (:Address {normalized_address: '(coinbase)'})-[:SPENT_INPUT]->(:Transaction)
        res_cb = await session.run(
            """
            MATCH (cb:Address {normalized_address: '(coinbase)'})-[r:SPENT_INPUT]->(t:Transaction {transaction_id: 'btc_coinbase_opreturn_tx'})
            RETURN cb, r
            """
        )
        rec_cb = await res_cb.single()
        assert rec_cb is not None
        assert rec_cb["r"]["coinbase"] is True
        assert rec_cb["cb"]["composite_id"] == f"bitcoin:{Network.BTC_MAINNET.value}:coinbase"

        # Verify OP_RETURN output edge: (:Transaction)-[:CREATED_OUTPUT]->(:Address {normalized_address: '(op_return:1)'})
        res_op = await session.run(
            """
            MATCH (t:Transaction {transaction_id: 'btc_coinbase_opreturn_tx'})-[r:CREATED_OUTPUT]->(op:Address)
            WHERE r.script_type = 'op_return'
            RETURN op, r
            """
        )
        rec_op = await res_op.single()
        assert rec_op is not None
        assert rec_op["r"]["n"] == 1
        assert rec_op["op"]["normalized_address"] == "(op_return:1)"
        assert rec_op["op"]["composite_id"] == f"bitcoin:{Network.BTC_MAINNET.value}:op_return:btc_coinbase_opreturn_tx:1"


# ===========================================================================
# 18. Neo4j Community Integration Test Evidence
# ===========================================================================
@pytest.mark.asyncio
async def test_18_neo4j_community_integration_metadata(neo4j_driver):
    """Verifies live connectivity to Neo4j Community edition and inspects version and constraints."""
    async with neo4j_driver.session() as session:
        # Check Neo4j edition & version
        res_ver = await session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
        rec_ver = await res_ver.single()
        assert rec_ver is not None
        edition = rec_ver["edition"].lower()
        versions = rec_ver["versions"]
        assert "community" in edition, f"Expected Community edition, got: {edition}"
        assert len(versions) > 0

        # Check that constraints are active
        res_c = await session.run("SHOW CONSTRAINTS")
        constraints = [record["name"] async for record in res_c]
        expected_constraints = {
            "address_composite_id",
            "address_identity",
            "transaction_composite_id",
            "transaction_identity",
            "transferred_transfer_id",
            "spent_input_id",
            "created_output_id",
        }
        for c in expected_constraints:
            assert c in constraints, f"Missing expected constraint: {c}"
