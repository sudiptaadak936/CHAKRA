import pytest
import asyncpg
import asyncio
import os
from datetime import datetime, timezone
import json

from app.schemas.transaction import (
    Transaction, Transfer, TransactionProvenance,
    BitcoinTransactionDetail, BitcoinVin, BitcoinVout,
    AssetType, TransactionStatus, TransactionType,
    Chain, Network
)
from app.db.repository import TransactionRepository

from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "chakra")

@pytest.fixture
async def db_pool():
    pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )
    yield pool
    await pool.close()

@pytest.fixture
async def repo(db_pool):
    # Clear tables before each test
    async with db_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
    return TransactionRepository(db_pool)

@pytest.mark.asyncio
async def test_save_and_get_evm_transaction(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="0x123",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        chain_id=1,
        block_number=1000,
        block_hash="0xabc",
        timestamp=datetime.now(timezone.utc),
        from_address="0xA",
        to_address="0xB",
        native_value=1000000000000000000, # 1 ETH
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21000,
        fee_asset="ETH",
        transfers=[],
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0x123",
            normalized_at=datetime.now(timezone.utc)
        )
    )
    
    await repo.save_transaction(tx)
    saved_tx = await repo.get_transaction("0x123", Chain.EVM, Network.ETH_MAINNET)
    
    assert saved_tx is not None
    assert saved_tx.transaction_id == "0x123"
    assert saved_tx.native_value == 1000000000000000000
    assert saved_tx.from_address == "0xA"
    assert saved_tx.fee == 21000
    assert saved_tx.provenance.provider == "test"

@pytest.mark.asyncio
async def test_save_erc20_transaction(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="0x456",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        fee=50000,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address="0xC",
                to_address="0xD",
                asset_type=AssetType.TOKEN,
                asset_symbol="USDC",
                asset_contract="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                amount=1000000, # 1 USDC
                amount_unit="mwei",
                decimals=6,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0x456"
        )
    )
    
    await repo.save_transaction(tx)
    saved_tx = await repo.get_transaction("0x456", Chain.EVM, Network.ETH_MAINNET)
    assert saved_tx is not None
    assert len(saved_tx.transfers) == 1
    assert saved_tx.transfers[0].amount == 1000000
    assert saved_tx.transfers[0].asset_symbol == "USDC"
    
@pytest.mark.asyncio
async def test_large_integer_precision(repo: TransactionRepository):
    # 9,007,199,254,740,991 is JS max safe int. Let's use something much bigger.
    huge_amount = 99999999999999999999999999999999999999999999999
    tx = Transaction(
        transaction_id="0xlarge",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=huge_amount,
        native_value_unit="wei",
        fee=huge_amount,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address="0xC",
                to_address="0xD",
                asset_type=AssetType.TOKEN,
                asset_symbol="HUGE",
                amount=huge_amount,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0xlarge"
        )
    )
    
    await repo.save_transaction(tx)
    saved_tx = await repo.get_transaction("0xlarge", Chain.EVM, Network.ETH_MAINNET)
    assert saved_tx is not None
    assert saved_tx.native_value == huge_amount
    assert saved_tx.fee == huge_amount
    assert saved_tx.transfers[0].amount == huge_amount

@pytest.mark.asyncio
async def test_duplicate_idempotency(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="0xdup",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=1,
        native_value_unit="wei",
        transfers=[
            Transfer(
                from_address="0xC", to_address="0xD", asset_type=AssetType.NATIVE,
                asset_symbol="ETH", amount=1, amount_unit="wei", chain=Chain.EVM, network=Network.ETH_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0xdup"
        )
    )
    
    # Save first time
    await repo.save_transaction(tx)
    # Save second time
    await repo.save_transaction(tx)
    
    saved_tx = await repo.get_transaction("0xdup", Chain.EVM, Network.ETH_MAINNET)
    assert saved_tx is not None
    # Ensure transfers are not duplicated
    assert len(saved_tx.transfers) == 1

@pytest.mark.asyncio
async def test_bitcoin_utxo_save(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="btc123",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=5000,
        native_value_unit="sat",
        transfers=[
            Transfer(
                from_address="bc1q_in", to_address=None, asset_type=AssetType.NATIVE,
                asset_symbol="BTC", amount=6000, amount_unit="sat", chain=Chain.BITCOIN, network=Network.BTC_MAINNET
            ),
            Transfer(
                from_address=None, to_address="bc1q_out", asset_type=AssetType.NATIVE,
                asset_symbol="BTC", amount=5000, amount_unit="sat", chain=Chain.BITCOIN, network=Network.BTC_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.BITCOIN, network=Network.BTC_MAINNET, original_id="btc123"
        )
    )
    
    btc_detail = BitcoinTransactionDetail(
        txid="btc123",
        inputs=[BitcoinVin(txid="prev1", vout=0, address="bc1q_in", value_sat=6000, coinbase=False)],
        outputs=[BitcoinVout(n=0, address="bc1q_out", value_sat=5000, script_type="v0_p2wpkh")],
        total_input_sat=6000,
        total_output_sat=5000,
        fee_sat=1000,
        version=2
    )
    
    await repo.save_transaction(tx, btc_detail)
    saved_tx = await repo.get_transaction("btc123", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_tx is not None
    assert len(saved_tx.transfers) == 2
    
    saved_detail = await repo.get_bitcoin_detail("btc123", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_detail is not None
    assert saved_detail.total_input_sat == 6000
    assert len(saved_detail.inputs) == 1
    assert saved_detail.inputs[0].address == "bc1q_in"
    assert len(saved_detail.outputs) == 1
    assert saved_detail.outputs[0].value_sat == 5000



@pytest.mark.asyncio
async def test_atomic_rollback_by_omitting_provenance(repo: TransactionRepository, db_pool):
    # If the python code throws inside the transaction block, it rolls back.
    # Let's insert directly with a missing field in the repo.
    tx = Transaction(
        transaction_id="rollback_test_2",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transfers=[],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="rollback_test_2"
        )
    )
    
    # Break the schema temporarily using object __dict__
    object.__setattr__(tx.provenance, 'provider', None) 
    # This will fail postgres constraint "provider NOT NULL"
    
    try:
        await repo.save_transaction(tx)
    except asyncpg.exceptions.NotNullViolationError:
        pass
        
    saved_tx = await repo.get_transaction("rollback_test_2", Chain.EVM, Network.ETH_MAINNET)
    assert saved_tx is None # The parent transaction was rolled back

@pytest.mark.asyncio
async def test_null_optional_fields(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="0xnulls",
        chain=Chain.SOLANA,
        network=Network.SOL_MAINNET,
        # Omit block_hash, timestamp, from_address, to_address, fee
        native_value=0,
        native_value_unit="lamports",
        transfers=[],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.SOLANA, network=Network.SOL_MAINNET, original_id="0xnulls"
        )
    )
    
    await repo.save_transaction(tx)
    saved_tx = await repo.get_transaction("0xnulls", Chain.SOLANA, Network.SOL_MAINNET)
    assert saved_tx is not None
    assert saved_tx.block_hash is None
    assert saved_tx.timestamp is None
    assert saved_tx.from_address is None
    assert saved_tx.to_address is None
    assert saved_tx.fee is None

@pytest.mark.asyncio
async def test_tron_transaction(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="tron123",
        chain=Chain.TRON,
        network=Network.TRON_MAINNET,
        native_value=1000000, # 1 TRX
        native_value_unit="sun",
        transfers=[
            Transfer(
                from_address="TFrom", to_address="TTo", asset_type=AssetType.TOKEN,
                asset_symbol="USDT", asset_contract="TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
                amount=5000000, amount_unit="sun", decimals=6, chain=Chain.TRON, network=Network.TRON_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="test", chain=Chain.TRON, network=Network.TRON_MAINNET, original_id="tron123"
        )
    )
    await repo.save_transaction(tx)
    saved_tx = await repo.get_transaction("tron123", Chain.TRON, Network.TRON_MAINNET)
    assert saved_tx.native_value == 1000000
    assert saved_tx.transfers[0].asset_symbol == "USDT"
    assert saved_tx.transfers[0].asset_contract == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

@pytest.mark.asyncio
async def test_solana_transactions(repo: TransactionRepository):
    # Native
    tx_native = Transaction(
        transaction_id="sol1", chain=Chain.SOLANA, network=Network.SOL_MAINNET,
        native_value=1000000000, native_value_unit="lamports",
        fee=5000, fee_asset="SOL", transfers=[],
        provenance=TransactionProvenance(provider="test", chain=Chain.SOLANA, network=Network.SOL_MAINNET, original_id="sol1")
    )
    
    # SPL
    tx_spl = Transaction(
        transaction_id="sol2", chain=Chain.SOLANA, network=Network.SOL_MAINNET,
        native_value=0, native_value_unit="lamports",
        fee=5000, fee_asset="SOL",
        transfers=[
            Transfer(
                from_address="SFrom", to_address="STo", asset_type=AssetType.TOKEN,
                asset_symbol="USDC", asset_contract="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                amount=1000000, amount_unit="lamports", decimals=6, chain=Chain.SOLANA, network=Network.SOL_MAINNET
            )
        ],
        provenance=TransactionProvenance(provider="test", chain=Chain.SOLANA, network=Network.SOL_MAINNET, original_id="sol2")
    )
    
    await repo.save_transaction(tx_native)
    await repo.save_transaction(tx_spl)
    
    saved_native = await repo.get_transaction("sol1", Chain.SOLANA, Network.SOL_MAINNET)
    assert saved_native.native_value == 1000000000
    
    saved_spl = await repo.get_transaction("sol2", Chain.SOLANA, Network.SOL_MAINNET)
    assert len(saved_spl.transfers) == 1
    assert saved_spl.transfers[0].asset_symbol == "USDC"

@pytest.mark.asyncio
async def test_transaction_exists(repo: TransactionRepository):
    tx = Transaction(
        transaction_id="exist1", chain=Chain.EVM, network=Network.ETH_MAINNET,
        native_value=0, native_value_unit="wei", transfers=[],
        provenance=TransactionProvenance(provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="exist1")
    )
    
    assert not await repo.transaction_exists("exist1", Chain.EVM, Network.ETH_MAINNET)
    await repo.save_transaction(tx)
    assert await repo.transaction_exists("exist1", Chain.EVM, Network.ETH_MAINNET)

@pytest.mark.asyncio
async def test_concurrent_duplicate_saves(repo: TransactionRepository, db_pool):
    """
    Test concurrent duplicate saves of an identical canonical transaction.
    Verifies that simultaneous save_transaction calls do not trigger UniqueViolationError
    and do not duplicate transaction, transfer, provenance, or Bitcoin detail records.
    """
    tx = Transaction(
        transaction_id="concurrent_btc_tx_1",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=7000,
        native_value_unit="sat",
        fee=1000,
        fee_asset="BTC",
        transfers=[
            Transfer(
                from_address="bc1q_concur_in", to_address=None, asset_type=AssetType.NATIVE,
                asset_symbol="BTC", amount=8000, amount_unit="sat", chain=Chain.BITCOIN, network=Network.BTC_MAINNET
            ),
            Transfer(
                from_address=None, to_address="bc1q_concur_out", asset_type=AssetType.NATIVE,
                asset_symbol="BTC", amount=7000, amount_unit="sat", chain=Chain.BITCOIN, network=Network.BTC_MAINNET
            ),
        ],
        provenance=TransactionProvenance(
            provider="test-concurrent",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
            original_id="concurrent_btc_tx_1",
            normalized_at=datetime.now(timezone.utc),
        ),
    )

    btc_detail = BitcoinTransactionDetail(
        txid="concurrent_btc_tx_1",
        inputs=[BitcoinVin(txid="prev_concur", vout=0, address="bc1q_concur_in", value_sat=8000, coinbase=False)],
        outputs=[BitcoinVout(n=0, address="bc1q_concur_out", value_sat=7000, script_type="v0_p2wpkh")],
        total_input_sat=8000,
        total_output_sat=7000,
        fee_sat=1000,
        version=2,
    )

    # Launch 10 simultaneous concurrent saves of the exact same transaction
    concurrency_count = 10
    tasks = [repo.save_transaction(tx, btc_detail) for _ in range(concurrency_count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify no exceptions (e.g. UniqueViolationError) occurred
    for r in results:
        assert not isinstance(r, Exception), f"Concurrent save raised exception: {r}"

    # Verify exact database row counts directly via SQL
    async with db_pool.acquire() as conn:
        tx_count = await conn.fetchval(
            "SELECT COUNT(*) FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
            "concurrent_btc_tx_1", Chain.BITCOIN.value, Network.BTC_MAINNET.value
        )
        assert tx_count == 1, f"Expected exactly 1 transaction row, found {tx_count}"

        tx_id = await conn.fetchval(
            "SELECT id FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
            "concurrent_btc_tx_1", Chain.BITCOIN.value, Network.BTC_MAINNET.value
        )

        transfer_count = await conn.fetchval(
            "SELECT COUNT(*) FROM transfers WHERE transaction_pk = $1", tx_id
        )
        assert transfer_count == 2, f"Expected exactly 2 transfer rows, found {transfer_count}"

        prov_count = await conn.fetchval(
            "SELECT COUNT(*) FROM transaction_provenance WHERE transaction_pk = $1", tx_id
        )
        assert prov_count == 1, f"Expected exactly 1 provenance row, found {prov_count}"

        detail_count = await conn.fetchval(
            "SELECT COUNT(*) FROM bitcoin_transaction_details WHERE transaction_pk = $1", tx_id
        )
        assert detail_count == 1, f"Expected exactly 1 bitcoin_transaction_details row, found {detail_count}"

        detail_id = await conn.fetchval(
            "SELECT id FROM bitcoin_transaction_details WHERE transaction_pk = $1", tx_id
        )

        vin_count = await conn.fetchval(
            "SELECT COUNT(*) FROM bitcoin_vins WHERE detail_pk = $1", detail_id
        )
        assert vin_count == 1, f"Expected exactly 1 bitcoin_vins row, found {vin_count}"

        vout_count = await conn.fetchval(
            "SELECT COUNT(*) FROM bitcoin_vouts WHERE detail_pk = $1", detail_id
        )
        assert vout_count == 1, f"Expected exactly 1 bitcoin_vouts row, found {vout_count}"

    # Verify canonical reconstruction
    saved_tx = await repo.get_transaction("concurrent_btc_tx_1", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_tx is not None
    assert len(saved_tx.transfers) == 2
    assert saved_tx.provenance.provider == "test-concurrent"

    saved_btc = await repo.get_bitcoin_detail("concurrent_btc_tx_1", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_btc is not None
    assert len(saved_btc.inputs) == 1
    assert len(saved_btc.outputs) == 1

