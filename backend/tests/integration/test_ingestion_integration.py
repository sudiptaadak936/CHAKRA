import pytest
import asyncpg
import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.schemas.transaction import (
    Chain, Network, AssetType, TransactionStatus, TransactionType,
    Transaction, Transfer, TransactionProvenance,
    BitcoinTransactionDetail, BitcoinVin, BitcoinVout
)
from app.schemas.ingestion import IngestionStatus, IngestionResult, AddressHistoryIngestionResult
from app.db.repository import TransactionRepository
from app.services.ingestion_service import IngestionService
from app.providers.data_registry import DataProviderRegistry
from app.providers.data_base import BaseDataProvider, ProviderCapability

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

@pytest.fixture
def sample_evm_tx():
    return Transaction(
        transaction_id="0xint_evm_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        chain_id=1,
        block_number=1000,
        block_hash="0xabc123",
        timestamp=datetime.now(timezone.utc),
        from_address="0xSender111111111111111111111111111111111111",
        to_address="0xReceiver2222222222222222222222222222222222",
        native_value=2500000000000000000,
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21000000000000,
        fee_asset="ETH",
        transfers=[
            Transfer(
                from_address="0xSender111111111111111111111111111111111111",
                to_address="0xReceiver2222222222222222222222222222222222",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=2500000000000000000,
                amount_unit="wei",
                decimals=18,
                chain=Chain.EVM,
                network=Network.ETH_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="etherscan",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0xint_evm_1",
            normalized_at=datetime.now(timezone.utc)
        )
    )

@pytest.fixture
def sample_bitcoin_tx():
    return Transaction(
        transaction_id="btc_int_tx_1",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        timestamp=datetime.now(timezone.utc),
        native_value=50000,
        native_value_unit="sat",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=2000,
        fee_asset="BTC",
        transfers=[
            Transfer(
                from_address="bc1q_in_addr",
                to_address=None,
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=52000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET
            ),
            Transfer(
                from_address=None,
                to_address="bc1q_out_addr",
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=50000,
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET
            )
        ],
        provenance=TransactionProvenance(
            provider="mempool.space",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
            original_id="btc_int_tx_1",
            normalized_at=datetime.now(timezone.utc)
        )
    )

@pytest.mark.asyncio
async def test_repeated_ingestion_idempotency(repo: TransactionRepository, db_pool, sample_evm_tx):
    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "etherscan"
    mock_adapter.supported_chains = [Chain.EVM]
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(return_value=sample_evm_tx)

    registry = DataProviderRegistry()
    registry.register(Chain.EVM, mock_adapter, primary=True)

    service = IngestionService(registry=registry, repository=repo)

    res1 = await service.ingest_transaction(
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        tx_id="0xint_evm_1"
    )
    assert res1.success is True
    assert res1.status == IngestionStatus.SUCCESS

    res2 = await service.ingest_transaction(
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        tx_id="0xint_evm_1"
    )
    assert res2.success is True
    assert res2.status == IngestionStatus.ALREADY_EXISTS

    async with db_pool.acquire() as conn:
        tx_count = await conn.fetchval("SELECT count(*) FROM transactions WHERE transaction_id = $1", "0xint_evm_1")
        assert tx_count == 1
        transfer_count = await conn.fetchval(
            "SELECT count(*) FROM transfers t JOIN transactions tx ON t.transaction_pk = tx.id WHERE tx.transaction_id = $1",
            "0xint_evm_1"
        )
        assert transfer_count == 1

@pytest.mark.asyncio
async def test_concurrent_duplicate_ingestion(repo: TransactionRepository, db_pool, sample_evm_tx):
    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "etherscan"
    mock_adapter.supported_chains = [Chain.EVM]
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(return_value=sample_evm_tx)

    registry = DataProviderRegistry()
    registry.register(Chain.EVM, mock_adapter, primary=True)

    service = IngestionService(registry=registry, repository=repo)

    concurrency = 8
    tasks = [
        service.ingest_transaction(
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            tx_id="0xint_evm_1"
        )
        for _ in range(concurrency)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        assert not isinstance(r, Exception), f"Task raised unexpected exception: {r}"
        assert r.success is True
        assert r.status in (IngestionStatus.SUCCESS, IngestionStatus.ALREADY_EXISTS)

    success_count = sum(1 for r in results if r.status == IngestionStatus.SUCCESS)
    assert success_count >= 1

    async with db_pool.acquire() as conn:
        tx_count = await conn.fetchval("SELECT count(*) FROM transactions WHERE transaction_id = $1", "0xint_evm_1")
        assert tx_count == 1
        transfer_count = await conn.fetchval(
            "SELECT count(*) FROM transfers t JOIN transactions tx ON t.transaction_pk = tx.id WHERE tx.transaction_id = $1",
            "0xint_evm_1"
        )
        assert transfer_count == 1

@pytest.mark.asyncio
async def test_persistence_failure_propagation():
    broken_tx = Transaction(
        transaction_id="0xbroken_1",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=0,
        native_value_unit="wei",
        transfers=[],
        provenance=TransactionProvenance(
            provider="test",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0xbroken_1"
        )
    )

    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "test"
    mock_adapter.supported_chains = [Chain.EVM]
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(return_value=broken_tx)

    registry = DataProviderRegistry()
    registry.register(Chain.EVM, mock_adapter, primary=True)

    mock_repo = AsyncMock(spec=TransactionRepository)
    mock_repo.transaction_exists.return_value = False
    mock_repo.save_transaction.side_effect = asyncpg.PostgresError("Simulated DB connection failure")

    service = IngestionService(registry=registry, repository=mock_repo)

    result = await service.ingest_transaction(
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        tx_id="0xbroken_1"
    )

    assert result.status == IngestionStatus.FAILED
    assert result.success is False
    assert "Simulated DB connection failure" in (result.error_message or "")

@pytest.mark.asyncio
async def test_end_to_end_bitcoin_utxo_ingestion(repo: TransactionRepository, db_pool, sample_bitcoin_tx):
    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "mempool.space"
    mock_adapter.supported_chains = [Chain.BITCOIN]
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(return_value=sample_bitcoin_tx)

    registry = DataProviderRegistry()
    registry.register(Chain.BITCOIN, mock_adapter, primary=True)

    service = IngestionService(registry=registry, repository=repo)

    res = await service.ingest_transaction(
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        tx_id="btc_int_tx_1"
    )

    assert res.status == IngestionStatus.SUCCESS
    assert res.transaction is not None
    assert res.transaction.transaction_id == "btc_int_tx_1"

    saved_tx = await repo.get_transaction("btc_int_tx_1", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_tx is not None
    assert len(saved_tx.transfers) == 2

    saved_detail = await repo.get_bitcoin_detail("btc_int_tx_1", Chain.BITCOIN, Network.BTC_MAINNET)
    assert saved_detail is not None
    assert saved_detail.total_input_sat == 52000
    assert saved_detail.total_output_sat == 50000
    assert saved_detail.fee_sat == 2000

@pytest.mark.asyncio
async def test_address_history_ingestion_pipeline(repo: TransactionRepository, db_pool, sample_evm_tx):
    tx2 = Transaction(
        transaction_id="0xint_evm_2",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=1000000000000000000,
        native_value_unit="wei",
        transfers=[],
        provenance=TransactionProvenance(
            provider="etherscan",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0xint_evm_2"
        )
    )

    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "etherscan"
    mock_adapter.supported_chains = [Chain.EVM]
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_address_transactions = AsyncMock(return_value=[sample_evm_tx, tx2])

    registry = DataProviderRegistry()
    registry.register(Chain.EVM, mock_adapter, primary=True)

    service = IngestionService(registry=registry, repository=repo)

    res = await service.ingest_address_history(
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        address="0xSender111111111111111111111111111111111111",
        limit=10
    )

    assert res.success is True
    assert res.transactions_discovered == 2
    assert res.transactions_persisted == 2
    assert res.transactions_already_existed == 0

    res_repeat = await service.ingest_address_history(
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        address="0xSender111111111111111111111111111111111111",
        limit=10
    )

    assert res_repeat.success is True
    assert res_repeat.transactions_discovered == 2
    assert res_repeat.transactions_persisted == 0
    assert res_repeat.transactions_already_existed == 2
