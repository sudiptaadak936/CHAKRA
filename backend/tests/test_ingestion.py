"""Unit and mock tests for CHAKRA Step 1D Live Blockchain Ingestion & Normalization Pipeline."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import httpx

from app.schemas.chain import Chain, Network
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)
from app.schemas.ingestion import IngestionResult, IngestionStatus, AddressHistoryIngestionResult
from app.providers.data_base import (
    BaseDataProvider,
    ProviderCapability,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TransactionNotFoundError,
    MalformedResponseError,
    NormalizationError,
    UnsupportedOperationError,
)
from app.providers.data_registry import DataProviderRegistry
from app.providers.adapters.evm import EVMTransactionAdapter, BlockscoutEVMAdapter
from app.providers.adapters.tron import TronTransactionAdapter
from app.providers.adapters.bitcoin import BitcoinTransactionAdapter, BlockstreamBitcoinAdapter
from app.providers.adapters.solana import HeliusSolanaAdapter, PublicSolanaRPCAdapter
from app.services.ingestion_service import IngestionService


# ---------------------------------------------------------------------------
# 1. EVM Native Transaction Normalization
# ---------------------------------------------------------------------------
def test_evm_native_normalization():
    adapter = EVMTransactionAdapter("dummy_key", default_chain_id=1)
    raw_tx = {
        "hash": "0x1111111111111111111111111111111111111111111111111111111111111111",
        "from": "0xSenderA",
        "to": "0xReceiverB",
        "value": "0xde0b6b3a7640000", # 1 ETH in hex wei
        "gasUsed": "21000",
        "gasPrice": "20000000000", # 20 gwei
        "blockNumber": "1000",
        "timeStamp": "1700000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0x1111", chain_id=1, network=Network.ETH_MAINNET)
    assert tx.transaction_id == "0x1111"
    assert tx.chain == Chain.EVM
    assert tx.network == Network.ETH_MAINNET
    assert tx.native_value == 1000000000000000000
    assert tx.native_value_unit == "wei"
    assert tx.fee == 21000 * 20000000000
    assert tx.fee_asset == "ETH"
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 1000000000000000000
    assert tx.transfers[0].asset_symbol == "ETH"
    assert tx.transfers[0].asset_type == AssetType.NATIVE
    assert tx.provenance.provider == "etherscan-evm"


# ---------------------------------------------------------------------------
# 2. EVM ERC20 Normalization
# ---------------------------------------------------------------------------
def test_evm_erc20_normalization():
    adapter = EVMTransactionAdapter("dummy_key", default_chain_id=1)
    raw_tx = {
        "hash": "0x2222",
        "from": "0xSenderA",
        "to": "0xTokenContract",
        "value": "0x0",
        "gasUsed": "50000",
        "gasPrice": "20000000000",
        "blockNumber": "1001",
        "timeStamp": "1700000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    token_transfers_raw = [{
        "from": "0xSenderA",
        "to": "0xReceiverB",
        "tokenSymbol": "USDC",
        "contractAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "value": "1000000", # 1 USDC (6 decimals)
        "tokenDecimal": "6",
    }]
    tx = adapter._normalize(raw_tx, token_transfers_raw, "0x2222", chain_id=1, network=Network.ETH_MAINNET)
    assert tx.native_value == 0
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDC"
    assert t.asset_contract == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    assert t.amount == 1000000
    assert t.decimals == 6
    assert t.amount_unit == "base_unit"


# ---------------------------------------------------------------------------
# 3. BNB EVM Normalization
# ---------------------------------------------------------------------------
def test_bnb_evm_normalization():
    adapter = EVMTransactionAdapter("dummy_key", default_chain_id=56)
    raw_tx = {
        "hash": "0xbnb1",
        "from": "0xUserBNB",
        "to": "0xTargetBNB",
        "value": "1000000000000000000",
        "gasUsed": "21000",
        "gasPrice": "5000000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0xbnb1", chain_id=56, network=Network.BNB_MAINNET)
    assert tx.chain == Chain.EVM
    assert tx.network == Network.BNB_MAINNET
    assert tx.chain_id == 56
    assert tx.fee_asset == "BNB"
    assert tx.transfers[0].asset_symbol == "BNB"


# ---------------------------------------------------------------------------
# 4. Polygon EVM Normalization
# ---------------------------------------------------------------------------
def test_polygon_evm_normalization():
    adapter = EVMTransactionAdapter("dummy_key", default_chain_id=137)
    raw_tx = {
        "hash": "0xpoly1",
        "from": "0xUserPoly",
        "to": "0xTargetPoly",
        "value": "2000000000000000000",
        "gasUsed": "21000",
        "gasPrice": "30000000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0xpoly1", chain_id=137, network=Network.POLYGON_MAINNET)
    assert tx.chain == Chain.EVM
    assert tx.network == Network.POLYGON_MAINNET
    assert tx.chain_id == 137
    assert tx.fee_asset == "MATIC"
    assert tx.transfers[0].asset_symbol == "MATIC"


# ---------------------------------------------------------------------------
# 5. TRON Native Transaction Normalization
# ---------------------------------------------------------------------------
def test_tron_native_normalization():
    adapter = TronTransactionAdapter("dummy_key")
    raw_tx = {
        "txID": "tron_tx_native",
        "raw_data": {
            "contract": [{
                "type": "TransferContract",
                "parameter": {
                    "value": {
                        "amount": 5000000, # 5 TRX in sun
                        "owner_address": "TFromAddr",
                        "to_address": "TToAddr",
                    }
                }
            }],
            "timestamp": 1700000000000,
        },
        "ret": [{"contractRet": "SUCCESS"}],
        "fee": 1000,
    }
    tx = adapter._normalize(raw_tx, [], "tron_tx_native", network=Network.TRON_MAINNET)
    assert tx.transaction_id == "tron_tx_native"
    assert tx.chain == Chain.TRON
    assert tx.network == Network.TRON_MAINNET
    assert tx.native_value == 5000000
    assert tx.native_value_unit == "sun"
    assert tx.fee == 1000
    assert tx.fee_asset == "TRX"
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 5000000
    assert tx.transfers[0].amount_unit == "sun"
    assert tx.transfers[0].asset_symbol == "TRX"


# ---------------------------------------------------------------------------
# 6. TRON TRC20 Normalization
# ---------------------------------------------------------------------------
def test_tron_trc20_normalization():
    adapter = TronTransactionAdapter("dummy_key")
    raw_tx = {
        "txID": "tron_tx_trc20",
        "raw_data": {
            "contract": [{
                "type": "TriggerSmartContract",
                "parameter": {
                    "value": {
                        "owner_address": "TFromAddr",
                        "to_address": "TContractAddr",
                    }
                }
            }],
            "timestamp": 1700000000000,
        },
        "ret": [{"contractRet": "SUCCESS"}],
        "fee": 25000,
    }
    trc20_events = [{
        "contract_address": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
        "result": {
            "from": "TFromAddr",
            "to": "TRecipientAddr",
            "value": "10000000", # 10 USDT (6 decimals)
        },
        "token_info": {
            "symbol": "USDT",
            "decimals": 6,
        }
    }]
    tx = adapter._normalize(raw_tx, trc20_events, "tron_tx_trc20", network=Network.TRON_MAINNET)
    assert tx.native_value == 0
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDT"
    assert t.asset_contract == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
    assert t.amount == 10000000
    assert t.decimals == 6


# ---------------------------------------------------------------------------
# 7. Bitcoin Transaction Normalization (1 In, 1 Out)
# ---------------------------------------------------------------------------
def test_bitcoin_normalization():
    adapter = BitcoinTransactionAdapter()
    raw = {
        "txid": "btc_tx_1in_1out",
        "vin": [{
            "txid": "prev_tx",
            "vout": 0,
            "prevout": {"scriptpubkey_address": "bc1q_in", "value": 50000},
            "is_coinbase": False,
        }],
        "vout": [
            {"scriptpubkey_address": "bc1q_out", "value": 48000, "scriptpubkey_type": "v0_p2wpkh"}
        ],
        "status": {"confirmed": True, "block_height": 800000, "block_time": 1700000000},
    }
    tx = adapter.normalize_mempool_tx(raw, network=Network.BTC_MAINNET)
    assert tx.transaction_id == "btc_tx_1in_1out"
    assert tx.chain == Chain.BITCOIN
    assert tx.network == Network.BTC_MAINNET
    assert tx.native_value == 48000
    assert tx.native_value_unit == "sat"
    assert tx.fee == 2000 # 50000 - 48000
    assert tx.fee_asset == "BTC"
    assert len(tx.transfers) == 2
    # Input transfer: from_address set, to_address is None
    assert tx.transfers[0].from_address == "bc1q_in"
    assert tx.transfers[0].to_address is None
    assert tx.transfers[0].amount == 50000
    # Output transfer: from_address is None, to_address set
    assert tx.transfers[1].from_address is None
    assert tx.transfers[1].to_address == "bc1q_out"
    assert tx.transfers[1].amount == 48000


# ---------------------------------------------------------------------------
# 8. Bitcoin Multiple-Input / Multiple-Output (No artificial pairs)
# ---------------------------------------------------------------------------
def test_bitcoin_multi_input_multi_output():
    adapter = BitcoinTransactionAdapter()
    raw = {
        "txid": "btc_multi_io",
        "vin": [
            {"prevout": {"scriptpubkey_address": "in_1", "value": 30000}, "is_coinbase": False},
            {"prevout": {"scriptpubkey_address": "in_2", "value": 25000}, "is_coinbase": False},
            {"prevout": {"scriptpubkey_address": "in_3", "value": 15000}, "is_coinbase": False},
        ],
        "vout": [
            {"scriptpubkey_address": "out_1", "value": 40000},
            {"scriptpubkey_address": "out_2", "value": 28000},
        ],
        "status": {"confirmed": True, "block_height": 800001, "block_time": 1700000000},
    }
    tx = adapter.normalize_mempool_tx(raw, network=Network.BTC_MAINNET)
    # Exactly 3 input transfers + 2 output transfers = 5 transfers
    assert len(tx.transfers) == 5
    input_transfers = [t for t in tx.transfers if t.from_address is not None]
    output_transfers = [t for t in tx.transfers if t.to_address is not None]
    assert len(input_transfers) == 3
    assert len(output_transfers) == 2
    # No artificial sender->receiver pairs: every transfer has either from=None or to=None
    for t in tx.transfers:
        assert (t.from_address is None) or (t.to_address is None)
    # Total input: 70000, Total output: 68000, Fee: 2000
    assert tx.fee == 2000
    assert tx.native_value == 68000


# ---------------------------------------------------------------------------
# 9. Solana Native Transaction Normalization
# ---------------------------------------------------------------------------
def test_solana_native_normalization():
    adapter = HeliusSolanaAdapter("dummy_key")
    raw = {
        "signature": "sol_sig_native_1",
        "feePayer": "SenderSol",
        "fee": 5000,
        "type": "TRANSFER",
        "slot": 250000000,
        "timestamp": 1700000000,
        "transactionError": None,
        "nativeTransfers": [
            {"fromUserAccount": "SenderSol", "toUserAccount": "ReceiverSol", "amount": 2000000000} # 2 SOL
        ],
        "tokenTransfers": [],
    }
    tx = adapter.normalize_helius_tx(raw, network=Network.SOL_MAINNET)
    assert tx.transaction_id == "sol_sig_native_1"
    assert tx.chain == Chain.SOLANA
    assert tx.network == Network.SOL_MAINNET
    assert tx.native_value == 2000000000
    assert tx.native_value_unit == "lamport"
    assert tx.fee == 5000
    assert tx.fee_asset == "SOL"
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 2000000000
    assert tx.transfers[0].asset_symbol == "SOL"


# ---------------------------------------------------------------------------
# 10. Solana SPL Token Normalization
# ---------------------------------------------------------------------------
def test_solana_spl_normalization():
    adapter = HeliusSolanaAdapter("dummy_key")
    raw = {
        "signature": "sol_sig_spl_1",
        "feePayer": "SenderSol",
        "fee": 5000,
        "type": "TRANSFER",
        "slot": 250000001,
        "timestamp": 1700000000,
        "transactionError": None,
        "nativeTransfers": [],
        "tokenTransfers": [{
            "fromUserAccount": "SenderSol",
            "toUserAccount": "ReceiverSol",
            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "symbol": "USDC",
            "rawTokenAmount": {"tokenAmount": "50000000", "decimals": 6},
        }],
    }
    tx = adapter.normalize_helius_tx(raw, network=Network.SOL_MAINNET)
    assert tx.native_value == 0
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDC"
    assert t.asset_contract == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert t.amount == 50000000
    assert t.decimals == 6


# ---------------------------------------------------------------------------
# 11. Large Integer Preservation (> 9,007,199,254,740,991)
# ---------------------------------------------------------------------------
def test_large_integer_preservation():
    adapter = EVMTransactionAdapter("dummy_key")
    huge = 99999999999999999999999999999999999999999999999
    raw_tx = {
        "hash": "0xhuge",
        "from": "0xA",
        "to": "0xB",
        "value": str(huge),
        "gasUsed": "1",
        "gasPrice": str(huge),
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0xhuge", chain_id=1, network=Network.ETH_MAINNET)
    assert tx.native_value == huge
    assert tx.fee == huge
    assert tx.transfers[0].amount == huge


# ---------------------------------------------------------------------------
# 12. Zero Native Value Preservation
# ---------------------------------------------------------------------------
def test_zero_native_value():
    adapter = EVMTransactionAdapter("dummy_key")
    raw_tx = {
        "hash": "0xzero",
        "from": "0xA",
        "to": "0xB",
        "value": "0",
        "gasUsed": "21000",
        "gasPrice": "1000000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0xzero", chain_id=1, network=Network.ETH_MAINNET)
    assert tx.native_value == 0
    assert tx.transfers[0].amount == 0


# ---------------------------------------------------------------------------
# 13. Multiple Token Transfers
# ---------------------------------------------------------------------------
def test_multiple_token_transfers():
    adapter = EVMTransactionAdapter("dummy_key")
    raw_tx = {
        "hash": "0xmultitoken",
        "from": "0xA",
        "to": "0xContract",
        "value": "0",
        "gasUsed": "100000",
        "gasPrice": "20000000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    token_transfers = [
        {"from": "0xA", "to": "0xB", "tokenSymbol": "DAI", "contractAddress": "0xDai", "value": "1000", "tokenDecimal": "18"},
        {"from": "0xB", "to": "0xA", "tokenSymbol": "USDC", "contractAddress": "0xUsdc", "value": "2000", "tokenDecimal": "6"},
    ]
    tx = adapter._normalize(raw_tx, token_transfers, "0xmultitoken", chain_id=1, network=Network.ETH_MAINNET)
    assert len(tx.transfers) == 2
    assert tx.transfers[0].asset_symbol == "DAI"
    assert tx.transfers[0].amount == 1000
    assert tx.transfers[1].asset_symbol == "USDC"
    assert tx.transfers[1].amount == 2000


# ---------------------------------------------------------------------------
# 14. Optional / NULL Fields Preservation
# ---------------------------------------------------------------------------
def test_optional_null_fields():
    adapter = EVMTransactionAdapter("dummy_key")
    # Contract creation: to is None / empty, blockNumber is None, timeStamp is None
    raw_tx = {
        "hash": "0xdeploy",
        "from": "0xDeployer",
        "to": None,
        "value": "0",
        "input": "0x60806040...",
        "gasUsed": "500000",
        "gasPrice": "20000000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0xdeploy", chain_id=1, network=Network.ETH_MAINNET)
    assert tx.to_address is None
    assert tx.block_number is None
    assert tx.timestamp is None
    assert tx.transaction_type == TransactionType.CONTRACT_CREATION


# ---------------------------------------------------------------------------
# 15. Malformed Provider Response
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_malformed_provider_response():
    adapter = BitcoinTransactionAdapter()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        # Returns invalid response body
        mock_get.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"no_txid_here": True}))
        with pytest.raises(MalformedResponseError):
            await adapter.get_transaction("bad_tx")


# ---------------------------------------------------------------------------
# 16. Transaction Not Found Handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transaction_not_found_handling():
    service = IngestionService(max_retries=0)
    with patch.object(EVMTransactionAdapter, "get_transaction", side_effect=TransactionNotFoundError("etherscan-evm", "0xnonexistent")):
        with patch.object(BlockscoutEVMAdapter, "get_transaction", side_effect=TransactionNotFoundError("blockscout-evm", "0xnonexistent")):
            result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xnonexistent")
            assert result.success is False
            assert result.status == IngestionStatus.FAILED
            assert result.error_category == "TransactionNotFoundError"


# ---------------------------------------------------------------------------
# 17. Provider Timeout & Bounded Retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_timeout():
    service = IngestionService(max_retries=1, retry_delay=0.01)
    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "mock-timeout"
    mock_adapter.supported_chains = [Chain.EVM]
    mock_adapter.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(side_effect=ProviderTimeoutError("mock-timeout", "Timed out"))

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, mock_adapter)
    service.registry = reg

    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xtimeout")
    assert result.success is False
    assert result.error_category == "ProviderTimeoutError"
    # Initial attempt + 1 retry = 2 attempts
    assert mock_adapter.get_transaction.call_count == 2


# ---------------------------------------------------------------------------
# 18. Provider Rate Limit Behavior
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_rate_limit_behavior():
    service = IngestionService(max_retries=1, retry_delay=0.01)
    mock_adapter = MagicMock(spec=BaseDataProvider)
    mock_adapter.provider_name = "mock-ratelimit"
    mock_adapter.supported_chains = [Chain.TRON]
    mock_adapter.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    mock_adapter.supports = MagicMock(return_value=True)
    mock_adapter.get_transaction = AsyncMock(side_effect=ProviderRateLimitError("mock-ratelimit", "Rate limited"))

    reg = DataProviderRegistry()
    reg.register(Chain.TRON, mock_adapter)
    service.registry = reg

    result = await service.ingest_transaction(Chain.TRON, Network.TRON_MAINNET, "tron_rate")
    assert result.success is False
    assert result.error_category == "ProviderRateLimitError"
    assert mock_adapter.get_transaction.call_count == 2


# ---------------------------------------------------------------------------
# 19. Provider Authentication Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_provider_authentication_failure():
    adapter = EVMTransactionAdapter(api_key=None)
    with pytest.raises(ProviderAuthError) as exc_info:
        await adapter.get_transaction("0xauth")
    assert "No API key configured" in str(exc_info.value)
    # Ensure error does not loop or leak keys
    assert "key" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 20. Fallback from Primary to Secondary Provider
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_from_primary_to_secondary():
    primary = MagicMock(spec=BaseDataProvider)
    primary.provider_name = "primary-provider"
    primary.supported_chains = [Chain.EVM]
    primary.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    primary.supports = MagicMock(return_value=True)
    primary.get_transaction = AsyncMock(side_effect=ProviderUnavailableError("primary-provider", "503 Down"))

    fallback_tx = Transaction(
        transaction_id="0xfallback",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        native_value=1000,
        native_value_unit="wei",
        transfers=[],
        provenance=TransactionProvenance(provider="fallback-provider", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id="0xfallback"),
    )
    fallback = MagicMock(spec=BaseDataProvider)
    fallback.provider_name = "fallback-provider"
    fallback.supported_chains = [Chain.EVM]
    fallback.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    fallback.supports = MagicMock(return_value=True)
    fallback.get_transaction = AsyncMock(return_value=fallback_tx)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=0)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xfallback")

    assert result.success is True
    assert result.fallback_used is True
    assert result.provider == "fallback-provider"
    assert result.transaction.provenance.provider == "fallback-provider"


# ---------------------------------------------------------------------------
# 21. Fallback Provenance Correctness
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fallback_provenance_correctness():
    # If fallback provider succeeds, provenance must record the fallback provider
    primary = MagicMock(spec=BaseDataProvider)
    primary.provider_name = "mempool-bitcoin"
    primary.supported_chains = [Chain.BITCOIN]
    primary.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    primary.supports = MagicMock(return_value=True)
    primary.get_transaction = AsyncMock(side_effect=ProviderTimeoutError("mempool-bitcoin", "Timeout"))

    btc_tx = Transaction(
        transaction_id="btc_fb",
        chain=Chain.BITCOIN,
        network=Network.BTC_MAINNET,
        native_value=10000,
        native_value_unit="sat",
        transfers=[],
        provenance=TransactionProvenance(provider="blockstream-bitcoin", chain=Chain.BITCOIN, network=Network.BTC_MAINNET, original_id="btc_fb"),
    )
    fallback = MagicMock(spec=BaseDataProvider)
    fallback.provider_name = "blockstream-bitcoin"
    fallback.supported_chains = [Chain.BITCOIN]
    fallback.capabilities = {ProviderCapability.TRANSACTION_LOOKUP}
    fallback.supports = MagicMock(return_value=True)
    fallback.get_transaction = AsyncMock(return_value=btc_tx)

    reg = DataProviderRegistry()
    reg.register(Chain.BITCOIN, primary, primary=True)
    reg.register(Chain.BITCOIN, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=0)
    result = await service.ingest_transaction(Chain.BITCOIN, Network.BTC_MAINNET, "btc_fb")

    assert result.success is True
    assert result.provider == "blockstream-bitcoin"
    assert result.transaction.provenance.provider == "blockstream-bitcoin"


# ---------------------------------------------------------------------------
# 22. Unsupported Capability
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unsupported_capability():
    mock_provider = MagicMock(spec=BaseDataProvider)
    mock_provider.provider_name = "mock-no-history"
    mock_provider.supported_chains = [Chain.BITCOIN]
    mock_provider.capabilities = {ProviderCapability.TRANSACTION_LOOKUP} # No ADDRESS_HISTORY
    mock_provider.supports = MagicMock(side_effect=lambda cap: cap == ProviderCapability.TRANSACTION_LOOKUP)

    reg = DataProviderRegistry()
    reg.register(Chain.BITCOIN, mock_provider)

    service = IngestionService(registry=reg)
    res = await service.ingest_address_history(Chain.BITCOIN, Network.BTC_MAINNET, "bc1q_addr")
    assert res.success is False
    assert res.error_category in ("ProviderError", "UnsupportedOperationError")


# ---------------------------------------------------------------------------
# 23. Bounded Pagination
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bounded_pagination():
    mock_provider = MagicMock(spec=BaseDataProvider)
    mock_provider.provider_name = "mock-paginated"
    mock_provider.supported_chains = [Chain.EVM]
    mock_provider.capabilities = {ProviderCapability.ADDRESS_HISTORY}
    mock_provider.supports = MagicMock(return_value=True)
    mock_provider.get_address_transactions = AsyncMock(return_value=[])

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, mock_provider)

    service = IngestionService(registry=reg)
    # Pass limit = 500
    await service.ingest_address_history(Chain.EVM, Network.ETH_MAINNET, "0xAddr", limit=500)

    # Assert bounded to <= 100
    called_limit = mock_provider.get_address_transactions.call_args[1]["limit"]
    assert called_limit == 100


# ---------------------------------------------------------------------------
# 27. No Secret Leakage in Errors / Logs
# ---------------------------------------------------------------------------
def test_no_secret_leakage_in_errors():
    secret_key = "MY_SUPER_SECRET_KEY_9999"
    err = ProviderAuthError("test-provider", "Auth failed")
    assert secret_key not in str(err)
    assert secret_key not in repr(err)

    adapter = EVMTransactionAdapter(api_key=secret_key)
    rep = repr(adapter)
    assert secret_key not in rep
