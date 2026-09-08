"""Tests for chain adapters."""

import pytest
from app.providers.adapters.evm import EVMTransactionAdapter
from app.providers.adapters.tron import TronTransactionAdapter
from app.providers.adapters.bitcoin import BitcoinTransactionAdapter
from app.providers.adapters.solana import HeliusSolanaAdapter
from app.schemas.chain import Network, Chain
from app.schemas.transaction import TransactionStatus, AssetType

def test_evm_adapter_normalization():
    adapter = EVMTransactionAdapter("dummy")
    raw_tx = {
        "hash": "0x123",
        "from": "0xA",
        "to": "0xB",
        "value": "0xde0b6b3a7640000",
        "gasUsed": "21000",
        "gasPrice": "1000000000",
        "blockNumber": "0x1",
        "timeStamp": "1700000000",
        "isError": "0",
        "txreceipt_status": "1",
    }
    tx = adapter._normalize(raw_tx, [], "0x123", 1, Network.ETH_MAINNET)
    assert tx.transaction_id == "0x123"
    assert tx.native_value == 1000000000000000000
    assert tx.fee == 21000000000000
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 1000000000000000000

def test_tron_adapter_normalization():
    adapter = TronTransactionAdapter("dummy")
    raw_tx = {
        "txID": "tx123",
        "raw_data": {
            "contract": [{"type": "TransferContract", "parameter": {"value": {"amount": 1000000, "owner_address": "T1", "to_address": "T2"}}}],
            "timestamp": 1700000000000
        },
        "ret": [{"contractRet": "SUCCESS"}],
        "fee": 100
    }
    tx = adapter._normalize(raw_tx, [], "tx123")
    assert tx.transaction_id == "tx123"
    assert tx.native_value == 1000000
    assert tx.fee == 100
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 1

def test_bitcoin_adapter_normalization():
    adapter = BitcoinTransactionAdapter()
    raw = {
        "txid": "btc123",
        "vin": [{"prevout": {"scriptpubkey_address": "1A", "value": 50000}, "is_coinbase": False}],
        "vout": [{"scriptpubkey_address": "1B", "value": 49000}],
        "status": {"confirmed": True, "block_height": 100}
    }
    tx = adapter.normalize_mempool_tx(raw)
    assert tx.transaction_id == "btc123"
    assert tx.native_value == 49000
    assert tx.fee == 1000
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 2
    # Input transfer
    assert tx.transfers[0].from_address == "1A"
    assert tx.transfers[0].to_address is None
    # Output transfer
    assert tx.transfers[1].from_address is None
    assert tx.transfers[1].to_address == "1B"

def test_solana_adapter_normalization():
    adapter = HeliusSolanaAdapter("dummy")
    raw = {
        "signature": "sig123",
        "feePayer": "S1",
        "fee": 5000,
        "type": "TRANSFER",
        "nativeTransfers": [{"fromUserAccount": "S1", "toUserAccount": "S2", "amount": 1000000000}],
        "tokenTransfers": []
    }
    tx = adapter.normalize_helius_tx(raw)
    assert tx.transaction_id == "sig123"
    assert tx.native_value == 1000000000
    assert tx.fee == 5000
    assert tx.status == TransactionStatus.SUCCESS
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 1000000000

def test_evm_adapter_erc20_token_transfers():
    adapter = EVMTransactionAdapter("dummy")
    raw_tx = {
        "hash": "0xabc789",
        "from": "0xSender123",
        "to": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "value": "0x0",
        "gasUsed": "45000",
        "gasPrice": "20000000000",
        "blockNumber": "19000000",
        "timeStamp": "1700000000",
        "isError": "0",
        "txreceipt_status": "1",
        "input": "0xa9059cbb",
    }
    token_transfers_raw = [
        {
            "from": "0xSender123",
            "to": "0xReceiver456",
            "contractAddress": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "tokenSymbol": "USDT",
            "value": "1500000000",
            "tokenDecimal": "6",
        }
    ]
    tx = adapter._normalize(raw_tx, token_transfers_raw, "0xabc789", 1, Network.ETH_MAINNET)
    assert tx.transaction_id == "0xabc789"
    assert tx.native_value == 0
    assert tx.native_value_unit == "wei"
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDT"
    assert t.asset_contract == "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    assert t.from_address == "0xSender123"
    assert t.to_address == "0xReceiver456"
    assert isinstance(t.amount, int)
    assert t.amount == 1500000000
    assert t.amount_unit == "base_unit"
    assert t.decimals == 6

def test_tron_adapter_trc20_token_transfers():
    adapter = TronTransactionAdapter("dummy")
    raw_tx = {
        "txID": "tron_tx_trc20_123",
        "raw_data": {
            "contract": [
                {
                    "type": "TriggerSmartContract",
                    "parameter": {
                        "value": {
                            "owner_address": "TFromAddress123",
                            "contract_address": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
                            "data": "a9059cbb",
                        }
                    },
                }
            ],
            "timestamp": 1700000000000,
        },
        "ret": [{"contractRet": "SUCCESS"}],
        "fee": 150000,
    }
    trc20_transfers = [
        {
            "contract_address": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
            "token_info": {
                "symbol": "USDT",
                "decimals": 6,
            },
            "result": {
                "from": "TFromAddress123",
                "to": "TToAddress456",
                "value": "2500000000",
            },
        }
    ]
    tx = adapter._normalize(raw_tx, trc20_transfers, "tron_tx_trc20_123")
    assert tx.transaction_id == "tron_tx_trc20_123"
    assert tx.native_value == 0
    assert tx.fee == 150000
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDT"
    assert t.asset_contract == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
    assert t.from_address == "TFromAddress123"
    assert t.to_address == "TToAddress456"
    assert isinstance(t.amount, int)
    assert t.amount == 2500000000
    assert t.amount_unit == "base_unit"
    assert t.decimals == 6

def test_solana_adapter_spl_token_transfers():
    from app.providers.data_base import NormalizationError

    adapter = HeliusSolanaAdapter("dummy")
    large_amount_str = "9007199254740993123456"
    large_amount_int = 9007199254740993123456
    raw = {
        "signature": "helius_spl_sig_123",
        "feePayer": "FeePayerSol1",
        "fee": 5000,
        "type": "TRANSFER",
        "nativeTransfers": [],
        "tokenTransfers": [
            {
                "fromUserAccount": "SolSenderTokenAccount",
                "toUserAccount": "SolReceiverTokenAccount",
                "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "symbol": "USDC",
                "decimals": 6,
                "tokenAmount": 9007199254740993.123456,
                "rawTokenAmount": {
                    "tokenAmount": large_amount_str,
                    "decimals": 6,
                },
            }
        ],
    }
    tx = adapter.normalize_helius_tx(raw)
    assert tx.transaction_id == "helius_spl_sig_123"
    assert tx.native_value == 0
    assert len(tx.transfers) == 1
    t = tx.transfers[0]
    assert t.asset_type == AssetType.TOKEN
    assert t.asset_symbol == "USDC"
    assert t.asset_contract == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    assert t.from_address == "SolSenderTokenAccount"
    assert t.to_address == "SolReceiverTokenAccount"
    assert isinstance(t.amount, int)
    assert t.amount == large_amount_int
    assert str(t.amount) == large_amount_str
    assert t.amount_unit == "base_unit"
    assert t.decimals == 6

def test_solana_adapter_spl_token_exact_decimal_fallback():
    adapter = HeliusSolanaAdapter("dummy")
    raw = {
        "signature": "sig_exact_fallback",
        "feePayer": "FeePayer",
        "fee": 5000,
        "type": "TRANSFER",
        "nativeTransfers": [],
        "tokenTransfers": [
            {
                "fromUserAccount": "UserA",
                "toUserAccount": "UserB",
                "mint": "TokenMint123",
                "symbol": "TEST",
                "decimals": 6,
                "tokenAmount": "12.345678",
            }
        ],
    }
    tx = adapter.normalize_helius_tx(raw)
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 12345678
    assert tx.transfers[0].decimals == 6

def test_solana_adapter_spl_token_missing_decimals_fails():
    from app.providers.data_base import NormalizationError

    adapter = HeliusSolanaAdapter("dummy")
    raw = {
        "signature": "sig_missing_decimals",
        "feePayer": "FeePayer",
        "fee": 5000,
        "type": "TRANSFER",
        "nativeTransfers": [],
        "tokenTransfers": [
            {
                "fromUserAccount": "UserA",
                "toUserAccount": "UserB",
                "mint": "TokenMint123",
                "symbol": "TEST",
                "tokenAmount": "12.345678",
            }
        ],
    }
    with pytest.raises(NormalizationError, match="missing both rawTokenAmount and decimals"):
        adapter.normalize_helius_tx(raw)

