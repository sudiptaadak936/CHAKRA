"""Unit tests for canonical Transaction and Transfer schemas (Step 1A)."""
import pytest
from datetime import datetime, timezone

from app.schemas.chain import Chain, Network, ChainInfo, EVM_CHAIN_ID_TO_NETWORK
from app.schemas.transaction import (
    AssetType,
    BitcoinTransactionDetail,
    BitcoinVin,
    BitcoinVout,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)

def make_provenance(provider="etherscan-evm", chain=Chain.EVM, network=Network.ETH_MAINNET,
                    original_id="0xabc123") -> TransactionProvenance:
    return TransactionProvenance(
        provider=provider,
        chain=chain,
        network=network,
        original_id=original_id,
    )

def make_eth_transfer(
    from_addr="0xFrom", to_addr="0xTo", amount=1_000_000_000_000_000_000
) -> Transfer:
    return Transfer(
        from_address=from_addr,
        to_address=to_addr,
        asset_type=AssetType.NATIVE,
        asset_symbol="ETH",
        amount=amount,
        amount_unit="wei",
        decimals=18,
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
    )

def make_erc20_transfer(
    from_addr="0xFrom", to_addr="0xTo", symbol="USDT",
    contract="0xDac17F958D2ee523a2206206994597C13D831ec7",
    amount=1_000_000,
) -> Transfer:
    return Transfer(
        from_address=from_addr,
        to_address=to_addr,
        asset_type=AssetType.TOKEN,
        asset_symbol=symbol,
        asset_contract=contract,
        amount=amount,
        amount_unit="base_unit",
        decimals=6,
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
    )

def make_transaction(
    tx_id="0xabc123",
    chain=Chain.EVM,
    network=Network.ETH_MAINNET,
    chain_id=1,
    native_value=1_000_000_000_000_000_000,
    transfers=None,
    provenance=None,
) -> Transaction:
    if provenance is None:
        provenance = make_provenance()
    return Transaction(
        transaction_id=tx_id,
        chain=chain,
        network=network,
        chain_id=chain_id,
        block_number=18_000_000,
        block_hash="0xblock",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        from_address="0xFrom",
        to_address="0xTo",
        native_value=native_value,
        native_value_unit="wei",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21_000 * 20_000_000_000,
        fee_asset="ETH",
        transfers=transfers or [make_eth_transfer()],
        provenance=provenance,
    )

class TestChainInfo:
    def test_evm_chain_info(self):
        info = ChainInfo(chain=Chain.EVM, network=Network.ETH_MAINNET, chain_id=1)
        assert info.chain == Chain.EVM
        assert info.network == Network.ETH_MAINNET
        assert info.chain_id == 1

    def test_non_evm_chain_id_is_none(self):
        info = ChainInfo(chain=Chain.BITCOIN, network=Network.BTC_MAINNET)
        assert info.chain_id is None

    def test_solana_chain_info(self):
        info = ChainInfo(chain=Chain.SOLANA, network=Network.SOL_MAINNET)
        assert info.chain == Chain.SOLANA
        assert info.chain_id is None

class TestTransfer:
    def test_native_eth_transfer(self):
        t = make_eth_transfer()
        assert t.asset_type == AssetType.NATIVE
        assert t.asset_symbol == "ETH"
        assert t.amount_unit == "wei"
        assert t.asset_contract is None
        assert t.decimals == 18

    def test_erc20_token_transfer(self):
        t = make_erc20_transfer()
        assert t.asset_type == AssetType.TOKEN
        assert t.asset_symbol == "USDT"

    def test_amount_is_integer(self):
        t = make_eth_transfer(amount=1_000_000_000_000_000_000)
        assert isinstance(t.amount, int)
        assert t.amount == 1_000_000_000_000_000_000

    def test_zero_amount_is_valid(self):
        t = make_eth_transfer(amount=0)
        assert t.amount == 0

    def test_from_address_can_be_none(self):
        t = Transfer(
            from_address=None,
            to_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf5",
            asset_type=AssetType.NATIVE,
            asset_symbol="BTC",
            amount=5_000_000_000,
            amount_unit="sat",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
        )
        assert t.from_address is None

    def test_to_address_can_be_none(self):
        t = Transfer(
            from_address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf5",
            to_address=None,
            asset_type=AssetType.NATIVE,
            asset_symbol="BTC",
            amount=0,
            amount_unit="sat",
            chain=Chain.BITCOIN,
            network=Network.BTC_MAINNET,
        )
        assert t.to_address is None

class TestTransactionProvenance:
    def test_provenance_fields(self):
        p = make_provenance()
        assert p.provider == "etherscan-evm"
        assert p.chain == Chain.EVM

class TestTransaction:
    def test_basic_eth_transaction(self):
        tx = make_transaction()
        assert tx.transaction_id == "0xabc123"
        assert tx.chain == Chain.EVM
        assert tx.native_value == 1_000_000_000_000_000_000
        assert tx.native_value_unit == "wei"
        assert tx.status == TransactionStatus.SUCCESS

    def test_transaction_with_multiple_transfers(self):
        native = make_eth_transfer(amount=500_000_000_000_000_000)
        token1 = make_erc20_transfer(symbol="USDT", amount=1_000_000)
        tx = make_transaction(transfers=[native, token1])
        assert len(tx.transfers) == 2

    def test_pure_token_transaction_value_is_zero(self):
        token_tx = Transaction(
            transaction_id="0xtoken",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            chain_id=1,
            block_number=18_000_001,
            from_address="0xFrom",
            to_address="0xContract",
            native_value=0,
            native_value_unit="wei",
            transaction_type=TransactionType.CONTRACT_CALL,
            status=TransactionStatus.SUCCESS,
            transfers=[make_erc20_transfer()],
            provenance=make_provenance(),
        )
        assert token_tx.native_value == 0
        assert len(token_tx.transfers) == 1

class TestBitcoinUTXO:
    def test_bitcoin_transaction_detail(self):
        vin = BitcoinVin(txid="prev", vout=0, address="1From", value_sat=100_000, coinbase=False)
        vout1 = BitcoinVout(n=0, address="1To", value_sat=90_000, script_type="p2pkh")
        detail = BitcoinTransactionDetail(
            txid="txidabc",
            inputs=[vin],
            outputs=[vout1],
            total_input_sat=100_000,
            total_output_sat=90_000,
            fee_sat=10_000,
        )
        assert detail.fee_sat == 10_000
