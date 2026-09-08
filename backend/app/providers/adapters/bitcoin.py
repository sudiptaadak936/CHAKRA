"""Bitcoin transaction adapter.

Supports Bitcoin mainnet via mempool.space API (primary) and
Blockstream Esplora (fallback).

IMPORTANT: Bitcoin uses the UTXO (Unspent Transaction Output) model.
Transactions do NOT have a single sender and single receiver.
We do not infer artificial 1-to-1 sender->receiver relationships.

This adapter preserves UTXO semantics in two ways:
1. BitcoinTransactionDetail: carries the full UTXO structure (inputs, outputs)
2. Transaction.transfers:
   - Inputs are represented as transfers with to_address=None
   - Outputs are represented as transfers with from_address=None
   This prevents downstream systems from counting the same movement twice.

Value unit: satoshis (1 BTC = 100,000,000 satoshis)
Fee = total_input_value - total_output_value (not explicitly in UTXO data)
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Set

import httpx

from app.providers.data_base import (
    BaseDataProvider,
    MalformedResponseError,
    NormalizationError,
    ProviderCapability,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TransactionNotFoundError,
)
from app.schemas.chain import Chain, Network
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

logger = logging.getLogger(__name__)

_DEFAULT_MEMPOOL_BASE = "https://mempool.space/api"
_DEFAULT_ESPLORA_BASE = "https://blockstream.info/api"
_TIMEOUT = 15.0


class BitcoinTransactionAdapter(BaseDataProvider):
    """Data provider adapter for Bitcoin using mempool.space REST API."""

    _PROVIDER_NAME = "mempool-bitcoin"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url = (base_url or _DEFAULT_MEMPOOL_BASE).rstrip("/")

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.BITCOIN]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.ADDRESS_HISTORY,
            ProviderCapability.NATIVE_TRANSFERS,
        }

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        """Fetch and normalize a Bitcoin transaction by txid from mempool.space."""
        network = kwargs.get("network", Network.BTC_MAINNET)
        url = f"{self._base_url}/tx/{tx_id}"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, f"Request timed out for tx {tx_id}")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        if resp.status_code == 404:
            raise TransactionNotFoundError(self.provider_name, tx_id)
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Server error {resp.status_code}")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            raw = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response: {exc}") from exc

        if not isinstance(raw, dict) or not raw.get("txid"):
            raise MalformedResponseError(self.provider_name, f"Malformed transaction response for {tx_id}")

        return self.normalize_mempool_tx(raw, network=network)

    async def get_address_transactions(
        self, address: str, limit: int = 25, offset: int = 0, **kwargs
    ) -> List[Transaction]:
        """Fetch and normalize recent transactions for a Bitcoin address."""
        network = kwargs.get("network", Network.BTC_MAINNET)
        url = f"{self._base_url}/address/{address}/txs"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, f"Request timed out for address {address}")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Server error {resp.status_code}")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            raw_list = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response: {exc}") from exc

        if not isinstance(raw_list, list):
            return []

        results: List[Transaction] = []
        for raw in raw_list[offset:offset + limit]:
            try:
                results.append(self.normalize_mempool_tx(raw, network=network))
            except Exception as exc:
                logger.warning("[%s] Skipping malformed tx: %s", self.provider_name, exc)

        return results

    def extract_bitcoin_detail(self, raw: Dict[str, Any]) -> BitcoinTransactionDetail:
        """Extract dedicated Bitcoin UTXO details (inputs, outputs, fees) from raw response."""
        txid = raw["txid"]
        vin_raw = raw.get("vin", [])
        vout_raw = raw.get("vout", [])

        is_coinbase = bool(vin_raw) and vin_raw[0].get("is_coinbase", False)
        inputs: List[BitcoinVin] = []
        total_input_sat: Optional[int] = 0 if not is_coinbase else None
        for vin in vin_raw:
            prevout = vin.get("prevout", {})
            value_sat = prevout.get("value") if prevout else None
            if value_sat is not None and total_input_sat is not None:
                total_input_sat += int(value_sat)
            inputs.append(BitcoinVin(
                txid=vin.get("txid"),
                vout=vin.get("vout"),
                address=prevout.get("scriptpubkey_address") if prevout else None,
                value_sat=int(value_sat) if value_sat is not None else None,
                coinbase=vin.get("is_coinbase", False),
            ))

        outputs: List[BitcoinVout] = []
        total_output_sat = 0
        for i, vout in enumerate(vout_raw):
            val_sat = int(vout.get("value", 0))
            total_output_sat += val_sat
            outputs.append(BitcoinVout(
                n=vout.get("n", i),
                address=vout.get("scriptpubkey_address"),
                value_sat=val_sat,
                script_type=vout.get("scriptpubkey_type"),
            ))

        fee_sat: Optional[int] = (
            total_input_sat - total_output_sat
            if total_input_sat is not None else None
        )
        if fee_sat is not None and fee_sat < 0:
            fee_sat = None

        return BitcoinTransactionDetail(
            txid=txid,
            inputs=inputs,
            outputs=outputs,
            total_input_sat=total_input_sat,
            total_output_sat=total_output_sat,
            fee_sat=fee_sat,
            locktime=raw.get("locktime"),
            version=raw.get("version"),
        )

    def normalize_mempool_tx(
        self, raw: Dict[str, Any], network: Network = Network.BTC_MAINNET
    ) -> Transaction:
        """Normalize mempool.space / Esplora transaction dict into canonical Transaction."""
        try:
            txid = raw["txid"]
            vin_raw = raw.get("vin", [])
            vout_raw = raw.get("vout", [])
            status_raw = raw.get("status", {})

            is_coinbase = bool(vin_raw) and vin_raw[0].get("is_coinbase", False)
            tx_type = TransactionType.COINBASE if is_coinbase else TransactionType.TRANSFER

            inputs: List[BitcoinVin] = []
            total_input_sat: Optional[int] = 0 if not is_coinbase else None
            for vin in vin_raw:
                prevout = vin.get("prevout", {})
                value_sat = prevout.get("value") if prevout else None
                if value_sat is not None and total_input_sat is not None:
                    total_input_sat += int(value_sat)
                inputs.append(BitcoinVin(
                    txid=vin.get("txid"),
                    vout=vin.get("vout"),
                    address=prevout.get("scriptpubkey_address") if prevout else None,
                    value_sat=int(value_sat) if value_sat is not None else None,
                    coinbase=vin.get("is_coinbase", False),
                ))

            outputs: List[BitcoinVout] = []
            total_output_sat = 0
            for i, vout in enumerate(vout_raw):
                val_sat = int(vout.get("value", 0))
                total_output_sat += val_sat
                outputs.append(BitcoinVout(
                    n=vout.get("n", i),
                    address=vout.get("scriptpubkey_address"),
                    value_sat=val_sat,
                    script_type=vout.get("scriptpubkey_type"),
                ))

            fee_sat: Optional[int] = (
                total_input_sat - total_output_sat
                if total_input_sat is not None else None
            )
            if fee_sat is not None and fee_sat < 0:
                fee_sat = None

            block_number: Optional[int] = status_raw.get("block_height")
            block_hash: Optional[str] = status_raw.get("block_hash")
            confirmed = status_raw.get("confirmed", False)

            block_time_raw = status_raw.get("block_time")
            try:
                ts = dt.datetime.fromtimestamp(int(block_time_raw), tz=dt.timezone.utc) if block_time_raw else None
            except (ValueError, TypeError, OSError):
                ts = None

            status = TransactionStatus.SUCCESS if confirmed else TransactionStatus.PENDING

            transfers: List[Transfer] = []

            # Input transfers: from_address=input address, to_address=None
            if not is_coinbase:
                for vin_obj in inputs:
                    if vin_obj.value_sat is not None and vin_obj.value_sat > 0:
                        transfers.append(Transfer(
                            from_address=vin_obj.address,
                            to_address=None,
                            asset_type=AssetType.NATIVE,
                            asset_symbol="BTC",
                            amount=vin_obj.value_sat,
                            amount_unit="sat",
                            decimals=8,
                            chain=Chain.BITCOIN,
                            network=network,
                        ))

            # Output transfers: from_address=None, to_address=output address
            for vout_obj in outputs:
                if vout_obj.value_sat > 0:
                    transfers.append(Transfer(
                        from_address=None,
                        to_address=vout_obj.address,
                        asset_type=AssetType.NATIVE,
                        asset_symbol="BTC",
                        amount=vout_obj.value_sat,
                        amount_unit="sat",
                        decimals=8,
                        chain=Chain.BITCOIN,
                        network=network,
                    ))

            provenance = TransactionProvenance(
                provider=self.provider_name,
                chain=Chain.BITCOIN,
                network=network,
                original_id=txid,
            )

            return Transaction(
                transaction_id=txid,
                chain=Chain.BITCOIN,
                network=network,
                chain_id=None,
                block_number=block_number,
                block_hash=block_hash,
                timestamp=ts,
                from_address=None,
                to_address=None,
                native_value=total_output_sat,
                native_value_unit="sat",
                transaction_type=tx_type,
                status=status,
                fee=fee_sat,
                fee_asset="BTC",
                transfers=transfers,
                provenance=provenance,
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise NormalizationError(
                self.provider_name,
                f"Failed to normalize Bitcoin transaction: {type(exc).__name__}",
            ) from exc


class BlockstreamBitcoinAdapter(BitcoinTransactionAdapter):
    """Data provider fallback adapter for Bitcoin using Blockstream Esplora API."""

    _PROVIDER_NAME = "blockstream-bitcoin"

    def __init__(self, base_url: Optional[str] = None) -> None:
        super().__init__(base_url=base_url or _DEFAULT_ESPLORA_BASE)

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME
