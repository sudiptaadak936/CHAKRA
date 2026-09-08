"""Tron transaction adapter.

Supports Tron mainnet via TronGrid (primary provider).

Normalizes:
- Native TRX transfers (value in sun; 1 TRX = 1,000,000 sun)
- TRC-20 token transfers

Tron address format: base58check (starts with T), 34 characters.
Addresses are preserved exactly as returned by TronGrid.

API key is passed at construction and stored privately.
It must never appear in error messages, logs, or repr output.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import httpx

from app.providers.data_base import (
    BaseDataProvider,
    MalformedResponseError,
    NormalizationError,
    ProviderAuthError,
    ProviderCapability,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TransactionNotFoundError,
    UnsupportedOperationError,
)
from app.schemas.chain import Chain, Network
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)

logger = logging.getLogger(__name__)

_TRONGRID_BASE = "https://api.trongrid.io"
_TIMEOUT = 15.0


class TronTransactionAdapter(BaseDataProvider):
    """Blockchain data provider for Tron via TronGrid.

    Supports TRX native transfers and TRC-20 token transfers.
    Does not support block lookup (deferred to future step).
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key   # private; never expose

    @property
    def provider_name(self) -> str:
        return "trongrid"

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.TRON]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.ADDRESS_HISTORY,
            ProviderCapability.TOKEN_TRANSFERS,
            ProviderCapability.NATIVE_TRANSFERS,
        }

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        """Retrieve and normalize a Tron transaction by hash.

        Args:
            tx_id: Tron transaction hash (hex string).

        Returns:
            Normalized canonical Transaction.

        Raises:
            ProviderAuthError:        API key rejected.
            ProviderRateLimitError:   Rate limited.
            ProviderTimeoutError:     Request timed out.
            ProviderUnavailableError: TronGrid server error.
            TransactionNotFoundError: Transaction not found.
            MalformedResponseError:   Unexpected response shape.
            NormalizationError:       Could not build Transaction from response.
        """
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        network = kwargs.get("network", Network.TRON_MAINNET)
        raw_tx = await self._fetch_transaction(tx_id)
        token_transfers = await self._fetch_trc20_transfers(tx_id)

        try:
            return self._normalize(raw_tx, token_transfers, tx_id, network=network)
        except (KeyError, ValueError, TypeError) as exc:
            raise NormalizationError(
                self.provider_name,
                f"Failed to normalize Tron transaction {tx_id}: {type(exc).__name__}",
            ) from exc

    async def get_address_transactions(
        self, address: str, limit: int = 25, offset: int = 0, **kwargs
    ) -> List[Transaction]:
        """List recent TRX transactions for a Tron address."""
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        network = kwargs.get("network", Network.TRON_MAINNET)
        url = f"{_TRONGRID_BASE}/v1/accounts/{address}/transactions"
        params = {"limit": min(limit, 200), "min_timestamp": 0}
        raw = await self._trongrid_get(url, params)
        txs_raw = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []

        results: List[Transaction] = []
        for raw_tx in txs_raw[:limit]:
            try:
                tx = self._normalize(raw_tx, [], raw_tx.get("txID", ""), network=network)
                results.append(tx)
            except Exception as exc:
                logger.debug("[%s] Skipping tx: %s", self.provider_name, type(exc).__name__)
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _fetch_transaction(self, tx_id: str) -> Dict[str, Any]:
        url = f"{_TRONGRID_BASE}/wallet/gettransactionbyid"
        result = await self._trongrid_post(url, {"value": tx_id})
        if not result or not isinstance(result, dict):
            raise TransactionNotFoundError(self.provider_name, tx_id)
        # TronGrid returns empty dict {} for not-found
        if not result.get("txID") and not result.get("id"):
            raise TransactionNotFoundError(self.provider_name, tx_id)
        return result

    async def _fetch_trc20_transfers(self, tx_id: str) -> List[Dict[str, Any]]:
        """Fetch TRC-20 token transfer events for a transaction."""
        url = f"{_TRONGRID_BASE}/v1/transactions/{tx_id}/events"
        try:
            result = await self._trongrid_get(url, {})
            if isinstance(result, dict):
                return result.get("data", [])
            if isinstance(result, list):
                return result
        except (TransactionNotFoundError, MalformedResponseError):
            pass
        return []

    async def _trongrid_post(self, url: str, body: Dict[str, Any]) -> Any:
        """POST to TronGrid with auth header (key never in errors)."""
        headers = {"TRON-PRO-API-KEY": self._api_key}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, "TronGrid request timed out.")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        return self._handle_response(resp)

    async def _trongrid_get(self, url: str, params: Dict[str, Any]) -> Any:
        """GET from TronGrid with auth header (key never in errors)."""
        headers = {"TRON-PRO-API-KEY": self._api_key}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, headers=headers, params=params)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, "TronGrid request timed out.")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        return self._handle_response(resp)

    def _handle_response(self, resp: httpx.Response) -> Any:
        if resp.status_code in (401, 403):
            raise ProviderAuthError(self.provider_name, "TronGrid authentication failed.")
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "TronGrid rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"TronGrid server error {resp.status_code}.")
        try:
            return resp.json()
        except Exception:
            raise MalformedResponseError(self.provider_name, "Non-JSON response from TronGrid.")

    def _normalize(
        self,
        raw_tx: Dict[str, Any],
        trc20_transfers: List[Dict[str, Any]],
        tx_id: str,
        network: Network = Network.TRON_MAINNET,
    ) -> Transaction:
        """Build a canonical Transaction from TronGrid response."""
        # TronGrid transaction structure
        raw_data = raw_tx.get("raw_data", {})
        contract_list = raw_data.get("contract", [])
        contract = contract_list[0] if contract_list else {}
        contract_type = contract.get("type", "")
        param = contract.get("parameter", {}).get("value", {})

        # Sender/receiver
        from_addr = param.get("owner_address")
        to_addr = param.get("to_address")

        # Amount in sun (TRX atomic unit)
        amount_sun = int(param.get("amount", 0))

        # Timestamp (ms -> datetime)
        ts_ms = raw_data.get("timestamp")
        try:
            ts: Optional[datetime] = (
                datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc) if ts_ms else None
            )
        except (ValueError, TypeError, OSError):
            ts = None

        # Status
        ret_list = raw_tx.get("ret", [{}])
        contract_ret = ret_list[0].get("contractRet", "UNKNOWN") if ret_list else "UNKNOWN"
        if contract_ret == "SUCCESS":
            status = TransactionStatus.SUCCESS
        elif contract_ret in ("REVERT", "FAILED", "OUT_OF_ENERGY", "OUT_OF_TIME"):
            status = TransactionStatus.FAILED
        else:
            status = TransactionStatus.UNKNOWN

        # Transaction type
        type_map = {
            "TransferContract": TransactionType.TRANSFER,
            "TriggerSmartContract": TransactionType.CONTRACT_CALL,
            "CreateSmartContract": TransactionType.CONTRACT_CREATION,
        }
        tx_type = type_map.get(contract_type, TransactionType.UNKNOWN)

        # Fee in sun
        fee_sun: Optional[int] = raw_tx.get("fee") or raw_tx.get("net_fee")
        if fee_sun is not None:
            try:
                fee_sun = int(fee_sun)
            except (ValueError, TypeError):
                fee_sun = None

        # Block
        block_number: Optional[int] = raw_tx.get("blockNumber") or raw_tx.get("block_number")
        if block_number is not None:
            try:
                block_number = int(block_number)
            except (ValueError, TypeError):
                block_number = None

        # Build transfers
        transfers: List[Transfer] = []

        if amount_sun > 0 or not trc20_transfers:
            transfers.append(
                Transfer(
                    from_address=from_addr,
                    to_address=to_addr,
                    
                    asset_type=AssetType.NATIVE,
                    asset_symbol="TRX",
                    amount=amount_sun,
                    amount_unit="sun",
                    decimals=6,
                    chain=Chain.TRON,
                    network=network,
                )
            )

        # TRC-20 transfers from events
        for event in trc20_transfers:
            result_data = event.get("result", {})
            try:
                transfers.append(
                    Transfer(
                        from_address=result_data.get("from"),
                        to_address=result_data.get("to"),
                        asset_type=AssetType.TOKEN,
                        asset_symbol=event.get("token_info", {}).get("symbol", "UNKNOWN"),
                        asset_contract=event.get("contract_address"),
                        amount=int(result_data.get("value", 0)),
                        amount_unit="base_unit",
                        decimals=event.get("token_info", {}).get("decimals"),
                        chain=Chain.TRON,
                        network=network,
                    )
                )
            except Exception:
                logger.debug("[%s] Skipping malformed TRC-20 event.", self.provider_name)

        provenance = TransactionProvenance(
            provider=self.provider_name,
            chain=Chain.TRON,
            network=network,
            original_id=tx_id or raw_tx.get("txID", ""),
        )

        return Transaction(
            transaction_id=tx_id or raw_tx.get("txID", ""),
            chain=Chain.TRON,
            network=network,
            chain_id=None,
            block_number=block_number,
            block_hash=None,
            timestamp=ts,
            from_address=from_addr,
            to_address=to_addr,
            native_value=amount_sun,
            native_value_unit="sun",
            
            transaction_type=tx_type,
            status=status,
            fee=fee_sun,
            fee_asset="TRX",
            transfers=transfers,
            provenance=provenance,
        )


