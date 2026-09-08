"""Solana transaction adapter.

Supports Solana mainnet via Helius (primary) and public Solana RPC (fallback stub).

Normalizes:
- Native SOL transfers (value in lamports; 1 SOL = 1,000,000,000 lamports)
- SPL token transfers

Solana identifiers:
- Transaction identifier: signature (base58 string)
- Slot: Solana block equivalent (uint64)
- Accounts: base58-encoded 32-byte public keys

Address format: base58check, 32-44 characters. Preserved exactly.

Current status:
- HeliusSolanaAdapter: normalization logic implemented and testable with mock data.
  Live HTTP calls are stubs (raise UnsupportedOperationError) until ingestion step.
- PublicSolanaRPCAdapter: full stub.

API key stored privately; never exposed in errors or logs.
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


class HeliusSolanaAdapter(BaseDataProvider):
    """Blockchain data provider for Solana via Helius.

    Helius provides enhanced transaction parsing via its REST API
    (/v0/transactions) in addition to standard Solana JSON-RPC.

    Current status: normalization logic testable with mock data.
    Live HTTP ingestion is a stub (raises UnsupportedOperationError)
    until the full ingestion step is implemented.
    """

    _PROVIDER_NAME = "helius"
    _HELIUS_BASE = "https://api.helius.xyz"

    def __init__(self, api_key: Optional[str]) -> None:
        self._api_key = api_key   # private; never expose

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.SOLANA]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.ADDRESS_HISTORY,
            ProviderCapability.TOKEN_TRANSFERS,
            ProviderCapability.NATIVE_TRANSFERS,
        }

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        """Retrieve and normalize a Solana transaction by signature from Helius."""
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        network = kwargs.get("network", Network.SOL_MAINNET)
        url = f"{self._HELIUS_BASE}/v0/transactions/?api-key={self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json={"transactions": [tx_id]})
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, "Helius request timed out.")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self.provider_name, "Helius authentication failed.")
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Helius rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Helius server error {resp.status_code}.")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response from Helius: {exc}") from exc

        if not isinstance(data, list) or len(data) == 0:
            raise TransactionNotFoundError(self.provider_name, tx_id)

        return self.normalize_helius_tx(data[0], network=network)

    async def get_address_transactions(
        self, address: str, limit: int = 25, offset: int = 0, **kwargs
    ) -> List[Transaction]:
        """List transactions for a Solana address from Helius."""
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        network = kwargs.get("network", Network.SOL_MAINNET)
        url = f"{self._HELIUS_BASE}/v0/addresses/{address}/transactions"
        params = {"api-key": self._api_key, "limit": min(limit, 100)}
        if "before" in kwargs:
            params["before"] = kwargs["before"]

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, "Helius request timed out.")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Network error: {type(exc).__name__}")

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self.provider_name, "Helius authentication failed.")
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Helius rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Helius server error {resp.status_code}.")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response from Helius: {exc}") from exc

        if not isinstance(data, list):
            return []

        results: List[Transaction] = []
        for raw in data[:limit]:
            try:
                results.append(self.normalize_helius_tx(raw, network=network))
            except Exception as exc:
                logger.warning("[%s] Skipping malformed tx: %s", self.provider_name, exc)

        return results

    # -----------------------------------------------------------------------
    # Normalization logic (testable with mock data)
    # -----------------------------------------------------------------------

    def _parse_spl_token_amount_and_decimals(
        self, tt: Dict[str, Any]
    ) -> tuple[int, Optional[int]]:
        """Extract exact integer atomic amount and decimals for an SPL token transfer.

        Guarantees:
        - Zero floating-point arithmetic.
        - Uses rawTokenAmount.tokenAmount as exact integer when present.
        - If rawTokenAmount is missing but decimal-form tokenAmount is provided,
          converts using exact fixed-point string/integer arithmetic using the token's actual decimals.
        - If neither raw atomic amount nor decimals are available, raises NormalizationError.
        """
        raw_amount = tt.get("rawTokenAmount")

        # 1. Prefer rawTokenAmount.tokenAmount if available
        if isinstance(raw_amount, dict) and raw_amount.get("tokenAmount") is not None:
            raw_val = str(raw_amount["tokenAmount"]).strip()
            decimals_raw = raw_amount.get("decimals")
            if decimals_raw is None:
                decimals_raw = tt.get("decimals")
            decimals = int(decimals_raw) if decimals_raw is not None else None
            try:
                return int(raw_val), decimals
            except (ValueError, TypeError) as exc:
                raise NormalizationError(
                    self.provider_name,
                    f"Malformed rawTokenAmount.tokenAmount: {raw_val!r}",
                ) from exc

        # 2. Raw atomic amount is missing: we require decimals to do exact conversion
        decimals_raw = tt.get("decimals")
        if decimals_raw is None and isinstance(raw_amount, dict):
            decimals_raw = raw_amount.get("decimals")

        if decimals_raw is None:
            raise NormalizationError(
                self.provider_name,
                f"SPL token transfer for mint {tt.get('mint')!r} missing both rawTokenAmount and decimals.",
            )

        try:
            decimals = int(decimals_raw)
        except (ValueError, TypeError) as exc:
            raise NormalizationError(
                self.provider_name, f"Invalid decimals value: {decimals_raw!r}"
            ) from exc

        token_amount_raw = tt.get("tokenAmount")
        if token_amount_raw is None:
            raise NormalizationError(
                self.provider_name,
                f"SPL token transfer for mint {tt.get('mint')!r} missing tokenAmount and rawTokenAmount.",
            )

        # Convert decimal-form string/int without float
        s = str(token_amount_raw).strip()
        if not s:
            raise NormalizationError(
                self.provider_name,
                f"SPL token transfer for mint {tt.get('mint')!r} has empty tokenAmount.",
            )

        try:
            if "." in s:
                int_part, frac_part = s.split(".", 1)
                int_part = int_part or "0"
                sign = -1 if int_part.startswith("-") else 1
                int_part = int_part.lstrip("-") or "0"
                if len(frac_part) > decimals:
                    frac_part = frac_part[:decimals]
                padded_frac = frac_part.ljust(decimals, "0")
                val = sign * (int(int_part) * (10 ** decimals) + (int(padded_frac) if padded_frac else 0))
            else:
                val = int(s) * (10 ** decimals)
            return val, decimals
        except (ValueError, TypeError) as exc:
            raise NormalizationError(
                self.provider_name,
                f"Failed to parse decimal token amount {s!r} with decimals {decimals}: {exc}",
            ) from exc

    def normalize_helius_tx(
        self,
        raw: Dict[str, Any],
        network: Network = Network.SOL_MAINNET,
    ) -> Transaction:
        """Normalize a Helius enhanced transaction response.

        This method is public for unit testing with mock data.
        It does NOT make HTTP calls.

        Helius enhanced transaction format includes:
        - signature: str
        - slot: int
        - blockTime: int (unix timestamp)
        - feePayer: str (account that paid the fee)
        - fee: int (lamports)
        - accountData: list of {account, nativeBalanceChange, tokenBalanceChanges}
        - nativeTransfers: list of {fromUserAccount, toUserAccount, amount (lamports)}
        - tokenTransfers: list of {fromUserAccount, toUserAccount, mint, tokenAmount, tokenStandard}
        - transactionError: null or error dict
        - type: str (e.g. "TRANSFER", "SWAP", "NFT_SALE", "UNKNOWN")

        Args:
            raw:     Dict as returned by Helius /v0/transactions (one item from list)
            network: Solana network (default: solana-mainnet)

        Returns:
            Canonical Transaction with all SOL and SPL token transfers.

        Raises:
            NormalizationError: If required fields are missing.
        """
        try:
            signature = raw.get("signature") or raw.get("txId", "")
            slot = raw.get("slot")
            block_time = raw.get("blockTime") or raw.get("timestamp")
            fee_lamports = int(raw.get("fee", 0))
            fee_payer = raw.get("feePayer")
            tx_error = raw.get("transactionError")
            helius_type = str(raw.get("type", "UNKNOWN")).upper()

            try:
                ts: Optional[datetime] = (
                    datetime.fromtimestamp(int(block_time), tz=timezone.utc)
                    if block_time else None
                )
            except (ValueError, TypeError, OSError):
                ts = None

            status = (
                TransactionStatus.FAILED if tx_error else TransactionStatus.SUCCESS
            )

            type_map = {
                "TRANSFER": TransactionType.TRANSFER,
                "SWAP": TransactionType.CONTRACT_CALL,
                "UNKNOWN": TransactionType.UNKNOWN,
            }
            tx_type = type_map.get(helius_type, TransactionType.UNKNOWN)

            # Build transfers
            transfers: List[Transfer] = []
            total_native_sent = 0

            # Native SOL transfers
            native_transfers = raw.get("nativeTransfers", [])
            for nt in native_transfers:
                amount_lam = int(nt.get("amount", 0))
                total_native_sent += amount_lam
                if amount_lam > 0:
                    transfers.append(Transfer(
                        from_address=nt.get("fromUserAccount"),
                        to_address=nt.get("toUserAccount"),
                        
                        asset_type=AssetType.NATIVE,
                        asset_symbol="SOL",
                        amount=amount_lam,
                        amount_unit="lamport",
                        decimals=9,
                        chain=Chain.SOLANA,
                        network=network,
                    ))

            # SPL token transfers
            token_transfers = raw.get("tokenTransfers", [])
            for tt in token_transfers:
                token_amount, token_decimals = self._parse_spl_token_amount_and_decimals(tt)

                symbol = tt.get("symbol") or tt.get("mint", "UNKNOWN")[:8]
                transfers.append(Transfer(
                    from_address=tt.get("fromUserAccount"),
                    to_address=tt.get("toUserAccount"),
                    asset_type=AssetType.TOKEN,
                    asset_symbol=symbol,
                    asset_contract=tt.get("mint"),
                    amount=token_amount,
                    amount_unit="base_unit",
                    decimals=token_decimals,
                    chain=Chain.SOLANA,
                    network=network,
                ))

            # Determine primary from/to from first native transfer or feePayer
            if native_transfers:
                primary_from = native_transfers[0].get("fromUserAccount") or fee_payer
                primary_to = native_transfers[0].get("toUserAccount")
            else:
                primary_from = fee_payer
                primary_to = None

            provenance = TransactionProvenance(
                provider=self.provider_name,
                chain=Chain.SOLANA,
                network=network,
                original_id=signature,
            )

            return Transaction(
                transaction_id=signature,
                chain=Chain.SOLANA,
                network=network,
                chain_id=None,
                block_number=slot,
                block_hash=None,
                timestamp=ts,
                from_address=primary_from,
                to_address=primary_to,
                native_value=total_native_sent,
                native_value_unit="lamport",
                
                transaction_type=tx_type,
                status=status,
                fee=fee_lamports,
                fee_asset="SOL",
                transfers=transfers,
                provenance=provenance,
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise NormalizationError(
                self.provider_name,
                f"Failed to normalize Solana transaction: {type(exc).__name__}",
            ) from exc


class PublicSolanaRPCAdapter(BaseDataProvider):
    """Public Solana RPC fallback adapter."""

    _PROVIDER_NAME = "solana-public-rpc"
    _DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
    _TIMEOUT = 15.0

    def __init__(self, rpc_url: Optional[str] = None) -> None:
        self._rpc_url = rpc_url or self._DEFAULT_RPC_URL

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.SOLANA]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.BLOCK_LOOKUP,
        }

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        network = kwargs.get("network", Network.SOL_MAINNET)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_id,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
                resp = await client.post(self._rpc_url, json=payload)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, f"RPC timeout for {tx_id}")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"RPC network error: {type(exc).__name__}")

        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Solana public RPC rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"RPC server error {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON RPC response: {exc}") from exc

        result = data.get("result")
        if result is None:
            raise TransactionNotFoundError(self.provider_name, tx_id)

        return self.normalize_rpc_tx(result, tx_id=tx_id, network=network)

    def normalize_rpc_tx(
        self, raw: Dict[str, Any], tx_id: str, network: Network = Network.SOL_MAINNET
    ) -> Transaction:
        try:
            slot = raw.get("slot")
            block_time = raw.get("blockTime")
            meta = raw.get("meta", {})
            transaction = raw.get("transaction", {})
            message = transaction.get("message", {})

            try:
                ts: Optional[datetime] = (
                    datetime.fromtimestamp(int(block_time), tz=timezone.utc)
                    if block_time else None
                )
            except (ValueError, TypeError, OSError):
                ts = None

            fee = int(meta.get("fee", 0))
            status = TransactionStatus.FAILED if meta.get("err") else TransactionStatus.SUCCESS

            account_keys = message.get("accountKeys", [])
            accounts: List[str] = []
            for ak in account_keys:
                if isinstance(ak, dict):
                    accounts.append(ak.get("pubkey", ""))
                else:
                    accounts.append(str(ak))

            from_addr = accounts[0] if accounts else None
            to_addr = accounts[1] if len(accounts) > 1 else None

            transfers: List[Transfer] = []
            total_native = 0

            instructions = message.get("instructions", [])
            for ix in instructions:
                parsed = ix.get("parsed") if isinstance(ix, dict) else None
                if isinstance(parsed, dict) and parsed.get("type") == "transfer":
                    info = parsed.get("info", {})
                    if ix.get("program") == "system":
                        lamports = int(info.get("lamports", 0))
                        total_native += lamports
                        transfers.append(Transfer(
                            from_address=info.get("source"),
                            to_address=info.get("destination"),
                            asset_type=AssetType.NATIVE,
                            asset_symbol="SOL",
                            amount=lamports,
                            amount_unit="lamport",
                            decimals=9,
                            chain=Chain.SOLANA,
                            network=network,
                        ))
                    elif ix.get("program") in ("spl-token", "spl-token-2022"):
                        raw_amt = info.get("amount") or info.get("tokenAmount", {}).get("amount", 0)
                        amt = int(raw_amt)
                        dec = info.get("tokenAmount", {}).get("decimals")
                        mint = info.get("mint")
                        transfers.append(Transfer(
                            from_address=info.get("source") or info.get("authority"),
                            to_address=info.get("destination"),
                            asset_type=AssetType.TOKEN,
                            asset_symbol="SPL",
                            asset_contract=mint,
                            amount=amt,
                            amount_unit="base_unit",
                            decimals=dec,
                            chain=Chain.SOLANA,
                            network=network,
                        ))

            provenance = TransactionProvenance(
                provider=self.provider_name,
                chain=Chain.SOLANA,
                network=network,
                original_id=tx_id,
            )

            return Transaction(
                transaction_id=tx_id,
                chain=Chain.SOLANA,
                network=network,
                chain_id=None,
                block_number=slot,
                block_hash=None,
                timestamp=ts,
                from_address=from_addr,
                to_address=to_addr,
                native_value=total_native,
                native_value_unit="lamport",
                transaction_type=TransactionType.TRANSFER if transfers else TransactionType.UNKNOWN,
                status=status,
                fee=fee,
                fee_asset="SOL",
                transfers=transfers,
                provenance=provenance,
            )
        except Exception as exc:
            raise NormalizationError(self.provider_name, f"Failed to normalize RPC tx {tx_id}: {exc}") from exc
