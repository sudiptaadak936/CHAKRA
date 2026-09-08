"""EVM transaction adapter for Ethereum-compatible chains.

Supports Ethereum mainnet, BNB Smart Chain, Polygon, and any EVM chain
accessible via Etherscan V2 (primary) or Blockscout (fallback stub).

Architecture:
    EtherscanClient (Step 0.5 health-check client, unchanged)
    EVMTransactionAdapter (this module)
        |
    Transaction (canonical model)

The adapter normalizes:
- Native ETH/BNB/MATIC transfers
- ERC-20 token transfers
- Contract calls (value=0 native, token movements in transfers list)
- Contract creation (to_address=None)
- Fee in wei (gas_used * gas_price)

EVM chain support:
    chain_id=1   -> ethereum-mainnet (Etherscan V2)
    chain_id=56  -> bsc-mainnet      (Etherscan V2 multi-chain)
    chain_id=137 -> polygon-mainnet  (Etherscan V2 multi-chain)

Etherscan V2 is the primary provider. Blockscout remains architecturally
available as a fallback but is not yet implemented (raises UnsupportedOperationError
for Blockscout-specific paths).

Security: API key must be passed at construction and stored privately.
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
from app.schemas.chain import Chain, EVM_CHAIN_ID_TO_NETWORK, NETWORK_TO_EVM_CHAIN_ID, Network
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)

logger = logging.getLogger(__name__)

_ETHERSCAN_V2_BASE = "https://api.etherscan.io/v2/api"
_TIMEOUT = 15.0

# Etherscan V2 status codes
_STATUS_OK = "1"
_STATUS_ERROR = "0"


class EVMTransactionAdapter(BaseDataProvider):
    """Blockchain data provider for EVM-compatible chains.

    Primary: Etherscan V2 (supports multi-chain via chain_id parameter)
    Fallback: Blockscout (stub only; raises UnsupportedOperationError)

    Constructor args:
        api_key: Etherscan API key (stored privately; never exposed).
        default_chain_id: Default EVM chain_id if not specified per-call (default: 1).
    """

    def __init__(self, api_key: Optional[str], default_chain_id: int = 1) -> None:
        self._api_key = api_key   # private; never log or include in errors
        self._default_chain_id = default_chain_id

    @property
    def provider_name(self) -> str:
        return "etherscan-evm"

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.EVM]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.ADDRESS_HISTORY,
            ProviderCapability.TOKEN_TRANSFERS,
            ProviderCapability.NATIVE_TRANSFERS,
        }

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        """Retrieve and normalize an EVM transaction by hash.

        Args:
            tx_id: Ethereum transaction hash (0x-prefixed).
            chain_id (kwarg): EVM chain ID (default: self._default_chain_id).

        Returns:
            Normalized canonical Transaction with all token and native transfers.

        Raises:
            ProviderAuthError:        API key rejected.
            ProviderRateLimitError:   Rate limited (no key in error).
            ProviderTimeoutError:     Request timed out.
            ProviderUnavailableError: Etherscan server error.
            TransactionNotFoundError: Transaction not found.
            MalformedResponseError:   Unexpected response shape.
            NormalizationError:       Could not build Transaction from response.
        """
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        if "chain_id" in kwargs:
            chain_id = int(kwargs["chain_id"])
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, kwargs.get("network", Network.UNKNOWN))
        elif "network" in kwargs and kwargs["network"] in NETWORK_TO_EVM_CHAIN_ID:
            network = kwargs["network"]
            chain_id = NETWORK_TO_EVM_CHAIN_ID[network]
        else:
            chain_id = self._default_chain_id
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, Network.UNKNOWN)

        # Fetch basic transaction data
        raw_tx = await self._fetch_transaction_by_hash(tx_id, chain_id)
        # Fetch token transfers for this hash
        token_transfers_raw = await self._fetch_token_transfers(tx_id, chain_id)

        try:
            return self._normalize(
                raw_tx=raw_tx,
                token_transfers_raw=token_transfers_raw,
                tx_id=tx_id,
                chain_id=chain_id,
                network=network,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise NormalizationError(
                self.provider_name,
                f"Failed to normalize EVM transaction {tx_id}: {type(exc).__name__}",
            ) from exc

    async def get_address_transactions(
        self,
        address: str,
        limit: int = 25,
        offset: int = 0,
        **kwargs,
    ) -> List[Transaction]:
        """List recent normal transactions for an EVM address via Etherscan.

        Args:
            address: EVM address (case preserved; Etherscan accepts checksummed and lowercase).
            limit:   Max transactions to return (maps to Etherscan `offset` page size).
            offset:  Page offset.
            chain_id (kwarg): EVM chain ID (default: self._default_chain_id).

        Returns:
            List of canonical Transactions (may be empty for new addresses).
        """
        if not self._api_key:
            raise ProviderAuthError(self.provider_name, "No API key configured.")

        if "chain_id" in kwargs:
            chain_id = int(kwargs["chain_id"])
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, kwargs.get("network", Network.UNKNOWN))
        elif "network" in kwargs and kwargs["network"] in NETWORK_TO_EVM_CHAIN_ID:
            network = kwargs["network"]
            chain_id = NETWORK_TO_EVM_CHAIN_ID[network]
        else:
            chain_id = self._default_chain_id
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, Network.UNKNOWN)

        params = {
            "chainid": str(chain_id),
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": (offset // limit) + 1 if limit > 0 else 1,
            "offset": limit,
            "sort": "desc",
            "apikey": self._api_key,
        }

        raw_list = await self._etherscan_get(params)
        if not isinstance(raw_list, list):
            return []

        results: List[Transaction] = []
        for raw_tx in raw_list:
            try:
                token_transfers_raw = []   # address-level listing omits token detail
                tx = self._normalize(
                    raw_tx=raw_tx,
                    token_transfers_raw=token_transfers_raw,
                    tx_id=raw_tx.get("hash", ""),
                    chain_id=chain_id,
                    network=network,
                )
                results.append(tx)
            except Exception as exc:
                logger.warning(
                    "[%s] Skipping tx due to normalization error: %s",
                    self.provider_name,
                    type(exc).__name__,
                )
        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _fetch_transaction_by_hash(
        self, tx_hash: str, chain_id: int
    ) -> Dict[str, Any]:
        """Fetch transaction receipt + info from Etherscan V2."""
        params = {
            "chainid": str(chain_id),
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash,
            "apikey": self._api_key,
        }
        result = await self._etherscan_get(params)
        if result is None:
            raise TransactionNotFoundError(self.provider_name, tx_hash)
        if not isinstance(result, dict):
            raise MalformedResponseError(
                self.provider_name,
                f"Expected dict for transaction, got {type(result).__name__}",
            )
        return result

    async def _fetch_token_transfers(
        self, tx_hash: str, chain_id: int
    ) -> List[Dict[str, Any]]:
        """Fetch ERC-20 token transfers for a transaction hash."""
        params = {
            "chainid": str(chain_id),
            "module": "account",
            "action": "tokentx",
            "txhash": tx_hash,
            "apikey": self._api_key,
        }
        try:
            result = await self._etherscan_get(params)
            if isinstance(result, list):
                return result
        except (TransactionNotFoundError, MalformedResponseError):
            pass  # No token transfers is not an error
        return []

    async def _etherscan_get(self, params: Dict[str, Any]) -> Any:
        """Execute an Etherscan V2 GET request and return the `result` field.

        Handles HTTP-level and application-level errors.
        API key is in params but never included in error messages.
        """
        safe_params = {k: v for k, v in params.items() if k != "apikey"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_ETHERSCAN_V2_BASE, params=params)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                self.provider_name, "Etherscan request timed out."
            )
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(
                self.provider_name, f"Network error: {type(exc).__name__}"
            )

        if resp.status_code in (401, 403):
            raise ProviderAuthError(self.provider_name, "Etherscan authentication failed.")
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Etherscan rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(
                self.provider_name, f"Etherscan server error {resp.status_code}."
            )

        try:
            data = resp.json()
        except Exception:
            raise MalformedResponseError(
                self.provider_name, "Non-JSON response from Etherscan."
            )

        status = str(data.get("status", ""))
        result = data.get("result")
        message = str(data.get("message", ""))

        # Application-level auth failure
        if status == _STATUS_ERROR:
            result_str = str(result or "")
            if "invalid api key" in result_str.lower() or "apikey" in result_str.lower():
                raise ProviderAuthError(
                    self.provider_name, "Etherscan rejected the API key."
                )
            if "max rate" in result_str.lower() or "rate limit" in result_str.lower():
                raise ProviderRateLimitError(
                    self.provider_name, "Etherscan rate limit reached."
                )
            if result == [] or result_str == "":
                return []
            return result  # May be an error string; callers check type

        return result

    def _normalize(
        self,
        raw_tx: Dict[str, Any],
        token_transfers_raw: List[Dict[str, Any]],
        tx_id: str,
        chain_id: int,
        network: Network,
    ) -> Transaction:
        """Build a canonical Transaction from Etherscan raw response dicts."""
        # Value in wei (hex string from eth_getTransactionByHash, or decimal from txlist)
        value_raw = raw_tx.get("value", "0x0")
        if isinstance(value_raw, str) and value_raw.startswith("0x"):
            value_wei = int(value_raw, 16)
        else:
            value_wei = int(value_raw or 0)

        from_addr = raw_tx.get("from") or raw_tx.get("from_address")
        to_addr = raw_tx.get("to") or raw_tx.get("to_address")

        # Gas fee in wei
        gas_used_raw = raw_tx.get("gasUsed") or raw_tx.get("gas", "0")
        gas_price_raw = raw_tx.get("gasPrice", "0")
        try:
            if isinstance(gas_used_raw, str) and gas_used_raw.startswith("0x"):
                gas_used = int(gas_used_raw, 16)
            else:
                gas_used = int(gas_used_raw or 0)
            if isinstance(gas_price_raw, str) and gas_price_raw.startswith("0x"):
                gas_price = int(gas_price_raw, 16)
            else:
                gas_price = int(gas_price_raw or 0)
            fee_wei: Optional[int] = gas_used * gas_price
        except (ValueError, TypeError):
            fee_wei = None

        # Block info
        block_number_raw = raw_tx.get("blockNumber")
        try:
            if isinstance(block_number_raw, str) and block_number_raw.startswith("0x"):
                block_number: Optional[int] = int(block_number_raw, 16)
            elif block_number_raw is not None:
                block_number = int(block_number_raw)
            else:
                block_number = None
        except (ValueError, TypeError):
            block_number = None

        ts_raw = raw_tx.get("timeStamp") or raw_tx.get("timestamp")
        try:
            ts: Optional[datetime] = (
                datetime.fromtimestamp(int(ts_raw), tz=timezone.utc) if ts_raw else None
            )
        except (ValueError, TypeError, OSError):
            ts = None

        # Transaction status
        is_error = str(raw_tx.get("isError", "0"))
        tx_receipt_status = str(raw_tx.get("txreceipt_status", ""))
        if is_error == "1":
            status = TransactionStatus.FAILED
        elif tx_receipt_status == "1" or (not is_error and not tx_receipt_status):
            status = TransactionStatus.SUCCESS
        else:
            status = TransactionStatus.UNKNOWN

        # Transaction type
        if to_addr is None or to_addr == "":
            tx_type = TransactionType.CONTRACT_CREATION
        elif raw_tx.get("input", "0x") not in ("0x", "", None):
            tx_type = TransactionType.CONTRACT_CALL
        else:
            tx_type = TransactionType.TRANSFER

        # Determine native asset symbol from chain_id
        native_symbol_map = {1: "ETH", 56: "BNB", 137: "MATIC"}
        native_symbol = native_symbol_map.get(chain_id, "ETH")

        # Build transfers list
        transfers: List[Transfer] = []

        # Native transfer (always present; value may be 0 for pure token txs)
        if value_wei > 0 or not token_transfers_raw:
            transfers.append(
                Transfer(
                    from_address=from_addr,
                    to_address=to_addr,
                    
                    asset_type=AssetType.NATIVE,
                asset_symbol=native_symbol,
                amount=value_wei,
                    amount_unit="wei",
                    decimals=18,
                    chain=Chain.EVM,
                    network=network,
                )
            )

        # ERC-20 token transfers
        for tt in token_transfers_raw:
            try:
                token_value_raw = tt.get("value", "0")
                token_decimals_raw = tt.get("tokenDecimal") or tt.get("decimals")
                transfers.append(
                    Transfer(
                        from_address=tt.get("from"),
                        to_address=tt.get("to"),
                        asset_type=AssetType.TOKEN,
                        asset_symbol=tt.get("tokenSymbol", "UNKNOWN"),
                        asset_contract=tt.get("contractAddress"),
                        amount=int(token_value_raw),
                        amount_unit="base_unit",
                        decimals=int(token_decimals_raw) if token_decimals_raw is not None else None,
                        chain=Chain.EVM,
                        network=network,
                    )
                )
            except Exception:
                logger.debug("[%s] Skipping malformed token transfer entry.", self.provider_name)

        provenance = TransactionProvenance(
            provider=self.provider_name,
            chain=Chain.EVM,
            network=network,
            original_id=tx_id,
        )

        return Transaction(
            transaction_id=tx_id,
            chain=Chain.EVM,
            network=network,
            chain_id=chain_id,
            block_number=block_number,
            block_hash=raw_tx.get("blockHash"),
            timestamp=ts,
            from_address=from_addr,
            to_address=to_addr or None,
            native_value=value_wei,
            native_value_unit="wei",
            
            transaction_type=tx_type,
            status=status,
            fee=fee_wei,
            fee_asset=native_symbol,
            transfers=transfers,
            provenance=provenance,
        )


class BlockscoutEVMAdapter(BaseDataProvider):
    """Blockscout EVM fallback adapter."""

    _PROVIDER_NAME = "blockscout-evm"
    _TIMEOUT = 15.0

    _CHAIN_BASE_URLS = {
        1: "https://eth.blockscout.com/api/v2",
        56: "https://bsc.blockscout.com/api/v2",
        137: "https://polygon.blockscout.com/api/v2",
    }

    def __init__(self, base_url: Optional[str] = None, default_chain_id: int = 1) -> None:
        self._base_url = base_url
        self._default_chain_id = default_chain_id

    @property
    def provider_name(self) -> str:
        return self._PROVIDER_NAME

    @property
    def supported_chains(self) -> List[Chain]:
        return [Chain.EVM]

    @property
    def capabilities(self) -> Set[ProviderCapability]:
        return {
            ProviderCapability.TRANSACTION_LOOKUP,
            ProviderCapability.ADDRESS_HISTORY,
            ProviderCapability.TOKEN_TRANSFERS,
            ProviderCapability.NATIVE_TRANSFERS,
        }

    def _get_base_url(self, chain_id: int) -> str:
        if self._base_url:
            return self._base_url.rstrip("/")
        return self._CHAIN_BASE_URLS.get(chain_id, "https://eth.blockscout.com/api/v2")

    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        if "chain_id" in kwargs:
            chain_id = int(kwargs["chain_id"])
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, kwargs.get("network", Network.UNKNOWN))
        elif "network" in kwargs and kwargs["network"] in NETWORK_TO_EVM_CHAIN_ID:
            network = kwargs["network"]
            chain_id = NETWORK_TO_EVM_CHAIN_ID[network]
        else:
            chain_id = self._default_chain_id
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, Network.UNKNOWN)

        base_url = self._get_base_url(chain_id)
        url = f"{base_url}/transactions/{tx_id}"

        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
                resp = await client.get(url)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, f"Blockscout request timed out for {tx_id}")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Blockscout network error: {type(exc).__name__}")

        if resp.status_code == 404:
            raise TransactionNotFoundError(self.provider_name, tx_id)
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Blockscout rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Blockscout server error {resp.status_code}")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            raw = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response: {exc}") from exc

        return self.normalize_blockscout_tx(raw, tx_id=tx_id, chain_id=chain_id, network=network)

    async def get_address_transactions(
        self, address: str, limit: int = 25, offset: int = 0, **kwargs
    ) -> List[Transaction]:
        if "chain_id" in kwargs:
            chain_id = int(kwargs["chain_id"])
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, kwargs.get("network", Network.UNKNOWN))
        elif "network" in kwargs and kwargs["network"] in NETWORK_TO_EVM_CHAIN_ID:
            network = kwargs["network"]
            chain_id = NETWORK_TO_EVM_CHAIN_ID[network]
        else:
            chain_id = self._default_chain_id
            network = EVM_CHAIN_ID_TO_NETWORK.get(chain_id, Network.UNKNOWN)

        base_url = self._get_base_url(chain_id)
        url = f"{base_url}/addresses/{address}/transactions"

        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT) as client:
                resp = await client.get(url)
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, f"Blockscout request timed out for {address}")
        except httpx.NetworkError as exc:
            raise ProviderUnavailableError(self.provider_name, f"Blockscout network error: {type(exc).__name__}")

        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            raise ProviderRateLimitError(self.provider_name, "Blockscout rate limit reached.")
        if resp.status_code >= 500:
            raise ProviderUnavailableError(self.provider_name, f"Blockscout server error {resp.status_code}")
        if resp.status_code != 200:
            raise ProviderUnavailableError(self.provider_name, f"Unexpected status code {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:
            raise MalformedResponseError(self.provider_name, f"Non-JSON response: {exc}") from exc

        items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        results: List[Transaction] = []
        for raw in items[:limit]:
            try:
                tx_hash = raw.get("hash") or ""
                results.append(self.normalize_blockscout_tx(raw, tx_id=tx_hash, chain_id=chain_id, network=network))
            except Exception as exc:
                logger.warning("[%s] Skipping malformed tx: %s", self.provider_name, exc)

        return results

    def normalize_blockscout_tx(
        self, raw: Dict[str, Any], tx_id: str, chain_id: int, network: Network
    ) -> Transaction:
        try:
            from_data = raw.get("from")
            from_addr = from_data.get("hash") if isinstance(from_data, dict) else from_data
            to_data = raw.get("to")
            to_addr = to_data.get("hash") if isinstance(to_data, dict) else to_data

            val_raw = raw.get("value", "0")
            value_wei = int(val_raw)

            fee_data = raw.get("fee")
            if isinstance(fee_data, dict) and "value" in fee_data:
                fee_wei: Optional[int] = int(fee_data["value"])
            elif raw.get("gas_used") and raw.get("gas_price"):
                fee_wei = int(raw["gas_used"]) * int(raw["gas_price"])
            else:
                fee_wei = None

            block_number = raw.get("block_number") or raw.get("block")
            block_num = int(block_number) if block_number is not None else None

            ts_str = raw.get("timestamp")
            if ts_str:
                try:
                    ts: Optional[datetime] = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    ts = None
            else:
                ts = None

            res_status = raw.get("result") or raw.get("status")
            if res_status in ("success", "ok", 1, "1"):
                status = TransactionStatus.SUCCESS
            elif res_status in ("error", "failed", 0, "0"):
                status = TransactionStatus.FAILED
            else:
                status = TransactionStatus.UNKNOWN

            native_symbol_map = {1: "ETH", 56: "BNB", 137: "MATIC", 11155111: "ETH"}
            native_symbol = native_symbol_map.get(chain_id, "ETH")

            transfers: List[Transfer] = []
            if value_wei > 0:
                transfers.append(Transfer(
                    from_address=from_addr,
                    to_address=to_addr,
                    asset_type=AssetType.NATIVE,
                    asset_symbol=native_symbol,
                    amount=value_wei,
                    amount_unit="wei",
                    decimals=18,
                    chain=Chain.EVM,
                    network=network,
                ))

            raw_token_transfers = raw.get("token_transfers", [])
            if isinstance(raw_token_transfers, list):
                for tt in raw_token_transfers:
                    try:
                        t_from = tt.get("from", {}).get("hash") if isinstance(tt.get("from"), dict) else tt.get("from")
                        t_to = tt.get("to", {}).get("hash") if isinstance(tt.get("to"), dict) else tt.get("to")
                        total = tt.get("total", {})
                        t_amt = int(total.get("value", 0)) if isinstance(total, dict) else int(tt.get("value", 0))
                        token_info = tt.get("token", {})
                        transfers.append(Transfer(
                            from_address=t_from,
                            to_address=t_to,
                            asset_type=AssetType.TOKEN,
                            asset_symbol=token_info.get("symbol", "TOKEN"),
                            asset_contract=token_info.get("address"),
                            amount=t_amt,
                            amount_unit="base_unit",
                            decimals=int(total.get("decimals")) if isinstance(total, dict) and total.get("decimals") else None,
                            chain=Chain.EVM,
                            network=network,
                        ))
                    except Exception:
                        pass

            provenance = TransactionProvenance(
                provider=self.provider_name,
                chain=Chain.EVM,
                network=network,
                original_id=tx_id or raw.get("hash", ""),
            )

            tx_type = TransactionType.CONTRACT_CREATION if to_addr is None else TransactionType.TRANSFER

            return Transaction(
                transaction_id=tx_id or raw.get("hash", ""),
                chain=Chain.EVM,
                network=network,
                chain_id=chain_id,
                block_number=block_num,
                block_hash=raw.get("block_hash"),
                timestamp=ts,
                from_address=from_addr,
                to_address=to_addr,
                native_value=value_wei,
                native_value_unit="wei",
                transaction_type=tx_type,
                status=status,
                fee=fee_wei,
                fee_asset=native_symbol,
                transfers=transfers,
                provenance=provenance,
            )
        except Exception as exc:
            raise NormalizationError(self.provider_name, f"Failed to normalize Blockscout tx: {exc}") from exc


