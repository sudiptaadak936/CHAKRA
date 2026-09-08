"""BaseDataProvider ABC, ProviderCapability enum, and ProviderError hierarchy.

This module defines the INGESTION-layer provider abstraction.
It is completely separate from BaseProviderClient (health-check layer).

Architecture:
    BaseDataProvider         <- this module
        |
    Chain-specific adapter   <- providers/adapters/{evm,tron,bitcoin,solana}.py
        |
    Canonical Transaction    <- schemas/transaction.py

The health-check provider hierarchy (BaseProviderClient in providers/base.py)
is NOT modified. Both hierarchies can coexist cleanly.

Security rules enforced by this module:
- API keys must NEVER appear in ProviderError messages or __repr__ outputs.
- Auth headers must NEVER be logged or included in exceptions.
- Callers should catch ProviderError (or subclasses) and handle gracefully.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Set, Union

from app.schemas.chain import Chain
from app.schemas.transaction import Transaction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability enum
# ---------------------------------------------------------------------------

class ProviderCapability(str, Enum):
    """Operations that a data provider may support.

    Not every provider supports every operation.
    Callers should check `provider.capabilities` before invoking an operation,
    or catch UnsupportedOperationError.

    Values:
        TRANSACTION_LOOKUP: Retrieve a single transaction by ID/hash/signature.
        ADDRESS_HISTORY:    List transactions for an address (paginated).
        BLOCK_LOOKUP:       Retrieve a block by height or hash.
        TOKEN_TRANSFERS:    Resolve token transfers within a transaction.
        NATIVE_TRANSFERS:   Resolve native coin transfers within a transaction.
    """

    TRANSACTION_LOOKUP = "transaction_lookup"
    ADDRESS_HISTORY = "address_history"
    BLOCK_LOOKUP = "block_lookup"
    TOKEN_TRANSFERS = "token_transfers"
    NATIVE_TRANSFERS = "native_transfers"


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base class for all CHAKRA data-provider errors.

    IMPORTANT: Subclass messages MUST NOT contain API keys, auth tokens,
    raw authorization headers, or any credential material.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderUnavailableError(ProviderError):
    """Provider is unreachable or returned an HTTP 5xx response."""


class ProviderTimeoutError(ProviderError):
    """Request to the provider timed out."""


class ProviderAuthError(ProviderError):
    """Provider rejected the request due to authentication failure (401/403).

    The error message MUST NOT contain the API key or auth header value.
    """


class ProviderRateLimitError(ProviderError):
    """Provider returned HTTP 429 (rate limited)."""


class TransactionNotFoundError(ProviderError):
    """The requested transaction ID was not found (404 or empty result)."""

    def __init__(self, provider: str, transaction_id: str) -> None:
        # transaction_id is a public blockchain hash, not a credential.
        super().__init__(provider, f"Transaction not found: {transaction_id}")
        self.transaction_id = transaction_id


class MalformedResponseError(ProviderError):
    """Provider returned a response that could not be parsed or validated."""


class UnsupportedOperationError(ProviderError):
    """The requested operation is not supported by this provider."""

    def __init__(self, provider: str, operation: str) -> None:
        super().__init__(provider, f"Operation not supported: {operation}")
        self.operation = operation


class NormalizationError(ProviderError):
    """The chain adapter could not normalize the provider response into a Transaction."""


# ---------------------------------------------------------------------------
# BaseDataProvider ABC
# ---------------------------------------------------------------------------

class BaseDataProvider(ABC):
    """Abstract base class for blockchain data ingestion providers.

    Subclasses implement chain-specific API calls and normalization.
    The public interface returns only canonical Transaction objects.

    API keys must be stored as private attributes and must never appear
    in returned objects, error messages, or log statements.

    Example implementation chain:
        EtherscanClient (HTTP layer)
            |
        EVMTransactionAdapter(BaseDataProvider)
            |
        Transaction (canonical model)
    """

    # -----------------------------------------------------------------------
    # Abstract properties
    # -----------------------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (no credential data)."""
        ...

    @property
    @abstractmethod
    def supported_chains(self) -> List[Chain]:
        """Blockchain families this provider can serve."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Set[ProviderCapability]:
        """Operations this provider supports."""
        ...

    # -----------------------------------------------------------------------
    # Required operations
    # -----------------------------------------------------------------------

    @abstractmethod
    async def get_transaction(self, tx_id: str, **kwargs) -> Transaction:
        """Retrieve and normalize a single transaction by its ID.

        Args:
            tx_id: Transaction hash (EVM/Tron), txid (Bitcoin), or signature (Solana).
            **kwargs: Chain-specific options (e.g. chain_id for EVM).

        Returns:
            Normalized canonical Transaction.

        Raises:
            TransactionNotFoundError: Transaction does not exist on chain.
            ProviderAuthError:        Authentication rejected (no key in message).
            ProviderRateLimitError:   Rate limit hit.
            ProviderTimeoutError:     Request timed out.
            ProviderUnavailableError: Provider is down.
            MalformedResponseError:   Response could not be parsed.
            NormalizationError:       Response parsed but cannot form a Transaction.
            UnsupportedOperationError: Provider does not support transaction lookup.
        """
        ...

    # -----------------------------------------------------------------------
    # Optional operations (raise UnsupportedOperationError by default)
    # -----------------------------------------------------------------------

    async def get_address_transactions(
        self,
        address: str,
        limit: int = 25,
        offset: int = 0,
        **kwargs,
    ) -> List[Transaction]:
        """List recent transactions for an address.

        Default implementation raises UnsupportedOperationError.
        Override if the provider supports address history.

        Args:
            address: Blockchain address (preserved as-is; not normalized).
            limit:   Maximum number of transactions to return.
            offset:  Pagination offset.
            **kwargs: Provider-specific options.

        Returns:
            List of canonical Transactions (may be empty).

        Raises:
            UnsupportedOperationError: If not overridden by subclass.
        """
        raise UnsupportedOperationError(self.provider_name, "get_address_transactions")

    async def get_block(
        self,
        block_id: Union[int, str],
        **kwargs,
    ) -> dict:
        """Retrieve raw block data by height or hash.

        Returns raw dict rather than a typed model because block schemas
        vary significantly across chains. Normalization is deferred.

        Default implementation raises UnsupportedOperationError.

        Args:
            block_id: Block height (int) or block hash (str).
            **kwargs: Provider-specific options.

        Raises:
            UnsupportedOperationError: If not overridden by subclass.
        """
        raise UnsupportedOperationError(self.provider_name, "get_block")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def supports(self, capability: ProviderCapability) -> bool:
        """Return True if this provider supports the given capability."""
        return capability in self.capabilities

    def __repr__(self) -> str:
        # Deliberately minimal; must not expose API keys.
        caps = ", ".join(c.value for c in sorted(self.capabilities))
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name!r}, "
            f"chains={[c.value for c in self.supported_chains]!r}, "
            f"capabilities=[{caps}])"
        )
