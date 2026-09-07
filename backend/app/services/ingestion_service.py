"""Blockchain data ingestion and normalization service.

Orchestrates:
    Blockchain Provider -> Data Provider Adapter -> Canonical Transaction -> PostgreSQL Repository

Enforces:
- Provider-independent canonical normalization
- Deterministic fallback between primary and secondary providers
- Bounded retry for transient provider errors (timeout, unavailable, rate-limit)
- Zero secret leakage in logs or returned error models
- Idempotent and concurrency-safe persistence via Step 1C TransactionRepository
- Exact integer monetary preservation (zero float usage)
- Bitcoin UTXO structure preservation
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from app.db.repository import TransactionRepository
from app.providers.adapters.bitcoin import BitcoinTransactionAdapter
from app.providers.data_base import (
    BaseDataProvider,
    MalformedResponseError,
    NormalizationError,
    ProviderAuthError,
    ProviderCapability,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TransactionNotFoundError,
    UnsupportedOperationError,
)
from app.providers.data_registry import DataProviderRegistry, get_default_data_provider_registry
from app.schemas.chain import Chain, Network
from app.schemas.ingestion import AddressHistoryIngestionResult, IngestionResult, IngestionStatus
from app.schemas.transaction import BitcoinTransactionDetail, Transaction, TransactionProvenance

logger = logging.getLogger(__name__)

# Transient errors eligible for bounded retry on the SAME provider
RETRYABLE_ERRORS = (
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
)


class FallbackPolicy:
    """Policy governing provider fallback behavior.

    By default:
    - fallback_on_auth: False (Authentication errors fail immediately without fallback unless explicitly enabled)
    - fallback_on_not_found: True (Fallback to secondary indexers in case of indexing lag)
    - fallback_on_malformed: True (Fallback to secondary in case primary returns unparseable shape)
    """

    def __init__(
        self,
        fallback_on_auth: bool = False,
        fallback_on_not_found: bool = True,
        fallback_on_malformed: bool = True,
    ) -> None:
        self.fallback_on_auth = fallback_on_auth
        self.fallback_on_not_found = fallback_on_not_found
        self.fallback_on_malformed = fallback_on_malformed


class IngestionService:
    """Service orchestrating blockchain transaction ingestion and persistence."""

    def __init__(
        self,
        registry: Optional[DataProviderRegistry] = None,
        repository: Optional[TransactionRepository] = None,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        max_delay: float = 5.0,
        fallback_policy: Optional[FallbackPolicy] = None,
    ) -> None:
        self.registry = registry or get_default_data_provider_registry()
        self.repository = repository
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_delay = max_delay
        self.fallback_policy = fallback_policy or FallbackPolicy()

    async def ingest_direct(self, tx: Transaction) -> IngestionResult:
        """Ingest a pre-constructed Transaction object directly without querying providers.
        
        This is intended for Demo Mode synthetic data ingestion and explicit data loading.
        Persists idempotently to PostgreSQL via Step 1C TransactionRepository.
        """
        # Extract Bitcoin UTXO details if applicable
        btc_detail: Optional[BitcoinTransactionDetail] = None
        if tx.chain == Chain.BITCOIN:
            from app.schemas.transaction import BitcoinVin, BitcoinVout
            inputs = [
                BitcoinVin(address=t.from_address, value_sat=t.amount)
                for t in tx.transfers if t.from_address is not None
            ]
            outputs = [
                BitcoinVout(n=i, address=t.to_address, value_sat=t.amount)
                for i, t in enumerate([t for t in tx.transfers if t.to_address is not None])
            ]
            total_in = sum(v.value_sat for v in inputs if v.value_sat is not None) if inputs else None
            total_out = sum(v.value_sat for v in outputs)
            btc_detail = BitcoinTransactionDetail(
                txid=tx.transaction_id,
                inputs=inputs,
                outputs=outputs,
                total_input_sat=total_in,
                total_output_sat=total_out,
                fee_sat=tx.fee,
            )

        already_existed = False
        if self.repository is not None:
            already_existed = await self.repository.transaction_exists(
                tx.transaction_id, tx.chain, tx.network
            )
            await self.repository.save_transaction(tx, btc_detail)

        status = (
            IngestionStatus.ALREADY_EXISTS
            if already_existed
            else IngestionStatus.SUCCESS
        )

        return IngestionResult(
            success=True,
            status=status,
            chain=tx.chain,
            network=tx.network,
            transaction_id=tx.transaction_id,
            provider="direct",
            fallback_used=False,
            transaction=tx,
        )

    async def ingest_transaction(
        self,
        chain: Chain,
        network: Network,
        tx_id: str,
        **kwargs,
    ) -> IngestionResult:
        """Ingest, normalize, and persist a single blockchain transaction.

        Executes deterministic fallback across registered providers for the chain.
        Applies bounded retries on transient errors.
        Guarantees idempotent persistence and accurate provenance.
        """
        providers = self.registry.get_all(chain)
        if not providers:
            logger.error(
                "[Ingestion] No data providers registered for chain %s (tx: %s, network: %s)",
                chain.value,
                tx_id,
                network.value,
            )
            return IngestionResult(
                success=False,
                status=IngestionStatus.FAILED,
                chain=chain,
                network=network,
                transaction_id=tx_id,
                error_category="UnsupportedOperationError",
                error_message=f"No provider registered for chain {chain.value}",
            )

        primary_provider = providers[0]
        last_error: Optional[Exception] = None
        stop_fallback = False

        for idx, provider in enumerate(providers):
            if stop_fallback:
                break

            if not provider.supports(ProviderCapability.TRANSACTION_LOOKUP):
                logger.debug(
                    "[Ingestion] Provider %s does not support TRANSACTION_LOOKUP for tx %s, skipping",
                    provider.provider_name,
                    tx_id,
                )
                continue

            fallback_used = provider != primary_provider
            logger.info(
                "[Ingestion] Ingesting transaction %s on %s/%s using provider %s (fallback=%s, attempt_limit=%d)",
                tx_id,
                chain.value,
                network.value,
                provider.provider_name,
                fallback_used,
                self.max_retries + 1,
            )

            # Bounded retry loop for the current provider
            for attempt in range(self.max_retries + 1):
                try:
                    tx = await provider.get_transaction(tx_id, network=network, **kwargs)

                    # Ensure provenance records the actual provider that succeeded
                    if tx.provenance.provider != provider.provider_name:
                        tx = tx.model_copy(
                            update={
                                "provenance": TransactionProvenance(
                                    provider=provider.provider_name,
                                    chain=tx.provenance.chain,
                                    network=tx.provenance.network,
                                    original_id=tx.provenance.original_id,
                                    normalized_at=tx.provenance.normalized_at,
                                )
                            }
                        )

                    # Extract Bitcoin UTXO details if applicable
                    btc_detail: Optional[BitcoinTransactionDetail] = None
                    if chain == Chain.BITCOIN:
                        from app.schemas.transaction import BitcoinVin, BitcoinVout
                        inputs = [
                            BitcoinVin(address=t.from_address, value_sat=t.amount)
                            for t in tx.transfers if t.from_address is not None
                        ]
                        outputs = [
                            BitcoinVout(n=i, address=t.to_address, value_sat=t.amount)
                            for i, t in enumerate([t for t in tx.transfers if t.to_address is not None])
                        ]
                        total_in = sum(v.value_sat for v in inputs if v.value_sat is not None) if inputs else None
                        total_out = sum(v.value_sat for v in outputs)
                        btc_detail = BitcoinTransactionDetail(
                            txid=tx.transaction_id,
                            inputs=inputs,
                            outputs=outputs,
                            total_input_sat=total_in,
                            total_output_sat=total_out,
                            fee_sat=tx.fee,
                        )

                    # Persist via Step 1C repository if available
                    already_existed = False
                    if self.repository is not None:
                        already_existed = await self.repository.transaction_exists(
                            tx.transaction_id, tx.chain, tx.network
                        )
                        await self.repository.save_transaction(tx, btc_detail)

                    status = (
                        IngestionStatus.ALREADY_EXISTS
                        if already_existed
                        else IngestionStatus.SUCCESS
                    )

                    logger.info(
                        "[Ingestion] Successfully ingested transaction %s (status=%s, provider=%s, fallback=%s)",
                        tx_id,
                        status.value,
                        provider.provider_name,
                        fallback_used,
                    )

                    return IngestionResult(
                        success=True,
                        status=status,
                        chain=chain,
                        network=network,
                        transaction_id=tx_id,
                        provider=provider.provider_name,
                        fallback_used=fallback_used,
                        transaction=tx,
                    )

                except RETRYABLE_ERRORS as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Transient error on provider %s for tx %s (%s/%s, attempt %d/%d): %s",
                        provider.provider_name,
                        tx_id,
                        chain.value,
                        network.value,
                        attempt + 1,
                        self.max_retries + 1,
                        type(exc).__name__,
                    )
                    if attempt < self.max_retries:
                        delay = min(self.retry_delay * (2**attempt), self.max_delay)
                        await asyncio.sleep(delay)
                        continue
                    # Retries exhausted for this provider; fall through to next provider
                    logger.info(
                        "[Ingestion] Retries exhausted on provider %s for tx %s. Transitioning to fallback provider if available.",
                        provider.provider_name,
                        tx_id,
                    )
                    break

                except ProviderAuthError as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Authentication failure on provider %s for tx %s (%s/%s): %s",
                        provider.provider_name,
                        tx_id,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    if not self.fallback_policy.fallback_on_auth:
                        logger.info(
                            "[Ingestion] Fallback on auth error disabled by policy for provider %s. Stopping provider transitions.",
                            provider.provider_name,
                        )
                        stop_fallback = True
                    break

                except TransactionNotFoundError as exc:
                    last_error = exc
                    logger.info(
                        "[Ingestion] Transaction %s not found on provider %s (%s/%s)",
                        tx_id,
                        provider.provider_name,
                        chain.value,
                        network.value,
                    )
                    if not self.fallback_policy.fallback_on_not_found:
                        stop_fallback = True
                    break

                except (MalformedResponseError, NormalizationError) as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Malformed response / normalization error on provider %s for tx %s (%s/%s): %s",
                        provider.provider_name,
                        tx_id,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    if not self.fallback_policy.fallback_on_malformed:
                        stop_fallback = True
                    break

                except (UnsupportedOperationError, ProviderError) as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Non-retryable provider error on %s for tx %s (%s/%s): %s",
                        provider.provider_name,
                        tx_id,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    break

                except Exception as exc:
                    last_error = exc
                    logger.error(
                        "[Ingestion] Unexpected exception on provider %s for tx %s (%s/%s): %s",
                        provider.provider_name,
                        tx_id,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    break

        # All providers failed or stopped
        error_category = type(last_error).__name__ if last_error else "ProviderError"
        error_message = str(last_error) if last_error else "All providers failed"

        logger.error(
            "[Ingestion] Ingestion failed for tx %s on %s/%s (error_category=%s): %s",
            tx_id,
            chain.value,
            network.value,
            error_category,
            error_message,
        )

        return IngestionResult(
            success=False,
            status=IngestionStatus.FAILED,
            chain=chain,
            network=network,
            transaction_id=tx_id,
            error_category=error_category,
            error_message=error_message,
        )

    async def ingest_address_history(
        self,
        chain: Chain,
        network: Network,
        address: str,
        limit: int = 25,
        offset: int = 0,
        **kwargs,
    ) -> AddressHistoryIngestionResult:
        """Ingest, normalize, and persist transaction history for an address.

        Enforces bounded pagination (maximum 100 transactions per call).
        Applies provider fallback and bounded retries.
        """
        bounded_limit = max(1, min(limit, 100))
        providers = self.registry.get_all(chain)
        if not providers:
            return AddressHistoryIngestionResult(
                success=False,
                chain=chain,
                network=network,
                address=address,
                error_category="UnsupportedOperationError",
                error_message=f"No provider registered for chain {chain.value}",
            )

        primary_provider = providers[0]
        last_error: Optional[Exception] = None
        stop_fallback = False

        for provider in providers:
            if stop_fallback:
                break

            if not provider.supports(ProviderCapability.ADDRESS_HISTORY):
                logger.debug(
                    "[Ingestion] Provider %s does not support ADDRESS_HISTORY for %s, skipping",
                    provider.provider_name,
                    address,
                )
                continue

            fallback_used = provider != primary_provider
            logger.info(
                "[Ingestion] Ingesting address history for %s on %s/%s using provider %s (fallback=%s, limit=%d)",
                address,
                chain.value,
                network.value,
                provider.provider_name,
                fallback_used,
                bounded_limit,
            )

            for attempt in range(self.max_retries + 1):
                try:
                    tx_list = await provider.get_address_transactions(
                        address,
                        limit=bounded_limit,
                        offset=offset,
                        network=network,
                        **kwargs,
                    )

                    results: List[IngestionResult] = []
                    persisted_count = 0
                    existed_count = 0

                    for tx in tx_list:
                        try:
                            # Ensure provenance reflects provider
                            if tx.provenance.provider != provider.provider_name:
                                tx = tx.model_copy(
                                    update={
                                        "provenance": TransactionProvenance(
                                            provider=provider.provider_name,
                                            chain=tx.provenance.chain,
                                            network=tx.provenance.network,
                                            original_id=tx.provenance.original_id,
                                            normalized_at=tx.provenance.normalized_at,
                                        )
                                    }
                                )

                            btc_detail: Optional[BitcoinTransactionDetail] = None
                            if chain == Chain.BITCOIN:
                                from app.schemas.transaction import BitcoinVin, BitcoinVout
                                inputs = [
                                    BitcoinVin(address=t.from_address, value_sat=t.amount)
                                    for t in tx.transfers if t.from_address is not None
                                ]
                                outputs = [
                                    BitcoinVout(n=i, address=t.to_address, value_sat=t.amount)
                                    for i, t in enumerate([t for t in tx.transfers if t.to_address is not None])
                                ]
                                total_in = sum(v.value_sat for v in inputs if v.value_sat is not None) if inputs else None
                                total_out = sum(v.value_sat for v in outputs)
                                btc_detail = BitcoinTransactionDetail(
                                    txid=tx.transaction_id,
                                    inputs=inputs,
                                    outputs=outputs,
                                    total_input_sat=total_in,
                                    total_output_sat=total_out,
                                    fee_sat=tx.fee,
                                )

                            already_existed = False
                            if self.repository is not None:
                                already_existed = await self.repository.transaction_exists(
                                    tx.transaction_id, tx.chain, tx.network
                                )
                                await self.repository.save_transaction(tx, btc_detail)

                            if already_existed:
                                existed_count += 1
                                item_status = IngestionStatus.ALREADY_EXISTS
                            else:
                                persisted_count += 1
                                item_status = IngestionStatus.SUCCESS

                            results.append(
                                IngestionResult(
                                    success=True,
                                    status=item_status,
                                    chain=chain,
                                    network=network,
                                    transaction_id=tx.transaction_id,
                                    provider=provider.provider_name,
                                    fallback_used=fallback_used,
                                    transaction=tx,
                                )
                            )
                        except Exception as item_exc:
                            logger.warning(
                                "[Ingestion] Error normalizing or persisting tx %s in address history for %s: %s",
                                getattr(tx, "transaction_id", "unknown"),
                                address,
                                item_exc,
                            )
                            continue

                    logger.info(
                        "[Ingestion] Address history completed for %s on %s/%s (discovered=%d, persisted=%d, existed=%d, provider=%s)",
                        address,
                        chain.value,
                        network.value,
                        len(tx_list),
                        persisted_count,
                        existed_count,
                        provider.provider_name,
                    )

                    return AddressHistoryIngestionResult(
                        success=True,
                        chain=chain,
                        network=network,
                        address=address,
                        provider=provider.provider_name,
                        fallback_used=fallback_used,
                        transactions_discovered=len(tx_list),
                        transactions_persisted=persisted_count,
                        transactions_already_existed=existed_count,
                        results=results,
                    )

                except RETRYABLE_ERRORS as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Transient error on provider %s for address %s (%s/%s, attempt %d/%d): %s",
                        provider.provider_name,
                        address,
                        chain.value,
                        network.value,
                        attempt + 1,
                        self.max_retries + 1,
                        type(exc).__name__,
                    )
                    if attempt < self.max_retries:
                        delay = min(self.retry_delay * (2**attempt), self.max_delay)
                        await asyncio.sleep(delay)
                        continue
                    logger.info(
                        "[Ingestion] Retries exhausted on provider %s for address %s. Transitioning to fallback.",
                        provider.provider_name,
                        address,
                    )
                    break

                except ProviderAuthError as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Auth failure on provider %s for address %s (%s/%s): %s",
                        provider.provider_name,
                        address,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    if not self.fallback_policy.fallback_on_auth:
                        logger.info(
                            "[Ingestion] Fallback on auth error disabled by policy for provider %s. Stopping fallback.",
                            provider.provider_name,
                        )
                        stop_fallback = True
                    break

                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "[Ingestion] Non-retryable error on provider %s for address %s (%s/%s): %s",
                        provider.provider_name,
                        address,
                        chain.value,
                        network.value,
                        type(exc).__name__,
                    )
                    break

        error_category = type(last_error).__name__ if last_error else "ProviderError"
        error_message = str(last_error) if last_error else "All providers failed"

        logger.error(
            "[Ingestion] Address history failed for %s on %s/%s (error_category=%s): %s",
            address,
            chain.value,
            network.value,
            error_category,
            error_message,
        )

        return AddressHistoryIngestionResult(
            success=False,
            chain=chain,
            network=network,
            address=address,
            error_category=error_category,
            error_message=error_message,
        )
