"""Tests for Step 1E: Ingestion Resilience & Orchestration.

Covers requirements A through N:
A. Primary success
B. Primary timeout -> retry -> success
C. Primary unavailable -> fallback success
D. Primary auth failure (no blind fallback by default; halts or falls back if policy enabled)
E. Primary rate limit -> bounded retry
F. Malformed provider response
G. Transaction not found
H. Fallback also fails (all providers exhausted)
I. Concurrent duplicate ingestion
J. Repeated address-history ingestion
K. Timeout enforcement
L. Secret redaction
M. Provenance correctness
N. Exact integer preservation
"""
import asyncio
from datetime import datetime, timezone
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.repository import TransactionRepository
from app.providers.adapters.bitcoin import BitcoinTransactionAdapter, BlockstreamBitcoinAdapter
from app.providers.adapters.evm import BlockscoutEVMAdapter, EVMTransactionAdapter
from app.providers.adapters.solana import HeliusSolanaAdapter, PublicSolanaRPCAdapter
from app.providers.adapters.tron import TronTransactionAdapter
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
from app.providers.data_registry import DataProviderRegistry
from app.schemas.chain import Chain, Network
from app.schemas.ingestion import IngestionResult, IngestionStatus
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)
from app.services.ingestion_service import FallbackPolicy, IngestionService


def make_sample_tx(
    tx_id: str = "0xsample1",
    provider: str = "primary-mock",
    chain: Chain = Chain.EVM,
    network: Network = Network.ETH_MAINNET,
    native_value: int = 1000000000000000000,
) -> Transaction:
    return Transaction(
        transaction_id=tx_id,
        chain=chain,
        network=network,
        chain_id=1 if chain == Chain.EVM else None,
        block_number=1000,
        block_hash="0xabc",
        timestamp=datetime.now(timezone.utc),
        from_address="0xSenderA",
        to_address="0xReceiverB",
        native_value=native_value,
        native_value_unit="wei" if chain == Chain.EVM else "sat",
        transaction_type=TransactionType.TRANSFER,
        status=TransactionStatus.SUCCESS,
        fee=21000,
        fee_asset="ETH" if chain == Chain.EVM else "BTC",
        transfers=[
            Transfer(
                from_address="0xSenderA",
                to_address="0xReceiverB",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH" if chain == Chain.EVM else "BTC",
                amount=native_value,
                amount_unit="wei" if chain == Chain.EVM else "sat",
                chain=chain,
                network=network,
            )
        ],
        provenance=TransactionProvenance(
            provider=provider,
            chain=chain,
            network=network,
            original_id=tx_id,
            normalized_at=datetime.now(timezone.utc),
        ),
    )


def make_mock_provider(
    name: str,
    chain: Chain = Chain.EVM,
    capabilities: set = None,
) -> MagicMock:
    prov = MagicMock(spec=BaseDataProvider)
    prov.provider_name = name
    prov.supported_chains = [chain]
    caps = capabilities or {ProviderCapability.TRANSACTION_LOOKUP, ProviderCapability.ADDRESS_HISTORY}
    prov.capabilities = caps
    prov.supports = MagicMock(side_effect=lambda c: c in caps)
    return prov


# ---------------------------------------------------------------------------
# A. Primary Success
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_a_primary_success():
    sample = make_sample_tx("0xsuccess", provider="primary")
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(return_value=sample)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    service = IngestionService(registry=reg, max_retries=1)

    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xsuccess")
    assert result.success is True
    assert result.status == IngestionStatus.SUCCESS
    assert result.provider == "primary"
    assert result.fallback_used is False
    assert result.transaction.transaction_id == "0xsuccess"
    assert primary.get_transaction.call_count == 1


# ---------------------------------------------------------------------------
# B. Primary Timeout -> Retry -> Success
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_b_primary_timeout_retry_success():
    sample = make_sample_tx("0xtimeout_retry", provider="primary")
    primary = make_mock_provider("primary")
    # First attempt times out, second attempt succeeds
    primary.get_transaction = AsyncMock(
        side_effect=[ProviderTimeoutError("primary", "Request timed out"), sample]
    )

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    service = IngestionService(registry=reg, max_retries=2, retry_delay=0.01)

    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xtimeout_retry")
    assert result.success is True
    assert result.status == IngestionStatus.SUCCESS
    assert result.provider == "primary"
    assert result.fallback_used is False
    assert primary.get_transaction.call_count == 2


# ---------------------------------------------------------------------------
# C. Primary Unavailable -> Fallback Success
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_c_primary_unavailable_fallback_success():
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=ProviderUnavailableError("primary", "503 Unavailable"))

    fallback_sample = make_sample_tx("0xunavail", provider="fallback")
    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(return_value=fallback_sample)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=1, retry_delay=0.01)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xunavail")

    assert result.success is True
    assert result.status == IngestionStatus.SUCCESS
    assert result.provider == "fallback"
    assert result.fallback_used is True
    # Primary attempted initial + 1 retry = 2 calls
    assert primary.get_transaction.call_count == 2
    assert fallback.get_transaction.call_count == 1


# ---------------------------------------------------------------------------
# D. Primary Auth Failure (No Blind Fallback vs Explicit Policy)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_d1_primary_auth_failure_halts_by_default():
    """By default, auth errors are not blindly converted into fallback."""
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=ProviderAuthError("primary", "Invalid API key (401)"))

    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(return_value=make_sample_tx("0xauth_test", provider="fallback"))

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=1)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xauth_test")

    assert result.success is False
    assert result.status == IngestionStatus.FAILED
    assert result.error_category == "ProviderAuthError"
    # Fallback was NOT called because fallback_on_auth is False by default
    assert fallback.get_transaction.call_count == 0


@pytest.mark.asyncio
async def test_resilience_d2_primary_auth_failure_with_explicit_policy():
    """If fallback_on_auth is explicitly enabled, it falls back."""
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=ProviderAuthError("primary", "Invalid API key (401)"))

    fallback_sample = make_sample_tx("0xauth_test", provider="fallback")
    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(return_value=fallback_sample)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    policy = FallbackPolicy(fallback_on_auth=True)
    service = IngestionService(registry=reg, max_retries=1, fallback_policy=policy)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xauth_test")

    assert result.success is True
    assert result.fallback_used is True
    assert result.provider == "fallback"
    assert fallback.get_transaction.call_count == 1


# ---------------------------------------------------------------------------
# E. Primary Rate Limit -> Bounded Retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_e_primary_rate_limit_bounded_retry():
    sample = make_sample_tx("0xratelimit", provider="primary")
    primary = make_mock_provider("primary")
    # Fails once with 429, then succeeds
    primary.get_transaction = AsyncMock(
        side_effect=[ProviderRateLimitError("primary", "HTTP 429 Rate limited"), sample]
    )

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    service = IngestionService(registry=reg, max_retries=2, retry_delay=0.01)

    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xratelimit")
    assert result.success is True
    assert primary.get_transaction.call_count == 2


# ---------------------------------------------------------------------------
# F. Malformed Provider Response
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_f_malformed_response_transitions_to_fallback():
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=MalformedResponseError("primary", "Invalid JSON shape"))

    fallback_sample = make_sample_tx("0xmalformed", provider="fallback")
    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(return_value=fallback_sample)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=1)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xmalformed")

    assert result.success is True
    assert result.fallback_used is True
    assert result.provider == "fallback"
    # MalformedResponseError is non-retryable on primary: attempt 1 then fallback
    assert primary.get_transaction.call_count == 1
    assert fallback.get_transaction.call_count == 1


# ---------------------------------------------------------------------------
# G. Transaction Not Found
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_g_transaction_not_found_handling():
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=TransactionNotFoundError("primary", "0xmissing"))

    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(side_effect=TransactionNotFoundError("fallback", "0xmissing"))

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=1)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xmissing")

    assert result.success is False
    assert result.status == IngestionStatus.FAILED
    assert result.error_category == "TransactionNotFoundError"
    # Primary not found is non-retryable; attempts fallback, which also reports not found
    assert primary.get_transaction.call_count == 1
    assert fallback.get_transaction.call_count == 1


# ---------------------------------------------------------------------------
# H. Fallback Also Fails (All Providers Exhausted)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_h_fallback_also_fails():
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(side_effect=ProviderUnavailableError("primary", "Primary Down"))

    fallback = make_mock_provider("fallback")
    fallback.get_transaction = AsyncMock(side_effect=ProviderUnavailableError("fallback", "Fallback Down"))

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=1, retry_delay=0.01)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xalldown")

    assert result.success is False
    assert result.status == IngestionStatus.FAILED
    assert result.error_category == "ProviderUnavailableError"
    assert "Fallback Down" in (result.error_message or "")


# ---------------------------------------------------------------------------
# I. Concurrent Duplicate Ingestion (Mock Repository)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_i_concurrent_duplicate_ingestion():
    sample = make_sample_tx("0xconcurrent", provider="primary")
    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(return_value=sample)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)

    # Mock repository simulating database row existence check & save
    saved_records = {}
    mock_repo = MagicMock(spec=TransactionRepository)

    async def mock_exists(tx_id, chain, network):
        return (tx_id, chain, network) in saved_records

    async def mock_save(tx, btc_detail=None):
        saved_records[(tx.transaction_id, tx.chain, tx.network)] = tx

    mock_repo.transaction_exists = AsyncMock(side_effect=mock_exists)
    mock_repo.save_transaction = AsyncMock(side_effect=mock_save)

    service = IngestionService(registry=reg, repository=mock_repo, max_retries=0)

    # Launch 10 simultaneous calls
    tasks = [
        service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xconcurrent")
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        assert not isinstance(r, Exception)
        assert r.success is True
        assert r.status in (IngestionStatus.SUCCESS, IngestionStatus.ALREADY_EXISTS)

    # Exact single stored record in repository
    assert len(saved_records) == 1


# ---------------------------------------------------------------------------
# J. Repeated Address-History Ingestion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_j_repeated_address_history_ingestion():
    tx1 = make_sample_tx("0xhist1", provider="primary")
    tx2 = make_sample_tx("0xhist2", provider="primary")
    primary = make_mock_provider("primary")
    primary.get_address_transactions = AsyncMock(return_value=[tx1, tx2])

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)

    persisted = {}
    mock_repo = MagicMock(spec=TransactionRepository)
    mock_repo.transaction_exists = AsyncMock(side_effect=lambda tid, c, n: (tid, c, n) in persisted)

    async def mock_save(tx, btc_detail=None):
        persisted[(tx.transaction_id, tx.chain, tx.network)] = tx

    mock_repo.save_transaction = AsyncMock(side_effect=mock_save)

    service = IngestionService(registry=reg, repository=mock_repo)

    # 1st run: 2 persisted, 0 existed
    res1 = await service.ingest_address_history(Chain.EVM, Network.ETH_MAINNET, "0xAddressA", limit=25)
    assert res1.success is True
    assert res1.transactions_discovered == 2
    assert res1.transactions_persisted == 2
    assert res1.transactions_already_existed == 0

    # 2nd run: 0 persisted, 2 existed (idempotent)
    res2 = await service.ingest_address_history(Chain.EVM, Network.ETH_MAINNET, "0xAddressA", limit=25)
    assert res2.success is True
    assert res2.transactions_discovered == 2
    assert res2.transactions_persisted == 0
    assert res2.transactions_already_existed == 2


# ---------------------------------------------------------------------------
# K. Timeout Enforcement Across Adapters
# ---------------------------------------------------------------------------
def test_resilience_k_timeout_enforcement():
    """Verify all adapters configure explicit HTTP timeouts."""
    import app.providers.adapters.evm as evm_mod
    import app.providers.adapters.bitcoin as btc_mod
    import app.providers.adapters.tron as tron_mod
    import app.providers.adapters.solana as sol_mod

    assert getattr(evm_mod, "_TIMEOUT", None) == 15.0
    assert getattr(btc_mod, "_TIMEOUT", None) == 15.0
    assert getattr(tron_mod, "_TIMEOUT", None) == 15.0
    assert getattr(sol_mod, "PublicSolanaRPCAdapter")._TIMEOUT == 15.0
    assert getattr(evm_mod, "BlockscoutEVMAdapter")._TIMEOUT == 15.0


# ---------------------------------------------------------------------------
# L. Secret Redaction
# ---------------------------------------------------------------------------
def test_resilience_l_secret_redaction():
    secret_key = "very_secret_api_key_xyz123"
    adapter = EVMTransactionAdapter(api_key=secret_key)

    # Check repr does not contain secret
    assert secret_key not in repr(adapter)
    assert secret_key not in str(adapter)

    # Check error message does not contain secret
    err = ProviderAuthError("etherscan", "Invalid key provided")
    assert secret_key not in str(err)
    assert secret_key not in repr(err)


# ---------------------------------------------------------------------------
# M. Provenance Correctness
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_m_provenance_correctness():
    primary = make_mock_provider("primary-failing")
    primary.get_transaction = AsyncMock(side_effect=ProviderUnavailableError("primary-failing", "Down"))

    # The fallback transaction might initially have any provenance, but service must enforce fallback provider
    raw_tx = make_sample_tx("0xprov", provider="fallback-succeeding")
    fallback = make_mock_provider("fallback-succeeding")
    fallback.get_transaction = AsyncMock(return_value=raw_tx)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    reg.register(Chain.EVM, fallback, primary=False)

    service = IngestionService(registry=reg, max_retries=0)
    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xprov")

    assert result.success is True
    assert result.provider == "fallback-succeeding"
    assert result.fallback_used is True
    assert result.transaction.provenance.provider == "fallback-succeeding"
    assert result.transaction.provenance.original_id == "0xprov"
    assert result.transaction.provenance.chain == Chain.EVM
    assert result.transaction.provenance.network == Network.ETH_MAINNET


# ---------------------------------------------------------------------------
# N. Exact Integer Preservation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilience_n_exact_integer_preservation():
    # Huge integer > 2^53 - 1 (JavaScript safe integer limit)
    huge_int = 99999999999999999999999999999999999999999999999
    huge_tx = make_sample_tx("0xhuge", provider="primary", native_value=huge_int)

    primary = make_mock_provider("primary")
    primary.get_transaction = AsyncMock(return_value=huge_tx)

    reg = DataProviderRegistry()
    reg.register(Chain.EVM, primary, primary=True)
    service = IngestionService(registry=reg)

    result = await service.ingest_transaction(Chain.EVM, Network.ETH_MAINNET, "0xhuge")
    assert result.success is True
    assert isinstance(result.transaction.native_value, int)
    assert result.transaction.native_value == huge_int
    assert result.transaction.transfers[0].amount == huge_int
    assert isinstance(result.transaction.transfers[0].amount, int)
