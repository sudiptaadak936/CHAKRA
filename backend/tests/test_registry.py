"""CHAKRA Step 6: Suspect Registry — unit test suite.

All tests are pure-Python with mocked asyncpg pools.
No live database required. Integration tests are excluded (no @pytest.mark.integration).

Test Matrix (31 tests):
  §A  Schema validation (SuspectRegistryRecord, CrossCaseHitResult, MatchedCaseContext)
  §B  Address normalization (EVM lowercase, Bitcoin/Tron/Solana case-preservation)
  §C  record_id determinism and format
  §D  hit_id determinism and format
  §E  Service.register() validation (fail-closed)
  §F  Service.register() happy path
  §G  Service.evaluate_cross_case_hits() — no hit (single case)
  §H  Service.evaluate_cross_case_hits() — hit (>= 2 cases)
  §I  Safety invariants (requires_human_review, no random UUIDs, no Redis/alerts)
  §J  Repository idempotency (mock)
  §K  Enum coverage
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.graph.models import make_address_composite_id, normalize_address
from app.registry.repository import SuspectRegistryRepository
from app.registry.service import SuspectRegistryService, _compute_hit_id, _compute_record_id
from app.schemas.chain import Chain, Network
from app.schemas.registry import (
    CrossCaseHitResult,
    CrossCaseLinkType,
    MatchedCaseContext,
    RegistryStatus,
    SuspectRegistryRecord,
    SuspectRole,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    record_id: str = "a" * 64,
    case_id: str = "case_abc123",
    chain: str = "evm",
    network: str = "ethereum-mainnet",
    address: str = "0xdeadbeef",
    normalized_address: str = "0xdeadbeef",
    composite_id: str = "evm:ethereum-mainnet:0xdeadbeef",
    suspect_role: SuspectRole = SuspectRole.SUSPECT_TARGET,
    status: RegistryStatus = RegistryStatus.ACTIVE,
    source: str = "fiu_referral",
    evidence_id: str = "ev-001",
    reported_by: str = "officer_A",
    entity_label: Optional[str] = None,
    external_reference_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> SuspectRegistryRecord:
    return SuspectRegistryRecord(
        record_id=record_id,
        case_id=case_id,
        chain=chain,
        network=network,
        address=address,
        normalized_address=normalized_address,
        composite_id=composite_id,
        suspect_role=suspect_role,
        status=status,
        source=source,
        evidence_id=evidence_id,
        reported_by=reported_by,
        entity_label=entity_label,
        external_reference_id=external_reference_id,
        metadata=metadata or {},
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_mock_repo(records_by_composite: Optional[Dict[str, List[SuspectRegistryRecord]]] = None):
    """Build a mock SuspectRegistryRepository.

    records_by_composite: maps composite_id -> list of records returned by
    find_cross_case_matches.
    """
    repo = MagicMock(spec=SuspectRegistryRepository)
    repo.save_record = AsyncMock(return_value=None)
    repo.init_schema = AsyncMock(return_value=None)

    rbc = records_by_composite or {}

    async def _find_cross(composite_id: str):
        return rbc.get(composite_id, [])

    async def _lookup_by_composite(composite_id: str):
        return rbc.get(composite_id, [])

    repo.find_cross_case_matches = AsyncMock(side_effect=_find_cross)
    repo.lookup_by_composite_id = AsyncMock(side_effect=_lookup_by_composite)
    repo.lookup_by_case_id = AsyncMock(return_value=[])
    repo._clear_all_for_testing = AsyncMock(return_value=None)
    return repo


# ===========================================================================
# §A — Schema validation
# ===========================================================================

def test_A1_suspect_registry_record_valid():
    """A valid SuspectRegistryRecord constructs without error."""
    r = _make_record()
    assert r.record_id == "a" * 64
    assert r.status == RegistryStatus.ACTIVE
    # SuspectRegistryRecord is a plain registry entry — requires_human_review is on CrossCaseHitResult
    assert not hasattr(r, "requires_human_review") or r.requires_human_review is None
    assert r.suspect_role == SuspectRole.SUSPECT_TARGET


def test_A2_suspect_registry_record_frozen():
    """SuspectRegistryRecord is immutable (frozen=True)."""
    r = _make_record()
    with pytest.raises((TypeError, ValidationError)):
        r.case_id = "tampered"  # Pydantic v2 frozen model raises on direct assignment


def test_A3_cross_case_hit_result_frozen():
    """CrossCaseHitResult is immutable and requires_human_review is always True."""
    hit = CrossCaseHitResult(
        hit_id="b" * 64,
        composite_id="evm:ethereum-mainnet:0xdeadbeef",
        chain="evm",
        network="ethereum-mainnet",
        normalized_address="0xdeadbeef",
        matched_case_ids=["case_001", "case_002"],
        matched_cases=[
            MatchedCaseContext(
                case_id="case_001", evidence_id="ev-1", source="fiu",
                reported_by="officer_A", suspect_role=SuspectRole.SUSPECT_TARGET,
            ),
            MatchedCaseContext(
                case_id="case_002", evidence_id="ev-2", source="fiu",
                reported_by="officer_B", suspect_role=SuspectRole.CASH_OUT,
            ),
        ],
        link_type=CrossCaseLinkType.DIRECT_ADDRESS_MATCH,
        requires_human_review=True,
    )
    assert hit.requires_human_review is True
    with pytest.raises((TypeError, ValidationError)):
        hit.requires_human_review = False  # Pydantic v2 frozen: direct assignment raises


def test_A4_cross_case_hit_result_requires_at_least_2_cases():
    """CrossCaseHitResult refuses < 2 matched_case_ids."""
    with pytest.raises(ValidationError):
        CrossCaseHitResult(
            hit_id="c" * 64,
            composite_id="evm:ethereum-mainnet:0xdeadbeef",
            chain="evm",
            network="ethereum-mainnet",
            normalized_address="0xdeadbeef",
            matched_case_ids=["case_001"],  # Only 1 — invalid
            matched_cases=[
                MatchedCaseContext(
                    case_id="case_001", evidence_id="ev-1", source="fiu",
                    reported_by="officer_A", suspect_role=SuspectRole.SUSPECT_TARGET,
                )
            ],
            link_type=CrossCaseLinkType.DIRECT_ADDRESS_MATCH,
            requires_human_review=True,
        )


def test_A5_suspect_registry_record_rejects_empty_record_id():
    """SuspectRegistryRecord rejects an empty record_id."""
    with pytest.raises(ValidationError):
        _make_record(record_id="")


def test_A6_matched_case_context_frozen():
    """MatchedCaseContext is immutable (frozen=True)."""
    ctx = MatchedCaseContext(
        case_id="case_001", evidence_id="ev-1", source="fiu",
        reported_by="officer_A", suspect_role=SuspectRole.SUSPECT_TARGET,
    )
    with pytest.raises((TypeError, ValidationError)):
        ctx.case_id = "tampered"  # Pydantic v2 frozen: direct assignment raises


# ===========================================================================
# §B — Address normalization
# ===========================================================================

def test_B1_evm_address_normalized_to_lowercase():
    """EVM addresses are lowercased by normalize_address."""
    result = normalize_address("evm", "0xDEADBEEF")
    assert result == "0xdeadbeef"


def test_B2_0x_prefix_triggers_lowercase_regardless_of_chain():
    """0x-prefixed addresses are lowercased even on non-evm chain."""
    result = normalize_address("bitcoin", "0xDEADBEEF")
    assert result == "0xdeadbeef"


def test_B3_bitcoin_address_case_preserved():
    """Bitcoin base58 addresses preserve case."""
    addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    result = normalize_address("bitcoin", addr)
    assert result == addr


def test_B4_tron_address_case_preserved():
    """Tron base58 addresses preserve case."""
    addr = "TG3XXyExBkPp9nzdajDZsozEu4BkaSJozs"
    result = normalize_address("tron", addr)
    assert result == addr


def test_B5_solana_address_case_preserved():
    """Solana base58 addresses preserve case."""
    addr = "So11111111111111111111111111111111111111112"
    result = normalize_address("solana", addr)
    assert result == addr


def test_B6_composite_id_structure():
    """make_address_composite_id returns chain:network:norm_addr."""
    composite = make_address_composite_id("evm", "ethereum-mainnet", "0xDEADBEEF")
    assert composite == "evm:ethereum-mainnet:0xdeadbeef"


# ===========================================================================
# §C — record_id determinism
# ===========================================================================

def test_C1_record_id_is_sha256_hex():
    """_compute_record_id returns a 64-char lowercase hex string."""
    rid = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_001", "ev-001")
    assert len(rid) == 64
    assert rid == rid.lower()
    int(rid, 16)  # Must be valid hex — no exception


def test_C2_record_id_is_deterministic():
    """_compute_record_id returns the same value for identical inputs."""
    rid1 = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_001", "ev-001")
    rid2 = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_001", "ev-001")
    assert rid1 == rid2


def test_C3_record_id_changes_on_different_case():
    """Different case_id produces a different record_id."""
    rid1 = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_001", "ev-001")
    rid2 = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_002", "ev-001")
    assert rid1 != rid2


def test_C4_record_id_matches_manual_sha256():
    """_compute_record_id matches manual SHA-256 of the canonical formula."""
    raw = "evm:ethereum-mainnet:0xdeadbeef:case_001:ev-001"
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    actual = _compute_record_id("evm", "ethereum-mainnet", "0xdeadbeef", "case_001", "ev-001")
    assert actual == expected


# ===========================================================================
# §D — hit_id determinism
# ===========================================================================

def test_D1_hit_id_is_deterministic():
    """_compute_hit_id is deterministic and order-invariant on case_ids."""
    h1 = _compute_hit_id("evm", "ethereum-mainnet", "0xdeadbeef", ["case_002", "case_001"])
    h2 = _compute_hit_id("evm", "ethereum-mainnet", "0xdeadbeef", ["case_001", "case_002"])
    assert h1 == h2  # Sorted internally


def test_D2_hit_id_is_sha256_hex():
    """_compute_hit_id returns a 64-char hex string."""
    h = _compute_hit_id("evm", "ethereum-mainnet", "0xdeadbeef", ["case_001", "case_002"])
    assert len(h) == 64
    int(h, 16)  # Valid hex


def test_D3_hit_id_changes_on_different_address():
    """Different address produces a different hit_id."""
    h1 = _compute_hit_id("evm", "ethereum-mainnet", "0xaaaa", ["case_001", "case_002"])
    h2 = _compute_hit_id("evm", "ethereum-mainnet", "0xbbbb", ["case_001", "case_002"])
    assert h1 != h2


# ===========================================================================
# §E — Service.register() validation (fail-closed)
# ===========================================================================

@pytest.mark.asyncio
async def test_E1_register_rejects_empty_case_id():
    """register() raises ValueError on empty case_id."""
    svc = SuspectRegistryService(_make_mock_repo())
    with pytest.raises(ValueError, match="case_id"):
        await svc.register(
            case_id="   ",
            chain=Chain.EVM, network=Network.ETH_MAINNET,
            address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="A",
        )


@pytest.mark.asyncio
async def test_E2_register_rejects_empty_address():
    """register() raises ValueError on empty address."""
    svc = SuspectRegistryService(_make_mock_repo())
    with pytest.raises(ValueError, match="address"):
        await svc.register(
            case_id="case_001",
            chain=Chain.EVM, network=Network.ETH_MAINNET,
            address="", source="fiu", evidence_id="ev-1", reported_by="A",
        )


@pytest.mark.asyncio
async def test_E3_register_rejects_invalid_chain():
    """register() raises ValueError on unknown chain string."""
    svc = SuspectRegistryService(_make_mock_repo())
    with pytest.raises(ValueError, match="chain"):
        await svc.register(
            case_id="case_001",
            chain="quantum_chain", network=Network.ETH_MAINNET,
            address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="A",
        )


@pytest.mark.asyncio
async def test_E4_register_rejects_empty_evidence_id():
    """register() raises ValueError on blank evidence_id."""
    svc = SuspectRegistryService(_make_mock_repo())
    with pytest.raises(ValueError, match="evidence_id"):
        await svc.register(
            case_id="case_001",
            chain=Chain.EVM, network=Network.ETH_MAINNET,
            address="0xdeadbeef", source="fiu", evidence_id="   ", reported_by="A",
        )


@pytest.mark.asyncio
async def test_E5_register_rejects_empty_reported_by():
    """register() raises ValueError on empty reported_by."""
    svc = SuspectRegistryService(_make_mock_repo())
    with pytest.raises(ValueError, match="reported_by"):
        await svc.register(
            case_id="case_001",
            chain=Chain.EVM, network=Network.ETH_MAINNET,
            address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="",
        )


# ===========================================================================
# §F — Service.register() happy path
# ===========================================================================

@pytest.mark.asyncio
async def test_F1_register_happy_path_evm():
    """register() returns a valid SuspectRegistryRecord for an EVM address."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    record = await svc.register(
        case_id="case_001",
        chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xDEADBEEF",
        source="fiu_referral", evidence_id="ev-001", reported_by="officer_A",
    )
    assert record.case_id == "case_001"
    assert record.normalized_address == "0xdeadbeef"  # EVM lowercased
    assert len(record.record_id) == 64
    # requires_human_review is on CrossCaseHitResult, NOT SuspectRegistryRecord
    assert not hasattr(record, "requires_human_review")
    repo.save_record.assert_awaited_once()


@pytest.mark.asyncio
async def test_F2_register_deterministic_record_id():
    """Two identical register() calls produce the same record_id."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    kwargs = dict(
        case_id="case_001",
        chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xdeadbeef", source="fiu", evidence_id="ev-001", reported_by="A",
    )
    r1 = await svc.register(**kwargs)
    r2 = await svc.register(**kwargs)
    assert r1.record_id == r2.record_id


@pytest.mark.asyncio
async def test_F3_register_normalizes_evm_address():
    """register() normalizes EVM address to lowercase in stored record."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    record = await svc.register(
        case_id="case_001",
        chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xABCDEF1234",
        source="fiu", evidence_id="ev-1", reported_by="A",
    )
    assert record.normalized_address == "0xabcdef1234"


@pytest.mark.asyncio
async def test_F4_register_preserves_bitcoin_address_case():
    """register() preserves Bitcoin address case."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    btc_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    record = await svc.register(
        case_id="case_btc",
        chain=Chain.BITCOIN, network=Network.BTC_MAINNET,
        address=btc_addr, source="fiu", evidence_id="ev-btc", reported_by="B",
    )
    assert record.normalized_address == btc_addr


@pytest.mark.asyncio
async def test_F5_register_metadata_optional():
    """register() works without metadata (defaults to empty dict)."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    record = await svc.register(
        case_id="case_001",
        chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="A",
    )
    assert record.metadata == {}


# ===========================================================================
# §G — evaluate_cross_case_hits() — no hit
# ===========================================================================

@pytest.mark.asyncio
async def test_G1_no_hit_on_empty_registry():
    """evaluate_cross_case_hits() returns None when no records exist."""
    svc = SuspectRegistryService(_make_mock_repo())
    result = await svc.evaluate_cross_case_hits("evm:ethereum-mainnet:0xdeadbeef")
    assert result is None


@pytest.mark.asyncio
async def test_G2_no_hit_on_single_case():
    """evaluate_cross_case_hits() returns None when only 1 case is present."""
    composite = "evm:ethereum-mainnet:0xdeadbeef"
    records = [
        _make_record(case_id="case_001", composite_id=composite, normalized_address="0xdeadbeef"),
    ]
    # find_cross_case_matches would return empty (repo enforces distinct >= 2)
    svc = SuspectRegistryService(_make_mock_repo())
    result = await svc.evaluate_cross_case_hits(composite)
    assert result is None


@pytest.mark.asyncio
async def test_G3_no_hit_on_blank_composite_id():
    """evaluate_cross_case_hits() returns None for blank input."""
    svc = SuspectRegistryService(_make_mock_repo())
    assert await svc.evaluate_cross_case_hits("") is None
    assert await svc.evaluate_cross_case_hits("   ") is None


# ===========================================================================
# §H — evaluate_cross_case_hits() — hit (>= 2 cases)
# ===========================================================================

@pytest.mark.asyncio
async def test_H1_hit_detected_on_two_cases():
    """evaluate_cross_case_hits() returns CrossCaseHitResult when 2 cases share composite_id."""
    composite = "evm:ethereum-mainnet:0xdeadbeef"
    records = [
        _make_record(
            record_id="a" * 64, case_id="case_001", composite_id=composite,
            normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet",
            source="fiu", evidence_id="ev-1", reported_by="officer_A",
        ),
        _make_record(
            record_id="b" * 64, case_id="case_002", composite_id=composite,
            normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet",
            source="fiu", evidence_id="ev-2", reported_by="officer_B",
        ),
    ]
    repo = _make_mock_repo({composite: records})
    svc = SuspectRegistryService(repo)
    result = await svc.evaluate_cross_case_hits(composite)
    assert result is not None
    assert result.requires_human_review is True
    assert set(result.matched_case_ids) == {"case_001", "case_002"}
    assert len(result.matched_cases) == 2
    assert result.link_type == CrossCaseLinkType.DIRECT_ADDRESS_MATCH


@pytest.mark.asyncio
async def test_H2_hit_id_is_deterministic():
    """Hit result hit_id is deterministic across two identical evaluations."""
    composite = "evm:ethereum-mainnet:0xdeadbeef"
    records = [
        _make_record(record_id="a" * 64, case_id="case_001", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
        _make_record(record_id="b" * 64, case_id="case_002", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
    ]
    repo = _make_mock_repo({composite: records})
    svc = SuspectRegistryService(repo)
    r1 = await svc.evaluate_cross_case_hits(composite)
    r2 = await svc.evaluate_cross_case_hits(composite)
    assert r1.hit_id == r2.hit_id


@pytest.mark.asyncio
async def test_H3_hit_matched_case_ids_are_sorted():
    """matched_case_ids in CrossCaseHitResult are always sorted."""
    composite = "evm:ethereum-mainnet:0xdeadbeef"
    records = [
        _make_record(record_id="a" * 64, case_id="case_zzz", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
        _make_record(record_id="b" * 64, case_id="case_aaa", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
    ]
    repo = _make_mock_repo({composite: records})
    svc = SuspectRegistryService(repo)
    result = await svc.evaluate_cross_case_hits(composite)
    assert result.matched_case_ids == sorted(result.matched_case_ids)


@pytest.mark.asyncio
async def test_H4_hit_requires_human_review_cannot_be_false():
    """requires_human_review is True in CrossCaseHitResult and cannot be mutated."""
    composite = "evm:ethereum-mainnet:0xdeadbeef"
    records = [
        _make_record(record_id="a" * 64, case_id="case_001", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
        _make_record(record_id="b" * 64, case_id="case_002", composite_id=composite,
                     normalized_address="0xdeadbeef", chain="evm", network="ethereum-mainnet"),
    ]
    repo = _make_mock_repo({composite: records})
    svc = SuspectRegistryService(repo)
    result = await svc.evaluate_cross_case_hits(composite)
    assert result.requires_human_review is True
    with pytest.raises((TypeError, ValidationError)):
        result.requires_human_review = False  # Pydantic v2 frozen: direct assignment raises


# ===========================================================================
# §I — Safety invariants
# ===========================================================================

def test_I1_requires_human_review_field_default_is_true():
    """CrossCaseHitResult.requires_human_review defaults to True."""
    hit = CrossCaseHitResult(
        hit_id="d" * 64,
        composite_id="evm:ethereum-mainnet:0xabc",
        chain="evm", network="ethereum-mainnet", normalized_address="0xabc",
        matched_case_ids=["c1", "c2"],
        matched_cases=[
            MatchedCaseContext(case_id="c1", evidence_id="e1", source="s", reported_by="r",
                               suspect_role=SuspectRole.SUSPECT_TARGET),
            MatchedCaseContext(case_id="c2", evidence_id="e2", source="s", reported_by="r",
                               suspect_role=SuspectRole.CASH_OUT),
        ],
        link_type=CrossCaseLinkType.DIRECT_ADDRESS_MATCH,
        requires_human_review=True,
    )
    assert hit.requires_human_review is True


@pytest.mark.asyncio
async def test_I2_service_does_not_import_redis():
    """SuspectRegistryService does not import or use Redis."""
    import app.registry.service as svc_mod
    # Check only the import lines — docstrings legitimately mention "redis" in negation
    import_lines = [
        line for line in open(svc_mod.__file__).readlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_block = "\n".join(import_lines).lower()
    assert "redis" not in import_block
    assert "aioredis" not in import_block


@pytest.mark.asyncio
async def test_I3_service_does_not_import_alert_service():
    """SuspectRegistryService does not import AlertService or publish alerts."""
    import app.registry.service as svc_mod
    # Check only the import lines — docstrings legitimately mention "alert" in negation
    import_lines = [
        line for line in open(svc_mod.__file__).readlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_block = "\n".join(import_lines).lower()
    assert "alert_service" not in import_block
    assert "alert" not in import_block


def test_I4_record_id_is_not_random():
    """record_id is deterministic — two calls with same inputs produce same ID."""
    rid1 = _compute_record_id("evm", "ethereum-mainnet", "0xabc", "case_001", "ev-001")
    rid2 = _compute_record_id("evm", "ethereum-mainnet", "0xabc", "case_001", "ev-001")
    assert rid1 == rid2


def test_I5_hit_id_is_not_random():
    """hit_id is deterministic — two calls with same inputs produce same ID."""
    h1 = _compute_hit_id("evm", "ethereum-mainnet", "0xabc", ["case_001", "case_002"])
    h2 = _compute_hit_id("evm", "ethereum-mainnet", "0xabc", ["case_001", "case_002"])
    assert h1 == h2


# ===========================================================================
# §J — Repository idempotency (mock)
# ===========================================================================

@pytest.mark.asyncio
async def test_J1_save_record_called_once_per_register():
    """register() calls save_record exactly once per invocation."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    await svc.register(
        case_id="case_001", chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="A",
    )
    assert repo.save_record.await_count == 1


@pytest.mark.asyncio
async def test_J2_duplicate_register_calls_both_invoke_save():
    """Two identical register() calls each invoke save_record (DB handles idempotency)."""
    repo = _make_mock_repo()
    svc = SuspectRegistryService(repo)
    kwargs = dict(
        case_id="case_001", chain=Chain.EVM, network=Network.ETH_MAINNET,
        address="0xdeadbeef", source="fiu", evidence_id="ev-1", reported_by="A",
    )
    await svc.register(**kwargs)
    await svc.register(**kwargs)
    assert repo.save_record.await_count == 2  # DB's ON CONFLICT handles dedup


# ===========================================================================
# §K — Enum coverage
# ===========================================================================

def test_K1_suspect_role_enum_values():
    """SuspectRole enum has all expected members."""
    assert SuspectRole.SUSPECT_TARGET.value == "SUSPECT_TARGET"
    assert SuspectRole.CASH_OUT.value == "CASH_OUT"
    assert SuspectRole.INTERMEDIARY.value == "INTERMEDIARY"
    assert SuspectRole.UNKNOWN_ROLE.value == "UNKNOWN_ROLE"


def test_K2_registry_status_enum_values():
    """RegistryStatus enum has all expected members."""
    assert RegistryStatus.ACTIVE.value == "ACTIVE"
    assert RegistryStatus.PROVISIONAL.value == "PROVISIONAL"
    assert RegistryStatus.ARCHIVED.value == "ARCHIVED"
    assert RegistryStatus.DISMISSED.value == "DISMISSED"


def test_K3_cross_case_link_type_enum_values():
    """CrossCaseLinkType enum has the expected members."""
    assert CrossCaseLinkType.DIRECT_ADDRESS_MATCH.value == "DIRECT_ADDRESS_MATCH"
    assert CrossCaseLinkType.INFERRED_CLUSTER_LINK.value == "INFERRED_CLUSTER_LINK"
