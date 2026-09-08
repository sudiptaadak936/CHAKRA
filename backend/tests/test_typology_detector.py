"""CHAKRA Step 3G: Typology Detection Test Suite.

Comprehensive unit and behavioral test suite covering:
1. Peel chain:
   - 3-hop genuine peel pattern -> positive
   - 3 CHANGE_CANDIDATE hops without peel/remainder evidence -> negative
   - insufficient peel hops -> negative
   - missing/incompatible amounts -> conservative behavior
   - peel evidence provenance (ratios, amounts, addresses)
   - non-change candidate hop breaking sequence -> negative
2. Fan-in:
   - exactly 3 sources within temporal window -> positive
   - 3 historical sources outside window -> negative
   - source addresses preserved
   - transaction IDs preserved
   - below threshold (< 3) -> negative
   - missing timestamps -> negative
3. Fan-out:
   - exactly 3 destinations within temporal window -> positive
   - 3 historical destinations outside window -> negative
   - destination addresses preserved
   - transaction IDs preserved
   - below threshold (< 3) -> negative
4. Rapid hopping:
   - exact boundary time -> positive
   - exceeding threshold -> negative
   - missing timestamps -> negative
5. Safety:
   - mixer safety (insufficient_evidence)
   - cross-chain safety (insufficient_evidence)
   - overlapping typologies on same path
   - read-only enforcement
   - deterministic ID generation
   - repeatability
   - empty path safety
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.forensics.models import ChangeAddressInference
from app.forensics.typology_detector import (
    TypologyDetector,
    TypologyType,
    MIN_PEEL_CHAIN_HOPS,
    MIN_FAN_IN_SOURCES,
    MIN_FAN_OUT_DESTINATIONS,
    MAX_RAPID_HOP_INTERVAL_MINUTES,
    MIN_RAPID_HOPS,
    DEFAULT_FAN_WINDOW_HOURS,
)
from app.graph.traversal import TraversalPath, TraversalNode


# ---------------------------------------------------------------------------
# Test Helpers & Dummies
# ---------------------------------------------------------------------------

class DummyTx:
    def __init__(self, txid: str, timestamp: str | None = None):
        self.transaction_id = txid
        self.timestamp = timestamp


class DummyOutput:
    def __init__(self, address: str, value_sat: int | None = None):
        self.address = address
        self.value_sat = value_sat


class DummyInput:
    def __init__(self, value_sat: int | None = None):
        self.value_sat = value_sat


class DummyUTXOStep:
    def __init__(
        self,
        txid: str,
        out_addr: str,
        timestamp: str | None = None,
        cont_sat: int | None = None,
        all_outputs: list | None = None,
    ):
        self.transaction = DummyTx(txid, timestamp)
        self.created_output = DummyOutput(out_addr, cont_sat)
        self.spent_input = DummyInput(cont_sat)
        self.all_outputs = all_outputs


def _make_node(addr: str, chain: str = "bitcoin") -> TraversalNode:
    return TraversalNode(
        composite_id=f"{chain}:mainnet:{addr}",
        chain=chain,
        network="mainnet",
        normalized_address=addr.lower(),
        raw_address=addr,
    )


def _make_pool_conn() -> tuple[MagicMock, AsyncMock]:
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    conn.fetch.return_value = []

    write_calls: list = []

    async def _track_execute(*a, **kw):
        write_calls.append(a)

    conn.execute = _track_execute
    conn.executemany = _track_execute

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire.return_value = ctx
    pool.write_calls = write_calls

    return pool, conn


def _make_repo(change_map: dict | None = None) -> AsyncMock:
    repo = AsyncMock()
    _map = change_map or {}

    async def _get_inferences(txid: str):
        return _map.get(txid, [])

    repo.get_change_inferences.side_effect = _get_inferences
    return repo


def _change_candidate(txid: str, address: str) -> ChangeAddressInference:
    return ChangeAddressInference(
        txid=txid,
        output_index=0,
        address=address,
        classification="CHANGE_CANDIDATE",
        confidence="HIGH",
        evidence_reason="Step 3D change candidate inference",
    )


def _make_bitcoin_peel_path(
    n_hops: int,
    *,
    has_peel_outputs: bool = True,
    has_amounts: bool = True,
    cont_sat: int = 950_000_000,
    peeled_sat: int = 50_000_000,
    timestamps: list | None = None,
) -> tuple[TraversalPath, dict]:
    nodes = [_make_node(f"addr{i}", chain="bitcoin") for i in range(n_hops + 1)]
    steps = []
    change_map = {}
    for i in range(n_hops):
        txid = f"tx{i+1}"
        cont_addr = f"addr{i+1}"
        ts = timestamps[i] if timestamps else None

        if has_peel_outputs:
            peel_addr = f"peel_addr{i+1}"
            all_outs = [
                {"output_index": 0, "address": peel_addr, "value_sat": peeled_sat if has_amounts else None},
                {"output_index": 1, "address": cont_addr, "value_sat": cont_sat if has_amounts else None},
            ]
        else:
            # 1-in-1-out self-sweep without separate peel output
            all_outs = [
                {"output_index": 0, "address": cont_addr, "value_sat": cont_sat if has_amounts else None}
            ]

        step = DummyUTXOStep(
            txid,
            cont_addr,
            timestamp=ts,
            cont_sat=cont_sat if has_amounts else None,
            all_outputs=all_outs,
        )
        steps.append(step)
        change_map[txid] = [_change_candidate(txid, cont_addr)]

    path = TraversalPath(nodes=nodes, hops=n_hops)
    path.utxo_steps = steps
    return path, change_map


# ---------------------------------------------------------------------------
# 1. PEEL_CHAIN Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_01_peel_chain_genuine_3_hops_positive():
    """Test 1: 3-hop genuine peel pattern with change remainder and peeled outputs -> positive."""
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        cont_sat=950_000_000,
        peeled_sat=50_000_000,
    )
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    peel = next((r for r in results if r.typology_type == TypologyType.PEEL_CHAIN), None)

    assert peel is not None, "Genuine peel chain must produce a positive PEEL_CHAIN detection"
    assert peel.confidence_level == "observed"
    assert peel.hop_count == MIN_PEEL_CHAIN_HOPS
    assert len(peel.counterparty_addresses) == MIN_PEEL_CHAIN_HOPS


@pytest.mark.asyncio
async def test_02_peel_chain_change_without_peel_remainder_negative():
    """Test 2: 3 CHANGE_CANDIDATE hops without separate peeled outputs (1-in-1-out sweeps) -> negative."""
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        has_peel_outputs=False,  # Single-output sweep
    )
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.PEEL_CHAIN for r in results), \
        "Single-output transactions without separate peeled output must NOT qualify as peel chain"


@pytest.mark.asyncio
async def test_03_peel_chain_insufficient_hops_negative():
    """Test 3: Fewer than MIN_PEEL_CHAIN_HOPS (2 hops) -> negative."""
    path, change_map = _make_bitcoin_peel_path(MIN_PEEL_CHAIN_HOPS - 1)
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.PEEL_CHAIN for r in results)


@pytest.mark.asyncio
async def test_04_peel_chain_missing_incompatible_amounts_conservative():
    """Test 4: 3 hops with change candidates but missing value amounts -> negative (no fabrication)."""
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        has_amounts=False,  # Missing amounts
    )
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.PEEL_CHAIN for r in results), \
        "Missing amounts must prevent unsupported peel chain classification"


@pytest.mark.asyncio
async def test_05_peel_chain_evidence_provenance():
    """Test 5: Positive peel chain preserves detailed evidence breakdown."""
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        cont_sat=900_000_000,
        peeled_sat=100_000_000,
    )
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    peel = next(r for r in results if r.typology_type == TypologyType.PEEL_CHAIN)

    assert len(peel.evidence) == 1
    ev = peel.evidence[0]
    assert ev.peel_details is not None
    assert len(ev.peel_details) == MIN_PEEL_CHAIN_HOPS

    hop0 = ev.peel_details[0]
    assert hop0["continuing_amount_sat"] == 900_000_000
    assert hop0["peeled_amount_sat"] == 100_000_000
    assert hop0["retention_ratio"] == 0.90
    assert hop0["peel_ratio"] == 0.10
    assert hop0["step_3d_classification"] == "CHANGE_CANDIDATE"


@pytest.mark.asyncio
async def test_06_peel_chain_broken_by_non_change_hop():
    """Test 6: Middle hop is not a CHANGE_CANDIDATE -> negative."""
    path, change_map = _make_bitcoin_peel_path(MIN_PEEL_CHAIN_HOPS)
    # Break hop 2
    change_map["tx2"] = [
        ChangeAddressInference(
            txid="tx2",
            output_index=0,
            address="addr2",
            classification="EXTERNAL_RECIPIENT",
            confidence="LOW",
            evidence_reason="not change",
        )
    ]
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.PEEL_CHAIN for r in results)


# ---------------------------------------------------------------------------
# 2. FAN_IN Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_07_fan_in_exact_3_sources_within_window_positive():
    """Test 7: Exactly 3 distinct upstream sources within 24h window -> positive."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo, max_fan_window_hours=24.0)

    t0 = datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "src_1", "transaction_id": "tx_in_1", "amount": 100, "amount_unit": "ETH", "timestamp": t0},
            {"from_address": "src_2", "transaction_id": "tx_in_2", "amount": 200, "amount_unit": "ETH", "timestamp": t0 + timedelta(hours=2)},
            {"from_address": "src_3", "transaction_id": "tx_in_3", "amount": 300, "amount_unit": "ETH", "timestamp": t0 + timedelta(hours=5)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("hub_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    fan_in = next((r for r in results if r.typology_type == TypologyType.FAN_IN), None)

    assert fan_in is not None, "Exactly 3 sources within window must produce positive Fan-In"
    assert fan_in.confidence_level == "observed"
    assert fan_in.counterparty_addresses == ["src_1", "src_2", "src_3"]
    assert fan_in.temporal_window is not None
    assert fan_in.temporal_window["duration_hours"] == 5.0


@pytest.mark.asyncio
async def test_08_fan_in_3_historical_sources_outside_window_negative():
    """Test 8: 3 distinct upstream sources spaced 30 days apart (outside 24h window) -> negative."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo, max_fan_window_hours=24.0)

    t0 = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "src_1", "transaction_id": "tx_1", "amount": 100, "amount_unit": "ETH", "timestamp": t0},
            {"from_address": "src_2", "transaction_id": "tx_2", "amount": 200, "amount_unit": "ETH", "timestamp": t0 + timedelta(days=30)},
            {"from_address": "src_3", "transaction_id": "tx_3", "amount": 300, "amount_unit": "ETH", "timestamp": t0 + timedelta(days=60)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("hub_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.FAN_IN for r in results), \
        "Sources spaced outside the temporal window must NOT trigger Fan-In"


@pytest.mark.asyncio
async def test_09_fan_in_source_addresses_and_txids_preserved():
    """Test 9: Fan-in preserves counterparty sources, txids, and deterministic ordering."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    t0 = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "0xCCC", "transaction_id": "tx_Z", "amount": 50, "amount_unit": "ETH", "timestamp": t0},
            {"from_address": "0xAAA", "transaction_id": "tx_A", "amount": 60, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=10)},
            {"from_address": "0xBBB", "transaction_id": "tx_M", "amount": 70, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=20)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("hub_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    fan_in = next(r for r in results if r.typology_type == TypologyType.FAN_IN)

    assert fan_in.counterparty_addresses == ["0xAAA", "0xBBB", "0xCCC"]
    assert fan_in.transaction_ids == ["tx_A", "tx_M", "tx_Z"]
    assert fan_in.addresses == ["hub_addr"]


@pytest.mark.asyncio
async def test_10_fan_in_below_source_threshold_negative():
    """Test 10: Fewer than MIN_FAN_IN_SOURCES (2 sources) -> negative."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    t0 = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "src_1", "transaction_id": "tx_1", "amount": 10, "amount_unit": "ETH", "timestamp": t0},
            {"from_address": "src_2", "transaction_id": "tx_2", "amount": 20, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=5)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("hub_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.FAN_IN for r in results)


@pytest.mark.asyncio
async def test_11_fan_in_missing_timestamps_negative():
    """Test 11: Sources exist but timestamps are missing -> negative (do not fabricate)."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "src_1", "transaction_id": "tx_1", "amount": 10, "amount_unit": "ETH", "timestamp": None},
            {"from_address": "src_2", "transaction_id": "tx_2", "amount": 20, "amount_unit": "ETH", "timestamp": None},
            {"from_address": "src_3", "transaction_id": "tx_3", "amount": 30, "amount_unit": "ETH", "timestamp": None},
        ]
    )

    path = TraversalPath(nodes=[_make_node("hub_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.FAN_IN for r in results)


# ---------------------------------------------------------------------------
# 3. FAN_OUT Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_12_fan_out_exact_3_destinations_within_window_positive():
    """Test 12: Exactly 3 distinct downstream destinations within 24h window -> positive."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo, max_fan_window_hours=24.0)

    t0 = datetime(2023, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_out_records = AsyncMock(
        return_value=[
            {"to_address": "dst_1", "transaction_id": "tx_out_1", "amount": 500, "amount_unit": "sat", "timestamp": t0},
            {"to_address": "dst_2", "transaction_id": "tx_out_2", "amount": 600, "amount_unit": "sat", "timestamp": t0 + timedelta(hours=3)},
            {"to_address": "dst_3", "transaction_id": "tx_out_3", "amount": 700, "amount_unit": "sat", "timestamp": t0 + timedelta(hours=6)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("dist_src", chain="bitcoin")], hops=0)
    results = await det.detect_path_typologies(path)
    fan_out = next((r for r in results if r.typology_type == TypologyType.FAN_OUT), None)

    assert fan_out is not None, "Exactly 3 destinations within window must produce positive Fan-Out"
    assert fan_out.confidence_level == "observed"
    assert fan_out.counterparty_addresses == ["dst_1", "dst_2", "dst_3"]
    assert fan_out.temporal_window is not None
    assert fan_out.temporal_window["duration_hours"] == 6.0


@pytest.mark.asyncio
async def test_13_fan_out_3_historical_destinations_outside_window_negative():
    """Test 13: 3 distinct downstream destinations spaced 15 days apart -> negative."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo, max_fan_window_hours=24.0)

    t0 = datetime(2023, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_out_records = AsyncMock(
        return_value=[
            {"to_address": "dst_1", "transaction_id": "tx_1", "amount": 100, "amount_unit": "sat", "timestamp": t0},
            {"to_address": "dst_2", "transaction_id": "tx_2", "amount": 200, "amount_unit": "sat", "timestamp": t0 + timedelta(days=15)},
            {"to_address": "dst_3", "transaction_id": "tx_3", "amount": 300, "amount_unit": "sat", "timestamp": t0 + timedelta(days=30)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("dist_src", chain="bitcoin")], hops=0)
    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.FAN_OUT for r in results)


@pytest.mark.asyncio
async def test_14_fan_out_destination_addresses_and_txids_preserved():
    """Test 14: Fan-out preserves counterparty destinations, txids, and deterministic ordering."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    t0 = datetime(2023, 6, 1, 14, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_out_records = AsyncMock(
        return_value=[
            {"to_address": "0xZZZ", "transaction_id": "tx_33", "amount": 100, "amount_unit": "ETH", "timestamp": t0},
            {"to_address": "0xXXX", "transaction_id": "tx_11", "amount": 200, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=15)},
            {"to_address": "0xYYY", "transaction_id": "tx_22", "amount": 300, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=30)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("source_addr", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    fan_out = next(r for r in results if r.typology_type == TypologyType.FAN_OUT)

    assert fan_out.counterparty_addresses == ["0xXXX", "0xYYY", "0xZZZ"]
    assert fan_out.transaction_ids == ["tx_11", "tx_22", "tx_33"]
    assert fan_out.addresses == ["source_addr"]


@pytest.mark.asyncio
async def test_15_fan_out_below_destination_threshold_negative():
    """Test 15: Fewer than MIN_FAN_OUT_DESTINATIONS (2 dests) -> negative."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    t0 = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_out_records = AsyncMock(
        return_value=[
            {"to_address": "dst_1", "transaction_id": "tx_1", "amount": 10, "amount_unit": "ETH", "timestamp": t0},
            {"to_address": "dst_2", "transaction_id": "tx_2", "amount": 20, "amount_unit": "ETH", "timestamp": t0 + timedelta(minutes=5)},
        ]
    )

    path = TraversalPath(nodes=[_make_node("dist_src", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.FAN_OUT for r in results)


# ---------------------------------------------------------------------------
# 4. RAPID_HOPPING Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_16_rapid_hopping_exact_boundary():
    """Test 16: Rapid hopping fires when window == MAX_RAPID_HOP_INTERVAL_MINUTES exactly."""
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    t_end = t0 + timedelta(minutes=MAX_RAPID_HOP_INTERVAL_MINUTES)
    mid = t0 + timedelta(minutes=MAX_RAPID_HOP_INTERVAL_MINUTES / 2)

    steps = [
        DummyUTXOStep("tx1", "addr1", t0.isoformat()),
        DummyUTXOStep("tx2", "addr2", mid.isoformat()),
        DummyUTXOStep("tx3", "addr3", t_end.isoformat()),
    ]
    nodes = [_make_node(f"addr{i}") for i in range(4)]
    path = TraversalPath(nodes=nodes, hops=MIN_RAPID_HOPS)
    path.utxo_steps = steps

    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    results = await det.detect_path_typologies(path)
    rapid = next((r for r in results if r.typology_type == TypologyType.RAPID_HOPPING), None)

    assert rapid is not None, "Rapid hopping must fire at exact boundary"
    assert rapid.confidence_level == "observed"


@pytest.mark.asyncio
async def test_17_rapid_hopping_exceeds_threshold():
    """Test 17: Rapid hopping does NOT fire when delta > threshold."""
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    t_end = t0 + timedelta(minutes=MAX_RAPID_HOP_INTERVAL_MINUTES, seconds=1)

    steps = [
        DummyUTXOStep("tx1", "addr1", t0.isoformat()),
        DummyUTXOStep("tx2", "addr2", (t0 + timedelta(minutes=10)).isoformat()),
        DummyUTXOStep("tx3", "addr3", t_end.isoformat()),
    ]
    nodes = [_make_node(f"addr{i}") for i in range(4)]
    path = TraversalPath(nodes=nodes, hops=MIN_RAPID_HOPS)
    path.utxo_steps = steps

    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.RAPID_HOPPING for r in results)


@pytest.mark.asyncio
async def test_18_rapid_hopping_missing_timestamps():
    """Test 18: Rapid hopping does NOT fire when timestamps are absent."""
    steps = [
        DummyUTXOStep("tx1", "addr1", None),
        DummyUTXOStep("tx2", "addr2", None),
        DummyUTXOStep("tx3", "addr3", None),
    ]
    nodes = [_make_node(f"addr{i}") for i in range(4)]
    path = TraversalPath(nodes=nodes, hops=MIN_RAPID_HOPS)
    path.utxo_steps = steps

    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    results = await det.detect_path_typologies(path)
    assert not any(r.typology_type == TypologyType.RAPID_HOPPING for r in results)


# ---------------------------------------------------------------------------
# 5. Safety & Determinism Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_19_mixer_always_insufficient_evidence():
    """Test 19: Mixer interaction always returns insufficient_evidence."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(nodes=[_make_node("addrM", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    mixer = next(r for r in results if r.typology_type == TypologyType.MIXER_INTERACTION)

    assert mixer.confidence_level == "insufficient_evidence"
    assert "no authoritative mixer identification source" in mixer.explanation.lower()


@pytest.mark.asyncio
async def test_20_cross_chain_always_insufficient_evidence():
    """Test 20: Cross-chain always returns insufficient_evidence."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(nodes=[_make_node("addrB", chain="evm")], hops=0)
    results = await det.detect_path_typologies(path)
    bridge = next(r for r in results if r.typology_type == TypologyType.CROSS_CHAIN)

    assert bridge.confidence_level == "insufficient_evidence"
    assert "no bridge detection" in bridge.explanation.lower()


@pytest.mark.asyncio
async def test_21_overlapping_typologies_all_returned():
    """Test 21: Multiple typologies on the same path are all returned."""
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        t0.isoformat(),
        (t0 + timedelta(minutes=5)).isoformat(),
        (t0 + timedelta(minutes=10)).isoformat(),
    ]
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        cont_sat=900_000_000,
        peeled_sat=100_000_000,
        timestamps=timestamps,
    )

    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)

    # Fan-in on the first node
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "ext1", "transaction_id": "tx_in1", "amount": 1000, "amount_unit": "sat", "timestamp": t0},
            {"from_address": "ext2", "transaction_id": "tx_in2", "amount": 2000, "amount_unit": "sat", "timestamp": t0 + timedelta(minutes=2)},
            {"from_address": "ext3", "transaction_id": "tx_in3", "amount": 3000, "amount_unit": "sat", "timestamp": t0 + timedelta(minutes=4)},
        ]
    )
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    results = await det.detect_path_typologies(path)
    types_found = {r.typology_type for r in results}

    assert TypologyType.PEEL_CHAIN in types_found
    assert TypologyType.RAPID_HOPPING in types_found
    assert TypologyType.FAN_IN in types_found
    assert TypologyType.MIXER_INTERACTION in types_found
    assert TypologyType.CROSS_CHAIN in types_found


@pytest.mark.asyncio
async def test_22_deterministic_sorting_by_type():
    """Test 22: Output list is sorted by (typology_type.value, detection_id)."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    det._fetch_fan_in_records = AsyncMock(
        return_value=[
            {"from_address": "s1", "transaction_id": "tx1", "amount": 1, "amount_unit": "sat", "timestamp": t0},
            {"from_address": "s2", "transaction_id": "tx2", "amount": 2, "amount_unit": "sat", "timestamp": t0},
            {"from_address": "s3", "transaction_id": "tx3", "amount": 3, "amount_unit": "sat", "timestamp": t0},
        ]
    )
    det._fetch_fan_out_records = AsyncMock(
        return_value=[
            {"to_address": "d1", "transaction_id": "tx4", "amount": 1, "amount_unit": "sat", "timestamp": t0},
            {"to_address": "d2", "transaction_id": "tx5", "amount": 2, "amount_unit": "sat", "timestamp": t0},
            {"to_address": "d3", "transaction_id": "tx6", "amount": 3, "amount_unit": "sat", "timestamp": t0},
        ]
    )

    path = TraversalPath(nodes=[_make_node("addrSort", chain="bitcoin")], hops=0)
    results = await det.detect_path_typologies(path)

    expected = sorted(results, key=lambda d: (d.typology_type.value, d.detection_id))
    assert results == expected


@pytest.mark.asyncio
async def test_23_read_only_no_db_writes():
    """Test 23: TypologyDetector never issues write queries."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(nodes=[_make_node("addrRO", chain="evm")], hops=0)
    await det.detect_path_typologies(path)

    assert len(pool.write_calls) == 0, "Detector must not emit write queries"


def test_24_deterministic_id_generation():
    """Test 24: Detection ID is identical regardless of input element order."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    id1 = det._make_detection_id("TYP", "bitcoin", ["Z", "A", "M"])
    id2 = det._make_detection_id("TYP", "bitcoin", ["M", "Z", "A"])
    id3 = det._make_detection_id("TYP", "bitcoin", ["A", "A", "M", "Z"])

    assert id1 == id2 == id3


@pytest.mark.asyncio
async def test_25_repeatability():
    """Test 25: Multiple invocations on same path produce identical output."""
    path, change_map = _make_bitcoin_peel_path(
        MIN_PEEL_CHAIN_HOPS,
        cont_sat=900_000_000,
        peeled_sat=100_000_000,
    )
    pool, _ = _make_pool_conn()
    repo = _make_repo(change_map)
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    run1 = await det.detect_path_typologies(path)
    run2 = await det.detect_path_typologies(path)

    assert len(run1) == len(run2)
    for d1, d2 in zip(run1, run2):
        assert d1.detection_id == d2.detection_id
        assert d1.typology_type == d2.typology_type
        assert d1.confidence_level == d2.confidence_level


@pytest.mark.asyncio
async def test_26_empty_path_returns_empty():
    """Test 26: Empty nodes list returns [] safely without error."""
    pool, _ = _make_pool_conn()
    repo = _make_repo()
    det = TypologyDetector(pool=pool, repo=repo)

    path = TraversalPath(nodes=[], hops=0)
    results = await det.detect_path_typologies(path)
    assert results == []
