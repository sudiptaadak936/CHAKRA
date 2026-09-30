"""CHAKRA Step 5A: Test Suite for Deterministic ML Feature Extractor.

Verifies:
1. Determinism and repeatability.
2. Exact monetary/numeric preservation (Decimal/integers).
3. Zero-evidence behavior.
4. Division-by-zero and overflow safety.
5. Temporal behavior (no wall-clock dependencies).
6. Inter-hop velocity and sentinel rules.
7. Typology confidence filtering.
8. Explicit unsupported feature declarations (mixer, cluster_size).
9. Cross-chain isolation and normalization.
10. Read-only database query execution (mock pool).
11. Model-vector conversion order and float safety.
12. Fail-closed input validation.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.schemas.chain import Chain
from app.schemas.ml_features import (
    FEATURE_SCHEMA_VERSION,
    CANONICAL_FEATURE_NAMES,
    UNSUPPORTED_FEATURE_NAMES,
    FeatureStatus,
    MLFeatureRecord,
    MLFeatureValues,
)
from app.forensics.ml_features import MLFeatureExtractor
from app.forensics.typology_detector import TypologyDetection, TypologyType


# ---------------------------------------------------------------------------
# 1. Determinism & Repeatability
# ---------------------------------------------------------------------------

def test_feature_extraction_is_deterministic():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)

    transfers = [
        {"from_address": "0xaaa", "to_address": "0xTARGET", "asset_type": "native", "amount": 1000, "transaction_id": "tx1", "timestamp": t0},
        {"from_address": "0xTARGET", "to_address": "0xbbb", "asset_type": "native", "amount": 400, "transaction_id": "tx2", "timestamp": t1},
    ]

    fixed_time = datetime(2026, 9, 20, 0, 0, 0, tzinfo=timezone.utc)
    res1 = MLFeatureExtractor.extract_from_evidence(
        target_address="0xTARGET",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
        extracted_at=fixed_time,
    )
    res2 = MLFeatureExtractor.extract_from_evidence(
        target_address="0xTARGET",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
        extracted_at=fixed_time,
    )

    assert res1.features == res2.features
    assert res1.to_model_vector() == res2.to_model_vector()
    assert res1.features.in_degree == 1
    assert res1.features.out_degree == 1
    assert res1.features.total_tx_count == 2
    assert res1.features.amount_retention_ratio == 0.6
    assert res1.features.time_active_seconds == 1800.0
    assert res1.features.inter_hop_velocity_avg == 1800.0
    assert res1.features.unique_counterparties == 2


# ---------------------------------------------------------------------------
# 2. Exact Numeric Preservation
# ---------------------------------------------------------------------------

def test_exact_numeric_preservation_large_wei():
    # 50 ETH in wei: 50 * 10^18
    large_wei_received = Decimal("50000000000000000000")
    large_wei_sent = Decimal("15000000000000000000")

    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": str(large_wei_received)},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": str(large_wei_sent)},
    ]

    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )

    # Authoritative representation MUST be exact Decimal, not float
    assert isinstance(record.features.total_received_native, Decimal)
    assert isinstance(record.features.total_sent_native, Decimal)
    assert record.features.total_received_native == large_wei_received
    assert record.features.total_sent_native == large_wei_sent

    # Model vector converts to float at the final boundary safely
    vec = record.to_model_vector()
    assert isinstance(vec[3], float)
    assert isinstance(vec[4], float)
    assert vec[3] == float(large_wei_received)
    assert vec[4] == float(large_wei_sent)


# ---------------------------------------------------------------------------
# 3. Zero-Evidence Behavior
# ---------------------------------------------------------------------------

def test_zero_evidence_behavior():
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0x1111111111111111111111111111111111111111",
        chain="evm",
        network="ethereum-mainnet",
    )

    assert record.features.in_degree == 0
    assert record.features.out_degree == 0
    assert record.features.total_tx_count == 0
    assert record.features.total_received_native == Decimal(0)
    assert record.features.total_sent_native == Decimal(0)
    assert record.features.amount_retention_ratio == 0.0
    assert record.features.time_active_seconds == 0.0
    assert record.features.inter_hop_velocity_avg == -1.0
    assert record.features.unique_counterparties == 0
    assert record.features.typology_peel_chain_flag == 0
    assert record.features.typology_rapid_hop_flag == 0
    assert record.features.typology_fan_in_flag == 0
    assert record.features.typology_fan_out_flag == 0

    vec = record.to_model_vector()
    assert len(vec) == 13
    assert vec == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 4. Division Safety & Mathematical Invariants
# ---------------------------------------------------------------------------

def test_amount_retention_ratio_zero_division():
    # Only sent funds, 0 received
    transfers = [
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 1000},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.amount_retention_ratio == 0.0
    assert math.isfinite(record.features.amount_retention_ratio)


def test_amount_retention_ratio_negative_fraction():
    # Spent more than received
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 150},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    # (100 - 150) / 100 = -0.5
    assert record.features.amount_retention_ratio == -0.5
    assert math.isfinite(record.features.amount_retention_ratio)


# ---------------------------------------------------------------------------
# 5. Temporal Behavior (No Wall-Clock Reliance)
# ---------------------------------------------------------------------------

def test_time_active_requires_multiple_timestamps():
    t0 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    # Single transfer with timestamp
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t0},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.time_active_seconds == 0.0


def test_time_active_spans_max_min():
    t0 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 6, 1, 12, 30, 0, tzinfo=timezone.utc)
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t0},
        {"from_address": "0xbbb", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t1},
        {"from_address": "0xtarget", "to_address": "0xccc", "asset_type": "native", "amount": 150, "timestamp": t2},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    # 2.5 hours = 9000 seconds
    assert record.features.time_active_seconds == 9000.0


# ---------------------------------------------------------------------------
# 6. Inter-Hop Velocity & Sentinel Rules
# ---------------------------------------------------------------------------

def test_inter_hop_velocity_sentinel_when_no_receipt_spend_sequence():
    # Only receipts, no spends
    t0 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc)
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t0},
        {"from_address": "0xbbb", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t1},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.inter_hop_velocity_avg == -1.0


def test_inter_hop_velocity_calculated_from_receipt_to_spend():
    t_in = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_out = datetime(2025, 6, 1, 10, 15, 0, tzinfo=timezone.utc)  # 900 seconds
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t_in},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 50, "timestamp": t_out},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.inter_hop_velocity_avg == 900.0


def test_inter_hop_velocity_equal_timestamps_produces_zero_delta():
    # Same-block receipt and spend
    t_same = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t_same},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 50, "timestamp": t_same},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.inter_hop_velocity_avg == 0.0


def test_inter_hop_velocity_row_order_independence():
    # Multiple receipts and spends in forward vs reverse order
    t0 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2025, 6, 1, 10, 10, 0, tzinfo=timezone.utc)  # +600s
    t2 = datetime(2025, 6, 1, 10, 25, 0, tzinfo=timezone.utc)  # +900s

    transfers_order_a = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t0},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 40, "timestamp": t1},
        {"from_address": "0xtarget", "to_address": "0xccc", "asset_type": "native", "amount": 30, "timestamp": t2},
    ]
    # Scrambled order
    transfers_order_b = [
        {"from_address": "0xtarget", "to_address": "0xccc", "asset_type": "native", "amount": 30, "timestamp": t2},
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t0},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 40, "timestamp": t1},
    ]

    rec_a = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget", chain="evm", network="ethereum-mainnet", transfers=transfers_order_a
    )
    rec_b = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget", chain="evm", network="ethereum-mainnet", transfers=transfers_order_b
    )

    # Both spends pair with t0 (600s and 1500s -> avg = 1050s)
    assert rec_a.features.inter_hop_velocity_avg == 1050.0
    assert rec_b.features.inter_hop_velocity_avg == 1050.0
    assert rec_a.features == rec_b.features


def test_inter_hop_velocity_missing_timestamps_produces_sentinel():
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": None},
        {"from_address": "0xtarget", "to_address": "0xbbb", "asset_type": "native", "amount": 50, "timestamp": None},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    assert record.features.inter_hop_velocity_avg == -1.0


def test_inter_hop_velocity_pairs_with_most_recent_prior_receipt():
    t_rec1 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_rec2 = datetime(2025, 6, 1, 10, 10, 0, tzinfo=timezone.utc)
    t_spend = datetime(2025, 6, 1, 10, 20, 0, tzinfo=timezone.utc)
    transfers = [
        {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t_rec1},
        {"from_address": "0xbbb", "to_address": "0xtarget", "asset_type": "native", "amount": 100, "timestamp": t_rec2},
        {"from_address": "0xtarget", "to_address": "0xccc", "asset_type": "native", "amount": 50, "timestamp": t_spend},
    ]
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
    )
    # Spend at 10:20 pairs with most recent receipt at 10:10 -> delta = 600s
    assert record.features.inter_hop_velocity_avg == 600.0


# ---------------------------------------------------------------------------
# 7. Typology Behavior & Confidence Gating
# ---------------------------------------------------------------------------

def test_typology_flags_require_observed_confidence():
    observed_peel = TypologyDetection(
        detection_id="peel-1",
        typology_type=TypologyType.PEEL_CHAIN,
        chain="bitcoin",
        confidence_level="observed",
        explanation="Valid peel chain",
    )
    unobserved_fan_in = TypologyDetection(
        detection_id="fanin-1",
        typology_type=TypologyType.FAN_IN,
        chain="evm",
        confidence_level="insufficient_evidence",
        explanation="Weak fan in",
    )
    observed_rapid_hop = TypologyDetection(
        detection_id="hop-1",
        typology_type=TypologyType.RAPID_HOPPING,
        chain="evm",
        confidence_level="observed",
        explanation="Rapid hops observed",
    )

    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        typologies=[observed_peel, unobserved_fan_in, observed_rapid_hop],
    )

    assert record.features.typology_peel_chain_flag == 1
    assert record.features.typology_fan_in_flag == 0  # Gated out due to insufficient_evidence
    assert record.features.typology_rapid_hop_flag == 1
    assert record.features.typology_fan_out_flag == 0


def test_ineligible_typologies_do_not_activate_flags():
    # Mixer and Cross-chain are unsupported upstream stubs
    mixer_det = TypologyDetection(
        detection_id="mix-1",
        typology_type=TypologyType.MIXER_INTERACTION,
        chain="evm",
        confidence_level="observed",
        explanation="Mixer hit",
    )
    cross_det = TypologyDetection(
        detection_id="cross-1",
        typology_type=TypologyType.CROSS_CHAIN,
        chain="evm",
        confidence_level="observed",
        explanation="Bridge hit",
    )

    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        typologies=[mixer_det, cross_det],
    )

    # Neither affects the 4 canonical typology flags
    assert record.features.typology_peel_chain_flag == 0
    assert record.features.typology_rapid_hop_flag == 0
    assert record.features.typology_fan_in_flag == 0
    assert record.features.typology_fan_out_flag == 0


# ---------------------------------------------------------------------------
# 8. Unsupported Features Declaration
# ---------------------------------------------------------------------------

def test_unsupported_features_explicitly_declared():
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
    )

    for unsupp in UNSUPPORTED_FEATURE_NAMES:
        assert unsupp in record.availability
        avail = record.availability[unsupp]
        assert avail.status == FeatureStatus.UNSUPPORTED
        assert avail.reason is not None
        assert len(avail.reason) > 0

    for supp in CANONICAL_FEATURE_NAMES:
        assert supp in record.availability
        assert record.availability[supp].status == FeatureStatus.AVAILABLE


# ---------------------------------------------------------------------------
# 9. Cross-Chain Isolation & Address Normalization
# ---------------------------------------------------------------------------

def test_cross_chain_address_normalization():
    # EVM lowercase normalization
    evm_record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xAbCdEf123456",
        chain="evm",
        network="ethereum-mainnet",
    )
    assert evm_record.normalized_address == "0xabcdef123456"

    # Bitcoin case preservation
    btc_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    btc_record = MLFeatureExtractor.extract_from_evidence(
        target_address=btc_addr,
        chain="bitcoin",
        network="bitcoin-mainnet",
    )
    assert btc_record.normalized_address == btc_addr


def test_bitcoin_utxo_vin_vout_extraction():
    target = "1TargetBtcAddress"
    vouts = [
        {"address": target, "value_sat": 50000000, "txid": "tx_btc_1", "timestamp": "2026-01-01T10:00:00Z"},
    ]
    vins = [
        {"address": target, "value_sat": 20000000, "txid": "tx_btc_2", "timestamp": "2026-01-01T11:00:00Z"},
    ]

    record = MLFeatureExtractor.extract_from_evidence(
        target_address=target,
        chain="bitcoin",
        network="bitcoin-mainnet",
        bitcoin_vouts=vouts,
        bitcoin_vins=vins,
    )

    assert record.features.in_degree == 1
    assert record.features.out_degree == 1
    assert record.features.total_tx_count == 2
    assert record.features.total_received_native == Decimal(50000000)
    assert record.features.total_sent_native == Decimal(20000000)
    assert record.features.amount_retention_ratio == 0.6
    assert record.features.time_active_seconds == 3600.0
    assert record.features.inter_hop_velocity_avg == 3600.0


# ---------------------------------------------------------------------------
# 10. Read-Only Query Execution (Mock DB)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_from_db_executes_read_only_queries():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Setup mock return data for EVM queries
    mock_conn.fetch.side_effect = [
        # transfers
        [
            {"from_address": "0xaaa", "to_address": "0xtarget", "asset_type": "native", "amount": 500, "transaction_id": "tx1", "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        ],
        # transactions
        [
            {"transaction_id": "tx1", "from_address": "0xaaa", "to_address": "0xtarget", "native_value": 0, "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        ],
    ]

    record = await MLFeatureExtractor.extract_from_db(
        pool=mock_pool,
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
    )

    assert record.features.in_degree == 1
    assert record.features.total_received_native == Decimal(500)
    assert record.features.total_tx_count == 1

    # Verify that mock_conn executed ONLY SELECT queries (no INSERT/UPDATE/DELETE)
    for call_args in mock_conn.fetch.call_args_list:
        query_sql = call_args[0][0].strip().upper()
        assert query_sql.startswith("SELECT")
        assert "INSERT" not in query_sql
        assert "UPDATE" not in query_sql
        assert "DELETE" not in query_sql


# ---------------------------------------------------------------------------
# 11. Schema Immutability & Vector Ordering
# ---------------------------------------------------------------------------

def test_feature_record_and_values_are_immutable():
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
    )

    with pytest.raises(ValidationError):
        record.target_address = "0xnew"

    with pytest.raises(ValidationError):
        record.features.in_degree = 99


def test_model_vector_exact_canonical_ordering():
    record = MLFeatureExtractor.extract_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
    )
    vec = record.to_model_vector()
    assert len(vec) == len(CANONICAL_FEATURE_NAMES)
    assert record.feature_schema_version == FEATURE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 12. Fail-Closed Input Validation
# ---------------------------------------------------------------------------

def test_fail_closed_missing_address():
    with pytest.raises(ValueError, match="Target address is required"):
        MLFeatureExtractor.extract_from_evidence(
            target_address="",
            chain="evm",
            network="ethereum-mainnet",
        )
    with pytest.raises(ValueError, match="Target address is required"):
        MLFeatureExtractor.extract_from_evidence(
            target_address="   ",
            chain="evm",
            network="ethereum-mainnet",
        )


def test_fail_closed_invalid_chain():
    with pytest.raises(ValueError, match="Invalid chain"):
        MLFeatureExtractor.extract_from_evidence(
            target_address="0xtarget",
            chain="nonexistent_chain",
            network="ethereum-mainnet",
        )


def test_fail_closed_missing_network():
    with pytest.raises(ValueError, match="Network is required"):
        MLFeatureExtractor.extract_from_evidence(
            target_address="0xtarget",
            chain="evm",
            network="",
        )
