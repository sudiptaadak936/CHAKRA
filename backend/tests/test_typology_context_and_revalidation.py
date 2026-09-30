"""CHAKRA Step 5H.4: Typology Context and 13-Feature Exact Revalidation Test Suite.

Validates:
1. HistoricalTypologyContextAdapter:
   - Fan-in detection (>=3 sources within 24h)
   - Fan-out detection (>=3 destinations within 24h)
   - Rapid-hop detection (>=3 hops within 60 min)
   - Cutoff enforcement (events after cutoff never contribute)
2. Exact 13-feature deterministic extraction and materialization without network.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import math

import pytest

from app.schemas.chain import Chain, Network
from app.schemas.ml_features import CANONICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)
from app.forensics.ml_features import MLFeatureExtractor
from ml_training.typology_context_adapter import HistoricalTypologyContextAdapter
from ml_training.address_materialization import (
    AddressCandidate,
    AddressDatasetMaterializer,
    LabelSemantics,
)


def _prov(txid: str) -> TransactionProvenance:
    return TransactionProvenance(
        provider="etherscan_v2",
        chain=Chain.EVM,
        network=Network.ETH_MAINNET,
        original_id=txid,
    )


# ---------------------------------------------------------------------------
# Part 1: Typology Context Adapter Tests
# ---------------------------------------------------------------------------

def test_fan_in_detection_positive_within_24h():
    t0 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    cutoff = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    transfers = [
        Transfer(
            transaction_id=f"tx_in_{i}",
            from_address=f"0xsource{i}",
            to_address="0xtarget",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000 * (i + 1),
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=t0 + timedelta(hours=i),
        )
        for i in range(3)
    ]

    detections = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
        cutoff_timestamp=cutoff,
    )

    types = [d.typology_type.value for d in detections]
    assert "FAN_IN" in types


def test_fan_in_detection_negative_outside_24h():
    t0 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    cutoff = datetime(2023, 1, 5, 0, 0, 0, tzinfo=timezone.utc)

    # 3 transfers, but spread across 3 days (>24h apart)
    transfers = [
        Transfer(
            transaction_id=f"tx_in_{i}",
            from_address=f"0xsource{i}",
            to_address="0xtarget",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=t0 + timedelta(days=i),
        )
        for i in range(3)
    ]

    detections = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
        cutoff_timestamp=cutoff,
    )

    types = [d.typology_type.value for d in detections]
    assert "FAN_IN" not in types


def test_fan_out_detection_positive_within_24h():
    t0 = datetime(2023, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    cutoff = datetime(2023, 2, 2, 0, 0, 0, tzinfo=timezone.utc)

    transfers = [
        Transfer(
            transaction_id=f"tx_out_{i}",
            from_address="0xtarget",
            to_address=f"0xdest{i}",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=500,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=t0 + timedelta(minutes=15 * i),
        )
        for i in range(3)
    ]

    detections = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transfers=transfers,
        cutoff_timestamp=cutoff,
    )

    types = [d.typology_type.value for d in detections]
    assert "FAN_OUT" in types


def test_rapid_hopping_positive_within_60m():
    t0 = datetime(2023, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    cutoff = datetime(2023, 3, 2, 0, 0, 0, tzinfo=timezone.utc)

    path = [
        {"transaction_id": "tx1", "timestamp": t0, "address": "0xaddrA"},
        {"transaction_id": "tx2", "timestamp": t0 + timedelta(minutes=10), "address": "0xaddrB"},
        {"transaction_id": "tx3", "timestamp": t0 + timedelta(minutes=25), "address": "0xaddrC"},
        {"transaction_id": "tx4", "timestamp": t0 + timedelta(minutes=40), "address": "0xaddrD"},
    ]

    detections = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xaddrA",
        chain="evm",
        network="ethereum-mainnet",
        multi_hop_paths=[path],
        cutoff_timestamp=cutoff,
    )

    types = [d.typology_type.value for d in detections]
    assert "RAPID_HOPPING" in types


def test_rapid_hopping_temporal_leakage():
    t0 = datetime(2023, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    cutoff = datetime(2023, 3, 1, 12, 30, 0, tzinfo=timezone.utc)

    path = [
        {"transaction_id": "tx1", "timestamp": t0, "address": "0xaddrA"},
        {"transaction_id": "tx2", "timestamp": t0 + timedelta(minutes=10), "address": "0xaddrB"},
        {"transaction_id": "tx3", "timestamp": t0 + timedelta(minutes=40), "address": "0xaddrC"}, # Post-cutoff
    ]

    detections = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xaddrA",
        chain="evm",
        network="ethereum-mainnet",
        multi_hop_paths=[path],
        cutoff_timestamp=cutoff,
    )

    types = [d.typology_type.value for d in detections]
    assert "RAPID_HOPPING" not in types


# ---------------------------------------------------------------------------
# Part 2: Exact 13-Feature Deterministic Revalidation
# ---------------------------------------------------------------------------

def test_full_13_feature_exact_revalidation():
    import math
    from datetime import datetime, timezone, timedelta
    from app.schemas.transaction import Network, Chain, TransactionStatus, TransactionType, AssetType
    from app.schemas.transaction import Transaction, Transfer, TransactionProvenance
    from ml_training.typology_context_adapter import HistoricalTypologyContextAdapter
    from ml_training.address_materialization import AddressCandidate, LabelSemantics
    from ml_training.materialize_dataset_v3 import AddressDatasetMaterializer
    from app.forensics.ml_features import FEATURE_SCHEMA_VERSION

    def _prov(txid):
        return TransactionProvenance(provider="test", chain=Chain.EVM, network=Network.ETH_MAINNET, original_id=txid)

    cutoff = datetime(2023, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    t0 = datetime(2023, 4, 1, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Inbound Native: 0xa -> target, 10000, 10:00
    # 2. Inbound Native: 0xb -> target, 20000, 10:15
    # 3. Inbound Native: 0xc -> target, 30000, 10:30
    # 4. Token 1 (tx9): 0xh -> target, 50, 10:45
    # 5. Token 2 (tx9): 0xi -> target, 100, 10:45
    # 6. Internal Native (tx10): 0xj -> target, 500, 10:50
    # 7. Self-transfer (tx11): target -> target, 300, 10:55
    # 8. Outbound Native: target -> 0xd, 15000, 11:00
    # 9. Outbound Native: target -> 0xe, 5000, 11:15
    # 10. Outbound Native: target -> 0xf, 4000, 11:30
    # 11. Exact-cutoff: target -> 0xg, 1000, 12:00
    # 12. Post-cutoff: target -> 0xz, 999999, 13:00

    txs = []
    transfers = []

    def add_tx(txid, ts, src, dst, amt, asset="ETH", type_=AssetType.NATIVE):
        txs.append(
            Transaction(
                transaction_id=txid, chain=Chain.EVM, network=Network.ETH_MAINNET,
                native_value=amt if type_ == AssetType.NATIVE else 0,
                native_value_unit="wei", timestamp=ts, from_address=src, to_address=dst,
                status=TransactionStatus.SUCCESS, transaction_type=TransactionType.TRANSFER,
                provenance=_prov(txid)
            )
        )
        transfers.append(
            Transfer(
                transaction_id=txid, from_address=src, to_address=dst,
                asset_type=type_, asset_symbol=asset, amount=amt, amount_unit="wei",
                chain=Chain.EVM, network=Network.ETH_MAINNET, timestamp=ts
            )
        )

    # Adding regular natives
    add_tx("tx1", t0, "0xa", "0xtarget", 10000)
    add_tx("tx2", t0 + timedelta(minutes=15), "0xb", "0xtarget", 20000)
    add_tx("tx3", t0 + timedelta(minutes=30), "0xc", "0xtarget", 30000)

    # Adding multiple token transfers in one TX (tx9)
    ts_9 = t0 + timedelta(minutes=45)
    txs.append(
        Transaction(
            transaction_id="tx9", chain=Chain.EVM, network=Network.ETH_MAINNET,
            native_value=0, native_value_unit="wei", timestamp=ts_9, from_address="0xh", to_address="0xtarget",
            status=TransactionStatus.SUCCESS, transaction_type=TransactionType.TRANSFER, provenance=_prov("tx9")
        )
    )
    transfers.append(
        Transfer(
            transaction_id="tx9", from_address="0xh", to_address="0xtarget",
            asset_type=AssetType.TOKEN, asset_symbol="USDC", amount=50, amount_unit="wei",
            chain=Chain.EVM, network=Network.ETH_MAINNET, timestamp=ts_9
        )
    )
    transfers.append(
        Transfer(
            transaction_id="tx9", from_address="0xi", to_address="0xtarget",
            asset_type=AssetType.TOKEN, asset_symbol="DAI", amount=100, amount_unit="wei",
            chain=Chain.EVM, network=Network.ETH_MAINNET, timestamp=ts_9
        )
    )

    # Internal native (tx10)
    add_tx("tx10", t0 + timedelta(minutes=50), "0xj", "0xtarget", 500)

    # Self-transfer (tx11)
    add_tx("tx11", t0 + timedelta(minutes=55), "0xtarget", "0xtarget", 300)

    # Outbound natives
    add_tx("tx4", t0 + timedelta(minutes=60), "0xtarget", "0xd", 15000)
    add_tx("tx5", t0 + timedelta(minutes=75), "0xtarget", "0xe", 5000)
    add_tx("tx6", t0 + timedelta(minutes=90), "0xtarget", "0xf", 4000)

    # Exact cutoff
    add_tx("tx8", cutoff, "0xtarget", "0xg", 1000)

    # Post-cutoff
    add_tx("tx7", t0 + timedelta(hours=3), "0xtarget", "0xz", 999999)

    multi_hop_paths = [[
        {"transaction_id": "tx12", "timestamp": t0, "address": "0xtarget"},
        {"transaction_id": "tx13", "timestamp": t0+timedelta(minutes=5), "address": "0xaddrB"},
        {"transaction_id": "tx14", "timestamp": t0+timedelta(minutes=10), "address": "0xaddrC"},
        {"transaction_id": "tx15", "timestamp": t0+timedelta(minutes=15), "address": "0xaddrD"}
    ]]

    typologies = HistoricalTypologyContextAdapter.detect_typologies_from_evidence(
        target_address="0xtarget",
        chain="evm",
        network="ethereum-mainnet",
        transactions=txs,
        transfers=transfers,
        cutoff_timestamp=cutoff,
        multi_hop_paths=multi_hop_paths
    )

    candidate = AddressCandidate(
        chain="evm",
        network="ethereum-mainnet",
        address="0xtarget",
        entity_id="test_entity",
        label=1,
        label_source="OFAC Test",
        label_timestamp=cutoff + timedelta(hours=24),
        label_semantics=LabelSemantics.POSITIVE_LEGAL_SANCTION,
    )

    sample, reason, detail = AddressDatasetMaterializer.materialize_sample_from_evidence(
        candidate=candidate,
        observation_end=cutoff,
        transactions=txs,
        transfers=transfers,
        typologies=typologies,
    )

    assert reason is None
    assert sample is not None

    fv = sample.feature_values
    assert len(fv) == 13

    # Calculations:
    # in_degree: tx1, tx2, tx3, tx9(x2), tx10, tx11 = 7
    # out_degree: tx11, tx4, tx5, tx6, tx8 = 5
    # total_tx_count: tx1, tx2, tx3, tx9, tx10, tx11, tx4, tx5, tx6, tx8 = 10
    # total_recv: 10000+20000+30000+50+100+500+300 = 60950
    # total_sent: 15000+5000+4000+1000+300 = 25300
    # retention: (60950 - 25300) / 60950 = 35650 / 60950 = 0.584905660377
    # time_active: 12:00:00 - 10:00:00 = 7200 seconds
    # inter_hop_velocity: spends at 10:55, 11:00, 11:15, 11:30, 12:00.
    # latest receipt for all spends is 10:55 (from tx11 self-transfer).
    # diffs: 0, 300, 1200, 2100, 3900. Average = 7500 / 5 = 1500
    # counterparties: 0xa, 0xb, 0xc, 0xh, 0xi, 0xj, 0xd, 0xe, 0xf, 0xg = 10 (self 0xtarget excluded)

    assert fv[0] == 7.0, f"in_degree expected 7.0, got {fv[0]}"
    assert fv[1] == 5.0, f"out_degree expected 5.0, got {fv[1]}"
    assert fv[2] == 10.0, f"total_tx_count expected 10.0, got {fv[2]}"
    assert fv[3] == 60800.0
    assert fv[4] == 25300.0
    assert math.isclose(fv[5], 0.58388157, rel_tol=1e-5), f"retention expected ~0.58388, got {fv[5]}"
    assert fv[6] == 7200.0, f"time_active expected 7200.0, got {fv[6]}"
    assert math.isclose(fv[7], 1500.0, rel_tol=1e-5), f"inter_hop_velocity expected 1500.0, got {fv[7]}"
    assert fv[8] == 10.0, f"unique_counterparties expected 10.0, got {fv[8]}"
    assert fv[9] == 0.0, f"peel_chain expected 0.0, got {fv[9]}"
    assert fv[10] == 1.0, f"rapid_hop expected 1.0, got {fv[10]}"
    assert fv[11] == 1.0, f"fan_in expected 1.0, got {fv[11]}"
    assert fv[12] == 1.0, f"fan_out expected 1.0, got {fv[12]}"

    assert all(math.isfinite(x) for x in fv)
