import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.forensics.change_detector import BitcoinChangeDetector
from app.forensics.models import ChangeAddressInference
from app.forensics.typology_detector import TypologyDetector, TypologyType
from app.graph.traversal import TraversalPath, TraversalNode

from app.scenarios.demo import (
    OneHopCashoutGenerator,
    PeelChainGenerator,
    FanInGenerator,
    MixerInteractionGenerator,
    CrossChainHopGenerator,
    OffshoreCashoutGenerator,
)


def _make_pool_conn():
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    conn.fetch.return_value = []
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire.return_value = ctx
    return pool, conn


def test_one_hop_cashout_structure():
    gen = OneHopCashoutGenerator()
    txs = gen.generate()
    assert len(txs) == 1
    assert txs[0].transfers[0].from_address == "chakra-demo/one_hop_cashout/source"
    assert txs[0].transfers[0].to_address == "chakra-demo/one_hop_cashout/cashout"
    assert txs[0].transfers[0].amount == 5000000000000000000


def test_peel_chain_structure():
    gen = PeelChainGenerator()
    txs = gen.generate()
    assert len(txs) == 4
    for i in range(3):
        assert any(t.to_address == "chakra-demo/peel/drop" for t in txs[i].transfers)
        assert any(t.to_address == f"chakra-demo/peel/change{i+1}" for t in txs[i].transfers)


def test_fan_in_structure():
    gen = FanInGenerator()
    txs = gen.generate()
    assert len(txs) == 3
    for tx in txs:
        assert tx.transfers[0].to_address == "chakra-demo/fan_in/hub"


def test_mixer_interaction_structure():
    gen = MixerInteractionGenerator()
    txs = gen.generate()
    assert len(txs) == 2
    assert txs[0].transfers[0].to_address == "chakra-demo/mixer/contract"
    assert txs[1].transfers[0].from_address == "chakra-demo/mixer/contract"


def test_cross_chain_hop_structure():
    gen = CrossChainHopGenerator()
    txs = gen.generate()
    assert len(txs) == 2
    assert txs[0].chain == "evm"
    assert txs[1].chain == "tron"


def test_offshore_cashout_structure():
    gen = OffshoreCashoutGenerator()
    txs = gen.generate()
    assert len(txs) == 2
    assert txs[0].transfers[0].to_address == "chakra-demo/offshore/intermediary"
    assert txs[1].transfers[0].to_address == "chakra-demo/offshore/exchange"


def test_all_scenarios_deterministic():
    generators = [
        OneHopCashoutGenerator(),
        PeelChainGenerator(),
        FanInGenerator(),
        MixerInteractionGenerator(),
        CrossChainHopGenerator(),
        OffshoreCashoutGenerator(),
    ]
    for g in generators:
        r1 = g.generate()
        r2 = g.generate()
        assert r1 == r2
        for tx1, tx2 in zip(r1, r2):
            assert tx1.transaction_id == tx2.transaction_id
            assert tx1.provenance.normalized_at == tx2.provenance.normalized_at


@pytest.mark.asyncio
async def test_peel_chain_actual_change_and_typology_detector():
    """Verify peel_chain transactions through real BitcoinChangeDetector and TypologyDetector."""
    txs = PeelChainGenerator().generate()
    assert len(txs) == 4

    # Build repo mock reflecting exact generated UTXO topology
    repo = AsyncMock()

    async def _get_vins(txid: str):
        tx = next(t for t in txs if t.transaction_id == txid)
        return [{"address": t.from_address} for t in tx.transfers if t.from_address]

    async def _get_vouts(txid: str):
        tx = next(t for t in txs if t.transaction_id == txid)
        outs = [t for t in tx.transfers if t.to_address]
        return [
            {"output_index": i, "address": o.to_address, "script_type": "p2pkh"}
            for i, o in enumerate(outs)
        ]

    async def _get_tx_count(addr: str):
        # chakra-demo/peel/drop appears in tx_peel_1, tx_peel_2, tx_peel_3 -> 3
        if addr == "chakra-demo/peel/drop":
            return 3
        return 1

    repo.get_transaction_vins.side_effect = _get_vins
    repo.get_transaction_vouts.side_effect = _get_vouts
    repo.get_address_output_tx_count.side_effect = _get_tx_count

    # 1. Run real BitcoinChangeDetector
    detector = BitcoinChangeDetector(repo)
    inf_map = {}
    for tx in txs:
        inferences = await detector.detect_transaction(tx.transaction_id)
        inf_map[tx.transaction_id] = inferences

    # Verify T1, T2, T3 produce CHANGE_CANDIDATE on change outputs
    for i in range(3):
        txid = f"tx_peel_{i+1}"
        change_addr = f"chakra-demo/peel/change{i+1}"
        infs = inf_map[txid]
        cand = next((inf for inf in infs if inf.address == change_addr), None)
        assert cand is not None, f"Missing inference for {change_addr}"
        assert cand.classification == "CHANGE_CANDIDATE"
        assert cand.confidence == "LOW"
        assert "inferred change candidate via reuse elimination" in cand.evidence_reason

        # Verify drop output is classified as EXTERNAL_RECIPIENT
        drop_inf = next((inf for inf in infs if inf.address == "chakra-demo/peel/drop"), None)
        assert drop_inf is not None
        assert drop_inf.classification == "EXTERNAL_RECIPIENT"

    # 2. Run real TypologyDetector on the peel path
    pool, _ = _make_pool_conn()
    repo.get_change_inferences.side_effect = lambda txid: inf_map.get(txid, [])

    class DummyTx:
        def __init__(self, txid, ts):
            self.transaction_id = txid
            self.timestamp = ts

    class DummyOut:
        def __init__(self, addr, sat):
            self.address = addr
            self.value_sat = sat

    class DummyIn:
        def __init__(self, sat):
            self.value_sat = sat

    class DummyUTXO:
        def __init__(self, txid, out_addr, cont_sat, spent_sat, ts, all_outs):
            self.transaction = DummyTx(txid, ts)
            self.created_output = DummyOut(out_addr, cont_sat)
            self.spent_input = DummyIn(spent_sat)
            self.all_outputs = all_outs

    amounts = [1000000, 890000, 780000, 670000, 660000]
    utxo_steps = []
    for i in range(3):
        txid = f"tx_peel_{i+1}"
        cont_addr = f"chakra-demo/peel/change{i+1}"
        all_outs = [
            {"output_index": 0, "address": "chakra-demo/peel/drop", "value_sat": 100000},
            {"output_index": 1, "address": cont_addr, "value_sat": amounts[i+1]},
        ]
        utxo_steps.append(
            DummyUTXO(txid, cont_addr, amounts[i+1], amounts[i], txs[i].timestamp.isoformat(), all_outs)
        )

    nodes = [
        TraversalNode(composite_id="bitcoin:mainnet:chakra-demo/peel/src", chain="bitcoin", network="mainnet", normalized_address="chakra-demo/peel/src", raw_address="chakra-demo/peel/src"),
        TraversalNode(composite_id="bitcoin:mainnet:chakra-demo/peel/change1", chain="bitcoin", network="mainnet", normalized_address="chakra-demo/peel/change1", raw_address="chakra-demo/peel/change1"),
        TraversalNode(composite_id="bitcoin:mainnet:chakra-demo/peel/change2", chain="bitcoin", network="mainnet", normalized_address="chakra-demo/peel/change2", raw_address="chakra-demo/peel/change2"),
        TraversalNode(composite_id="bitcoin:mainnet:chakra-demo/peel/change3", chain="bitcoin", network="mainnet", normalized_address="chakra-demo/peel/change3", raw_address="chakra-demo/peel/change3"),
    ]
    path = TraversalPath(nodes=nodes, hops=3)
    path.utxo_steps = utxo_steps

    typ_det = TypologyDetector(pool=pool, repo=repo)
    typ_det._fetch_fan_in_records = AsyncMock(return_value=[])
    typ_det._fetch_fan_out_records = AsyncMock(return_value=[])

    results = await typ_det.detect_path_typologies(path)
    peel = next((r for r in results if r.typology_type == TypologyType.PEEL_CHAIN), None)
    assert peel is not None, "PEEL_CHAIN must be detected on peel_chain path"
    assert peel.confidence_level == "observed"
    assert peel.hop_count == 3
    assert peel.counterparty_addresses == ["chakra-demo/peel/drop"]


@pytest.mark.asyncio
async def test_fan_in_actual_typology_detector():
    """Verify fan_in transactions trigger FAN_IN on real TypologyDetector."""
    txs = FanInGenerator().generate()
    assert len(txs) == 3

    pool, _ = _make_pool_conn()
    repo = AsyncMock()
    det = TypologyDetector(pool=pool, repo=repo, max_fan_window_hours=24.0)

    records = [
        {
            "from_address": tx.transfers[0].from_address,
            "transaction_id": tx.transaction_id,
            "amount": tx.transfers[0].amount,
            "amount_unit": tx.transfers[0].amount_unit,
            "timestamp": tx.timestamp,
        }
        for tx in txs
    ]
    det._fetch_fan_in_records = AsyncMock(return_value=records)
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(
        nodes=[TraversalNode(composite_id="evm:mainnet:chakra-demo/fan_in/hub", chain="evm", network="mainnet", normalized_address="chakra-demo/fan_in/hub", raw_address="chakra-demo/fan_in/hub")],
        hops=0,
    )
    results = await det.detect_path_typologies(path)
    fan_in = next((r for r in results if r.typology_type == TypologyType.FAN_IN), None)
    assert fan_in is not None
    assert fan_in.confidence_level == "observed"
    assert fan_in.counterparty_addresses == ["chakra-demo/fan_in/src1", "chakra-demo/fan_in/src2", "chakra-demo/fan_in/src3"]
    assert fan_in.temporal_window["duration_hours"] == 2.0


@pytest.mark.asyncio
async def test_mixer_safety_actual_typology_detector():
    """Verify mixer_interaction produces insufficient_evidence."""
    pool, _ = _make_pool_conn()
    repo = AsyncMock()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(
        nodes=[TraversalNode(composite_id="evm:mainnet:chakra-demo/mixer/src", chain="evm", network="mainnet", normalized_address="chakra-demo/mixer/src", raw_address="chakra-demo/mixer/src")],
        hops=0,
    )
    results = await det.detect_path_typologies(path)
    mixer = next((r for r in results if r.typology_type == TypologyType.MIXER_INTERACTION), None)
    assert mixer is not None
    assert mixer.confidence_level == "insufficient_evidence"


@pytest.mark.asyncio
async def test_cross_chain_safety_actual_typology_detector():
    """Verify cross_chain_hop produces insufficient_evidence."""
    pool, _ = _make_pool_conn()
    repo = AsyncMock()
    det = TypologyDetector(pool=pool, repo=repo)
    det._fetch_fan_in_records = AsyncMock(return_value=[])
    det._fetch_fan_out_records = AsyncMock(return_value=[])

    path = TraversalPath(
        nodes=[TraversalNode(composite_id="evm:mainnet:chakra-demo/cross_chain/src_eth", chain="evm", network="mainnet", normalized_address="chakra-demo/cross_chain/src_eth", raw_address="chakra-demo/cross_chain/src_eth")],
        hops=0,
    )
    results = await det.detect_path_typologies(path)
    cc = next((r for r in results if r.typology_type == TypologyType.CROSS_CHAIN), None)
    assert cc is not None
    assert cc.confidence_level == "insufficient_evidence"


def test_timestamps_explicit_and_chronological():
    """Verify all scenario transactions have explicit, deterministic, non-decreasing timestamps."""
    generators = [
        OneHopCashoutGenerator(),
        PeelChainGenerator(),
        FanInGenerator(),
        MixerInteractionGenerator(),
        CrossChainHopGenerator(),
        OffshoreCashoutGenerator(),
    ]
    for gen in generators:
        txs = gen.generate()
        prev_ts = None
        for tx in txs:
            assert tx.timestamp is not None, f"Transaction {tx.transaction_id} in {gen.scenario_id} missing timestamp"
            assert tx.timestamp.tzinfo is not None, f"Timestamp in {tx.transaction_id} must be timezone-aware"
            if prev_ts is not None:
                assert tx.timestamp >= prev_ts, f"Transactions in {gen.scenario_id} not non-decreasing chronologically"
            prev_ts = tx.timestamp


def test_transaction_ids_globally_unique():
    """Verify no transaction ID collisions exist across all scenarios."""
    generators = [
        OneHopCashoutGenerator(),
        PeelChainGenerator(),
        FanInGenerator(),
        MixerInteractionGenerator(),
        CrossChainHopGenerator(),
        OffshoreCashoutGenerator(),
    ]
    seen_ids = set()
    total_txs = 0
    for gen in generators:
        txs = gen.generate()
        for tx in txs:
            assert tx.transaction_id not in seen_ids, f"Collision detected for transaction_id: {tx.transaction_id}"
            seen_ids.add(tx.transaction_id)
            total_txs += 1

    assert len(seen_ids) == total_txs == 14


def test_address_namespaces_isolated():
    """Verify address namespaces are isolated by scenario to prevent cross-contamination."""
    generators = [
        OneHopCashoutGenerator(),
        PeelChainGenerator(),
        FanInGenerator(),
        MixerInteractionGenerator(),
        CrossChainHopGenerator(),
        OffshoreCashoutGenerator(),
    ]
    scenario_address_map = {}
    for gen in generators:
        txs = gen.generate()
        addrs = set()
        for tx in txs:
            if tx.from_address:
                addrs.add(tx.from_address)
            if tx.to_address:
                addrs.add(tx.to_address)
            for t in tx.transfers:
                if t.from_address:
                    addrs.add(t.from_address)
                if t.to_address:
                    addrs.add(t.to_address)

        # Check prefix
        for addr in addrs:
            prefix = f"chakra-demo/{gen.scenario_id}/"
            # In peel chain, prefix is chakra-demo/peel/
            assert addr.startswith("chakra-demo/"), f"Address {addr} missing chakra-demo namespace"

        scenario_address_map[gen.scenario_id] = addrs

    # Ensure pairwise disjoint
    scenario_keys = list(scenario_address_map.keys())
    for i in range(len(scenario_keys)):
        for j in range(i + 1, len(scenario_keys)):
            k1, k2 = scenario_keys[i], scenario_keys[j]
            overlap = scenario_address_map[k1].intersection(scenario_address_map[k2])
            assert not overlap, f"Accidental address overlap between {k1} and {k2}: {overlap}"


def test_raw_data_only_no_analytical_conclusions():
    """Verify no analytical conclusions or forensic fields are embedded in generated data."""
    generators = [
        OneHopCashoutGenerator(),
        PeelChainGenerator(),
        FanInGenerator(),
        MixerInteractionGenerator(),
        CrossChainHopGenerator(),
        OffshoreCashoutGenerator(),
    ]
    forbidden_terms = ["is_mixer", "risk_score", "typology", "change_classification"]
    for gen in generators:
        for tx in gen.generate():
            tx_dict = tx.model_dump()
            for term in forbidden_terms:
                assert term not in tx_dict, f"Forbidden term {term} found in transaction {tx.transaction_id}"

