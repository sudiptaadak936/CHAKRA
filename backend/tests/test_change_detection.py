"""CHAKRA Step 3D: Bitcoin Change-Address Detection Test Suite.

Tests for the deterministic Bitcoin Change-Address Detector.
Covering Cases 1-8 and adversarial scenarios with audited classification semantics.
"""
from __future__ import annotations

import os
from typing import List

import asyncpg
import pytest

from app.clustering.engine import ClusteringEngine
from app.clustering.repository import ClusteringRepository
from app.forensics.change_detector import BitcoinChangeDetector
from app.forensics.repository import ForensicsRepository
from app.forensics.models import ChangeAddressInference

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "chakra_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "your_actual_password")
PG_DB = os.getenv("POSTGRES_DB", "chakra_db")


@pytest.fixture
async def pg_pool():
    pool = await asyncpg.create_pool(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database=PG_DB,
        min_size=1, max_size=5,
    )
    yield pool
    await pool.close()


@pytest.fixture
async def repo(pg_pool):
    r = ForensicsRepository(pg_pool)
    await r.init_schema()
    yield r


@pytest.fixture
async def clustering_repo(pg_pool):
    cr = ClusteringRepository(pg_pool)
    await cr.init_schema()
    yield cr


@pytest.fixture
async def clean_pg(pg_pool, repo, clustering_repo):
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
        await conn.execute("TRUNCATE TABLE bitcoin_change_inferences CASCADE")
        await conn.execute("TRUNCATE TABLE clustering_evidence CASCADE")
        await conn.execute("TRUNCATE TABLE address_clusters CASCADE")
        await conn.execute("TRUNCATE TABLE cluster_members CASCADE")
    yield pg_pool


@pytest.fixture
async def detector(repo):
    yield BitcoinChangeDetector(repo)


async def _insert_btc_tx(
    conn: asyncpg.Connection,
    txid: str,
    network: str = "bitcoin-mainnet",
    input_addresses: List[str] = None,
    output_addresses: List[str] = None,
) -> None:
    tx_pk = await conn.fetchval(
        """
        INSERT INTO transactions (
            transaction_id, chain, network, native_value, native_value_unit,
            transaction_type, status
        ) VALUES ($1, 'bitcoin', $2, 0, 'sat', 'transfer', 'success')
        ON CONFLICT (transaction_id, chain, network) DO UPDATE SET chain=EXCLUDED.chain
        RETURNING id
        """,
        txid, network,
    )
    detail_pk = await conn.fetchval(
        """
        INSERT INTO bitcoin_transaction_details (transaction_pk, txid, total_output_sat)
        VALUES ($1, $2, 0) RETURNING id
        """,
        tx_pk, txid,
    )
    for addr in (input_addresses or []):
        await conn.execute(
            "INSERT INTO bitcoin_vins (detail_pk, address, coinbase) VALUES ($1, $2, FALSE)",
            detail_pk, addr,
        )
    for i, addr in enumerate(output_addresses or []):
        await conn.execute(
            "INSERT INTO bitcoin_vouts (detail_pk, n, address, value_sat) VALUES ($1, $2, $3, 0)",
            detail_pk, i, addr,
        )


# ---------------------------------------------------------------------------
# Case 1: Obvious Change Candidate (Self-Change)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_1_obvious_change_self_change(clean_pg, detector):
    """Case 1 — obvious change candidate (self-change).
    
    Output address matches an input address of the same transaction.
    Expected: CHANGE_CANDIDATE with HIGH confidence.
    The other fresh output remains UNKNOWN (cannot assume external without reuse proof).
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_1",
            input_addresses=["sender1", "sender2"],
            output_addresses=["recipient1", "sender1"] # sender1 is self-change
        )
    
    inferences = await detector.detect_transaction("tx_1")
    assert len(inferences) == 2
    
    sender_inf = next(i for i in inferences if i.address == "sender1")
    assert sender_inf.classification == "CHANGE_CANDIDATE"
    assert sender_inf.confidence == "HIGH"
    assert "matches an input" in sender_inf.evidence_reason

    recipient_inf = next(i for i in inferences if i.address == "recipient1")
    assert recipient_inf.classification == "UNKNOWN"
    assert recipient_inf.confidence == "UNKNOWN"


# ---------------------------------------------------------------------------
# Case 2: Single-Output Ambiguity & External Recipient Classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_2_single_output_is_unknown(clean_pg, detector):
    """Case 2A — single output transaction is NOT assumed to be external recipient.
    
    A single output could be an external sweep, an internal consolidation, or cold storage move.
    Expected: UNKNOWN with UNKNOWN confidence (conservative classification).
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_2_single",
            input_addresses=["sender1"],
            output_addresses=["dest1"]
        )
    
    inferences = await detector.detect_transaction("tx_2_single")
    assert len(inferences) == 1
    assert inferences[0].classification == "UNKNOWN"
    assert inferences[0].confidence == "UNKNOWN"
    assert "Single-output transaction" in inferences[0].evidence_reason


@pytest.mark.asyncio
async def test_case_2_single_output_self_sweep(clean_pg, detector):
    """Case 2B — single output returning to input address is self-sweep/change."""
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_2_sweep",
            input_addresses=["sender1"],
            output_addresses=["sender1"]
        )
    
    inferences = await detector.detect_transaction("tx_2_sweep")
    assert len(inferences) == 1
    assert inferences[0].classification == "CHANGE_CANDIDATE"
    assert inferences[0].confidence == "HIGH"
    assert "self-sweep" in inferences[0].evidence_reason


@pytest.mark.asyncio
async def test_case_2_external_recipient_observed_reuse(clean_pg, detector):
    """Case 2C — address reuse is evidence of recipient behavior, not absolute proof.
    
    When an output address is seen receiving funds in multiple transactions,
    it is classified as EXTERNAL_RECIPIENT with LOW confidence.
    """
    async with clean_pg.acquire() as conn:
        # Prior transaction
        await _insert_btc_tx(conn, "tx_prev", input_addresses=["other1"], output_addresses=["merchant"])
        # Current transaction
        await _insert_btc_tx(conn, "tx_curr", input_addresses=["sender1"], output_addresses=["merchant", "change1"])
    
    inferences = await detector.detect_transaction("tx_curr")
    merchant_inf = next(i for i in inferences if i.address == "merchant")
    assert merchant_inf.classification == "EXTERNAL_RECIPIENT"
    assert merchant_inf.confidence == "LOW"
    assert "Observed address reuse" in merchant_inf.evidence_reason


# ---------------------------------------------------------------------------
# Case 3: Ambiguous Transaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_3_ambiguous_transaction(clean_pg, detector):
    """Case 3 — ambiguous transaction (multiple fresh outputs).
    
    Neither address has prior history, and neither matches inputs.
    Expected: UNKNOWN / INCONCLUSIVE for both outputs. Must NOT force a classification.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_3",
            input_addresses=["sender1"],
            output_addresses=["fresh1", "fresh2"]
        )
    
    inferences = await detector.detect_transaction("tx_3")
    assert len(inferences) == 2
    for inf in inferences:
        assert inf.classification == "UNKNOWN"
        assert inf.confidence == "UNKNOWN"
        assert "Ambiguous transaction" in inf.evidence_reason


# ---------------------------------------------------------------------------
# Case 4: One-Time Change via Reuse Elimination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_4_one_time_change_reuse_elim(clean_pg, detector):
    """Case 4 — one-time change via reuse elimination.
    
    Address appears only once initially and subsequent reuse history of the
    other output provides elimination evidence.
    Expected:
      - Reused output: EXTERNAL_RECIPIENT (LOW confidence)
      - Fresh output: CHANGE_CANDIDATE (LOW confidence, inferred by elimination)
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_payment",
            input_addresses=["other_sender"],
            output_addresses=["recipient1"]
        )
        await _insert_btc_tx(
            conn, "tx_4",
            input_addresses=["sender1"],
            output_addresses=["recipient1", "fresh_change"]
        )
        
    inferences = await detector.detect_transaction("tx_4")
    assert len(inferences) == 2
    
    recipient_inf = next(i for i in inferences if i.address == "recipient1")
    assert recipient_inf.classification == "EXTERNAL_RECIPIENT"
    assert recipient_inf.confidence == "LOW"
    
    change_inf = next(i for i in inferences if i.address == "fresh_change")
    assert change_inf.classification == "CHANGE_CANDIDATE"
    assert change_inf.confidence == "LOW"
    assert "reuse elimination" in change_inf.evidence_reason


# ---------------------------------------------------------------------------
# Case 5: Multiple Outputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_5_multiple_outputs_all_reused(clean_pg, detector):
    """Case 5A — multiple outputs, all with observed reuse.
    
    Both outputs are known recipients from earlier transactions.
    Expected: Both are EXTERNAL_RECIPIENT (LOW confidence), 0 change candidates.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_h1", input_addresses=["a"], output_addresses=["reused1"])
        await _insert_btc_tx(conn, "tx_h2", input_addresses=["b"], output_addresses=["reused2"])
        await _insert_btc_tx(conn, "tx_multi", input_addresses=["c"], output_addresses=["reused1", "reused2"])

    inferences = await detector.detect_transaction("tx_multi")
    assert len(inferences) == 2
    assert all(i.classification == "EXTERNAL_RECIPIENT" for i in inferences)
    assert all(i.confidence == "LOW" for i in inferences)
    assert not any(i.classification == "CHANGE_CANDIDATE" for i in inferences)


@pytest.mark.asyncio
async def test_case_5_multiple_outputs_one_reused_two_fresh(clean_pg, detector):
    """Case 5B — multiple outputs: 1 reused, 2 fresh.
    
    Elimination of the reused output leaves 2 fresh candidates -> still ambiguous.
    Expected: Reused output is EXTERNAL_RECIPIENT, remaining fresh outputs are UNKNOWN.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_hist", input_addresses=["x"], output_addresses=["deposit_addr"])
        await _insert_btc_tx(
            conn, "tx_3outs",
            input_addresses=["y"],
            output_addresses=["deposit_addr", "fresh_a", "fresh_b"]
        )

    inferences = await detector.detect_transaction("tx_3outs")
    assert len(inferences) == 3

    reused = next(i for i in inferences if i.address == "deposit_addr")
    assert reused.classification == "EXTERNAL_RECIPIENT"
    assert reused.confidence == "LOW"

    fresh_infs = [i for i in inferences if i.address in ("fresh_a", "fresh_b")]
    assert len(fresh_infs) == 2
    assert all(i.classification == "UNKNOWN" for i in fresh_infs)
    assert all(i.confidence == "UNKNOWN" for i in fresh_infs)


# ---------------------------------------------------------------------------
# Case 6: Regression Against 3C (Common-Input Clustering)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_6_regression_against_3c(clean_pg, detector, pg_pool):
    """Case 6 — regression against 3C Common-Input/H1.
    
    Ensures that change detection operations do NOT mutate or interfere with
    the deterministic common-input clustering engine.
    """
    async with clean_pg.acquire() as conn:
        # A 2-input Bitcoin transaction for H1 clustering
        await _insert_btc_tx(
            conn, "tx_coinput",
            input_addresses=["alice", "bob"],
            output_addresses=["merchant", "alice"] # alice is change
        )

    # 1. Run change detection
    change_infs = await detector.detect_and_save("tx_coinput")
    assert len(change_infs) == 2
    alice_inf = next(i for i in change_infs if i.address == "alice")
    assert alice_inf.classification == "CHANGE_CANDIDATE"

    # 2. Run Step 3C clustering engine
    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline()
    assert result.success is True
    assert result.evidence_summary.evidence_relationships_generated == 1
    assert result.clusters_generated == 1

    # Verify cluster has exactly alice and bob (co-input preserved)
    async with pg_pool.acquire() as conn:
        members = await conn.fetch("SELECT normalized_address FROM cluster_members")
        member_addrs = {m["normalized_address"] for m in members}
        assert member_addrs == {"alice", "bob"}


# ---------------------------------------------------------------------------
# Case 7: Idempotence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_7_idempotence(clean_pg, detector):
    """Case 7 — idempotence.
    
    Running detection repeatedly on the same transaction yields identical results
    and does not corrupt or duplicate database records.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_idem",
            input_addresses=["sender1"],
            output_addresses=["recipient1", "sender1"]
        )
    
    inf1 = await detector.detect_and_save("tx_idem")
    inf2 = await detector.detect_and_save("tx_idem")
    
    assert len(inf1) == len(inf2) == 2
    assert [i.classification for i in inf1] == [i.classification for i in inf2]
    assert [i.confidence for i in inf1] == [i.confidence for i in inf2]
    
    async with clean_pg.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM bitcoin_change_inferences WHERE txid = 'tx_idem'"
        )
        assert count == 2


# ---------------------------------------------------------------------------
# Case 8: Malformed / Incomplete Data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_case_8_malformed(clean_pg, detector):
    """Case 8 — malformed/incomplete transaction data.
    
    A transaction with no outputs returns an empty result safely.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_mal",
            input_addresses=["sender1"],
            output_addresses=[]
        )
    inferences = await detector.detect_transaction("tx_mal")
    assert len(inferences) == 0


# ---------------------------------------------------------------------------
# Adversarial Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_equal_value(clean_pg, detector):
    """Adversarial: equal-value outputs / multiple fresh outputs.
    
    The detector must not guess between fresh outputs of equal standing.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_adv",
            input_addresses=["sender1"],
            output_addresses=["fresh1", "fresh2"]
        )
    inferences = await detector.detect_transaction("tx_adv")
    assert all(i.classification == "UNKNOWN" for i in inferences)
    assert all(i.confidence == "UNKNOWN" for i in inferences)


@pytest.mark.asyncio
async def test_adversarial_insufficient_history(clean_pg, detector):
    """Adversarial: isolated transaction with zero historical context.
    
    Without prior transaction history, reuse elimination cannot be used.
    Both outputs must remain UNKNOWN.
    """
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_isolated",
            input_addresses=["solo_sender"],
            output_addresses=["out_a", "out_b"]
        )
    inferences = await detector.detect_transaction("tx_isolated")
    assert len(inferences) == 2
    assert all(i.classification == "UNKNOWN" for i in inferences)
