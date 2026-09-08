"""CHAKRA Step 3E: Account-Model Clustering Tests.

Validates the semantic boundaries of Account Relationship Evidence vs. Hard Entity Clustering:
1. Bitcoin co-input spending (H1) forms hard entity clusters via Union-Find.
2. Account deposit-address reuse generates relationship evidence in clustering_evidence,
   but does NOT merge depositors and does NOT put the deposit address into a hard cluster.
3. Account shared-funding generates relationship evidence in clustering_evidence,
   but does NOT merge recipients and does NOT put the funder into a hard cluster.
4. Mixed-heuristic transitive collapse is strictly prevented.
5. Evidence provenance, idempotency, determinism, and candidate-selection boundaries.
6. Cross-chain isolation between Bitcoin hard clusters and Account relationship records.
"""
from __future__ import annotations

import os
import pytest
from datetime import datetime, timezone
from typing import List

import asyncpg

from app.schemas.chain import Chain, Network
from app.clustering.evidence_account import (
    AccountDepositReuseGenerator,
    AccountSharedFundingGenerator,
    DEPOSIT_REUSE_TYPE,
    SHARED_FUNDING_TYPE,
    MAX_COUNTERPARTIES,
)
from app.clustering.engine import ClusteringEngine
from app.graph.models import make_address_composite_id

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
async def setup_account_transfers(pg_pool: asyncpg.Pool):
    """Inserts mock transactions and transfers for account clustering tests."""
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence, "
            "transfers, bitcoin_vouts, bitcoin_vins, bitcoin_transaction_details, "
            "transaction_provenance, transactions RESTART IDENTITY CASCADE"
        )

        async def insert_transfer(txid: str, chain: str, from_addr: str, to_addr: str):
            tx_row = await conn.fetchrow(
                """
                INSERT INTO transactions (
                    transaction_id, chain, network, timestamp, 
                    native_value, native_value_unit, transaction_type, status
                ) VALUES ($1, $2, 'mainnet', $3, 0, 'wei', 'transfer', 'success')
                RETURNING id
                """,
                txid, chain, datetime.now(timezone.utc),
            )
            await conn.execute(
                """
                INSERT INTO transfers (
                    transaction_pk, chain, network, from_address, to_address, 
                    asset_type, asset_symbol, amount, amount_unit
                ) VALUES ($1, $2, 'mainnet', $3, $4, 'native', 'ETH', 100, 'wei')
                """,
                tx_row["id"], chain, from_addr, to_addr,
            )

        # Deposit Reuse: User1 and User2 send to Deposit1
        await insert_transfer("tx_dep1", "ethereum", "user1", "deposit1")
        await insert_transfer("tx_dep2", "ethereum", "user2", "deposit1")

        # Shared Funding: Funding1 sends to User3 and User4
        await insert_transfer("tx_fund1", "ethereum", "funding1", "user3")
        await insert_transfer("tx_fund2", "ethereum", "funding1", "user4")

        # Exceeds candidate-selection limit: 51 senders to massive_deposit
        for i in range(MAX_COUNTERPARTIES + 1):
            await insert_transfer(f"tx_massive_dep_{i}", "ethereum", f"bot{i}", "massive_deposit")

        # Exceeds candidate-selection limit: massive_funding to 51 receivers
        for i in range(MAX_COUNTERPARTIES + 1):
            await insert_transfer(f"tx_massive_fund_{i}", "ethereum", "massive_funding", f"bot{i}")

        # Cross-chain isolation check: Same address names on Tron
        await insert_transfer("tx_tron_dep", "tron", "user1", "deposit1")
        await insert_transfer("tx_tron_dep2", "tron", "user3", "deposit1")


# ---------------------------------------------------------------------------
# Test 1 — Bitcoin hard clustering remains intact
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bitcoin_hard_clustering_remains_intact(pg_pool: asyncpg.Pool):
    """Test 1: Two addresses participating in the same Bitcoin co-input transaction
    still enter the exact same hard entity cluster.
    """
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence, "
            "transfers, bitcoin_vouts, bitcoin_vins, bitcoin_transaction_details, transactions "
            "RESTART IDENTITY CASCADE"
        )
        tx_pk = await conn.fetchval(
            """
            INSERT INTO transactions (transaction_id, chain, network, native_value, native_value_unit, transaction_type, status)
            VALUES ('btc_coinput_tx', 'bitcoin', 'bitcoin-mainnet', 0, 'sat', 'transfer', 'success')
            RETURNING id
            """
        )
        dt_pk = await conn.fetchval(
            "INSERT INTO bitcoin_transaction_details (transaction_pk, txid, total_output_sat) VALUES ($1, 'btc_coinput_tx', 0) RETURNING id",
            tx_pk,
        )
        await conn.execute("INSERT INTO bitcoin_vins (detail_pk, address, coinbase) VALUES ($1, 'btc_in_1', FALSE)", dt_pk)
        await conn.execute("INSERT INTO bitcoin_vins (detail_pk, address, coinbase) VALUES ($1, 'btc_in_2', FALSE)", dt_pk)

    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline(rebuild=True)
    assert result.success is True
    assert result.clusters_generated == 1
    assert result.addresses_assigned_to_clusters == 2

    async with pg_pool.acquire() as conn:
        comp_1 = make_address_composite_id("bitcoin", "bitcoin-mainnet", "btc_in_1")
        comp_2 = make_address_composite_id("bitcoin", "bitcoin-mainnet", "btc_in_2")
        members = await conn.fetch("SELECT cluster_id, composite_id FROM cluster_members WHERE composite_id = ANY($1)", [comp_1, comp_2])
        assert len(members) == 2
        # Both share the same cluster_id
        assert members[0]["cluster_id"] == members[1]["cluster_id"]


# ---------------------------------------------------------------------------
# Test 2 — Shared funding does NOT create hard cluster
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shared_funding_does_not_create_hard_cluster(pg_pool: asyncpg.Pool, setup_account_transfers):
    """Test 2: Given F -> A and F -> B:
    - account_shared_funding evidence exists in clustering_evidence
    - F, A, B are NOT merged into a hard entity cluster by account_shared_funding alone
    """
    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline(rebuild=True)
    assert result.success is True

    async with pg_pool.acquire() as conn:
        # 1. Evidence exists in clustering_evidence
        ev_rows = await conn.fetch(
            "SELECT * FROM clustering_evidence WHERE evidence_type = $1", SHARED_FUNDING_TYPE
        )
        assert len(ev_rows) == 2  # funding1 -> user3, funding1 -> user4

        # 2. None of funding1, user3, user4 enter hard cluster_members
        fund_comps = [
            make_address_composite_id("ethereum", "mainnet", a)
            for a in ["funding1", "user3", "user4"]
        ]
        cluster_rows = await conn.fetch(
            "SELECT * FROM cluster_members WHERE composite_id = ANY($1)", fund_comps
        )
        assert len(cluster_rows) == 0, "Shared funding recipients must NOT enter hard entity clusters"


# ---------------------------------------------------------------------------
# Test 3 — Deposit infrastructure does NOT create hard cluster
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deposit_infrastructure_does_not_create_hard_cluster(pg_pool: asyncpg.Pool, setup_account_transfers):
    """Test 3: Given U1 -> D and U2 -> D:
    - deposit reuse evidence exists in clustering_evidence
    - D is NOT placed into the hard entity cluster merely because of this evidence
    - U1 and U2 are NOT merged solely by this evidence
    """
    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline(rebuild=True)
    assert result.success is True

    async with pg_pool.acquire() as conn:
        # 1. Evidence exists in clustering_evidence
        dep_ev = await conn.fetch(
            "SELECT * FROM clustering_evidence WHERE evidence_type = $1 AND chain = 'ethereum'",
            DEPOSIT_REUSE_TYPE,
        )
        assert len(dep_ev) == 2  # user1 -> deposit1, user2 -> deposit1

        # 2. Neither deposit1 nor depositors enter hard cluster_members
        dep_comps = [
            make_address_composite_id("ethereum", "mainnet", a)
            for a in ["user1", "user2", "deposit1"]
        ]
        cluster_rows = await conn.fetch(
            "SELECT * FROM cluster_members WHERE composite_id = ANY($1)", dep_comps
        )
        assert len(cluster_rows) == 0, "Deposit reuse must NOT merge depositors or deposit address into hard clusters"


# ---------------------------------------------------------------------------
# Test 4 — Mixed heuristic transitive collapse is prevented
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mixed_heuristic_transitive_collapse_prevented(pg_pool: asyncpg.Pool):
    """Test 4: Construct:
        F -> A
        F -> I
        A -> D
        S -> D
    Verify that account relationship evidence does NOT create {F, A, I, D, S}
    as one hard entity cluster.
    """
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence, transfers, transactions "
            "RESTART IDENTITY CASCADE"
        )

        async def add_transfer(txid: str, f: str, t: str):
            tx_row = await conn.fetchrow(
                "INSERT INTO transactions (transaction_id, chain, network, timestamp, native_value, native_value_unit, transaction_type, status) "
                "VALUES ($1, 'ethereum', 'mainnet', NOW(), 0, 'wei', 'transfer', 'success') RETURNING id",
                txid,
            )
            await conn.execute(
                "INSERT INTO transfers (transaction_pk, chain, network, from_address, to_address, asset_type, asset_symbol, amount, amount_unit) "
                "VALUES ($1, 'ethereum', 'mainnet', $2, $3, 'native', 'ETH', 1, 'wei')",
                tx_row["id"], f, t,
            )

        # Funder funds A and Innocent
        await add_transfer("tx_f1", "funder_corp", "user_a")
        await add_transfer("tx_f2", "funder_corp", "innocent_i")

        # user_a and scammer_s deposit to deposit_binance
        await add_transfer("tx_d1", "user_a", "deposit_binance")
        await add_transfer("tx_d2", "scammer_s", "deposit_binance")

    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline(rebuild=True)
    assert result.success is True

    async with pg_pool.acquire() as conn:
        # All 4 relationship evidence records exist in clustering_evidence
        ev_rows = await conn.fetch("SELECT * FROM clustering_evidence")
        assert len(ev_rows) == 4

        # ZERO hard clusters are formed from relationship evidence
        clusters = await conn.fetch("SELECT * FROM address_clusters")
        members = await conn.fetch("SELECT * FROM cluster_members")
        assert len(clusters) == 0, "No hard clusters must be formed from account relationship evidence"
        assert len(members) == 0


# ---------------------------------------------------------------------------
# Test 5 — Evidence provenance
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_account_evidence_provenance(pg_pool: asyncpg.Pool, setup_account_transfers):
    """Test 5: Every retained account-model evidence record must still have deterministic
    provenance back to the relevant transaction(s) in the transactions table.
    """
    engine = ClusteringEngine(pg_pool)
    await engine.generate_evidence()

    async with pg_pool.acquire() as conn:
        evidence_rows = await conn.fetch(
            "SELECT evidence_id, transaction_id, chain, network FROM clustering_evidence "
            "WHERE evidence_type IN ($1, $2)",
            DEPOSIT_REUSE_TYPE, SHARED_FUNDING_TYPE,
        )
        assert len(evidence_rows) > 0

        # Verify every transaction_id exists in canonical transactions table
        orphan_txs = await conn.fetch(
            "SELECT e.evidence_id, e.transaction_id FROM clustering_evidence e "
            "LEFT JOIN transactions t ON e.transaction_id = t.transaction_id "
            "WHERE t.id IS NULL AND e.evidence_type IN ($1, $2)",
            DEPOSIT_REUSE_TYPE, SHARED_FUNDING_TYPE,
        )
        assert len(orphan_txs) == 0, f"Found orphan evidence without transaction provenance: {orphan_txs}"


# ---------------------------------------------------------------------------
# Test 6 — Repeated evidence idempotency
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_account_clustering_idempotency(pg_pool: asyncpg.Pool, setup_account_transfers):
    """Test 6: Running evidence generation and pipeline repeatedly must not create
    duplicate logical evidence records.
    """
    engine = ClusteringEngine(pg_pool)

    r1 = await engine.run_full_pipeline(rebuild=True)
    assert r1.success is True
    total_ev_1 = r1.total_evidence_records

    # Second run without rebuild (incremental / repeat)
    r2 = await engine.run_full_pipeline(rebuild=False)
    assert r2.success is True
    assert r2.total_evidence_records == total_ev_1

    # Third run with full rebuild
    r3 = await engine.rebuild()
    assert r3.success is True
    assert r3.total_evidence_records == total_ev_1

    async with pg_pool.acquire() as conn:
        # Check uniqueness constraint on evidence_id
        count = await conn.fetchval("SELECT count(*) FROM clustering_evidence")
        distinct_count = await conn.fetchval("SELECT count(DISTINCT evidence_id) FROM clustering_evidence")
        assert count == distinct_count == total_ev_1


# ---------------------------------------------------------------------------
# Test 7 — Deterministic identifiers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deterministic_account_evidence_identifiers(pg_pool: asyncpg.Pool, setup_account_transfers):
    """Test 7: Identical input must produce identical evidence identifiers across runs."""
    gen = AccountDepositReuseGenerator(pg_pool)
    ev1, _ = await gen.generate_all_evidence()
    ev2, _ = await gen.generate_all_evidence()

    assert len(ev1) == len(ev2)
    ev_ids_1 = [e.evidence_id for e in ev1]
    ev_ids_2 = [e.evidence_id for e in ev2]
    assert ev_ids_1 == ev_ids_2


# ---------------------------------------------------------------------------
# Test 8 — Counterparty boundary behavior (Candidate-Selection Filter)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_counterparty_boundary_behavior(pg_pool: asyncpg.Pool):
    """Test 8: Preserves counterparty boundary tests (1, 2, 50, 51).
    Explicitly documents that 2–50 is an engineering candidate-selection filter
    to exclude high-throughput contracts/hot-wallets from pairwise indexing,
    NOT an entity-ownership threshold.
    """
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence, transfers, transactions "
            "RESTART IDENTITY CASCADE"
        )

        async def add_transfer(txid: str, f: str, t: str):
            tx_row = await conn.fetchrow(
                "INSERT INTO transactions (transaction_id, chain, network, timestamp, native_value, native_value_unit, transaction_type, status) "
                "VALUES ($1, 'ethereum', 'mainnet', NOW(), 0, 'wei', 'transfer', 'success') RETURNING id",
                txid,
            )
            await conn.execute(
                "INSERT INTO transfers (transaction_pk, chain, network, from_address, to_address, asset_type, asset_symbol, amount, amount_unit) "
                "VALUES ($1, 'ethereum', 'mainnet', $2, $3, 'native', 'ETH', 1, 'wei')",
                tx_row["id"], f, t,
            )

        # 1 counterparty: solo_dep -> 0 evidence
        await add_transfer("tx_b1", "solo_sender", "solo_dep")

        # 2 counterparties: lower candidate bound -> 2 evidence records
        await add_transfer("tx_b2_1", "sender2_a", "bound2_dep")
        await add_transfer("tx_b2_2", "sender2_b", "bound2_dep")

        # 50 counterparties: upper candidate bound -> 50 evidence records
        for i in range(50):
            await add_transfer(f"tx_b50_{i}", f"sender50_{i}", "bound50_dep")

        # 51 counterparties: exceeds candidate bound -> 0 evidence records (treated as high-degree infrastructure)
        for i in range(51):
            await add_transfer(f"tx_b51_{i}", f"sender51_{i}", "bound51_dep")

    gen = AccountDepositReuseGenerator(pg_pool)
    evidence, _ = await gen.generate_all_evidence()

    solo_ev = [e for e in evidence if "solo_dep" in (e.address_a_normalized, e.address_b_normalized)]
    b2_ev = [e for e in evidence if "bound2_dep" in (e.address_a_normalized, e.address_b_normalized)]
    b50_ev = [e for e in evidence if "bound50_dep" in (e.address_a_normalized, e.address_b_normalized)]
    b51_ev = [e for e in evidence if "bound51_dep" in (e.address_a_normalized, e.address_b_normalized)]

    assert len(solo_ev) == 0
    assert len(b2_ev) == 2
    assert len(b50_ev) == 50
    assert len(b51_ev) == 0


# ---------------------------------------------------------------------------
# Test 9 — Bitcoin/account isolation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bitcoin_account_isolation(pg_pool: asyncpg.Pool):
    """Test 9: Account-model relationship evidence does NOT alter Bitcoin
    hard-clustering semantics. When Bitcoin co-input and Account transfers
    co-exist in the database:
    - Bitcoin addresses enter hard entity clusters.
    - Account addresses do NOT enter hard entity clusters.
    - No cross-chain merging occurs.
    """
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence, "
            "transfers, bitcoin_vouts, bitcoin_vins, bitcoin_transaction_details, transactions "
            "RESTART IDENTITY CASCADE"
        )

        # 1. Insert Bitcoin 2-input transaction
        tx_pk = await conn.fetchval(
            """
            INSERT INTO transactions (transaction_id, chain, network, native_value, native_value_unit, transaction_type, status)
            VALUES ('btc_iso_tx', 'bitcoin', 'bitcoin-mainnet', 0, 'sat', 'transfer', 'success')
            RETURNING id
            """
        )
        dt_pk = await conn.fetchval(
            "INSERT INTO bitcoin_transaction_details (transaction_pk, txid, total_output_sat) VALUES ($1, 'btc_iso_tx', 0) RETURNING id",
            tx_pk,
        )
        await conn.execute("INSERT INTO bitcoin_vins (detail_pk, address, coinbase) VALUES ($1, 'btc_user_1', FALSE)", dt_pk)
        await conn.execute("INSERT INTO bitcoin_vins (detail_pk, address, coinbase) VALUES ($1, 'btc_user_2', FALSE)", dt_pk)

        # 2. Insert Ethereum deposit-reuse transfers
        eth_tx_1 = await conn.fetchval(
            "INSERT INTO transactions (transaction_id, chain, network, timestamp, native_value, native_value_unit, transaction_type, status) "
            "VALUES ('eth_iso_1', 'ethereum', 'mainnet', NOW(), 0, 'wei', 'transfer', 'success') RETURNING id"
        )
        eth_tx_2 = await conn.fetchval(
            "INSERT INTO transactions (transaction_id, chain, network, timestamp, native_value, native_value_unit, transaction_type, status) "
            "VALUES ('eth_iso_2', 'ethereum', 'mainnet', NOW(), 0, 'wei', 'transfer', 'success') RETURNING id"
        )
        await conn.execute(
            "INSERT INTO transfers (transaction_pk, chain, network, from_address, to_address, asset_type, asset_symbol, amount, amount_unit) "
            "VALUES ($1, 'ethereum', 'mainnet', 'eth_depositor_1', 'eth_dep_addr', 'native', 'ETH', 1, 'wei')",
            eth_tx_1,
        )
        await conn.execute(
            "INSERT INTO transfers (transaction_pk, chain, network, from_address, to_address, asset_type, asset_symbol, amount, amount_unit) "
            "VALUES ($1, 'ethereum', 'mainnet', 'eth_depositor_2', 'eth_dep_addr', 'native', 'ETH', 1, 'wei')",
            eth_tx_2,
        )

    engine = ClusteringEngine(pg_pool)
    result = await engine.run_full_pipeline(rebuild=True)
    assert result.success is True

    # 1 hard cluster generated (from Bitcoin co-input)
    assert result.clusters_generated == 1
    assert result.addresses_assigned_to_clusters == 2

    async with pg_pool.acquire() as conn:
        # Bitcoin cluster exists
        btc_members = await conn.fetch("SELECT * FROM cluster_members WHERE chain = 'bitcoin'")
        assert len(btc_members) == 2

        # ZERO Ethereum cluster members exist
        eth_members = await conn.fetch("SELECT * FROM cluster_members WHERE chain = 'ethereum'")
        assert len(eth_members) == 0

        # BUT Ethereum relationship evidence is fully recorded in clustering_evidence
        eth_ev = await conn.fetch("SELECT * FROM clustering_evidence WHERE chain = 'ethereum'")
        assert len(eth_ev) == 2
