"""CHAKRA Step 2C: Comprehensive Clustering Test Suite.

Tests A through X covering all required scenarios:
A.  Two-input Bitcoin transaction -> one co-input evidence relationship.
B.  Three-input Bitcoin transaction -> exactly three pairwise evidence.
C.  Four-input Bitcoin transaction -> exactly six pairwise evidence.
D.  Single-input Bitcoin transaction -> zero clustering evidence.
E.  Coinbase transaction -> zero clustering evidence.
F.  Addressless input -> excluded from clustering.
G.  Duplicate input address -> deduplicated before pair generation.
H.  No self-clustering -> A never clusters with itself.
I.  Chain/network isolation -> same text on different chains stays distinct.
J.  Connected components A-B and B-C -> cluster {A, B, C}.
K.  Separate components A-B and C-D -> two clusters.
L.  Deterministic representative -> smallest canonical address identity.
M.  Deterministic cluster ID -> same members always same ID.
N.  Input-order invariance -> shuffled source records produce identical results.
O.  Database-order invariance -> different retrieval ordering identical results.
P.  Repeat execution -> identical output, no duplicates.
Q.  Large dataset/component -> no recursion-depth failure.
R.  Cross-chain collision -> no accidental merging.
S.  Token/asset irrelevance -> token symbol/contract does not cause clustering.
T.  Bitcoin output addresses -> outputs must NOT create co-input evidence.
U.  No synthetic sender->receiver inference.
V.  Evidence provenance -> every evidence record traceable to source transaction.
W.  Deterministic pair canonicalization -> A-B and B-A resolve to one evidence ID.
X.  Regression -> all existing Step 0-2B tests continue to pass.
Additional: Evidence generation summary accuracy.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import List

import asyncpg
import pytest

from app.clustering.clusterer import DeterministicClusterer, make_cluster_id, _DSU
from app.clustering.engine import ClusteringEngine
from app.clustering.evidence import (
    CoInputEvidenceGenerator,
    make_evidence_id,
    _generate_pairs_for_transaction,
)
from app.clustering.models import (
    AddressCluster,
    ClusteringEvidence,
    ClusterMember,
    EvidenceGenerationSummary,
)
from app.clustering.repository import ClusteringRepository
from app.graph.models import make_address_composite_id, normalize_address
from app.schemas.chain import Chain, Network

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
    r = ClusteringRepository(pg_pool)
    await r.init_schema()
    await r.clear_clustering_state()
    yield r
    await r.clear_clustering_state()


@pytest.fixture
async def engine(pg_pool, repo):
    e = ClusteringEngine(pg_pool)
    yield e


@pytest.fixture
async def clean_pg(pg_pool, repo):
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")
    yield pg_pool


async def _insert_btc_tx(
    conn,
    txid: str,
    network: str = "bitcoin-mainnet",
    input_addresses: List[str] = None,
    coinbase_inputs: int = 0,
    addressless_inputs: int = 0,
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
    for _ in range(coinbase_inputs):
        await conn.execute(
            "INSERT INTO bitcoin_vins (detail_pk, coinbase) VALUES ($1, TRUE)", detail_pk,
        )
    for _ in range(addressless_inputs):
        await conn.execute(
            "INSERT INTO bitcoin_vins (detail_pk, coinbase) VALUES ($1, FALSE)", detail_pk,
        )
    for i, addr in enumerate(output_addresses or []):
        await conn.execute(
            "INSERT INTO bitcoin_vouts (detail_pk, n, address, value_sat) VALUES ($1, $2, $3, 0)",
            detail_pk, i, addr,
        )


def _comp(addr: str, network: str = "bitcoin-mainnet") -> str:
    return make_address_composite_id("bitcoin", network, addr)


@pytest.mark.asyncio
async def test_a_two_input_one_evidence(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_a", input_addresses=["addr1", "addr2"])
    result = await engine.run_full_pipeline()
    assert result.success is True
    assert result.evidence_summary.evidence_relationships_generated == 1
    assert result.clusters_generated == 1
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.evidence_type == "bitcoin_co_input"
    assert ev.evidence_status == "observed_heuristic"
    assert ev.transaction_id == "tx_a"
    assert ev.chain == "bitcoin"


@pytest.mark.asyncio
async def test_b_three_input_three_evidence(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_b", input_addresses=["a1", "a2", "a3"])
    result = await engine.run_full_pipeline()
    assert result.success is True
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 3, f"Expected 3 evidence pairs, got {len(evidence)}"
    for ev in evidence:
        assert ev.transaction_id == "tx_b"
    pair_ids = {ev.evidence_id for ev in evidence}
    assert len(pair_ids) == 3


@pytest.mark.asyncio
async def test_c_four_input_six_evidence(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_c", input_addresses=["c1", "c2", "c3", "c4"])
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 6, f"Expected 6 pairs (4 choose 2), got {len(evidence)}"


@pytest.mark.asyncio
async def test_d_single_input_zero_evidence(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_d", input_addresses=["solo_addr"])
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 0
    assert result.clusters_generated == 0
    assert result.evidence_summary.evidence_relationships_generated == 0


@pytest.mark.asyncio
async def test_e_coinbase_zero_evidence(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_e", coinbase_inputs=1)
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 0
    assert result.evidence_summary.coinbase_inputs_skipped == 1
    assert result.evidence_summary.evidence_relationships_generated == 0


@pytest.mark.asyncio
async def test_f_addressless_excluded(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_f1", input_addresses=["real_addr"], addressless_inputs=1)
        await _insert_btc_tx(conn, "tx_f2", input_addresses=["f_a", "f_b"], addressless_inputs=1)
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 1
    assert result.evidence_summary.addressless_inputs_skipped == 2


@pytest.mark.asyncio
async def test_g_duplicate_input_deduplication(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_g", input_addresses=["dup_addr", "dup_addr", "other_addr"])
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 1, f"Expected 1 pair after dedup, got {len(evidence)}"
    assert result.evidence_summary.duplicate_addresses_removed == 1


@pytest.mark.asyncio
async def test_h_no_self_clustering():
    pairs = _generate_pairs_for_transaction("bitcoin-mainnet", "tx_h", ["self_addr"])
    assert len(pairs) == 0
    pairs = _generate_pairs_for_transaction("bitcoin-mainnet", "tx_h2", ["self_addr", "self_addr", "self_addr"])
    assert len(pairs) == 0


@pytest.mark.asyncio
async def test_i_chain_network_isolation(clean_pg, repo, engine):
    same_addr = "0xshared_address"
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_i_mainnet", network="bitcoin-mainnet",
                             input_addresses=[same_addr, "btc_mainnet_only"])
        await _insert_btc_tx(conn, "tx_i_testnet", network="bitcoin-testnet",
                             input_addresses=[same_addr, "btc_testnet_only"])
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 2
    networks_in_evidence = {ev.network for ev in evidence}
    assert "bitcoin-mainnet" in networks_in_evidence
    assert "bitcoin-testnet" in networks_in_evidence
    for ev in evidence:
        assert ev.network in ev.address_a_composite_id
        assert ev.network in ev.address_b_composite_id
    summary = await repo.get_cluster_summary()
    assert summary["total_clusters"] == 2


@pytest.mark.asyncio
async def test_j_connected_components_merge(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_j1", input_addresses=["j_addr_a", "j_addr_b"])
        await _insert_btc_tx(conn, "tx_j2", input_addresses=["j_addr_b", "j_addr_c"])
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 2
    assert result.clusters_generated == 1
    assert result.addresses_assigned_to_clusters == 3
    summary = await repo.get_cluster_summary()
    assert summary["total_cluster_members"] == 3


@pytest.mark.asyncio
async def test_k_separate_components_two_clusters(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_k1", input_addresses=["k_a", "k_b"])
        await _insert_btc_tx(conn, "tx_k2", input_addresses=["k_c", "k_d"])
    result = await engine.run_full_pipeline()
    assert result.clusters_generated == 2
    summary = await repo.get_cluster_summary()
    assert summary["total_cluster_members"] == 4
    assert summary["total_clusters"] == 2


@pytest.mark.asyncio
async def test_l_deterministic_representative(clean_pg, repo, engine):
    addr_a = "zzz_addr"
    addr_b = "aaa_addr"
    comp_a = _comp(addr_a)
    comp_b = _comp(addr_b)
    expected_rep = min(comp_a, comp_b)
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_l", input_addresses=[addr_a, addr_b])
    await engine.run_full_pipeline()
    async with repo.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT representative_composite_id FROM address_clusters LIMIT 1")
    assert row is not None
    assert row["representative_composite_id"] == expected_rep


@pytest.mark.asyncio
async def test_m_deterministic_cluster_id():
    members = [
        "bitcoin:bitcoin-mainnet:addr_c",
        "bitcoin:bitcoin-mainnet:addr_a",
        "bitcoin:bitcoin-mainnet:addr_b",
    ]
    sorted_members = sorted(members)
    id1 = make_cluster_id(sorted_members)
    id2 = make_cluster_id(sorted_members)
    assert id1 == id2
    id3 = make_cluster_id(sorted(members[::-1]))
    assert id1 == id3
    assert len(id1) == 64
    int(id1, 16)


@pytest.mark.asyncio
async def test_n_input_order_invariance():
    clusterer = DeterministicClusterer()
    ev1 = ClusteringEvidence(
        evidence_id="e1", evidence_type="bitcoin_co_input", evidence_status="observed_heuristic",
        chain="bitcoin", network="bitcoin-mainnet", transaction_id="tx1",
        address_a_composite_id="bitcoin:bitcoin-mainnet:addr_a",
        address_b_composite_id="bitcoin:bitcoin-mainnet:addr_b",
        address_a_normalized="addr_a", address_b_normalized="addr_b",
    )
    ev2 = ClusteringEvidence(
        evidence_id="e2", evidence_type="bitcoin_co_input", evidence_status="observed_heuristic",
        chain="bitcoin", network="bitcoin-mainnet", transaction_id="tx2",
        address_a_composite_id="bitcoin:bitcoin-mainnet:addr_b",
        address_b_composite_id="bitcoin:bitcoin-mainnet:addr_c",
        address_a_normalized="addr_b", address_b_normalized="addr_c",
    )
    clusters_fwd = clusterer.compute_clusters([ev1, ev2])
    clusters_rev = clusterer.compute_clusters([ev2, ev1])
    assert len(clusters_fwd) == len(clusters_rev) == 1
    assert clusters_fwd[0].cluster_id == clusters_rev[0].cluster_id
    fwd_members = {m.composite_id for m in clusters_fwd[0].members}
    rev_members = {m.composite_id for m in clusters_rev[0].members}
    assert fwd_members == rev_members


@pytest.mark.asyncio
async def test_o_database_order_invariance():
    clusterer = DeterministicClusterer()
    base_evidence = [
        ClusteringEvidence(
            evidence_id=f"ev_{i}", evidence_type="bitcoin_co_input",
            evidence_status="observed_heuristic",
            chain="bitcoin", network="bitcoin-mainnet", transaction_id=f"tx_{i}",
            address_a_composite_id=f"bitcoin:bitcoin-mainnet:z_addr_{i}",
            address_b_composite_id=f"bitcoin:bitcoin-mainnet:a_addr_{i}",
            address_a_normalized=f"z_addr_{i}", address_b_normalized=f"a_addr_{i}",
        )
        for i in range(5)
    ]
    shuffled = list(base_evidence)
    random.shuffle(shuffled)
    clusters_orig = clusterer.compute_clusters(base_evidence)
    clusters_shuf = clusterer.compute_clusters(shuffled)
    orig_ids = {c.cluster_id for c in clusters_orig}
    shuf_ids = {c.cluster_id for c in clusters_shuf}
    assert orig_ids == shuf_ids


@pytest.mark.asyncio
async def test_p_repeat_execution_idempotent(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_p", input_addresses=["p_a", "p_b", "p_c"])
    r1 = await engine.run_full_pipeline()
    r2 = await engine.run_full_pipeline()
    assert r1.success and r2.success
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 3
    summary = await repo.get_cluster_summary()
    assert summary["total_clusters"] == 1
    assert summary["total_cluster_members"] == 3


@pytest.mark.asyncio
async def test_q_large_component_no_recursion():
    dsu = _DSU()
    n = 2000
    for i in range(n + 1):
        dsu.add(f"node_{i:04d}")
    for i in range(n):
        dsu.union(f"node_{i:04d}", f"node_{i+1:04d}")
    root = dsu.find("node_0000")
    for i in range(1, n + 1):
        assert dsu.find(f"node_{i:04d}") == root
    components = dsu.components()
    assert len(components) == 1
    members = list(components.values())[0]
    assert len(members) == n + 1


@pytest.mark.asyncio
async def test_r_cross_chain_collision():
    same_text = "same_address_text"
    ev1 = ClusteringEvidence(
        evidence_id="ev_mainnet", evidence_type="bitcoin_co_input",
        evidence_status="observed_heuristic",
        chain="bitcoin", network="bitcoin-mainnet", transaction_id="tx_mainnet",
        address_a_composite_id=f"bitcoin:bitcoin-mainnet:{same_text}",
        address_b_composite_id="bitcoin:bitcoin-mainnet:other_mainnet",
        address_a_normalized=same_text, address_b_normalized="other_mainnet",
    )
    ev2 = ClusteringEvidence(
        evidence_id="ev_testnet", evidence_type="bitcoin_co_input",
        evidence_status="observed_heuristic",
        chain="bitcoin", network="bitcoin-testnet", transaction_id="tx_testnet",
        address_a_composite_id=f"bitcoin:bitcoin-testnet:{same_text}",
        address_b_composite_id="bitcoin:bitcoin-testnet:other_testnet",
        address_a_normalized=same_text, address_b_normalized="other_testnet",
    )
    clusterer = DeterministicClusterer()
    clusters = clusterer.compute_clusters([ev1, ev2])
    assert len(clusters) == 2
    cluster_ids = {c.cluster_id for c in clusters}
    assert len(cluster_ids) == 2
    mainnet_cluster = next(c for c in clusters if "bitcoin-mainnet" in c.representative_composite_id)
    testnet_cluster = next(c for c in clusters if "bitcoin-testnet" in c.representative_composite_id)
    mainnet_comp_ids = {m.composite_id for m in mainnet_cluster.members}
    testnet_comp_ids = {m.composite_id for m in testnet_cluster.members}
    assert mainnet_comp_ids.isdisjoint(testnet_comp_ids)


@pytest.mark.asyncio
async def test_s_token_asset_irrelevance(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        tx_pk = await conn.fetchval(
            """
            INSERT INTO transactions (
                transaction_id, chain, network, native_value, native_value_unit,
                transaction_type, status, from_address, to_address
            ) VALUES ('evm_tx_s', 'evm', 'ethereum-mainnet', 1000, 'wei',
                      'transfer', 'success', '0xfromaddr', '0xtoaddr')
            RETURNING id
            """,
        )
        await conn.execute(
            """
            INSERT INTO transfers (
                transaction_pk, from_address, to_address, asset_type, asset_symbol,
                amount, amount_unit, chain, network
            ) VALUES ($1, '0xfromaddr', '0xtoaddr', 'token', 'USDC',
                      1000, 'wei', 'evm', 'ethereum-mainnet')
            """,
            tx_pk,
        )
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 0
    assert result.clusters_generated == 0


@pytest.mark.asyncio
async def test_t_bitcoin_outputs_not_clustered(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_t",
            input_addresses=["t_input_only"],
            output_addresses=["t_out_1", "t_out_2", "t_out_3"],
        )
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 0, "Outputs must not generate co-input evidence"
    assert result.clusters_generated == 0


@pytest.mark.asyncio
async def test_u_no_synthetic_sender_receiver(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_u",
            input_addresses=["u_in_1", "u_in_2"],
            output_addresses=["u_out_1", "u_out_2"],
        )
    result = await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 1
    ev = evidence[0]
    assert "u_out_1" not in ev.address_a_normalized
    assert "u_out_2" not in ev.address_a_normalized
    assert "u_out_1" not in ev.address_b_normalized
    assert "u_out_2" not in ev.address_b_normalized


@pytest.mark.asyncio
async def test_v_evidence_provenance(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_v_source_tx", input_addresses=["v_a", "v_b"])
    await engine.run_full_pipeline()
    evidence = await repo.get_all_evidence()
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.transaction_id == "tx_v_source_tx"
    assert ev.chain == "bitcoin"
    assert ev.evidence_type == "bitcoin_co_input"
    async with repo.pool.acquire() as conn:
        tx_row = await conn.fetchrow(
            "SELECT id FROM transactions WHERE transaction_id = $1 AND chain = 'bitcoin'",
            ev.transaction_id,
        )
    assert tx_row is not None


@pytest.mark.asyncio
async def test_w_pair_canonicalization():
    network = "bitcoin-mainnet"
    txid = "tx_w"
    addr_x = "zzz_addr_x"
    addr_y = "aaa_addr_y"
    comp_x = _comp(addr_x, network)
    comp_y = _comp(addr_y, network)
    lo, hi = (comp_x, comp_y) if comp_x <= comp_y else (comp_y, comp_x)
    ev_id_expected = make_evidence_id(network, txid, lo, hi)
    pairs_fwd = _generate_pairs_for_transaction(network, txid, [addr_x, addr_y])
    pairs_rev = _generate_pairs_for_transaction(network, txid, [addr_y, addr_x])
    assert len(pairs_fwd) == 1
    assert len(pairs_rev) == 1
    assert pairs_fwd[0].evidence_id == pairs_rev[0].evidence_id
    assert pairs_fwd[0].evidence_id == ev_id_expected
    for pairs in [pairs_fwd, pairs_rev]:
        ev = pairs[0]
        assert ev.address_a_composite_id <= ev.address_b_composite_id


@pytest.mark.asyncio
async def test_evidence_summary_accuracy(clean_pg, repo, engine):
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(
            conn, "tx_summary_1",
            input_addresses=["s1", "s2", "s3"],
            coinbase_inputs=1,
            addressless_inputs=1,
        )
        await _insert_btc_tx(conn, "tx_summary_2", input_addresses=["s4"])
    result = await engine.run_full_pipeline()
    s = result.evidence_summary
    assert s.transactions_examined == 2
    assert s.eligible_bitcoin_transactions == 1
    assert s.coinbase_inputs_skipped == 1
    assert s.addressless_inputs_skipped == 1
    assert s.input_addresses_examined == 4
    assert s.duplicate_addresses_removed == 0
    assert s.evidence_relationships_generated == 3


# ===========================================================================
# Lifecycle Refinement Tests: A through H
# ===========================================================================

@pytest.mark.asyncio
async def test_lifecycle_rebuild_dataset_expansion(clean_pg, repo, engine):
    """Lifecycle audit test verifying full-rebuild semantics upon dataset expansion:
    A. Initial dataset produces A-B cluster.
    B. Canonical dataset is expanded with B-C evidence.
    C. Rebuild produces exactly A-B-C.
    D. No obsolete cluster membership remains.
    E. No obsolete cluster remains.
    F. No duplicate evidence exists.
    G. Direct evidence remains exactly A-B and B-C; no synthetic A-C evidence appears.
    H. Re-running rebuild with identical source data is deterministic and idempotent.
    """
    comp_a = _comp("life_a")
    comp_b = _comp("life_b")
    comp_c = _comp("life_c")

    # A. Initial dataset produces A-B cluster
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_life_1", input_addresses=["life_a", "life_b"])

    r1 = await engine.run_full_pipeline()
    assert r1.success is True
    assert r1.clusters_generated == 1
    assert r1.addresses_assigned_to_clusters == 2

    # Capture initial cluster ID
    async with repo.pool.acquire() as conn:
        initial_cluster_rows = await conn.fetch("SELECT * FROM address_clusters")
        initial_member_rows = await conn.fetch("SELECT * FROM cluster_members")
    assert len(initial_cluster_rows) == 1
    assert len(initial_member_rows) == 2
    initial_cluster_id = initial_cluster_rows[0]["cluster_id"]

    # B. Canonical dataset is expanded with B-C evidence
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_life_2", input_addresses=["life_b", "life_c"])

    # C. Rebuild produces exactly A-B-C
    r2 = await engine.rebuild()
    assert r2.success is True
    assert r2.clusters_generated == 1
    assert r2.addresses_assigned_to_clusters == 3

    async with repo.pool.acquire() as conn:
        clusters_after = await conn.fetch("SELECT * FROM address_clusters")
        members_after = await conn.fetch("SELECT * FROM cluster_members")
        evidence_after = await conn.fetch("SELECT * FROM clustering_evidence ORDER BY evidence_id")

    # E. No obsolete cluster remains (only 1 cluster total)
    assert len(clusters_after) == 1, "Obsolete cluster must not remain after rebuild"
    new_cluster_id = clusters_after[0]["cluster_id"]
    assert new_cluster_id != initial_cluster_id, "Merged cluster must have new deterministic cluster_id"
    assert clusters_after[0]["member_count"] == 3

    # D. No obsolete cluster membership remains
    assert len(members_after) == 3, "Obsolete cluster memberships must not remain after rebuild"
    member_comp_ids = {m["composite_id"] for m in members_after}
    assert member_comp_ids == {comp_a, comp_b, comp_c}
    for m in members_after:
        assert m["cluster_id"] == new_cluster_id, "All members must point to the new merged cluster"

    # F. No duplicate evidence exists
    assert len(evidence_after) == 2, "Exactly 2 direct evidence records expected"
    ev_ids = {e["evidence_id"] for e in evidence_after}
    assert len(ev_ids) == 2

    # G. Direct evidence remains exactly A-B and B-C; no synthetic A-C evidence appears
    pair_tuples = {
        (e["address_a_composite_id"], e["address_b_composite_id"])
        for e in evidence_after
    }
    expected_ab = (min(comp_a, comp_b), max(comp_a, comp_b))
    expected_bc = (min(comp_b, comp_c), max(comp_b, comp_c))
    assert pair_tuples == {expected_ab, expected_bc}

    # Verify no synthetic A-C pair in evidence
    forbidden_ac = (min(comp_a, comp_c), max(comp_a, comp_c))
    assert forbidden_ac not in pair_tuples, "Direct evidence must not contain synthetic A-C pair"

    # H. Re-running rebuild with identical source data is deterministic and idempotent
    r3 = await engine.rebuild()
    assert r3.success is True
    assert r3.clusters_generated == 1
    assert r3.addresses_assigned_to_clusters == 3

    summary_after = await repo.get_cluster_summary()
    assert summary_after["total_clusters"] == 1
    assert summary_after["total_cluster_members"] == 3
    assert summary_after["total_evidence_records"] == 2

    async with repo.pool.acquire() as conn:
        re_clusters = await conn.fetch("SELECT * FROM address_clusters")
    assert re_clusters[0]["cluster_id"] == new_cluster_id


@pytest.mark.asyncio
async def test_lifecycle_pipeline_replaces_clusters_on_expansion(clean_pg, repo, engine):
    """Verify that run_full_pipeline with default replace_clusters=True also
    prevents obsolete clusters and obsolete memberships when data expands incrementally.
    """
    comp_x = _comp("life_x")
    comp_y = _comp("life_y")
    comp_z = _comp("life_z")

    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_inc_1", input_addresses=["life_x", "life_y"])

    r1 = await engine.run_full_pipeline()
    assert r1.clusters_generated == 1

    # Expand dataset
    async with clean_pg.acquire() as conn:
        await _insert_btc_tx(conn, "tx_inc_2", input_addresses=["life_y", "life_z"])

    r2 = await engine.run_full_pipeline()
    assert r2.clusters_generated == 1
    assert r2.addresses_assigned_to_clusters == 3

    # Check that old cluster and memberships were replaced
    summary = await repo.get_cluster_summary()
    assert summary["total_clusters"] == 1
    assert summary["total_cluster_members"] == 3
    assert summary["total_evidence_records"] == 2

