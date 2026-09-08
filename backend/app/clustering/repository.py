"""CHAKRA Step 2C: Clustering persistence repository.

Manages the three derived analytical tables:
  - clustering_evidence
  - address_clusters
  - cluster_members

These tables are DERIVED STATE only. Canonical tables are never modified.

All writes use deterministic keys with ON CONFLICT DO NOTHING for idempotency.
Schema initialization applies the 002_clustering_schema.sql migration.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import asyncpg

from app.clustering.models import AddressCluster, ClusteringEvidence

logger = logging.getLogger(__name__)

# Path to the migration file relative to this file's location
_MIGRATION_FILE = Path(__file__).parent.parent / "db" / "migrations" / "002_clustering_schema.sql"


class ClusteringRepository:
    """Repository for derived Step 2C clustering analytical state.

    All methods are idempotent via ON CONFLICT DO NOTHING.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_schema(self) -> None:
        """Apply the clustering schema migration idempotently.

        Reads 002_clustering_schema.sql and executes it. All CREATE TABLE and
        CREATE INDEX statements use IF NOT EXISTS, so repeated calls are safe.
        """
        sql = _MIGRATION_FILE.read_text(encoding="utf-8-sig")
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("Clustering schema initialized (002_clustering_schema.sql applied).")

    async def upsert_evidence(self, records: List[ClusteringEvidence]) -> int:
        """Persist evidence records, ignoring conflicts on evidence_id.

        Returns the number of records inserted (0 for pre-existing records).
        """
        if not records:
            return 0

        data = [
            (
                r.evidence_id,
                r.evidence_type,
                r.evidence_status,
                r.chain,
                r.network,
                r.transaction_id,
                r.address_a_composite_id,
                r.address_b_composite_id,
                r.address_a_normalized,
                r.address_b_normalized,
            )
            for r in records
        ]

        async with self.pool.acquire() as conn:
            result = await conn.executemany(
                """
                INSERT INTO clustering_evidence (
                    evidence_id, evidence_type, evidence_status,
                    chain, network, transaction_id,
                    address_a_composite_id, address_b_composite_id,
                    address_a_normalized, address_b_normalized
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (evidence_id) DO NOTHING
                """,
                data,
            )

        inserted = len(records)
        logger.info("Upserted %d evidence records (conflicts silently ignored).", inserted)
        return inserted

    async def get_all_evidence(self) -> List[ClusteringEvidence]:
        """Read all persisted evidence records, ordered deterministically."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    evidence_id, evidence_type, evidence_status,
                    chain, network, transaction_id,
                    address_a_composite_id, address_b_composite_id,
                    address_a_normalized, address_b_normalized
                FROM clustering_evidence
                ORDER BY evidence_id ASC
                """
            )

        return [
            ClusteringEvidence(
                evidence_id=r["evidence_id"],
                evidence_type=r["evidence_type"],
                evidence_status=r["evidence_status"],
                chain=r["chain"],
                network=r["network"],
                transaction_id=r["transaction_id"],
                address_a_composite_id=r["address_a_composite_id"],
                address_b_composite_id=r["address_b_composite_id"],
                address_a_normalized=r["address_a_normalized"],
                address_b_normalized=r["address_b_normalized"],
            )
            for r in rows
        ]

    async def upsert_clusters(self, clusters: List[AddressCluster]) -> None:
        """Persist cluster and member records, ignoring conflicts.

        Clusters and members use deterministic unique keys, so repeated
        execution produces no duplicates.
        """
        if not clusters:
            return

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Upsert address_clusters
                cluster_data = [
                    (
                        c.cluster_id,
                        c.member_count,
                        c.representative_composite_id,
                        c.chain,
                        c.network,
                    )
                    for c in clusters
                ]
                await conn.executemany(
                    """
                    INSERT INTO address_clusters (
                        cluster_id, member_count, representative_composite_id, chain, network
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (cluster_id) DO NOTHING
                    """,
                    cluster_data,
                )

                # Upsert cluster_members
                member_data = [
                    (
                        m.cluster_id,
                        m.composite_id,
                        m.chain,
                        m.network,
                        m.normalized_address,
                    )
                    for c in clusters
                    for m in c.members
                ]
                if member_data:
                    await conn.executemany(
                        """
                        INSERT INTO cluster_members (
                            cluster_id, composite_id, chain, network, normalized_address
                        ) VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (cluster_id, composite_id) DO NOTHING
                        """,
                        member_data,
                    )

        logger.info(
            "Upserted %d clusters and %d member records.",
            len(clusters),
            sum(c.member_count for c in clusters),
        )

    async def replace_clusters(self, clusters: List[AddressCluster]) -> None:
        """Atomically replace all cluster and member records with current clusters.

        Runs in a single transaction:
        1. Deletes existing rows in cluster_members and address_clusters.
        2. Inserts current clusters and members.
        This guarantees obsolete clusters and obsolete memberships do not survive.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM cluster_members")
                await conn.execute("DELETE FROM address_clusters")

                if not clusters:
                    return

                cluster_data = [
                    (
                        c.cluster_id,
                        c.member_count,
                        c.representative_composite_id,
                        c.chain,
                        c.network,
                    )
                    for c in clusters
                ]
                await conn.executemany(
                    """
                    INSERT INTO address_clusters (
                        cluster_id, member_count, representative_composite_id, chain, network
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (cluster_id) DO NOTHING
                    """,
                    cluster_data,
                )

                member_data = [
                    (
                        m.cluster_id,
                        m.composite_id,
                        m.chain,
                        m.network,
                        m.normalized_address,
                    )
                    for c in clusters
                    for m in c.members
                ]
                if member_data:
                    await conn.executemany(
                        """
                        INSERT INTO cluster_members (
                            cluster_id, composite_id, chain, network, normalized_address
                        ) VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (cluster_id, composite_id) DO NOTHING
                        """,
                        member_data,
                    )

        logger.info(
            "Replaced clusters: %d clusters and %d member records persisted.",
            len(clusters),
            sum(c.member_count for c in clusters),
        )

    async def get_cluster_summary(self) -> dict:
        """Return aggregate cluster observability metrics."""
        async with self.pool.acquire() as conn:
            evidence_count = await conn.fetchval("SELECT COUNT(*) FROM clustering_evidence")
            cluster_count = await conn.fetchval("SELECT COUNT(*) FROM address_clusters")
            member_count = await conn.fetchval("SELECT COUNT(*) FROM cluster_members")

        return {
            "total_evidence_records": evidence_count or 0,
            "total_clusters": cluster_count or 0,
            "total_cluster_members": member_count or 0,
        }

    async def clear_clustering_state(self) -> None:
        """Remove all derived clustering data. FOR TEST ISOLATION ONLY.

        Does NOT affect canonical transaction or transfer tables.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE cluster_members, address_clusters, clustering_evidence")
        logger.info("Clustering state cleared (test isolation).")
