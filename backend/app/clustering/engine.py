"""CHAKRA Step 2C: Clustering engine orchestrator.

Orchestrates the full Step 2C pipeline:
  Phase 1: Evidence generation — reads canonical Bitcoin UTXO data from PostgreSQL,
           generates co-input evidence, persists to clustering_evidence.
  Phase 2: Cluster computation — reads all evidence, runs deterministic Union-Find,
           persists cluster results to address_clusters and cluster_members.

CRITICAL SEMANTIC BOUNDARY:
  Clustering results represent addresses linked by deterministic blockchain
  co-spending heuristics. They do NOT represent confirmed real-world entities.

No canonical data is modified. No Neo4j writes are performed.
"""
from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from app.clustering.evidence import CoInputEvidenceGenerator, DEFAULT_BATCH_SIZE
from app.clustering.evidence_account import AccountDepositReuseGenerator, AccountSharedFundingGenerator
from app.clustering.clusterer import DeterministicClusterer
from app.clustering.repository import ClusteringRepository
from app.clustering.models import ClusteringError, ClusteringResult

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """Orchestrates the full Step 2C deterministic clustering pipeline.

    Usage:
        engine = ClusteringEngine(pg_pool)
        result = await engine.run_full_pipeline()
    """

    def __init__(self, pg_pool: asyncpg.Pool, batch_size: int = DEFAULT_BATCH_SIZE):
        self.pg_pool = pg_pool
        self.batch_size = batch_size
        self._repo = ClusteringRepository(pg_pool)
        self._evidence_gen = CoInputEvidenceGenerator(pg_pool, batch_size=batch_size)
        self._deposit_gen = AccountDepositReuseGenerator(pg_pool, batch_size=batch_size)
        self._funding_gen = AccountSharedFundingGenerator(pg_pool, batch_size=batch_size)
        self._clusterer = DeterministicClusterer()

    async def init_schema(self) -> None:
        """Apply the clustering schema migration idempotently."""
        await self._repo.init_schema()

    async def generate_evidence(self) -> ClusteringResult:
        """Phase 1: Generate and persist clustering evidence.

        Returns a ClusteringResult with evidence_summary populated.
        """
        result = ClusteringResult()
        try:
            # Bitcoin Co-Input
            btc_records, btc_summary = await self._evidence_gen.generate_all_evidence()
            
            # Account-Model Deposit Reuse
            dep_records, dep_summary = await self._deposit_gen.generate_all_evidence()
            
            # Account-Model Shared Funding
            fund_records, fund_summary = await self._funding_gen.generate_all_evidence()

            # Aggregate
            all_records = btc_records + dep_records + fund_records
            
            # Aggregate summary
            summary = btc_summary
            summary.account_deposit_evidence_generated = dep_summary.account_deposit_evidence_generated
            summary.account_funding_evidence_generated = fund_summary.account_funding_evidence_generated

            await self._repo.upsert_evidence(all_records)
            result.evidence_summary = summary
            result.total_evidence_records = len(all_records)
            logger.info(
                "Phase 1 complete: %d evidence records generated and persisted.",
                len(all_records),
            )
        except Exception as e:
            logger.exception("Phase 1 (evidence generation) failed.")
            result.success = False
            result.error = str(e)
            raise ClusteringError("evidence_generation", str(e), original_error=e) from e
        return result

    async def compute_clusters(self, replace: bool = True) -> ClusteringResult:
        """Phase 2: Compute clusters from persisted evidence and persist results.

        Args:
            replace: If True (default), atomically replaces cluster_members and
                     address_clusters so obsolete clusters and memberships do not remain.
                     If False, upserts with ON CONFLICT DO NOTHING.

        Returns a ClusteringResult with cluster metrics populated.
        """
        result = ClusteringResult()
        try:
            evidence_records = await self._repo.get_all_evidence()
            result.total_evidence_records = len(evidence_records)

            clusters = self._clusterer.compute_clusters(evidence_records)

            if replace:
                await self._repo.replace_clusters(clusters)
            else:
                await self._repo.upsert_clusters(clusters)

            result.clusters_generated = len(clusters)
            result.addresses_assigned_to_clusters = sum(c.member_count for c in clusters)

            logger.info(
                "Phase 2 complete: %d clusters, %d addresses assigned.",
                result.clusters_generated,
                result.addresses_assigned_to_clusters,
            )
        except ClusteringError:
            raise
        except Exception as e:
            logger.exception("Phase 2 (cluster computation) failed.")
            result.success = False
            result.error = str(e)
            raise ClusteringError("cluster_computation", str(e), original_error=e) from e
        return result

    async def run_full_pipeline(
        self, rebuild: bool = False, replace_clusters: bool = True
    ) -> ClusteringResult:
        """Run the Step 2C pipeline: canonical data -> evidence -> clusters.

        Args:
            rebuild: If True, clears all derived clustering state first.
            replace_clusters: If True (default), atomically replaces cluster tables.

        Returns a merged ClusteringResult with all observability metrics.
        """
        final_result = ClusteringResult()

        if rebuild:
            await self._repo.clear_clustering_state()

        # Phase 1
        try:
            p1 = await self.generate_evidence()
            final_result.evidence_summary = p1.evidence_summary
            final_result.total_evidence_records = p1.total_evidence_records
        except ClusteringError as e:
            final_result.success = False
            final_result.error = str(e)
            return final_result

        # Phase 2
        try:
            p2 = await self.compute_clusters(replace=replace_clusters)
            final_result.clusters_generated = p2.clusters_generated
            final_result.addresses_assigned_to_clusters = p2.addresses_assigned_to_clusters
        except ClusteringError as e:
            final_result.success = False
            final_result.error = str(e)
            return final_result

        logger.info(
            "Step 2C pipeline complete. Transactions examined: %d, Evidence: %d, "
            "Clusters: %d, Addresses assigned: %d",
            final_result.evidence_summary.transactions_examined,
            final_result.total_evidence_records,
            final_result.clusters_generated,
            final_result.addresses_assigned_to_clusters,
        )
        return final_result

    async def rebuild(self) -> ClusteringResult:
        """Perform a complete, deterministic rebuild of derived clustering state.

        Clears all derived clustering state (clustering_evidence, address_clusters,
        cluster_members) and recomputes everything from canonical PostgreSQL data.
        Ensures obsolete evidence, clusters, and memberships cannot survive.
        """
        return await self.run_full_pipeline(rebuild=True, replace_clusters=True)
