"""CHAKRA Step 3E: Account-model clustering evidence generators.

Generates deterministic clustering relationship evidence for Ethereum and Tron using:
1. Deposit-address reuse (account_deposit_reuse)
2. Shared-funding-source (account_shared_funding)

SEMANTIC BOUNDARY & EVIDENCE CLASS:
  These generators produce ACCOUNT RELATIONSHIP EVIDENCE, NOT hard entity-ownership
  equivalence. They establish forensic linkages between counterparties and intermediaries.
  They do NOT authorize Union-Find merging of user identities into hard clusters.

  - Deposit-address reuse:
      U1 -> D <- U2
      Represents interaction with a common deposit intermediary / VASP infrastructure.
      The deposit address D is an intermediary, NOT an equivalent owner of U1 or U2.
      U1 and U2 are related by shared deposit infrastructure, NOT confirmed same-entity.

  - Shared funding source:
      F -> A, F -> B
      Represents directed distribution from a common funding source.
      Recipients A and B are NOT inferred to be the same entity as each other or F.

ENGINEERING CANDIDATE-SELECTION THRESHOLD (MAX_COUNTERPARTIES = 50):
  The counterparty boundary (between 2 and 50 counterparties) is an engineering
  candidate-selection / performance safeguard to prevent indexing massive high-degree
  contracts (e.g. Uniswap routers, exchange omnibus hot wallets) in high-throughput
  ingestion. It is NOT a forensic confidence threshold or proof of entity ownership.
  This threshold is subject to tuning before production-scale deployment.

Evidence eligibility rules:
  - Only Ethereum and Tron chains are processed.
  - Directional transfer counterparties are stored canonically in clustering_evidence.
"""
from __future__ import annotations

import logging
from typing import List

import asyncpg

from app.graph.models import normalize_address, make_address_composite_id
from app.clustering.models import ClusteringEvidence, EvidenceGenerationSummary

logger = logging.getLogger(__name__)

# Heuristic constants
DEPOSIT_REUSE_TYPE: str = "account_deposit_reuse"
SHARED_FUNDING_TYPE: str = "account_shared_funding"
EVIDENCE_STATUS: str = "observed_heuristic"

# Engineering candidate-selection filter (not forensic ownership proof)
MAX_COUNTERPARTIES: int = 50
DEFAULT_BATCH_SIZE: int = 1000


def make_account_evidence_id(
    evidence_type: str,
    network: str,
    transaction_id: str,
    composite_a: str,
    composite_b: str,
) -> str:
    """Deterministic evidence identity for account-model pairs."""
    return f"{evidence_type}:{network}:{transaction_id}:{composite_a}:{composite_b}"


class AccountDepositReuseGenerator:
    """Generates account_deposit_reuse evidence from canonical PostgreSQL data."""

    def __init__(self, pg_pool: asyncpg.Pool, batch_size: int = DEFAULT_BATCH_SIZE):
        self.pg_pool = pg_pool
        self.batch_size = batch_size

    async def generate_all_evidence(self) -> tuple[List[ClusteringEvidence], EvidenceGenerationSummary]:
        summary = EvidenceGenerationSummary()
        all_evidence: List[ClusteringEvidence] = []

        # Find target addresses receiving from between 2 and MAX_COUNTERPARTIES distinct senders
        query = f"""
        SELECT 
            t.chain, 
            t.network, 
            tx.transaction_id,
            t.from_address, 
            t.to_address
        FROM transfers t
        JOIN transactions tx ON t.transaction_pk = tx.id
        JOIN (
            SELECT chain, network, to_address
            FROM transfers
            WHERE chain IN ('ethereum', 'tron') AND to_address IS NOT NULL AND from_address IS NOT NULL
            GROUP BY chain, network, to_address
            HAVING COUNT(DISTINCT from_address) BETWEEN 2 AND $1
        ) candidates ON t.chain = candidates.chain 
                    AND t.network = candidates.network 
                    AND t.to_address = candidates.to_address
        WHERE t.from_address IS NOT NULL
        """

        async with self.pg_pool.acquire() as conn:
            # We fetch all rows since we are pre-filtering strictly with the candidates join
            rows = await conn.fetch(query, MAX_COUNTERPARTIES)

            for row in rows:
                chain = row["chain"]
                network = row["network"]
                txid = str(row["transaction_id"]).strip()
                from_addr = str(row["from_address"]).strip()
                to_addr = str(row["to_address"]).strip()

                comp_a = make_address_composite_id(chain, network, from_addr)
                comp_b = make_address_composite_id(chain, network, to_addr)
                norm_a = normalize_address(chain, from_addr)
                norm_b = normalize_address(chain, to_addr)

                if comp_a > comp_b:
                    comp_a, comp_b = comp_b, comp_a
                    norm_a, norm_b = norm_b, norm_a
                
                # Exclude self-transfers
                if comp_a == comp_b:
                    continue

                ev_id = make_account_evidence_id(
                    DEPOSIT_REUSE_TYPE, network, txid, comp_a, comp_b
                )

                all_evidence.append(
                    ClusteringEvidence(
                        evidence_id=ev_id,
                        evidence_type=DEPOSIT_REUSE_TYPE,
                        evidence_status=EVIDENCE_STATUS,
                        chain=chain,
                        network=network,
                        transaction_id=txid,
                        address_a_composite_id=comp_a,
                        address_b_composite_id=comp_b,
                        address_a_normalized=norm_a,
                        address_b_normalized=norm_b,
                    )
                )

        summary.account_deposit_evidence_generated = len(all_evidence)
        logger.info("Generated %d account_deposit_reuse evidence records.", len(all_evidence))
        return all_evidence, summary


class AccountSharedFundingGenerator:
    """Generates account_shared_funding evidence from canonical PostgreSQL data."""

    def __init__(self, pg_pool: asyncpg.Pool, batch_size: int = DEFAULT_BATCH_SIZE):
        self.pg_pool = pg_pool
        self.batch_size = batch_size

    async def generate_all_evidence(self) -> tuple[List[ClusteringEvidence], EvidenceGenerationSummary]:
        summary = EvidenceGenerationSummary()
        all_evidence: List[ClusteringEvidence] = []

        # Find source addresses sending to between 2 and MAX_COUNTERPARTIES distinct receivers
        query = f"""
        SELECT 
            t.chain, 
            t.network, 
            tx.transaction_id,
            t.from_address, 
            t.to_address
        FROM transfers t
        JOIN transactions tx ON t.transaction_pk = tx.id
        JOIN (
            SELECT chain, network, from_address
            FROM transfers
            WHERE chain IN ('ethereum', 'tron') AND to_address IS NOT NULL AND from_address IS NOT NULL
            GROUP BY chain, network, from_address
            HAVING COUNT(DISTINCT to_address) BETWEEN 2 AND $1
        ) candidates ON t.chain = candidates.chain 
                    AND t.network = candidates.network 
                    AND t.from_address = candidates.from_address
        WHERE t.to_address IS NOT NULL
        """

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, MAX_COUNTERPARTIES)

            for row in rows:
                chain = row["chain"]
                network = row["network"]
                txid = str(row["transaction_id"]).strip()
                from_addr = str(row["from_address"]).strip()
                to_addr = str(row["to_address"]).strip()

                comp_a = make_address_composite_id(chain, network, from_addr)
                comp_b = make_address_composite_id(chain, network, to_addr)
                norm_a = normalize_address(chain, from_addr)
                norm_b = normalize_address(chain, to_addr)

                if comp_a > comp_b:
                    comp_a, comp_b = comp_b, comp_a
                    norm_a, norm_b = norm_b, norm_a
                
                # Exclude self-transfers
                if comp_a == comp_b:
                    continue

                ev_id = make_account_evidence_id(
                    SHARED_FUNDING_TYPE, network, txid, comp_a, comp_b
                )

                all_evidence.append(
                    ClusteringEvidence(
                        evidence_id=ev_id,
                        evidence_type=SHARED_FUNDING_TYPE,
                        evidence_status=EVIDENCE_STATUS,
                        chain=chain,
                        network=network,
                        transaction_id=txid,
                        address_a_composite_id=comp_a,
                        address_b_composite_id=comp_b,
                        address_a_normalized=norm_a,
                        address_b_normalized=norm_b,
                    )
                )

        summary.account_funding_evidence_generated = len(all_evidence)
        logger.info("Generated %d account_shared_funding evidence records.", len(all_evidence))
        return all_evidence, summary
