"""CHAKRA Step 2C: Bitcoin co-input evidence generator.

Reads canonical Bitcoin UTXO data from PostgreSQL and generates deterministic
pairwise co-input clustering evidence.

CRITICAL SEMANTIC BOUNDARY:
  Co-input evidence means two addresses appear as spending inputs in the same
  Bitcoin transaction. This is a blockchain observation heuristic.
  It does NOT prove common ownership of those addresses.

Evidence eligibility rules:
  - Only Bitcoin (chain = 'bitcoin') transactions are processed.
  - Only transactions with >= 2 distinct eligible input addresses produce evidence.
  - Coinbase inputs (coinbase = TRUE) are EXCLUDED.
  - Addressless inputs (address IS NULL) are EXCLUDED.
  - Duplicate input addresses within a single transaction are deduplicated.
  - Outputs (vouts) are never used as co-input evidence.
  - Self-links (address A with itself) are never generated.
  - Canonical pair ordering: (min_composite_id, max_composite_id) prevents
    duplicate A-B / B-A records.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import asyncpg

from app.graph.models import normalize_address, make_address_composite_id
from app.clustering.models import ClusteringEvidence, EvidenceGenerationSummary

logger = logging.getLogger(__name__)

# Evidence metadata constants
EVIDENCE_TYPE: str = "bitcoin_co_input"
EVIDENCE_STATUS: str = "observed_heuristic"
CHAIN: str = "bitcoin"

# Default batch size for iterating over Bitcoin transactions
DEFAULT_BATCH_SIZE: int = 200


def make_evidence_id(
    network: str,
    transaction_id: str,
    composite_a: str,
    composite_b: str,
) -> str:
    """Deterministic evidence identity for a bitcoin co-input pair.

    Format: bitcoin_co_input:<network>:<txid>:<addr_a_composite>:<addr_b_composite>

    Assumes composite_a <= composite_b (caller is responsible for ordering).
    """
    return f"bitcoin_co_input:{network}:{transaction_id}:{composite_a}:{composite_b}"


def _is_eligible_address(address: Optional[str], coinbase: bool) -> bool:
    """Return True only if the vin address qualifies for co-input clustering.

    Ineligible:
    - NULL / empty address
    - coinbase = TRUE inputs
    """
    if coinbase:
        return False
    if not address or not address.strip():
        return False
    return True


def _generate_pairs_for_transaction(
    network: str,
    transaction_id: str,
    eligible_addresses: List[str],
) -> List[ClusteringEvidence]:
    """Generate deterministic pairwise co-input evidence for one transaction.

    For N distinct eligible input addresses, produces N*(N-1)/2 evidence records.
    All pairs are canonical: (min_composite_id, max_composite_id).
    No self-links. No duplicates.
    """
    # Compute composite IDs and deduplicate
    composite_to_normalized: dict[str, str] = {}
    for addr in eligible_addresses:
        norm = normalize_address(CHAIN, addr)
        comp = make_address_composite_id(CHAIN, network, addr)
        composite_to_normalized[comp] = norm

    # Deterministic sort of distinct composite IDs
    sorted_composites = sorted(composite_to_normalized.keys())
    n = len(sorted_composites)

    if n < 2:
        return []

    evidence_list: List[ClusteringEvidence] = []
    for i in range(n):
        for j in range(i + 1, n):
            comp_a = sorted_composites[i]
            comp_b = sorted_composites[j]
            # comp_a < comp_b is guaranteed by sorted order
            ev_id = make_evidence_id(network, transaction_id, comp_a, comp_b)
            evidence_list.append(
                ClusteringEvidence(
                    evidence_id=ev_id,
                    evidence_type=EVIDENCE_TYPE,
                    evidence_status=EVIDENCE_STATUS,
                    chain=CHAIN,
                    network=network,
                    transaction_id=transaction_id,
                    address_a_composite_id=comp_a,
                    address_b_composite_id=comp_b,
                    address_a_normalized=composite_to_normalized[comp_a],
                    address_b_normalized=composite_to_normalized[comp_b],
                )
            )

    return evidence_list


class CoInputEvidenceGenerator:
    """Generates bitcoin_co_input evidence from canonical PostgreSQL data.

    Reads bitcoin_vins + bitcoin_transaction_details + transactions.
    Never writes to canonical tables.
    Processes transactions in bounded batches to avoid unbounded memory usage.
    """

    def __init__(self, pg_pool: asyncpg.Pool, batch_size: int = DEFAULT_BATCH_SIZE):
        self.pg_pool = pg_pool
        self.batch_size = batch_size

    async def generate_all_evidence(self) -> tuple[List[ClusteringEvidence], EvidenceGenerationSummary]:
        """Scan all Bitcoin transactions and generate co-input evidence.

        Returns:
            (list of ClusteringEvidence records, EvidenceGenerationSummary)
        """
        summary = EvidenceGenerationSummary()
        all_evidence: List[ClusteringEvidence] = []

        last_detail_pk: int = 0

        async with self.pg_pool.acquire() as conn:
            while True:
                # Fetch a bounded batch of Bitcoin transaction details by keyset pagination
                detail_rows = await conn.fetch(
                    """
                    SELECT
                        btd.id         AS detail_pk,
                        btd.txid       AS txid,
                        t.network      AS network,
                        t.transaction_id AS transaction_id
                    FROM bitcoin_transaction_details btd
                    JOIN transactions t ON t.id = btd.transaction_pk
                    WHERE t.chain = 'bitcoin'
                      AND btd.id > $1
                    ORDER BY btd.id ASC
                    LIMIT $2
                    """,
                    last_detail_pk,
                    self.batch_size,
                )

                if not detail_rows:
                    break

                detail_pks = [r["detail_pk"] for r in detail_rows]

                # Fetch all vins for this batch
                vin_rows = await conn.fetch(
                    """
                    SELECT
                        bv.detail_pk,
                        bv.address,
                        bv.coinbase
                    FROM bitcoin_vins bv
                    WHERE bv.detail_pk = ANY($1)
                    ORDER BY bv.id ASC
                    """,
                    detail_pks,
                )

                # Group vins by detail_pk
                vins_by_detail: dict[int, list] = {}
                for vin in vin_rows:
                    vins_by_detail.setdefault(vin["detail_pk"], []).append(vin)

                # Process each transaction in this batch
                for row in detail_rows:
                    detail_pk = row["detail_pk"]
                    txid = str(row["transaction_id"]).strip()
                    network = str(row["network"]).lower()

                    summary.transactions_examined += 1

                    vins = vins_by_detail.get(detail_pk, [])
                    eligible: List[str] = []
                    raw_coinbase_count = 0
                    raw_addressless_count = 0
                    raw_total = 0

                    for vin in vins:
                        raw_total += 1
                        address = vin["address"]
                        coinbase = bool(vin["coinbase"])

                        if coinbase:
                            raw_coinbase_count += 1
                            summary.coinbase_inputs_skipped += 1
                            continue

                        if not address or not str(address).strip():
                            raw_addressless_count += 1
                            summary.addressless_inputs_skipped += 1
                            continue

                        summary.input_addresses_examined += 1
                        eligible.append(str(address).strip())

                    # Deduplicate (preserving all for count tracking)
                    unique_eligible = list(dict.fromkeys(eligible))  # preserves first-seen order
                    duplicates_removed = len(eligible) - len(unique_eligible)
                    summary.duplicate_addresses_removed += duplicates_removed

                    if len(unique_eligible) < 2:
                        continue

                    summary.eligible_bitcoin_transactions += 1

                    tx_evidence = _generate_pairs_for_transaction(
                        network=network,
                        transaction_id=txid,
                        eligible_addresses=unique_eligible,
                    )

                    summary.evidence_relationships_generated += len(tx_evidence)
                    all_evidence.extend(tx_evidence)

                last_detail_pk = detail_pks[-1]

        logger.info(
            "Co-input evidence generation complete: %d evidence records from %d eligible transactions "
            "(%d transactions examined, %d addressless skipped, %d coinbase skipped, %d duplicates removed)",
            summary.evidence_relationships_generated,
            summary.eligible_bitcoin_transactions,
            summary.transactions_examined,
            summary.addressless_inputs_skipped,
            summary.coinbase_inputs_skipped,
            summary.duplicate_addresses_removed,
        )

        return all_evidence, summary
