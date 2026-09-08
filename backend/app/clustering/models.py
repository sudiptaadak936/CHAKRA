"""CHAKRA Step 2C: Clustering data models.

All models in this module represent DERIVED ANALYTICAL STATE only.
They do NOT represent canonical blockchain transactions or transfers.

CRITICAL SEMANTIC BOUNDARY:
  ClusteringEvidence represents an observed blockchain heuristic or relationship signal.
  It does NOT prove common ownership.
  AddressCluster represents a derived analytical construct for addresses connected
  by authorized hard-clustering evidence.
  It does NOT represent a confirmed real-world person, organization, or entity.

  SEMANTIC CLASSES:
    A. Hard Entity-Clustering Evidence:
       Evidence authorized for Union-Find entity clustering (e.g. 'bitcoin_co_input').
    B. Account Relationship Evidence:
       Evidence establishing forensic linkages between counterparties/intermediaries
       (e.g. 'account_deposit_reuse', 'account_shared_funding'). Preserved in the
       evidence store for traversal and path scoring, but NOT merged into hard clusters.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence Models
# ---------------------------------------------------------------------------

class ClusteringEvidence(BaseModel):
    """One deterministic evidence relationship between two address identities.

    evidence_type:
        - 'bitcoin_co_input': Hard clustering evidence. Addresses A and B appear as
          spending inputs in the same Bitcoin transaction.
        - 'account_deposit_reuse': Account relationship evidence. Transfer to a shared
          deposit intermediary.
        - 'account_shared_funding': Account relationship evidence. Transfer from a shared
          funding source.

    evidence_status = 'observed_heuristic':
        Signals that this is an observed heuristic/relationship signal, NOT a proof of
        common ownership.

    address_a_composite_id is always <= address_b_composite_id (lexicographic order).
    This canonical ordering prevents duplicate A-B / B-A evidence records.
    """

    evidence_id: str = Field(..., description="Deterministic unique evidence identifier")
    evidence_type: str = Field(..., description="Evidence category ('bitcoin_co_input', 'account_deposit_reuse', 'account_shared_funding')")
    evidence_status: str = Field(..., description="'observed_heuristic' — not ownership proof")
    chain: str
    network: str
    transaction_id: str
    address_a_composite_id: str = Field(..., description="Lexicographically smaller composite ID")
    address_b_composite_id: str = Field(..., description="Lexicographically larger composite ID")
    address_a_normalized: str
    address_b_normalized: str

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Cluster Models
# ---------------------------------------------------------------------------

class ClusterMember(BaseModel):
    """One address identity within a derived cluster."""

    cluster_id: str
    composite_id: str
    chain: str
    network: str
    normalized_address: str

    model_config = {"frozen": True}


class AddressCluster(BaseModel):
    """A derived analytical address cluster.

    Represents a connected component of addresses linked by co-input evidence.

    cluster_id: Deterministic SHA-256 hex digest of the canonical sorted member set.
    representative_composite_id: Lexicographically smallest member composite ID.
    members: All member addresses.

    DOES NOT imply common real-world ownership.
    """

    cluster_id: str = Field(..., description="Deterministic SHA-256 hex of sorted member composite IDs")
    representative_composite_id: str = Field(..., description="Lexicographically smallest member")
    chain: str
    network: str
    members: List[ClusterMember] = Field(default_factory=list)

    @property
    def member_count(self) -> int:
        return len(self.members)

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Summary / Result Models
# ---------------------------------------------------------------------------

class EvidenceGenerationSummary(BaseModel):
    """Structured observability output for the evidence generation phase."""

    transactions_examined: int = 0
    eligible_bitcoin_transactions: int = 0
    input_addresses_examined: int = 0
    addressless_inputs_skipped: int = 0
    coinbase_inputs_skipped: int = 0
    duplicate_addresses_removed: int = 0
    evidence_relationships_generated: int = 0
    malformed_records_encountered: int = 0
    account_deposit_evidence_generated: int = 0
    account_funding_evidence_generated: int = 0


class ClusteringResult(BaseModel):
    """Complete result of a full Step 2C clustering pipeline run."""

    evidence_summary: EvidenceGenerationSummary = Field(
        default_factory=EvidenceGenerationSummary
    )
    clusters_generated: int = 0
    addresses_assigned_to_clusters: int = 0
    total_evidence_records: int = 0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ClusteringError(Exception):
    """Raised when clustering pipeline encounters a non-recoverable error."""

    def __init__(self, phase: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Clustering error in phase '{phase}': {message}")
        self.phase = phase
        self.original_error = original_error
