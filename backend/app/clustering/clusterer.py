"""CHAKRA Step 2C: Deterministic address cluster computation.

Implements iterative Union-Find (Disjoint Set Union) with path compression
and union by rank to compute connected components from clustering evidence.

Algorithm properties:
  - Fully deterministic: given the same evidence set, always produces identical
    cluster membership, representatives, and cluster IDs.
  - Iterative (no recursion): safe for large component sizes.
  - Input-order invariant: records are sorted before processing.
  - Hash-seed invariant: uses sorted canonical string construction, not Python dicts.

Cluster ID construction:
  1. Collect all composite IDs in the component.
  2. Sort lexicographically.
  3. Compute SHA-256 of the pipe-joined sorted string: "id1|id2|id3|..."
  4. Cluster ID = first 64 hex characters of the SHA-256 digest.

Representative:
  Lexicographically smallest composite ID in the component.

SEMANTIC NOTE:
  An AddressCluster represents addresses connected by evidence types explicitly
  authorized for hard entity clustering (HARD_CLUSTER_EVIDENCE_TYPES).
  It does NOT represent every address that has interacted with another address.
  It does NOT establish real-world entity ownership.

  Relationship evidence (such as account-model deposit-address reuse and shared-funding
  source) establishes forensic linkage, but does NOT constitute hard entity equivalence.
  Such relationship evidence is preserved in clustering_evidence for downstream
  traversal and path scoring (Step 3F), but does NOT participate in DSU union.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple

from app.clustering.models import AddressCluster, ClusterMember, ClusteringEvidence

logger = logging.getLogger(__name__)

# Evidence types explicitly authorized to form hard entity clusters via Union-Find.
# Currently, only Bitcoin co-input spending (H1) is authorized.
# Account-model heuristics (deposit-reuse, shared-funding) represent relationship
# linkages and are intentionally excluded from hard entity clustering.
HARD_CLUSTER_EVIDENCE_TYPES: frozenset[str] = frozenset({
    "bitcoin_co_input",
})


# ---------------------------------------------------------------------------
# Iterative Union-Find
# ---------------------------------------------------------------------------

class _DSU:
    """Iterative Disjoint Set Union with path compression and union by rank.

    Operates on arbitrary hashable node identities (composite_id strings).
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def add(self, node: str) -> None:
        """Register a node if not already present."""
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, node: str) -> str:
        """Find root of the set containing `node`, with iterative path compression."""
        # Iterative path finding (two-pass)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression: point all nodes on the path directly to root
        current = node
        while self._parent[current] != root:
            next_node = self._parent[current]
            self._parent[current] = root
            current = next_node

        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing `a` and `b` using union by rank."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return  # Already in the same set

        # Union by rank — smaller rank tree attaches under larger rank tree
        if self._rank[root_a] < self._rank[root_b]:
            self._parent[root_a] = root_b
        elif self._rank[root_a] > self._rank[root_b]:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

    def components(self) -> Dict[str, List[str]]:
        """Return a mapping {root_composite_id -> [member composite IDs]}."""
        groups: Dict[str, List[str]] = {}
        # Sort nodes first for determinism
        for node in sorted(self._parent.keys()):
            root = self.find(node)
            groups.setdefault(root, []).append(node)
        return groups


# ---------------------------------------------------------------------------
# Cluster ID / Representative
# ---------------------------------------------------------------------------

def make_cluster_id(sorted_member_ids: List[str]) -> str:
    """Deterministic cluster ID from sorted member composite IDs.

    Algorithm:
      1. Members must already be sorted lexicographically.
      2. Join with '|' separator.
      3. SHA-256 hash of the UTF-8 encoded string.
      4. Return the full 64-character hex digest.
    """
    canonical_string = "|".join(sorted_member_ids)
    digest = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------

class DeterministicClusterer:
    """Computes deterministic connected-component clusters from evidence records.

    Input: a list of ClusteringEvidence records.
    Output: a list of AddressCluster objects.

    Properties:
      - Deterministic: same evidence → same clusters.
      - Input-order invariant: evidence is sorted before processing.
      - No ML, no randomization, no probabilities.
      - Iterative Union-Find: safe for large datasets.
      - Selective: only authorized hard-clustering evidence types participate in
        connected-component entity formation. Relationship evidence records are
        preserved in the database for path scoring but excluded from DSU union.
    """

    def __init__(
        self,
        authorized_evidence_types: frozenset[str] = HARD_CLUSTER_EVIDENCE_TYPES,
    ) -> None:
        self.authorized_evidence_types = authorized_evidence_types

    def compute_clusters(
        self,
        evidence_records: List[ClusteringEvidence],
    ) -> List[AddressCluster]:
        """Compute clusters from a list of evidence records.

        Only records matching authorized_evidence_types participate in DSU union.
        Returns clusters sorted by cluster_id for deterministic output ordering.
        """
        if not evidence_records:
            return []

        # Filter for evidence authorized for hard entity clustering
        eligible_evidence = [
            ev for ev in evidence_records if ev.evidence_type in self.authorized_evidence_types
        ]
        if not eligible_evidence:
            return []

        dsu = _DSU()

        # Sort evidence records for determinism before processing
        sorted_evidence = sorted(eligible_evidence, key=lambda e: e.evidence_id)

        # Build a mapping from composite_id -> (chain, network, normalized_address)
        # for all addresses encountered in evidence.
        addr_meta: Dict[str, Tuple[str, str, str]] = {}

        for ev in sorted_evidence:
            dsu.add(ev.address_a_composite_id)
            dsu.add(ev.address_b_composite_id)
            dsu.union(ev.address_a_composite_id, ev.address_b_composite_id)

            # Store address metadata (chain, network, normalized)
            # composite_id format: "{chain}:{network}:{normalized_address}"
            if ev.address_a_composite_id not in addr_meta:
                addr_meta[ev.address_a_composite_id] = (
                    ev.chain, ev.network, ev.address_a_normalized
                )
            if ev.address_b_composite_id not in addr_meta:
                addr_meta[ev.address_b_composite_id] = (
                    ev.chain, ev.network, ev.address_b_normalized
                )

        # Extract connected components
        components = dsu.components()

        clusters: List[AddressCluster] = []
        for _root, member_ids in components.items():
            # Sort members for determinism
            sorted_member_ids = sorted(member_ids)

            # Representative: lexicographically smallest composite ID
            representative = sorted_member_ids[0]

            # Cluster ID: SHA-256 of sorted members
            cluster_id = make_cluster_id(sorted_member_ids)

            # Determine chain/network from representative (all members in same network
            # because evidence is chain-scoped, but we derive from metadata)
            chain, network, _ = addr_meta[representative]

            # Build ClusterMember list
            members = []
            for comp_id in sorted_member_ids:
                m_chain, m_network, m_norm = addr_meta[comp_id]
                members.append(
                    ClusterMember(
                        cluster_id=cluster_id,
                        composite_id=comp_id,
                        chain=m_chain,
                        network=m_network,
                        normalized_address=m_norm,
                    )
                )

            clusters.append(
                AddressCluster(
                    cluster_id=cluster_id,
                    representative_composite_id=representative,
                    chain=chain,
                    network=network,
                    members=members,
                )
            )

        # Sort clusters for deterministic output order
        clusters.sort(key=lambda c: c.cluster_id)

        logger.info(
            "Cluster computation complete: %d clusters from %d evidence records",
            len(clusters),
            len(evidence_records),
        )

        return clusters
