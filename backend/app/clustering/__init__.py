"""CHAKRA Step 2C: Deterministic Address/Entity Clustering.

Derives candidate address clusters from canonical Bitcoin co-input evidence.

CRITICAL SEMANTIC BOUNDARY:
  Clusters are analytical constructs derived from observable blockchain evidence.
  Step 2C establishes:
    1. OBSERVATION: addresses co-appear as inputs in the same transaction.
    2. CLUSTERING EVIDENCE: deterministic bitcoin_co_input evidence records.
    3. DETERMINISTIC CLUSTER MEMBERSHIP: connected components via Union-Find.

  Step 2C does NOT establish:
    4. REAL-WORLD ENTITY ATTRIBUTION: clusters do not represent confirmed persons,
       organizations, exchanges, VASPs, or scammers.

PostgreSQL remains authoritative. Clustering state is derived analytical output.
Neo4j graph projection (Steps 2A/2B) is not modified.
"""
from app.clustering.models import (
    ClusteringEvidence,
    ClusterMember,
    AddressCluster,
    EvidenceGenerationSummary,
    ClusteringResult,
    ClusteringError,
)
from app.clustering.evidence import (
    CoInputEvidenceGenerator,
    make_evidence_id,
    EVIDENCE_TYPE,
    EVIDENCE_STATUS,
    DEFAULT_BATCH_SIZE,
)
from app.clustering.clusterer import DeterministicClusterer, make_cluster_id
from app.clustering.repository import ClusteringRepository
from app.clustering.engine import ClusteringEngine

__all__ = [
    # Models
    "ClusteringEvidence",
    "ClusterMember",
    "AddressCluster",
    "EvidenceGenerationSummary",
    "ClusteringResult",
    "ClusteringError",
    # Evidence
    "CoInputEvidenceGenerator",
    "make_evidence_id",
    "EVIDENCE_TYPE",
    "EVIDENCE_STATUS",
    "DEFAULT_BATCH_SIZE",
    # Clustering
    "DeterministicClusterer",
    "make_cluster_id",
    # Repository
    "ClusteringRepository",
    # Engine
    "ClusteringEngine",
]
