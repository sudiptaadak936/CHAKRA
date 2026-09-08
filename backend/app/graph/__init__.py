"""CHAKRA Graph Projection Package.

Step 2A: PostgreSQL -> Neo4j Graph Projection.
Step 2B: Deterministic Money-Flow Traversal.
"""
from app.graph.models import (
    normalize_address,
    make_address_composite_id,
    make_transaction_composite_id,
    make_transfer_relationship_id,
    make_bitcoin_input_id,
    make_bitcoin_output_id,
    to_neo4j_numeric,
    BatchProjectionResult,
    ProjectionSummary,
    BatchProjectionError,
)
from app.graph.projector import GraphProjector, DEFAULT_BATCH_SIZE
from app.graph.traversal import (
    MoneyFlowTraversal,
    TraversalEdge,
    TraversalNode,
    SpentInputEdge,
    TransactionNode,
    CreatedOutputEdge,
    UtxoSpendStep,
    TraversalPath,
    TraversalResult,
    TraversalParameterError,
    MAX_HOPS_CEILING,
    MAX_NODES_CEILING,
    MAX_EDGES_CEILING,
)

__all__ = [
    # Step 2A — Graph Projection
    "normalize_address",
    "make_address_composite_id",
    "make_transaction_composite_id",
    "make_transfer_relationship_id",
    "make_bitcoin_input_id",
    "make_bitcoin_output_id",
    "to_neo4j_numeric",
    "BatchProjectionResult",
    "ProjectionSummary",
    "BatchProjectionError",
    "GraphProjector",
    "DEFAULT_BATCH_SIZE",
    # Step 2B — Money-Flow Traversal
    "MoneyFlowTraversal",
    "TraversalEdge",
    "TraversalNode",
    "SpentInputEdge",
    "TransactionNode",
    "CreatedOutputEdge",
    "UtxoSpendStep",
    "TraversalPath",
    "TraversalResult",
    "TraversalParameterError",
    "MAX_HOPS_CEILING",
    "MAX_NODES_CEILING",
    "MAX_EDGES_CEILING",
]


