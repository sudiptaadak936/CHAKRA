"""CHAKRA Step 3F: Deterministic Weighted Beam Search."""
from __future__ import annotations

import logging
from typing import List, Optional

from app.graph.traversal import MoneyFlowTraversal, TraversalPath, TraversalResult, TraversalNode
from app.forensics.path_scorer import PathScorer
from app.forensics.scoring_models import ScoredPath

logger = logging.getLogger(__name__)


def _get_path_sort_key(scored_path: ScoredPath):
    """Deterministic sorting key for tie-breaking.
    1. total score descending (-score)
    2. hop count ascending (hops)
    3. canonical address sequence lexicographically
    4. canonical transaction sequence lexicographically
    """
    path = scored_path.path
    
    # 1. Total score (descending) -> use negative for ascending sort
    score_key = -scored_path.score
    
    # 2. Hop count (ascending)
    hops_key = path.hops
    
    # 3. Canonical address sequence
    addr_seq = ",".join(n.composite_id for n in path.nodes)
    
    # 4. Canonical transaction sequence
    if path.edges:
        tx_seq = ",".join(e.transaction_id for e in path.edges)
    elif path.utxo_steps:
        tx_seq = ",".join(step.transaction.transaction_id for step in path.utxo_steps)
    else:
        tx_seq = ""
        
    return (score_key, hops_key, addr_seq, tx_seq)


class WeightedBeamSearch:
    """Performs deterministic weighted beam search over money-flow paths."""

    def __init__(self, traversal: MoneyFlowTraversal, scorer: PathScorer):
        self.traversal = traversal
        self.scorer = scorer

    async def search(
        self,
        chain: str,
        network: str,
        source_address: str,
        target_address: Optional[str] = None,
        beam_width: int = 5,
        max_hops: int = 5,
        max_nodes_per_query: int = 1000,
        max_edges_per_query: int = 10000,
    ) -> List[ScoredPath]:
        """Execute the beam search algorithm."""
        
        # Start with a 0-hop initial path. To get the start node, we run traversal for 0 hops?
        # MoneyFlowTraversal doesn't support 0 hops, min is 1.
        # But we can just use 1-hop traversal on the source to get started.
        
        # Initial candidate generation from source
        res = await self.traversal.traverse(
            chain=chain,
            network=network,
            address=source_address,
            max_hops=1,
            max_nodes=max_nodes_per_query,
            max_edges=max_edges_per_query
        )
        
        if res.error or not res.paths:
            return []

        # Filter out cycles (self-transfers on hop 1)
        valid_initial_paths = []
        for p in res.paths:
            if p.nodes[0].composite_id != p.nodes[-1].composite_id:
                valid_initial_paths.append(p)
                
        if not valid_initial_paths:
            return []

        # Score and prune initial candidates
        scored_candidates = await self.scorer.score_paths(valid_initial_paths)
        scored_candidates.sort(key=_get_path_sort_key)
        active_candidates = scored_candidates[:beam_width]

        completed_paths: List[ScoredPath] = []
        
        target_norm = None
        if target_address:
            # We don't have access to normalize_address here directly without importing,
            # but we can do a basic normalization or check raw_address.
            # We'll just compare normalized_address later if needed, assuming target_address is pre-normalized.
            target_norm = target_address.lower() if chain.lower() != "bitcoin" else target_address

        for current_depth in range(1, max_hops):
            if not active_candidates:
                break
                
            next_generation_paths: List[TraversalPath] = []
            
            for candidate in active_candidates:
                path = candidate.path
                last_node = path.nodes[-1]
                
                # If target reached, move to completed and do not expand
                if target_norm and (last_node.normalized_address == target_norm or last_node.raw_address == target_address):
                    completed_paths.append(candidate)
                    continue
                    
                # Expand 1 hop from the tip
                hop_res = await self.traversal.traverse(
                    chain=last_node.chain,
                    network=last_node.network,
                    address=last_node.raw_address,
                    max_hops=1,
                    max_nodes=max_nodes_per_query,
                    max_edges=max_edges_per_query
                )
                
                if not hop_res.paths:
                    if not target_norm or (last_node.normalized_address == target_norm or last_node.raw_address == target_address):
                        completed_paths.append(candidate)
                    continue
                    
                path_node_ids = {n.composite_id for n in path.nodes}
                expanded = False
                
                for hop_path in hop_res.paths:
                    hop_dest = hop_path.nodes[-1]
                    
                    # Cycle check
                    if hop_dest.composite_id in path_node_ids:
                        continue
                        
                    # Stitch path
                    new_nodes = list(path.nodes) + [hop_dest]
                    new_edges = list(path.edges) + hop_path.edges
                    new_utxo = list(path.utxo_steps) + hop_path.utxo_steps
                    
                    new_path = TraversalPath(
                        nodes=new_nodes,
                        edges=new_edges,
                        utxo_steps=new_utxo,
                        hops=path.hops + 1
                    )
                    next_generation_paths.append(new_path)
                    expanded = True
                    
                if not expanded:
                    if not target_norm or (last_node.normalized_address == target_norm or last_node.raw_address == target_address):
                        completed_paths.append(candidate)
            
            if not next_generation_paths:
                active_candidates = []
                break
                
            # Score and prune next generation
            scored_next_gen = await self.scorer.score_paths(next_generation_paths)
            scored_next_gen.sort(key=_get_path_sort_key)
            active_candidates = scored_next_gen[:beam_width]
            
        # Add any remaining active paths to completed
        for candidate in active_candidates:
            if target_norm:
                last_node = candidate.path.nodes[-1]
                if last_node.normalized_address != target_norm and last_node.raw_address != target_address:
                    continue
            completed_paths.append(candidate)
            
        # Final deterministic sort
        completed_paths.sort(key=_get_path_sort_key)
        
        return completed_paths
