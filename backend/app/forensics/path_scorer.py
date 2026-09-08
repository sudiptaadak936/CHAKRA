"""CHAKRA Step 3F: Deterministic Path Scorer."""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import asyncpg

from app.graph.traversal import TraversalPath, TraversalNode, TraversalEdge, _BITCOIN_CHAIN
from app.forensics.scoring_models import PathScoreExplanation, ScoredPath, EvidenceReason

logger = logging.getLogger(__name__)

# Configurable Engineering Defaults
HOP_PENALTY = 10.0
TIME_PENALTY_PER_DAY = 2.0
MAX_AMOUNT_LOSS_PENALTY = 20.0

class PathScorer:
    """Deterministic path scorer."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def score_paths(self, paths: List[TraversalPath]) -> List[ScoredPath]:
        if not paths:
            return []

        # Gather context required for scoring all paths
        tx_pks = set()
        cids = set()
        
        for p in paths:
            for node in p.nodes:
                cids.add(node.composite_id)
            for edge in p.edges:
                if edge.pg_transaction_pk is not None:
                    tx_pks.add(edge.pg_transaction_pk)

        # Batch fetch timestamps
        tx_timestamps: Dict[int, datetime] = {}
        if tx_pks:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT id, timestamp FROM transactions WHERE id = ANY($1)", list(tx_pks))
                for r in rows:
                    if r["timestamp"]:
                        tx_timestamps[r["id"]] = r["timestamp"]

        # Batch fetch cluster memberships
        cluster_memberships: Dict[str, str] = {}
        if cids:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT composite_id, cluster_id FROM cluster_members WHERE composite_id = ANY($1)", list(cids))
                for r in rows:
                    cluster_memberships[r["composite_id"]] = r["cluster_id"]

        # Batch fetch relationship evidence
        relationship_evidence_map: Dict[Tuple[str, str], List[str]] = {}
        if cids:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT address_a_composite_id, address_b_composite_id, evidence_type
                    FROM clustering_evidence
                    WHERE address_a_composite_id = ANY($1) AND address_b_composite_id = ANY($1)
                    """, list(cids)
                )
                for r in rows:
                    k1 = (r["address_a_composite_id"], r["address_b_composite_id"])
                    k2 = (r["address_b_composite_id"], r["address_a_composite_id"])
                    relationship_evidence_map.setdefault(k1, []).append(r["evidence_type"])
                    relationship_evidence_map.setdefault(k2, []).append(r["evidence_type"])

        scored_paths = []
        for p in paths:
            exp = self._score_single_path(p, tx_timestamps, cluster_memberships, relationship_evidence_map)
            scored_paths.append(ScoredPath(path=p, explanation=exp))
            
        return scored_paths

    def _score_single_path(
        self, 
        path: TraversalPath, 
        tx_timestamps: Dict[int, datetime],
        cluster_memberships: Dict[str, str],
        relationship_evidence_map: Dict[Tuple[str, str], List[str]]
    ) -> PathScoreExplanation:
        exp = PathScoreExplanation()
        
        # 1. Hop Penalty
        exp.hop_penalty = float(path.hops * HOP_PENALTY)
        if exp.hop_penalty > 0:
            exp.hop_penalty_evidence = EvidenceReason(
                value=-exp.hop_penalty,
                reason=f"{path.hops} hops at {HOP_PENALTY} per hop"
            )

        # Sequence of timestamps and amounts
        times: List[datetime] = []
        amounts: List[Tuple[str, float]] = [] # (asset_id, amount)
        
        if path.edges:
            for edge in path.edges:
                # Time
                if edge.pg_transaction_pk in tx_timestamps:
                    times.append(tx_timestamps[edge.pg_transaction_pk])
                # Amount
                asset_id = f"{edge.chain}:{edge.asset_symbol or edge.asset_type}"
                if edge.amount is not None:
                    amounts.append((asset_id, float(edge.amount)))
                elif edge.amount_str:
                    try:
                        amounts.append((asset_id, float(edge.amount_str)))
                    except ValueError:
                        amounts.append((asset_id, -1.0))
                else:
                    amounts.append((asset_id, -1.0))
        elif path.utxo_steps:
            for step in path.utxo_steps:
                tx = step.transaction
                if tx.timestamp:
                    try:
                        times.append(datetime.fromisoformat(tx.timestamp))
                    except ValueError:
                        pass
                
                asset_id = f"{step.spent_input.chain}:BTC"
                if step.created_output.value_sat is not None:
                    amounts.append((asset_id, float(step.created_output.value_sat)))
                elif step.created_output.value_sat_str:
                    try:
                        amounts.append((asset_id, float(step.created_output.value_sat_str)))
                    except ValueError:
                        amounts.append((asset_id, -1.0))
                else:
                    amounts.append((asset_id, -1.0))
        
        # 2. Time Penalty
        total_time_penalty = 0.0
        time_gaps = []
        for i in range(1, len(times)):
            gap = (times[i] - times[i-1]).total_seconds()
            if gap > 0:
                gap_days = gap / 86400.0
                penalty = gap_days * TIME_PENALTY_PER_DAY
                total_time_penalty += penalty
                time_gaps.append(f"{gap_days:.2f} days")
                
        if total_time_penalty > 0:
            exp.time_penalty = total_time_penalty
            exp.time_penalty_evidence = EvidenceReason(
                value=-total_time_penalty,
                reason=f"Time gaps: {', '.join(time_gaps)}"
            )

        # 3. Amount Loss Penalty
        total_loss_penalty = 0.0
        loss_reasons = []
        for i in range(1, len(amounts)):
            prev_asset, prev_amt = amounts[i-1]
            curr_asset, curr_amt = amounts[i]
            
            if prev_asset != curr_asset:
                loss_reasons.append(f"Amount retention not scored: incompatible or unavailable amount data.")
                continue
            if prev_amt <= 0 or curr_amt < 0:
                loss_reasons.append(f"Amount retention not scored: incompatible or unavailable amount data.")
                continue
                
            retention_ratio = curr_amt / prev_amt
            if retention_ratio > 1.0:
                retention_ratio = 1.0
            if retention_ratio < 0.0:
                retention_ratio = 0.0
                
            penalty = MAX_AMOUNT_LOSS_PENALTY * (1.0 - retention_ratio)
            total_loss_penalty += penalty
            if penalty > 0:
                loss_reasons.append(f"Hop {i}: Retention {retention_ratio:.2f}, penalty {penalty:.2f}")

        if total_loss_penalty > 0:
            exp.amount_loss_penalty = total_loss_penalty
            exp.amount_loss_evidence = EvidenceReason(
                value=-total_loss_penalty,
                reason=" | ".join(loss_reasons)
            )
        elif loss_reasons:
            exp.amount_loss_evidence = EvidenceReason(
                value=0.0,
                reason=" | ".join(loss_reasons)
            )

        # 4. Label Strength / 5. Risk Evidence (Neutral for now)
        exp.label_strength = 0.0
        exp.risk_evidence = 0.0

        # 6. Cluster Strength and Relationship Bonus
        hard_bonus_count = 0
        rel_evidence_found = set()
        
        for i in range(1, len(path.nodes)):
            cid_prev = path.nodes[i-1].composite_id
            cid_curr = path.nodes[i].composite_id
            
            # Hard cluster
            c1 = cluster_memberships.get(cid_prev)
            c2 = cluster_memberships.get(cid_curr)
            if c1 and c2 and c1 == c2:
                hard_bonus_count += 1
                
            # Relationship evidence
            rels = relationship_evidence_map.get((cid_prev, cid_curr), [])
            for r in rels:
                rel_evidence_found.add(r)
                
        # HARD CLUSTER BONUS
        HARD_CLUSTER_VALUE = 5.0
        if hard_bonus_count > 0:
            exp.hard_cluster_bonus = float(hard_bonus_count * HARD_CLUSTER_VALUE)
            exp.hard_cluster_evidence = EvidenceReason(
                value=exp.hard_cluster_bonus,
                reason=f"{hard_bonus_count} consecutive addresses share hard cluster"
            )

        # RELATIONSHIP BONUS
        if rel_evidence_found:
            exp.relationship_bonus = 0.0
            exp.relationship_evidence = EvidenceReason(
                value=0.0,
                reason=f"Relationships observed: {', '.join(sorted(rel_evidence_found))}"
            )

        exp.total_score = (
            - exp.hop_penalty
            - exp.time_penalty
            - exp.amount_loss_penalty
            + exp.label_strength
            + exp.hard_cluster_bonus
            + exp.relationship_bonus
            + exp.risk_evidence
        )
        return exp