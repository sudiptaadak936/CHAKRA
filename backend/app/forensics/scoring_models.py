"""CHAKRA Step 3F: Path Scoring Models.

These models define the deterministic scoring components and the 
explained evidence for candidate paths.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from app.graph.traversal import TraversalPath


class EvidenceReason(BaseModel):
    """Structured explanation for a scoring component."""
    value: float
    reason: str


class PathScoreExplanation(BaseModel):
    """Deterministic score components and provenance for a path."""

    total_score: float = 0.0
    hop_penalty: float = 0.0
    time_penalty: float = 0.0
    amount_loss_penalty: float = 0.0
    label_strength: float = 0.0
    hard_cluster_bonus: float = 0.0
    relationship_bonus: float = 0.0
    risk_evidence: float = 0.0

    # Structured explanations
    hop_penalty_evidence: Optional[EvidenceReason] = None
    time_penalty_evidence: Optional[EvidenceReason] = None
    amount_loss_evidence: Optional[EvidenceReason] = None
    label_strength_evidence: Optional[EvidenceReason] = None
    hard_cluster_evidence: Optional[EvidenceReason] = None
    relationship_evidence: Optional[EvidenceReason] = None
    risk_evidence_reason: Optional[EvidenceReason] = None


class ScoredPath(BaseModel):
    """A TraversalPath annotated with its deterministic score and explanation."""
    
    path: TraversalPath
    explanation: PathScoreExplanation

    @property
    def score(self) -> float:
        return self.explanation.total_score
