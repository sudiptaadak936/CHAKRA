"""CHAKRA Step 5: Risk Engine.

Deterministic risk scoring engine for CHAKRA.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.schemas.risk import (
    RiskLevel,
    RiskComponentContribution,
    RiskScoreRecord,
)
from app.forensics.risk_policy import (
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    LOW_THRESHOLD,
    MAX_SCORE,
    MIN_SCORE,
    SANCTIONS_HIT,
    TYPOLOGY_WEIGHTS,
    INELIGIBLE_TYPOLOGIES,
)
from app.schemas.alert import SanctionedAddressHit
from app.schemas.chain import Chain
from app.forensics.typology_detector import TypologyDetection
from app.attribution.models import VASPAddressRecord
from app.attribution.authority import AuthorityPolicy, AuthorityClass
from app.forensics.models import ChangeAddressInference


SYNTHETIC_MARKERS = {
    "demo", "synthetic", "test", "mock", "dummy", "sample", "fake"
}

class RiskEngine:
    """Deterministic, evidence-backed risk engine."""

    @staticmethod
    def _determine_risk_level(score: float) -> RiskLevel:
        if score >= CRITICAL_THRESHOLD:
            return RiskLevel.CRITICAL
        elif score >= HIGH_THRESHOLD:
            return RiskLevel.HIGH
        elif score >= MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM
        elif score >= LOW_THRESHOLD:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEUTRAL

    @staticmethod
    def _generate_deterministic_hash(
        target_address: str,
        chain: str,
        network: str,
        components: List[RiskComponentContribution]
    ) -> str:
        """Generate a deterministic hash from canonicalized inputs."""
        canonical_components = []
        for comp in sorted(components, key=lambda c: c.component_name):
            canonical_components.append({
                "component_name": comp.component_name,
                "score_contribution": round(comp.score_contribution, 2),
                "evidence_ids": sorted(comp.evidence_ids),
                "detection_ids": sorted(comp.detection_ids),
                "reason": comp.reason,
            })
        
        canonical_state = {
            "target_address": target_address,
            "chain": chain,
            "network": network,
            "components": canonical_components,
        }
        
        json_str = json.dumps(canonical_state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_synthetic(source: str, evidence_id: str) -> bool:
        src = (source or "").strip().lower()
        ev = (evidence_id or "").strip().lower()
        for marker in SYNTHETIC_MARKERS:
            if marker in src or marker in ev:
                return True
        return False

    @staticmethod
    def evaluate(
        target_address: str,
        chain: str,
        network: str,
        sanctions_hits: Optional[List[SanctionedAddressHit]] = None,
        typologies: Optional[List[TypologyDetection]] = None,
        attributions: Optional[List[VASPAddressRecord]] = None,
        change_inferences: Optional[List[ChangeAddressInference]] = None,
        cluster_contagions: Optional[List[Dict[str, Any]]] = None,
        computed_at: Optional[datetime] = None
    ) -> RiskScoreRecord:
        """
        Evaluate risk for a target address deterministically.
        """
        # 1. Input Validation - Fail closed
        if not target_address or not target_address.strip():
            raise ValueError("Target address is required.")
        if not chain or not chain.strip():
            raise ValueError("Chain is required.")
            
        try:
            Chain(chain)
        except ValueError:
            raise ValueError(f"Invalid chain: {chain}")
            
        if not network or not network.strip():
            raise ValueError("Network is required.")
            
        sanctions_hits = sanctions_hits or []
        typologies = typologies or []
        attributions = attributions or []
        change_inferences = change_inferences or []
        cluster_contagions = cluster_contagions or []
        
        computed_at = computed_at or datetime.now(timezone.utc)
        
        components: List[RiskComponentContribution] = []
        
        # 2. Sanctions component
        sanctions_evidence_ids = set()
        sanctions_score = 0.0
        for hit in sanctions_hits:
            if RiskEngine._is_synthetic(hit.source, hit.evidence_id):
                continue
            sanctions_evidence_ids.add(hit.evidence_id)
            sanctions_score = SANCTIONS_HIT
            
        if sanctions_score > 0:
            components.append(
                RiskComponentContribution(
                    component_name="SANCTIONS",
                    score_contribution=sanctions_score,
                    evidence_ids=sorted(list(sanctions_evidence_ids)),
                    detection_ids=[],
                    reason="Valid direct sanctioned-address hit from authoritative provenance."
                )
            )
            
        # 3. Typologies component
        typologies_score = 0.0
        typology_detection_ids = set()
        for typo in typologies:
            if typo.confidence_level != "observed":
                continue
                
            typo_type = typo.typology_type.value
            if typo_type in INELIGIBLE_TYPOLOGIES:
                continue
                
            if typo.detection_id in typology_detection_ids:
                continue
                
            weight = TYPOLOGY_WEIGHTS.get(typo_type, 0.0)
            if weight > 0.0:
                typologies_score += weight
                typology_detection_ids.add(typo.detection_id)
                
        if typologies_score > 0:
            components.append(
                RiskComponentContribution(
                    component_name="TYPOLOGIES",
                    score_contribution=typologies_score,
                    evidence_ids=[],
                    detection_ids=sorted(list(typology_detection_ids)),
                    reason="Observed structural typologies."
                )
            )
            
        # 4. Attribution (UNKNOWN does not increase risk, synthetic excluded)
        # 5. Change Inferences (LOW/UNKNOWN excluded)
        # 6. Cluster Contagion
        
        raw_score = sanctions_score + typologies_score
        overall_score = max(MIN_SCORE, min(MAX_SCORE, round(raw_score, 2)))
        
        risk_level = RiskEngine._determine_risk_level(overall_score)
        deterministic_hash = RiskEngine._generate_deterministic_hash(
            target_address, chain, network, components
        )
        
        return RiskScoreRecord(
            target_address=target_address,
            chain=chain,
            network=network,
            overall_score=overall_score,
            risk_level=risk_level,
            components=sorted(components, key=lambda x: x.component_name),
            computed_at=computed_at,
            requires_human_review=True,
            deterministic_hash=deterministic_hash
        )

