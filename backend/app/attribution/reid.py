from typing import List, Optional
from app.attribution.engine_models import AttributionDecision, AttributionConfidence
from app.attribution.reid_models import (
    ReIDStatus,
    ReIDBranchDecision,
    ReIdentificationRequest,
)
from app.attribution.fiu_registry import FIURegistryRepository, FIURegistrationStatus


class FIUReIDBrancher:
    """
    Deterministic FIU-IND RE-ID branching policy.

    This does NOT submit real requests. It only models the decision branch.

    Two SEPARATE and INDEPENDENT decisions are made:
    1. Branch decision — driven by AttributionConfidence tier.
    2. Registration status — driven by FIU registry lookup (SEPARATE from confidence).
       UNKNOWN registration must never silently become NOT_REGISTERED.
    """

    @staticmethod
    def evaluate(
        decision: AttributionDecision,
        fiu_registry: Optional[FIURegistryRepository] = None,
    ) -> ReIdentificationRequest:
        """
        Evaluate a RE-ID branch decision.

        Args:
            decision:      The attribution decision from the attribution engine.
            fiu_registry:  Optional FIURegistryRepository for registration-status lookup.
                           If None, registration_status will be UNKNOWN.

        Returns:
            ReIdentificationRequest with branch_decision and registration_status
            as INDEPENDENT fields.
        """
        if decision is None:
            raise ValueError("AttributionDecision is required for RE-ID branching.")

        # ---------------------------------------------------------------
        # DECISION 1: Branch decision — based purely on confidence tier.
        # This is NOT influenced by registration status.
        # ---------------------------------------------------------------
        branch_decision = ReIDBranchDecision.REID_NOT_JUSTIFIED
        reason = "Unknown attribution"

        if decision.confidence == AttributionConfidence.CONFIRMED:
            branch_decision = ReIDBranchDecision.NO_REID_REQUIRED
            reason = "VASP identity already established through authoritative provenance."
        elif decision.confidence == AttributionConfidence.HIGH_CONFIDENCE:
            branch_decision = ReIDBranchDecision.REID_REQUIRED
            reason = "Attribution is high confidence, but requires FIU-IND workflow for formal identity confirmation."
        elif decision.confidence == AttributionConfidence.PROBABLE_INFERRED:
            branch_decision = ReIDBranchDecision.REID_REQUIRED
            reason = "Probable inferred attribution hypothesis exists, requires FIU-IND workflow for identity."
        elif decision.confidence == AttributionConfidence.UNKNOWN:
            branch_decision = ReIDBranchDecision.REID_NOT_JUSTIFIED
            reason = "No defensible VASP attribution exists. Do not manufacture a target."

        if not decision.address or not decision.chain:
            branch_decision = ReIDBranchDecision.REID_NOT_JUSTIFIED
            reason = "Missing address or chain."

        if (
            decision.confidence in [AttributionConfidence.HIGH_CONFIDENCE, AttributionConfidence.PROBABLE_INFERRED]
            and not decision.vasp_name
        ):
            branch_decision = ReIDBranchDecision.REID_NOT_JUSTIFIED
            reason = "Missing VASP candidate for RE-ID."

        # ---------------------------------------------------------------
        # DECISION 2: Registration status — SEPARATE from confidence.
        # Absence from the demo registry => UNKNOWN, not NOT_REGISTERED.
        # UNKNOWN must NEVER silently become NOT_REGISTERED.
        # ---------------------------------------------------------------
        registration_status: FIURegistrationStatus = FIURegistrationStatus.UNKNOWN
        if fiu_registry is not None and decision.vasp_name:
            registration_status = fiu_registry.is_registered(decision.vasp_name)
        # If fiu_registry is None, or vasp_name is absent, status remains UNKNOWN.

        # ---------------------------------------------------------------
        # Collect evidence IDs safely
        # ---------------------------------------------------------------
        evidence_ids: List[str] = []
        if decision.provenance:
            for obs in decision.provenance.direct_observations:
                if getattr(obs, "evidence_id", None):
                    evidence_ids.append(str(obs.evidence_id))

            for obs in decision.provenance.inferred_observations:
                if getattr(obs, "evidence_id", None):
                    evidence_ids.append(str(obs.evidence_id))

            if decision.provenance.cluster_id:
                evidence_ids.append(str(decision.provenance.cluster_id))

        # Deduplicate while preserving order
        evidence_ids = list(dict.fromkeys(evidence_ids))

        return ReIdentificationRequest(
            chain=decision.chain,
            address=decision.address,
            attributed_vasp=decision.vasp_name,
            attribution_confidence=decision.confidence,
            evidence_ids=evidence_ids,
            provenance=decision.provenance,
            reason=reason,
            status=ReIDStatus.REQUEST_PREPARED,
            branch_decision=branch_decision,
            registration_status=registration_status,
        )
