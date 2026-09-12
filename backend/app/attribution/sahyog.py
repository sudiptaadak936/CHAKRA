import logging
from app.schemas.chain import Chain
from app.attribution.engine_models import AttributionConfidence
from app.attribution.reid_models import ReIdentificationRequest, ReIDBranchDecision
from app.attribution.sahyog_models import EscalationRecord, EscalationStatus, SahyogEscalationPacket

logger = logging.getLogger(__name__)


class SAHYOGEscalationStub:
    """
    Local SAHYOG-style escalation preparation stub.
    No SAHYOG/FIU-IND network request is performed.
    """

    @staticmethod
    def _validate_request(request: ReIdentificationRequest) -> None:
        if request is None:
            raise ValueError("Request cannot be null.")

        if not isinstance(request.branch_decision, ReIDBranchDecision):
            raise ValueError("Invalid branch decision.")

        if not isinstance(request.attribution_confidence, AttributionConfidence):
            raise ValueError("Invalid attribution confidence.")

        if not isinstance(request.chain, Chain):
            raise ValueError("Invalid chain.")

        if not request.address or not str(request.address).strip():
            raise ValueError("Invalid address.")

        if not request.provenance:
            raise ValueError("Missing provenance.")

        if request.evidence_ids is None or not isinstance(request.evidence_ids, list):
            raise ValueError("Invalid evidence IDs.")

        for eid in request.evidence_ids:
            if not isinstance(eid, str) or not eid.strip():
                raise ValueError("Invalid evidence ID element.")

        # Contradictions
        if request.branch_decision == ReIDBranchDecision.REID_REQUIRED:
            if request.attribution_confidence == AttributionConfidence.UNKNOWN:
                raise ValueError("Contradiction: REID_REQUIRED but confidence is UNKNOWN.")
            if not request.evidence_ids:
                raise ValueError("Contradiction: REID_REQUIRED but no evidence IDs provided.")

    @staticmethod
    def notify_sahyog(
        request: ReIdentificationRequest,
        recommended_action: str = None,
    ) -> EscalationRecord:
        """
        Prepares a SAHYOG-style escalation packet locally.
        It does NOT contact SAHYOG or any government system.
        No network request is made. No credentials are used.
        """
        try:
            SAHYOGEscalationStub._validate_request(request)
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}") from e

        escalation_status = EscalationStatus.NOT_ESCALATED
        packet = None
        reason = request.reason

        if request.branch_decision == ReIDBranchDecision.REID_REQUIRED:
            escalation_status = EscalationStatus.PREPARED
            reason = f"SAHYOG escalation packet prepared. Original reason: {request.reason}"
            packet = SahyogEscalationPacket(
                chain=request.chain,
                address=request.address,
                attributed_vasp=request.attributed_vasp,
                attribution_confidence=request.attribution_confidence,
                branch_decision=request.branch_decision,
                evidence_ids=list(request.evidence_ids),
                provenance=request.provenance,
                reason=reason,
                recommended_action=recommended_action,
            )
            # Structured log: no secrets, no credentials, clearly marks no network call
            logger.info(
                "SAHYOG escalation packet PREPARED (NO SAHYOG REQUEST MADE). "
                "chain=%s address=%r attributed_vasp=%r confidence=%s "
                "branch_decision=%s evidence_count=%d recommended_action=%r",
                request.chain.value if request.chain else "?",
                request.address,
                request.attributed_vasp,
                request.attribution_confidence.value if request.attribution_confidence else "?",
                request.branch_decision.value,
                len(request.evidence_ids),
                recommended_action,
            )
        elif request.branch_decision == ReIDBranchDecision.NO_REID_REQUIRED:
            escalation_status = EscalationStatus.NOT_ESCALATED
            reason = "Escalation not justified: RE-ID not required."
            logger.info(
                "SAHYOG escalation NOT triggered (NO_REID_REQUIRED). "
                "chain=%s address=%r — NO SAHYOG REQUEST MADE.",
                request.chain.value if request.chain else "?",
                request.address,
            )
        elif request.branch_decision == ReIDBranchDecision.REID_NOT_JUSTIFIED:
            escalation_status = EscalationStatus.NOT_ESCALATED
            reason = "Escalation not justified: Insufficient defensible attribution."
            logger.info(
                "SAHYOG escalation NOT triggered (REID_NOT_JUSTIFIED). "
                "chain=%s address=%r — NO SAHYOG REQUEST MADE.",
                request.chain.value if request.chain else "?",
                request.address,
            )
        else:
            raise ValueError("Unsupported branch decision.")

        return EscalationRecord(
            chain=request.chain,
            address=request.address,
            attributed_vasp=request.attributed_vasp,
            attribution_confidence=request.attribution_confidence,
            branch_decision=request.branch_decision,
            evidence_ids=list(request.evidence_ids),
            provenance=request.provenance,
            reason=reason,
            escalation_status=escalation_status,
            packet=packet,
        )

