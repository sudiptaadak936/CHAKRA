"""
KYC preparation — Step 4 prototype. LOCAL ONLY — no external calls.

This module takes a ReIdentificationRequest and the FIU registration status
and produces a KYCRequestRecord with a clear KYC pathway:

  REGISTERED_VASP_PATH   — standard KYC workflow for registered VASPs.
  UNREGISTERED_VASP_PATH — escalation workflow (fires only when explicitly NOT_REGISTERED).
  UNKNOWN_REGISTRATION   — blocked; missing evidence, not treated as either.

IMPORTANT INVARIANT:
  UNKNOWN registration status must NEVER trigger UNREGISTERED_VASP_PATH.
  Missing upstream evidence remains UNKNOWN / UNAVAILABLE.
"""
import logging
from typing import Optional

from app.attribution.engine_models import AttributionConfidence
from app.attribution.fiu_registry import FIURegistrationStatus, FIURegistryRepository
from app.attribution.kyc_models import (
    KYCRequestPacket,
    KYCRequestPath,
    KYCRequestRecord,
    KYCRequestStatus,
)
from app.attribution.reid_models import ReIdentificationRequest, ReIDBranchDecision

logger = logging.getLogger(__name__)


class KYCPreparationService:
    """
    Determines the KYC pathway and assembles a KYC preparation packet.

    No external network calls are made. This is a local prototype only.
    Human investigator review is always required for the resulting packet.
    """

    @staticmethod
    def prepare(
        request: ReIdentificationRequest,
        fiu_registry: Optional[FIURegistryRepository] = None,
    ) -> KYCRequestRecord:
        """
        Prepare a KYC request record from a RE-ID request.

        Args:
            request:      The ReIdentificationRequest from FIUReIDBrancher.
            fiu_registry: Optional repository for registration lookup.
                          If None, registration_status is taken from request.registration_status.

        Returns:
            KYCRequestRecord with status PREPARED, NOT_REQUIRED, or BLOCKED.
        """
        if request is None:
            raise ValueError("ReIdentificationRequest is required for KYC preparation.")

        # Only attempt KYC when REID_REQUIRED — all other branches are not applicable.
        if request.branch_decision != ReIDBranchDecision.REID_REQUIRED:
            reason_map = {
                ReIDBranchDecision.NO_REID_REQUIRED: "KYC not required: attribution is already confirmed.",
                ReIDBranchDecision.REID_NOT_JUSTIFIED: "KYC blocked: insufficient attribution to justify RE-ID.",
            }
            reason = reason_map.get(request.branch_decision, "KYC not applicable.")
            return KYCRequestRecord(
                chain=request.chain,
                address=request.address,
                attributed_vasp=request.attributed_vasp,
                kyc_status=KYCRequestStatus.NOT_REQUIRED,
                kyc_path=KYCRequestPath.UNKNOWN_REGISTRATION,
                packet=None,
                reason=reason,
            )

        # Determine registration status — prefer live lookup, fall back to request field.
        registration_status_value: str = request.registration_status
        if fiu_registry is not None and request.attributed_vasp:
            live_status = fiu_registry.is_registered(request.attributed_vasp)
            registration_status_value = live_status.value
        re_id: Optional[str] = None
        if fiu_registry is not None and request.attributed_vasp:
            re_id = fiu_registry.get_re_id(request.attributed_vasp)

        # ---------------------------------------------------------------
        # INVARIANT: UNKNOWN != NOT_REGISTERED.
        # Only fire UNREGISTERED_VASP_PATH when status is EXPLICITLY NOT_REGISTERED.
        # ---------------------------------------------------------------
        if registration_status_value == FIURegistrationStatus.REGISTERED.value:
            kyc_path = KYCRequestPath.REGISTERED_VASP_PATH
            reason = (
                f"VASP '{request.attributed_vasp}' is REGISTERED in the demonstration FIU registry "
                f"(RE-ID: {re_id}). Standard KYC workflow applies. DEMONSTRATION DATA ONLY."
            )
        elif registration_status_value == FIURegistrationStatus.NOT_REGISTERED.value:
            kyc_path = KYCRequestPath.UNREGISTERED_VASP_PATH
            reason = (
                f"VASP '{request.attributed_vasp}' is explicitly NOT_REGISTERED in the demonstration "
                f"FIU registry. Escalation pathway applies. Human investigator review required. "
                f"DEMONSTRATION DATA ONLY."
            )
        else:
            # UNKNOWN — never silently becomes unregistered
            kyc_path = KYCRequestPath.UNKNOWN_REGISTRATION
            reason = (
                f"Registration status for VASP '{request.attributed_vasp}' is UNKNOWN. "
                f"Absent from demonstration FIU registry. Missing evidence remains UNKNOWN. "
                f"KYC pathway cannot be determined without further investigation."
            )
            logger.info(
                "KYC preparation: VASP %r has UNKNOWN registration status — "
                "not treated as NOT_REGISTERED. address=%r chain=%s",
                request.attributed_vasp,
                request.address,
                request.chain.value if request.chain else "?",
            )
            return KYCRequestRecord(
                chain=request.chain,
                address=request.address,
                attributed_vasp=request.attributed_vasp,
                kyc_status=KYCRequestStatus.BLOCKED,
                kyc_path=kyc_path,
                packet=None,
                reason=reason,
            )

        # Assemble packet for REGISTERED or NOT_REGISTERED paths
        packet = KYCRequestPacket(
            chain=request.chain,
            address=request.address,
            attributed_vasp=request.attributed_vasp,
            attribution_confidence=request.attribution_confidence,
            branch_decision=request.branch_decision,
            kyc_path=kyc_path,
            registration_status=registration_status_value,
            re_id=re_id,
            evidence_ids=list(request.evidence_ids),
            reason=reason,
        )

        logger.info(
            "KYC packet prepared: vasp=%r path=%s status=%s address=%r",
            request.attributed_vasp,
            kyc_path.value,
            registration_status_value,
            request.address,
        )

        return KYCRequestRecord(
            chain=request.chain,
            address=request.address,
            attributed_vasp=request.attributed_vasp,
            kyc_status=KYCRequestStatus.PREPARED,
            kyc_path=kyc_path,
            packet=packet,
            reason=reason,
        )
