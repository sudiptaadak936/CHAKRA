"""Service for deterministic alert generation and Redis stream publishing."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.database import db_manager
from app.schemas.alert import (
    AlertRecord,
    AlertType,
    AlertSeverity,
    AlertStatus,
    SanctionedAddressHit,
    UnregisteredVASPAttribution,
    RiskScoreThresholdExceeded,
    SuspectRegistryLink,
)

logger = logging.getLogger(__name__)

# Registration status constant — must match FIURegistrationStatus.NOT_REGISTERED.value
_NOT_REGISTERED_VALUE = "NOT_REGISTERED"
_MATCH_VALUE = "MATCH"


class AlertService:
    """Deterministic alert generation service."""

    async def generate_sanction_alert(self, hit: SanctionedAddressHit) -> Optional[AlertRecord]:
        """
        Generate an alert for a sanctioned address hit and publish to Redis Stream.
        Fails closed if the input is invalid.
        """
        if not self._is_valid_hit(hit):
            logger.warning("Invalid sanctioned address hit input, fail closed.")
            return None

        # Deterministic identity key for deduplication
        dedup_key = f"{AlertType.SANCTIONED_ADDRESS_HIT.value}:{hit.chain.value}:{hit.address}:{hit.evidence_id}"
        alert_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()

        # Deterministic alert record
        alert = AlertRecord(
            alert_id=alert_id,
            alert_type=AlertType.SANCTIONED_ADDRESS_HIT,
            chain=hit.chain,
            address=hit.address,
            severity=AlertSeverity.CRITICAL,
            reason=hit.reason,
            evidence_ids=[hit.evidence_id],
            source=hit.source,
            created_at=datetime.now(timezone.utc),
            requires_human_review=True,
            status=AlertStatus.PENDING_REVIEW,
        )

        return await self._publish_alert(alert)

    async def generate_unregistered_vasp_alert(
        self, hit: UnregisteredVASPAttribution
    ) -> Optional[AlertRecord]:
        """
        Generate an alert when attribution resolves to an explicitly NOT_REGISTERED VASP.

        INVARIANT: This method ONLY fires when registration_status == NOT_REGISTERED.
        It must NEVER fire when registration_status is UNKNOWN.
        UNKNOWN != NOT_REGISTERED.

        Fails closed if the invariant is violated.
        """
        if hit is None:
            logger.warning("UnregisteredVASPAttribution input is None, fail closed.")
            return None

        # Strict invariant: ONLY NOT_REGISTERED triggers this alert.
        if hit.registration_status != _NOT_REGISTERED_VALUE:
            logger.warning(
                "UNREGISTERED_VASP_ATTRIBUTION alert suppressed: "
                "registration_status=%r is not NOT_REGISTERED (UNKNOWN != NOT_REGISTERED). "
                "address=%r attributed_vasp=%r",
                hit.registration_status,
                hit.address,
                hit.attributed_vasp,
            )
            return None

        if not hit.address or not hit.address.strip():
            logger.warning("UNREGISTERED_VASP_ATTRIBUTION: missing address, fail closed.")
            return None
        if not hit.attributed_vasp or not hit.attributed_vasp.strip():
            logger.warning("UNREGISTERED_VASP_ATTRIBUTION: missing attributed_vasp, fail closed.")
            return None
        if not hit.evidence_ids:
            logger.warning("UNREGISTERED_VASP_ATTRIBUTION: no evidence_ids, fail closed.")
            return None

        dedup_key = (
            f"{AlertType.UNREGISTERED_VASP_ATTRIBUTION.value}:"
            f"{hit.chain.value}:{hit.address}:{hit.attributed_vasp}"
        )
        alert_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()

        alert = AlertRecord(
            alert_id=alert_id,
            alert_type=AlertType.UNREGISTERED_VASP_ATTRIBUTION,
            chain=hit.chain,
            address=hit.address,
            severity=AlertSeverity.HIGH,
            reason=hit.reason,
            evidence_ids=list(hit.evidence_ids),
            source=hit.source,
            created_at=datetime.now(timezone.utc),
            requires_human_review=True,
            status=AlertStatus.PENDING_REVIEW,
        )

        return await self._publish_alert(alert)

    async def generate_risk_threshold_alert(
        self, hit: RiskScoreThresholdExceeded
    ) -> Optional[AlertRecord]:
        """
        Generate an alert when a risk score exceeds a policy threshold.

        NOTE: This is a FORWARD CONTRACT for Step 5 Risk Scoring.
        The risk engine will call this when overall_score >= threshold.
        Human review is always required.
        """
        if hit is None:
            logger.warning("RiskScoreThresholdExceeded input is None, fail closed.")
            return None
        if not hit.address or not hit.address.strip():
            logger.warning("RISK_SCORE_THRESHOLD_EXCEEDED: missing address, fail closed.")
            return None
        if hit.overall_score < 0 or hit.overall_score > 100:
            logger.warning(
                "RISK_SCORE_THRESHOLD_EXCEEDED: invalid overall_score=%f, fail closed.",
                hit.overall_score,
            )
            return None

        dedup_key = (
            f"{AlertType.RISK_SCORE_THRESHOLD_EXCEEDED.value}:"
            f"{hit.chain.value}:{hit.address}:{hit.risk_score_record_id}"
        )
        alert_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()

        # Severity derived from risk level
        severity = AlertSeverity.HIGH
        if hit.risk_level in ("CRITICAL",):
            severity = AlertSeverity.CRITICAL
        elif hit.risk_level in ("MEDIUM",):
            severity = AlertSeverity.MEDIUM

        alert = AlertRecord(
            alert_id=alert_id,
            alert_type=AlertType.RISK_SCORE_THRESHOLD_EXCEEDED,
            chain=hit.chain,
            address=hit.address,
            severity=severity,
            reason=hit.reason,
            evidence_ids=list(hit.evidence_ids),
            source=hit.source,
            created_at=datetime.now(timezone.utc),
            requires_human_review=True,
            status=AlertStatus.PENDING_REVIEW,
        )

        return await self._publish_alert(alert)

    async def generate_suspect_registry_alert(
        self, hit: SuspectRegistryLink
    ) -> Optional[AlertRecord]:
        """
        Generate an alert when an address matches the suspect registry.

        FORWARD CONTRACT for Step 6 Suspect Registry.
        INVARIANT: Only fires on explicit MATCH result. Never on UNKNOWN or NO_MATCH.
        """
        if hit is None:
            logger.warning("SuspectRegistryLink input is None, fail closed.")
            return None
        if hit.match_result != _MATCH_VALUE:
            logger.warning(
                "SUSPECT_REGISTRY_LINK alert suppressed: match_result=%r is not MATCH. "
                "address=%r",
                hit.match_result,
                hit.address,
            )
            return None
        if not hit.address or not hit.address.strip():
            logger.warning("SUSPECT_REGISTRY_LINK: missing address, fail closed.")
            return None

        dedup_key = (
            f"{AlertType.SUSPECT_REGISTRY_LINK.value}:"
            f"{hit.chain.value}:{hit.address}:{hit.registry_match_id}"
        )
        alert_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()

        alert = AlertRecord(
            alert_id=alert_id,
            alert_type=AlertType.SUSPECT_REGISTRY_LINK,
            chain=hit.chain,
            address=hit.address,
            severity=AlertSeverity.HIGH,
            reason=hit.reason,
            evidence_ids=list(hit.evidence_ids),
            source=hit.source,
            created_at=datetime.now(timezone.utc),
            requires_human_review=True,
            status=AlertStatus.PENDING_REVIEW,
        )

        return await self._publish_alert(alert)

    async def approve_alert(self, alert: AlertRecord) -> AlertRecord:
        """
        Mark an alert as REVIEWED by a human investigator.

        Returns a new AlertRecord with status=REVIEWED.
        AlertRecord is frozen; a new instance is created.
        Human review is always required — this records that it happened.
        """
        if alert is None:
            raise ValueError("AlertRecord is required for approval.")
        if alert.status == AlertStatus.DISMISSED:
            raise ValueError(f"Alert {alert.alert_id} is already DISMISSED and cannot be approved.")
        # Return new frozen instance with updated status
        return AlertRecord(
            **{
                **alert.model_dump(),
                "status": AlertStatus.REVIEWED,
                "requires_human_review": True,  # invariant preserved
            }
        )

    async def dismiss_alert(self, alert: AlertRecord) -> AlertRecord:
        """
        Mark an alert as DISMISSED by a human investigator.

        Returns a new AlertRecord with status=DISMISSED.
        AlertRecord is frozen; a new instance is created.
        Dismissal is a human action — never automatic.
        """
        if alert is None:
            raise ValueError("AlertRecord is required for dismissal.")
        if alert.status == AlertStatus.REVIEWED:
            raise ValueError(f"Alert {alert.alert_id} is already REVIEWED and cannot be dismissed.")
        return AlertRecord(
            **{
                **alert.model_dump(),
                "status": AlertStatus.DISMISSED,
                "requires_human_review": True,  # invariant preserved
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _publish_alert(self, alert: AlertRecord) -> Optional[AlertRecord]:
        """Publish alert to Redis stream with deduplication. Returns alert on success."""
        if db_manager.redis_client is None:
            logger.error("Redis client is not initialized, cannot publish alert.")
            return None

        try:
            # Deterministic deduplication via Redis SETNX
            dedup_redis_key = f"chakra:alert:dedup:{alert.alert_id}"
            is_new = await db_manager.redis_client.setnx(dedup_redis_key, "1")

            if not is_new:
                logger.info("Duplicate alert suppressed for alert_id=%s", alert.alert_id)
                # Return same logical result; don't publish duplicate event
                return alert

            # Expiry for deduplication key: 30 days
            await db_manager.redis_client.expire(dedup_redis_key, 30 * 86400)

            # Redis Streams field values must be strings
            redis_payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "chain": alert.chain.value,
                "address": alert.address,
                "severity": alert.severity.value,
                "reason": alert.reason,
                "evidence_ids": json.dumps(alert.evidence_ids),
                "source": alert.source,
                "created_at": alert.created_at.isoformat(),
                "requires_human_review": "True",
                "status": alert.status.value,
            }

            await db_manager.redis_client.xadd(
                name="alerts",
                fields=redis_payload,
            )
            logger.info("Published alert %s to Redis alerts stream", alert.alert_id)

            return alert

        except Exception as e:
            logger.error("Failed to publish alert to Redis: %s", e)
            return None

    def _is_valid_hit(self, hit: SanctionedAddressHit) -> bool:
        """Strict validation to fail closed."""
        if not hit:
            return False
        if not getattr(hit, "chain", None) or not hit.chain.value:
            return False
        if not getattr(hit, "address", None) or not hit.address.strip():
            return False
        if not getattr(hit, "source", None) or not hit.source.strip():
            return False
        if not getattr(hit, "evidence_id", None) or not hit.evidence_id.strip():
            return False
        return True
