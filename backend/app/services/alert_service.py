"""Service for deterministic alert generation and Redis stream publishing."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.database import db_manager
from app.schemas.alert import AlertRecord, AlertType, AlertSeverity, AlertStatus, SanctionedAddressHit

logger = logging.getLogger(__name__)


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

        # Publish to Redis
        if db_manager.redis_client is None:
            logger.error("Redis client is not initialized, cannot publish alert.")
            return None

        try:
            # Deterministic deduplication via Redis SETNX
            dedup_redis_key = f"chakra:alert:dedup:{alert_id}"
            is_new = await db_manager.redis_client.setnx(dedup_redis_key, "1")
            
            if not is_new:
                logger.info("Duplicate alert suppressed for alert_id=%s", alert_id)
                # Ensure the same logical result is returned, just don't publish duplicate event
                return alert
                
            # Expiry for deduplication key to prevent it growing unbounded forever, 
            # or keep it if strictly needed. Let's set 30 days as reasonable.
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
            logger.info("Published alert %s to Redis alerts stream", alert_id)
            
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

