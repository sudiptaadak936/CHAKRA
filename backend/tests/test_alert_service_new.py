"""
Tests for FIX 6/7/8/10: New alert types and alert service methods.

Invariants tested:
- UNREGISTERED_VASP_ATTRIBUTION only fires on NOT_REGISTERED
- UNKNOWN registration status does NOT trigger UNREGISTERED_VASP_ATTRIBUTION
- RISK_SCORE_THRESHOLD_EXCEEDED forward contract works
- SUSPECT_REGISTRY_LINK only fires on MATCH
- approve_alert / dismiss_alert preserve requires_human_review=True
- All 4 AlertType values exist in the enum
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.alert import (
    AlertRecord,
    AlertSeverity,
    AlertStatus,
    AlertType,
    SanctionedAddressHit,
    UnregisteredVASPAttribution,
    RiskScoreThresholdExceeded,
    SuspectRegistryLink,
)
from app.schemas.chain import Chain
from app.services.alert_service import AlertService


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_mock_redis():
    redis = AsyncMock()
    redis.setnx = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    redis.xadd = AsyncMock(return_value=b"1-0")
    return redis


def _make_alert_record(alert_type=AlertType.SANCTIONED_ADDRESS_HIT, status=AlertStatus.PENDING_REVIEW):
    from datetime import datetime, timezone
    return AlertRecord(
        alert_id="test-alert-id-001",
        alert_type=alert_type,
        chain=Chain.BITCOIN,
        address="1TestAddress001",
        severity=AlertSeverity.HIGH,
        reason="test reason",
        evidence_ids=["ev-001"],
        source="test",
        created_at=datetime.now(timezone.utc),
        requires_human_review=True,
        status=status,
    )


# -----------------------------------------------------------------------
# AlertType enum coverage
# -----------------------------------------------------------------------

class TestAlertTypeEnum:
    def test_all_four_types_present(self):
        values = {t.value for t in AlertType}
        assert "SANCTIONED_ADDRESS_HIT" in values
        assert "UNREGISTERED_VASP_ATTRIBUTION" in values
        assert "RISK_SCORE_THRESHOLD_EXCEEDED" in values
        assert "SUSPECT_REGISTRY_LINK" in values

    def test_exactly_four_types(self):
        assert len(list(AlertType)) == 4


# -----------------------------------------------------------------------
# UNREGISTERED_VASP_ATTRIBUTION — FIX 7
# -----------------------------------------------------------------------

class TestUnregisteredVASPAlert:
    @pytest.mark.asyncio
    async def test_fires_on_not_registered(self):
        service = AlertService()
        hit = UnregisteredVASPAttribution(
            chain=Chain.BITCOIN,
            address="1TestAddress001",
            attributed_vasp="DarkExchange",
            attribution_confidence="HIGH_CONFIDENCE",
            registration_status="NOT_REGISTERED",
            evidence_ids=["ev-001"],
            source="test",
            reason="Unregistered VASP detected",
        )
        mock_redis = _make_mock_redis()
        with patch("app.services.alert_service.db_manager") as mock_db:
            mock_db.redis_client = mock_redis
            result = await service.generate_unregistered_vasp_alert(hit)
        assert result is not None
        assert result.alert_type == AlertType.UNREGISTERED_VASP_ATTRIBUTION
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_suppressed_on_unknown(self):
        """INVARIANT: UNKNOWN must NEVER trigger UNREGISTERED_VASP_ATTRIBUTION."""
        service = AlertService()
        hit = UnregisteredVASPAttribution(
            chain=Chain.BITCOIN,
            address="1TestAddress002",
            attributed_vasp="UnknownVASP",
            attribution_confidence="HIGH_CONFIDENCE",
            registration_status="UNKNOWN",
            evidence_ids=["ev-002"],
            source="test",
            reason="Should not fire",
        )
        mock_redis = _make_mock_redis()
        with patch("app.services.alert_service.db_manager") as mock_db:
            mock_db.redis_client = mock_redis
            result = await service.generate_unregistered_vasp_alert(hit)
        assert result is None

    @pytest.mark.asyncio
    async def test_suppressed_on_registered(self):
        service = AlertService()
        hit = UnregisteredVASPAttribution(
            chain=Chain.BITCOIN,
            address="1TestAddress003",
            attributed_vasp="RegisteredVASP",
            attribution_confidence="CONFIRMED",
            registration_status="REGISTERED",
            evidence_ids=["ev-003"],
            source="test",
            reason="Should not fire for registered",
        )
        result = await service.generate_unregistered_vasp_alert(hit)
        assert result is None

    @pytest.mark.asyncio
    async def test_suppressed_on_none_input(self):
        service = AlertService()
        result = await service.generate_unregistered_vasp_alert(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_suppressed_on_missing_address(self):
        service = AlertService()
        hit = UnregisteredVASPAttribution(
            chain=Chain.BITCOIN,
            address="",
            attributed_vasp="DarkExchange",
            attribution_confidence="HIGH_CONFIDENCE",
            registration_status="NOT_REGISTERED",
            evidence_ids=["ev-001"],
            source="test",
            reason="test",
        )
        result = await service.generate_unregistered_vasp_alert(hit)
        assert result is None


# -----------------------------------------------------------------------
# RISK_SCORE_THRESHOLD_EXCEEDED — FIX 8 forward contract
# -----------------------------------------------------------------------

class TestRiskThresholdAlert:
    @pytest.mark.asyncio
    async def test_fires_on_valid_input(self):
        service = AlertService()
        hit = RiskScoreThresholdExceeded(
            chain=Chain.BITCOIN,
            address="1TestAddress010",
            overall_score=85.0,
            risk_level="CRITICAL",
            threshold_crossed=80.0,
            risk_score_record_id="hash-001",
            evidence_ids=["ev-010"],
            source="risk_engine",
            reason="Risk score exceeded critical threshold",
        )
        mock_redis = _make_mock_redis()
        with patch("app.services.alert_service.db_manager") as mock_db:
            mock_db.redis_client = mock_redis
            result = await service.generate_risk_threshold_alert(hit)
        assert result is not None
        assert result.alert_type == AlertType.RISK_SCORE_THRESHOLD_EXCEEDED
        assert result.severity == AlertSeverity.CRITICAL
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_suppressed_on_none(self):
        service = AlertService()
        result = await service.generate_risk_threshold_alert(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_suppressed_on_invalid_score(self):
        service = AlertService()
        hit = RiskScoreThresholdExceeded(
            chain=Chain.BITCOIN,
            address="1TestAddress011",
            overall_score=150.0,  # invalid
            risk_level="HIGH",
            threshold_crossed=60.0,
            risk_score_record_id="hash-002",
            evidence_ids=["ev-011"],
            source="risk_engine",
            reason="test",
        )
        result = await service.generate_risk_threshold_alert(hit)
        assert result is None

    @pytest.mark.asyncio
    async def test_high_risk_level_severity(self):
        service = AlertService()
        hit = RiskScoreThresholdExceeded(
            chain=Chain.BITCOIN,
            address="1TestAddress012",
            overall_score=65.0,
            risk_level="HIGH",
            threshold_crossed=60.0,
            risk_score_record_id="hash-003",
            evidence_ids=["ev-012"],
            source="risk_engine",
            reason="Risk score HIGH",
        )
        mock_redis = _make_mock_redis()
        with patch("app.services.alert_service.db_manager") as mock_db:
            mock_db.redis_client = mock_redis
            result = await service.generate_risk_threshold_alert(hit)
        assert result is not None
        assert result.severity == AlertSeverity.HIGH


# -----------------------------------------------------------------------
# SUSPECT_REGISTRY_LINK — Step 6 forward contract
# -----------------------------------------------------------------------

class TestSuspectRegistryAlert:
    @pytest.mark.asyncio
    async def test_fires_on_match(self):
        service = AlertService()
        hit = SuspectRegistryLink(
            chain=Chain.BITCOIN,
            address="1TestAddress020",
            registry_match_id="registry-match-001",
            match_result="MATCH",
            evidence_ids=["ev-020"],
            source="suspect_registry",
            reason="Address matched suspect registry",
        )
        mock_redis = _make_mock_redis()
        with patch("app.services.alert_service.db_manager") as mock_db:
            mock_db.redis_client = mock_redis
            result = await service.generate_suspect_registry_alert(hit)
        assert result is not None
        assert result.alert_type == AlertType.SUSPECT_REGISTRY_LINK
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_suppressed_on_unknown(self):
        service = AlertService()
        hit = SuspectRegistryLink(
            chain=Chain.BITCOIN,
            address="1TestAddress021",
            registry_match_id="registry-match-002",
            match_result="UNKNOWN",
            evidence_ids=["ev-021"],
            source="suspect_registry",
            reason="Should not fire",
        )
        result = await service.generate_suspect_registry_alert(hit)
        assert result is None

    @pytest.mark.asyncio
    async def test_suppressed_on_no_match(self):
        service = AlertService()
        hit = SuspectRegistryLink(
            chain=Chain.BITCOIN,
            address="1TestAddress022",
            registry_match_id="registry-match-003",
            match_result="NO_MATCH",
            evidence_ids=["ev-022"],
            source="suspect_registry",
            reason="Should not fire",
        )
        result = await service.generate_suspect_registry_alert(hit)
        assert result is None


# -----------------------------------------------------------------------
# approve_alert / dismiss_alert — FIX 10
# -----------------------------------------------------------------------

class TestAlertApprovalDismissal:
    @pytest.mark.asyncio
    async def test_approve_pending_alert(self):
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.PENDING_REVIEW)
        result = await service.approve_alert(alert)
        assert result.status == AlertStatus.REVIEWED
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_dismiss_pending_alert(self):
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.PENDING_REVIEW)
        result = await service.dismiss_alert(alert)
        assert result.status == AlertStatus.DISMISSED
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_cannot_approve_dismissed_alert(self):
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.DISMISSED)
        with pytest.raises(ValueError, match="DISMISSED"):
            await service.approve_alert(alert)

    @pytest.mark.asyncio
    async def test_cannot_dismiss_reviewed_alert(self):
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.REVIEWED)
        with pytest.raises(ValueError, match="REVIEWED"):
            await service.dismiss_alert(alert)

    @pytest.mark.asyncio
    async def test_approve_none_raises(self):
        service = AlertService()
        with pytest.raises(ValueError):
            await service.approve_alert(None)

    @pytest.mark.asyncio
    async def test_dismiss_none_raises(self):
        service = AlertService()
        with pytest.raises(ValueError):
            await service.dismiss_alert(None)

    @pytest.mark.asyncio
    async def test_requires_human_review_immutable_after_approve(self):
        """requires_human_review must always be True after approval."""
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.PENDING_REVIEW)
        result = await service.approve_alert(alert)
        assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_requires_human_review_immutable_after_dismiss(self):
        """requires_human_review must always be True after dismissal."""
        service = AlertService()
        alert = _make_alert_record(status=AlertStatus.PENDING_REVIEW)
        result = await service.dismiss_alert(alert)
        assert result.requires_human_review is True
