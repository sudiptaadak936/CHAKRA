"""Unit tests for Step 4.5A LEA Alerting Layer."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.alert import AlertRecord, AlertSeverity, AlertStatus, AlertType, SanctionedAddressHit
from app.schemas.chain import Chain
from app.services.alert_service import AlertService


@pytest.fixture
def mock_redis():
    mock = AsyncMock()
    # setnx returns True for new keys, False for duplicates
    mock.setnx.return_value = True
    return mock


@pytest.fixture
def alert_service(mock_redis):
    with patch("app.services.alert_service.db_manager") as mock_db:
        mock_db.redis_client = mock_redis
        yield AlertService(), mock_redis


class TestAlertGenerationValid:
    """Valid hit scenarios."""

    @pytest.mark.asyncio
    async def test_valid_ofac_hit_generates_alert(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna",
            source="OFAC",
            evidence_id="evidence-123",
            reason="SDN List match"
        )
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is not None
        assert isinstance(alert, AlertRecord)
        # Correct propagation
        assert alert.chain == Chain.BITCOIN
        assert alert.address == "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna"
        assert alert.source == "OFAC"
        assert alert.evidence_ids == ["evidence-123"]
        assert alert.reason == "SDN List match"
        
        # Guardrails & invariants
        assert alert.requires_human_review is True
        assert alert.alert_type == AlertType.SANCTIONED_ADDRESS_HIT
        assert alert.status == AlertStatus.PENDING_REVIEW
        assert alert.severity == AlertSeverity.CRITICAL

        # Redis verification
        mock_redis.xadd.assert_called_once()
        call_kwargs = mock_redis.xadd.call_args.kwargs
        assert call_kwargs["name"] == "alerts"
        
        fields = call_kwargs["fields"]
        assert fields["alert_id"] == alert.alert_id
        assert fields["alert_type"] == AlertType.SANCTIONED_ADDRESS_HIT.value
        assert fields["chain"] == Chain.BITCOIN.value
        assert fields["address"] == "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna"
        assert fields["severity"] == AlertSeverity.CRITICAL.value
        assert fields["reason"] == "SDN List match"
        assert fields["evidence_ids"] == '["evidence-123"]'
        assert fields["source"] == "OFAC"
        assert fields["requires_human_review"] == "True"
        assert fields["status"] == AlertStatus.PENDING_REVIEW.value

    @pytest.mark.asyncio
    async def test_valid_chainabuse_hit_generates_alert(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.EVM,
            address="0x1234567890123456789012345678901234567890",
            source="Chainabuse",
            evidence_id="ca-ev-999",
            reason="Reported scam"
        )
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is not None
        assert alert.source == "Chainabuse"
        assert alert.chain == Chain.EVM
        mock_redis.xadd.assert_called_once()


class TestFailClosedValidation:
    """Fail-closed contract tests."""

    @pytest.mark.asyncio
    async def test_null_input(self, alert_service):
        service, mock_redis = alert_service
        
        # Type hinting allows passing None here for testing the internal fail-closed check
        alert = await service.generate_sanction_alert(None)
        
        assert alert is None
        mock_redis.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_chain(self, alert_service):
        service, mock_redis = alert_service
        
        with pytest.raises(ValidationError):
            SanctionedAddressHit(
                chain="UNKNOWN_CHAIN",
                address="1A1",
                source="OFAC",
                evidence_id="123",
                reason="reason"
            )

    @pytest.mark.asyncio
    async def test_missing_or_empty_address(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="   ", # Empty after strip
            source="OFAC",
            evidence_id="123",
            reason="reason"
        )
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is None
        mock_redis.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_source(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="123",
            source="", # Empty source
            evidence_id="123",
            reason="reason"
        )
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is None
        mock_redis.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_evidence_id(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="123",
            source="OFAC",
            evidence_id="", # Empty evidence id
            reason="reason"
        )
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is None
        mock_redis.xadd.assert_not_called()
        
    @pytest.mark.asyncio
    async def test_malformed_hit_schema(self):
        # Pydantic handles structural malformation (e.g. missing fields entirely)
        with pytest.raises(ValidationError):
            SanctionedAddressHit(
                chain=Chain.BITCOIN,
                address="1A1",
                # missing source, evidence_id, reason
            )

    def test_unsupported_alert_type(self):
        # AlertType Enum strictly limits possible types.
        with pytest.raises(ValidationError):
            AlertRecord(
                alert_id="1",
                alert_type="UNSUPPORTED_TYPE",
                chain=Chain.BITCOIN,
                address="123",
                severity=AlertSeverity.CRITICAL,
                reason="test",
                evidence_ids=["1"],
                source="OFAC",
                created_at="2026-01-01T00:00:00Z"
            )


class TestSafetyInvariants:
    """Invariant and safety behavior tests."""

    @pytest.mark.asyncio
    async def test_no_external_http_calls(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="123",
            source="OFAC",
            evidence_id="123",
            reason="test"
        )
        
        with patch("httpx.AsyncClient.get") as mock_get:
            await service.generate_sanction_alert(hit)
            mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_deterministic_deduplication(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="123",
            source="OFAC",
            evidence_id="ev-123",
            reason="test"
        )
        
        # Simulate that SETNX returns False (key already exists)
        mock_redis.setnx.return_value = False
        
        alert = await service.generate_sanction_alert(hit)
        
        assert alert is not None # Returns the alert logically
        mock_redis.xadd.assert_not_called() # But does NOT publish to stream again

    @pytest.mark.asyncio
    async def test_repeated_event_deduplication_lifecycle(self, alert_service):
        service, mock_redis = alert_service

        existing_keys = set()

        async def fake_setnx(key: str, value: str) -> bool:
            if key in existing_keys:
                return False
            existing_keys.add(key)
            return True

        mock_redis.setnx.side_effect = fake_setnx

        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna",
            source="OFAC",
            evidence_id="ev-sdn-999",
            reason="Sanctioned match"
        )

        # 1. First processing: alert published
        alert_first = await service.generate_sanction_alert(hit)
        assert alert_first is not None
        assert mock_redis.xadd.call_count == 1
        expected_dedup_key = f"chakra:alert:dedup:{alert_first.alert_id}"
        assert expected_dedup_key in existing_keys

        # 2. Same SanctionedAddressHit processed again: duplicate suppressed
        alert_second = await service.generate_sanction_alert(hit)
        assert alert_second is not None
        assert alert_second.alert_id == alert_first.alert_id
        # xadd NOT called again (call count remains 1)
        assert mock_redis.xadd.call_count == 1

        # 3. Genuinely different evidence event (different evidence_id) remains distinguishable
        hit_different_evidence = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="1A1zP1eP5QGefi2DMPTfTL5SLmv7Divfna",
            source="OFAC",
            evidence_id="ev-sdn-1000",
            reason="Sanctioned match"
        )
        alert_diff_ev = await service.generate_sanction_alert(hit_different_evidence)
        assert alert_diff_ev is not None
        assert alert_diff_ev.alert_id != alert_first.alert_id
        assert mock_redis.xadd.call_count == 2
        assert f"chakra:alert:dedup:{alert_diff_ev.alert_id}" in existing_keys

        # 4. Genuinely different address remains distinguishable
        hit_different_address = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
            source="OFAC",
            evidence_id="ev-sdn-999",
            reason="Sanctioned match"
        )
        alert_diff_addr = await service.generate_sanction_alert(hit_different_address)
        assert alert_diff_addr is not None
        assert alert_diff_addr.alert_id != alert_first.alert_id
        assert alert_diff_addr.alert_id != alert_diff_ev.alert_id
        assert mock_redis.xadd.call_count == 3
        assert f"chakra:alert:dedup:{alert_diff_addr.alert_id}" in existing_keys


    @pytest.mark.asyncio
    async def test_no_redis_publish_on_invalid_input(self, alert_service):
        service, mock_redis = alert_service
        
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="",
            source="OFAC",
            evidence_id="ev-123",
            reason="test"
        )
        
        await service.generate_sanction_alert(hit)
        mock_redis.xadd.assert_not_called()
        
    def test_alert_record_frozen(self):
        """Ensure the human review constraint and record are immutable."""
        hit = SanctionedAddressHit(
            chain=Chain.BITCOIN,
            address="1A1",
            source="OFAC",
            evidence_id="123",
            reason="reason"
        )
        
        record = AlertRecord(
            alert_id="1",
            alert_type=AlertType.SANCTIONED_ADDRESS_HIT,
            chain=hit.chain,
            address=hit.address,
            severity=AlertSeverity.CRITICAL,
            reason=hit.reason,
            evidence_ids=[hit.evidence_id],
            source=hit.source,
            created_at="2026-01-01T00:00:00Z"
        )
        
        with pytest.raises(ValidationError):
            record.requires_human_review = False
            
        with pytest.raises(ValidationError):
            record.status = AlertStatus.REVIEWED
