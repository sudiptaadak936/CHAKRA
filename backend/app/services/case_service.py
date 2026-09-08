"""Canonical case intake service for CHAKRA.

Provides the single authoritative entry point for case registration.
Does NOT perform forensic traversal, clustering, scoring, or external network calls.
"""
import uuid
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from app.schemas.case import CaseIntakeRequest, CaseIntakeResponse


@runtime_checkable
class CaseIntakeProtocol(Protocol):
    """Protocol for case intake implementations."""

    def submit_case(self, request: CaseIntakeRequest) -> CaseIntakeResponse:
        ...


class CaseIntakeService:
    """Canonical service handling case registration in CHAKRA.

    This service is the single internal intake flow into which all intake
    channels (manual, API, or external stubs like NCRP) delegate.
    """

    def submit_case(self, request: CaseIntakeRequest) -> CaseIntakeResponse:
        """Register a case into CHAKRA.

        Generates an internal case identifier (`case_<uuid4>`) and produces
        the canonical `CaseIntakeResponse`.
        """
        # Deterministic generation prefix for CHAKRA internal cases
        internal_case_id = f"case_{uuid.uuid4().hex[:12]}"

        return CaseIntakeResponse(
            case_id=internal_case_id,
            status="registered",
            wallet_address=request.wallet_address,
            chain=request.chain,
            created_at=datetime.now(timezone.utc),
        )


_case_service_instance = CaseIntakeService()


def get_case_service() -> CaseIntakeService:
    """FastAPI dependency provider for CaseIntakeService.

    Allows test suites to override the canonical service using app.dependency_overrides.
    """
    return _case_service_instance
