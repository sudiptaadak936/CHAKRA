"""
Tests for FIX 1/2: FIURegistryRepository — demonstration FIU-IND registry.

Invariants tested:
- UNKNOWN is never silently treated as NOT_REGISTERED
- Data classification label is always DEMONSTRATION_ONLY
- Tri-state status is precise
"""
import json
import tempfile
from pathlib import Path

import pytest

from app.attribution.fiu_registry import (
    FIURegistrationStatus,
    FIURegistryRecord,
    FIURegistryRepository,
)


@pytest.fixture()
def demo_registry_file(tmp_path: Path) -> Path:
    """Create a minimal valid demo_fiu_registry.json."""
    data = {
        "_metadata": {
            "data_classification": "DEMONSTRATION_ONLY",
            "disclaimer": "TEST DATA",
            "source": "TEST",
            "schema_version": "1.0.0",
        },
        "vasps": [
            {
                "re_id": "FIU-TEST-001",
                "vasp_name": "RegisteredExchange",
                "registration_status": "REGISTERED",
                "registered_jurisdiction": "IN",
                "registration_date": "2023-01-01",
                "compliance_officer_contact": "test@example.invalid",
                "notes": "TEST DEMO RECORD",
            },
            {
                "re_id": None,
                "vasp_name": "UnregisteredExchange",
                "registration_status": "NOT_REGISTERED",
                "registered_jurisdiction": None,
                "registration_date": None,
                "compliance_officer_contact": None,
                "notes": "TEST DEMO RECORD",
            },
        ],
    }
    path = tmp_path / "demo_fiu_registry.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestFIURegistryRepository:
    def test_loads_registered_vasp(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        record = repo.lookup_vasp("RegisteredExchange")
        assert record is not None
        assert record.registration_status == FIURegistrationStatus.REGISTERED
        assert record.re_id == "FIU-TEST-001"
        assert record.data_classification == "DEMONSTRATION_ONLY"

    def test_loads_not_registered_vasp(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        record = repo.lookup_vasp("UnregisteredExchange")
        assert record is not None
        assert record.registration_status == FIURegistrationStatus.NOT_REGISTERED
        assert record.re_id is None

    def test_unknown_vasp_returns_unknown_not_not_registered(self, demo_registry_file):
        """INVARIANT: absent VASP => UNKNOWN, never NOT_REGISTERED."""
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        status = repo.is_registered("CompletelyUnknownVASP")
        assert status == FIURegistrationStatus.UNKNOWN
        assert status != FIURegistrationStatus.NOT_REGISTERED

    def test_lookup_case_insensitive(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.lookup_vasp("registeredexchange") is not None
        assert repo.lookup_vasp("REGISTEREDEXCHANGE") is not None

    def test_lookup_none_returns_none(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.lookup_vasp(None) is None
        assert repo.lookup_vasp("") is None

    def test_is_registered_returns_correct_status(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.is_registered("RegisteredExchange") == FIURegistrationStatus.REGISTERED
        assert repo.is_registered("UnregisteredExchange") == FIURegistrationStatus.NOT_REGISTERED
        assert repo.is_registered("DoesNotExist") == FIURegistrationStatus.UNKNOWN

    def test_get_re_id_registered(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.get_re_id("RegisteredExchange") == "FIU-TEST-001"

    def test_get_re_id_not_registered_returns_none(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.get_re_id("UnregisteredExchange") is None

    def test_get_re_id_unknown_returns_none(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.get_re_id("CompletelyUnknownVASP") is None

    def test_data_classification_property(self, demo_registry_file):
        repo = FIURegistryRepository(registry_path=demo_registry_file)
        assert repo.data_classification == "DEMONSTRATION_ONLY"

    def test_rejects_file_without_classification_label(self, tmp_path):
        """Must refuse to load a file without the DEMONSTRATION_ONLY label."""
        data = {
            "_metadata": {"data_classification": "REAL_DATA"},
            "vasps": [],
        }
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="DEMONSTRATION_ONLY"):
            FIURegistryRepository(registry_path=path)

    def test_missing_file_does_not_raise(self, tmp_path):
        """Missing file should not crash — returns UNKNOWN for all lookups."""
        path = tmp_path / "nonexistent.json"
        repo = FIURegistryRepository(registry_path=path)
        assert repo.is_registered("Anything") == FIURegistrationStatus.UNKNOWN


class TestFIURegistrationStatusEnum:
    def test_tri_state_values(self):
        assert FIURegistrationStatus.REGISTERED.value == "REGISTERED"
        assert FIURegistrationStatus.NOT_REGISTERED.value == "NOT_REGISTERED"
        assert FIURegistrationStatus.UNKNOWN.value == "UNKNOWN"

    def test_unknown_is_not_not_registered(self):
        assert FIURegistrationStatus.UNKNOWN != FIURegistrationStatus.NOT_REGISTERED

    def test_registered_is_not_unknown(self):
        assert FIURegistrationStatus.REGISTERED != FIURegistrationStatus.UNKNOWN
