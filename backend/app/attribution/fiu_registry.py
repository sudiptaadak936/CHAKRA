"""
FIU-IND Registry lookup service — DEMONSTRATION ONLY prototype.

This module loads demonstration data from demo_fiu_registry.json.
ALL DATA IS FABRICATED. No real FIU-IND registrations are represented.
No network calls are made. No government API is contacted.
"""
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical path to demonstration data file
_DEMO_REGISTRY_PATH = Path(__file__).parent.parent.parent / "data" / "demo_fiu_registry.json"

_DEMO_CLASSIFICATION_LABEL = "DEMONSTRATION_ONLY"


class FIURegistrationStatus(str, Enum):
    """
    Explicit tri-state registration status.

    REGISTERED     — Demonstration record shows VASP as registered.
    NOT_REGISTERED — Demonstration record shows VASP as NOT registered.
    UNKNOWN        — VASP is not in the demonstration registry.
                     UNKNOWN must NEVER be silently treated as NOT_REGISTERED.
    """
    REGISTERED = "REGISTERED"
    NOT_REGISTERED = "NOT_REGISTERED"
    UNKNOWN = "UNKNOWN"


class FIURegistryRecord:
    """
    Holds one resolved FIU-IND registry entry.
    All fields are from DEMONSTRATION data; none are real government records.
    """
    __slots__ = (
        "vasp_name",
        "registration_status",
        "re_id",
        "registered_jurisdiction",
        "notes",
        "data_classification",
    )

    def __init__(
        self,
        vasp_name: str,
        registration_status: FIURegistrationStatus,
        re_id: Optional[str],
        registered_jurisdiction: Optional[str],
        notes: str,
        data_classification: str,
    ) -> None:
        self.vasp_name = vasp_name
        self.registration_status = registration_status
        self.re_id = re_id
        self.registered_jurisdiction = registered_jurisdiction
        self.notes = notes
        self.data_classification = data_classification

    def __repr__(self) -> str:
        return (
            f"FIURegistryRecord("
            f"vasp_name={self.vasp_name!r}, "
            f"status={self.registration_status.value}, "
            f"re_id={self.re_id!r}, "
            f"classification={self.data_classification!r})"
        )


class FIURegistryRepository:
    """
    In-memory lookup against the demonstration FIU-IND registry.

    Data source: backend/data/demo_fiu_registry.json
    Classification: DEMONSTRATION_ONLY — all records are fabricated.

    This class does NOT:
    - Contact any FIU-IND API or government system.
    - Make any network request.
    - Claim any real regulatory authority.

    Lookup is case-insensitive on vasp_name.
    If a VASP name is not found, status is UNKNOWN — never NOT_REGISTERED.
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        path = registry_path or _DEMO_REGISTRY_PATH
        self._index: dict[str, FIURegistryRecord] = {}
        self._classification: str = _DEMO_CLASSIFICATION_LABEL
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.error(
                "demo_fiu_registry.json not found at %s — FIU registry will return UNKNOWN for all lookups.",
                path,
            )
            return

        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load demo_fiu_registry.json: %s", exc)
            return

        # Validate top-level classification label
        metadata = raw.get("_metadata", {})
        classification = metadata.get("data_classification", "")
        if classification != _DEMO_CLASSIFICATION_LABEL:
            raise ValueError(
                f"demo_fiu_registry.json does not carry the required "
                f"data_classification=DEMONSTRATION_ONLY label. "
                f"Found: {classification!r}. Refusing to load."
            )
        self._classification = classification
        disclaimer = metadata.get("disclaimer", "")
        logger.info(
            "Loaded FIU registry (DEMONSTRATION_ONLY). Disclaimer: %s", disclaimer
        )

        for entry in raw.get("vasps", []):
            raw_status = entry.get("registration_status", "UNKNOWN").upper()
            try:
                status = FIURegistrationStatus(raw_status)
            except ValueError:
                logger.warning(
                    "Unknown registration_status value %r for VASP %r — treating as UNKNOWN.",
                    raw_status,
                    entry.get("vasp_name"),
                )
                status = FIURegistrationStatus.UNKNOWN

            record = FIURegistryRecord(
                vasp_name=entry.get("vasp_name", ""),
                registration_status=status,
                re_id=entry.get("re_id"),
                registered_jurisdiction=entry.get("registered_jurisdiction"),
                notes=entry.get("notes", ""),
                data_classification=_DEMO_CLASSIFICATION_LABEL,
            )
            key = record.vasp_name.lower()
            self._index[key] = record

        logger.info(
            "FIU demonstration registry loaded: %d entries (DEMONSTRATION_ONLY, NO REAL DATA).",
            len(self._index),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lookup_vasp(self, vasp_name: str) -> Optional[FIURegistryRecord]:
        """
        Look up a VASP by name (case-insensitive).

        Returns FIURegistryRecord if found, or None if absent.
        Callers MUST treat None as UNKNOWN, never as NOT_REGISTERED.
        """
        if not vasp_name or not vasp_name.strip():
            return None
        return self._index.get(vasp_name.strip().lower())

    def is_registered(self, vasp_name: str) -> FIURegistrationStatus:
        """
        Return the explicit tri-state registration status.

        - REGISTERED      if the demo record says REGISTERED.
        - NOT_REGISTERED  if the demo record explicitly says NOT_REGISTERED.
        - UNKNOWN         if the VASP is absent from the demo registry.
                          UNKNOWN != NOT_REGISTERED.
        """
        record = self.lookup_vasp(vasp_name)
        if record is None:
            return FIURegistrationStatus.UNKNOWN
        return record.registration_status

    def get_re_id(self, vasp_name: str) -> Optional[str]:
        """
        Return the demonstration RE-ID for a registered VASP, or None.
        None is returned for NOT_REGISTERED and UNKNOWN VASPs.
        """
        record = self.lookup_vasp(vasp_name)
        if record is None:
            return None
        return record.re_id

    @property
    def data_classification(self) -> str:
        """Always DEMONSTRATION_ONLY."""
        return self._classification
