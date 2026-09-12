"""CHAKRA Step 6: Suspect Registry — internal query/result models.

These models are used internally by the repository and service layers.
They are distinct from the public Pydantic schemas in app.schemas.registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RegistryLookupResult:
    """Raw result returned by a repository lookup, before domain-model hydration.

    Fields mirror the suspect_registry_records table columns exactly.
    Consumers should hydrate these into SuspectRegistryRecord via the repository.
    """
    record_id: str
    case_id: str
    chain: str
    network: str
    address: str
    normalized_address: str
    composite_id: str
    suspect_role: str
    status: str
    source: str
    evidence_id: str
    reported_by: str
    entity_label: Optional[str]
    external_reference_id: Optional[str]
    metadata: dict
    created_at: Optional[object]
    updated_at: Optional[object]


@dataclass(frozen=True)
class CrossCaseCandidate:
    """Aggregated composite-id cross-case candidate before CrossCaseHitResult is built.

    Used internally in the service when multiple registry records for the same
    composite_id span >= 2 distinct case_ids.
    """
    composite_id: str
    chain: str
    network: str
    normalized_address: str
    case_ids: List[str] = field(default_factory=list)
