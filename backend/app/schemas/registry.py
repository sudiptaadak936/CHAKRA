"""CHAKRA Step 6: Suspect Registry — Pydantic schemas.

IMPORTANT DISCLAIMER:
    Registry entries represent investigative observations only.
    The presence of an address in the Suspect Registry does NOT imply
    criminality, proven guilt, confirmed ownership, or legal culpability.
    All cross-case hit results ALWAYS require human review.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SuspectRole(str, Enum):
    """Role assigned to the flagged address within a case.

    IMPORTANT: These labels are investigative classifications only.
    They do not constitute legal determinations.
    """
    SUSPECT_TARGET = "SUSPECT_TARGET"
    CASH_OUT = "CASH_OUT"
    INTERMEDIARY = "INTERMEDIARY"
    UNKNOWN_ROLE = "UNKNOWN_ROLE"


class RegistryStatus(str, Enum):
    """Lifecycle status of a registry record."""
    ACTIVE = "ACTIVE"
    PROVISIONAL = "PROVISIONAL"
    ARCHIVED = "ARCHIVED"
    DISMISSED = "DISMISSED"


class CrossCaseLinkType(str, Enum):
    """The mechanism by which the cross-case match was established.

    - DIRECT_ADDRESS_MATCH: The exact normalized address appears in >= 2 cases.
    - INFERRED_CLUSTER_LINK: A clustering sibling appears in >= 2 cases.
      NOTE: INFERRED_CLUSTER_LINK is conservatively deferred in Step 6 initial
      implementation. ClusteringRepository does not expose a
      get_cluster_for_composite_id() query. This variant is reserved for a
      future sub-step when that interface is available.
    """
    DIRECT_ADDRESS_MATCH = "DIRECT_ADDRESS_MATCH"
    INFERRED_CLUSTER_LINK = "INFERRED_CLUSTER_LINK"


class SuspectRegistryRecord(BaseModel):
    """Immutable domain model for a single Suspect Registry entry.

    Identity is determined by (chain, network, normalized_address, case_id, evidence_id).
    record_id is a deterministic SHA-256 of those five fields — NEVER a random UUID.

    DISCLAIMER: The presence of this record does NOT imply guilt or criminality.
    """
    record_id: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Deterministic SHA-256 hex digest of the canonical identity fields.",
    )
    case_id: str = Field(..., min_length=1)
    chain: str = Field(..., min_length=1)
    network: str = Field(..., min_length=1)
    address: str = Field(..., min_length=1)
    normalized_address: str = Field(..., min_length=1)
    composite_id: str = Field(..., min_length=1)
    suspect_role: SuspectRole = Field(default=SuspectRole.SUSPECT_TARGET)
    status: RegistryStatus = Field(default=RegistryStatus.ACTIVE)
    source: str = Field(..., min_length=1)
    evidence_id: str = Field(..., min_length=1)
    reported_by: str = Field(..., min_length=1)
    entity_label: Optional[str] = None
    external_reference_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"frozen": True}


class MatchedCaseContext(BaseModel):
    """Per-case provenance context within a cross-case hit.

    Captures which case reported this address, with what evidence, and by whom.
    """
    case_id: str = Field(..., min_length=1)
    evidence_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    reported_by: str = Field(..., min_length=1)
    suspect_role: SuspectRole
    registered_at: Optional[datetime] = None

    model_config = {"frozen": True}


class CrossCaseHitResult(BaseModel):
    """Result produced when the same address appears in >= 2 distinct cases.

    MANDATORY SAFETY INVARIANTS (enforced at construction time):
      - requires_human_review is ALWAYS True and cannot be overridden.
      - hit_id is a deterministic SHA-256 — never random.
      - link_type is always DIRECT_ADDRESS_MATCH in Step 6 initial scope.

    DISCLAIMER: A cross-case hit is an investigative signal, NOT a determination
    of guilt, criminality, confirmed ownership, or legal liability. All results
    MUST be reviewed by an authorized human investigator before any operational
    use.
    """
    hit_id: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Deterministic SHA-256 hex digest of (chain:network:normalized_address:sorted_case_ids).",
    )
    composite_id: str = Field(..., min_length=1)
    chain: str = Field(..., min_length=1)
    network: str = Field(..., min_length=1)
    normalized_address: str = Field(..., min_length=1)
    matched_case_ids: List[str] = Field(
        ...,
        min_length=2,
        description="Sorted list of distinct case IDs that share this address. Min 2.",
    )
    matched_cases: List[MatchedCaseContext] = Field(
        ...,
        min_length=2,
        description="Per-case provenance contexts, one per matched case.",
    )
    link_type: CrossCaseLinkType = Field(default=CrossCaseLinkType.DIRECT_ADDRESS_MATCH)
    # Safety invariant: this field is permanently True and cannot be set to False.
    requires_human_review: bool = Field(
        default=True,
        frozen=True,
        description=(
            "ALWAYS True. Cross-case hits MUST be reviewed by an authorized human "
            "investigator. This field cannot be set to False."
        ),
    )

    model_config = {"frozen": True}
