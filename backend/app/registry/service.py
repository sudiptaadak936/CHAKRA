"""CHAKRA Step 6: Suspect Registry — domain service.

Provides two primary operations:
  1. register()               — Validate inputs, normalize address, compute deterministic
                                record_id, and persist via repository.
  2. evaluate_cross_case_hits() — Query repository and produce a CrossCaseHitResult
                                   when the same address appears in >= 2 distinct cases.

Safety invariants enforced by this service:
  - No random UUIDs: all identity fields are deterministic SHA-256 hex digests.
  - No Redis publishing, no alerts, no external API calls.
  - requires_human_review is always True in CrossCaseHitResult (schema-enforced).
  - Fail-closed: missing / invalid inputs raise ValueError before any DB write.
  - Registry presence never implies criminality, guilt, or legal culpability.

INFERRED_CLUSTER_LINK cross-case detection is conservatively deferred:
  ClusteringRepository does not expose get_cluster_for_composite_id().
  Only DIRECT_ADDRESS_MATCH is supported in Step 6 initial scope.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from app.graph.models import make_address_composite_id, normalize_address
from app.registry.repository import SuspectRegistryRepository
from app.schemas.chain import Chain, Network
from app.schemas.registry import (
    CrossCaseHitResult,
    CrossCaseLinkType,
    MatchedCaseContext,
    RegistryStatus,
    SuspectRegistryRecord,
    SuspectRole,
)

logger = logging.getLogger(__name__)


def _compute_record_id(
    chain: str,
    network: str,
    normalized_address: str,
    case_id: str,
    evidence_id: str,
) -> str:
    """Deterministic SHA-256 record identity.

    Formula: SHA-256( "{chain}:{network}:{normalized_address}:{case_id}:{evidence_id}" )
    """
    raw_key = f"{chain}:{network}:{normalized_address}:{case_id}:{evidence_id}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _compute_hit_id(
    chain: str,
    network: str,
    normalized_address: str,
    case_ids: List[str],
) -> str:
    """Deterministic SHA-256 cross-case hit identity.

    Formula: SHA-256( "{chain}:{network}:{normalized_address}:{':'.join(sorted(case_ids))}" )
    """
    sorted_cases = sorted(case_ids)
    raw_hit = f"{chain}:{network}:{normalized_address}:{':'.join(sorted_cases)}"
    return hashlib.sha256(raw_hit.encode("utf-8")).hexdigest()


class SuspectRegistryService:
    """Domain service for the CHAKRA Suspect Registry (Step 6).

    DISCLAIMER: The presence of an address in the registry is an investigative
    observation only. It does NOT imply criminality, guilt, confirmed ownership,
    or legal liability. All cross-case hits require authorized human review.
    """

    def __init__(self, repository: SuspectRegistryRepository) -> None:
        self._repo = repository

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_chain(chain: Any) -> Chain:
        """Validate chain value; raises ValueError if invalid."""
        if isinstance(chain, Chain):
            return chain
        try:
            return Chain(str(chain).lower())
        except ValueError:
            raise ValueError(
                f"Invalid chain '{chain}'. Must be one of: "
                f"{[c.value for c in Chain]}"
            )

    @staticmethod
    def _validate_network(network: Any) -> Network:
        """Validate network value; raises ValueError if invalid."""
        if isinstance(network, Network):
            return network
        try:
            return Network(str(network).lower())
        except ValueError:
            raise ValueError(
                f"Invalid network '{network}'. Must be one of: "
                f"{[n.value for n in Network]}"
            )

    @staticmethod
    def _require_non_empty(value: Any, field_name: str) -> str:
        """Require a non-empty, non-whitespace string field."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Field '{field_name}' must be a non-empty string, got: {value!r}"
            )
        return value.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register(
        self,
        *,
        case_id: str,
        chain: Any,
        network: Any,
        address: str,
        source: str,
        evidence_id: str,
        reported_by: str,
        suspect_role: SuspectRole = SuspectRole.SUSPECT_TARGET,
        status: RegistryStatus = RegistryStatus.ACTIVE,
        entity_label: Optional[str] = None,
        external_reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SuspectRegistryRecord:
        """Validate inputs, normalize address, compute record_id, and persist.

        Returns the fully populated SuspectRegistryRecord (idempotent: repeated
        calls with the same natural key are safe — the record already in the DB
        is the authoritative entry and this call silently succeeds).

        Raises ValueError for any invalid input before any DB write (fail-closed).
        """
        # --- Validate mandatory fields ---
        case_id = self._require_non_empty(case_id, "case_id")
        address = self._require_non_empty(address, "address")
        source = self._require_non_empty(source, "source")
        evidence_id = self._require_non_empty(evidence_id, "evidence_id")
        reported_by = self._require_non_empty(reported_by, "reported_by")

        chain_enum = self._validate_chain(chain)
        network_enum = self._validate_network(network)

        # --- Normalize address ---
        norm_addr = normalize_address(chain_enum.value, address)
        if not norm_addr:
            raise ValueError(
                f"Address normalization produced empty string for address={address!r}, "
                f"chain={chain_enum.value!r}"
            )

        # --- Compute deterministic composite_id and record_id ---
        composite_id = make_address_composite_id(chain_enum.value, network_enum.value, address)
        record_id = _compute_record_id(
            chain=chain_enum.value,
            network=network_enum.value,
            normalized_address=norm_addr,
            case_id=case_id,
            evidence_id=evidence_id,
        )

        record = SuspectRegistryRecord(
            record_id=record_id,
            case_id=case_id,
            chain=chain_enum.value,
            network=network_enum.value,
            address=address.strip(),
            normalized_address=norm_addr,
            composite_id=composite_id,
            suspect_role=suspect_role,
            status=status,
            source=source,
            evidence_id=evidence_id,
            reported_by=reported_by,
            entity_label=entity_label,
            external_reference_id=external_reference_id,
            metadata=metadata or {},
        )

        await self._repo.save_record(record)
        logger.info(
            "Suspect registry record saved: record_id=%s composite_id=%s case_id=%s",
            record_id,
            composite_id,
            case_id,
        )
        return record

    async def evaluate_cross_case_hits(
        self, composite_id: str
    ) -> Optional[CrossCaseHitResult]:
        """Evaluate whether the given composite_id appears in >= 2 distinct cases.

        Returns a CrossCaseHitResult (requires_human_review=True, always) if a
        cross-case signal exists, or None if fewer than 2 distinct cases share
        this composite_id.

        DISCLAIMER: A cross-case hit is an investigative signal only. It does NOT
        imply guilt, criminality, or legal culpability. Authorized human review is
        always required before any operational action.

        NOTE: Only DIRECT_ADDRESS_MATCH is implemented in Step 6 initial scope.
        INFERRED_CLUSTER_LINK is conservatively deferred — ClusteringRepository
        does not expose a get_cluster_for_composite_id() interface.
        """
        if not composite_id or not composite_id.strip():
            return None

        records = await self._repo.find_cross_case_matches(composite_id.strip())
        if not records:
            return None

        # Group records by case_id to build MatchedCaseContext per case
        cases_seen: Dict[str, List[SuspectRegistryRecord]] = {}
        for r in records:
            cases_seen.setdefault(r.case_id, []).append(r)

        if len(cases_seen) < 2:
            return None

        # Take the first (oldest by created_at order) record per case for context
        matched_cases: List[MatchedCaseContext] = []
        for cid in sorted(cases_seen.keys()):
            rep = cases_seen[cid][0]
            matched_cases.append(
                MatchedCaseContext(
                    case_id=rep.case_id,
                    evidence_id=rep.evidence_id,
                    source=rep.source,
                    reported_by=rep.reported_by,
                    suspect_role=rep.suspect_role,
                    registered_at=rep.created_at,
                )
            )

        sorted_case_ids = sorted(cases_seen.keys())

        # All records share the same chain/network/normalized_address (same composite_id)
        first = records[0]
        hit_id = _compute_hit_id(
            chain=first.chain,
            network=first.network,
            normalized_address=first.normalized_address,
            case_ids=sorted_case_ids,
        )

        result = CrossCaseHitResult(
            hit_id=hit_id,
            composite_id=composite_id.strip(),
            chain=first.chain,
            network=first.network,
            normalized_address=first.normalized_address,
            matched_case_ids=sorted_case_ids,
            matched_cases=matched_cases,
            link_type=CrossCaseLinkType.DIRECT_ADDRESS_MATCH,
            requires_human_review=True,
        )

        logger.info(
            "Cross-case hit detected: hit_id=%s composite_id=%s case_count=%d requires_human_review=%s",
            hit_id,
            composite_id,
            len(sorted_case_ids),
            result.requires_human_review,
        )
        return result
