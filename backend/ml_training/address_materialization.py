"""CHAKRA Step 5H.4: Controlled Address-Level Supervised Dataset Materialization.

Implements the deterministic, provenance-preserving materialization foundation for
address-level supervised ML datasets strictly adhering to CHAKRA's frozen 13-feature
contract (FEATURE_SCHEMA_VERSION = "1.0.0").

Core Guarantees:
- Target analytical unit is ADDRESS.
- Strictly adheres to the Absolute Non-Fabrication Rule (UNKNOWN != BENIGN).
- Enforces temporal label semantics: T_cutoff <= T_label - 24 hours.
- Enforces bounded observation windows: [T_start, T_cutoff].
- Enforces minimum observation requirements: >= 2 distinct canonical transactions.
- Preserves entity grouping metadata for downstream leakage prevention.
- Deterministic conflict handling and fail-closed validation.
- Deterministic JSONL serialization with SHA-256 checksums.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.forensics.ml_features import MLFeatureExtractor
from app.graph.models import normalize_address
from app.schemas.chain import Chain, Network
from app.schemas.ml_features import (
    CANONICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MLFeatureRecord,
)

logger = logging.getLogger(__name__)


class LabelSemantics(str, Enum):
    """Formal ontology tiers for address supervision labels."""
    POSITIVE_LEGAL_SANCTION = "POSITIVE_LEGAL_SANCTION"      # Tier 1: OFAC SDN direct designation
    POSITIVE_VERIFIED_ILLICIT = "POSITIVE_VERIFIED_ILLICIT"  # Tier 2: Documented academic research attribution
    NEGATIVE_VERIFIED_VASP = "NEGATIVE_VERIFIED_VASP"        # Regulated exchange / custodial infrastructure
    NEGATIVE_VERIFIED_MINING = "NEGATIVE_VERIFIED_MINING"    # Verified mining pool / infrastructure
    NEGATIVE_VERIFIED_BENIGN = "NEGATIVE_VERIFIED_BENIGN"    # Verified commercial / benign service
    UNKNOWN = "UNKNOWN"                                      # Default unverified state (CANNOT BE USED AS NEGATIVE)


class ExclusionReason(str, Enum):
    """Deterministic classifications for excluded candidate addresses."""
    INSUFFICIENT_OBSERVATION = "INSUFFICIENT_OBSERVATION"        # < 2 transactions in observation window
    MISSING_LABEL_TIMESTAMP = "MISSING_LABEL_TIMESTAMP"          # No verified T_label for positive sample
    INVALID_CUTOFF_WINDOW = "INVALID_CUTOFF_WINDOW"              # T_cutoff > T_label - 24h (leakage risk)
    LABEL_CONFLICT = "LABEL_CONFLICT"                            # Conflicting positive vs negative attribution
    UNKNOWN_NEGATIVE = "UNKNOWN_NEGATIVE"                        # Unknown address attempted as negative (UNKNOWN != BENIGN)
    UNSUPPORTED_CHAIN_OR_ADDRESS = "UNSUPPORTED_CHAIN_OR_ADDRESS"# Cannot be mapped to supported chain/network
    ENVIRONMENTAL_ACQUISITION_BLOCKED = "ENVIRONMENTAL_ACQUISITION_BLOCKED" # External data missing / blocked


class AddressCandidate(BaseModel):
    """Input specification for an address candidate undergoing supervised materialization."""
    chain: str
    network: str
    address: str
    normalized_address: Optional[str] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    label: Optional[int] = None
    label_source: str
    label_source_version: Optional[str] = None
    label_timestamp: Optional[datetime] = None
    label_semantics: LabelSemantics = LabelSemantics.UNKNOWN

    model_config = {"frozen": True}


class MaterializedAddressSample(BaseModel):
    """Immutable domain model for an address-level supervised training sample.

    Strictly satisfies the Step 5H.4 Materialized Sample Contract.
    """
    sample_id: str = Field(..., description="Deterministic unique identifier: sha256(chain:network:norm_addr:cutoff)")
    chain: str
    network: str
    normalized_address: str
    entity_id: Optional[str] = Field(default=None, description="Entity identifier for leakage-controlled grouped splitting")
    entity_name: Optional[str] = None
    observation_start: Optional[datetime] = Field(default=None, description="T_start of observation window (None = lifetime up to cutoff)")
    observation_end: datetime = Field(..., description="T_cutoff upper temporal boundary")
    label: int = Field(..., ge=0, le=1, description="Binary supervision target: 1=positive, 0=negative")
    label_source: str
    label_source_version: Optional[str] = None
    label_timestamp: datetime = Field(..., description="T_label earliest public designation timestamp")
    label_semantics: str
    feature_schema_version: str = Field(default=FEATURE_SCHEMA_VERSION, frozen=True)
    feature_values: List[float] = Field(..., description="Canonical 13-feature ordered float vector")
    feature_provenance: Dict[str, Any] = Field(default_factory=dict)
    materialized_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": True}


def compute_sample_id(
    chain: str,
    network: str,
    normalized_address: str,
    observation_end: datetime,
) -> str:
    """Generate a deterministic, reproducible sample ID."""
    iso_cutoff = observation_end.astimezone(timezone.utc).isoformat()
    raw = f"{chain.lower()}:{network.lower()}:{normalized_address}:{iso_cutoff}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def validate_candidate_eligibility(
    candidate: AddressCandidate,
    observation_end: datetime,
    observation_start: Optional[datetime] = None,
) -> Tuple[bool, Optional[ExclusionReason], Optional[str]]:
    """Validate that candidate address satisfies all ontology, temporal, and leakage invariants."""
    # 1. Address and Chain Normalization
    try:
        norm_addr = normalize_address(candidate.chain, candidate.address)
        if not norm_addr:
            return False, ExclusionReason.UNSUPPORTED_CHAIN_OR_ADDRESS, "Normalization produced empty string"
    except Exception as e:
        return False, ExclusionReason.UNSUPPORTED_CHAIN_OR_ADDRESS, f"Address normalization failed: {e}"

    # 2. Temporal Window Bounds
    if observation_start is not None and observation_start > observation_end:
        return False, ExclusionReason.INVALID_CUTOFF_WINDOW, f"observation_start ({observation_start}) > observation_end ({observation_end})"

    # 3. Positive Label Validation
    if candidate.label == 1:
        if candidate.label_timestamp is None:
            return False, ExclusionReason.MISSING_LABEL_TIMESTAMP, "Supervised positive candidate lacks label_timestamp (T_label)"

        # 24-hour pre-designation buffer: T_cutoff <= T_label - 24h
        max_allowed_cutoff = candidate.label_timestamp - timedelta(hours=24)
        if observation_end > max_allowed_cutoff:
            return False, ExclusionReason.INVALID_CUTOFF_WINDOW, (
                f"Cutoff {observation_end.isoformat()} exceeds max allowed cutoff "
                f"{max_allowed_cutoff.isoformat()} (T_label - 24h)"
            )

        if candidate.label_semantics not in {
            LabelSemantics.POSITIVE_LEGAL_SANCTION,
            LabelSemantics.POSITIVE_VERIFIED_ILLICIT,
        }:
            return False, ExclusionReason.LABEL_CONFLICT, (
                f"Invalid label semantics for positive sample: {candidate.label_semantics}"
            )

    # 4. Negative Label Validation — UNKNOWN != BENIGN
    elif candidate.label == 0:
        if candidate.label_semantics not in {
            LabelSemantics.NEGATIVE_VERIFIED_VASP,
            LabelSemantics.NEGATIVE_VERIFIED_MINING,
            LabelSemantics.NEGATIVE_VERIFIED_BENIGN,
        }:
            return False, ExclusionReason.UNKNOWN_NEGATIVE, (
                f"Negative candidate lacks verified reference-negative ontology classification: {candidate.label_semantics}"
            )
    else:
        return False, ExclusionReason.LABEL_CONFLICT, f"Candidate has invalid label: {candidate.label}"

    return True, None, None


class AddressDatasetMaterializer:
    """Materialization engine converting canonical blockchain evidence into address-level supervised records."""

    @classmethod
    def materialize_sample_from_evidence(
        cls,
        candidate: AddressCandidate,
        observation_end: datetime,
        observation_start: Optional[datetime] = None,
        transfers: Optional[List[Any]] = None,
        transactions: Optional[List[Any]] = None,
        bitcoin_vins: Optional[List[Any]] = None,
        bitcoin_vouts: Optional[List[Any]] = None,
        typologies: Optional[List[Any]] = None,
        extracted_at: Optional[datetime] = None,
        materialized_at: Optional[datetime] = None,
    ) -> Tuple[Optional[MaterializedAddressSample], Optional[ExclusionReason], Optional[str]]:
        """Materialize a single address sample from pre-loaded canonical evidence."""
        # 1. Eligibility Check
        eligible, reason, msg = validate_candidate_eligibility(
            candidate=candidate,
            observation_end=observation_end,
            observation_start=observation_start,
        )
        if not eligible:
            return None, reason, msg

        norm_addr = normalize_address(candidate.chain, candidate.address)
        ext_time = extracted_at or observation_end
        mat_time = materialized_at or ext_time

        # 2. Extract Features using Temporal Cutoff & Bounded Window
        try:
            record: MLFeatureRecord = MLFeatureExtractor.extract_from_evidence(
                target_address=candidate.address,
                chain=candidate.chain,
                network=candidate.network,
                transfers=transfers,
                transactions=transactions,
                bitcoin_vins=bitcoin_vins,
                bitcoin_vouts=bitcoin_vouts,
                typologies=typologies,
                extracted_at=ext_time,
                cutoff_timestamp=observation_end,
                start_timestamp=observation_start,
            )
        except Exception as e:
            return None, ExclusionReason.UNSUPPORTED_CHAIN_OR_ADDRESS, f"Feature extraction failed: {e}"

        # 3. Minimum Observation Check (>= 2 transactions in observation window)
        tx_count = record.features.total_tx_count
        if tx_count < 2:
            return None, ExclusionReason.INSUFFICIENT_OBSERVATION, (
                f"Candidate address {norm_addr} has {tx_count} transactions in observation window (< 2 required)"
            )

        # 4. Canonical Feature Vector Validation
        vector = record.to_model_vector()
        if len(vector) != 13:
            raise ValueError(f"Feature vector length {len(vector)} does not match canonical 13 features")

        for idx, val in enumerate(vector):
            if not math.isfinite(val):
                raise ValueError(f"Feature {CANONICAL_FEATURE_NAMES[idx]} value {val} is not finite")

        if record.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Feature schema version {record.feature_schema_version} != {FEATURE_SCHEMA_VERSION}")

        # 5. Build Materialized Sample
        sample_id = compute_sample_id(
            chain=candidate.chain,
            network=candidate.network,
            normalized_address=norm_addr,
            observation_end=observation_end,
        )

        provenance = {
            "total_tx_count": tx_count,
            "in_degree": record.features.in_degree,
            "out_degree": record.features.out_degree,
            "total_received_native": float(record.features.total_received_native),
            "total_sent_native": float(record.features.total_sent_native),
            "amount_retention_ratio": record.features.amount_retention_ratio,
            "time_active_seconds": record.features.time_active_seconds,
            "inter_hop_velocity_avg": record.features.inter_hop_velocity_avg,
            "extracted_at": ext_time.isoformat(),
        }

        sample = MaterializedAddressSample(
            sample_id=sample_id,
            chain=candidate.chain,
            network=candidate.network,
            normalized_address=norm_addr,
            entity_id=candidate.entity_id,
            entity_name=candidate.entity_name,
            observation_start=observation_start,
            observation_end=observation_end,
            label=candidate.label,  # type: ignore[arg-type]
            label_source=candidate.label_source,
            label_source_version=candidate.label_source_version,
            label_timestamp=candidate.label_timestamp,  # type: ignore[arg-type]
            label_semantics=candidate.label_semantics.value,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_values=vector,
            feature_provenance=provenance,
            materialized_at=mat_time,
        )

        return sample, None, None


def serialize_materialized_dataset(
    samples: List[MaterializedAddressSample],
    output_path: Path,
) -> Dict[str, Any]:
    """Serialize materialized samples deterministically to JSONL with SHA-256 hash calculation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic sorting by sample_id
    sorted_samples = sorted(samples, key=lambda s: s.sample_id)

    sha256 = hashlib.sha256()

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for s in sorted_samples:
            row_dict = s.model_dump(mode="json")
            # Deterministic JSON with sorted keys
            line = json.dumps(row_dict, sort_keys=True) + "\n"
            f.write(line)
            sha256.update(line.encode("utf-8"))

    file_hash = sha256.hexdigest()

    checksums_path = output_path.with_suffix(".sha256")
    with open(checksums_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{file_hash}  {output_path.name}\n")

    return {
        "output_path": str(output_path),
        "record_count": len(sorted_samples),
        "sha256": file_hash,
    }
