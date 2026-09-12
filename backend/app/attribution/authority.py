"""CHAKRA Step 4.2: Authority Policy for VASP Attribution Provenance.

This module formalizes the explicit, centralized authority policy governing
whether a VASP attribution observation satisfies the strict authoritative
contract required to support CONFIRMED attribution confidence.
"""
from enum import Enum
from typing import Set
from app.attribution.models import VASPAddressRecord


class AuthorityClass(str, Enum):
    """Categorical classification of observation provenance authority."""
    AUTHORITATIVE = "AUTHORITATIVE"
    NON_AUTHORITATIVE = "NON_AUTHORITATIVE"
    DEMO = "DEMO"
    UNKNOWN = "UNKNOWN"


# Canonical, explicitly recognized authoritative source identifiers.
# Every retained value has defined semantics within CHAKRA:
# - "official": Direct, cryptographically signed or legally attested primary disclosure
#   from the VASP entity (e.g. proof of reserves, official regulatory filing).
# - "verified_institutional": Attested counterparty verification record with verified chain-of-custody.
# - "statutory_register": Statutory registry entry published by an official financial authority / regulator.
# - "regulatory_gazette": Official government or regulatory gazette publication.
RECOGNIZED_AUTHORITATIVE_SOURCES: Set[str] = {
    "official",
    "verified_institutional",
    "statutory_register",
    "regulatory_gazette",
}

# Substring markers indicating synthetic, demo, or testing data
SYNTHETIC_MARKERS: Set[str] = {
    "demo",
    "synthetic",
    "test",
    "mock",
    "dummy",
    "sample",
    "fake",
}

# Placeholder strings that do NOT constitute valid evidence IDs
INVALID_EVIDENCE_IDS: Set[str] = {
    "",
    "none",
    "null",
    "unknown",
    "0",
    "undefined",
    "n/a",
    "na",
}


class AuthorityPolicy:
    """
    Deterministic, auditable policy governing provenance authority in Step 4.2.
    
    Determines whether a VASPAddressRecord observation satisfies the strict
    authoritative contract required to support CONFIRMED attribution confidence.
    """

    @classmethod
    def is_synthetic(cls, record: VASPAddressRecord) -> bool:
        """Check if an observation is synthetic/demo/test data."""
        source_clean = record.source.strip().lower() if record.source else ""
        evidence_clean = record.evidence_id.strip().lower() if record.evidence_id else ""
        
        for marker in SYNTHETIC_MARKERS:
            if marker in source_clean or marker in evidence_clean:
                return True
        return False

    @classmethod
    def classify_observation(cls, record: VASPAddressRecord) -> AuthorityClass:
        """
        Classify the authority of a VASP observation strictly based on provenance.
        
        CRITICAL RULES:
        1. Arbitrary metadata is NEVER used to manufacture authority.
        2. Loose matching or fuzzy strings ("official-demo", "not-official") are rejected.
        3. Demo/synthetic records can NEVER be classified as AUTHORITATIVE.
        4. Missing or placeholder evidence_id invalidates authority.
        """
        # 1. Validate mandatory provenance fields
        source_raw = record.source.strip() if record.source else ""
        evidence_raw = record.evidence_id.strip() if record.evidence_id else ""
        
        if not source_raw:
            return AuthorityClass.UNKNOWN
            
        if not evidence_raw or evidence_raw.lower() in INVALID_EVIDENCE_IDS:
            return AuthorityClass.UNKNOWN

        # 2. Check for synthetic / demo / test provenance
        if cls.is_synthetic(record):
            return AuthorityClass.DEMO

        # 3. Exact match against recognized authoritative sources (no loose matching!)
        # e.g. "official-demo", "unofficial", "not-official" MUST NOT pass.
        source_normalized = source_raw.lower()
        if source_normalized in RECOGNIZED_AUTHORITATIVE_SOURCES:
            return AuthorityClass.AUTHORITATIVE

        # 4. Valid observation from non-authoritative source (e.g. public_tag, heuristic, explorer)
        return AuthorityClass.NON_AUTHORITATIVE

    @classmethod
    def is_authoritative(cls, record: VASPAddressRecord) -> bool:
        """Convenience method returning True iff observation is strictly AUTHORITATIVE."""
        return cls.classify_observation(record) == AuthorityClass.AUTHORITATIVE
