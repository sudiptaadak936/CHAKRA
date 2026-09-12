-- 005_suspect_registry_schema.sql
-- Schema for Suspect Registry (CHAKRA Step 6)
--
-- IMPORTANT DISCLAIMER: The presence of an address in this registry does NOT
-- imply criminality, proven guilt, confirmed ownership, or legal culpability.
-- Registry entries are investigative observations only and ALWAYS require human review.

CREATE TABLE IF NOT EXISTS suspect_registry_records (
    id BIGSERIAL PRIMARY KEY,
    record_id VARCHAR(64) NOT NULL UNIQUE,
    case_id VARCHAR(255) NOT NULL,
    chain VARCHAR(50) NOT NULL,
    network VARCHAR(50) NOT NULL,
    address TEXT NOT NULL,
    normalized_address TEXT NOT NULL,
    composite_id TEXT NOT NULL,
    suspect_role VARCHAR(50) NOT NULL DEFAULT 'SUSPECT_TARGET',
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    source VARCHAR(255) NOT NULL,
    evidence_id VARCHAR(255) NOT NULL,
    reported_by VARCHAR(255) NOT NULL,
    entity_label VARCHAR(255),
    external_reference_id VARCHAR(255),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_suspect_record UNIQUE (chain, network, normalized_address, case_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_suspect_registry_lookup
    ON suspect_registry_records(chain, network, normalized_address);

CREATE INDEX IF NOT EXISTS idx_suspect_registry_composite
    ON suspect_registry_records(composite_id);

CREATE INDEX IF NOT EXISTS idx_suspect_registry_case
    ON suspect_registry_records(case_id);

CREATE INDEX IF NOT EXISTS idx_suspect_registry_evidence
    ON suspect_registry_records(evidence_id);
