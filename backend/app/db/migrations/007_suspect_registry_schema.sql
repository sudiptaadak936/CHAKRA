-- Migration 007: Suspect Registry Schema
-- Enables pgvector and creates tables for cross-case linkage.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS suspect_registry (
    registry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    address TEXT NOT NULL,
    chain TEXT NOT NULL,
    first_seen_case_id TEXT NOT NULL,
    risk_score NUMERIC(5, 2) NOT NULL,
    graph_fingerprint vector(6), -- 6-dimensional behavioral fingerprint
    status TEXT NOT NULL DEFAULT 'Active', -- Active, Monitored, Resolved
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (address, chain)
);

CREATE INDEX IF NOT EXISTS idx_suspect_registry_address ON suspect_registry(address);
CREATE INDEX IF NOT EXISTS idx_suspect_registry_chain ON suspect_registry(chain);

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_suspect_registry_fingerprint ON suspect_registry USING hnsw (graph_fingerprint vector_cosine_ops);

CREATE TABLE IF NOT EXISTS case_linkages (
    id BIGSERIAL PRIMARY KEY,
    analysis_id TEXT NOT NULL,
    registry_id UUID NOT NULL REFERENCES suspect_registry(registry_id) ON DELETE CASCADE,
    linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (analysis_id, registry_id)
);

CREATE INDEX IF NOT EXISTS idx_case_linkages_analysis_id ON case_linkages(analysis_id);

COMMENT ON TABLE suspect_registry IS 'CHAKRA Step 6 — Persistent tracking of suspicious addresses across cases.';
COMMENT ON TABLE case_linkages IS 'CHAKRA Step 6 — Many-to-many relationship tracking which cases encountered which registry suspects.';
