-- Migration 006: Step 5 Risk Scoring Schema
-- Persistent storage for end-to-end risk assessments.

CREATE TABLE IF NOT EXISTS risk_assessments (
    id BIGSERIAL PRIMARY KEY,
    analysis_id TEXT NOT NULL UNIQUE,
    target_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    risk_score NUMERIC(5, 2) NOT NULL,
    risk_category TEXT NOT NULL,
    is_prototype_fusion BOOLEAN NOT NULL DEFAULT TRUE,
    model_status JSONB NOT NULL DEFAULT '{}',
    signal_status JSONB NOT NULL DEFAULT '{}',
    features JSONB NOT NULL DEFAULT '{}',
    explanations JSONB NOT NULL DEFAULT '{}',
    evidence_ids JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_assessments_address ON risk_assessments (target_address);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_chain ON risk_assessments (chain);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_created_at ON risk_assessments (created_at DESC);

COMMENT ON TABLE risk_assessments IS 'CHAKRA Step 5 — Fused risk score results.';
