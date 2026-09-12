-- 004_vasp_attribution_schema.sql
-- Schema for VASP Attribution Registry (CHAKRA Step 4.1)

CREATE TABLE IF NOT EXISTS vasp_attribution_registry (
    id BIGSERIAL PRIMARY KEY,
    address TEXT NOT NULL,
    chain VARCHAR(50) NOT NULL,
    vasp_name VARCHAR(255) NOT NULL,
    service_type VARCHAR(100) NOT NULL,
    source VARCHAR(255) NOT NULL,
    evidence_id VARCHAR(255) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    CONSTRAINT uq_vasp_attribution UNIQUE (address, chain, vasp_name, source)
);

CREATE INDEX IF NOT EXISTS idx_vasp_attribution_lookup ON vasp_attribution_registry(chain, address);
