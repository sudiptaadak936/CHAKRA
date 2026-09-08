-- 003_change_detection_schema.sql
-- Step 3D: Derived analytical change-address inferences.
-- These tables are DERIVED and ANALYTICAL only.
-- Canonical tables are NOT modified.

CREATE TABLE IF NOT EXISTS bitcoin_change_inferences (
    id BIGSERIAL PRIMARY KEY,
    txid VARCHAR(255) NOT NULL,
    output_index INTEGER NOT NULL,
    address TEXT NOT NULL,
    classification VARCHAR(50) NOT NULL, -- CHANGE_CANDIDATE, EXTERNAL_RECIPIENT, UNKNOWN
    confidence VARCHAR(50) NOT NULL, -- HIGH, LOW, UNKNOWN
    evidence_reason TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_bitcoin_change_inference UNIQUE (txid, output_index)
);

CREATE INDEX IF NOT EXISTS idx_bitcoin_change_inferences_txid
    ON bitcoin_change_inferences(txid);
CREATE INDEX IF NOT EXISTS idx_bitcoin_change_inferences_address
    ON bitcoin_change_inferences(address);
