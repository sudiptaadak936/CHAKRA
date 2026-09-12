-- Migration 005: Alerts and Audit Schema
-- CHAKRA Step 4.5A LEA Alerting Layer — persistent storage
-- No private data. All alert_ids are deterministic SHA-256 hashes.
-- Human review fields are immutable once written.

-- ===================================================================
-- TABLE: alerts
-- Persistent store for all generated alert records.
-- alert_id is a deterministic SHA-256 deduplication key.
-- requires_human_review is always TRUE — never set to false.
-- ===================================================================
CREATE TABLE IF NOT EXISTS alerts (
    id                   BIGSERIAL PRIMARY KEY,
    alert_id             TEXT NOT NULL UNIQUE,       -- SHA-256 dedup key
    alert_type           TEXT NOT NULL,              -- AlertType enum value
    chain                TEXT NOT NULL,              -- Chain enum value
    address              TEXT NOT NULL,
    severity             TEXT NOT NULL,              -- AlertSeverity enum value
    reason               TEXT NOT NULL,
    evidence_ids         JSONB NOT NULL DEFAULT '[]',
    source               TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- INVARIANT: requires_human_review is always TRUE. Never set to FALSE.
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE
        CONSTRAINT alerts_human_review_immutable CHECK (requires_human_review = TRUE),
    status               TEXT NOT NULL DEFAULT 'PENDING_REVIEW'
        CONSTRAINT alerts_status_valid
            CHECK (status IN ('PENDING_REVIEW', 'REVIEWED', 'DISMISSED'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_alert_type   ON alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_chain        ON alerts (chain);
CREATE INDEX IF NOT EXISTS idx_alerts_address      ON alerts (address);
CREATE INDEX IF NOT EXISTS idx_alerts_status       ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at   ON alerts (created_at DESC);

-- ===================================================================
-- TABLE: alert_audit_log
-- Append-only audit trail for all alert state transitions.
-- Never deletes rows. actor_id is human investigator identifier.
-- No automated actors may write to this table.
-- ===================================================================
CREATE TABLE IF NOT EXISTS alert_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    alert_id        TEXT NOT NULL REFERENCES alerts(alert_id) ON DELETE RESTRICT,
    from_status     TEXT,                           -- NULL for creation events
    to_status       TEXT NOT NULL,
    action          TEXT NOT NULL,                  -- e.g. CREATED, REVIEWED, DISMISSED
    actor_id        TEXT,                           -- human investigator identifier (nullable for system)
    reason          TEXT,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_alert_audit_alert_id    ON alert_audit_log (alert_id);
CREATE INDEX IF NOT EXISTS idx_alert_audit_occurred_at ON alert_audit_log (occurred_at DESC);

-- ===================================================================
-- TABLE: escalation_audit_log
-- Append-only record of SAHYOG-style escalation packet preparations.
-- No external submissions are recorded here (there are none).
-- ===================================================================
CREATE TABLE IF NOT EXISTS escalation_audit_log (
    id                      BIGSERIAL PRIMARY KEY,
    chain                   TEXT NOT NULL,
    address                 TEXT NOT NULL,
    attributed_vasp         TEXT,
    attribution_confidence  TEXT NOT NULL,
    branch_decision         TEXT NOT NULL,
    escalation_status       TEXT NOT NULL,
    registration_status     TEXT NOT NULL DEFAULT 'UNKNOWN',
    recommended_action      TEXT,
    evidence_ids            JSONB NOT NULL DEFAULT '[]',
    reason                  TEXT NOT NULL,
    -- INVARIANT: no_network_request is always TRUE — no SAHYOG API was called
    no_network_request      BOOLEAN NOT NULL DEFAULT TRUE
        CONSTRAINT escalation_no_network_immutable CHECK (no_network_request = TRUE),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_escalation_audit_address ON escalation_audit_log (address);
CREATE INDEX IF NOT EXISTS idx_escalation_audit_chain   ON escalation_audit_log (chain);
CREATE INDEX IF NOT EXISTS idx_escalation_audit_date    ON escalation_audit_log (created_at DESC);

COMMENT ON TABLE alerts IS
    'CHAKRA Step 4.5A — alert records. requires_human_review is always TRUE.';
COMMENT ON TABLE alert_audit_log IS
    'CHAKRA Step 4.5A — append-only audit trail for alert state transitions.';
COMMENT ON TABLE escalation_audit_log IS
    'CHAKRA Step 4H — audit trail for SAHYOG-style escalation preparations. no_network_request always TRUE.';
