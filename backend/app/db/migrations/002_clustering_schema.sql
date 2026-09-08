-- 002_clustering_schema.sql
-- Step 2C: Derived analytical clustering state.
-- These tables are DERIVED and ANALYTICAL only.
-- They do NOT represent canonical transaction or transfer data.
-- Canonical tables (transactions, transfers, bitcoin_*) are NOT modified.

-- Clustering Evidence: one row per deterministic co-input address pair.
-- evidence_type = 'bitcoin_co_input' is the only supported type in Step 2C.
-- evidence_status = 'observed_heuristic' signals this is an observed blockchain
-- interaction, NOT ownership proof.
CREATE TABLE IF NOT EXISTS clustering_evidence (
    id BIGSERIAL PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    chain TEXT NOT NULL,
    network TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    address_a_composite_id TEXT NOT NULL,
    address_b_composite_id TEXT NOT NULL,
    address_a_normalized TEXT NOT NULL,
    address_b_normalized TEXT NOT NULL,

    CONSTRAINT uq_clustering_evidence_id UNIQUE (evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_clustering_evidence_chain_network
    ON clustering_evidence(chain, network);
CREATE INDEX IF NOT EXISTS idx_clustering_evidence_txid
    ON clustering_evidence(transaction_id);
CREATE INDEX IF NOT EXISTS idx_clustering_evidence_addr_a
    ON clustering_evidence(address_a_composite_id);
CREATE INDEX IF NOT EXISTS idx_clustering_evidence_addr_b
    ON clustering_evidence(address_b_composite_id);


-- Address Clusters: one row per derived cluster.
-- cluster_id is a deterministic SHA-256 digest of the sorted canonical member set.
-- representative_composite_id is the lexicographically smallest member.
CREATE TABLE IF NOT EXISTS address_clusters (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    representative_composite_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    network TEXT NOT NULL,

    CONSTRAINT uq_address_cluster_id UNIQUE (cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_address_clusters_chain_network
    ON address_clusters(chain, network);
CREATE INDEX IF NOT EXISTS idx_address_clusters_representative
    ON address_clusters(representative_composite_id);


-- Cluster Members: one row per address per cluster.
CREATE TABLE IF NOT EXISTS cluster_members (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    composite_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    network TEXT NOT NULL,
    normalized_address TEXT NOT NULL,

    CONSTRAINT uq_cluster_member UNIQUE (cluster_id, composite_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id
    ON cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_composite_id
    ON cluster_members(composite_id);
