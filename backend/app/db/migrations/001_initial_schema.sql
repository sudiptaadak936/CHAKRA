-- 001_initial_schema.sql
-- Initial schema for Canonical CHAKRA Transactions and Transfers

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    chain VARCHAR(50) NOT NULL,
    network VARCHAR(50) NOT NULL,
    chain_id INTEGER,
    block_number BIGINT,
    block_hash VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE,
    from_address TEXT,
    to_address TEXT,
    native_value NUMERIC(78, 0) NOT NULL DEFAULT 0,
    native_value_unit VARCHAR(50) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    fee NUMERIC(78, 0),
    fee_asset VARCHAR(255),
    
    CONSTRAINT uq_transaction_identity UNIQUE (transaction_id, chain, network)
);

CREATE INDEX IF NOT EXISTS idx_transactions_txid ON transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_chain_network ON transactions(chain, network);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_from_address ON transactions(from_address) WHERE from_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transactions_to_address ON transactions(to_address) WHERE to_address IS NOT NULL;


CREATE TABLE IF NOT EXISTS transfers (
    id BIGSERIAL PRIMARY KEY,
    transaction_pk BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    from_address TEXT,
    to_address TEXT,
    asset_type VARCHAR(50) NOT NULL,
    asset_symbol VARCHAR(50) NOT NULL,
    asset_contract VARCHAR(255),
    amount NUMERIC(78, 0) NOT NULL CHECK (amount >= 0),
    amount_unit VARCHAR(50) NOT NULL,
    decimals INTEGER,
    chain VARCHAR(50) NOT NULL,
    network VARCHAR(50) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transfers_transaction_pk ON transfers(transaction_pk);
CREATE INDEX IF NOT EXISTS idx_transfers_from_address ON transfers(from_address) WHERE from_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transfers_to_address ON transfers(to_address) WHERE to_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transfers_asset_contract ON transfers(asset_contract) WHERE asset_contract IS NOT NULL;


CREATE TABLE IF NOT EXISTS transaction_provenance (
    id BIGSERIAL PRIMARY KEY,
    transaction_pk BIGINT NOT NULL UNIQUE REFERENCES transactions(id) ON DELETE CASCADE,
    provider VARCHAR(100) NOT NULL,
    chain VARCHAR(50) NOT NULL,
    network VARCHAR(50) NOT NULL,
    original_id VARCHAR(255) NOT NULL,
    normalized_at TIMESTAMP WITH TIME ZONE NOT NULL
);


CREATE TABLE IF NOT EXISTS bitcoin_transaction_details (
    id BIGSERIAL PRIMARY KEY,
    transaction_pk BIGINT NOT NULL UNIQUE REFERENCES transactions(id) ON DELETE CASCADE,
    txid VARCHAR(255) NOT NULL,
    total_input_sat NUMERIC(78, 0),
    total_output_sat NUMERIC(78, 0) NOT NULL,
    fee_sat NUMERIC(78, 0),
    locktime BIGINT,
    version INTEGER
);


CREATE TABLE IF NOT EXISTS bitcoin_vins (
    id BIGSERIAL PRIMARY KEY,
    detail_pk BIGINT NOT NULL REFERENCES bitcoin_transaction_details(id) ON DELETE CASCADE,
    txid VARCHAR(255),
    vout INTEGER,
    address TEXT,
    value_sat NUMERIC(78, 0),
    coinbase BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_bitcoin_vins_detail_pk ON bitcoin_vins(detail_pk);
CREATE INDEX IF NOT EXISTS idx_bitcoin_vins_address ON bitcoin_vins(address) WHERE address IS NOT NULL;


CREATE TABLE IF NOT EXISTS bitcoin_vouts (
    id BIGSERIAL PRIMARY KEY,
    detail_pk BIGINT NOT NULL REFERENCES bitcoin_transaction_details(id) ON DELETE CASCADE,
    n INTEGER NOT NULL,
    address TEXT,
    value_sat NUMERIC(78, 0) NOT NULL CHECK (value_sat >= 0),
    script_type VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_bitcoin_vouts_detail_pk ON bitcoin_vouts(detail_pk);
CREATE INDEX IF NOT EXISTS idx_bitcoin_vouts_address ON bitcoin_vouts(address) WHERE address IS NOT NULL;
