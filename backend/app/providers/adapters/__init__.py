"""Chain-specific transaction adapters.

Each adapter implements BaseDataProvider for one blockchain family,
translating provider API responses into canonical Transaction objects.

Adapters:
    EVMTransactionAdapter   -- Ethereum-compatible chains via Etherscan V2 (primary)
                               or Blockscout (fallback stub)
    TronTransactionAdapter  -- Tron network via TronGrid
    BitcoinTransactionAdapter -- Bitcoin via mempool.space / Blockstream Esplora (stub)
    SolanaTransactionAdapter  -- Solana via Helius (primary) or public RPC (fallback stub)

ChainAbuse is NOT in this package. It is an intelligence provider, not a
blockchain data source, and must not be used in the transaction ingestion pipeline.
"""
