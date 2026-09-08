"""Chain and network identifiers for the CHAKRA canonical transaction model.

Defines:
- Chain: the top-level blockchain family (TRON, EVM, BITCOIN, SOLANA)
- Network: specific network/mainnet/testnet within a chain family
- ChainInfo: structured chain metadata bundling chain, network, and optional EVM chain_id

Design notes:
- EVM chains (Ethereum, BNB Smart Chain, Polygon, etc.) all share Chain.EVM;
  the actual network is identified by `network` and `chain_id`.
- Non-EVM chains do not use chain_id (it remains None).
- Address normalization is chain-aware: callers must not blindly lowercase
  addresses without knowing the target chain.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Chain(str, Enum):
    """Top-level blockchain family identifier."""

    TRON = "tron"
    EVM = "evm"
    BITCOIN = "bitcoin"
    SOLANA = "solana"


class Network(str, Enum):
    """Specific network within a chain family."""

    TRON_MAINNET = "tron-mainnet"
    TRON_SHASTA = "tron-shasta"
    ETH_MAINNET = "ethereum-mainnet"
    ETH_SEPOLIA = "ethereum-sepolia"
    BNB_MAINNET = "bsc-mainnet"
    POLYGON_MAINNET = "polygon-mainnet"
    BTC_MAINNET = "bitcoin-mainnet"
    BTC_TESTNET = "bitcoin-testnet"
    SOL_MAINNET = "solana-mainnet"
    SOL_DEVNET = "solana-devnet"
    UNKNOWN = "unknown"


EVM_CHAIN_ID_TO_NETWORK: dict[int, Network] = {
    1: Network.ETH_MAINNET,
    56: Network.BNB_MAINNET,
    137: Network.POLYGON_MAINNET,
    11155111: Network.ETH_SEPOLIA,
}

NETWORK_TO_EVM_CHAIN_ID: dict[Network, int] = {
    net: cid for cid, net in EVM_CHAIN_ID_TO_NETWORK.items()
}


class ChainInfo(BaseModel):
    """Structured chain metadata for a transaction or transfer."""

    chain: Chain
    network: Network
    chain_id: Optional[int] = None

    model_config = {"frozen": True}
