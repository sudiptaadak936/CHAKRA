"""Data provider registry for blockchain ingestion providers.

This registry is separate from the Step 0.5 health-check provider registry
(providers/registry.py). Its purpose is:
- Register chain-specific data adapters (BaseDataProvider subclasses)
- Look up the appropriate provider for a given chain
- Support provider fallback chains per blockchain family

Health-check providers and ingestion providers have different lifecycles
and must not be conflated.

ChainAbuse is NOT registered here. It is an intelligence provider,
not a blockchain data source.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.providers.data_base import BaseDataProvider, UnsupportedOperationError
from app.schemas.chain import Chain

logger = logging.getLogger(__name__)


class DataProviderRegistry:
    """Registry mapping blockchain families to ordered lists of data providers.

    Providers are ordered by preference (primary first, fallback(s) last).
    The registry does not automatically fall back; callers decide fallback logic.

    Example usage:
        registry = DataProviderRegistry()
        registry.register(Chain.EVM, evm_adapter)
        registry.register(Chain.TRON, tron_adapter)

        evm_provider = registry.get_primary(Chain.EVM)
        all_evm = registry.get_all(Chain.EVM)
    """

    def __init__(self) -> None:
        self._providers: Dict[Chain, List[BaseDataProvider]] = {
            chain: [] for chain in Chain
        }

    def register(self, chain: Chain, provider: BaseDataProvider, primary: bool = False) -> None:
        """Register a data provider for a chain.

        Args:
            chain:    The blockchain family this provider serves.
            provider: A BaseDataProvider instance.
            primary:  If True, insert at the front of the list (becomes primary).
                      If False, append to the end (becomes last fallback).
        """
        if not isinstance(provider, BaseDataProvider):
            raise TypeError(
                f"Provider must be a BaseDataProvider instance, got {type(provider).__name__}"
            )
        if chain not in provider.supported_chains:
            raise ValueError(
                f"Provider {provider.provider_name!r} does not support chain {chain.value!r}. "
                f"Supported chains: {[c.value for c in provider.supported_chains]}"
            )
        if primary:
            self._providers[chain].insert(0, provider)
        else:
            self._providers[chain].append(provider)
        logger.debug(
            "Registered provider %r for chain %r (primary=%s)",
            provider.provider_name,
            chain.value,
            primary,
        )

    def get_primary(self, chain: Chain) -> BaseDataProvider:
        """Return the primary (first-registered or most-preferred) provider for a chain.

        Raises:
            UnsupportedOperationError: No provider registered for the chain.
        """
        providers = self._providers.get(chain, [])
        if not providers:
            raise UnsupportedOperationError(
                "data-registry",
                f"No provider registered for chain {chain.value!r}",
            )
        return providers[0]

    def get_all(self, chain: Chain) -> List[BaseDataProvider]:
        """Return all registered providers for a chain (primary first).

        Returns an empty list if no providers are registered.
        """
        return list(self._providers.get(chain, []))

    def get_by_name(self, provider_name: str) -> Optional[BaseDataProvider]:
        """Return the first provider matching the given provider_name, or None."""
        for providers in self._providers.values():
            for provider in providers:
                if provider.provider_name == provider_name:
                    return provider
        return None

    def registered_chains(self) -> List[Chain]:
        """Return chains that have at least one provider registered."""
        return [chain for chain, providers in self._providers.items() if providers]

    def summary(self) -> Dict[str, List[str]]:
        """Return a dict mapping chain name -> list of provider names (for diagnostics)."""
        return {
            chain.value: [p.provider_name for p in providers]
            for chain, providers in self._providers.items()
            if providers
        }


def get_default_data_provider_registry() -> DataProviderRegistry:
    """Construct and return a DataProviderRegistry with default configured adapters.

    Providers:
    - EVM: EVMTransactionAdapter (primary), BlockscoutEVMAdapter (fallback)
    - TRON: TronTransactionAdapter (primary)
    - BITCOIN: BitcoinTransactionAdapter (primary), BlockstreamBitcoinAdapter (fallback)
    - SOLANA: HeliusSolanaAdapter (primary), PublicSolanaRPCAdapter (fallback)
    """
    from app.core.config import settings
    from app.providers.adapters.evm import EVMTransactionAdapter, BlockscoutEVMAdapter
    from app.providers.adapters.tron import TronTransactionAdapter
    from app.providers.adapters.bitcoin import BitcoinTransactionAdapter, BlockstreamBitcoinAdapter
    from app.providers.adapters.solana import HeliusSolanaAdapter, PublicSolanaRPCAdapter

    registry = DataProviderRegistry()

    # EVM: Etherscan (primary) -> Blockscout (fallback)
    registry.register(Chain.EVM, EVMTransactionAdapter(api_key=settings.ETHERSCAN_API_KEY), primary=True)
    registry.register(Chain.EVM, BlockscoutEVMAdapter(), primary=False)

    # TRON: TronGrid (primary)
    registry.register(Chain.TRON, TronTransactionAdapter(api_key=settings.TRONGRID_API_KEY), primary=True)

    # BITCOIN: mempool.space (primary) -> Blockstream Esplora (fallback)
    registry.register(Chain.BITCOIN, BitcoinTransactionAdapter(), primary=True)
    registry.register(Chain.BITCOIN, BlockstreamBitcoinAdapter(), primary=False)

    # SOLANA: Helius (primary) -> public RPC (fallback)
    registry.register(Chain.SOLANA, HeliusSolanaAdapter(api_key=settings.HELIUS_API_KEY), primary=True)
    registry.register(Chain.SOLANA, PublicSolanaRPCAdapter(), primary=False)

    return registry

