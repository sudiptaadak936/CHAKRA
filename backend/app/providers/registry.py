"""Provider registry — constructs all provider clients from settings.

The registry is the only place that reads API keys from settings.
Keys are passed to client constructors and stored privately.
They never appear in returned objects, logs, or API responses.
"""
from typing import List

from app.core.config import settings
from app.providers.base import BaseProviderClient
from app.providers.trongrid import TronGridClient
from app.providers.etherscan import EtherscanClient
from app.providers.helius import HeliusClient
from app.providers.solscan import SolscanClient
from app.providers.chainabuse import ChainAbuseClient


def get_all_providers() -> List[BaseProviderClient]:
    """Return one client instance per provider, keyed from settings."""
    from app.core.config import settings

    return [
        TronGridClient(settings.TRONGRID_API_KEY),
        EtherscanClient(settings.ETHERSCAN_API_KEY),
        HeliusClient(settings.HELIUS_API_KEY),
        SolscanClient(settings.SOLSCAN_API_KEY),
        ChainAbuseClient(settings.CHAINABUSE_API_KEY),
    ]
