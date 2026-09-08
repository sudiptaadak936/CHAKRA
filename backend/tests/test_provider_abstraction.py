"""Tests for the DataProvider abstraction and registry."""

import pytest
from app.schemas.chain import Chain
from app.providers.data_base import (
    BaseDataProvider,
    ProviderCapability,
    ProviderError,
    ProviderAuthError,
    UnsupportedOperationError,
)
from app.providers.data_registry import DataProviderRegistry


class DummyAdapter(BaseDataProvider):
    @property
    def provider_name(self) -> str:
        return "dummy"
    @property
    def supported_chains(self) -> list[Chain]:
        return [Chain.EVM]
    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.TRANSACTION_LOOKUP}
    async def get_transaction(self, tx_id: str, **kwargs):
        pass


def test_registry_registration():
    registry = DataProviderRegistry()
    adapter = DummyAdapter()
    registry.register(Chain.EVM, adapter)
    
    assert registry.get_primary(Chain.EVM) is adapter
    assert registry.get_all(Chain.EVM) == [adapter]
    assert registry.get_by_name("dummy") is adapter

def test_registry_unsupported_chain():
    registry = DataProviderRegistry()
    adapter = DummyAdapter()
    with pytest.raises(ValueError, match="does not support chain"):
        registry.register(Chain.TRON, adapter)

def test_registry_no_primary():
    registry = DataProviderRegistry()
    with pytest.raises(UnsupportedOperationError):
        registry.get_primary(Chain.EVM)

def test_error_hierarchy():
    err = ProviderAuthError("dummy", "Bad key")
    assert isinstance(err, ProviderError)
    assert err.provider == "dummy"
    assert "dummy" in str(err)
    assert "Bad key" in str(err)

def test_unsupported_operation_default():
    adapter = DummyAdapter()
    assert adapter.supports(ProviderCapability.TRANSACTION_LOOKUP) is True
    assert adapter.supports(ProviderCapability.ADDRESS_HISTORY) is False
