"""Strict contract tests for CHAKRA Step 2.2: Scenario Framework + Registry."""
from datetime import datetime, timezone
from typing import Any, List
import pytest

from app.scenarios.generator import ScenarioGenerator
from app.scenarios.registry import (
    DuplicateScenarioError,
    InvalidScenarioIdError,
    ScenarioNotFoundError,
    ScenarioRegistry,
    scenario_registry,
)
from app.scenarios.stubs import (
    CrossChainHopGenerator,
    FanInGenerator,
    MixerInteractionGenerator,
    OffshoreCashoutGenerator,
    OneHopCashoutGenerator,
    PeelChainGenerator,
)
from app.schemas.chain import Chain, Network
from app.schemas.transaction import (
    AssetType,
    Transaction,
    TransactionProvenance,
    TransactionStatus,
    TransactionType,
    Transfer,
)


class DummyGenerator(ScenarioGenerator):
    def __init__(self, sid: Any):
        self._sid = sid

    @property
    def scenario_id(self) -> str:
        return self._sid

    def generate(self) -> List[Transaction]:
        return []


class ConcreteValidTransactionGenerator(ScenarioGenerator):
    """Test generator that returns real, valid canonical Transaction instances."""
    @property
    def scenario_id(self) -> str:
        return "concrete_valid_test"

    def generate(self) -> List[Transaction]:
        prov = TransactionProvenance(
            provider="test-provider",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="0xtest123",
            normalized_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        transfer = Transfer(
            from_address="0xSender",
            to_address="0xReceiver",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        tx = Transaction(
            transaction_id="tx_test_001",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            native_value=1000,
            native_value_unit="wei",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[transfer],
            provenance=prov,
        )
        return [tx]


# ---------------------------------------------------------------------------
# Section 2 & 3: Generator Contract & Determinism
# ---------------------------------------------------------------------------

def test_scenario_generator_cannot_be_instantiated_directly():
    """ScenarioGenerator is an abstract base class; direct instantiation must fail."""
    with pytest.raises(TypeError):
        ScenarioGenerator()


def test_scenario_generator_subclass_missing_scenario_id():
    """Omitting scenario_id must prevent instantiation."""
    class IncompleteGen(ScenarioGenerator):
        def generate(self) -> List[Transaction]:
            return []

    with pytest.raises(TypeError):
        IncompleteGen()


def test_scenario_generator_subclass_missing_generate():
    """Omitting generate() must prevent instantiation."""
    class IncompleteGen(ScenarioGenerator):
        @property
        def scenario_id(self) -> str:
            return "test_missing_generate"

    with pytest.raises(TypeError):
        IncompleteGen()


def test_scenario_generator_determinism_contract():
    """A generator must produce the same ordered transaction sequence on repeated invocations."""
    gen = ConcreteValidTransactionGenerator()
    run_1 = gen.generate()
    run_2 = gen.generate()

    assert len(run_1) == len(run_2)
    for tx1, tx2 in zip(run_1, run_2):
        assert tx1 == tx2
        assert tx1.transaction_id == tx2.transaction_id


# ---------------------------------------------------------------------------
# Section 4: Transaction Schema Return Compatibility
# ---------------------------------------------------------------------------

def test_scenario_generator_returns_valid_transaction_schema():
    gen = ConcreteValidTransactionGenerator()
    txs = gen.generate()

    assert isinstance(txs, list)
    assert len(txs) == 1
    tx = txs[0]
    assert isinstance(tx, Transaction)
    assert tx.transaction_id == "tx_test_001"
    assert tx.chain == Chain.EVM
    assert len(tx.transfers) == 1
    assert tx.transfers[0].amount == 1000


# ---------------------------------------------------------------------------
# Section 5 & 6: The Six Canonical Scenarios & Stubs
# ---------------------------------------------------------------------------

CANONICAL_SCENARIOS = [
    ("one_hop_cashout", OneHopCashoutGenerator),
    ("peel_chain", PeelChainGenerator),
    ("fan_in", FanInGenerator),
    ("mixer_interaction", MixerInteractionGenerator),
    ("cross_chain_hop", CrossChainHopGenerator),
    ("offshore_cashout", OffshoreCashoutGenerator),
]


def test_canonical_scenarios_exact_set_and_classes():
    """Verify the exact six target scenario identifiers and generator classes."""
    expected_ids = [s[0] for s in CANONICAL_SCENARIOS]
    assert scenario_registry.list() == sorted(expected_ids)

    for scenario_id, cls in CANONICAL_SCENARIOS:
        assert issubclass(cls, ScenarioGenerator)
        gen = scenario_registry.get(scenario_id)
        assert isinstance(gen, cls)
        assert gen.scenario_id == scenario_id


@pytest.mark.parametrize("scenario_id,cls", CANONICAL_SCENARIOS)
def test_each_stub_raises_not_implemented_error(scenario_id, cls):
    """Calling generate() on any of the 6 canonical stubs must raise NotImplementedError."""
    gen = scenario_registry.get(scenario_id)
    with pytest.raises(NotImplementedError) as exc_info:
        gen.generate()
    assert scenario_id in str(exc_info.value)


# ---------------------------------------------------------------------------
# Section 7 & 8: Registry Contract, ID Validation, and Errors
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    registry = ScenarioRegistry()
    gen = DummyGenerator("test_scenario")
    registry.register(gen)

    assert registry.contains("test_scenario")
    assert registry.get("test_scenario") is gen


def test_registry_duplicate_registration_same_instance():
    registry = ScenarioRegistry()
    gen = DummyGenerator("duplicate_id")
    registry.register(gen)

    with pytest.raises(DuplicateScenarioError):
        registry.register(gen)


def test_registry_duplicate_registration_different_instance_same_id():
    registry = ScenarioRegistry()
    gen1 = DummyGenerator("duplicate_id")
    gen2 = DummyGenerator("duplicate_id")
    registry.register(gen1)

    with pytest.raises(DuplicateScenarioError):
        registry.register(gen2)


def test_registry_unknown_id_raises_not_found():
    registry = ScenarioRegistry()
    with pytest.raises(ScenarioNotFoundError):
        registry.get("unknown_scenario")


def test_registry_contains_true_and_false():
    registry = ScenarioRegistry()
    registry.register(DummyGenerator("present_id"))

    assert registry.contains("present_id") is True
    assert registry.contains("absent_id") is False


def test_registry_list_deterministic_lexicographic_order():
    registry = ScenarioRegistry()
    registry.register(DummyGenerator("zeta"))
    registry.register(DummyGenerator("alpha"))
    registry.register(DummyGenerator("beta"))
    registry.register(DummyGenerator("gamma"))

    assert registry.list() == ["alpha", "beta", "gamma", "zeta"]


@pytest.mark.parametrize("valid_id", [
    "a",
    "abc",
    "one_hop_cashout",
    "peel_chain",
    "scenario123",
    "test_123_valid",
])
def test_registry_valid_id_format(valid_id):
    registry = ScenarioRegistry()
    gen = DummyGenerator(valid_id)
    registry.register(gen)
    assert registry.contains(valid_id)


@pytest.mark.parametrize("invalid_id", [
    "",
    " ",
    "UPPERCASE",
    "hyphen-name",
    "contains space",
    "contains.dot",
    "contains/slash",
    "123!@#",
])
def test_registry_invalid_string_id_rejected(invalid_id):
    registry = ScenarioRegistry()
    with pytest.raises(InvalidScenarioIdError):
        registry.register(DummyGenerator(invalid_id))


@pytest.mark.parametrize("non_string_id", [
    None,
    123,
    12.34,
    ["list"],
    {"dict": "val"},
])
def test_registry_non_string_id_rejected(non_string_id):
    registry = ScenarioRegistry()
    with pytest.raises(InvalidScenarioIdError):
        registry.register(DummyGenerator(non_string_id))


# ---------------------------------------------------------------------------
# Section 9 & 10: Registry State Isolation & Clean Imports
# ---------------------------------------------------------------------------

def test_registry_instance_isolation():
    """Independent ScenarioRegistry instances must not share state."""
    reg1 = ScenarioRegistry()
    reg2 = ScenarioRegistry()

    reg1.register(DummyGenerator("custom_scenario"))
    assert reg1.contains("custom_scenario")
    assert not reg2.contains("custom_scenario")
    assert reg2.list() == []


def test_package_clean_imports():
    """Confirm package imports work cleanly without circular dependencies."""
    from app.scenarios import (
        DuplicateScenarioError as DSE,
        InvalidScenarioIdError as ISIE,
        ScenarioGenerator as SG,
        ScenarioNotFoundError as SNFE,
        ScenarioRegistry as SR,
        scenario_registry as sr,
    )
    assert SG is ScenarioGenerator
    assert SR is ScenarioRegistry
    assert sr is scenario_registry

