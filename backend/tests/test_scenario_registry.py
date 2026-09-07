import pytest
from typing import List

from app.scenarios.generator import ScenarioGenerator
from app.scenarios.registry import (
    DuplicateScenarioError,
    InvalidScenarioIdError,
    ScenarioNotFoundError,
    ScenarioRegistry,
    scenario_registry,
)
from app.schemas.transaction import Transaction


class DummyGenerator(ScenarioGenerator):
    def __init__(self, sid: str):
        self._sid = sid

    @property
    def scenario_id(self) -> str:
        return self._sid

    def generate(self) -> List[Transaction]:
        return []


def test_scenario_registry_singleton_has_required_stubs():
    # Phase C verification: Ensure the 6 required scenarios are registered
    expected_ids = {
        "one_hop_cashout",
        "peel_chain",
        "fan_in",
        "mixer_interaction",
        "cross_chain_hop",
        "offshore_cashout",
    }
    registered = set(scenario_registry.list())
    assert expected_ids.issubset(registered)


def test_scenario_registry_stub_not_implemented():
    # Verify that calling generate() on a stub raises NotImplementedError
    gen = scenario_registry.get("one_hop_cashout")
    with pytest.raises(NotImplementedError):
        gen.generate()


def test_scenario_registry_register_and_get():
    registry = ScenarioRegistry()
    gen = DummyGenerator("test_scenario")
    registry.register(gen)
    
    assert registry.contains("test_scenario")
    assert registry.get("test_scenario") is gen


def test_scenario_registry_duplicate_registration():
    registry = ScenarioRegistry()
    gen = DummyGenerator("duplicate_me")
    registry.register(gen)
    
    with pytest.raises(DuplicateScenarioError):
        registry.register(DummyGenerator("duplicate_me"))


@pytest.mark.parametrize("invalid_id", [
    "UpperCase",
    "has-hyphen",
    "has space",
    "123!@#",
    "",
])
def test_scenario_registry_invalid_id(invalid_id):
    registry = ScenarioRegistry()
    with pytest.raises(InvalidScenarioIdError):
        registry.register(DummyGenerator(invalid_id))


def test_scenario_registry_not_found():
    registry = ScenarioRegistry()
    with pytest.raises(ScenarioNotFoundError):
        registry.get("does_not_exist")


def test_scenario_registry_list_deterministic_order():
    registry = ScenarioRegistry()
    registry.register(DummyGenerator("zeta"))
    registry.register(DummyGenerator("alpha"))
    registry.register(DummyGenerator("beta"))
    
    # list() should be deterministically sorted
    assert registry.list() == ["alpha", "beta", "zeta"]
