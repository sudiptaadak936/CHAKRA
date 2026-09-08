from app.scenarios.generator import ScenarioGenerator
from app.scenarios.registry import (
    DuplicateScenarioError,
    InvalidScenarioIdError,
    ScenarioNotFoundError,
    ScenarioRegistry,
    scenario_registry,
)
from app.scenarios.demo import (
    CrossChainHopGenerator,
    FanInGenerator,
    MixerInteractionGenerator,
    OffshoreCashoutGenerator,
    OneHopCashoutGenerator,
    PeelChainGenerator,
)

# Register the deterministic scenario stubs
scenario_registry.register(OneHopCashoutGenerator())
scenario_registry.register(PeelChainGenerator())
scenario_registry.register(FanInGenerator())
scenario_registry.register(MixerInteractionGenerator())
scenario_registry.register(CrossChainHopGenerator())
scenario_registry.register(OffshoreCashoutGenerator())

__all__ = [
    "ScenarioGenerator",
    "ScenarioRegistry",
    "scenario_registry",
    "DuplicateScenarioError",
    "InvalidScenarioIdError",
    "ScenarioNotFoundError",
]
