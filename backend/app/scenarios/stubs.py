from typing import List

from app.scenarios.generator import ScenarioGenerator
from app.schemas.transaction import Transaction


class OneHopCashoutGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "one_hop_cashout"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario one_hop_cashout is not yet implemented")


class PeelChainGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "peel_chain"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario peel_chain is not yet implemented")


class FanInGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "fan_in"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario fan_in is not yet implemented")


class MixerInteractionGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "mixer_interaction"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario mixer_interaction is not yet implemented")


class CrossChainHopGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "cross_chain_hop"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario cross_chain_hop is not yet implemented")


class OffshoreCashoutGenerator(ScenarioGenerator):
    @property
    def scenario_id(self) -> str:
        return "offshore_cashout"

    def generate(self) -> List[Transaction]:
        raise NotImplementedError("Scenario offshore_cashout is not yet implemented")
