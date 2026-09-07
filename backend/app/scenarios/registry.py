import re
from typing import Dict, List

from app.scenarios.generator import ScenarioGenerator


class InvalidScenarioIdError(ValueError):
    """Raised when a scenario ID does not meet format requirements."""
    pass


class DuplicateScenarioError(ValueError):
    """Raised when attempting to register a scenario ID that is already registered."""
    pass


class ScenarioNotFoundError(KeyError):
    """Raised when a requested scenario ID is not found in the registry."""
    pass


class ScenarioRegistry:
    """Registry for deterministic scenario generators."""

    def __init__(self):
        self._generators: Dict[str, ScenarioGenerator] = {}

    def register(self, generator: ScenarioGenerator) -> None:
        """Registers a scenario generator.
        
        Args:
            generator: The ScenarioGenerator instance to register.
            
        Raises:
            InvalidScenarioIdError: If the scenario_id is malformed.
            DuplicateScenarioError: If the scenario_id is already registered.
        """
        sid = generator.scenario_id
        if not sid or not re.match(r"^[a-z0-9_]+$", sid):
            raise InvalidScenarioIdError(
                f"Invalid scenario ID format: '{sid}'. Must be lowercase alphanumeric and underscores."
            )
            
        if sid in self._generators:
            raise DuplicateScenarioError(f"Scenario '{sid}' is already registered.")
            
        self._generators[sid] = generator

    def get(self, scenario_id: str) -> ScenarioGenerator:
        """Retrieves a scenario generator by ID.
        
        Args:
            scenario_id: The stable identifier for the scenario.
            
        Returns:
            The registered ScenarioGenerator instance.
            
        Raises:
            ScenarioNotFoundError: If the scenario is not found.
        """
        if scenario_id not in self._generators:
            raise ScenarioNotFoundError(f"Scenario '{scenario_id}' not found.")
        return self._generators[scenario_id]

    def list(self) -> List[str]:
        """Returns a deterministic, sorted list of registered scenario IDs."""
        return sorted(list(self._generators.keys()))

    def contains(self, scenario_id: str) -> bool:
        """Checks if a scenario ID is registered."""
        return scenario_id in self._generators

# Global singleton registry
scenario_registry = ScenarioRegistry()
