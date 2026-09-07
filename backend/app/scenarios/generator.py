import abc
from typing import List

from app.schemas.transaction import Transaction


class ScenarioGenerator(abc.ABC):
    """Base protocol for deterministic scenario generators.
    
    Architectural invariant:
        A generator with identical configuration produces the same ordered
        transaction sequence on repeated invocation.
        
    A scenario generator generates input data (canonical Transaction objects)
    and does NOT perform forensic analysis.
    """

    @property
    @abc.abstractmethod
    def scenario_id(self) -> str:
        """Stable, machine-friendly identifier for the scenario."""
        pass

    @abc.abstractmethod
    def generate(self) -> List[Transaction]:
        """Generate a deterministic list of canonical transactions.
        
        Architectural invariant:
            Repeated invocations of generate() with identical configuration
            must return the exact same ordered transaction sequence.
            Do NOT rely on Python hash ordering, wall-clock time,
            random UUIDs, or database ordering.
        """
        pass

