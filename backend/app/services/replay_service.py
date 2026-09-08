import copy
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from app.schemas.transaction import Transaction
from app.scenarios.registry import scenario_registry
from app.services.mode_service import ModeService


@dataclass(frozen=True)
class ReplayEvent:
    scenario_id: str
    sequence_index: int
    transaction_id: str
    event_timestamp: datetime
    virtual_elapsed_seconds: float
    transaction: Transaction
    is_first: bool
    is_last: bool


class InvalidChronologicalOrderError(ValueError):
    """Raised when scenario transactions are not strictly chronologically ordered."""
    pass


class ReplaySession:
    """Virtual chronological replay session for demo scenarios.
    
    Provides a deterministic, step-by-step presentation view of a scenario's
    canonical transactions without mutating any underlying datastores.
    
    Cursor Contract:
    - Initial State:
      * cursor = 0
      * has_next() is True (for non-empty scenarios)
      * current() is None (no event emitted yet)
    - start() / reset():
      * Sets cursor = 0
      * current() returns None
      * has_next() returns True (for non-empty scenarios)
    - next():
      * If has_next() is False, returns None (no silent wraparound).
      * Otherwise, retrieves transaction at cursor, emits a ReplayEvent with
        an isolated, deep-copied Transaction, advances cursor by 1, and returns the event.
    - current():
      * If cursor == 0, returns None.
      * If cursor > 0, returns the event corresponding to the most recently
        emitted transaction (cursor - 1) with an isolated, deep-copied Transaction.
      * At final/end state (has_next() is False), current() returns the final event.
    - End State:
      * cursor = len(transactions)
      * has_next() is False
      * next() returns None
      * current() returns the final event (is_last = True)
    """

    def __init__(self, scenario_id: str, mode_service: ModeService):
        if not mode_service.is_demo():
            raise RuntimeError("ReplaySession is only permitted in DEMO mode.")
        
        self.scenario_id = scenario_id
        generator = scenario_registry.get(scenario_id)
        # Store an isolated deep copy to prevent mutation of generator state
        self.transactions = copy.deepcopy(generator.generate())
        
        # Validate chronological ordering
        for i in range(1, len(self.transactions)):
            if self.transactions[i].timestamp < self.transactions[i-1].timestamp:
                raise InvalidChronologicalOrderError(
                    f"Transactions out of order at index {i}"
                )
                
        self.cursor = 0

    def start(self) -> None:
        """Initializes or restarts the replay session, placing cursor at index 0."""
        self.cursor = 0

    def has_next(self) -> bool:
        """Returns True if there are more events to replay."""
        return self.cursor < len(self.transactions)

    def next(self) -> Optional[ReplayEvent]:
        """Advances the cursor and returns the next ReplayEvent.
        
        Returns None when exhausted (no silent wraparound).
        Emits an isolated deep copy of the transaction to guarantee immutability.
        """
        if not self.has_next():
            return None
            
        idx = self.cursor
        tx = copy.deepcopy(self.transactions[idx])
        is_first = (idx == 0)
        is_last = (idx == len(self.transactions) - 1)
        
        first_tx_timestamp = self.transactions[0].timestamp
        virtual_elapsed_seconds = (tx.timestamp - first_tx_timestamp).total_seconds()
        
        event = ReplayEvent(
            scenario_id=self.scenario_id,
            sequence_index=idx,
            transaction_id=tx.transaction_id,
            event_timestamp=tx.timestamp,
            virtual_elapsed_seconds=virtual_elapsed_seconds,
            transaction=tx,
            is_first=is_first,
            is_last=is_last
        )
        self.cursor += 1
        return event

    def current(self) -> Optional[ReplayEvent]:
        """Returns the most recently emitted ReplayEvent without advancing the cursor.
        
        Returns None if no event has been emitted yet (cursor == 0).
        Emits an isolated deep copy of the transaction to guarantee immutability.
        """
        if self.cursor == 0 or self.cursor > len(self.transactions):
            return None
            
        idx = self.cursor - 1
        tx = copy.deepcopy(self.transactions[idx])
        is_first = (idx == 0)
        is_last = (idx == len(self.transactions) - 1)
        
        first_tx_timestamp = self.transactions[0].timestamp
        virtual_elapsed_seconds = (tx.timestamp - first_tx_timestamp).total_seconds()
        
        event = ReplayEvent(
            scenario_id=self.scenario_id,
            sequence_index=idx,
            transaction_id=tx.transaction_id,
            event_timestamp=tx.timestamp,
            virtual_elapsed_seconds=virtual_elapsed_seconds,
            transaction=tx,
            is_first=is_first,
            is_last=is_last
        )
        return event

    def reset(self) -> None:
        """Resets the cursor to the initial deterministic state (index 0)."""
        self.cursor = 0

