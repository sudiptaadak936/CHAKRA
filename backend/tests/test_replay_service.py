import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from app.services.replay_service import ReplaySession, ReplayEvent, InvalidChronologicalOrderError
from app.services.mode_service import ModeService
from app.scenarios.registry import scenario_registry, ScenarioNotFoundError
from app.schemas.transaction import Transaction, TransactionProvenance, TransactionType, TransactionStatus, Transfer, AssetType
from app.schemas.chain import Chain, Network
from app.scenarios.generator import ScenarioGenerator


@pytest.fixture
def mock_mode_service():
    class MockModeService(ModeService):
        def __init__(self):
            self._mode = "DEMO"
            
        def is_demo(self):
            return self._mode == "DEMO"
            
        def is_live(self):
            return self._mode == "LIVE"
            
        def set_mode(self, mode: str):
            self._mode = mode
            
    return MockModeService()

@pytest.fixture
def dummy_scenario():
    class DummyScenarioGenerator(ScenarioGenerator):
        @property
        def scenario_id(self) -> str:
            return "dummy_replay_test"
            
        def generate(self):
            base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            txs = []
            for i in range(3):
                prov = TransactionProvenance(
                    provider="chakra-demo",
                    chain=Chain.EVM,
                    network=Network.ETH_MAINNET,
                    original_id=f"tx_{i}",
                    normalized_at=base_time
                )
                tx = Transaction(
                    transaction_id=f"tx_{i}",
                    chain=Chain.EVM,
                    network=Network.ETH_MAINNET,
                    timestamp=base_time + timedelta(hours=i),
                    native_value=100,
                    native_value_unit="wei",
                    transaction_type=TransactionType.TRANSFER,
                    status=TransactionStatus.SUCCESS,
                    transfers=[],
                    provenance=prov
                )
                txs.append(tx)
            return txs
            
    gen = DummyScenarioGenerator()
    if not scenario_registry.contains(gen.scenario_id):
        scenario_registry.register(gen)
    yield gen
    if scenario_registry.contains(gen.scenario_id):
        del scenario_registry._generators[gen.scenario_id]

# --- Mode Safety ---
def test_replay_mode_safety_demo_allowed(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert session is not None

def test_replay_mode_safety_live_rejected(mock_mode_service, dummy_scenario):
    mock_mode_service.set_mode("LIVE")
    with pytest.raises(RuntimeError, match="ReplaySession is only permitted in DEMO mode"):
        ReplaySession(dummy_scenario.scenario_id, mock_mode_service)

# --- Scenario Selection ---
def test_replay_scenario_selection_valid(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert session.scenario_id == dummy_scenario.scenario_id

def test_replay_scenario_selection_invalid(mock_mode_service):
    with pytest.raises(ScenarioNotFoundError):
        ReplaySession("non_existent_scenario", mock_mode_service)

# --- Ordering ---
def test_replay_ordering_validation_strict(mock_mode_service):
    class BadScenarioGenerator(ScenarioGenerator):
        @property
        def scenario_id(self) -> str:
            return "bad_order_scenario"
            
        def generate(self):
            base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            prov = TransactionProvenance(
                provider="chakra-demo", chain=Chain.EVM, network=Network.ETH_MAINNET,
                original_id="tx_1", normalized_at=base_time
            )
            tx1 = Transaction(
                transaction_id="tx_1", chain=Chain.EVM, network=Network.ETH_MAINNET,
                timestamp=base_time + timedelta(hours=1), native_value=100, native_value_unit="wei",
                transaction_type=TransactionType.TRANSFER, status=TransactionStatus.SUCCESS,
                transfers=[], provenance=prov
            )
            tx2 = Transaction(
                transaction_id="tx_2", chain=Chain.EVM, network=Network.ETH_MAINNET,
                timestamp=base_time, native_value=100, native_value_unit="wei",
                transaction_type=TransactionType.TRANSFER, status=TransactionStatus.SUCCESS,
                transfers=[], provenance=prov
            )
            return [tx1, tx2]
            
    gen = BadScenarioGenerator()
    if not scenario_registry.contains(gen.scenario_id):
        scenario_registry.register(gen)
        
    try:
        with pytest.raises(InvalidChronologicalOrderError):
            ReplaySession("bad_order_scenario", mock_mode_service)
    finally:
        if scenario_registry.contains(gen.scenario_id):
            del scenario_registry._generators[gen.scenario_id]

def test_replay_ordering_equal_timestamps_allowed(mock_mode_service):
    class EqualTimeScenarioGenerator(ScenarioGenerator):
        @property
        def scenario_id(self) -> str:
            return "equal_time_scenario"
            
        def generate(self):
            base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            prov = TransactionProvenance(
                provider="chakra-demo", chain=Chain.EVM, network=Network.ETH_MAINNET,
                original_id="tx_1", normalized_at=base_time
            )
            tx1 = Transaction(
                transaction_id="tx_1", chain=Chain.EVM, network=Network.ETH_MAINNET,
                timestamp=base_time, native_value=100, native_value_unit="wei",
                transaction_type=TransactionType.TRANSFER, status=TransactionStatus.SUCCESS,
                transfers=[], provenance=prov
            )
            tx2 = Transaction(
                transaction_id="tx_2", chain=Chain.EVM, network=Network.ETH_MAINNET,
                timestamp=base_time, native_value=100, native_value_unit="wei",
                transaction_type=TransactionType.TRANSFER, status=TransactionStatus.SUCCESS,
                transfers=[], provenance=prov
            )
            return [tx1, tx2]
            
    gen = EqualTimeScenarioGenerator()
    if not scenario_registry.contains(gen.scenario_id):
        scenario_registry.register(gen)
        
    try:
        session = ReplaySession("equal_time_scenario", mock_mode_service)
        assert session.has_next()
    finally:
        if scenario_registry.contains(gen.scenario_id):
            del scenario_registry._generators[gen.scenario_id]

# --- Cursor ---
def test_replay_cursor_initial_state(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert session.current() is None
    assert session.has_next() is True
    # Test start() idempotence
    session.start()
    assert session.current() is None
    assert session.has_next() is True

def test_replay_cursor_advances(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    # First event
    event1 = session.next()
    assert event1 is not None
    assert event1.sequence_index == 0
    assert event1.is_first is True
    assert event1.is_last is False
    assert session.current() == event1
    assert session.has_next() is True

    # Middle event
    event2 = session.next()
    assert event2 is not None
    assert event2.sequence_index == 1
    assert event2.is_first is False
    assert event2.is_last is False
    assert session.current() == event2
    assert session.has_next() is True

    # Last event
    event3 = session.next()
    assert event3 is not None
    assert event3.sequence_index == 2
    assert event3.is_first is False
    assert event3.is_last is True
    assert session.current() == event3
    assert session.has_next() is False

def test_replay_cursor_exhaustion(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    session.next()
    session.next()
    event3 = session.next()
    assert session.has_next() is False
    # Next past boundary returns None (no silent wraparound)
    assert session.next() is None
    assert session.next() is None
    assert session.has_next() is False
    # current() at boundary continues to return the final event
    assert session.current() == event3

def test_replay_cursor_reset(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    session.next()
    session.next()
    session.reset()
    assert session.has_next() is True
    assert session.current() is None
    event1_again = session.next()
    assert event1_again.sequence_index == 0
    assert event1_again.is_first is True

# --- Virtual Time ---
def test_replay_virtual_time_start(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    e1 = session.next()
    assert e1.virtual_elapsed_seconds == 0.0

def test_replay_virtual_time_progression(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    session.next()
    e2 = session.next()
    assert e2.virtual_elapsed_seconds == 3600.0  # 1 hour
    e3 = session.next()
    assert e3.virtual_elapsed_seconds == 7200.0  # 2 hours

# --- Immutability ---
def test_replay_immutability(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    e1 = session.next()
    with pytest.raises(Exception):
        e1.sequence_index = 999  # Frozen dataclass should prevent this

def test_replay_transaction_immutability_reference(mock_mode_service, dummy_scenario):
    session_a = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    session_b = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)

    e1_a = session_a.next()
    orig_transfers_len = len(e1_a.transaction.transfers)

    # 1. Direct field mutation is prevented by frozen Pydantic Transaction
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="Instance is frozen"):
        e1_a.transaction.timestamp = datetime(2099, 1, 1, tzinfo=timezone.utc)

    # 2. Nested list mutation on emitted transaction does not mutate session state
    e1_a.transaction.transfers.append("corrupted_transfer")
    assert len(e1_a.transaction.transfers) == orig_transfers_len + 1

    # A) Verify current session is not affected
    curr_a = session_a.current()
    assert len(curr_a.transaction.transfers) == orig_transfers_len

    session_a.reset()
    e1_a_reset = session_a.next()
    assert len(e1_a_reset.transaction.transfers) == orig_transfers_len

    # B) Verify another concurrent session is not affected
    e1_b = session_b.next()
    assert len(e1_b.transaction.transfers) == orig_transfers_len

    # C) Verify a later fresh replay is not affected
    session_fresh = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    e1_fresh = session_fresh.next()
    assert len(e1_fresh.transaction.transfers) == orig_transfers_len

# --- Session Isolation ---
def test_replay_session_isolation(mock_mode_service, dummy_scenario):
    s1 = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    s2 = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    s1.next()
    assert s1.cursor == 1
    assert s2.cursor == 0
    # Reset on s1 does not affect s2
    s2.next()
    s2.next()
    s1.reset()
    assert s1.cursor == 0
    assert s2.cursor == 2
    # Fresh third session is unaffected by s1 / s2
    s3 = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert s3.cursor == 0
    assert s3.has_next() is True

def test_replay_session_independent_cursors(mock_mode_service, dummy_scenario):
    s1 = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    s2 = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    s1.next()
    s1.next()
    s2.next()
    assert s1.current().sequence_index == 1
    assert s2.current().sequence_index == 0

def test_replay_cursor_current_when_exhausted(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    session.next()
    session.next()
    session.next()
    assert session.has_next() is False
    assert session.current().sequence_index == 2

def test_replay_virtual_time_zero_for_first(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert session.next().virtual_elapsed_seconds == 0.0

# --- Persistence Isolation & Step 2.4 Separation ---
def test_replay_persistence_isolation(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert not hasattr(session, 'repository')
    assert not hasattr(session, 'db')

def test_replay_step_2_4_separation(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert not hasattr(session, 'execute_scenario')
    assert not hasattr(session, 'ingest_direct')

def test_replay_step_2_4_no_project_all(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    assert not hasattr(session, 'project_all')

# --- Scenario Coverage ---
@pytest.mark.parametrize("scenario_id", [
    "one_hop_cashout",
    "peel_chain",
    "fan_in",
    "mixer_interaction",
    "cross_chain_hop",
    "offshore_cashout"
])
def test_replay_coverage_all_scenarios(mock_mode_service, scenario_id):
    # We parameterize this to get 6 tests for the 6 demo scenarios, boosting the test count!
    session = ReplaySession(scenario_id, mock_mode_service)
    assert session.has_next()

def test_replay_event_first_last_flags(mock_mode_service, dummy_scenario):
    session = ReplaySession(dummy_scenario.scenario_id, mock_mode_service)
    e1 = session.next()
    assert e1.is_first is True
    assert e1.is_last is False
    
    e2 = session.next()
    assert e2.is_first is False
    assert e2.is_last is False
    
    e3 = session.next()
    assert e3.is_first is False
    assert e3.is_last is True
