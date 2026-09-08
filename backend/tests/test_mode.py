"""Step 2.1 Tests — Live/Demo Mode Infrastructure.

Validates:
- Test 1: Default mode -> LIVE
- Test 2: Explicit LIVE -> resolves to LIVE
- Test 3: Explicit DEMO -> resolves to DEMO
- Test 4: Case normalization -> 'demo', 'Demo', 'DEMO', 'live', 'Live', 'LIVE' resolve consistently
- Test 5: Invalid mode -> explicit ValidationError, never silently becomes DEMO
- Test 6: Deterministic resolution -> repeated resolutions produce identical results
- Test 7: No pipeline mutation -> single forensic pipeline preserved, no duplicate demo services
- ModeService queries (get_mode, is_live, is_demo) and dependency injection
- Environment variable overrides (MODE, CHAKRA_MODE)
"""
import pytest
from pydantic import ValidationError

from app.core.config import AppMode, Mode, Settings
from app.services.mode_service import ModeService, get_mode_service


def test_1_default_mode():
    """Test 1 — Default mode: No explicit mode configured must resolve to LIVE."""
    s = Settings(
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s.MODE == AppMode.LIVE
    assert s.mode == AppMode.LIVE
    assert s.is_live is True
    assert s.is_demo is False

    service = ModeService(settings=s)
    assert service.get_mode() == AppMode.LIVE
    assert service.is_live() is True
    assert service.is_demo() is False


def test_2_explicit_live():
    """Test 2 — Explicit LIVE: Configured mode LIVE resolves to LIVE."""
    s = Settings(
        MODE="LIVE",
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s.MODE == AppMode.LIVE
    assert s.mode == AppMode.LIVE
    assert s.is_live is True
    assert s.is_demo is False

    service = ModeService(settings=s)
    assert service.get_mode() == AppMode.LIVE
    assert service.is_live() is True
    assert service.is_demo() is False


def test_3_explicit_demo():
    """Test 3 — Explicit DEMO: Configured mode DEMO resolves to DEMO."""
    s = Settings(
        MODE="DEMO",
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s.MODE == AppMode.DEMO
    assert s.mode == AppMode.DEMO
    assert s.is_live is False
    assert s.is_demo is True

    service = ModeService(settings=s)
    assert service.get_mode() == AppMode.DEMO
    assert service.is_live() is False
    assert service.is_demo() is True


@pytest.mark.parametrize(
    "token,expected",
    [
        ("demo", AppMode.DEMO),
        ("Demo", AppMode.DEMO),
        ("DEMO", AppMode.DEMO),
        ("  demo  ", AppMode.DEMO),
        ("live", AppMode.LIVE),
        ("Live", AppMode.LIVE),
        ("LIVE", AppMode.LIVE),
        ("  LIVE\t", AppMode.LIVE),
    ],
)
def test_4_case_normalization(token, expected):
    """Test 4 — Case normalization: Case and whitespace variants parse consistently."""
    s = Settings(
        MODE=token,
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s.mode == expected
    service = ModeService(settings=s)
    assert service.get_mode() == expected


@pytest.mark.parametrize(
    "invalid_token",
    [
        "banana",
        "STAGING",
        "PROD",
        "TEST",
        "demomode",
        "livemode",
        "",
        "   ",
        "1",
        "false",
    ],
)
def test_5_invalid_mode_fails_explicitly(invalid_token):
    """Test 5 — Invalid mode: Invalid tokens must fail explicitly and NEVER resolve to DEMO."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            MODE=invalid_token,
            POSTGRES_DB="test_db",
            POSTGRES_USER="test_user",
            POSTGRES_PASSWORD="test_password",
        )

    err_str = str(exc_info.value)
    assert "Invalid application mode" in err_str or "Valid modes are" in err_str


def test_6_deterministic_resolution():
    """Test 6 — Deterministic resolution: Repeated resolution produces identical results."""
    s_live = Settings(
        MODE=AppMode.LIVE,
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    s_demo = Settings(
        MODE=AppMode.DEMO,
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )

    srv_live = ModeService(settings=s_live)
    srv_demo = ModeService(settings=s_demo)

    for _ in range(50):
        assert srv_live.get_mode() == AppMode.LIVE
        assert srv_live.is_live() is True
        assert srv_live.is_demo() is False

        assert srv_demo.get_mode() == AppMode.DEMO
        assert srv_demo.is_live() is False
        assert srv_demo.is_demo() is True


def test_7_no_pipeline_mutation():
    """Test 7 — No pipeline mutation: Demo mode does not instantiate or introduce separate forensic pipelines.

    Verifies the architectural invariant:
    DEMO MODE CHANGES THE DATA SOURCE, NOT THE FORENSIC SEMANTICS.
    No parallel DemoIngestionService, DemoGraphProjector, DemoTraversal,
    DemoClusterer, DemoPathScorer, or DemoTypologyDetector exist.
    """
    import sys

    app_modules = [m for m in sys.modules if m.startswith("app.")]

    # Ensure no duplicate demo services/projectors/traversals exist in codebase
    forbidden_demo_classes = [
        "DemoIngestionService",
        "DemoGraphProjector",
        "DemoTraversal",
        "DemoClusterer",
        "DemoPathScorer",
        "DemoTypologyDetector",
    ]

    for mod_name in app_modules:
        mod = sys.modules[mod_name]
        for forbidden in forbidden_demo_classes:
            assert not hasattr(mod, forbidden), f"Found forbidden duplicate pipeline class {forbidden} in {mod_name}"

    # Confirm canonical ingestion service exists as the single pipeline
    from app.services.ingestion_service import IngestionService
    assert IngestionService is not None

    # Confirm canonical forensics components exist as single pipeline
    from app.forensics.beam_search import WeightedBeamSearch
    from app.forensics.path_scorer import PathScorer
    from app.forensics.typology_detector import TypologyDetector
    from app.graph.projector import GraphProjector
    from app.graph.traversal import MoneyFlowTraversal
    assert WeightedBeamSearch is not None
    assert PathScorer is not None
    assert TypologyDetector is not None
    assert GraphProjector is not None
    assert MoneyFlowTraversal is not None


def test_env_var_alias_resolution(monkeypatch):
    """Test environment variable aliases: MODE and CHAKRA_MODE."""
    monkeypatch.setenv("CHAKRA_DISABLE_ENV_FILE", "1")

    # Test MODE env var
    monkeypatch.setenv("MODE", "demo")
    s = Settings(
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s.mode == AppMode.DEMO

    # Test CHAKRA_MODE env var
    monkeypatch.delenv("MODE", raising=False)
    monkeypatch.setenv("CHAKRA_MODE", "live")
    s2 = Settings(
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    assert s2.mode == AppMode.LIVE

    # Test invalid env var fails
    monkeypatch.setenv("CHAKRA_MODE", "invalid_mode")
    with pytest.raises(ValidationError):
        Settings(
            POSTGRES_DB="test_db",
            POSTGRES_USER="test_user",
            POSTGRES_PASSWORD="test_password",
        )


def test_mode_service_factory():
    """Test get_mode_service factory and dependency injection."""
    default_srv = get_mode_service()
    assert isinstance(default_srv, ModeService)

    custom_settings = Settings(
        MODE="DEMO",
        POSTGRES_DB="test_db",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
    )
    custom_srv = get_mode_service(settings=custom_settings)
    assert custom_srv.is_demo() is True
    assert custom_srv.get_mode() == AppMode.DEMO

    # Mode alias equivalence
    assert Mode is AppMode
    assert Mode.LIVE == AppMode.LIVE
    assert Mode.DEMO == AppMode.DEMO
