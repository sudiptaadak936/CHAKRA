"""Application mode resolution service.

Provides a clean, authoritative interface for querying the current operational mode
(LIVE vs DEMO) without reading environment variables directly across modules
or introducing mutable global state.

Architectural principle:
- LIVE: normal CHAKRA operation consuming real/external blockchain providers.
- DEMO: synthetic scenario data may be selected as the input source.
- Demo mode changes the data source, NOT the forensic semantics.
- No separate demo forensic pipeline is instantiated.
"""
from __future__ import annotations

from typing import Optional

from app.core.config import AppMode, Mode, Settings, settings as default_settings


class ModeService:
    """Canonical resolver for CHAKRA operational mode."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Return the bound Settings instance or fall back to default global settings."""
        return self._settings if self._settings is not None else default_settings

    def get_mode(self) -> AppMode:
        """Return the current operational mode (AppMode.LIVE or AppMode.DEMO)."""
        return self.settings.mode

    def is_live(self) -> bool:
        """Return True if the system is configured in LIVE mode."""
        return self.settings.is_live

    def is_demo(self) -> bool:
        """Return True if the system is configured in DEMO mode."""
        return self.settings.is_demo


def get_mode_service(settings: Optional[Settings] = None) -> ModeService:
    """Dependency / factory helper for ModeService.

    When settings is provided, returns an isolated ModeService bound to those settings.
    When settings is None, returns a ModeService bound to the default global settings.
    """
    return ModeService(settings=settings)


__all__ = ["AppMode", "Mode", "ModeService", "get_mode_service"]
