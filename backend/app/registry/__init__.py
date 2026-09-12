"""CHAKRA Step 6: Suspect Registry package.

Public API for the Suspect Registry:
  - SuspectRegistryRepository  — PostgreSQL persistence layer
  - SuspectRegistryService     — Domain logic and cross-case evaluation
"""
from app.registry.repository import SuspectRegistryRepository
from app.registry.service import SuspectRegistryService

__all__ = [
    "SuspectRegistryRepository",
    "SuspectRegistryService",
]
