"""Schemas for blockchain data ingestion results and statuses."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.chain import Chain, Network
from app.schemas.transaction import Transaction


class IngestionStatus(str, Enum):
    SUCCESS = "success"
    ALREADY_EXISTS = "already_exists"
    FAILED = "failed"


class IngestionResult(BaseModel):
    """Result of a single transaction ingestion attempt."""

    success: bool
    status: IngestionStatus
    chain: Chain
    network: Network
    transaction_id: str
    provider: Optional[str] = None
    fallback_used: bool = False
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    transaction: Optional[Transaction] = None

    model_config = {"frozen": True}


class AddressHistoryIngestionResult(BaseModel):
    """Result of an address transaction history ingestion attempt."""

    success: bool
    chain: Chain
    network: Network
    address: str
    provider: Optional[str] = None
    fallback_used: bool = False
    transactions_discovered: int = 0
    transactions_persisted: int = 0
    transactions_already_existed: int = 0
    results: List[IngestionResult] = Field(default_factory=list)
    error_category: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"frozen": True}
