"""Canonical CHAKRA transaction and transfer models.

Architecture:
    Blockchain API
        |
    Provider Client
        |
    Chain-specific Adapter
        |
    Transaction / Transfer  (this module)
        |
    Future: Persistence -> Graph / Risk layers

Key design decisions:
1. Monetary values are always INTEGERS in atomic units:
   - Bitcoin:    satoshis (1 BTC  = 100,000,000 sat)
   - Ethereum:   wei      (1 ETH  = 1e18 wei)
   - Tron TRX:  sun      (1 TRX  = 1,000,000 sun)
   - Solana:    lamports (1 SOL  = 1,000,000,000 lamports)
   - ERC-20/TRC-20/SPL: token base units (decimals stored separately)
   Never store floating-point for monetary values.

2. A transaction is a container (Q1):
   A blockchain transaction may contain zero, one, or many asset movements.
   The `Transaction` model uses `native_value` and `native_value_unit` to
   represent the top-level native coin movement (0 if none).
   The `transfers` list is the AUTHORITATIVE source for all asset movements.

3. Bitcoin UTXO model (Q2):
   Bitcoin transactions use inputs and outputs, not a single sender/receiver.
   We do NOT create artificial 1-to-1 sender->receiver mappings.
   BitcoinVin/BitcoinVout preserve full UTXO semantics (available in BitcoinTransactionDetail).
   When producing canonical `Transfer` objects for Bitcoin:
     - Output transfers will have from_address=None.
     - Input transfers will have to_address=None.
   This prevents downstream systems from counting the same movement twice or
   inferring fake relationships.

4. Address preservation:
   Addresses are stored exactly as received from the provider.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.chain import Chain, Network


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AssetType(str, Enum):
    NATIVE = "native"
    TOKEN = "token"

class TransactionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    UNKNOWN = "unknown"

class TransactionType(str, Enum):
    TRANSFER = "transfer"
    CONTRACT_CALL = "contract_call"
    CONTRACT_CREATION = "contract_creation"
    COINBASE = "coinbase"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

class Transfer(BaseModel):
    """One normalized asset movement.

    Authoritative representation for asset movements.
    For account-based chains, from_address and to_address are typically both present.
    For UTXO chains (Bitcoin):
      - output movements have from_address=None
      - input movements have to_address=None
    """

    from_address: Optional[str] = None
    to_address: Optional[str] = None
    asset_type: AssetType
    asset_symbol: str = Field(..., min_length=1, max_length=20)
    asset_contract: Optional[str] = None
    amount: int = Field(..., ge=0)
    amount_unit: str = Field(..., min_length=1)
    decimals: Optional[int] = Field(None, ge=0, le=77)
    chain: Chain
    network: Network

    @field_validator("asset_contract")
    @classmethod
    def contract_required_for_token(cls, v: Optional[str], info) -> Optional[str]:
        return v

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# TransactionProvenance
# ---------------------------------------------------------------------------

class TransactionProvenance(BaseModel):
    provider: str = Field(..., min_length=1)
    chain: Chain
    network: Network
    original_id: str = Field(..., min_length=1)
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Canonical CHAKRA blockchain transaction.

    A container for zero, one, or many asset movements.
    """

    transaction_id: str = Field(..., min_length=1)
    chain: Chain
    network: Network
    chain_id: Optional[int] = None

    block_number: Optional[int] = Field(None, ge=0)
    block_hash: Optional[str] = None
    timestamp: Optional[datetime] = None

    # Primary participants (Optional; usually None for UTXO chains)
    from_address: Optional[str] = None
    to_address: Optional[str] = None

    # Top-level native movement (0 if pure token tx)
    native_value: int = Field(0, ge=0)
    native_value_unit: str = Field(..., min_length=1)

    transaction_type: TransactionType = TransactionType.UNKNOWN
    status: TransactionStatus = TransactionStatus.UNKNOWN

    fee: Optional[int] = Field(None, ge=0)
    fee_asset: Optional[str] = None

    # Authoritative list of all asset movements
    transfers: List[Transfer] = Field(default_factory=list)
    provenance: TransactionProvenance

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Bitcoin UTXO extensions
# ---------------------------------------------------------------------------

class BitcoinVin(BaseModel):
    txid: Optional[str] = None
    vout: Optional[int] = None
    address: Optional[str] = None
    value_sat: Optional[int] = Field(None, ge=0)
    coinbase: bool = False

    model_config = {"frozen": True}

class BitcoinVout(BaseModel):
    n: int = Field(..., ge=0)
    address: Optional[str] = None
    value_sat: int = Field(..., ge=0)
    script_type: Optional[str] = None

    model_config = {"frozen": True}

class BitcoinTransactionDetail(BaseModel):
    txid: str
    inputs: List[BitcoinVin]
    outputs: List[BitcoinVout]
    total_input_sat: Optional[int] = Field(None, ge=0)
    total_output_sat: int = Field(..., ge=0)
    fee_sat: Optional[int] = Field(None, ge=0)
    locktime: Optional[int] = None
    version: Optional[int] = None

    model_config = {"frozen": True}
