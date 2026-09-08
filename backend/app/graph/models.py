"""Models and identity helpers for CHAKRA Step 2A Neo4j Graph Projection.

PostgreSQL is the canonical source of truth.
Neo4j is downstream/derived state.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

# 64-bit signed integer limits for Neo4j Bolt protocol
INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807


def normalize_address(chain: str, address: Optional[str]) -> str:
    """Normalize blockchain address following canonical rules.

    - If address is None or empty, returns empty string.
    - For EVM addresses, normalizes to lowercase and strips whitespace.
    - For case-sensitive addresses (Bitcoin base58, Solana base58, Tron base58),
      preserves case as-is and strips whitespace.
    """
    if not address:
        return ""
    addr = address.strip()
    chain_lower = (chain or "").lower()
    if chain_lower == "evm" or addr.startswith("0x"):
        return addr.lower()
    return addr


def make_address_composite_id(chain: str, network: str, address: str) -> str:
    """Deterministic graph identity for Address nodes: (chain, network, normalized_address)."""
    norm = normalize_address(chain, address)
    return f"{chain.lower()}:{network.lower()}:{norm}"


def make_transaction_composite_id(chain: str, network: str, transaction_id: str) -> str:
    """Deterministic graph identity for Transaction nodes: (chain, network, transaction_id)."""
    return f"{chain.lower()}:{network.lower()}:{transaction_id.strip()}"


def make_transfer_relationship_id(
    chain: str, network: str, transaction_id: str, transfer_pk: int
) -> str:
    """Deterministic relationship identity for canonical Transfer records.

    Preserves a one-to-one correspondence with the canonical PostgreSQL Transfer record.
    """
    return f"{chain.lower()}:{network.lower()}:{transaction_id.strip()}:{transfer_pk}"


def make_bitcoin_input_id(
    network: str,
    transaction_id: str,
    vin_txid: Optional[str],
    vin_vout: Optional[int],
    vin_pk: int,
) -> str:
    """Deterministic relationship identity for Bitcoin SPENT_INPUT relationship."""
    prev_tx = (vin_txid or "coinbase").strip()
    prev_vout = vin_vout if vin_vout is not None else "cb"
    return f"bitcoin:{network.lower()}:{transaction_id.strip()}:vin:{prev_tx}_{prev_vout}_{vin_pk}"


def make_bitcoin_output_id(
    network: str, transaction_id: str, vout_n: int, vout_pk: int
) -> str:
    """Deterministic relationship identity for Bitcoin CREATED_OUTPUT relationship."""
    return f"bitcoin:{network.lower()}:{transaction_id.strip()}:vout:{vout_n}_{vout_pk}"


def to_neo4j_numeric(val: Optional[int | str]) -> tuple[Optional[int], Optional[str]]:
    """Convert an arbitrary-precision integer into exact Neo4j representations.

    Returns:
        (amount, amount_str)
        - amount: Neo4j signed 64-bit integer ONLY when representable in [-2^63, 2^63 - 1];
                  otherwise None (null in Neo4j).
        - amount_str: Always the exact canonical decimal string representation of the integer.
    """
    if val is None:
        return None, None
    int_val = int(val)
    str_val = str(int_val)
    if INT64_MIN <= int_val <= INT64_MAX:
        return int_val, str_val
    return None, str_val


class BatchProjectionResult(BaseModel):
    batch_index: int
    transactions_count: int
    transfers_count: int
    bitcoin_inputs_count: int
    bitcoin_outputs_count: int
    start_tx_id: Optional[int] = None
    end_tx_id: Optional[int] = None
    success: bool = True


class ProjectionSummary(BaseModel):
    total_batches: int = 0
    total_transactions: int = 0
    total_transfers: int = 0
    total_bitcoin_inputs: int = 0
    total_bitcoin_outputs: int = 0
    success: bool = True
    batch_results: list[BatchProjectionResult] = Field(default_factory=list)


class BatchProjectionError(Exception):
    """Raised when a projection batch fails inside its transaction boundary."""

    def __init__(
        self,
        batch_index: int,
        start_tx_id: Optional[int],
        end_tx_id: Optional[int],
        message: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            f"Batch {batch_index} (PostgreSQL tx pk range {start_tx_id}..{end_tx_id}) failed: {message}"
        )
        self.batch_index = batch_index
        self.start_tx_id = start_tx_id
        self.end_tx_id = end_tx_id
        self.original_error = original_error
