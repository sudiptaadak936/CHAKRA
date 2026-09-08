"""CHAKRA Step 3D: Forensics evidence models.

These models represent DERIVED ANALYTICAL STATE and forensic inferences.
They do NOT represent canonical blockchain transactions.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ChangeAddressInference(BaseModel):
    """Forensic inference regarding whether a Bitcoin output is a change address.
    
    This is an inference based on deterministic heuristics.
    It does NOT prove common ownership automatically.
    """

    txid: str = Field(..., description="The transaction ID")
    output_index: int = Field(..., description="The index of the output in the transaction (vout.n)")
    address: str = Field(..., description="The candidate output address")
    classification: str = Field(..., description="CHANGE_CANDIDATE, EXTERNAL_RECIPIENT, or UNKNOWN")
    confidence: str = Field(..., description="HIGH, LOW, or UNKNOWN")
    evidence_reason: str = Field(..., description="Explanation of the deterministic signals used")

    model_config = {"frozen": True}
