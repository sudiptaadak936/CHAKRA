"""CHAKRA Step 3D: Bitcoin Change-Address Detection.

Deterministic detection of Bitcoin change outputs using heuristics.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Set

from app.forensics.models import ChangeAddressInference
from app.forensics.repository import ForensicsRepository

logger = logging.getLogger(__name__)


class BitcoinChangeDetector:
    """Service for inferring Bitcoin change outputs deterministically."""

    def __init__(self, repository: ForensicsRepository):
        self.repository = repository

    async def detect_transaction(self, txid: str) -> List[ChangeAddressInference]:
        """Detect change addresses for a given transaction."""
        
        vins = await self.repository.get_transaction_vins(txid)
        vouts = await self.repository.get_transaction_vouts(txid)

        if not vouts:
            return []

        input_addrs: Set[str] = {vin["address"] for vin in vins}
        inferences: List[ChangeAddressInference] = []

        # Case: Single output transaction
        if len(vouts) == 1:
            vout = vouts[0]
            if vout["address"] in input_addrs:
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=vout["address"],
                        classification="CHANGE_CANDIDATE",
                        confidence="HIGH",
                        evidence_reason="Single output address matches input address (self-sweep / consolidation)"
                    )
                )
            else:
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=vout["address"],
                        classification="UNKNOWN",
                        confidence="UNKNOWN",
                        evidence_reason="Single-output transaction without distinct change output; ambiguous between external payment and internal sweep"
                    )
                )
            return inferences

        # Case: Multiple outputs
        fresh_vouts: List[Dict[str, Any]] = []

        for vout in vouts:
            addr = vout["address"]
            
            # Heuristic 1: Self-Change (strongest)
            if addr in input_addrs:
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=addr,
                        classification="CHANGE_CANDIDATE",
                        confidence="HIGH",
                        evidence_reason="Address matches an input address (self-change)"
                    )
                )
                continue

            # Heuristic 2: Reuse Elimination
            # If the address appears as an output in >1 distinct transactions, it's reused.
            tx_count = await self.repository.get_address_output_tx_count(addr)
            
            if tx_count > 1:
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=addr,
                        classification="EXTERNAL_RECIPIENT",
                        confidence="LOW",
                        evidence_reason="Observed address reuse across multiple transactions (heuristic signal of recipient behavior; does not prove external ownership)"
                    )
                )
                continue

            # If it's not self-change and not reused, it is a one-time fresh output
            fresh_vouts.append(vout)

        # Heuristic 3: One-time change via elimination resolution
        has_self_change = any(i.classification == "CHANGE_CANDIDATE" for i in inferences)
        
        if len(fresh_vouts) == 1:
            vout = fresh_vouts[0]
            if has_self_change:
                # If there's already a self-change output, the remaining fresh output is likely the payment recipient.
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=vout["address"],
                        classification="UNKNOWN",
                        confidence="UNKNOWN",
                        evidence_reason="Remaining fresh output in a transaction that already has a self-change output"
                    )
                )
            else:
                # No self-change, and all other outputs are known external recipients.
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=vout["address"],
                        classification="CHANGE_CANDIDATE",
                        confidence="LOW",
                        evidence_reason="Only newly observed address among multiple outputs with reused counterparts (inferred change candidate via reuse elimination; heuristic inference)"
                    )
                )
        elif len(fresh_vouts) > 1:
            # Ambiguous: multiple fresh outputs
            for vout in fresh_vouts:
                inferences.append(
                    ChangeAddressInference(
                        txid=txid,
                        output_index=vout["output_index"],
                        address=vout["address"],
                        classification="UNKNOWN",
                        confidence="UNKNOWN",
                        evidence_reason="Ambiguous transaction with multiple fresh output addresses"
                    )
                )

        # Sort inferences by output_index for deterministic ordering
        inferences.sort(key=lambda x: x.output_index)
        
        return inferences

    async def detect_and_save(self, txid: str) -> List[ChangeAddressInference]:
        """Detect change addresses and save them to the database."""
        inferences = await self.detect_transaction(txid)
        if inferences:
            await self.repository.upsert_change_inferences(inferences)
        return inferences
