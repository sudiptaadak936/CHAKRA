"""CHAKRA Step 3G: Typology Detector.

Deterministic forensic typology detection layer built upon CHAKRA's
traversal, clustering, and path-scoring infrastructure.

Strictly read-only and analytical:
- Identifies transaction flow patterns (Peel chains, Fan-in, Fan-out, Rapid hopping).
- mixer and cross-chain/bridge safety stubs return insufficient_evidence.
- Makes NO claims of criminality, intent, guilt, or ownership.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import asyncpg
from pydantic import BaseModel, Field

from app.forensics.models import ChangeAddressInference
from app.forensics.repository import ForensicsRepository
from app.graph.traversal import TraversalPath, TraversalNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable engineering defaults (NOT probabilities or legal thresholds)
# ---------------------------------------------------------------------------

MIN_PEEL_CHAIN_HOPS: int = 3
MIN_FAN_IN_SOURCES: int = 3
MIN_FAN_OUT_DESTINATIONS: int = 3
MAX_RAPID_HOP_INTERVAL_MINUTES: float = 60.0
MIN_RAPID_HOPS: int = 3
DEFAULT_FAN_WINDOW_HOURS: float = 24.0


# ---------------------------------------------------------------------------
# Typology classification models
# ---------------------------------------------------------------------------

class TypologyType(str, Enum):
    PEEL_CHAIN = "PEEL_CHAIN"
    FAN_IN = "FAN_IN"
    FAN_OUT = "FAN_OUT"
    RAPID_HOPPING = "RAPID_HOPPING"
    MIXER_INTERACTION = "MIXER_INTERACTION"
    CROSS_CHAIN = "CROSS_CHAIN"


class TypologyEvidence(BaseModel):
    evidence_type: str
    evidence_details: str
    transaction_ids: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    counterparty_addresses: List[str] = Field(default_factory=list)
    amounts: List[Dict[str, Any]] = Field(default_factory=list)
    timestamps: List[str] = Field(default_factory=list)
    temporal_window: Optional[Dict[str, Any]] = None
    peel_details: Optional[List[Dict[str, Any]]] = None


class TypologyDetection(BaseModel):
    detection_id: str
    typology_type: TypologyType
    chain: str
    confidence_level: str  # "observed", "strongly_observed", "insufficient_evidence"
    explanation: str
    evidence: List[TypologyEvidence] = Field(default_factory=list)

    # Context & provenance
    hop_count: Optional[int] = None
    transaction_ids: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    counterparty_addresses: List[str] = Field(default_factory=list)
    temporal_window: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_timestamp(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Typology Detector Service
# ---------------------------------------------------------------------------

class TypologyDetector:
    """Deterministic forensic typology detector (strictly read-only)."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        repo: ForensicsRepository = None,
        *,
        min_peel_chain_hops: int = MIN_PEEL_CHAIN_HOPS,
        min_fan_in_sources: int = MIN_FAN_IN_SOURCES,
        min_fan_out_destinations: int = MIN_FAN_OUT_DESTINATIONS,
        max_rapid_hop_interval_minutes: float = MAX_RAPID_HOP_INTERVAL_MINUTES,
        min_rapid_hops: int = MIN_RAPID_HOPS,
        max_fan_window_hours: float = DEFAULT_FAN_WINDOW_HOURS,
    ):
        self.pool = pool
        self.repo = repo or ForensicsRepository(pool)
        self.min_peel_chain_hops = min_peel_chain_hops
        self.min_fan_in_sources = min_fan_in_sources
        self.min_fan_out_destinations = min_fan_out_destinations
        self.max_rapid_hop_interval_minutes = max_rapid_hop_interval_minutes
        self.min_rapid_hops = min_rapid_hops
        self.max_fan_window_hours = max_fan_window_hours

    def _make_detection_id(self, typ: str, chain: str, elements: List[str]) -> str:
        """Create a deterministic SHA-256 detection ID from canonical inputs."""
        canonical = f"{typ}:{chain}:" + "|".join(sorted(set(str(e) for e in elements if e)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def detect_path_typologies(self, path: TraversalPath) -> List[TypologyDetection]:
        """Detect all applicable typologies on a single TraversalPath."""
        detections: List[TypologyDetection] = []

        if not path.nodes:
            return detections

        chain = path.nodes[0].chain

        # 1. Peel Chain (Bitcoin UTXO only)
        if chain.lower() == "bitcoin":
            peel_det = await self._detect_peel_chain(path)
            if peel_det:
                detections.append(peel_det)

        # 2. Rapid Hopping
        rapid_det = await self._detect_rapid_hopping(path)
        if rapid_det:
            detections.append(rapid_det)

        # 3. Fan-in (Aggregation)
        fan_in_dets = await self._detect_fan_in(path)
        detections.extend(fan_in_dets)

        # 4. Fan-out (Distribution)
        fan_out_dets = await self._detect_fan_out(path)
        detections.extend(fan_out_dets)

        # 5. Mixer Interaction (Safety default)
        mixer_det = self._detect_mixer(path)
        if mixer_det:
            detections.append(mixer_det)

        # 6. Cross-Chain / Bridge (Safety default)
        bridge_det = self._detect_cross_chain(path)
        if bridge_det:
            detections.append(bridge_det)

        # Deterministic ordering by (typology_type.value, detection_id)
        detections.sort(key=lambda d: (d.typology_type.value, d.detection_id))
        return detections

    # -----------------------------------------------------------------------
    # PEEL_CHAIN
    # -----------------------------------------------------------------------

    async def _get_bitcoin_vouts(self, txid: str) -> List[Dict[str, Any]]:
        """Fetch outputs with amounts for a Bitcoin transaction."""
        query = """
            SELECT v.n as output_index, v.address, v.value_sat
            FROM bitcoin_vouts v
            JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
            WHERE d.txid = $1 AND v.address IS NOT NULL
            ORDER BY v.n
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, txid)
            return [
                {
                    "output_index": r["output_index"],
                    "address": r["address"],
                    "value_sat": int(r["value_sat"]) if r.get("value_sat") is not None else None,
                }
                for r in rows
            ]

    async def _detect_peel_chain(self, path: TraversalPath) -> Optional[TypologyDetection]:
        """Detect genuine Bitcoin peel chains with remainder & peeled outputs.
        
        Requires:
        1. At least min_peel_chain_hops consecutive hops.
        2. Each transaction must have:
           - A continuing/remainder output (matching the path) classified as CHANGE_CANDIDATE.
           - A separate peeled/non-change output with positive amount.
        3. Continuing output must represent a meaningful remainder (retention_ratio >= 0.50).
        4. Amounts must be available and positive; missing amounts result in no detection.
        """
        if path.hops < self.min_peel_chain_hops or not path.utxo_steps:
            return None

        valid_hops: List[Dict[str, Any]] = []

        for step in path.utxo_steps:
            txid = step.transaction.transaction_id
            continuing_addr = step.created_output.address

            # 1. Step 3D change inference check
            inferences = await self.repo.get_change_inferences(txid)
            change_inf = next(
                (i for i in inferences if i.address == continuing_addr and i.classification == "CHANGE_CANDIDATE"),
                None,
            )
            if not change_inf:
                # Sequence broken
                break

            # 2. Output topology & split check
            vouts = getattr(step, "all_outputs", None)
            if vouts is None:
                vouts = await self._get_bitcoin_vouts(txid)

            if not vouts or len(vouts) < 2:
                # 1-in-1-out self-sweep or consolidation cannot be a peel step
                break

            continuing_vout = next((v for v in vouts if v["address"] == continuing_addr), None)
            peeled_vouts = [v for v in vouts if v["address"] != continuing_addr]

            if not continuing_vout or not peeled_vouts:
                break

            # 3. Amount & ratio check
            cont_sat = continuing_vout.get("value_sat")
            if cont_sat is None or int(cont_sat) <= 0:
                # Amount missing or zero: cannot establish peel relationship
                break
            cont_sat = int(cont_sat)

            peeled_amounts = [v.get("value_sat") for v in peeled_vouts]
            if any(p is None for p in peeled_amounts):
                # Missing amounts: cannot establish peel relationship
                break
            peeled_sat = sum(int(p) for p in peeled_amounts)
            if peeled_sat <= 0:
                break

            total_out = cont_sat + peeled_sat
            retention_ratio = round(cont_sat / total_out, 4)
            peel_ratio = round(peeled_sat / total_out, 4)

            # Continuing output must represent the primary remainder (retention >= 50%)
            if cont_sat < peeled_sat or retention_ratio < 0.50:
                break

            upstream_sat = None
            if getattr(step, "spent_input", None) and getattr(step.spent_input, "value_sat", None) is not None:
                upstream_sat = int(step.spent_input.value_sat)
            else:
                upstream_sat = total_out

            hop_detail = {
                "txid": txid,
                "continuing_address": continuing_addr,
                "continuing_amount_sat": cont_sat,
                "peeled_addresses": sorted(list(set(v["address"] for v in peeled_vouts))),
                "peeled_amount_sat": peeled_sat,
                "upstream_amount_sat": upstream_sat,
                "retention_ratio": retention_ratio,
                "peel_ratio": peel_ratio,
                "timestamp": getattr(step.transaction, "timestamp", None),
                "step_3d_classification": change_inf.classification,
                "step_3d_reason": change_inf.evidence_reason,
            }
            valid_hops.append(hop_detail)

        if len(valid_hops) >= self.min_peel_chain_hops:
            txids = sorted(list(set(h["txid"] for h in valid_hops)))
            cont_addrs = sorted(list(set(h["continuing_address"] for h in valid_hops)))
            peeled_addrs = sorted(list(set(a for h in valid_hops for a in h["peeled_addresses"])))
            all_addrs = sorted(list(set(cont_addrs + peeled_addrs)))

            det_id = self._make_detection_id("PEEL_CHAIN", path.nodes[0].chain, txids + cont_addrs + peeled_addrs)
            explanation = (
                f"Observed sequential peel chain pattern across {len(valid_hops)} consecutive hops. "
                f"Each transaction transfers a primary remainder forward via Step 3D change candidates "
                f"(average retention: {round(sum(h['retention_ratio'] for h in valid_hops) / len(valid_hops), 2)}) "
                "while peeling smaller non-change outputs."
            )
            evidence = [
                TypologyEvidence(
                    evidence_type="peel_chain_sequence",
                    evidence_details=explanation,
                    transaction_ids=txids,
                    addresses=cont_addrs,
                    counterparty_addresses=peeled_addrs,
                    peel_details=valid_hops,
                )
            ]
            return TypologyDetection(
                detection_id=det_id,
                typology_type=TypologyType.PEEL_CHAIN,
                chain=path.nodes[0].chain,
                confidence_level="observed",
                explanation=explanation,
                evidence=evidence,
                hop_count=len(valid_hops),
                transaction_ids=txids,
                addresses=cont_addrs,
                counterparty_addresses=peeled_addrs,
            )

        return None

    # -----------------------------------------------------------------------
    # RAPID_HOPPING
    # -----------------------------------------------------------------------

    async def _detect_rapid_hopping(self, path: TraversalPath) -> Optional[TypologyDetection]:
        """Detect rapid movement through multiple addresses within a short temporal window."""
        if path.hops < self.min_rapid_hops:
            return None

        times: List[datetime] = []
        txids: List[str] = []
        addresses = sorted(list(set(n.raw_address for n in path.nodes)))

        if path.edges:
            for edge in path.edges:
                txids.append(edge.transaction_id)
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT transaction_id, timestamp FROM transactions WHERE transaction_id = ANY($1)",
                    txids,
                )
                tx_times = {r["transaction_id"]: _parse_timestamp(r["timestamp"]) for r in rows if r["timestamp"]}
                for txid in txids:
                    t = tx_times.get(txid)
                    if t:
                        times.append(t)
        elif path.utxo_steps:
            for step in path.utxo_steps:
                txids.append(step.transaction.transaction_id)
                t = _parse_timestamp(getattr(step.transaction, "timestamp", None))
                if t:
                    times.append(t)

        if len(times) < self.min_rapid_hops:
            # Missing timestamps prevent unsupported rapid-hop classification
            return None

        window_size = self.min_rapid_hops
        for i in range(len(times) - window_size + 1):
            t_start = times[i]
            t_end = times[i + window_size - 1]
            diff_minutes = (t_end - t_start).total_seconds() / 60.0

            if 0.0 <= diff_minutes <= self.max_rapid_hop_interval_minutes:
                rapid_txids = sorted(list(set(txids[i : i + window_size])))
                det_id = self._make_detection_id("RAPID_HOPPING", path.nodes[0].chain, rapid_txids + addresses)
                explanation = (
                    f"Observed {window_size} consecutive hops within {diff_minutes:.2f} minutes "
                    f"(threshold: <= {self.max_rapid_hop_interval_minutes} min). "
                    "Temporal proximity observation only; does not establish automated or programmatic execution."
                )
                temporal_win = {
                    "start": t_start.isoformat(),
                    "end": t_end.isoformat(),
                    "duration_minutes": round(diff_minutes, 2),
                    "max_interval_minutes": self.max_rapid_hop_interval_minutes,
                }
                return TypologyDetection(
                    detection_id=det_id,
                    typology_type=TypologyType.RAPID_HOPPING,
                    chain=path.nodes[0].chain,
                    confidence_level="observed",
                    explanation=explanation,
                    evidence=[
                        TypologyEvidence(
                            evidence_type="temporal_proximity",
                            evidence_details=explanation,
                            transaction_ids=rapid_txids,
                            addresses=addresses,
                            timestamps=[t.isoformat() for t in times[i : i + window_size]],
                            temporal_window=temporal_win,
                        )
                    ],
                    hop_count=window_size,
                    transaction_ids=rapid_txids,
                    addresses=addresses,
                    temporal_window=temporal_win,
                )

        return None

    # -----------------------------------------------------------------------
    # FAN_IN (Aggregation)
    # -----------------------------------------------------------------------

    async def _fetch_fan_in_records(self, node: TraversalNode) -> List[Dict[str, Any]]:
        """Fetch incoming transfers to the node address."""
        addr = node.raw_address
        chain = node.chain
        network = node.network
        if chain.lower() == "bitcoin":
            query = """
                SELECT vin.address as from_address, d.txid as transaction_id, 
                       vout.value_sat as amount, 'sat' as amount_unit, tx.timestamp
                FROM bitcoin_vouts vout
                JOIN bitcoin_transaction_details d ON vout.detail_pk = d.id
                JOIN transactions tx ON d.transaction_pk = tx.id
                JOIN bitcoin_vins vin ON vin.detail_pk = d.id
                WHERE vout.address = $1 AND vin.address != $1 AND vin.address IS NOT NULL
                ORDER BY tx.timestamp ASC NULLS LAST, d.txid ASC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, addr)
                return [dict(r) for r in rows]
        else:
            query = """
                SELECT t.from_address, tx.transaction_id, t.amount, t.amount_unit, tx.timestamp
                FROM transfers t
                JOIN transactions tx ON t.transaction_pk = tx.id
                WHERE t.chain = $1 AND t.network = $2 AND t.to_address = $3 AND t.from_address != $3
                ORDER BY tx.timestamp ASC NULLS LAST, tx.transaction_id ASC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, chain, network, addr)
                return [dict(r) for r in rows]

    async def _detect_fan_in(self, path: TraversalPath) -> List[TypologyDetection]:
        """Detect fan-in aggregation patterns into nodes along the path within a temporal window."""
        detections: List[TypologyDetection] = []
        chain = path.nodes[0].chain

        for node in path.nodes:
            addr = node.raw_address
            records = await self._fetch_fan_in_records(node)
            if not records:
                continue

            # Process timestamp-bearing records
            parsed_records = []
            for r in records:
                t = _parse_timestamp(r.get("timestamp"))
                if t:
                    parsed_records.append((t, r))

            if not parsed_records:
                # Without timestamps, temporal aggregation window cannot be established
                continue

            parsed_records.sort(key=lambda x: x[0])

            # Find best sliding window <= max_fan_window_hours
            window_seconds = self.max_fan_window_hours * 3600.0
            best_window = None
            best_sources_count = 0

            for i in range(len(parsed_records)):
                t_start = parsed_records[i][0]
                cur_records = []
                cur_sources = set()

                for j in range(i, len(parsed_records)):
                    t_cur, rec = parsed_records[j]
                    if (t_cur - t_start).total_seconds() <= window_seconds:
                        cur_records.append(rec)
                        cur_sources.add(rec["from_address"])
                    else:
                        break

                if len(cur_sources) >= self.min_fan_in_sources and len(cur_sources) > best_sources_count:
                    best_sources_count = len(cur_sources)
                    t_end = cur_records[-1]["timestamp"]
                    if not isinstance(t_end, datetime):
                        t_end = _parse_timestamp(t_end) or t_start
                    best_window = (t_start, t_end, cur_records, cur_sources)

            if best_window:
                t_start, t_end, win_records, win_sources = best_window
                sorted_sources = sorted(list(win_sources))
                sorted_txids = sorted(list(set(r["transaction_id"] for r in win_records)))
                amounts = [
                    {
                        "transaction_id": r["transaction_id"],
                        "from_address": r["from_address"],
                        "amount": str(r["amount"]),
                        "unit": r.get("amount_unit", "native"),
                    }
                    for r in win_records
                    if r.get("amount") is not None
                ]
                duration_hrs = round(max((t_end - t_start).total_seconds() / 3600.0, 0.0), 2)
                temporal_win = {
                    "window_start": t_start.isoformat(),
                    "window_end": t_end.isoformat(),
                    "duration_hours": duration_hrs,
                    "max_window_hours": self.max_fan_window_hours,
                }

                det_id = self._make_detection_id("FAN_IN", chain, [addr] + sorted_sources + sorted_txids)
                explanation = (
                    f"Address {addr} received funds from {len(sorted_sources)} distinct upstream sources "
                    f"across {len(sorted_txids)} transactions within a {duration_hrs}h window "
                    f"(threshold: >= {self.min_fan_in_sources} sources in <= {self.max_fan_window_hours}h). "
                    "Structural topology observation only; does not infer common control or intent."
                )
                detections.append(
                    TypologyDetection(
                        detection_id=det_id,
                        typology_type=TypologyType.FAN_IN,
                        chain=chain,
                        confidence_level="observed",
                        explanation=explanation,
                        evidence=[
                            TypologyEvidence(
                                evidence_type="aggregation_topology",
                                evidence_details=explanation,
                                transaction_ids=sorted_txids,
                                addresses=[addr],
                                counterparty_addresses=sorted_sources,
                                amounts=amounts,
                                timestamps=sorted(list(set(r["timestamp"].isoformat() if isinstance(r["timestamp"], datetime) else str(r["timestamp"]) for r in win_records if r.get("timestamp")))),
                                temporal_window=temporal_win,
                            )
                        ],
                        transaction_ids=sorted_txids,
                        addresses=[addr],
                        counterparty_addresses=sorted_sources,
                        temporal_window=temporal_win,
                    )
                )

        return detections

    # -----------------------------------------------------------------------
    # FAN_OUT (Distribution)
    # -----------------------------------------------------------------------

    async def _fetch_fan_out_records(self, node: TraversalNode) -> List[Dict[str, Any]]:
        """Fetch outgoing transfers from the node address."""
        addr = node.raw_address
        chain = node.chain
        network = node.network
        if chain.lower() == "bitcoin":
            query = """
                SELECT vout.address as to_address, d.txid as transaction_id, 
                       vout.value_sat as amount, 'sat' as amount_unit, tx.timestamp
                FROM bitcoin_vins vin
                JOIN bitcoin_transaction_details d ON vin.detail_pk = d.id
                JOIN transactions tx ON d.transaction_pk = tx.id
                JOIN bitcoin_vouts vout ON vout.detail_pk = d.id
                WHERE vin.address = $1 AND vout.address != $1 AND vout.address IS NOT NULL
                ORDER BY tx.timestamp ASC NULLS LAST, d.txid ASC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, addr)
                return [dict(r) for r in rows]
        else:
            query = """
                SELECT t.to_address, tx.transaction_id, t.amount, t.amount_unit, tx.timestamp
                FROM transfers t
                JOIN transactions tx ON t.transaction_pk = tx.id
                WHERE t.chain = $1 AND t.network = $2 AND t.from_address = $3 AND t.to_address != $3
                ORDER BY tx.timestamp ASC NULLS LAST, tx.transaction_id ASC
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, chain, network, addr)
                return [dict(r) for r in rows]

    async def _detect_fan_out(self, path: TraversalPath) -> List[TypologyDetection]:
        """Detect fan-out distribution patterns from nodes along the path within a temporal window."""
        detections: List[TypologyDetection] = []
        chain = path.nodes[0].chain

        for node in path.nodes:
            addr = node.raw_address
            records = await self._fetch_fan_out_records(node)
            if not records:
                continue

            parsed_records = []
            for r in records:
                t = _parse_timestamp(r.get("timestamp"))
                if t:
                    parsed_records.append((t, r))

            if not parsed_records:
                continue

            parsed_records.sort(key=lambda x: x[0])

            window_seconds = self.max_fan_window_hours * 3600.0
            best_window = None
            best_dests_count = 0

            for i in range(len(parsed_records)):
                t_start = parsed_records[i][0]
                cur_records = []
                cur_dests = set()

                for j in range(i, len(parsed_records)):
                    t_cur, rec = parsed_records[j]
                    if (t_cur - t_start).total_seconds() <= window_seconds:
                        cur_records.append(rec)
                        cur_dests.add(rec["to_address"])
                    else:
                        break

                if len(cur_dests) >= self.min_fan_out_destinations and len(cur_dests) > best_dests_count:
                    best_dests_count = len(cur_dests)
                    t_end = cur_records[-1]["timestamp"]
                    if not isinstance(t_end, datetime):
                        t_end = _parse_timestamp(t_end) or t_start
                    best_window = (t_start, t_end, cur_records, cur_dests)

            if best_window:
                t_start, t_end, win_records, win_dests = best_window
                sorted_dests = sorted(list(win_dests))
                sorted_txids = sorted(list(set(r["transaction_id"] for r in win_records)))
                amounts = [
                    {
                        "transaction_id": r["transaction_id"],
                        "to_address": r["to_address"],
                        "amount": str(r["amount"]),
                        "unit": r.get("amount_unit", "native"),
                    }
                    for r in win_records
                    if r.get("amount") is not None
                ]
                duration_hrs = round(max((t_end - t_start).total_seconds() / 3600.0, 0.0), 2)
                temporal_win = {
                    "window_start": t_start.isoformat(),
                    "window_end": t_end.isoformat(),
                    "duration_hours": duration_hrs,
                    "max_window_hours": self.max_fan_window_hours,
                }

                det_id = self._make_detection_id("FAN_OUT", chain, [addr] + sorted_dests + sorted_txids)
                explanation = (
                    f"Address {addr} distributed funds to {len(sorted_dests)} distinct downstream destinations "
                    f"across {len(sorted_txids)} transactions within a {duration_hrs}h window "
                    f"(threshold: >= {self.min_fan_out_destinations} destinations in <= {self.max_fan_window_hours}h). "
                    "Structural topology observation only; does not classify routine service distribution as suspicious."
                )
                detections.append(
                    TypologyDetection(
                        detection_id=det_id,
                        typology_type=TypologyType.FAN_OUT,
                        chain=chain,
                        confidence_level="observed",
                        explanation=explanation,
                        evidence=[
                            TypologyEvidence(
                                evidence_type="distribution_topology",
                                evidence_details=explanation,
                                transaction_ids=sorted_txids,
                                addresses=[addr],
                                counterparty_addresses=sorted_dests,
                                amounts=amounts,
                                timestamps=sorted(list(set(r["timestamp"].isoformat() if isinstance(r["timestamp"], datetime) else str(r["timestamp"]) for r in win_records if r.get("timestamp")))),
                                temporal_window=temporal_win,
                            )
                        ],
                        transaction_ids=sorted_txids,
                        addresses=[addr],
                        counterparty_addresses=sorted_dests,
                        temporal_window=temporal_win,
                    )
                )

        return detections

    # -----------------------------------------------------------------------
    # MIXER_INTERACTION (Safety stub)
    # -----------------------------------------------------------------------

    def _detect_mixer(self, path: TraversalPath) -> Optional[TypologyDetection]:
        chain = path.nodes[0].chain
        det_id = self._make_detection_id("MIXER_INTERACTION", chain, ["unavailable"])
        return TypologyDetection(
            detection_id=det_id,
            typology_type=TypologyType.MIXER_INTERACTION,
            chain=chain,
            confidence_level="insufficient_evidence",
            explanation="Mixer detection unavailable: no authoritative mixer identification source currently configured.",
            evidence=[],
        )

    # -----------------------------------------------------------------------
    # CROSS_CHAIN (Safety stub)
    # -----------------------------------------------------------------------

    def _detect_cross_chain(self, path: TraversalPath) -> Optional[TypologyDetection]:
        chain = path.nodes[0].chain
        det_id = self._make_detection_id("CROSS_CHAIN", chain, ["unavailable"])
        return TypologyDetection(
            detection_id=det_id,
            typology_type=TypologyType.CROSS_CHAIN,
            chain=chain,
            confidence_level="insufficient_evidence",
            explanation="No bridge detection. Explicit cross-chain bridge data is not currently configured.",
            evidence=[],
        )
