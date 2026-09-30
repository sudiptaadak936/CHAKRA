"""CHAKRA Step 5H: Historical Typology Context Adapter.

Deterministic, offline adapter that constructs canonical TypologyDetection
evidence from bounded local historical evidence graphs without requiring live
PostgreSQL or Neo4j connections.

Guarantees:
- Strict temporal isolation: timestamp <= cutoff_timestamp.
- Exact amount and address preservation.
- Canonical SHA-256 deterministic detection IDs matching Step 3G typology detector.
- No synthetic edges, no ownership inferences, no fabricated probabilities.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from app.forensics.typology_detector import (
    DEFAULT_FAN_WINDOW_HOURS,
    MAX_RAPID_HOP_INTERVAL_MINUTES,
    MIN_FAN_IN_SOURCES,
    MIN_FAN_OUT_DESTINATIONS,
    MIN_PEEL_CHAIN_HOPS,
    MIN_RAPID_HOPS,
    TypologyDetection,
    TypologyEvidence,
    TypologyType,
)
from app.graph.models import normalize_address


def _make_detection_id(typ: str, chain: str, elements: List[str]) -> str:
    """Create a deterministic SHA-256 detection ID from canonical inputs."""
    canonical = f"{typ}:{chain}:" + "|".join(sorted(set(str(e) for e in elements if e)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_ts(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


class HistoricalTypologyContextAdapter:
    """Evaluates canonical forensic typologies on bounded historical evidence."""

    @classmethod
    def detect_typologies_from_evidence(
        cls,
        target_address: str,
        chain: str,
        network: str,
        transactions: Optional[List[Any]] = None,
        transfers: Optional[List[Any]] = None,
        cutoff_timestamp: Optional[datetime] = None,
        start_timestamp: Optional[datetime] = None,
        multi_hop_paths: Optional[List[Any]] = None,
    ) -> List[TypologyDetection]:
        """Detect typologies applicable to target_address within [start, cutoff]."""
        norm_target = normalize_address(chain, target_address)
        if not norm_target:
            return []

        cutoff_dt = _parse_ts(cutoff_timestamp)
        start_dt = _parse_ts(start_timestamp)

        detections: List[TypologyDetection] = []

        # Build tx_lookup for transfer timestamps if needed
        tx_ts_map: Dict[str, datetime] = {}
        for tx in (transactions or []):
            txid = getattr(tx, "transaction_id", None) if not isinstance(tx, dict) else tx.get("transaction_id")
            txts = getattr(tx, "timestamp", None) if not isinstance(tx, dict) else tx.get("timestamp")
            parsed_txts = _parse_ts(txts)
            if txid and parsed_txts:
                tx_ts_map[str(txid).lower()] = parsed_txts

        # -------------------------------------------------------------------
        # 1. FAN-IN Detection (Aggregation into target_address)
        # -------------------------------------------------------------------
        in_records: List[Tuple[datetime, str, str, Any, str]] = []  # (ts, from_addr, txid, amount, unit)
        for tr in (transfers or []):
            to_addr = getattr(tr, "to_address", None) if not isinstance(tr, dict) else tr.get("to_address")
            from_addr = getattr(tr, "from_address", None) if not isinstance(tr, dict) else tr.get("from_address")
            norm_to = normalize_address(chain, to_addr) if to_addr else None
            norm_from = normalize_address(chain, from_addr) if from_addr else None

            if norm_to == norm_target and norm_from and norm_from != norm_target:
                ts_raw = getattr(tr, "timestamp", None) if not isinstance(tr, dict) else tr.get("timestamp")
                tx_id = getattr(tr, "transaction_id", None) if not isinstance(tr, dict) else tr.get("transaction_id")
                ts = _parse_ts(ts_raw)
                if ts is None and tx_id:
                    ts = tx_ts_map.get(str(tx_id).lower())

                if ts is None:
                    continue
                if cutoff_dt is not None and ts > cutoff_dt:
                    continue
                if start_dt is not None and ts < start_dt:
                    continue

                amt = getattr(tr, "amount", 0) if not isinstance(tr, dict) else tr.get("amount", 0)
                unit = getattr(tr, "amount_unit", "native") if not isinstance(tr, dict) else tr.get("amount_unit", "native")
                in_records.append((ts, norm_from, str(tx_id or ""), amt, unit))

        in_records.sort(key=lambda x: x[0])
        fan_in_det = cls._evaluate_fan_in(norm_target, chain, in_records, cutoff_dt)
        if fan_in_det:
            detections.append(fan_in_det)

        # -------------------------------------------------------------------
        # 2. FAN-OUT Detection (Distribution from target_address)
        # -------------------------------------------------------------------
        out_records: List[Tuple[datetime, str, str, Any, str]] = []  # (ts, to_addr, txid, amount, unit)
        for tr in (transfers or []):
            from_addr = getattr(tr, "from_address", None) if not isinstance(tr, dict) else tr.get("from_address")
            to_addr = getattr(tr, "to_address", None) if not isinstance(tr, dict) else tr.get("to_address")
            norm_from = normalize_address(chain, from_addr) if from_addr else None
            norm_to = normalize_address(chain, to_addr) if to_addr else None

            if norm_from == norm_target and norm_to and norm_to != norm_target:
                ts_raw = getattr(tr, "timestamp", None) if not isinstance(tr, dict) else tr.get("timestamp")
                tx_id = getattr(tr, "transaction_id", None) if not isinstance(tr, dict) else tr.get("transaction_id")
                ts = _parse_ts(ts_raw)
                if ts is None and tx_id:
                    ts = tx_ts_map.get(str(tx_id).lower())

                if ts is None:
                    continue
                if cutoff_dt is not None and ts > cutoff_dt:
                    continue
                if start_dt is not None and ts < start_dt:
                    continue

                amt = getattr(tr, "amount", 0) if not isinstance(tr, dict) else tr.get("amount", 0)
                unit = getattr(tr, "amount_unit", "native") if not isinstance(tr, dict) else tr.get("amount_unit", "native")
                out_records.append((ts, norm_to, str(tx_id or ""), amt, unit))

        out_records.sort(key=lambda x: x[0])
        fan_out_det = cls._evaluate_fan_out(norm_target, chain, out_records, cutoff_dt)
        if fan_out_det:
            detections.append(fan_out_det)

        # -------------------------------------------------------------------
        # 3. RAPID HOPPING (if multi-hop sequence is provided in context)
        # -------------------------------------------------------------------
        if multi_hop_paths:
            for path in multi_hop_paths:
                rapid_det = cls._evaluate_rapid_hop_path(path, chain, cutoff_dt, start_dt)
                if rapid_det:
                    detections.append(rapid_det)

        detections.sort(key=lambda d: (d.typology_type.value, d.detection_id))
        return detections

    @classmethod
    def _evaluate_fan_in(
        cls,
        target_addr: str,
        chain: str,
        records: List[Tuple[datetime, str, str, Any, str]],
        cutoff_dt: Optional[datetime],
    ) -> Optional[TypologyDetection]:
        """Evaluate sliding 24h window for >= 3 upstream sources."""
        if not records:
            return None

        window_seconds = DEFAULT_FAN_WINDOW_HOURS * 3600.0
        best_window = None
        best_sources_count = 0

        for i in range(len(records)):
            t_start = records[i][0]
            cur_records = []
            cur_sources = set()

            for j in range(i, len(records)):
                t_cur, src, txid, amt, unit = records[j]
                if (t_cur - t_start).total_seconds() <= window_seconds:
                    cur_records.append(records[j])
                    cur_sources.add(src)
                else:
                    break

            if len(cur_sources) >= MIN_FAN_IN_SOURCES and len(cur_sources) > best_sources_count:
                best_sources_count = len(cur_sources)
                t_end = cur_records[-1][0]
                best_window = (t_start, t_end, cur_records, cur_sources)

        if best_window:
            t_start, t_end, win_records, win_sources = best_window
            sorted_sources = sorted(list(win_sources))
            sorted_txids = sorted(list(set(r[2] for r in win_records if r[2])))
            duration_hrs = round(max((t_end - t_start).total_seconds() / 3600.0, 0.0), 2)

            det_id = _make_detection_id("FAN_IN", chain, [target_addr] + sorted_sources + sorted_txids)
            explanation = (
                f"Address {target_addr} received funds from {len(sorted_sources)} distinct upstream sources "
                f"across {len(sorted_txids)} transactions within a {duration_hrs}h window "
                f"(threshold: >= {MIN_FAN_IN_SOURCES} sources in <= {DEFAULT_FAN_WINDOW_HOURS}h). "
                "Structural topology observation only; does not infer common control or intent."
            )
            temporal_win = {
                "window_start": t_start.isoformat(),
                "window_end": t_end.isoformat(),
                "duration_hours": duration_hrs,
                "max_window_hours": DEFAULT_FAN_WINDOW_HOURS,
            }
            return TypologyDetection(
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
                        addresses=[target_addr],
                        counterparty_addresses=sorted_sources,
                        amounts=[{"transaction_id": r[2], "from_address": r[1], "amount": str(r[3]), "unit": r[4]} for r in win_records],
                        timestamps=sorted(list(set(r[0].isoformat() for r in win_records))),
                        temporal_window=temporal_win,
                    )
                ],
                transaction_ids=sorted_txids,
                addresses=[target_addr],
                counterparty_addresses=sorted_sources,
                temporal_window=temporal_win,
            )
        return None

    @classmethod
    def _evaluate_fan_out(
        cls,
        target_addr: str,
        chain: str,
        records: List[Tuple[datetime, str, str, Any, str]],
        cutoff_dt: Optional[datetime],
    ) -> Optional[TypologyDetection]:
        """Evaluate sliding 24h window for >= 3 downstream destinations."""
        if not records:
            return None

        window_seconds = DEFAULT_FAN_WINDOW_HOURS * 3600.0
        best_window = None
        best_dests_count = 0

        for i in range(len(records)):
            t_start = records[i][0]
            cur_records = []
            cur_dests = set()

            for j in range(i, len(records)):
                t_cur, dst, txid, amt, unit = records[j]
                if (t_cur - t_start).total_seconds() <= window_seconds:
                    cur_records.append(records[j])
                    cur_dests.add(dst)
                else:
                    break

            if len(cur_dests) >= MIN_FAN_OUT_DESTINATIONS and len(cur_dests) > best_dests_count:
                best_dests_count = len(cur_dests)
                t_end = cur_records[-1][0]
                best_window = (t_start, t_end, cur_records, cur_dests)

        if best_window:
            t_start, t_end, win_records, win_dests = best_window
            sorted_dests = sorted(list(win_dests))
            sorted_txids = sorted(list(set(r[2] for r in win_records if r[2])))
            duration_hrs = round(max((t_end - t_start).total_seconds() / 3600.0, 0.0), 2)

            det_id = _make_detection_id("FAN_OUT", chain, [target_addr] + sorted_dests + sorted_txids)
            explanation = (
                f"Address {target_addr} distributed funds to {len(sorted_dests)} distinct downstream destinations "
                f"across {len(sorted_txids)} transactions within a {duration_hrs}h window "
                f"(threshold: >= {MIN_FAN_OUT_DESTINATIONS} destinations in <= {DEFAULT_FAN_WINDOW_HOURS}h). "
                "Structural topology observation only; does not infer common control or intent."
            )
            temporal_win = {
                "window_start": t_start.isoformat(),
                "window_end": t_end.isoformat(),
                "duration_hours": duration_hrs,
                "max_window_hours": DEFAULT_FAN_WINDOW_HOURS,
            }
            return TypologyDetection(
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
                        addresses=[target_addr],
                        counterparty_addresses=sorted_dests,
                        amounts=[{"transaction_id": r[2], "to_address": r[1], "amount": str(r[3]), "unit": r[4]} for r in win_records],
                        timestamps=sorted(list(set(r[0].isoformat() for r in win_records))),
                        temporal_window=temporal_win,
                    )
                ],
                transaction_ids=sorted_txids,
                addresses=[target_addr],
                counterparty_addresses=sorted_dests,
                temporal_window=temporal_win,
            )
        return None

    @classmethod
    def _evaluate_rapid_hop_path(
        cls,
        path: Any,
        chain: str,
        cutoff_dt: Optional[datetime],
        start_dt: Optional[datetime],
    ) -> Optional[TypologyDetection]:
        """Evaluate a multi-hop sequence for rapid movement within 60 minutes."""
        # path is a sequence of (txid, timestamp, address)
        steps = getattr(path, "steps", path) if not isinstance(path, list) else path
        valid_steps = []
        for step in steps:
            if isinstance(step, dict):
                txid = step.get("transaction_id")
                ts = _parse_ts(step.get("timestamp"))
                addr = step.get("address")
            else:
                txid = getattr(step, "transaction_id", None)
                ts = _parse_ts(getattr(step, "timestamp", None))
                addr = getattr(step, "address", None)

            if ts is None:
                continue
            if cutoff_dt is not None and ts > cutoff_dt:
                continue
            if start_dt is not None and ts < start_dt:
                continue
            valid_steps.append((ts, str(txid or ""), str(addr or "")))

        if len(valid_steps) < MIN_RAPID_HOPS:
            return None

        for i in range(len(valid_steps) - MIN_RAPID_HOPS + 1):
            t_start = valid_steps[i][0]
            t_end = valid_steps[i + MIN_RAPID_HOPS - 1][0]
            diff_min = (t_end - t_start).total_seconds() / 60.0
            if 0.0 <= diff_min <= MAX_RAPID_HOP_INTERVAL_MINUTES:
                window_steps = valid_steps[i : i + MIN_RAPID_HOPS]
                txids = sorted(list(set(s[1] for s in window_steps if s[1])))
                addrs = sorted(list(set(s[2] for s in window_steps if s[2])))
                det_id = _make_detection_id("RAPID_HOPPING", chain, txids + addrs)
                explanation = (
                    f"Observed {MIN_RAPID_HOPS} consecutive hops within {diff_min:.2f} minutes "
                    f"(threshold: <= {MAX_RAPID_HOP_INTERVAL_MINUTES} min)."
                )
                temporal_win = {
                    "start": t_start.isoformat(),
                    "end": t_end.isoformat(),
                    "duration_minutes": round(diff_min, 2),
                    "max_interval_minutes": MAX_RAPID_HOP_INTERVAL_MINUTES,
                }
                return TypologyDetection(
                    detection_id=det_id,
                    typology_type=TypologyType.RAPID_HOPPING,
                    chain=chain,
                    confidence_level="observed",
                    explanation=explanation,
                    evidence=[
                        TypologyEvidence(
                            evidence_type="temporal_proximity",
                            evidence_details=explanation,
                            transaction_ids=txids,
                            addresses=addrs,
                            timestamps=[s[0].isoformat() for s in window_steps],
                            temporal_window=temporal_win,
                        )
                    ],
                    hop_count=MIN_RAPID_HOPS,
                    transaction_ids=txids,
                    addresses=addrs,
                    temporal_window=temporal_win,
                )
        return None
