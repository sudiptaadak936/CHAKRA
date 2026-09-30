"""CHAKRA Step 5A: Deterministic ML Feature Extractor.

Extracts a versioned, immutable feature vector from canonical blockchain evidence
without mutating any underlying state, without external API calls, and preserving
exact monetary representations.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg

from app.graph.models import normalize_address
from app.schemas.chain import Chain
from app.forensics.typology_detector import TypologyDetection
from app.schemas.transaction import AssetType
from app.schemas.ml_features import (
    FEATURE_SCHEMA_VERSION,
    CANONICAL_FEATURE_NAMES,
    UNSUPPORTED_FEATURE_NAMES,
    FeatureStatus,
    FeatureAvailability,
    MLFeatureValues,
    MLFeatureRecord,
)

logger = logging.getLogger(__name__)


def _parse_timestamp(val: Any) -> Optional[datetime]:
    """Helper to parse a timestamp into a timezone-aware datetime."""
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
def _is_typology_within_cutoff(
    typo: TypologyDetection,
    cutoff_dt: Optional[datetime],
    start_dt: Optional[datetime] = None,
) -> bool:
    """Verify if a typology detection occurred entirely within [start_dt, cutoff_dt].

    Fail closed: If the typology contains events outside the observation window, return False.
    If the typology has timestamps, all evidence timestamps must satisfy:
    (start_dt is None or ts >= start_dt) and (cutoff_dt is None or ts <= cutoff_dt).
    If the typology has no timestamps whatsoever when an observation boundary is provided,
    fail closed and return False.
    """
    if cutoff_dt is None and start_dt is None:
        return True

    found_timestamps: List[datetime] = []

    # 1. Top-level temporal_window
    if isinstance(typo.temporal_window, dict):
        for k in ("end", "window_end", "end_time", "max_timestamp", "timestamp", "start", "start_time", "min_timestamp"):
            val = typo.temporal_window.get(k)
            if val:
                dt = _parse_timestamp(val)
                if dt:
                    found_timestamps.append(dt)

    # 2. Evidence timestamps and temporal windows
    for ev in getattr(typo, "evidence", []) or []:
        if isinstance(ev.temporal_window, dict):
            for k in ("end", "window_end", "end_time", "max_timestamp", "timestamp", "start", "start_time", "min_timestamp"):
                val = ev.temporal_window.get(k)
                if val:
                    dt = _parse_timestamp(val)
                    if dt:
                        found_timestamps.append(dt)
        for ts_item in getattr(ev, "timestamps", []) or []:
            dt = _parse_timestamp(ts_item)
            if dt:
                found_timestamps.append(dt)
        for hop in getattr(ev, "peel_details", []) or []:
            if isinstance(hop, dict) and hop.get("timestamp"):
                dt = _parse_timestamp(hop.get("timestamp"))
                if dt:
                    found_timestamps.append(dt)

    if not found_timestamps:
        return False

    return all(
        (cutoff_dt is None or ts <= cutoff_dt) and (start_dt is None or ts >= start_dt)
        for ts in found_timestamps
    )


class MLFeatureExtractor:
    """Deterministic, read-only feature extraction engine for CHAKRA ML sub-layers."""

    @staticmethod
    def extract_from_evidence(
        target_address: str,
        chain: str,
        network: str,
        transfers: Optional[List[Any]] = None,
        transactions: Optional[List[Any]] = None,
        bitcoin_vins: Optional[List[Any]] = None,
        bitcoin_vouts: Optional[List[Any]] = None,
        typologies: Optional[List[TypologyDetection]] = None,
        extracted_at: Optional[datetime] = None,
        cutoff_timestamp: Optional[datetime | str] = None,
        start_timestamp: Optional[datetime | str] = None,
    ) -> MLFeatureRecord:
        """Extract features deterministically from pre-loaded canonical evidence objects.

        Guarantees:
        - No database writes or external network calls.
        - Exact integer / Decimal monetary preservation.
        - Deterministic, sorted canonical output.
        - Fail-closed on invalid inputs.
        """
        # 1. Input Validation — Fail closed
        if not target_address or not target_address.strip():
            raise ValueError("Target address is required.")
        if not chain or not chain.strip():
            raise ValueError("Chain is required.")

        try:
            chain_enum = Chain(chain.lower())
        except ValueError:
            raise ValueError(f"Invalid chain: {chain}")

        if not network or not network.strip():
            raise ValueError("Network is required.")

        norm_target = normalize_address(chain, target_address)
        if not norm_target:
            raise ValueError("Target address normalization yielded empty string.")

        extracted_at = extracted_at or datetime.now(timezone.utc)

        cutoff_dt: Optional[datetime] = None
        if cutoff_timestamp is not None:
            cutoff_dt = _parse_timestamp(cutoff_timestamp)
            if cutoff_dt is None:
                raise ValueError(f"Invalid cutoff_timestamp: {cutoff_timestamp}")

        start_dt: Optional[datetime] = None
        if start_timestamp is not None:
            start_dt = _parse_timestamp(start_timestamp)
            if start_dt is None:
                raise ValueError(f"Invalid start_timestamp: {start_timestamp}")

        if start_dt is not None and cutoff_dt is not None and start_dt > cutoff_dt:
            raise ValueError("start_timestamp cannot be greater than cutoff_timestamp")

        # 2. Accumulators for metrics
        in_degree = 0
        out_degree = 0
        relevant_tx_ids: Set[str] = set()
        counterparties: Set[str] = set()
        total_received = Decimal(0)
        total_sent = Decimal(0)
        all_timestamps: List[datetime] = []
        receipt_events: List[Tuple[datetime, Decimal]] = []
        spend_events: List[Tuple[datetime, Decimal]] = []

        # Pre-index transaction timestamps for transfers that reference transaction_id
        tx_ts_map: Dict[str, datetime] = {}
        for tx in (transactions or []):
            _txid = getattr(tx, "transaction_id", None) if not isinstance(tx, dict) else tx.get("transaction_id")
            _txts_raw = getattr(tx, "timestamp", None) if not isinstance(tx, dict) else tx.get("timestamp")
            _txts = _parse_timestamp(_txts_raw)
            if _txid and _txts:
                tx_ts_map[str(_txid).lower()] = _txts

        # 3. Process Transfers
        # Either canonical Transfer models or dict representations
        for tr in (transfers or []):
            from_addr = getattr(tr, "from_address", None) if not isinstance(tr, dict) else tr.get("from_address")
            to_addr = getattr(tr, "to_address", None) if not isinstance(tr, dict) else tr.get("to_address")
            amt_raw = getattr(tr, "amount", 0) if not isinstance(tr, dict) else tr.get("amount", 0)
            tx_id = getattr(tr, "transaction_id", None) if not isinstance(tr, dict) else tr.get("transaction_id")
            ts_raw = getattr(tr, "timestamp", None) if not isinstance(tr, dict) else tr.get("timestamp")
            ts = _parse_timestamp(ts_raw)

            # Inherit parent transaction timestamp when no explicit transfer timestamp exists
            if ts is None and tx_id and str(tx_id).lower() in tx_ts_map:
                ts = tx_ts_map[str(tx_id).lower()]

            # Temporal isolation: Exclude evidence outside observation window or lacking timestamp when active
            if cutoff_dt is not None and (ts is None or ts > cutoff_dt):
                continue
            if start_dt is not None and (ts is None or ts < start_dt):
                continue


            norm_from = normalize_address(chain, from_addr) if from_addr else None
            norm_to = normalize_address(chain, to_addr) if to_addr else None

            amt_dec = Decimal(str(amt_raw)) if amt_raw is not None else Decimal(0)

            if tx_id:
                relevant_tx_ids.add(str(tx_id))
            if ts:
                all_timestamps.append(ts)

            # Incoming transfer to target
            if norm_to == norm_target:
                in_degree += 1
                if getattr(tr, 'asset_type', None) == AssetType.NATIVE or (isinstance(tr, dict) and tr.get('asset_type') == AssetType.NATIVE.value):
                    total_received += amt_dec
                if norm_from and norm_from != norm_target:
                    counterparties.add(norm_from)
                if ts:
                    receipt_events.append((ts, amt_dec))

            # Outgoing transfer from target
            if norm_from == norm_target:
                out_degree += 1
                if getattr(tr, 'asset_type', None) == AssetType.NATIVE or (isinstance(tr, dict) and tr.get('asset_type') == AssetType.NATIVE.value):
                    total_sent += amt_dec
                if norm_to and norm_to != norm_target:
                    counterparties.add(norm_to)
                if ts:
                    spend_events.append((ts, amt_dec))

        # 4. Process Top-level Transactions
        # Used when transactions have native movements without separate transfers,
        # or to establish canonical transaction counts and timestamps.
        for tx in (transactions or []):
            tx_id = getattr(tx, "transaction_id", None) if not isinstance(tx, dict) else tx.get("transaction_id")
            from_addr = getattr(tx, "from_address", None) if not isinstance(tx, dict) else tx.get("from_address")
            to_addr = getattr(tx, "to_address", None) if not isinstance(tx, dict) else tx.get("to_address")
            native_val_raw = getattr(tx, "native_value", 0) if not isinstance(tx, dict) else tx.get("native_value", 0)
            ts_raw = getattr(tx, "timestamp", None) if not isinstance(tx, dict) else tx.get("timestamp")
            tx_transfers = getattr(tx, "transfers", []) if not isinstance(tx, dict) else tx.get("transfers", [])
            ts = _parse_timestamp(ts_raw)

            # Temporal isolation: Exclude transactions outside observation window or lacking timestamp when active
            if cutoff_dt is not None and (ts is None or ts > cutoff_dt):
                continue
            if start_dt is not None and (ts is None or ts < start_dt):
                continue

            norm_from = normalize_address(chain, from_addr) if from_addr else None
            norm_to = normalize_address(chain, to_addr) if to_addr else None

            is_relevant = (norm_from == norm_target) or (norm_to == norm_target)
            if is_relevant and tx_id:
                relevant_tx_ids.add(str(tx_id))
            if is_relevant and ts:
                all_timestamps.append(ts)

            # If the transaction has native_value and no sub-transfers were provided,
            # account for top-level native transfer directly
            if not tx_transfers and not transfers and (native_val_raw or 0) > 0:
                val_dec = Decimal(str(native_val_raw))
                if norm_to == norm_target:
                    in_degree += 1
                    total_received += val_dec
                    if norm_from and norm_from != norm_target:
                        counterparties.add(norm_from)
                    if ts:
                        receipt_events.append((ts, val_dec))

                if norm_from == norm_target:
                    out_degree += 1
                    total_sent += val_dec
                    if norm_to and norm_to != norm_target:
                        counterparties.add(norm_to)
                    if ts:
                        spend_events.append((ts, val_dec))

        # 5. Process Bitcoin UTXO Inputs (vins) & Outputs (vouts)
        for vout in (bitcoin_vouts or []):
            vout_addr = getattr(vout, "address", None) if not isinstance(vout, dict) else vout.get("address")
            norm_addr = normalize_address(chain, vout_addr) if vout_addr else None
            val_raw = getattr(vout, "value_sat", 0) if not isinstance(vout, dict) else vout.get("value_sat", 0)
            tx_id = getattr(vout, "txid", None) if not isinstance(vout, dict) else vout.get("txid")
            ts_raw = getattr(vout, "timestamp", None) if not isinstance(vout, dict) else vout.get("timestamp")
            ts = _parse_timestamp(ts_raw)

            # Temporal isolation: Exclude outputs outside observation window or lacking timestamp when active
            if cutoff_dt is not None and (ts is None or ts > cutoff_dt):
                continue
            if start_dt is not None and (ts is None or ts < start_dt):
                continue

            if norm_addr == norm_target:
                in_degree += 1
                val_dec = Decimal(str(val_raw)) if val_raw is not None else Decimal(0)
                total_received += val_dec
                if tx_id:
                    relevant_tx_ids.add(str(tx_id))
                if ts:
                    all_timestamps.append(ts)
                    receipt_events.append((ts, val_dec))

        for vin in (bitcoin_vins or []):
            vin_addr = getattr(vin, "address", None) if not isinstance(vin, dict) else vin.get("address")
            norm_addr = normalize_address(chain, vin_addr) if vin_addr else None
            val_raw = getattr(vin, "value_sat", 0) if not isinstance(vin, dict) else vin.get("value_sat", 0)
            tx_id = getattr(vin, "txid", None) if not isinstance(vin, dict) else vin.get("txid")
            ts_raw = getattr(vin, "timestamp", None) if not isinstance(vin, dict) else vin.get("timestamp")
            ts = _parse_timestamp(ts_raw)

            # Temporal isolation: Exclude inputs outside observation window or lacking timestamp when active
            if cutoff_dt is not None and (ts is None or ts > cutoff_dt):
                continue
            if start_dt is not None and (ts is None or ts < start_dt):
                continue

            if norm_addr == norm_target:
                out_degree += 1
                val_dec = Decimal(str(val_raw)) if val_raw is not None else Decimal(0)
                total_sent += val_dec
                if tx_id:
                    relevant_tx_ids.add(str(tx_id))
                if ts:
                    all_timestamps.append(ts)
                    spend_events.append((ts, val_dec))

        # 6. Calculate Amount Retention Ratio
        if total_received == Decimal(0):
            amount_retention_ratio = 0.0
        else:
            diff = total_received - total_sent
            ratio_float = float(diff / total_received)
            amount_retention_ratio = round(ratio_float, 6) if math.isfinite(ratio_float) else 0.0

        # 7. Calculate Time Active Seconds
        valid_timestamps = [ts for ts in all_timestamps if ts is not None]
        if len(valid_timestamps) >= 2:
            min_ts = min(valid_timestamps)
            max_ts = max(valid_timestamps)
            time_active_seconds = max(0.0, float((max_ts - min_ts).total_seconds()))
        else:
            time_active_seconds = 0.0

        # 8. Calculate Inter-Hop Velocity Avg
        # Defined as mean elapsed seconds between a receipt and subsequent spend
        receipt_events.sort(key=lambda x: x[0])
        spend_events.sort(key=lambda x: x[0])

        deltas: List[float] = []
        for spend_ts, _ in spend_events:
            # Find the most recent prior receipt
            prior_receipts = [r_ts for r_ts, _ in receipt_events if r_ts <= spend_ts]
            if prior_receipts:
                latest_receipt = max(prior_receipts)
                delta = float((spend_ts - latest_receipt).total_seconds())
                if delta >= 0:
                    deltas.append(delta)

        if deltas:
            inter_hop_velocity_avg = round(sum(deltas) / len(deltas), 6)
        else:
            inter_hop_velocity_avg = -1.0  # Documented sentinel for unavailable velocity

        # 9. Evaluate Typology Flags
        peel_flag = 0
        rapid_flag = 0
        fan_in_flag = 0
        fan_out_flag = 0

        for typo in (typologies or []):
            if typo.confidence_level != "observed":
                continue
            # Temporal isolation: Exclude typology detections outside observation window
            if (cutoff_dt is not None or start_dt is not None) and not _is_typology_within_cutoff(typo, cutoff_dt, start_dt):
                continue

            t_type = typo.typology_type.value if hasattr(typo.typology_type, "value") else str(typo.typology_type)
            if t_type == "PEEL_CHAIN":
                peel_flag = 1
            elif t_type == "RAPID_HOPPING":
                rapid_flag = 1
            elif t_type == "FAN_IN":
                fan_in_flag = 1
            elif t_type == "FAN_OUT":
                fan_out_flag = 1

        # 10. Assemble Feature Values
        features = MLFeatureValues(
            in_degree=in_degree,
            out_degree=out_degree,
            total_tx_count=len(relevant_tx_ids),
            total_received_native=total_received,
            total_sent_native=total_sent,
            amount_retention_ratio=amount_retention_ratio,
            time_active_seconds=time_active_seconds,
            inter_hop_velocity_avg=inter_hop_velocity_avg,
            unique_counterparties=len(counterparties),
            typology_peel_chain_flag=peel_flag,
            typology_rapid_hop_flag=rapid_flag,
            typology_fan_in_flag=fan_in_flag,
            typology_fan_out_flag=fan_out_flag,
        )

        # 11. Build Feature Availability Declarations
        availability: Dict[str, FeatureAvailability] = {}
        for fname in CANONICAL_FEATURE_NAMES:
            availability[fname] = FeatureAvailability(
                feature_name=fname,
                status=FeatureStatus.AVAILABLE,
                reason="Derived deterministically from canonical evidence."
            )

        availability["hop_distance_to_mixer"] = FeatureAvailability(
            feature_name="hop_distance_to_mixer",
            status=FeatureStatus.UNSUPPORTED,
            reason="Authoritative mixer registry is not integrated in CHAKRA Steps 0-4.5A."
        )
        availability["cluster_size"] = FeatureAvailability(
            feature_name="cluster_size",
            status=FeatureStatus.UNSUPPORTED,
            reason="ClusteringRepository does not expose address-to-cluster lookup without extending Step 2/3 contracts."
        )

        return MLFeatureRecord(
            target_address=target_address,
            chain=chain,
            network=network,
            normalized_address=norm_target,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features=features,
            availability=availability,
            extracted_at=extracted_at,
        )

    @classmethod
    async def extract_from_db(
        cls,
        pool: asyncpg.Pool,
        target_address: str,
        chain: str,
        network: str,
        typologies: Optional[List[TypologyDetection]] = None,
        extracted_at: Optional[datetime] = None,
        cutoff_timestamp: Optional[datetime | str] = None,
        start_timestamp: Optional[datetime | str] = None,
    ) -> MLFeatureRecord:
        """Extract features by executing strictly read-only parameterized queries against PostgreSQL.

        Zero database writes, zero mutations.
        """
        if not target_address or not target_address.strip():
            raise ValueError("Target address is required.")
        if not chain or not chain.strip():
            raise ValueError("Chain is required.")
        if not network or not network.strip():
            raise ValueError("Network is required.")

        norm_target = normalize_address(chain, target_address)
        chain_lower = chain.lower()

        cutoff_dt: Optional[datetime] = None
        if cutoff_timestamp is not None:
            cutoff_dt = _parse_timestamp(cutoff_timestamp)
            if cutoff_dt is None:
                raise ValueError(f"Invalid cutoff_timestamp: {cutoff_timestamp}")

        start_dt: Optional[datetime] = None
        if start_timestamp is not None:
            start_dt = _parse_timestamp(start_timestamp)
            if start_dt is None:
                raise ValueError(f"Invalid start_timestamp: {start_timestamp}")

        if start_dt is not None and cutoff_dt is not None and start_dt > cutoff_dt:
            raise ValueError("start_timestamp cannot be greater than cutoff_timestamp")

        transfers_data: List[Dict[str, Any]] = []
        txs_data: List[Dict[str, Any]] = []
        vins_data: List[Dict[str, Any]] = []
        vouts_data: List[Dict[str, Any]] = []

        async with pool.acquire() as conn:
            if chain_lower == "bitcoin":
                # Bitcoin UTXO queries
                vout_query = """
                    SELECT v.address, v.value_sat, tx.transaction_id, tx.timestamp
                    FROM bitcoin_vouts v
                    JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
                    JOIN transactions tx ON d.transaction_pk = tx.id
                    WHERE v.address = $1 AND tx.chain = $2 AND tx.network = $3
                """
                vin_query = """
                    SELECT v.address, v.value_sat, tx.transaction_id, tx.timestamp
                    FROM bitcoin_vins v
                    JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
                    JOIN transactions tx ON d.transaction_pk = tx.id
                    WHERE v.address = $1 AND tx.chain = $2 AND tx.network = $3
                """
                params = [norm_target, chain_lower, network]
                if start_dt is not None:
                    p_idx = len(params) + 1
                    vout_query += f" AND tx.timestamp >= ${p_idx}"
                    vin_query += f" AND tx.timestamp >= ${p_idx}"
                    params.append(start_dt)
                if cutoff_dt is not None:
                    p_idx = len(params) + 1
                    vout_query += f" AND tx.timestamp <= ${p_idx}"
                    vin_query += f" AND tx.timestamp <= ${p_idx}"
                    params.append(cutoff_dt)

                vout_rows = await conn.fetch(vout_query, *params)
                vin_rows = await conn.fetch(vin_query, *params)
                vouts_data = [dict(r) for r in vout_rows]
                vins_data = [dict(r) for r in vin_rows]
            else:
                # Account-based chain queries (EVM, Tron, Solana)
                tr_query = """
                    SELECT t.from_address, t.to_address, t.amount, tx.transaction_id, tx.timestamp
                    FROM transfers t
                    JOIN transactions tx ON t.transaction_pk = tx.id
                    WHERE (t.from_address = $1 OR t.to_address = $1)
                      AND t.chain = $2 AND t.network = $3
                """
                tx_query = """
                    SELECT transaction_id, from_address, to_address, native_value, timestamp
                    FROM transactions
                    WHERE (from_address = $1 OR to_address = $1)
                      AND chain = $2 AND network = $3
                """
                params_tr = [norm_target, chain_lower, network]
                params_tx = [norm_target, chain_lower, network]
                if start_dt is not None:
                    p_tr = len(params_tr) + 1
                    tr_query += f" AND tx.timestamp >= ${p_tr}"
                    params_tr.append(start_dt)
                    p_tx = len(params_tx) + 1
                    tx_query += f" AND timestamp >= ${p_tx}"
                    params_tx.append(start_dt)
                if cutoff_dt is not None:
                    p_tr = len(params_tr) + 1
                    tr_query += f" AND tx.timestamp <= ${p_tr}"
                    params_tr.append(cutoff_dt)
                    p_tx = len(params_tx) + 1
                    tx_query += f" AND timestamp <= ${p_tx}"
                    params_tx.append(cutoff_dt)

                tr_rows = await conn.fetch(tr_query, *params_tr)
                tx_rows = await conn.fetch(tx_query, *params_tx)
                transfers_data = [dict(r) for r in tr_rows]
                txs_data = [dict(r) for r in tx_rows]

        return cls.extract_from_evidence(
            target_address=target_address,
            chain=chain,
            network=network,
            transfers=transfers_data,
            transactions=txs_data,
            bitcoin_vins=vins_data,
            bitcoin_vouts=vouts_data,
            typologies=typologies,
            extracted_at=extracted_at,
            cutoff_timestamp=cutoff_dt,
            start_timestamp=start_dt,
        )

