"""CHAKRA Step 3D: Forensics persistence repository.

Manages derived analytical state for forensics, like change address inferences.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import asyncpg

from app.forensics.models import ChangeAddressInference

logger = logging.getLogger(__name__)

_MIGRATION_FILE = Path(__file__).parent.parent / "db" / "migrations" / "003_change_detection_schema.sql"


class ForensicsRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_schema(self) -> None:
        """Apply the forensics schema migration idempotently."""
        sql = _MIGRATION_FILE.read_text(encoding="utf-8-sig")
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("Forensics schema initialized (003_change_detection_schema.sql applied).")

    async def get_transaction_vins(self, txid: str) -> List[Dict[str, Any]]:
        """Fetch all input addresses for a given transaction."""
        query = """
            SELECT v.address
            FROM bitcoin_vins v
            JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
            WHERE d.txid = $1 AND v.address IS NOT NULL
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, txid)
            return [{"address": row["address"]} for row in rows]

    async def get_transaction_vouts(self, txid: str) -> List[Dict[str, Any]]:
        """Fetch all outputs for a given transaction."""
        query = """
            SELECT v.n as output_index, v.address, v.script_type
            FROM bitcoin_vouts v
            JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
            WHERE d.txid = $1 AND v.address IS NOT NULL
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, txid)
            return [
                {
                    "output_index": row["output_index"],
                    "address": row["address"],
                    "script_type": row["script_type"],
                }
                for row in rows
            ]

    async def get_address_output_tx_count(self, address: str) -> int:
        """Count the number of distinct transactions where this address is an output.
        
        This is used for one-time-change-via-reuse-elimination.
        """
        query = """
            SELECT COUNT(DISTINCT d.txid) as tx_count
            FROM bitcoin_vouts v
            JOIN bitcoin_transaction_details d ON v.detail_pk = d.id
            WHERE v.address = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, address)
            return row["tx_count"] if row else 0

    async def upsert_change_inferences(self, inferences: List[ChangeAddressInference]) -> int:
        """Persist change inferences idempotently."""
        if not inferences:
            return 0

        data = [
            (
                i.txid,
                i.output_index,
                i.address,
                i.classification,
                i.confidence,
                i.evidence_reason,
            )
            for i in inferences
        ]

        query = """
            INSERT INTO bitcoin_change_inferences
                (txid, output_index, address, classification, confidence, evidence_reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (txid, output_index) DO UPDATE SET
                address = EXCLUDED.address,
                classification = EXCLUDED.classification,
                confidence = EXCLUDED.confidence,
                evidence_reason = EXCLUDED.evidence_reason
        """

        async with self.pool.acquire() as conn:
            # We use executemany, which doesn't return rows affected easily in asyncpg.
            # We'll return len(data) as an approximation of processed records.
            await conn.executemany(query, data)
        
        return len(data)
