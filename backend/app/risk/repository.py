"""CHAKRA Step 5: Risk Feature Repository."""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import asyncpg
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class RiskRepository:
    """Fetches raw data needed for feature engineering and risk scoring."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_address_stats(self, chain: str, address: str) -> Dict[str, Any]:
        """Fetch basic transaction stats for an address."""
        # Note: In a real system, you'd aggregate this efficiently.
        # This is a safe baseline query against the canonical `transfers` and `transactions` tables.
        query = """
            SELECT 
                COUNT(*) as degree,
                SUM(amount) as transaction_volume,
                SUM(CASE WHEN to_address = $2 THEN amount ELSE 0 END) as total_incoming,
                SUM(CASE WHEN from_address = $2 THEN amount ELSE 0 END) as total_outgoing,
                MIN(tx.timestamp) as first_activity_at,
                MAX(tx.timestamp) as last_activity_at
            FROM transfers tr
            JOIN transactions tx ON tr.transaction_pk = tx.id
            WHERE tr.chain = $1 AND (tr.from_address = $2 OR tr.to_address = $2)
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, chain, address)
            if not row or row["degree"] == 0:
                return {}
            
            return {
                "degree": row["degree"],
                "transaction_volume": float(row["transaction_volume"]) if row["transaction_volume"] else 0.0,
                "total_incoming": float(row["total_incoming"]) if row["total_incoming"] else 0.0,
                "total_outgoing": float(row["total_outgoing"]) if row["total_outgoing"] else 0.0,
                "first_activity_at": row["first_activity_at"],
                "last_activity_at": row["last_activity_at"],
            }

    async def get_cluster_size(self, chain: str, address: str) -> Optional[int]:
        """Fetch the cluster size for the address if it belongs to a cluster."""
        composite_id = f"{chain.lower()}:mainnet:{address.lower()}"
        query = """
            SELECT c.member_count 
            FROM address_clusters c
            JOIN cluster_members m ON c.cluster_id = m.cluster_id
            WHERE m.composite_id = $1
            LIMIT 1
        """
        async with self.pool.acquire() as conn:
            # We catch exceptions just in case the tables aren't populated or don't exist in testing env
            try:
                row = await conn.fetchrow(query, composite_id)
                return row["member_count"] if row else None
            except asyncpg.UndefinedTableError:
                return None

    async def get_inter_hop_velocity_hours(self, chain: str, address: str) -> Optional[float]:
        """Estimate inter-hop velocity (avg time between receiving and sending)."""
        # A simple approximation: (last_activity - first_activity) / degree
        stats = await self.get_address_stats(chain, address)
        if not stats or stats["degree"] < 2:
            return None
            
        first = stats["first_activity_at"]
        last = stats["last_activity_at"]
        if not first or not last:
            return None
            
        diff_hours = (last - first).total_seconds() / 3600.0
        return diff_hours / (stats["degree"] - 1)
