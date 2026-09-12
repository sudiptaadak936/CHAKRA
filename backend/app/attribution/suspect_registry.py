"""
Suspect registry interface — Step 6.

Provides persistent tracking of suspicious entities, exact cross-case matching,
and similar-case structural search using graph fingerprints (pgvector).
"""
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
import asyncpg

logger = logging.getLogger(__name__)


class SuspectRegistryMatchResult(str, Enum):
    """Explicit tri-state result of a suspect registry lookup."""
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    UNKNOWN = "UNKNOWN"


class SuspectRegistryLookupRecord:
    """Outcome of a suspect registry lookup."""
    __slots__ = ("address", "chain", "match_result", "match_id", "reason")

    def __init__(
        self,
        address: str,
        chain: str,
        match_result: SuspectRegistryMatchResult,
        match_id: Optional[str] = None,
        reason: str = "",
    ) -> None:
        self.address = address
        self.chain = chain
        self.match_result = match_result
        self.match_id = match_id
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"SuspectRegistryLookupRecord("
            f"address={self.address!r}, chain={self.chain!r}, "
            f"match_result={self.match_result.value!r}, match_id={self.match_id!r})"
        )


class SuspectRegistryService:
    """
    Step 6 — Suspect Registry.
    
    Handles persistent storage, exact-match lookup, and similarity search
    using pgvector fingerprints.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def lookup(self, address: str, chain: str) -> SuspectRegistryLookupRecord:
        """
        Look up an address in the suspect registry for exact matches.
        Maintains the Step 4.5 boundary invariants.
        """
        if not address or not address.strip():
            raise ValueError("Address is required for suspect registry lookup.")
        if not chain or not chain.strip():
            raise ValueError("Chain is required for suspect registry lookup.")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT registry_id, risk_score, first_seen_case_id FROM suspect_registry WHERE address = $1 AND chain = $2",
                    address, chain
                )
                
                if row:
                    return SuspectRegistryLookupRecord(
                        address=address,
                        chain=chain,
                        match_result=SuspectRegistryMatchResult.MATCH,
                        match_id=str(row["registry_id"]),
                        reason=f"Address found in registry. First seen in case {row['first_seen_case_id']} with score {row['risk_score']}."
                    )
                else:
                    return SuspectRegistryLookupRecord(
                        address=address,
                        chain=chain,
                        match_result=SuspectRegistryMatchResult.NO_MATCH,
                        reason="Address queried but not found in registry."
                    )
        except asyncpg.UndefinedTableError:
            logger.warning("Suspect registry table not found. Returning UNKNOWN.")
            return SuspectRegistryLookupRecord(
                address=address, chain=chain,
                match_result=SuspectRegistryMatchResult.UNKNOWN,
                reason="Database table unavailable."
            )
        except Exception as e:
            logger.error(f"Registry lookup failed: {e}")
            return SuspectRegistryLookupRecord(
                address=address, chain=chain,
                match_result=SuspectRegistryMatchResult.UNKNOWN,
                reason="Registry lookup failed due to internal error."
            )

    async def add_to_registry(
        self, 
        address: str, 
        chain: str, 
        case_id: str, 
        risk_score: float, 
        fingerprint: List[float]
    ) -> Optional[str]:
        """Add a new suspect to the registry with its graph fingerprint."""
        query = """
            INSERT INTO suspect_registry (address, chain, first_seen_case_id, risk_score, graph_fingerprint)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (address, chain) DO UPDATE
            SET risk_score = EXCLUDED.risk_score,
                graph_fingerprint = EXCLUDED.graph_fingerprint,
                updated_at = NOW()
            RETURNING registry_id
        """
        try:
            async with self.pool.acquire() as conn:
                # Convert list to string format for pgvector casting '[val1, val2, ...]'
                vec_str = "[" + ",".join(map(str, fingerprint)) + "]"
                registry_id = await conn.fetchval(query, address, chain, case_id, risk_score, vec_str)
                return str(registry_id)
        except Exception as e:
            logger.error(f"Failed to add to registry: {e}")
            return None

    async def link_case(self, analysis_id: str, registry_id: str) -> bool:
        """Link an analysis case to a registry entry."""
        query = """
            INSERT INTO case_linkages (analysis_id, registry_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, analysis_id, registry_id)
                return True
        except Exception as e:
            logger.error(f"Failed to link case: {e}")
            return False

    async def find_similar_cases(self, fingerprint: List[float], limit: int = 5, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search against registry fingerprints using pgvector.
        Returns the most structurally similar historical cases.
        """
        # vector_cosine_ops calculates cosine distance (1 - cosine similarity)
        # So cosine similarity = 1 - (fingerprint <=> column)
        query = """
            SELECT 
                registry_id, address, chain, first_seen_case_id, risk_score,
                1 - (graph_fingerprint <=> $1::vector) AS similarity
            FROM suspect_registry
            WHERE 1 - (graph_fingerprint <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT $3
        """
        try:
            async with self.pool.acquire() as conn:
                vec_str = "[" + ",".join(map(str, fingerprint)) + "]"
                rows = await conn.fetch(query, vec_str, threshold, limit)
                return [
                    {
                        "registry_id": str(r["registry_id"]),
                        "address": r["address"],
                        "chain": r["chain"],
                        "first_seen_case_id": r["first_seen_case_id"],
                        "risk_score": float(r["risk_score"]),
                        "similarity": float(r["similarity"])
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed similarity search: {e}")
            return []
