import json
import logging
from pathlib import Path
from typing import List, Optional
import asyncpg
from app.schemas.chain import Chain
from app.attribution.models import VASPAddressRecord
from app.graph.models import normalize_address

logger = logging.getLogger(__name__)

_MIGRATION_FILE = Path(__file__).parent.parent / "db" / "migrations" / "004_vasp_attribution_schema.sql"

class VASPAttributionRepository:
    """
    PostgreSQL repository for the VASP / Exchange Attribution Registry.
    """
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_schema(self) -> None:
        """Apply the VASP attribution schema migration idempotently."""
        sql = _MIGRATION_FILE.read_text(encoding="utf-8-sig")
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("VASP Attribution schema initialized (004_vasp_attribution_schema.sql applied).")

    async def save_record(self, record: VASPAddressRecord) -> None:
        """
        Saves a VASP attribution record.
        Idempotent on (address, chain, vasp_name, source).
        """
        async with self.pool.acquire() as conn:
            normalized_address = normalize_address(record.chain.value, record.address)
            await conn.execute(
                """
                INSERT INTO vasp_attribution_registry (
                    address, chain, vasp_name, service_type, source, evidence_id, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (address, chain, vasp_name, source) DO UPDATE
                SET service_type = EXCLUDED.service_type,
                    evidence_id = EXCLUDED.evidence_id,
                    metadata = EXCLUDED.metadata
                """,
                normalized_address,
                record.chain.value,
                record.vasp_name,
                record.service_type,
                record.source,
                record.evidence_id,
                json.dumps(record.metadata)
            )

    async def lookup(self, address: str, chain: str) -> List[VASPAddressRecord]:
        """
        Finds all VASP attribution records for a given normalized address on a given chain.
        """
        async with self.pool.acquire() as conn:
            normalized_address = normalize_address(chain, address)
            rows = await conn.fetch(
                """
                SELECT address, chain, vasp_name, service_type, source, evidence_id, metadata
                FROM vasp_attribution_registry
                WHERE chain = $1 AND address = $2
                """,
                chain, normalized_address
            )
            
            records = []
            for row in rows:
                meta = json.loads(row['metadata']) if row['metadata'] else {}
                records.append(VASPAddressRecord(
                    address=row['address'],
                    chain=Chain(row['chain']),
                    vasp_name=row['vasp_name'],
                    service_type=row['service_type'],
                    source=row['source'],
                    evidence_id=row['evidence_id'],
                    metadata=meta
                ))
            return records

    async def _clear_all_for_testing(self) -> None:
        """Test-only utility to clear the registry."""
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE vasp_attribution_registry")
