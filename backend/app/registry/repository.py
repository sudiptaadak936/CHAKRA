"""CHAKRA Step 6: Suspect Registry — PostgreSQL persistence repository.

Follows the exact same pattern as VASPAttributionRepository (Step 4.1):
  - asyncpg.Pool injection
  - init_schema() reads migration SQL from disk, idempotent
  - save_record() uses ON CONFLICT DO NOTHING for idempotency
  - All lookups return typed SuspectRegistryRecord domain objects

PostgreSQL is the ONLY authoritative source of truth.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import asyncpg

from app.schemas.registry import (
    CrossCaseLinkType,
    MatchedCaseContext,
    RegistryStatus,
    SuspectRegistryRecord,
    SuspectRole,
)

logger = logging.getLogger(__name__)

_MIGRATION_FILE = (
    Path(__file__).parent.parent / "db" / "migrations" / "005_suspect_registry_schema.sql"
)


def _row_to_record(row: asyncpg.Record) -> SuspectRegistryRecord:
    """Hydrate a raw asyncpg Record into a SuspectRegistryRecord domain model."""
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta) if meta else {}
    elif meta is None:
        meta = {}
    return SuspectRegistryRecord(
        record_id=row["record_id"],
        case_id=row["case_id"],
        chain=row["chain"],
        network=row["network"],
        address=row["address"],
        normalized_address=row["normalized_address"],
        composite_id=row["composite_id"],
        suspect_role=SuspectRole(row["suspect_role"]),
        status=RegistryStatus(row["status"]),
        source=row["source"],
        evidence_id=row["evidence_id"],
        reported_by=row["reported_by"],
        entity_label=row["entity_label"],
        external_reference_id=row["external_reference_id"],
        metadata=meta,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SuspectRegistryRepository:
    """PostgreSQL repository for the Suspect Registry (CHAKRA Step 6).

    All methods are idempotent. save_record() silently skips pre-existing records
    via ON CONFLICT DO NOTHING on the natural key
    (chain, network, normalized_address, case_id, evidence_id).
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def init_schema(self) -> None:
        """Apply the suspect registry schema migration idempotently."""
        sql = _MIGRATION_FILE.read_text(encoding="utf-8-sig")
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info(
            "Suspect Registry schema initialized (005_suspect_registry_schema.sql applied)."
        )

    async def save_record(self, record: SuspectRegistryRecord) -> None:
        """Persist a suspect registry record.

        Idempotent on the natural key (chain, network, normalized_address, case_id, evidence_id).
        Silently ignores duplicates via ON CONFLICT DO NOTHING.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO suspect_registry_records (
                    record_id,
                    case_id,
                    chain,
                    network,
                    address,
                    normalized_address,
                    composite_id,
                    suspect_role,
                    status,
                    source,
                    evidence_id,
                    reported_by,
                    entity_label,
                    external_reference_id,
                    metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (chain, network, normalized_address, case_id, evidence_id)
                DO NOTHING
                """,
                record.record_id,
                record.case_id,
                record.chain,
                record.network,
                record.address,
                record.normalized_address,
                record.composite_id,
                record.suspect_role.value,
                record.status.value,
                record.source,
                record.evidence_id,
                record.reported_by,
                record.entity_label,
                record.external_reference_id,
                json.dumps(record.metadata),
            )

    async def lookup_by_composite_id(self, composite_id: str) -> List[SuspectRegistryRecord]:
        """Return all registry records that share a given composite_id.

        composite_id is '{chain}:{network}:{normalized_address}'.
        Returns an empty list if none found (fail-closed).
        """
        if not composite_id or not composite_id.strip():
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    record_id, case_id, chain, network, address,
                    normalized_address, composite_id, suspect_role, status,
                    source, evidence_id, reported_by, entity_label,
                    external_reference_id, metadata, created_at, updated_at
                FROM suspect_registry_records
                WHERE composite_id = $1
                ORDER BY created_at ASC
                """,
                composite_id.strip(),
            )
        return [_row_to_record(r) for r in rows]

    async def lookup_by_case_id(self, case_id: str) -> List[SuspectRegistryRecord]:
        """Return all registry records belonging to a specific case.

        Returns an empty list if none found (fail-closed).
        """
        if not case_id or not case_id.strip():
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    record_id, case_id, chain, network, address,
                    normalized_address, composite_id, suspect_role, status,
                    source, evidence_id, reported_by, entity_label,
                    external_reference_id, metadata, created_at, updated_at
                FROM suspect_registry_records
                WHERE case_id = $1
                ORDER BY created_at ASC
                """,
                case_id.strip(),
            )
        return [_row_to_record(r) for r in rows]

    async def find_cross_case_matches(self, composite_id: str) -> List[SuspectRegistryRecord]:
        """Return all records for composite_id that span >= 2 distinct case_ids.

        If the composite_id is found in only 0 or 1 distinct cases, returns an
        empty list — there is no cross-case signal.
        """
        if not composite_id or not composite_id.strip():
            return []
        async with self.pool.acquire() as conn:
            # First confirm there are >= 2 distinct case_ids for this composite_id
            distinct_cases = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT case_id)
                FROM suspect_registry_records
                WHERE composite_id = $1
                """,
                composite_id.strip(),
            )
            if (distinct_cases or 0) < 2:
                return []

            rows = await conn.fetch(
                """
                SELECT
                    record_id, case_id, chain, network, address,
                    normalized_address, composite_id, suspect_role, status,
                    source, evidence_id, reported_by, entity_label,
                    external_reference_id, metadata, created_at, updated_at
                FROM suspect_registry_records
                WHERE composite_id = $1
                ORDER BY case_id ASC, created_at ASC
                """,
                composite_id.strip(),
            )
        return [_row_to_record(r) for r in rows]

    async def _clear_all_for_testing(self) -> None:
        """Test-only utility to clear the entire registry table."""
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE suspect_registry_records")
