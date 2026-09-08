"""PostgreSQL to Neo4j Graph Projector.

PostgreSQL is the authoritative canonical source of truth.
Neo4j is a deterministic, idempotent, downstream projection.
Strictly read-only with respect to PostgreSQL.

Monetary Integer Representation:
Any monetary value exceeding Neo4j's signed 64-bit integer range cannot be stored
as a native Neo4j integer; it is preserved exactly in amount_str, with amount set to null.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import asyncpg
from neo4j import AsyncDriver

from app.schemas.chain import Chain
from app.graph.models import (
    normalize_address,
    make_address_composite_id,
    make_transaction_composite_id,
    make_transfer_relationship_id,
    make_bitcoin_input_id,
    make_bitcoin_output_id,
    to_neo4j_numeric,
    BatchProjectionResult,
    ProjectionSummary,
    BatchProjectionError,
)

logger = logging.getLogger(__name__)

# Default batch size for PostgreSQL -> Neo4j projection
DEFAULT_BATCH_SIZE = 100


class GraphProjector:
    """Deterministic, idempotent projector from canonical PostgreSQL data into Neo4j."""

    CONSTRAINTS = [
        "CREATE CONSTRAINT address_composite_id IF NOT EXISTS FOR (a:Address) REQUIRE a.composite_id IS UNIQUE",
        "CREATE CONSTRAINT address_identity IF NOT EXISTS FOR (a:Address) REQUIRE (a.chain, a.network, a.normalized_address) IS UNIQUE",
        "CREATE CONSTRAINT transaction_composite_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.composite_id IS UNIQUE",
        "CREATE CONSTRAINT transaction_identity IF NOT EXISTS FOR (t:Transaction) REQUIRE (t.chain, t.network, t.transaction_id) IS UNIQUE",
        "CREATE CONSTRAINT transferred_transfer_id IF NOT EXISTS FOR ()-[r:TRANSFERRED]-() REQUIRE r.transfer_id IS UNIQUE",
        "CREATE CONSTRAINT spent_input_id IF NOT EXISTS FOR ()-[r:SPENT_INPUT]-() REQUIRE r.input_id IS UNIQUE",
        "CREATE CONSTRAINT created_output_id IF NOT EXISTS FOR ()-[r:CREATED_OUTPUT]-() REQUIRE r.output_id IS UNIQUE",
    ]

    def __init__(self, pg_pool: asyncpg.Pool, neo4j_driver: AsyncDriver):
        self.pg_pool = pg_pool
        self.neo4j_driver = neo4j_driver

    async def init_constraints(self) -> None:
        """Create all required Neo4j uniqueness constraints and indexes idempotently."""
        async with self.neo4j_driver.session() as session:
            for query in self.CONSTRAINTS:
                await session.run(query)
        logger.info("Initialized Neo4j constraints and indexes for graph projection.")

    async def clear_graph(self) -> None:
        """Clear all nodes and relationships in Neo4j. Used primarily for test cleanup."""
        async with self.neo4j_driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all graph entities from Neo4j.")

    async def project_batch_data(
        self,
        tx_rows: List[Dict[str, Any]],
        transfer_rows: List[Dict[str, Any]],
        btc_details: Dict[int, Dict[str, Any]],
        btc_vins: Dict[int, List[Dict[str, Any]]],
        btc_vouts: Dict[int, List[Dict[str, Any]]],
        batch_index: int = 0,
    ) -> BatchProjectionResult:
        """Projects a pre-fetched set of canonical records into Neo4j inside a single atomic transaction.

        If any error occurs, the Neo4j transaction is rolled back and a BatchProjectionError is raised.
        """
        if not tx_rows:
            return BatchProjectionResult(
                batch_index=batch_index,
                transactions_count=0,
                transfers_count=0,
                bitcoin_inputs_count=0,
                bitcoin_outputs_count=0,
                start_tx_id=None,
                end_tx_id=None,
                success=True,
            )

        start_tx_id = tx_rows[0]["id"]
        end_tx_id = tx_rows[-1]["id"]

        try:
            addresses_dict: Dict[str, Dict[str, Any]] = {}
            transactions_list: List[Dict[str, Any]] = []
            transfers_list: List[Dict[str, Any]] = []
            bitcoin_inputs_list: List[Dict[str, Any]] = []
            bitcoin_outputs_list: List[Dict[str, Any]] = []

            # 1. Process Transactions
            for tx in tx_rows:
                chain = str(tx["chain"]).lower()
                network = str(tx["network"]).lower()
                tx_id = str(tx["transaction_id"]).strip()
                tx_comp_id = make_transaction_composite_id(chain, network, tx_id)

                fee_val, fee_str = to_neo4j_numeric(tx["fee"])
                native_val, native_str = to_neo4j_numeric(tx["native_value"])

                transactions_list.append({
                    "composite_id": tx_comp_id,
                    "chain": chain,
                    "network": network,
                    "transaction_id": tx_id,
                    "pg_id": tx["id"],
                    "block_number": tx["block_number"],
                    "block_hash": tx["block_hash"],
                    "timestamp": tx["timestamp"].isoformat() if tx["timestamp"] else None,
                    "transaction_type": str(tx["transaction_type"]),
                    "status": str(tx["status"]),
                    "fee": fee_val,
                    "fee_str": fee_str,
                    "fee_asset": tx["fee_asset"],
                    "native_value": native_val,
                    "native_value_str": native_str,
                    "native_value_unit": tx["native_value_unit"],
                })

                # For account-model chains, register from/to addresses as Address nodes if present
                if chain != Chain.BITCOIN.value:
                    if tx.get("from_address"):
                        from_comp = make_address_composite_id(chain, network, tx["from_address"])
                        addresses_dict[from_comp] = {
                            "composite_id": from_comp,
                            "chain": chain,
                            "network": network,
                            "normalized_address": normalize_address(chain, tx["from_address"]),
                            "raw_address": str(tx["from_address"]).strip(),
                        }
                    if tx.get("to_address"):
                        to_comp = make_address_composite_id(chain, network, tx["to_address"])
                        addresses_dict[to_comp] = {
                            "composite_id": to_comp,
                            "chain": chain,
                            "network": network,
                            "normalized_address": normalize_address(chain, tx["to_address"]),
                            "raw_address": str(tx["to_address"]).strip(),
                        }

            # 2. Process Transfers (Account-model chains)
            # CRITICAL: Bitcoin UTXO transactions do NOT project :TRANSFERRED edges.
            # They project :SPENT_INPUT and :CREATED_OUTPUT below.
            for tr in transfer_rows:
                tr_chain = str(tr["chain"]).lower()
                tr_network = str(tr["network"]).lower()
                tx_id = tr.get("transaction_id") or ""
                tr_pk = tr["id"]

                if tr_chain == Chain.BITCOIN.value:
                    # Bitcoin transfers (from_address=None for output, to_address=None for input)
                    # are canonical Transfer representations of UTXO movements.
                    # Do NOT convert them into sender->receiver edges.
                    continue

                from_addr = tr.get("from_address")
                to_addr = tr.get("to_address")

                # In account-model chains, transfers must connect existing endpoints
                if from_addr and to_addr:
                    from_comp = make_address_composite_id(tr_chain, tr_network, from_addr)
                    to_comp = make_address_composite_id(tr_chain, tr_network, to_addr)

                    addresses_dict[from_comp] = {
                        "composite_id": from_comp,
                        "chain": tr_chain,
                        "network": tr_network,
                        "normalized_address": normalize_address(tr_chain, from_addr),
                        "raw_address": str(from_addr).strip(),
                    }
                    addresses_dict[to_comp] = {
                        "composite_id": to_comp,
                        "chain": tr_chain,
                        "network": tr_network,
                        "normalized_address": normalize_address(tr_chain, to_addr),
                        "raw_address": str(to_addr).strip(),
                    }

                    amt_val, amt_str = to_neo4j_numeric(tr["amount"])
                    transfer_rel_id = make_transfer_relationship_id(
                        tr_chain, tr_network, tx_id, tr_pk
                    )

                    transfers_list.append({
                        "transfer_id": transfer_rel_id,
                        "from_composite_id": from_comp,
                        "to_composite_id": to_comp,
                        "chain": tr_chain,
                        "network": tr_network,
                        "transaction_id": tx_id,
                        "asset_type": str(tr["asset_type"]),
                        "asset_symbol": str(tr["asset_symbol"]),
                        "asset_contract": tr["asset_contract"],
                        "amount": amt_val,
                        "amount_str": amt_str,
                        "amount_unit": tr["amount_unit"],
                        "decimals": tr["decimals"],
                        "pg_transfer_id": tr_pk,
                        "pg_transaction_pk": tr["transaction_pk"],
                    })

            # 3. Process Bitcoin Explicit UTXO Details
            for tx in tx_rows:
                chain = str(tx["chain"]).lower()
                if chain != Chain.BITCOIN.value:
                    continue

                tx_pk = tx["id"]
                tx_id = str(tx["transaction_id"]).strip()
                network = str(tx["network"]).lower()
                tx_comp_id = make_transaction_composite_id(chain, network, tx_id)

                detail = btc_details.get(tx_pk)
                if detail:
                    detail_pk = detail["id"]
                    vins = btc_vins.get(detail_pk, [])
                    vouts = btc_vouts.get(detail_pk, [])

                    # Project inputs: (:Address)-[:SPENT_INPUT]->(:Transaction)
                    for vin in vins:
                        vin_addr = vin.get("address")
                        if vin_addr:
                            addr_comp = make_address_composite_id(chain, network, vin_addr)
                            norm_addr = normalize_address(chain, vin_addr)
                            raw_addr = str(vin_addr).strip()
                        elif vin.get("coinbase"):
                            addr_comp = f"bitcoin:{network}:coinbase"
                            norm_addr = "(coinbase)"
                            raw_addr = "(coinbase)"
                        else:
                            addr_comp = f"bitcoin:{network}:addressless_vin:{vin['id']}"
                            norm_addr = f"(addressless_vin:{vin['id']})"
                            raw_addr = f"(addressless_vin:{vin['id']})"

                        addresses_dict[addr_comp] = {
                            "composite_id": addr_comp,
                            "chain": chain,
                            "network": network,
                            "normalized_address": norm_addr,
                            "raw_address": raw_addr,
                        }
                        val_num, val_str = to_neo4j_numeric(vin["value_sat"])
                        input_rel_id = make_bitcoin_input_id(
                            network, tx_id, vin.get("txid"), vin.get("vout"), vin["id"]
                        )
                        bitcoin_inputs_list.append({
                            "input_id": input_rel_id,
                            "from_composite_id": addr_comp,
                            "tx_composite_id": tx_comp_id,
                            "spent_txid": vin.get("txid"),
                            "spent_vout": vin.get("vout"),
                            "value_sat": val_num,
                            "value_sat_str": val_str,
                            "coinbase": vin.get("coinbase", False),
                            "pg_vin_id": vin["id"],
                            "chain": chain,
                            "network": network,
                        })

                    # Project outputs: (:Transaction)-[:CREATED_OUTPUT]->(:Address)
                    for vout in vouts:
                        vout_addr = vout.get("address")
                        if vout_addr:
                            addr_comp = make_address_composite_id(chain, network, vout_addr)
                            norm_addr = normalize_address(chain, vout_addr)
                            raw_addr = str(vout_addr).strip()
                        elif vout.get("script_type") == "op_return":
                            addr_comp = f"bitcoin:{network}:op_return:{tx_id}:{vout['n']}"
                            norm_addr = f"(op_return:{vout['n']})"
                            raw_addr = f"(op_return:{vout['n']})"
                        else:
                            addr_comp = f"bitcoin:{network}:addressless_vout:{vout['id']}"
                            norm_addr = f"(addressless_vout:{vout['id']})"
                            raw_addr = f"(addressless_vout:{vout['id']})"

                        addresses_dict[addr_comp] = {
                            "composite_id": addr_comp,
                            "chain": chain,
                            "network": network,
                            "normalized_address": norm_addr,
                            "raw_address": raw_addr,
                        }
                        val_num, val_str = to_neo4j_numeric(vout["value_sat"])
                        output_rel_id = make_bitcoin_output_id(
                            network, tx_id, vout["n"], vout["id"]
                        )
                        bitcoin_outputs_list.append({
                            "output_id": output_rel_id,
                            "tx_composite_id": tx_comp_id,
                            "to_composite_id": addr_comp,
                            "n": vout["n"],
                            "value_sat": val_num,
                            "value_sat_str": val_str,
                            "script_type": vout.get("script_type"),
                            "pg_vout_id": vout["id"],
                            "chain": chain,
                            "network": network,
                        })

            # 4. Atomic Neo4j Transaction Execution
            async with self.neo4j_driver.session() as session:
                neo4j_tx = await session.begin_transaction()
                try:
                    # Merge Address nodes
                    if addresses_dict:
                        await neo4j_tx.run(
                            """
                            UNWIND $addresses AS addr
                            MERGE (a:Address {composite_id: addr.composite_id})
                            ON CREATE SET
                                a.chain = addr.chain,
                                a.network = addr.network,
                                a.normalized_address = addr.normalized_address,
                                a.raw_address = addr.raw_address
                            ON MATCH SET
                                a.raw_address = addr.raw_address
                            """,
                            addresses=list(addresses_dict.values()),
                        )

                    # Merge Transaction nodes
                    if transactions_list:
                        await neo4j_tx.run(
                            """
                            UNWIND $transactions AS tx
                            MERGE (t:Transaction {composite_id: tx.composite_id})
                            ON CREATE SET
                                t.chain = tx.chain,
                                t.network = tx.network,
                                t.transaction_id = tx.transaction_id,
                                t.pg_id = tx.pg_id,
                                t.block_number = tx.block_number,
                                t.block_hash = tx.block_hash,
                                t.timestamp = tx.timestamp,
                                t.transaction_type = tx.transaction_type,
                                t.status = tx.status,
                                t.fee = tx.fee,
                                t.fee_str = tx.fee_str,
                                t.fee_asset = tx.fee_asset,
                                t.native_value = tx.native_value,
                                t.native_value_str = tx.native_value_str,
                                t.native_value_unit = tx.native_value_unit
                            ON MATCH SET
                                t.pg_id = tx.pg_id,
                                t.block_number = tx.block_number,
                                t.block_hash = tx.block_hash,
                                t.timestamp = tx.timestamp,
                                t.transaction_type = tx.transaction_type,
                                t.status = tx.status,
                                t.fee = tx.fee,
                                t.fee_str = tx.fee_str,
                                t.fee_asset = tx.fee_asset,
                                t.native_value = tx.native_value,
                                t.native_value_str = tx.native_value_str,
                                t.native_value_unit = tx.native_value_unit
                            """,
                            transactions=transactions_list,
                        )

                    # Merge Transfer relationships (:TRANSFERRED)
                    if transfers_list:
                        await neo4j_tx.run(
                            """
                            UNWIND $transfers AS tr
                            MATCH (from:Address {composite_id: tr.from_composite_id})
                            MATCH (to:Address {composite_id: tr.to_composite_id})
                            MERGE (from)-[r:TRANSFERRED {transfer_id: tr.transfer_id}]->(to)
                            ON CREATE SET
                                r.chain = tr.chain,
                                r.network = tr.network,
                                r.transaction_id = tr.transaction_id,
                                r.asset_type = tr.asset_type,
                                r.asset_symbol = tr.asset_symbol,
                                r.asset_contract = tr.asset_contract,
                                r.amount = tr.amount,
                                r.amount_str = tr.amount_str,
                                r.amount_unit = tr.amount_unit,
                                r.decimals = tr.decimals,
                                r.pg_transfer_id = tr.pg_transfer_id,
                                r.pg_transaction_pk = tr.pg_transaction_pk
                            ON MATCH SET
                                r.asset_type = tr.asset_type,
                                r.asset_symbol = tr.asset_symbol,
                                r.asset_contract = tr.asset_contract,
                                r.amount = tr.amount,
                                r.amount_str = tr.amount_str,
                                r.amount_unit = tr.amount_unit,
                                r.decimals = tr.decimals,
                                r.pg_transfer_id = tr.pg_transfer_id,
                                r.pg_transaction_pk = tr.pg_transaction_pk
                            """,
                            transfers=transfers_list,
                        )

                    # Merge Bitcoin SPENT_INPUT relationships
                    if bitcoin_inputs_list:
                        await neo4j_tx.run(
                            """
                            UNWIND $inputs AS inp
                            MATCH (addr:Address {composite_id: inp.from_composite_id})
                            MATCH (tx:Transaction {composite_id: inp.tx_composite_id})
                            MERGE (addr)-[r:SPENT_INPUT {input_id: inp.input_id}]->(tx)
                            ON CREATE SET
                                r.spent_txid = inp.spent_txid,
                                r.spent_vout = inp.spent_vout,
                                r.value_sat = inp.value_sat,
                                r.value_sat_str = inp.value_sat_str,
                                r.coinbase = inp.coinbase,
                                r.pg_vin_id = inp.pg_vin_id,
                                r.chain = inp.chain,
                                r.network = inp.network
                            ON MATCH SET
                                r.spent_txid = inp.spent_txid,
                                r.spent_vout = inp.spent_vout,
                                r.value_sat = inp.value_sat,
                                r.value_sat_str = inp.value_sat_str,
                                r.coinbase = inp.coinbase,
                                r.pg_vin_id = inp.pg_vin_id,
                                r.chain = inp.chain,
                                r.network = inp.network
                            """,
                            inputs=bitcoin_inputs_list,
                        )

                    # Merge Bitcoin CREATED_OUTPUT relationships
                    if bitcoin_outputs_list:
                        await neo4j_tx.run(
                            """
                            UNWIND $outputs AS outp
                            MATCH (tx:Transaction {composite_id: outp.tx_composite_id})
                            MATCH (addr:Address {composite_id: outp.to_composite_id})
                            MERGE (tx)-[r:CREATED_OUTPUT {output_id: outp.output_id}]->(addr)
                            ON CREATE SET
                                r.n = outp.n,
                                r.value_sat = outp.value_sat,
                                r.value_sat_str = outp.value_sat_str,
                                r.script_type = outp.script_type,
                                r.pg_vout_id = outp.pg_vout_id,
                                r.chain = outp.chain,
                                r.network = outp.network
                            ON MATCH SET
                                r.n = outp.n,
                                r.value_sat = outp.value_sat,
                                r.value_sat_str = outp.value_sat_str,
                                r.script_type = outp.script_type,
                                r.pg_vout_id = outp.pg_vout_id,
                                r.chain = outp.chain,
                                r.network = outp.network
                            """,
                            outputs=bitcoin_outputs_list,
                        )

                    await neo4j_tx.commit()
                except Exception:
                    await neo4j_tx.rollback()
                    raise
                finally:
                    await neo4j_tx.close()

            return BatchProjectionResult(
                batch_index=batch_index,
                transactions_count=len(transactions_list),
                transfers_count=len(transfers_list),
                bitcoin_inputs_count=len(bitcoin_inputs_list),
                bitcoin_outputs_count=len(bitcoin_outputs_list),
                start_tx_id=start_tx_id,
                end_tx_id=end_tx_id,
                success=True,
            )

        except Exception as e:
            logger.error(
                f"Batch {batch_index} (tx id {start_tx_id}..{end_tx_id}) failed: {e}. Rolled back."
            )
            raise BatchProjectionError(
                batch_index=batch_index,
                start_tx_id=start_tx_id,
                end_tx_id=end_tx_id,
                message=str(e),
                original_error=e,
            ) from e

    async def project_batch(
        self, start_id: int = 0, batch_size: int = DEFAULT_BATCH_SIZE, batch_index: int = 0
    ) -> BatchProjectionResult:
        """Fetch a bounded batch of canonical PostgreSQL records and project them into Neo4j."""
        async with self.pg_pool.acquire() as conn:
            # 1. Fetch bounded transactions by keyset pagination (read-only SELECT)
            tx_records = await conn.fetch(
                """
                SELECT * FROM transactions
                WHERE id > $1
                ORDER BY id ASC
                LIMIT $2
                """,
                start_id,
                batch_size,
            )

            if not tx_records:
                return BatchProjectionResult(
                    batch_index=batch_index,
                    transactions_count=0,
                    transfers_count=0,
                    bitcoin_inputs_count=0,
                    bitcoin_outputs_count=0,
                    start_tx_id=None,
                    end_tx_id=None,
                    success=True,
                )

            tx_rows = [dict(r) for r in tx_records]
            tx_pks = [r["id"] for r in tx_rows]
            txid_by_pk = {r["id"]: r["transaction_id"] for r in tx_rows}

            # 2. Fetch associated transfers
            transfer_records = await conn.fetch(
                """
                SELECT * FROM transfers
                WHERE transaction_pk = ANY($1)
                ORDER BY id ASC
                """,
                tx_pks,
            )
            transfer_rows = []
            for tr in transfer_records:
                d = dict(tr)
                d["transaction_id"] = txid_by_pk.get(d["transaction_pk"], "")
                transfer_rows.append(d)

            # 3. Fetch Bitcoin details if present
            btc_detail_records = await conn.fetch(
                """
                SELECT * FROM bitcoin_transaction_details
                WHERE transaction_pk = ANY($1)
                """,
                tx_pks,
            )
            btc_details = {r["transaction_pk"]: dict(r) for r in btc_detail_records}

            btc_vins: Dict[int, List[Dict[str, Any]]] = {}
            btc_vouts: Dict[int, List[Dict[str, Any]]] = {}

            if btc_details:
                detail_pks = [d["id"] for d in btc_details.values()]
                vin_records = await conn.fetch(
                    """
                    SELECT * FROM bitcoin_vins
                    WHERE detail_pk = ANY($1)
                    ORDER BY id ASC
                    """,
                    detail_pks,
                )
                for vin in vin_records:
                    btc_vins.setdefault(vin["detail_pk"], []).append(dict(vin))

                vout_records = await conn.fetch(
                    """
                    SELECT * FROM bitcoin_vouts
                    WHERE detail_pk = ANY($1)
                    ORDER BY id ASC
                    """,
                    detail_pks,
                )
                for vout in vout_records:
                    btc_vouts.setdefault(vout["detail_pk"], []).append(dict(vout))

        # Project into Neo4j
        return await self.project_batch_data(
            tx_rows=tx_rows,
            transfer_rows=transfer_rows,
            btc_details=btc_details,
            btc_vins=btc_vins,
            btc_vouts=btc_vouts,
            batch_index=batch_index,
        )

    async def project_all(
        self, batch_size: int = DEFAULT_BATCH_SIZE, on_error: str = "abort"
    ) -> ProjectionSummary:
        """Projects all canonical PostgreSQL transactions into Neo4j using bounded pagination.

        Args:
            batch_size: Number of transactions to process per batch.
            on_error: 'abort' to stop immediately on batch failure (default),
                      'continue' to record failure and proceed to subsequent batches.
        """
        summary = ProjectionSummary()
        last_id = 0
        batch_index = 0

        while True:
            try:
                result = await self.project_batch(
                    start_id=last_id, batch_size=batch_size, batch_index=batch_index
                )
            except BatchProjectionError as err:
                if on_error == "abort":
                    summary.success = False
                    raise err
                else:
                    logger.error(f"Batch failed but continuing: {err}")
                    summary.success = False
                    if err.end_tx_id is not None:
                        last_id = err.end_tx_id
                    batch_index += 1
                    continue

            if result.transactions_count == 0:
                break

            summary.total_batches += 1
            summary.total_transactions += result.transactions_count
            summary.total_transfers += result.transfers_count
            summary.total_bitcoin_inputs += result.bitcoin_inputs_count
            summary.total_bitcoin_outputs += result.bitcoin_outputs_count
            summary.batch_results.append(result)

            if result.end_tx_id is not None:
                last_id = result.end_tx_id
            batch_index += 1

        return summary
