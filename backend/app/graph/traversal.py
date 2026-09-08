"""CHAKRA Step 2B: Deterministic Money-Flow Traversal.

Read-only consumer of the Neo4j graph projected by Step 2A.

Supported models:
- Account-model chains (EVM, TRON, Solana):
  Traverses explicit (:Address)-[:TRANSFERRED]->(:Address) relationships.
- Bitcoin (UTXO model):
  Traverses explicit UTXO spend transitions:
  (:Address)-[:SPENT_INPUT]->(:Transaction)-[:CREATED_OUTPUT]->(:Address)
  Never creates or fabricates synthetic Address -> Address TRANSFERRED edges.
  Never infers entity clustering, co-input ownership, or 1-to-1 input-to-output pairings.

Design constraints:
- Strictly read-only: never writes to Neo4j or PostgreSQL.
- Bounded traversal: max_hops, max_nodes, max_edges (all mandatory and enforced).
- Simple-path semantics: nodes/transitions in the active path stack cannot be revisited.
- Deterministic ordering by canonical graph identifiers (transfer_id, input_id, output_id).
- Exact integer / satoshi preservation: no floats, authoritative string representations.
- No analytics, risk scoring, entity attribution, or laundering detection.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from neo4j import AsyncDriver
from pydantic import BaseModel, Field

from app.graph.models import normalize_address, make_address_composite_id
from app.schemas.chain import Chain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HOPS_CEILING: int = 10
MAX_NODES_CEILING: int = 1000
MAX_EDGES_CEILING: int = 10000

_BITCOIN_CHAIN: str = Chain.BITCOIN.value  # "bitcoin"


# ---------------------------------------------------------------------------
# Result models — Account Model
# ---------------------------------------------------------------------------


class TraversalEdge(BaseModel):
    """A single TRANSFERRED relationship traversed from one Address to another."""

    transfer_id: str
    from_composite_id: str
    to_composite_id: str
    chain: str
    network: str
    transaction_id: str
    asset_type: str
    asset_symbol: str
    asset_contract: Optional[str] = None
    amount: Optional[int] = None
    """Neo4j int when representable; None for values beyond signed 64-bit."""
    amount_str: Optional[str] = None
    """Always exact canonical decimal string; authoritative when amount is None."""
    amount_unit: Optional[str] = None
    decimals: Optional[int] = None
    pg_transfer_id: Optional[int] = None
    pg_transaction_pk: Optional[int] = None


class TraversalNode(BaseModel):
    """An Address node encountered during traversal."""

    composite_id: str
    chain: str
    network: str
    normalized_address: str
    raw_address: str


# ---------------------------------------------------------------------------
# Result models — Bitcoin UTXO Model
# ---------------------------------------------------------------------------


class SpentInputEdge(BaseModel):
    """An explicit SPENT_INPUT relationship: (:Address)-[:SPENT_INPUT]->(:Transaction)."""

    input_id: str
    from_composite_id: str
    tx_composite_id: str
    spent_txid: Optional[str] = None
    spent_vout: Optional[int] = None
    value_sat: Optional[int] = None
    """Exact satoshi value when representable in signed 64-bit."""
    value_sat_str: Optional[str] = None
    """Authoritative exact string representation of satoshi value."""
    coinbase: bool = False
    pg_vin_id: Optional[int] = None
    chain: str
    network: str


class TransactionNode(BaseModel):
    """An explicit Transaction node encountered during Bitcoin UTXO traversal."""

    composite_id: str
    chain: str
    network: str
    transaction_id: str
    pg_id: Optional[int] = None
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    timestamp: Optional[str] = None
    transaction_type: Optional[str] = None
    status: Optional[str] = None
    fee: Optional[int] = None
    fee_str: Optional[str] = None
    fee_asset: Optional[str] = None
    native_value: Optional[int] = None
    native_value_str: Optional[str] = None
    native_value_unit: Optional[str] = None


class CreatedOutputEdge(BaseModel):
    """An explicit CREATED_OUTPUT relationship: (:Transaction)-[:CREATED_OUTPUT]->(:Address)."""

    output_id: str
    tx_composite_id: str
    to_composite_id: str
    n: int
    value_sat: Optional[int] = None
    """Exact satoshi value when representable in signed 64-bit."""
    value_sat_str: Optional[str] = None
    """Authoritative exact string representation of satoshi value."""
    script_type: Optional[str] = None
    pg_vout_id: Optional[int] = None
    chain: str
    network: str


class UtxoSpendStep(BaseModel):
    """An explicit UTXO spend transition.

    Represents:
        from_address --SPENT_INPUT--> transaction --CREATED_OUTPUT--> to_address
    Never asserts a direct Address -> Address relationship.
    Does not imply that the entire output amount originated solely from the input address.
    """

    from_address: TraversalNode
    spent_input: SpentInputEdge
    transaction: TransactionNode
    created_output: CreatedOutputEdge
    to_address: TraversalNode


# ---------------------------------------------------------------------------
# Path & Result models
# ---------------------------------------------------------------------------


class TraversalPath(BaseModel):
    """A single simple path through the graph starting from the origin address."""

    nodes: List[TraversalNode]
    """Ordered list of Address nodes visited along this path."""

    edges: List[TraversalEdge] = Field(default_factory=list)
    """Ordered list of TRANSFERRED edges traversed (account-model only; empty for Bitcoin)."""

    hops: int
    """Number of transitions traversed (TRANSFERRED edges for account-model;
    UTXO spend transitions for Bitcoin)."""

    utxo_steps: List[UtxoSpendStep] = Field(default_factory=list)
    """Explicit UTXO spend transitions (Bitcoin only; empty for account-model)."""


class TraversalResult(BaseModel):
    """Complete result of a bounded money-flow traversal."""

    start_chain: str
    start_network: str
    start_normalized_address: str
    start_composite_id: str

    paths: List[TraversalPath] = Field(default_factory=list)
    total_paths: int = 0
    total_edges_visited: int = 0
    total_nodes_visited: int = 0

    max_hops: int
    max_nodes: int
    max_edges: int

    truncated: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------


class TraversalParameterError(ValueError):
    """Raised when traversal parameters are invalid or out of bounds."""


# ---------------------------------------------------------------------------
# Traversal service
# ---------------------------------------------------------------------------


class MoneyFlowTraversal:
    """Read-only deterministic money-flow traversal over the Step 2A Neo4j graph.

    Supports:
    1. Account-model chains (EVM, TRON, Solana) via (:Address)-[:TRANSFERRED]->(:Address).
    2. Bitcoin (UTXO model) via explicit (:Address)-[:SPENT_INPUT]->(:Transaction)-[:CREATED_OUTPUT]->(:Address).

    Strictly read-only with respect to both Neo4j and PostgreSQL.
    """

    def __init__(self, neo4j_driver: AsyncDriver) -> None:
        self.neo4j_driver = neo4j_driver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def traverse(
        self,
        chain: str,
        network: str,
        address: str,
        max_hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> TraversalResult:
        """Traverse money-flow paths starting from a given address.

        Args:
            chain:     Blockchain chain identifier (e.g. "evm", "tron", "solana", "bitcoin").
            network:   Specific network (e.g. "ethereum-mainnet", "bitcoin-mainnet").
            address:   Starting blockchain address (Step 2A normalization applied).
            max_hops:  Maximum edge/spend transitions per path. [1, MAX_HOPS_CEILING].
            max_nodes: Maximum total distinct nodes visited before truncation. [1, MAX_NODES_CEILING].
            max_edges: Maximum total edge visits before truncation. [1, MAX_EDGES_CEILING].

        Returns:
            TraversalResult with all discovered simple paths within limits.
        """
        chain_lower = chain.lower()
        network_lower = network.lower()

        try:
            self._validate_params(chain_lower, max_hops, max_nodes, max_edges)
        except TraversalParameterError as exc:
            return TraversalResult(
                start_chain=chain_lower,
                start_network=network_lower,
                start_normalized_address="",
                start_composite_id="",
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                error=str(exc),
            )

        if chain_lower == _BITCOIN_CHAIN:
            return await self._traverse_bitcoin(
                chain=chain_lower,
                network=network_lower,
                address=address,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        else:
            return await self._traverse_account(
                chain=chain_lower,
                network=network_lower,
                address=address,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_params(chain: str, max_hops: int, max_nodes: int, max_edges: int) -> None:
        supported_chains = {c.value for c in Chain}
        if chain not in supported_chains:
            raise TraversalParameterError(
                f"Unsupported chain: '{chain}'. Supported chains: {sorted(supported_chains)}"
            )
        if not (1 <= max_hops <= MAX_HOPS_CEILING):
            raise TraversalParameterError(
                f"max_hops must be between 1 and {MAX_HOPS_CEILING}, got {max_hops}."
            )
        if not (1 <= max_nodes <= MAX_NODES_CEILING):
            raise TraversalParameterError(
                f"max_nodes must be between 1 and {MAX_NODES_CEILING}, got {max_nodes}."
            )
        if not (1 <= max_edges <= MAX_EDGES_CEILING):
            raise TraversalParameterError(
                f"max_edges must be between 1 and {MAX_EDGES_CEILING}, got {max_edges}."
            )

    # ------------------------------------------------------------------
    # Common Graph Access
    # ------------------------------------------------------------------

    async def _fetch_address_node(
        self,
        composite_id: str,
        chain: Optional[str] = None,
        network: Optional[str] = None,
        normalized_address: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single Address node by composite_id or (chain, network, normalized_address)."""
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Address)
                WHERE a.composite_id = $cid
                   OR ($chain IS NOT NULL AND a.chain = $chain AND a.network = $network AND a.normalized_address = $norm)
                RETURN
                    a.composite_id       AS composite_id,
                    a.chain              AS chain,
                    a.network            AS network,
                    a.normalized_address AS normalized_address,
                    a.raw_address        AS raw_address
                LIMIT 1
                """,
                cid=composite_id,
                chain=chain,
                network=network,
                norm=normalized_address,
            )
            record = await result.single()
        if record is None:
            return None
        return dict(record)


    # ------------------------------------------------------------------
    # Account-Model Traversal (EVM, TRON, Solana)
    # ------------------------------------------------------------------

    async def _traverse_account(
        self,
        chain: str,
        network: str,
        address: str,
        max_hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> TraversalResult:
        norm_addr = normalize_address(chain, address)
        start_composite_id = make_address_composite_id(chain, network, address)

        start_node_data = await self._fetch_address_node(
            composite_id=start_composite_id,
            chain=chain,
            network=network,
            normalized_address=norm_addr,
        )
        if start_node_data is None:
            return TraversalResult(
                start_chain=chain,
                start_network=network,
                start_normalized_address=norm_addr,
                start_composite_id=start_composite_id,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                error=f"Start address not found in graph: {start_composite_id}",
            )

        start_composite_id = start_node_data["composite_id"]


        start_traversal_node = TraversalNode(
            composite_id=start_node_data["composite_id"],
            chain=start_node_data["chain"],
            network=start_node_data["network"],
            normalized_address=start_node_data["normalized_address"],
            raw_address=start_node_data["raw_address"],
        )

        paths: List[TraversalPath] = []
        visited_in_path: List[str] = [start_composite_id]
        nodes_seen: Set[str] = {start_composite_id}
        edges_visited_counter: List[int] = [0]
        truncated: List[bool] = [False]

        await self._dfs_account(
            current_node=start_traversal_node,
            current_path_nodes=[start_traversal_node],
            current_path_edges=[],
            depth=0,
            visited_in_path=visited_in_path,
            nodes_seen=nodes_seen,
            edges_visited_counter=edges_visited_counter,
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            paths=paths,
            truncated=truncated,
        )

        return TraversalResult(
            start_chain=chain,
            start_network=network,
            start_normalized_address=norm_addr,
            start_composite_id=start_composite_id,
            paths=paths,
            total_paths=len(paths),
            total_edges_visited=edges_visited_counter[0],
            total_nodes_visited=len(nodes_seen),
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            truncated=truncated[0],
        )

    async def _fetch_outgoing_transfers(self, composite_id: str) -> List[Dict[str, Any]]:
        """Fetch all outgoing TRANSFERRED edges ordered deterministically by transfer_id."""
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Address {composite_id: $cid})-[r:TRANSFERRED]->(b:Address)
                RETURN
                    r.transfer_id        AS transfer_id,
                    a.composite_id       AS from_composite_id,
                    b.composite_id       AS to_composite_id,
                    r.chain              AS chain,
                    r.network            AS network,
                    r.transaction_id     AS transaction_id,
                    r.asset_type         AS asset_type,
                    r.asset_symbol       AS asset_symbol,
                    r.asset_contract     AS asset_contract,
                    r.amount             AS amount,
                    r.amount_str         AS amount_str,
                    r.amount_unit        AS amount_unit,
                    r.decimals           AS decimals,
                    r.pg_transfer_id     AS pg_transfer_id,
                    r.pg_transaction_pk  AS pg_transaction_pk,
                    b.composite_id       AS dest_composite_id,
                    b.chain              AS dest_chain,
                    b.network            AS dest_network,
                    b.normalized_address AS dest_normalized_address,
                    b.raw_address        AS dest_raw_address
                ORDER BY r.transfer_id ASC
                """,
                cid=composite_id,
            )
            records = await result.data()
        return records

    async def _dfs_account(
        self,
        current_node: TraversalNode,
        current_path_nodes: List[TraversalNode],
        current_path_edges: List[TraversalEdge],
        depth: int,
        visited_in_path: List[str],
        nodes_seen: Set[str],
        edges_visited_counter: List[int],
        max_hops: int,
        max_nodes: int,
        max_edges: int,
        paths: List[TraversalPath],
        truncated: List[bool],
    ) -> None:
        if truncated[0]:
            return

        if depth > 0:
            paths.append(
                TraversalPath(
                    nodes=list(current_path_nodes),
                    edges=list(current_path_edges),
                    hops=depth,
                )
            )

        if depth >= max_hops:
            return

        outgoing = await self._fetch_outgoing_transfers(current_node.composite_id)

        for edge_record in outgoing:
            if truncated[0]:
                return

            if edges_visited_counter[0] >= max_edges:
                truncated[0] = True
                return

            edges_visited_counter[0] += 1

            dest_cid: str = edge_record["dest_composite_id"]

            if dest_cid in visited_in_path:
                continue

            if dest_cid not in nodes_seen:
                if len(nodes_seen) >= max_nodes:
                    truncated[0] = True
                    return
                nodes_seen.add(dest_cid)

            dest_node = TraversalNode(
                composite_id=dest_cid,
                chain=edge_record["dest_chain"],
                network=edge_record["dest_network"],
                normalized_address=edge_record["dest_normalized_address"],
                raw_address=edge_record["dest_raw_address"],
            )

            traversal_edge = TraversalEdge(
                transfer_id=edge_record["transfer_id"],
                from_composite_id=edge_record["from_composite_id"],
                to_composite_id=edge_record["to_composite_id"],
                chain=edge_record["chain"],
                network=edge_record["network"],
                transaction_id=edge_record["transaction_id"],
                asset_type=edge_record["asset_type"],
                asset_symbol=edge_record["asset_symbol"],
                asset_contract=edge_record.get("asset_contract"),
                amount=edge_record.get("amount"),
                amount_str=edge_record.get("amount_str"),
                amount_unit=edge_record.get("amount_unit"),
                decimals=edge_record.get("decimals"),
                pg_transfer_id=edge_record.get("pg_transfer_id"),
                pg_transaction_pk=edge_record.get("pg_transaction_pk"),
            )

            visited_in_path.append(dest_cid)
            await self._dfs_account(
                current_node=dest_node,
                current_path_nodes=current_path_nodes + [dest_node],
                current_path_edges=current_path_edges + [traversal_edge],
                depth=depth + 1,
                visited_in_path=visited_in_path,
                nodes_seen=nodes_seen,
                edges_visited_counter=edges_visited_counter,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                paths=paths,
                truncated=truncated,
            )
            visited_in_path.pop()

    # ------------------------------------------------------------------
    # Bitcoin UTXO Traversal
    # ------------------------------------------------------------------

    async def _traverse_bitcoin(
        self,
        chain: str,
        network: str,
        address: str,
        max_hops: int,
        max_nodes: int,
        max_edges: int,
    ) -> TraversalResult:
        norm_addr = normalize_address(chain, address)
        start_composite_id = make_address_composite_id(chain, network, address)

        start_node_data = await self._fetch_address_node(
            composite_id=start_composite_id,
            chain=chain,
            network=network,
            normalized_address=norm_addr,
        )
        if start_node_data is None:
            return TraversalResult(
                start_chain=chain,
                start_network=network,
                start_normalized_address=norm_addr,
                start_composite_id=start_composite_id,
                max_hops=max_hops,
                max_nodes=max_nodes,
                max_edges=max_edges,
                error=f"Start address not found in graph: {start_composite_id}",
            )

        start_composite_id = start_node_data["composite_id"]


        start_traversal_node = TraversalNode(
            composite_id=start_node_data["composite_id"],
            chain=start_node_data["chain"],
            network=start_node_data["network"],
            normalized_address=start_node_data["normalized_address"],
            raw_address=start_node_data["raw_address"],
        )

        paths: List[TraversalPath] = []
        visited_in_path: List[str] = [start_composite_id]
        nodes_seen: Set[str] = {start_composite_id}
        edges_visited_counter: List[int] = [0]
        truncated: List[bool] = [False]

        await self._dfs_bitcoin(
            current_address=start_traversal_node,
            current_path_addresses=[start_traversal_node],
            current_utxo_steps=[],
            depth=0,
            visited_in_path=visited_in_path,
            nodes_seen=nodes_seen,
            edges_visited_counter=edges_visited_counter,
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            paths=paths,
            truncated=truncated,
        )

        return TraversalResult(
            start_chain=chain,
            start_network=network,
            start_normalized_address=norm_addr,
            start_composite_id=start_composite_id,
            paths=paths,
            total_paths=len(paths),
            total_edges_visited=edges_visited_counter[0],
            total_nodes_visited=len(nodes_seen),
            max_hops=max_hops,
            max_nodes=max_nodes,
            max_edges=max_edges,
            truncated=truncated[0],
        )

    async def _fetch_bitcoin_spent_inputs(self, composite_id: str) -> List[Dict[str, Any]]:
        """Fetch all outgoing SPENT_INPUT edges from an Address, ordered by input_id ASC."""
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (a:Address {composite_id: $cid})-[r:SPENT_INPUT]->(tx:Transaction)
                RETURN
                    r.input_id           AS input_id,
                    r.spent_txid         AS spent_txid,
                    r.spent_vout         AS spent_vout,
                    r.value_sat          AS value_sat,
                    r.value_sat_str      AS value_sat_str,
                    r.coinbase           AS coinbase,
                    r.pg_vin_id          AS pg_vin_id,
                    r.chain              AS chain,
                    r.network            AS network,
                    tx.composite_id      AS tx_composite_id,
                    tx.chain             AS tx_chain,
                    tx.network           AS tx_network,
                    tx.transaction_id    AS tx_transaction_id,
                    tx.pg_id             AS tx_pg_id,
                    tx.block_number      AS tx_block_number,
                    tx.block_hash        AS tx_block_hash,
                    tx.timestamp         AS tx_timestamp,
                    tx.transaction_type  AS tx_transaction_type,
                    tx.status            AS tx_status,
                    tx.fee               AS tx_fee,
                    tx.fee_str           AS tx_fee_str,
                    tx.fee_asset         AS tx_fee_asset,
                    tx.native_value      AS tx_native_value,
                    tx.native_value_str  AS tx_native_value_str,
                    tx.native_value_unit AS tx_native_value_unit
                ORDER BY r.input_id ASC
                """,
                cid=composite_id,
            )
            records = await result.data()
        return records

    async def _fetch_bitcoin_created_outputs(self, tx_composite_id: str) -> List[Dict[str, Any]]:
        """Fetch all outgoing CREATED_OUTPUT edges from a Transaction, ordered by output_id ASC."""
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (tx:Transaction {composite_id: $tx_cid})-[r:CREATED_OUTPUT]->(b:Address)
                RETURN
                    r.output_id          AS output_id,
                    r.n                  AS n,
                    r.value_sat          AS value_sat,
                    r.value_sat_str      AS value_sat_str,
                    r.script_type        AS script_type,
                    r.pg_vout_id         AS pg_vout_id,
                    r.chain              AS chain,
                    r.network            AS network,
                    b.composite_id       AS dest_composite_id,
                    b.chain              AS dest_chain,
                    b.network            AS dest_network,
                    b.normalized_address AS dest_normalized_address,
                    b.raw_address        AS dest_raw_address
                ORDER BY r.output_id ASC
                """,
                tx_cid=tx_composite_id,
            )
            records = await result.data()
        return records

    async def _dfs_bitcoin(
        self,
        current_address: TraversalNode,
        current_path_addresses: List[TraversalNode],
        current_utxo_steps: List[UtxoSpendStep],
        depth: int,
        visited_in_path: List[str],
        nodes_seen: Set[str],
        edges_visited_counter: List[int],
        max_hops: int,
        max_nodes: int,
        max_edges: int,
        paths: List[TraversalPath],
        truncated: List[bool],
    ) -> None:
        """Recursive DFS for Bitcoin UTXO spend transitions.

        Definition of bounds:
        - 1 hop: complete spend transition Address -> SPENT_INPUT -> Tx -> CREATED_OUTPUT -> Address.
        - Visited node: any Address or Transaction node admitted into the traversal.
        - Visited edge: any SPENT_INPUT or CREATED_OUTPUT relationship visited.

        Simple-path semantics:
        - visited_in_path tracks composite_ids of Address and Transaction nodes currently
          in the active path stack. A transaction or address already in the active path
          cannot be revisited.
        """
        if truncated[0]:
            return

        if depth >= max_hops:
            return

        inputs = await self._fetch_bitcoin_spent_inputs(current_address.composite_id)

        for input_rec in inputs:
            if truncated[0]:
                return

            # Check max_edges for SPENT_INPUT visit
            if edges_visited_counter[0] >= max_edges:
                truncated[0] = True
                return
            edges_visited_counter[0] += 1

            tx_cid: str = input_rec["tx_composite_id"]

            # Simple-path cycle guard: skip if transaction is already in this active path
            if tx_cid in visited_in_path:
                continue

            # Check max_nodes for Transaction node
            if tx_cid not in nodes_seen:
                if len(nodes_seen) >= max_nodes:
                    truncated[0] = True
                    return
                nodes_seen.add(tx_cid)

            tx_node = TransactionNode(
                composite_id=tx_cid,
                chain=input_rec["tx_chain"],
                network=input_rec["tx_network"],
                transaction_id=input_rec["tx_transaction_id"],
                pg_id=input_rec.get("tx_pg_id"),
                block_number=input_rec.get("tx_block_number"),
                block_hash=input_rec.get("tx_block_hash"),
                timestamp=input_rec.get("tx_timestamp"),
                transaction_type=input_rec.get("tx_transaction_type"),
                status=input_rec.get("tx_status"),
                fee=input_rec.get("tx_fee"),
                fee_str=input_rec.get("tx_fee_str"),
                fee_asset=input_rec.get("tx_fee_asset"),
                native_value=input_rec.get("tx_native_value"),
                native_value_str=input_rec.get("tx_native_value_str"),
                native_value_unit=input_rec.get("tx_native_value_unit"),
            )

            spent_edge = SpentInputEdge(
                input_id=input_rec["input_id"],
                from_composite_id=current_address.composite_id,
                tx_composite_id=tx_cid,
                spent_txid=input_rec.get("spent_txid"),
                spent_vout=input_rec.get("spent_vout"),
                value_sat=input_rec.get("value_sat"),
                value_sat_str=input_rec.get("value_sat_str"),
                coinbase=input_rec.get("coinbase", False),
                pg_vin_id=input_rec.get("pg_vin_id"),
                chain=input_rec["chain"],
                network=input_rec["network"],
            )

            outputs = await self._fetch_bitcoin_created_outputs(tx_cid)

            for output_rec in outputs:
                if truncated[0]:
                    return

                # Check max_edges for CREATED_OUTPUT visit
                if edges_visited_counter[0] >= max_edges:
                    truncated[0] = True
                    return
                edges_visited_counter[0] += 1

                dest_cid: str = output_rec["dest_composite_id"]

                # Simple-path cycle guard: skip if destination Address is already in this active path
                if dest_cid in visited_in_path:
                    continue

                # Check max_nodes for destination Address node
                if dest_cid not in nodes_seen:
                    if len(nodes_seen) >= max_nodes:
                        truncated[0] = True
                        return
                    nodes_seen.add(dest_cid)

                dest_node = TraversalNode(
                    composite_id=dest_cid,
                    chain=output_rec["dest_chain"],
                    network=output_rec["dest_network"],
                    normalized_address=output_rec["dest_normalized_address"],
                    raw_address=output_rec["dest_raw_address"],
                )

                created_edge = CreatedOutputEdge(
                    output_id=output_rec["output_id"],
                    tx_composite_id=tx_cid,
                    to_composite_id=dest_cid,
                    n=output_rec["n"],
                    value_sat=output_rec.get("value_sat"),
                    value_sat_str=output_rec.get("value_sat_str"),
                    script_type=output_rec.get("script_type"),
                    pg_vout_id=output_rec.get("pg_vout_id"),
                    chain=output_rec["chain"],
                    network=output_rec["network"],
                )

                step = UtxoSpendStep(
                    from_address=current_address,
                    spent_input=spent_edge,
                    transaction=tx_node,
                    created_output=created_edge,
                    to_address=dest_node,
                )

                next_path_addresses = current_path_addresses + [dest_node]
                next_utxo_steps = current_utxo_steps + [step]

                # Record discovered spend path
                paths.append(
                    TraversalPath(
                        nodes=next_path_addresses,
                        edges=[],
                        hops=depth + 1,
                        utxo_steps=next_utxo_steps,
                    )
                )

                # Expand deeper if hop limit permits
                if depth + 1 < max_hops:
                    visited_in_path.append(tx_cid)
                    visited_in_path.append(dest_cid)
                    await self._dfs_bitcoin(
                        current_address=dest_node,
                        current_path_addresses=next_path_addresses,
                        current_utxo_steps=next_utxo_steps,
                        depth=depth + 1,
                        visited_in_path=visited_in_path,
                        nodes_seen=nodes_seen,
                        edges_visited_counter=edges_visited_counter,
                        max_hops=max_hops,
                        max_nodes=max_nodes,
                        max_edges=max_edges,
                        paths=paths,
                        truncated=truncated,
                    )
                    visited_in_path.pop()
                    visited_in_path.pop()
