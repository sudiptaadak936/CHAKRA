import asyncpg
from typing import Optional, List, Dict, Any
from app.schemas.transaction import (
    Transaction,
    Transfer,
    TransactionProvenance,
    BitcoinTransactionDetail,
    BitcoinVin,
    BitcoinVout,
    AssetType,
    TransactionStatus,
    TransactionType,
    Chain,
    Network,
)

class TransactionRepository:
    """
    Repository for persisting Canonical CHAKRA Transactions and Transfers.
    
    Design constraints:
    - Monetary values are stored exactly as NUMERIC(78,0) (no precision loss).
    - Writes are atomic per transaction.
    - Idempotency: duplicate attempts to save the same (transaction_id, chain, network)
      will result in an update/ignore without duplicating transfers.
    - Bitcoin: UTXO structures are optionally stored in dedicated tables.
      If both generic Transfer rows and Bitcoin UTXO records are stored, the generic
      Transfer rows use from_address=None for outputs and to_address=None for inputs.
      Consumers should query EITHER transfers OR bitcoin_transaction_details to avoid double counting.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def save_transaction(self, tx: Transaction, bitcoin_detail: Optional[BitcoinTransactionDetail] = None) -> None:
        """
        Atomically saves a Transaction, its Transfers, Provenance, and optional Bitcoin details.
        If the transaction already exists, it is ignored (idempotent).
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Atomic Insert Transaction with ON CONFLICT DO NOTHING
                tx_pk = await conn.fetchval(
                    """
                    INSERT INTO transactions (
                        transaction_id, chain, network, chain_id, block_number,
                        block_hash, timestamp, from_address, to_address,
                        native_value, native_value_unit, transaction_type, status,
                        fee, fee_asset
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (transaction_id, chain, network) DO NOTHING
                    RETURNING id
                    """,
                    tx.transaction_id,
                    tx.chain.value,
                    tx.network.value,
                    tx.chain_id,
                    tx.block_number,
                    tx.block_hash,
                    tx.timestamp,
                    tx.from_address,
                    tx.to_address,
                    tx.native_value,
                    tx.native_value_unit,
                    tx.transaction_type.value,
                    tx.status.value,
                    tx.fee,
                    tx.fee_asset
                )

                if not tx_pk:
                    # Transaction already exists (concurrent insert or prior save).
                    # Retrieve the existing transaction primary key for canonical identity confirmation.
                    existing_id = await conn.fetchval(
                        "SELECT id FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
                        tx.transaction_id, tx.chain.value, tx.network.value
                    )
                    # Treat as idempotent duplicate; do not insert child records, return cleanly.
                    return


                # 3. Insert Transfers
                if tx.transfers:
                    transfer_data = [
                        (
                            tx_pk,
                            t.from_address,
                            t.to_address,
                            t.asset_type.value,
                            t.asset_symbol,
                            t.asset_contract,
                            t.amount,
                            t.amount_unit,
                            t.decimals,
                            t.chain.value,
                            t.network.value
                        ) for t in tx.transfers
                    ]
                    
                    await conn.executemany(
                        """
                        INSERT INTO transfers (
                            transaction_pk, from_address, to_address, asset_type,
                            asset_symbol, asset_contract, amount, amount_unit,
                            decimals, chain, network
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        transfer_data
                    )

                # 4. Insert Provenance
                await conn.execute(
                    """
                    INSERT INTO transaction_provenance (
                        transaction_pk, provider, chain, network, original_id, normalized_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    tx_pk,
                    tx.provenance.provider,
                    tx.provenance.chain.value,
                    tx.provenance.network.value,
                    tx.provenance.original_id,
                    tx.provenance.normalized_at
                )

                # 5. Insert Bitcoin Detail (if provided)
                if bitcoin_detail:
                    detail_pk = await conn.fetchval(
                        """
                        INSERT INTO bitcoin_transaction_details (
                            transaction_pk, txid, total_input_sat, total_output_sat,
                            fee_sat, locktime, version
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        tx_pk,
                        bitcoin_detail.txid,
                        bitcoin_detail.total_input_sat,
                        bitcoin_detail.total_output_sat,
                        bitcoin_detail.fee_sat,
                        bitcoin_detail.locktime,
                        bitcoin_detail.version
                    )

                    if bitcoin_detail.inputs:
                        vin_data = [
                            (
                                detail_pk,
                                vin.txid,
                                vin.vout,
                                vin.address,
                                vin.value_sat,
                                vin.coinbase
                            ) for vin in bitcoin_detail.inputs
                        ]
                        await conn.executemany(
                            """
                            INSERT INTO bitcoin_vins (
                                detail_pk, txid, vout, address, value_sat, coinbase
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                            """,
                            vin_data
                        )
                        
                    if bitcoin_detail.outputs:
                        vout_data = [
                            (
                                detail_pk,
                                vout.n,
                                vout.address,
                                vout.value_sat,
                                vout.script_type
                            ) for vout in bitcoin_detail.outputs
                        ]
                        await conn.executemany(
                            """
                            INSERT INTO bitcoin_vouts (
                                detail_pk, n, address, value_sat, script_type
                            ) VALUES ($1, $2, $3, $4, $5)
                            """,
                            vout_data
                        )

    async def transaction_exists(self, transaction_id: str, chain: Chain, network: Network) -> bool:
        async with self.pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT 1 FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
                transaction_id, chain.value, network.value
            )
            return bool(val)
            
    async def get_transaction(self, transaction_id: str, chain: Chain, network: Network) -> Optional[Transaction]:
        """
        Reads a transaction, its transfers, and provenance back from PostgreSQL.
        Constructs the canonical Transaction object.
        """
        async with self.pool.acquire() as conn:
            tx_row = await conn.fetchrow(
                "SELECT * FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
                transaction_id, chain.value, network.value
            )
            
            if not tx_row:
                return None
                
            tx_pk = tx_row['id']
            
            # Fetch transfers
            transfer_rows = await conn.fetch(
                "SELECT * FROM transfers WHERE transaction_pk = $1", tx_pk
            )
            
            # Fetch provenance
            prov_row = await conn.fetchrow(
                "SELECT * FROM transaction_provenance WHERE transaction_pk = $1", tx_pk
            )
            
            if not prov_row:
                raise ValueError(f"Transaction {tx_pk} has no provenance")
                
            transfers = []
            for row in transfer_rows:
                transfers.append(Transfer(
                    from_address=row['from_address'],
                    to_address=row['to_address'],
                    asset_type=AssetType(row['asset_type']),
                    asset_symbol=row['asset_symbol'],
                    asset_contract=row['asset_contract'],
                    amount=int(row['amount']),
                    amount_unit=row['amount_unit'],
                    decimals=row['decimals'],
                    chain=Chain(row['chain']),
                    network=Network(row['network'])
                ))
                
            provenance = TransactionProvenance(
                provider=prov_row['provider'],
                chain=Chain(prov_row['chain']),
                network=Network(prov_row['network']),
                original_id=prov_row['original_id'],
                normalized_at=prov_row['normalized_at']
            )
            
            return Transaction(
                transaction_id=tx_row['transaction_id'],
                chain=Chain(tx_row['chain']),
                network=Network(tx_row['network']),
                chain_id=tx_row['chain_id'],
                block_number=tx_row['block_number'],
                block_hash=tx_row['block_hash'],
                timestamp=tx_row['timestamp'],
                from_address=tx_row['from_address'],
                to_address=tx_row['to_address'],
                native_value=int(tx_row['native_value']),
                native_value_unit=tx_row['native_value_unit'],
                transaction_type=TransactionType(tx_row['transaction_type']),
                status=TransactionStatus(tx_row['status']),
                fee=int(tx_row['fee']) if tx_row['fee'] is not None else None,
                fee_asset=tx_row['fee_asset'],
                transfers=transfers,
                provenance=provenance
            )
            
    async def get_bitcoin_detail(self, transaction_id: str, chain: Chain, network: Network) -> Optional[BitcoinTransactionDetail]:
        """
        Reads Bitcoin UTXO details back.
        """
        async with self.pool.acquire() as conn:
            tx_row = await conn.fetchrow(
                "SELECT id FROM transactions WHERE transaction_id = $1 AND chain = $2 AND network = $3",
                transaction_id, chain.value, network.value
            )
            if not tx_row:
                return None
                
            detail_row = await conn.fetchrow(
                "SELECT * FROM bitcoin_transaction_details WHERE transaction_pk = $1", tx_row['id']
            )
            if not detail_row:
                return None
                
            detail_pk = detail_row['id']
            vin_rows = await conn.fetch("SELECT * FROM bitcoin_vins WHERE detail_pk = $1", detail_pk)
            vout_rows = await conn.fetch("SELECT * FROM bitcoin_vouts WHERE detail_pk = $1", detail_pk)
            
            inputs = [
                BitcoinVin(
                    txid=r['txid'],
                    vout=r['vout'],
                    address=r['address'],
                    value_sat=int(r['value_sat']) if r['value_sat'] is not None else None,
                    coinbase=r['coinbase']
                ) for r in vin_rows
            ]
            
            outputs = [
                BitcoinVout(
                    n=r['n'],
                    address=r['address'],
                    value_sat=int(r['value_sat']),
                    script_type=r['script_type']
                ) for r in vout_rows
            ]
            
            return BitcoinTransactionDetail(
                txid=detail_row['txid'],
                inputs=inputs,
                outputs=outputs,
                total_input_sat=int(detail_row['total_input_sat']) if detail_row['total_input_sat'] is not None else None,
                total_output_sat=int(detail_row['total_output_sat']),
                fee_sat=int(detail_row['fee_sat']) if detail_row['fee_sat'] is not None else None,
                locktime=detail_row['locktime'],
                version=detail_row['version']
            )
