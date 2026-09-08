from typing import List

from app.scenarios.generator import ScenarioGenerator
from app.schemas.transaction import Transaction


from datetime import datetime, timezone

class OneHopCashoutGenerator(ScenarioGenerator):
    """Minimal baseline scenario: a direct single-hop transfer.
    
    Serves as an analytical control/negative baseline to confirm that simple,
    routine transactions do not accidentally trigger complex typologies.
    """
    @property
    def scenario_id(self) -> str:
        return "one_hop_cashout"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        prov = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_one_hop_cashout_001",
            normalized_at=base_time,
        )
        transfer = Transfer(
            from_address="chakra-demo/one_hop_cashout/source",
            to_address="chakra-demo/one_hop_cashout/cashout",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=5000000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        tx = Transaction(
            transaction_id="tx_one_hop_cashout_001",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time,
            native_value=5000000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[transfer],
            provenance=prov,
        )
        return [tx]


class PeelChainGenerator(ScenarioGenerator):
    """Bitcoin UTXO peel chain scenario forming 3 consecutive qualifying hops.
    
    Generates 4 Bitcoin transactions where T1-T3 transfer a primary remainder forward
    to fresh change candidates (>50% retention) while peeling smaller amounts to a reused
    external recipient address. T4 performs a final single-output sweep.
    Triggers Step 3D change candidate detection and Step 3G PEEL_CHAIN classification.
    """
    @property
    def scenario_id(self) -> str:
        return "peel_chain"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        from datetime import timedelta
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        txs = []
        
        amounts = [1000000, 890000, 780000, 670000, 660000]
        drop_amount = 100000
        fee = 10000
        
        for i in range(4):
            prov = TransactionProvenance(
                provider="chakra-demo",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
                original_id=f"tx_peel_{i+1}",
                normalized_at=base_time,
            )
            
            in_addr = "chakra-demo/peel/src" if i == 0 else f"chakra-demo/peel/change{i}"
            
            in_transfer = Transfer(
                from_address=in_addr,
                to_address=None,
                asset_type=AssetType.NATIVE,
                asset_symbol="BTC",
                amount=amounts[i],
                amount_unit="sat",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
            )
            
            transfers = [in_transfer]
            
            if i < 3:
                # Peeling hop
                out_drop = Transfer(
                    from_address=None,
                    to_address="chakra-demo/peel/drop",
                    asset_type=AssetType.NATIVE,
                    asset_symbol="BTC",
                    amount=drop_amount,
                    amount_unit="sat",
                    chain=Chain.BITCOIN,
                    network=Network.BTC_MAINNET,
                )
                out_change = Transfer(
                    from_address=None,
                    to_address=f"chakra-demo/peel/change{i+1}",
                    asset_type=AssetType.NATIVE,
                    asset_symbol="BTC",
                    amount=amounts[i+1],
                    amount_unit="sat",
                    chain=Chain.BITCOIN,
                    network=Network.BTC_MAINNET,
                )
                transfers.extend([out_drop, out_change])
            else:
                # Final cashout
                out_cashout = Transfer(
                    from_address=None,
                    to_address="chakra-demo/peel/cashout",
                    asset_type=AssetType.NATIVE,
                    asset_symbol="BTC",
                    amount=amounts[i+1],
                    amount_unit="sat",
                    chain=Chain.BITCOIN,
                    network=Network.BTC_MAINNET,
                )
                transfers.append(out_cashout)
                
            tx = Transaction(
                transaction_id=f"tx_peel_{i+1}",
                chain=Chain.BITCOIN,
                network=Network.BTC_MAINNET,
                timestamp=base_time + timedelta(hours=i),
                native_value=amounts[i],
                native_value_unit="sat",
                transaction_type=TransactionType.TRANSFER,
                status=TransactionStatus.SUCCESS,
                fee=fee,
                transfers=transfers,
                provenance=prov,
            )
            txs.append(tx)
            
        return txs


class FanInGenerator(ScenarioGenerator):
    """Fan-in aggregation scenario with 3 distinct sources converging on 1 hub.
    
    Generates 3 EVM transactions from distinct source addresses to a single hub address
    spaced 1 hour apart (total 2.0h window <= 24.0h threshold).
    Triggers Step 3G FAN_IN classification.
    """
    @property
    def scenario_id(self) -> str:
        return "fan_in"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        from datetime import timedelta
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        txs = []
        
        for i in range(3):
            prov = TransactionProvenance(
                provider="chakra-demo",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
                original_id=f"tx_fan_in_{i+1}",
                normalized_at=base_time,
            )
            
            transfer = Transfer(
                from_address=f"chakra-demo/fan_in/src{i+1}",
                to_address="chakra-demo/fan_in/hub",
                asset_type=AssetType.NATIVE,
                asset_symbol="ETH",
                amount=1000000000000000000,
                amount_unit="wei",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
            )
            
            tx = Transaction(
                transaction_id=f"tx_fan_in_{i+1}",
                chain=Chain.EVM,
                network=Network.ETH_MAINNET,
                timestamp=base_time + timedelta(hours=i),
                native_value=1000000000000000000,
                native_value_unit="wei",
                transaction_type=TransactionType.TRANSFER,
                status=TransactionStatus.SUCCESS,
                transfers=[transfer],
                provenance=prov,
            )
            txs.append(tx)
            
        return txs


class MixerInteractionGenerator(ScenarioGenerator):
    """Mixer interaction safety demonstration (structural mimic).
    
    Generates a deposit and withdrawal pattern around a synthetic contract address.
    Strictly emits raw transaction structures with NO authoritative mixer attribution,
    NO forced metadata, and NO risk scores. Step 3G typology detection returns
    'insufficient_evidence' as required.
    """
    @property
    def scenario_id(self) -> str:
        return "mixer_interaction"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        from datetime import timedelta
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        prov1 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_mixer_deposit",
            normalized_at=base_time,
        )
        
        t1 = Transfer(
            from_address="chakra-demo/mixer/src",
            to_address="chakra-demo/mixer/contract",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        
        tx1 = Transaction(
            transaction_id="tx_mixer_deposit",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time,
            native_value=1000000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.CONTRACT_CALL,
            status=TransactionStatus.SUCCESS,
            transfers=[t1],
            provenance=prov1,
        )
        
        prov2 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_mixer_withdrawal",
            normalized_at=base_time,
        )
        
        t2 = Transfer(
            from_address="chakra-demo/mixer/contract",
            to_address="chakra-demo/mixer/dest",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=990000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        
        tx2 = Transaction(
            transaction_id="tx_mixer_withdrawal",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time + timedelta(hours=5),
            native_value=990000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[t2],
            provenance=prov2,
        )
        
        return [tx1, tx2]


class CrossChainHopGenerator(ScenarioGenerator):
    """Cross-chain hop safety demonstration (structural mimic).
    
    Generates two transactions on different chains (EVM and TRON).
    Strictly emits raw transaction structures with NO bridge identity fabrication,
    NO bridge attribution, and NO exchange-rate conversion. Step 3G typology
    detection returns 'insufficient_evidence' as required.
    """
    @property
    def scenario_id(self) -> str:
        return "cross_chain_hop"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        from datetime import timedelta
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        prov1 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_cross_chain_eth",
            normalized_at=base_time,
        )
        
        t1 = Transfer(
            from_address="chakra-demo/cross_chain/src_eth",
            to_address="chakra-demo/cross_chain/bridge_eth",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        
        tx1 = Transaction(
            transaction_id="tx_cross_chain_eth",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time,
            native_value=1000000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.CONTRACT_CALL,
            status=TransactionStatus.SUCCESS,
            transfers=[t1],
            provenance=prov1,
        )
        
        prov2 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.TRON,
            network=Network.TRON_MAINNET,
            original_id="tx_cross_chain_trx",
            normalized_at=base_time,
        )
        
        t2 = Transfer(
            from_address="chakra-demo/cross_chain/bridge_trx",
            to_address="chakra-demo/cross_chain/dest_trx",
            asset_type=AssetType.NATIVE,
            asset_symbol="TRX",
            amount=2000000000,
            amount_unit="sun",
            chain=Chain.TRON,
            network=Network.TRON_MAINNET,
        )
        
        tx2 = Transaction(
            transaction_id="tx_cross_chain_trx",
            chain=Chain.TRON,
            network=Network.TRON_MAINNET,
            timestamp=base_time + timedelta(minutes=15),
            native_value=2000000000,
            native_value_unit="sun",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[t2],
            provenance=prov2,
        )
        
        return [tx1, tx2]


class OffshoreCashoutGenerator(ScenarioGenerator):
    """Offshore cashout scenario (synthetic structural cashout chain).
    
    Generates a two-hop movement: source -> intermediary -> synthetic offshore exchange.
    Strictly emits raw transaction structures with NO real VASP attribution,
    NO Step 4 intelligence invocation, and NO fabricated jurisdiction metadata.
    """
    @property
    def scenario_id(self) -> str:
        return "offshore_cashout"

    def generate(self) -> List[Transaction]:
        from app.schemas.transaction import (
            AssetType,
            TransactionProvenance,
            TransactionStatus,
            TransactionType,
            Transfer,
        )
        from app.schemas.chain import Chain, Network
        from datetime import timedelta
        
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        prov1 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_offshore_1",
            normalized_at=base_time,
        )
        
        t1 = Transfer(
            from_address="chakra-demo/offshore/src",
            to_address="chakra-demo/offshore/intermediary",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        
        tx1 = Transaction(
            transaction_id="tx_offshore_1",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time,
            native_value=1000000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[t1],
            provenance=prov1,
        )
        
        prov2 = TransactionProvenance(
            provider="chakra-demo",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            original_id="tx_offshore_2",
            normalized_at=base_time,
        )
        
        t2 = Transfer(
            from_address="chakra-demo/offshore/intermediary",
            to_address="chakra-demo/offshore/exchange",
            asset_type=AssetType.NATIVE,
            asset_symbol="ETH",
            amount=1000000000000000000,
            amount_unit="wei",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
        )
        
        tx2 = Transaction(
            transaction_id="tx_offshore_2",
            chain=Chain.EVM,
            network=Network.ETH_MAINNET,
            timestamp=base_time + timedelta(hours=1),
            native_value=1000000000000000000,
            native_value_unit="wei",
            transaction_type=TransactionType.TRANSFER,
            status=TransactionStatus.SUCCESS,
            transfers=[t2],
            provenance=prov2,
        )
        
        return [tx1, tx2]
