from typing import List, Optional
import logging

from app.schemas.transaction import Transaction
from app.schemas.mode import AppMode
from app.scenarios.registry import ScenarioRegistry
from app.services.ingestion_service import IngestionService
from app.services.mode_service import ModeService
from app.graph.projector import GraphProjector

logger = logging.getLogger(__name__)

class DemoExecutionResult:
    def __init__(self, scenario_id: str, transactions_generated: int, projected_transactions: int, projected_transfers: int):
        self.scenario_id = scenario_id
        self.transactions_generated = transactions_generated
        self.projected_transactions = projected_transactions
        self.projected_transfers = projected_transfers

class DemoScenarioService:
    """Orchestrates deterministic demo scenario generation and real pipeline ingestion."""

    def __init__(
        self,
        registry: ScenarioRegistry,
        ingestion_service: IngestionService,
        mode_service: ModeService,
        projector: GraphProjector
    ):
        self.registry = registry
        self.ingestion_service = ingestion_service
        self.mode_service = mode_service
        self.projector = projector

    async def execute_scenario(self, scenario_id: str) -> DemoExecutionResult:
        """Executes a scenario explicitly in Demo Mode.
        
        Flow: Generator -> IngestionService (direct) -> PostgreSQL -> GraphProjector (Neo4j).
        """
        # Strict mode safety: ONLY run in DEMO mode
        if not self.mode_service.is_demo():
            raise RuntimeError("Demo operations are disabled in LIVE mode.")

        # 1. Resolve Scenario
        try:
            generator = self.registry.get(scenario_id)
        except Exception:
            raise ValueError(f"Unknown scenario ID: {scenario_id}")

        # 2. Generate Transactions deterministically
        transactions = generator.generate()
        logger.info("Generated %d transactions for demo scenario %s", len(transactions), scenario_id)

        # 3. Direct Pipeline Ingestion (PostgreSQL via IngestionService, bypassing external providers)
        for tx in transactions:
            await self.ingestion_service.ingest_direct(tx)

        # 4. Trigger Graph Projection (Neo4j)
        # Using the existing projection pipeline.
        summary = await self.projector.project_all()
        logger.info(
            "Demo scenario %s projection summary: %d txs, %d transfers",
            scenario_id, summary.total_transactions, summary.total_transfers
        )

        return DemoExecutionResult(
            scenario_id=scenario_id,
            transactions_generated=len(transactions),
            projected_transactions=summary.total_transactions,
            projected_transfers=summary.total_transfers
        )
