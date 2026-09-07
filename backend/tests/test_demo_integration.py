import os
import pytest
import asyncpg
from unittest.mock import MagicMock
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.schemas.mode import AppMode
from app.services.mode_service import ModeService
from app.scenarios.registry import ScenarioRegistry
from app.services.ingestion_service import IngestionService
from app.db.repository import TransactionRepository
from app.graph.projector import GraphProjector
from app.services.demo_service import DemoScenarioService
from app.forensics.typology_detector import TypologyDetector

# Environment connection config
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "chakra_user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "your_actual_password")
PG_DB = os.getenv("POSTGRES_DB", "chakra_db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
if "bolt://neo4j" in NEO4J_URI and PG_HOST == "localhost":
    NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "your_actual_password")

@pytest.fixture
async def pg_pool():
    pool = await asyncpg.create_pool(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
        database=PG_DB,
        min_size=1,
        max_size=5,
    )
    yield pool
    await pool.close()

@pytest.fixture
async def neo4j_driver():
    driver: AsyncDriver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
    )
    yield driver
    await driver.close()

@pytest.fixture
async def clean_slate(pg_pool, neo4j_driver):
    """Clean slate before each test and ensure constraints are applied."""
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE transactions CASCADE")

    projector = GraphProjector(pg_pool, neo4j_driver)
    await projector.init_constraints()
    await projector.clear_graph()
    
    yield

from app.scenarios import scenario_registry

@pytest.fixture
def demo_service(pg_pool, neo4j_driver):
    repo = TransactionRepository(pg_pool)
    ingestion = IngestionService(repository=repo)
    mode_service = ModeService()
    projector = GraphProjector(pg_pool, neo4j_driver)
    
    return DemoScenarioService(
        registry=scenario_registry,
        ingestion_service=ingestion,
        mode_service=mode_service,
        projector=projector
    )

@pytest.mark.asyncio
async def test_demo_service_mode_safety(demo_service):
    """Ensure DemoScenarioService rejects execution when in LIVE mode."""
    # Force LIVE mode
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = False
    
    with pytest.raises(RuntimeError, match="Demo operations are disabled"):
        await demo_service.execute_scenario("one_hop_cashout")

@pytest.mark.asyncio
async def test_demo_service_unknown_scenario(demo_service):
    """Ensure DemoScenarioService rejects unknown scenario IDs."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True
    
    with pytest.raises(ValueError, match="Unknown scenario ID"):
        await demo_service.execute_scenario("non_existent_scenario")

@pytest.mark.asyncio
async def test_demo_service_successful_execution(demo_service, clean_slate, neo4j_driver):
    """Verify end-to-end execution of a deterministic scenario."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True
    
    # Execute the peel chain scenario
    result = await demo_service.execute_scenario("peel_chain")
    
    assert result.scenario_id == "peel_chain"
    assert result.transactions_generated > 0
    assert result.projected_transactions == result.transactions_generated
    # Note: peel_chain is a Bitcoin scenario, so it generates 0 TRANSFERRED edges
    # (it generates SPENT_INPUT / CREATED_OUTPUT instead).
    assert result.projected_transfers == 0
    
    # Verify Neo4j data directly
    async with neo4j_driver.session() as session:
        res = await session.run("MATCH (t:Transaction) RETURN count(t) as count")
        record = await res.single()
        assert record["count"] == result.transactions_generated

@pytest.mark.asyncio
async def test_demo_service_idempotency(demo_service, clean_slate):
    """Verify running the same scenario twice doesn't duplicate data."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True
    
    res1 = await demo_service.execute_scenario("one_hop_cashout")
    res2 = await demo_service.execute_scenario("one_hop_cashout")
    
    assert res1.transactions_generated == res2.transactions_generated
    assert res2.projected_transactions == res1.projected_transactions

@pytest.mark.asyncio
async def test_demo_service_forensic_integration(demo_service, clean_slate, neo4j_driver):
    """Verify that generated demo data is properly structured in Neo4j for forensic analysis."""
    demo_service.mode_service = MagicMock()
    demo_service.mode_service.is_demo.return_value = True
    
    # Peel chain scenario has known forensic traces
    await demo_service.execute_scenario("peel_chain")
    
    # Run a generic Cypher query to find the length of the longest SPENT_INPUT/CREATED_OUTPUT chain
    # This proves the Bitcoin UTXO projection worked and the peel chain is continuous.
    query = """
    MATCH p = (start:Transaction)-[:CREATED_OUTPUT]->(:Address)-[:SPENT_INPUT]->(:Transaction)-[:CREATED_OUTPUT]->(:Address)-[:SPENT_INPUT]->(:Transaction)
    RETURN length(p) as path_length
    LIMIT 1
    """
    async with neo4j_driver.session() as session:
        res = await session.run(query)
        record = await res.single()
        # The peel chain generates 4 transactions, so we should find a path connecting them.
        assert record is not None
        assert record["path_length"] > 0
