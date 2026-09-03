import pytest
from app.core.database import db_manager
from app.services.health_service import HealthService


@pytest.mark.asyncio
async def test_live_dependencies_connectivity():
    """
    Integration test validating real connectivity to PostgreSQL, Neo4j, and Redis.
    Intended to be run when the Docker stack or live local services are running.
    """
    await db_manager.connect()
    try:
        health = await HealthService.get_all_dependencies_health()
        assert health.status == "healthy", f"Dependencies not fully healthy: {health.dependencies}"
        assert health.dependencies["postgres"].status == "connected"
        assert health.dependencies["neo4j"].status == "connected"
        assert health.dependencies["redis"].status == "connected"
    finally:
        await db_manager.disconnect()
