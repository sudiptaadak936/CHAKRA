import pytest
from app.core.database import db_manager
from app.services.health_service import HealthService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_dependencies_connectivity():
    """
    Integration test validating real connectivity to PostgreSQL, Neo4j, and Redis.
    
    Execution Model:
      - Docker Network: Inside the containerized compose environment, container hostnames
        ('postgres', 'neo4j', 'redis') resolve across the internal Docker bridge network.
      - Local Host: Outside Docker, services listen on localhost-mapped ports. Running this
        test on the host OS requires setting POSTGRES_HOST=localhost, REDIS_HOST=localhost,
        and NEO4J_URI=bolt://localhost:7687, or running inside the Docker backend container.
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
