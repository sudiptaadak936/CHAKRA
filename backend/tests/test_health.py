from unittest.mock import patch
import pytest
from app.schemas.health import DependencyStatus, DependenciesHealthResponse
from app.services.health_service import HealthService


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "chakra-backend"


@pytest.mark.asyncio
async def test_dependencies_health_endpoint_structure(async_client):
    mock_response = DependenciesHealthResponse(
        status="healthy",
        service="chakra-backend",
        dependencies={
            "postgres": DependencyStatus(
                status="connected",
                details="PostgreSQL active on postgres:5432",
                latency_ms=1.5,
            ),
            "neo4j": DependencyStatus(
                status="connected",
                details="Neo4j Bolt reachable at bolt://neo4j:7687",
                latency_ms=2.1,
            ),
            "redis": DependencyStatus(
                status="connected",
                details="Redis Ping successful on redis:6379",
                latency_ms=0.9,
            ),
        },
    )

    with patch.object(HealthService, "get_all_dependencies_health", return_value=mock_response):
        response = await async_client.get("/health/dependencies")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "dependencies" in data
        assert data["dependencies"]["postgres"]["status"] == "connected"
        assert data["dependencies"]["neo4j"]["status"] == "connected"
        assert data["dependencies"]["redis"]["status"] == "connected"
        for dep_key, dep_info in data["dependencies"].items():
            assert "status" in dep_info
            assert "latency_ms" in dep_info
            assert "details" in dep_info


@pytest.mark.asyncio
async def test_health_service_all_connected_mock():
    mock_pg = DependencyStatus(status="connected", details="PostgreSQL active", latency_ms=1.2)
    mock_neo = DependencyStatus(status="connected", details="Neo4j Bolt reachable", latency_ms=2.5)
    mock_redis = DependencyStatus(status="connected", details="Redis Ping successful", latency_ms=0.8)

    with patch.object(HealthService, "check_postgres", return_value=mock_pg), \
         patch.object(HealthService, "check_neo4j", return_value=mock_neo), \
         patch.object(HealthService, "check_redis", return_value=mock_redis):
        result = await HealthService.get_all_dependencies_health()
        assert isinstance(result, DependenciesHealthResponse)
        assert result.status == "healthy"
        assert result.dependencies["postgres"].status == "connected"
        assert result.dependencies["neo4j"].status == "connected"
        assert result.dependencies["redis"].status == "connected"


@pytest.mark.asyncio
async def test_health_service_degraded_mock():
    mock_pg = DependencyStatus(status="connected", details="PostgreSQL active", latency_ms=1.2)
    mock_neo = DependencyStatus(status="disconnected", details="Connection refused", latency_ms=5.0)
    mock_redis = DependencyStatus(status="connected", details="Redis Ping successful", latency_ms=0.8)

    with patch.object(HealthService, "check_postgres", return_value=mock_pg), \
         patch.object(HealthService, "check_neo4j", return_value=mock_neo), \
         patch.object(HealthService, "check_redis", return_value=mock_redis):
        result = await HealthService.get_all_dependencies_health()
        assert result.status == "degraded"


@pytest.mark.asyncio
async def test_health_service_unhealthy_mock():
    mock_pg = DependencyStatus(status="disconnected", details="Host unreachable", latency_ms=5.0)
    mock_neo = DependencyStatus(status="disconnected", details="Connection refused", latency_ms=5.0)
    mock_redis = DependencyStatus(status="disconnected", details="Timeout", latency_ms=5.0)

    with patch.object(HealthService, "check_postgres", return_value=mock_pg), \
         patch.object(HealthService, "check_neo4j", return_value=mock_neo), \
         patch.object(HealthService, "check_redis", return_value=mock_redis):
        result = await HealthService.get_all_dependencies_health()
        assert result.status == "unhealthy"