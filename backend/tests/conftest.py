import os
import socket

# Execution model configuration:
# When pytest runs inside the Docker network, internal service names
# ('postgres', 'neo4j', 'redis') resolve via Docker DNS.
# When running directly on the host machine outside Docker, those container
# names do not resolve in host DNS. Set localhost defaults so host test runs
# connect to the Docker-exposed ports on the host.
def _is_local_docker_service(hostname: str, port: int) -> bool:
    try:
        sock = socket.create_connection((hostname, port), timeout=0.2)
        sock.close()
        return True
    except (socket.gaierror, OSError):
        return False

# When running on host, container hostnames fail to connect directly; fall back to 127.0.0.1
if not _is_local_docker_service("postgres", 5432):
    if os.environ.get("POSTGRES_HOST") in (None, "postgres", "localhost"):
        os.environ["POSTGRES_HOST"] = "127.0.0.1"

if not _is_local_docker_service("neo4j", 7687):
    current_uri = os.environ.get("NEO4J_URI", "")
    if not current_uri or "bolt://neo4j" in current_uri or "bolt://localhost" in current_uri:
        os.environ["NEO4J_URI"] = "bolt://127.0.0.1:7687"

if not _is_local_docker_service("redis", 6379):
    if os.environ.get("REDIS_HOST") in (None, "redis", "localhost"):
        os.environ["REDIS_HOST"] = "127.0.0.1"

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client