import time
import logging
import asyncpg
from neo4j.exceptions import ServiceUnavailable as Neo4jServiceUnavailable, AuthError as Neo4jAuthError
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
from app.core.config import settings
from app.core.database import db_manager
from app.schemas.health import DependencyStatus, DependenciesHealthResponse

logger = logging.getLogger(__name__)

# Exception types that indicate the dependency is simply unreachable or
# refusing connections (expected network-level failures → "disconnected").
_PG_DISCONNECTED = (
    asyncpg.PostgresConnectionError,
    asyncpg.ConnectionFailureError,
    asyncpg.TooManyConnectionsError,
    asyncpg.CannotConnectNowError,
    OSError,
    ConnectionRefusedError,
)


class HealthService:
    @staticmethod
    async def check_postgres() -> DependencyStatus:
        start_time = time.perf_counter()
        try:
            if db_manager.pg_pool is None:
                await db_manager.connect_postgres()

            if db_manager.pg_pool is None:
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return DependencyStatus(
                    status="disconnected",
                    details=f"PostgreSQL connection pool could not be established on "
                            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}",
                    latency_ms=latency,
                )

            async with db_manager.pg_pool.acquire() as conn:
                val = await conn.fetchval("SELECT 1")
                latency = round((time.perf_counter() - start_time) * 1000, 2)

                if val == 1:
                    return DependencyStatus(
                        status="connected",
                        details=f"PostgreSQL active on {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}",
                        latency_ms=latency,
                    )
                return DependencyStatus(
                    status="error",
                    details=f"Unexpected query response from PostgreSQL: {val!r}",
                    latency_ms=latency,
                )

        except _PG_DISCONNECTED as e:
            # Expected: server is down, port closed, host unreachable
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"PostgreSQL unreachable: {e}")
            return DependencyStatus(
                status="disconnected",
                details=str(e),
                latency_ms=latency,
            )
        except Exception as e:
            # Unexpected: auth failure, protocol error, pool exhaustion, etc.
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"PostgreSQL health check error: {e}")
            return DependencyStatus(
                status="error",
                details=str(e),
                latency_ms=latency,
            )

    @staticmethod
    async def check_neo4j() -> DependencyStatus:
        start_time = time.perf_counter()
        try:
            if db_manager.neo4j_driver is None:
                await db_manager.connect_neo4j()

            driver = db_manager.neo4j_driver
            if driver is None:
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return DependencyStatus(
                    status="disconnected",
                    details=f"Neo4j driver could not be initialized for {settings.NEO4J_URI}",
                    latency_ms=latency,
                )

            async with driver.session() as session:
                result = await session.run("RETURN 1 AS num")
                record = await result.single()
                latency = round((time.perf_counter() - start_time) * 1000, 2)

                if record and record["num"] == 1:
                    return DependencyStatus(
                        status="connected",
                        details=f"Neo4j Bolt reachable at {settings.NEO4J_URI}",
                        latency_ms=latency,
                    )

                return DependencyStatus(
                    status="error",
                    details="Unexpected Neo4j query response",
                    latency_ms=latency,
                )

        except (Neo4jServiceUnavailable, OSError, ConnectionRefusedError) as e:
            # Expected: Bolt port closed, Neo4j not started
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"Neo4j unreachable: {e}")
            return DependencyStatus(
                status="disconnected",
                details=str(e),
                latency_ms=latency,
            )
        except Neo4jAuthError as e:
            # Auth failure is an unexpected configuration error
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Neo4j authentication error: {e}")
            return DependencyStatus(
                status="error",
                details=f"Neo4j authentication failed: {e}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Neo4j health check error: {e}")
            return DependencyStatus(
                status="error",
                details=str(e),
                latency_ms=latency,
            )

    @staticmethod
    async def check_redis() -> DependencyStatus:
        start_time = time.perf_counter()
        try:
            if db_manager.redis_client is None:
                await db_manager.connect_redis()

            redis_client = db_manager.redis_client
            if redis_client is None:
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return DependencyStatus(
                    status="disconnected",
                    details=f"Redis client could not be initialized for "
                            f"{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                    latency_ms=latency,
                )

            pong = await redis_client.ping()
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            if pong:
                return DependencyStatus(
                    status="connected",
                    details=f"Redis ping successful on {settings.REDIS_HOST}:{settings.REDIS_PORT}",
                    latency_ms=latency,
                )
            return DependencyStatus(
                status="error",
                details="Redis ping returned falsy response",
                latency_ms=latency,
            )

        except (RedisConnectionError, RedisTimeoutError, OSError, ConnectionRefusedError) as e:
            # Expected: Redis is down or port is closed
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(f"Redis unreachable: {e}")
            return DependencyStatus(
                status="disconnected",
                details=str(e),
                latency_ms=latency,
            )
        except Exception as e:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Redis health check error: {e}")
            return DependencyStatus(
                status="error",
                details=str(e),
                latency_ms=latency,
            )

    @classmethod
    async def get_all_dependencies_health(cls) -> DependenciesHealthResponse:
        pg_res = await cls.check_postgres()
        neo_res = await cls.check_neo4j()
        redis_res = await cls.check_redis()

        dependencies = {
            "postgres": pg_res,
            "neo4j": neo_res,
            "redis": redis_res,
        }

        all_connected = all(dep.status == "connected" for dep in dependencies.values())
        any_connected = any(dep.status == "connected" for dep in dependencies.values())

        if all_connected:
            overall_status = "healthy"
        elif any_connected:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return DependenciesHealthResponse(
            status=overall_status,
            service="chakra-backend",
            dependencies=dependencies,
        )
