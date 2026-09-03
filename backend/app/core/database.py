import logging
from typing import Optional
import asyncpg
import redis.asyncio as aioredis
from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.redis_client: Optional[aioredis.Redis] = None

    async def connect_postgres(self):
        if self.pg_pool is None:
            try:
                self.pg_pool = await asyncpg.create_pool(
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    database=settings.POSTGRES_DB,
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    min_size=1,
                    max_size=10,
                    timeout=3.0,
                    command_timeout=3.0,
                )
                logger.info("PostgreSQL connection pool initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
                self.pg_pool = None

    async def connect_neo4j(self):
        if self.neo4j_driver is None:
            try:
                self.neo4j_driver = AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                    connection_timeout=3.0,
                )
                logger.info("Neo4j async driver initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                self.neo4j_driver = None

    async def connect_redis(self):
        if self.redis_client is None:
            try:
                self.redis_client = aioredis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD or None,
                    decode_responses=True,
                    socket_timeout=3.0,
                )
                logger.info("Redis async client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                self.redis_client = None

    async def connect(self):
        await self.connect_postgres()
        await self.connect_neo4j()
        await self.connect_redis()

    async def disconnect(self):
        if self.pg_pool:
            try:
                await self.pg_pool.close()
                logger.info("PostgreSQL connection pool closed.")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection pool: {e}")
            finally:
                self.pg_pool = None

        if self.neo4j_driver:
            try:
                await self.neo4j_driver.close()
                logger.info("Neo4j driver closed.")
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")
            finally:
                self.neo4j_driver = None

        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis client closed.")
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self.redis_client = None


db_manager = DatabaseManager()
