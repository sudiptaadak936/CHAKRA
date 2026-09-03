from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "CHAKRA"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # PostgreSQL
    POSTGRES_DB: str = "chakra_db"
    POSTGRES_USER: str = "chakra_user"
    POSTGRES_PASSWORD: str = "chakra_secure_password"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "chakra_neo4j_password"

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None

    # Server
    BACKEND_PORT: int = 8000

    # Provider API keys — all optional; missing keys are reported as not_configured
    # Never log, print, or expose these values in API responses or frontend code.
    TRONGRID_API_KEY: Optional[str] = None
    ETHERSCAN_API_KEY: Optional[str] = None
    HELIUS_API_KEY: Optional[str] = None
    SOLSCAN_API_KEY: Optional[str] = None
    CHAINABUSE_API_KEY: Optional[str] = None


    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
