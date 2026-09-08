import os
from enum import Enum
from typing import Any, Optional
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    """Operational mode for CHAKRA data ingestion and analysis.

    LIVE: Standard operation consuming live external blockchain providers.
    DEMO: Demonstration mode allowing synthetic scenario data ingestion.
    """

    LIVE = "LIVE"
    DEMO = "DEMO"


# Canonical alias for concise referencing
Mode = AppMode


class Settings(BaseSettings):
    PROJECT_NAME: str = "CHAKRA"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Application / Data Mode (Default is strictly LIVE)
    MODE: AppMode = Field(
        default=AppMode.LIVE,
        validation_alias=AliasChoices("MODE", "CHAKRA_MODE", "mode", "chakra_mode"),
        description="Operational mode: LIVE (default) or DEMO.",
    )

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


    @field_validator("MODE", mode="before")
    @classmethod
    def parse_app_mode(cls, v: Any) -> AppMode:
        """Robustly parse and normalize application mode.

        Accepts case-insensitive strings ('live', 'demo', 'LIVE', 'DEMO').
        Invalid or unknown tokens raise an explicit ValueError/ValidationError.
        Never silently falls back to DEMO.
        """
        if isinstance(v, AppMode):
            return v
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in AppMode.__members__:
                return AppMode[v_clean]
            valid_modes = [m.value for m in AppMode]
            raise ValueError(
                f"Invalid application mode: {v!r}. Valid modes are: {valid_modes}"
            )
        raise ValueError(
            f"Expected string or AppMode enum for MODE, got {type(v).__name__}"
        )

    @property
    def mode(self) -> AppMode:
        """Authoritative operational mode."""
        return self.MODE

    @property
    def is_live(self) -> bool:
        """Return True if running in LIVE mode."""
        return self.MODE == AppMode.LIVE

    @property
    def is_demo(self) -> bool:
        """Return True if running in DEMO mode."""
        return self.MODE == AppMode.DEMO

    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=(
            None
            if os.getenv("CHAKRA_DISABLE_ENV_FILE", "").lower() in ("1", "true", "yes")
            else os.getenv("CHAKRA_ENV_FILE", ".env")
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
