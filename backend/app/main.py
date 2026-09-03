import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.health import router as health_router
from app.api.v1.providers import router as providers_router
from app.core.config import settings
from app.core.database import db_manager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chakra-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing CHAKRA backend services and database connections...")
    await db_manager.connect()
    logger.info("CHAKRA backend startup completed.")
    yield
    # Shutdown
    logger.info("Shutting down CHAKRA backend and closing database connections...")
    await db_manager.disconnect()
    logger.info("CHAKRA backend shutdown completed.")


app = FastAPI(
    title="CHAKRA API",
    description="Cryptocurrency Fraud Investigation Platform - Core API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health endpoints and versioned routes
app.include_router(health_router)
app.include_router(providers_router)
