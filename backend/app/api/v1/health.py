from fastapi import APIRouter
from app.schemas.health import HealthResponse, DependenciesHealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    return HealthResponse(status="ok", service="chakra-backend")


@router.get("/health/dependencies", response_model=DependenciesHealthResponse)
async def get_dependencies_health():
    return await HealthService.get_all_dependencies_health()

