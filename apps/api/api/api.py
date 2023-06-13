from fastapi import APIRouter
from api.routes import index, status, models

api_router = APIRouter()

api_router.include_router(index.router)
api_router.include_router(status.router, prefix="/status")
api_router.include_router(models.router, prefix="/models")
