from fastapi import APIRouter
from .routes import index, status

api_router = APIRouter()

api_router.include_router(index.router)
api_router.include_router(status.router, prefix="/status")
