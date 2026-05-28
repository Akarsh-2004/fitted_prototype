from fastapi import APIRouter
from pipeline.api.upload import router as upload_router
from pipeline.api.processing import router as processing_router
from pipeline.api.selection import router as selection_router
from pipeline.api.wardrobe import router as wardrobe_router
from pipeline.api.composer import router as composer_router

api_router = APIRouter()

# Include sub-routers
api_router.include_router(upload_router)
api_router.include_router(processing_router)
api_router.include_router(selection_router)
api_router.include_router(wardrobe_router)
api_router.include_router(composer_router)
