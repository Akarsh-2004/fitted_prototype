import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pipeline.config import settings
from pipeline.database.storage import init_db
from pipeline.router import api_router

app = FastAPI(
    title="Vestir AI - Upgraded Wardrobe Ingestion Pipeline",
    description="Production-grade AI wardrobe ingestion backend with adaptive multi-scenario processing.",
    version="2.0.0"
)

# Enable CORS for full frontend accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories are initialized
settings.storage_dir.mkdir(parents=True, exist_ok=True)

# Mount the local asset storage folder so crops can be fetched by the browser
app.mount("/data/storage", StaticFiles(directory=str(settings.storage_dir)), name="storage")

# Include the main API router
app.include_router(api_router)

@app.on_event("startup")
def on_startup():
    """Initializes SQLite databases and FAISS indexes on server startup."""
    print("🔮 [Vestir AI] Starting services...")
    init_db()
    print("✅ [Vestir AI] SQLite database tables initialized.")

@app.get("/health")
def health_check():
    from pipeline.services.model_registry import model_registry
    return {
        "status": "healthy",
        "service": "Vestir AI Upgraded Pipeline",
        "yolo_path": settings.yolo_model_path,
        "sqlite_path": str(settings.sqlite_db_path),
        "models": model_registry.get_status()
    }


if __name__ == "__main__":
    import uvicorn
    # Bound to 127.0.0.1 and port 8011 as standardized
    uvicorn.run("main:app", host="127.0.0.1", port=8011, reload=True)
