from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multimodal RAG API for PDF text, tables, and image summaries.",
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "message": "Multimodal RAG System is running.",
        "docs": "/docs",
        "health": "/health",
    }
