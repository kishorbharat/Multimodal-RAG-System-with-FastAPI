from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    DocumentsResponse,
    HealthResponse,
    ImagesListResponse,
    ImageMetadata,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from src.system import system

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    stats = system.vector_manager.stats()
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - system.started_at, 3),
        model_readiness=system.readiness(),
        indexed_documents=stats["indexed_documents"],
        index_size=stats["indexed_chunks"],
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    summary = await system.ingestion_service.ingest_pdf(file)
    return IngestResponse(**summary)


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    stats = system.vector_manager.stats()
    if stats["indexed_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No indexed content. Ingest a PDF first.")

    answer, refs = system.rag_chain.run(payload.question)
    sources = [
        SourceReference(
            filename=r.source,
            page=r.page,
            chunk_type=r.chunk_type,
            chunk_index=r.chunk_index,
        )
        for r in refs
    ]
    return QueryResponse(answer=answer, sources=sources)


@router.get("/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    stats = system.vector_manager.stats()
    return DocumentsResponse(
        documents=stats["documents"],
        indexed_documents=stats["indexed_documents"],
        index_size=stats["indexed_chunks"],
    )


@router.get("/images", response_model=ImagesListResponse)
def list_images(filename: str | None = None) -> ImagesListResponse:
    """List all extracted images from ingested PDFs.
    
    Query parameters:
    - filename: Optional, filter by PDF filename (e.g., "AIS_197-1_BNCAP.pdf")
    
    Returns array of image metadata with IDs that can be used with /images/{image_id} endpoint.
    """
    images = system.image_extractor.list_images(filename=filename)
    return ImagesListResponse(images=[ImageMetadata(**img) for img in images], total=len(images))


@router.get("/images/{image_id}")
def get_image(image_id: str):
    """Retrieve a specific image by ID as binary data.
    
    Use /images endpoint first to get available image IDs.
    Returns: Image file (PNG, JPG, or JPEG format)
    """
    image_data = system.image_extractor.get_image_by_id(image_id)
    if not image_data:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")
    
    # Determine media type from image_id
    if ".jpg" in image_id or "_jpg" in image_id:
        media_type = "image/jpeg"
    elif ".jpeg" in image_id or "_jpeg" in image_id:
        media_type = "image/jpeg"
    else:
        media_type = "image/png"
    
    return StreamingResponse(
        iter([image_data]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={image_id}.png"},
    )
