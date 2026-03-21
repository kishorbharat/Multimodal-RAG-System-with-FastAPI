from __future__ import annotations

import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import (
    DocumentsResponse,
    HealthResponse,
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
