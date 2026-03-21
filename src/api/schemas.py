from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_readiness: dict
    indexed_documents: int
    index_size: int


class IngestResponse(BaseModel):
    filename: str | None
    text_chunks: int
    table_chunks: int
    image_summary_chunks: int
    total_chunks: int
    processing_time_seconds: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)


class SourceReference(BaseModel):
    filename: str
    page: int
    chunk_type: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceReference]


class DocumentsResponse(BaseModel):
    documents: List[str]
    indexed_documents: int
    index_size: int
