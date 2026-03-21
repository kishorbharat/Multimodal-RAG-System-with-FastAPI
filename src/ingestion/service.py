from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import UploadFile

from src.ingestion.pdf_parser import PDFParser


class IngestionService:
    def __init__(self, parser: PDFParser, vector_manager, vision_summarizer) -> None:
        self.parser = parser
        self.vector_manager = vector_manager
        self.vision_summarizer = vision_summarizer

    async def ingest_pdf(self, file: UploadFile) -> dict:
        start = time.perf_counter()
        suffix = Path(file.filename or "uploaded.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            payload = await file.read()
            tmp.write(payload)
            tmp_path = Path(tmp.name)

        parsed = self.parser.parse(
            pdf_path=tmp_path,
            source_name=file.filename or tmp_path.name,
            vision_summarizer=self.vision_summarizer,
        )

        all_docs = [*parsed.text_chunks, *parsed.table_chunks, *parsed.image_chunks]
        self.vector_manager.add_documents(all_docs)

        elapsed = time.perf_counter() - start
        return {
            "filename": file.filename,
            "text_chunks": len(parsed.text_chunks),
            "table_chunks": len(parsed.table_chunks),
            "image_summary_chunks": len(parsed.image_chunks),
            "total_chunks": len(all_docs),
            "processing_time_seconds": round(elapsed, 3),
        }
