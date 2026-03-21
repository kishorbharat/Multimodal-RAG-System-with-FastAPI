from __future__ import annotations

import time

from src.config import settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.service import IngestionService
from src.models.llm import load_llm
from src.models.vision import VisionSummarizer
from src.retrieval.rag_chain import RAGChain
from src.retrieval.vector_store import VectorStoreManager


class AppSystem:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.vector_manager = VectorStoreManager()
        self.parser = PDFParser(
            enable_ocr_fallback=settings.enable_ocr_fallback,
            ocr_dpi=settings.ocr_dpi,
        )
        self.vision = VisionSummarizer(settings.vlm_model)
        self.llm = load_llm()
        self.ingestion_service = IngestionService(
            parser=self.parser,
            vector_manager=self.vector_manager,
            vision_summarizer=self.vision,
        )
        self.rag_chain = RAGChain(vector_manager=self.vector_manager, llm=self.llm)

    def readiness(self) -> dict:
        return {
            "embeddings_ready": self.vector_manager is not None,
            "vector_store_ready": self.vector_manager is not None,
            "vlm_ready": self.vision is not None,
            "llm_ready": self.llm is not None,
        }


system = AppSystem()
