from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Multimodal RAG System"
    app_version: str = "1.0.0"
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./data/chroma"))
    collection_name: str = os.getenv("COLLECTION_NAME", "multimodal_rag")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    vlm_model: str = os.getenv(
        "VLM_MODEL", "Salesforce/blip-image-captioning-base"
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai_or_local")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "6"))
    enable_ocr_fallback: bool = os.getenv("ENABLE_OCR_FALLBACK", "true").lower() == "true"
    ocr_dpi: int = int(os.getenv("OCR_DPI", "220"))


settings = Settings()
