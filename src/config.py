from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _resolve_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on what hardware is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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
    # Device for all torch/transformers models: auto-detected unless DEVICE is set.
    # Set DEVICE=cuda in your .env (or Colab runtime) to force GPU.
    device: str = os.getenv("DEVICE") or _resolve_device()


settings = Settings()
