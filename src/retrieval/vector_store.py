from __future__ import annotations

from typing import List

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import settings


class VectorStoreManager:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def add_documents(self, docs: List[Document]) -> None:
        if docs:
            self.store.add_documents(documents=docs)

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return self.store.similarity_search(query=query, k=k)

    def stats(self) -> dict:
        collection = self.store._collection
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) if data else []
        unique_sources = sorted({m.get("source", "unknown") for m in metadatas if m})
        return {
            "indexed_chunks": len(metadatas),
            "indexed_documents": len(unique_sources),
            "documents": unique_sources,
        }
