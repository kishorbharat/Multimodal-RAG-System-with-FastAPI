from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain.docstore.document import Document

from src.config import settings
from src.models.llm import invoke_llm


RAG_PROMPT_TEMPLATE = """You are a grounded assistant for multimodal document QA.
Use only the provided context to answer the question.
If the answer is missing from context, say that explicitly.
Do not fabricate values.
Write the answer in clear, human-readable language.
Do not copy large table fragments verbatim.
Summarize extracted values and conditions instead of repeating raw rows.

Question:
{question}

Context:
{context}

Return a concise answer in 3-6 sentences.
"""


@dataclass
class SourceRef:
    source: str
    page: int
    chunk_type: str
    chunk_index: int


class RAGChain:
    def __init__(self, vector_manager, llm) -> None:
        self.vector_manager = vector_manager
        self.llm = llm

    def run(self, question: str) -> tuple[str, List[SourceRef]]:
        docs = self.vector_manager.similarity_search(question, k=settings.retrieval_k)
        prompt = RAG_PROMPT_TEMPLATE.format(
            question=question,
            context=self._format_context(docs),
        )
        answer = invoke_llm(self.llm, prompt)
        refs = [
            SourceRef(
                source=str(doc.metadata.get("source", "unknown")),
                page=int(doc.metadata.get("page", -1)),
                chunk_type=str(doc.metadata.get("chunk_type", "unknown")),
                chunk_index=int(doc.metadata.get("chunk_index", -1)),
            )
            for doc in docs
        ]
        return answer, refs

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        lines: List[str] = []
        seen: set[tuple[str, int, str, int]] = set()
        total_chars = 0
        max_total_chars = 6000

        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            ref = (
                str(meta.get("source", "unknown")),
                int(meta.get("page", -1)),
                str(meta.get("chunk_type", "unknown")),
                int(meta.get("chunk_index", -1)),
            )
            if ref in seen:
                continue
            seen.add(ref)

            content = " ".join(str(doc.page_content).split())
            content = content[:800]
            if len(content) == 800:
                content += " ..."

            chunk = (
                f"[{i}] source={meta.get('source')} page={meta.get('page')} "
                f"type={meta.get('chunk_type')} idx={meta.get('chunk_index')}\n{content}\n"
            )
            if total_chars + len(chunk) > max_total_chars:
                break

            lines.append(
                chunk
            )
            total_chars += len(chunk)
        return "\n".join(lines)
