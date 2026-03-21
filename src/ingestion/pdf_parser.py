from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz
import pdfplumber
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from PIL import UnidentifiedImageError


@dataclass
class ParsedChunks:
    text_chunks: List[Document]
    table_chunks: List[Document]
    image_chunks: List[Document]


class PDFParser:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        enable_ocr_fallback: bool = True,
        ocr_dpi: int = 220,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_dpi = ocr_dpi

    def parse(self, pdf_path: Path, source_name: str, vision_summarizer) -> ParsedChunks:
        text_docs, text_pages = self._extract_text_chunks(pdf_path, source_name)
        table_docs, table_pages = self._extract_table_chunks(pdf_path, source_name)
        if self.enable_ocr_fallback:
            ocr_text_docs, ocr_table_docs = self._extract_ocr_fallback_chunks(
                pdf_path=pdf_path,
                source_name=source_name,
                pages_with_text_or_table=text_pages | table_pages,
            )
            text_docs.extend(ocr_text_docs)
            table_docs.extend(ocr_table_docs)
        image_docs = self._extract_image_summary_chunks(pdf_path, source_name, vision_summarizer)
        return ParsedChunks(text_chunks=text_docs, table_chunks=table_docs, image_chunks=image_docs)

    def _extract_text_chunks(self, pdf_path: Path, source_name: str) -> tuple[List[Document], set[int]]:
        docs: List[Document] = []
        pages_with_text: set[int] = set()
        pdf = fitz.open(pdf_path)
        try:
            for page_idx, page in enumerate(pdf, start=1):
                content = page.get_text("text").strip()
                if not content:
                    continue
                pages_with_text.add(page_idx)
                for i, chunk in enumerate(self.splitter.split_text(content), start=1):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "chunk_type": "text",
                                "chunk_index": i,
                            },
                        )
                    )
        finally:
            pdf.close()
        return docs, pages_with_text

    def _extract_table_chunks(self, pdf_path: Path, source_name: str) -> tuple[List[Document], set[int]]:
        docs: List[Document] = []
        pages_with_tables: set[int] = set()
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables, start=1):
                    markdown = self._table_to_markdown(table)
                    if not markdown.strip():
                        continue
                    pages_with_tables.add(page_idx)
                    docs.append(
                        Document(
                            page_content=markdown,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "chunk_type": "table",
                                "chunk_index": t_idx,
                            },
                        )
                    )
        return docs, pages_with_tables

    def _extract_ocr_fallback_chunks(
        self,
        pdf_path: Path,
        source_name: str,
        pages_with_text_or_table: set[int],
    ) -> tuple[List[Document], List[Document]]:
        try:
            import pytesseract
        except ImportError:
            return [], []

        text_docs: List[Document] = []
        table_docs: List[Document] = []
        scale = max(self.ocr_dpi, 72) / 72.0
        matrix = fitz.Matrix(scale, scale)

        pdf = fitz.open(pdf_path)
        try:
            for page_idx, page in enumerate(pdf, start=1):
                if page_idx in pages_with_text_or_table:
                    continue
                try:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(image).strip()
                except (RuntimeError, OSError, ValueError):
                    continue

                if not ocr_text:
                    continue

                for i, chunk in enumerate(self.splitter.split_text(ocr_text), start=1):
                    text_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "chunk_type": "text",
                                "chunk_index": i,
                                "extraction_method": "ocr_fallback",
                            },
                        )
                    )

                table_markdown = self._ocr_text_to_table_markdown(ocr_text)
                if table_markdown:
                    table_docs.append(
                        Document(
                            page_content=table_markdown,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "chunk_type": "table",
                                "chunk_index": 1,
                                "extraction_method": "ocr_fallback",
                            },
                        )
                    )
        finally:
            pdf.close()

        return text_docs, table_docs

    def _extract_image_summary_chunks(
        self, pdf_path: Path, source_name: str, vision_summarizer
    ) -> List[Document]:
        docs: List[Document] = []
        pdf = fitz.open(pdf_path)
        try:
            for page_idx, page in enumerate(pdf, start=1):
                images = page.get_images(full=True)
                for img_idx, img in enumerate(images, start=1):
                    try:
                        xref = img[0]
                        image_data = pdf.extract_image(xref)
                        image_bytes = image_data["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        summary = vision_summarizer.summarize(pil_image)
                    except (KeyError, RuntimeError, ValueError, OSError, UnidentifiedImageError):
                        # Some PDFs contain image streams that fail to decode cleanly.
                        continue
                    if not summary:
                        continue
                    docs.append(
                        Document(
                            page_content=summary,
                            metadata={
                                "source": source_name,
                                "page": page_idx,
                                "chunk_type": "image_summary",
                                "chunk_index": img_idx,
                            },
                        )
                    )
        finally:
            pdf.close()
        return docs

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        if not table:
            return ""
        normalized = [[(cell or "").strip() for cell in row] for row in table if row]
        if not normalized:
            return ""
        header = normalized[0]
        rows = normalized[1:] if len(normalized) > 1 else []
        header_line = "| " + " | ".join(header) + " |"
        sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
        row_lines = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header_line, sep_line, *row_lines]).strip()

    @staticmethod
    def _ocr_text_to_table_markdown(ocr_text: str) -> str:
        lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        candidate_rows: list[list[str]] = []
        for line in lines:
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
            else:
                cells = [c.strip() for c in re.split(r"\s{2,}|\t", line) if c.strip()]
            if len(cells) >= 3:
                candidate_rows.append(cells)

        if len(candidate_rows) < 3:
            return ""

        col_count = max(len(row) for row in candidate_rows)
        normalized_rows = [row + [""] * (col_count - len(row)) for row in candidate_rows]
        header = normalized_rows[0]
        body = normalized_rows[1:]
        header_line = "| " + " | ".join(header) + " |"
        sep_line = "| " + " | ".join(["---"] * col_count) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join([header_line, sep_line, *body_lines]).strip()
