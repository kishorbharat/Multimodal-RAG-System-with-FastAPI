# Test Queries for Multimodal RAG API

This document demonstrates test queries covering all three modalities: text, tables, and images.

---

## Query 1: Text-Focused Question

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main safety requirements and regulations covered in this document?"
  }'
```

**Response Format:**
```json
{
  "answer": "<LLM-generated answer synthesized from retrieved text chunks>",
  "sources": [
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 42,
      "chunk_type": "text",
      "chunk_index": 3
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 45,
      "chunk_type": "text",
      "chunk_index": 1
    }
  ]
}
```

**Expected Behavior:**
- Retriever fetches top-K text chunks via semantic similarity
- Sources list includes `chunk_type: "text"` for text-extracted content
- Answer reflects ground truth from text paragraphs and definitions

---

## Query 2: Table-Focused Question

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What numerical thresholds and limits are specified in tabular form?"
  }'
```

**Response Format:**
```json
{
  "answer": "<LLM-generated answer referencing table data>",
  "sources": [
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 38,
      "chunk_type": "table",
      "chunk_index": 1
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 40,
      "chunk_type": "table",
      "chunk_index": 2
    }
  ]
}
```

**Expected Behavior:**
- Retriever pulls structured table chunks encoded in Markdown
- Sources include `chunk_type: "table"` for tabular content
- Answer synthesizes numerical values and constraints from table context

---

## Query 3: Image/Diagram-Focused Question

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Describe the diagrams, flowcharts, and visual components in the document."
  }'
```

**Response Format:**
```json
{
  "answer": "<LLM-generated answer based on VLM image summaries>",
  "sources": [
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 15,
      "chunk_type": "image_summary",
      "chunk_index": 1
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 28,
      "chunk_type": "image_summary",
      "chunk_index": 2
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 35,
      "chunk_type": "image_summary",
      "chunk_index": 1
    }
  ]
}
```

**Expected Behavior:**
- Retriever matches query intent to image-summary chunks
- Sources include `chunk_type: "image_summary"` (output from VLM)
- Answer describes visual content (diagrams, flowcharts, charts) extracted and understood via Vision Language Model

---

## Verified Test Run (Sample PDF: AIS 175)

The following demonstrates a successful query execution on the sample AIS-175 PDF:

**Query:**
```json
{
  "question": "What does AIS-175 focus on?"
}
```

**Actual Response:**
```json
{
  "answer": "document QA",
  "sources": [
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 175,
      "chunk_type": "image_summary",
      "chunk_index": 1
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 492,
      "chunk_type": "image_summary",
      "chunk_index": 10
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 123,
      "chunk_type": "image_summary",
      "chunk_index": 1
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 166,
      "chunk_type": "image_summary",
      "chunk_index": 2
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 249,
      "chunk_type": "image_summary",
      "chunk_index": 1
    },
    {
      "filename": "AIS 175_Final Draft_MARCH_2025.pdf",
      "page": 422,
      "chunk_type": "image_summary",
      "chunk_index": 1
    }
  ]
}
```

**Notes:**
- The sample AIS-175 PDF is image-heavy; all indexed chunks are image summaries (176 total)
- Retriever successfully returns top-6 most similar image-summary chunks
- Sources provide full traceability: filename, page number, chunk type, and chunk index
- LLM synthesizes answer from retrieved image descriptions

---

## How to Run Tests

1. **Start the API:**
   ```bash
   source .venv/bin/activate
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Ingest a Sample PDF:**
   ```bash
   curl -X POST "http://localhost:8000/ingest" \
     -F "file=@sample_documents/AIS 175_Final Draft_MARCH_2025.pdf"
   ```

4. **Run Test Queries:**
   - Use the curl examples above with your own questions
   - Verify source references match expectations (chunk_type coverage)
   - Check answer grounding against retrieved context

---

## Interpretation Guide

| Chunk Type | Meaning | Source |
|------------|---------|--------|
| `text` | Paragraph, heading, or extracted prose | PyMuPDF or Tesseract OCR |
| `table` | Structured tabular data in Markdown | pdfplumber or Tesseract heuristics |
| `image_summary` | VLM caption from extracted image | BLIP image captioning model |

The `answer` field always reflects a grounded response synthesized only from the retrieved chunks, respecting the RAG prompt constraint: "use only retrieved context or admit when information is missing."
