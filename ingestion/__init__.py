"""
ingestion
─────────
Document loading and chunking pipeline for Hybrid-RAG.

Exports
───────
    DocumentScraper  — loads PDF, DOCX, TXT, HTML, CSV from disk or URL
    DocumentChunker  — token-aware recursive chunking with overlap

Typical usage
─────────────
    from ingestion import DocumentScraper, DocumentChunker

    scraper = DocumentScraper()
    docs    = scraper.load_directory("data/raw/")

    chunker = DocumentChunker()
    chunks  = chunker.chunk_batch(docs)
    # chunks → list[Chunk], ready for embedding or BM25 indexing

Supported formats
─────────────────
    .pdf  .docx  .txt  .html  .htm  .csv  .xlsx
    Scanned PDFs fall back to OCR via unstructured + pytesseract.
"""
from ingestion.scraper import DocumentScraper
from ingestion.chunker import DocumentChunker

__all__ = [
    "DocumentScraper",
    "DocumentChunker",
]