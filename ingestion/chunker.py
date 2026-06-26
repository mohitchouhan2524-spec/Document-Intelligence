"""
ingestion/chunker.py
────────────────────
Token-aware chunking with overlap. Produces Chunk objects
with metadata preserved from the parent Document.
"""
from __future__ import annotations
import hashlib
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.models import Chunk, Document
from src.config import get_settings


def _chunk_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DocumentChunker:
    """
    Splits documents into overlapping chunks.
    Uses tiktoken for accurate token counts.
    """

    def __init__(self):
        cfg = get_settings()
        self.chunk_size = cfg.embedding.batch_size   # reuse from config
        # read chunking params directly from yaml via raw load
        import yaml, pathlib
        raw = yaml.safe_load(
            (pathlib.Path(__file__).parent.parent / "configs" / "config.yaml").read_text()
        )
        ing = raw.get("ingestion", {})
        self.chunk_size = ing.get("chunk_size", 512)
        self.chunk_overlap = ing.get("chunk_overlap", 64)
        self.min_chunk_size = ing.get("min_chunk_size", 50)

        self._enc = tiktoken.get_encoding("cl100k_base")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,length_function=lambda text: len(self._enc.encode(text)),
            separators=["\n\n","\n",". "," ","",],
        )
    def chunk(self, doc: Document) -> list[Chunk]:
        raw_chunks = self._splitter.split_text(doc.content)
        chunks = []
        for i, text in enumerate(raw_chunks):
            token_count = len(self._enc.encode(text))
            if token_count < self.min_chunk_size:
                continue
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, i),
                    doc_id=doc.doc_id,
                    content=text,
                    chunk_index=i,
                    token_count=token_count,
                    metadata={**doc.metadata, "source": doc.source},
                )
            )
        logger.debug(f"Doc {doc.doc_id}: {len(raw_chunks)} raw → {len(chunks)} chunks kept")
        return chunks

    def chunk_batch(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        logger.info(f"Chunked {len(docs)} docs → {len(all_chunks)} total chunks")
        return all_chunks