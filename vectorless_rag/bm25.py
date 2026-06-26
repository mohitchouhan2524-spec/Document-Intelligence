"""
vectorless_rag/bm25.py
──────────────────────
Two-tier BM25 retrieval:
  1. Elasticsearch (production) — full inverted index with filters
  2. rank_bm25 (dev fallback) — in-memory, no infra needed
"""
from __future__ import annotations
import pickle
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.models import Chunk, RetrievedChunk, RouteType


class ElasticsearchBM25:
    """Production BM25 via Elasticsearch."""

    def __init__(self):
        from elasticsearch import Elasticsearch
        cfg = get_settings().elasticsearch
        self.client = Elasticsearch(f"http://{cfg.host}:{cfg.port}")
        self.index = cfg.index_name
        self.top_k = cfg.top_k
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(
                index=self.index,
                body={
                    "settings": {"analysis": {"analyzer": {"default": {"type": "english"}}}},
                    "mappings": {
                        "properties": {
                            "content": {"type": "text", "analyzer": "english"},
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "filename": {"type": "keyword"},
                            "filetype": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                        }
                    },
                },
            )
            logger.info(f"Created ES index: {self.index}")

    def index_chunks(self, chunks: list[Chunk]) -> int:
        from elasticsearch.helpers import bulk
        actions = [
            {
                "_index": self.index,
                "_id": chunk.chunk_id,
                "_source": {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            }
            for chunk in chunks
        ]
        success, _ = bulk(self.client, actions)
        self.client.indices.refresh(index=self.index)
        logger.info(f"Indexed {success} chunks to Elasticsearch")
        return success

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        must_clauses = [{"match": {"content": {"query": query, "operator": "or"}}}]
        filter_clauses = []
        if filters:
            for k, v in filters.items():
                filter_clauses.append({"term": {k: v}})

        body = {
            "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
            "size": top_k,
        }
        resp = self.client.search(index=self.index, body=body)
        hits = resp["hits"]["hits"]

        return [
            RetrievedChunk(
                chunk_id=h["_source"].get("chunk_id", h["_id"]),
                doc_id=h["_source"].get("doc_id", ""),
                content=h["_source"].get("content", ""),
                score=h["_score"],
                source=RouteType.VECTORLESS,
                metadata={k: v for k, v in h["_source"].items() if k not in ("content", "chunk_id", "doc_id")},
            )
            for h in hits
        ]


class InMemoryBM25:
    """
    Dev-mode BM25 using rank_bm25. No Elasticsearch needed.
    Serialises corpus to disk for reuse.
    """

    def __init__(self, index_path: str = "data/indexes/bm25_index.pkl"):
        from rank_bm25 import BM25Okapi
        self._BM25 = BM25Okapi
        self.index_path = Path(index_path)
        self._chunks: list[Chunk] = []
        self._bm25 = None
        if self.index_path.exists():
            self._load()

    def index_chunks(self, chunks: list[Chunk]):
        self._chunks = chunks
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25 = self._BM25(tokenized)
        self._save()
        logger.info(f"Built in-memory BM25 index over {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if self._bm25 is None:
            logger.warning("BM25 index not built. Call index_chunks() first.")
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i].chunk_id,
                doc_id=self._chunks[i].doc_id,
                content=self._chunks[i].content,
                score=float(scores[i]),
                source=RouteType.VECTORLESS,
                metadata=self._chunks[i].metadata,
            )
            for i in top_indices
            if scores[i] > 0
        ]

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)

    def _load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._bm25 = data["bm25"]
        logger.info(f"Loaded BM25 index: {len(self._chunks)} chunks")