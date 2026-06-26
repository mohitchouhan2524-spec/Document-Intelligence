"""
vector_rag/embed.py
───────────────────
Generates embeddings for chunks and upserts them into Qdrant.
Supports local (sentence-transformers) and OpenAI embeddings.
"""
from __future__ import annotations
import os
import uuid
from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    UpdateStatus,
)
from tqdm import tqdm

from src.config import get_settings
from src.models import Chunk
class EmbeddingEngine:
    """Wraps local or Google embedding model."""

    def __init__(self):
        cfg = get_settings()

        self.provider = cfg.embedding.provider
        self.dimension = cfg.embedding.dimension
        self.batch_size = cfg.embedding.batch_size

        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(cfg.embedding.local_model)
            logger.info(
                f"Loaded local embedding model: {cfg.embedding.local_model}"
            )

        elif self.provider == "google":
            from google import genai

            self._client = genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            self._model_name = cfg.embedding.google_model

            logger.info(
                f"Using Google embedding model: {cfg.embedding.google_model}"
            )

        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.provider}"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""

        if not texts:
            return []

        if self.provider == "local":
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            return vectors.tolist()

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self._client.models.embed_content(
                model=self._model_name,
                contents=batch,
            )

            embeddings.extend(
                emb.values for emb in response.embeddings
            )

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed_texts([query])[0]

class QdrantStore:
    """Manages Qdrant collection and upsert operations."""

    def __init__(self):
        cfg = get_settings()
        self.cfg = cfg.qdrant
        self.dim = cfg.embedding.dimension
        self.client = QdrantClient(host=self.cfg.host, port=self.cfg.port)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.cfg.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.cfg.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance[self.cfg.distance.upper()],
                ),
                on_disk_payload=True,
            )
            logger.info(f"Created Qdrant collection: {self.cfg.collection_name}")
        else:
            logger.debug(f"Qdrant collection already exists: {self.cfg.collection_name}")

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            
            # FIX 2: Deterministic ID generation using UUID5
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(chunk.chunk_id)))
            
            points.append(
                PointStruct(
                    id=deterministic_id,
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        result = self.client.upsert(
            collection_name=self.cfg.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} chunks to Qdrant")
        return len(points)

    def delete_collection(self):
        self.client.delete_collection(self.cfg.collection_name)
        logger.warning(f"Deleted Qdrant collection: {self.cfg.collection_name}")


class VectorIndexer:
    """Orchestrates embedding + Qdrant upsert."""

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.store = QdrantStore()

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        total = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embedder.embed_texts(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
            total += self.store.upsert_chunks(batch)
        return total