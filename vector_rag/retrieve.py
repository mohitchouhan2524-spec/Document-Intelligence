"""
vector_rag/retrieve.py
──────────────────────
Dense retrieval from Qdrant + cross-encoder reranking.
"""
from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import get_settings
from src.models import RetrievedChunk, RouteType
from vector_rag.embed import EmbeddingEngine


class VectorRetriever:
    def __init__(self):
        cfg = get_settings()
        self.cfg_qdrant = cfg.qdrant
        self.cfg_ret = cfg.retrieval
        self.client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        self.embedder = EmbeddingEngine()
        self._reranker = None  # lazy-loaded

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.cfg_qdrant.top_k
        query_vec = self.embedder.embed_query(query)

        qdrant_filter = None
        if metadata_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.cfg_qdrant.collection_name,
            query=query_vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks = [
            RetrievedChunk(
                chunk_id=r.payload.get("chunk_id", str(r.id)),
                doc_id=r.payload.get("doc_id", ""),
                content=r.payload.get("content", ""),
                score=r.score,
                source=RouteType.VECTOR,
                metadata={k: v for k, v in r.payload.items() if k not in ("content", "chunk_id", "doc_id")},
            )
            for r in results.points
        ]
        logger.debug(f"Vector retrieval: {len(chunks)} results for query '{query[:60]}...'")
        return chunks

    def retrieve_and_rerank(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve wider set then rerank with cross-encoder."""
        candidates = self.retrieve(query, top_k=(top_k or self.cfg_ret.reranker_top_k) * 3)
        return self._rerank(query, candidates, top_k or self.cfg_ret.reranker_top_k)

    def _rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        reranker = self._get_reranker()
        pairs = [(query, c.content) for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        for score, chunk in ranked[:top_k]:
            chunk.score = float(score)
        return [chunk for _, chunk in ranked[:top_k]]

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg_ret.reranker_model)
            logger.info(f"Loaded reranker: {self.cfg_ret.reranker_model}")
        return self._reranker