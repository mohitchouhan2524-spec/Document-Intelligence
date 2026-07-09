"""
vector_rag/retrieve.py
────────────────────────────────────────────────────────────────────────────────
Supabase pgvector retriever for Hybrid-RAG Document Intelligence.

Replaces the supabase-based VectorRetriever.
All searches are scoped to the authenticated user's chunks only.

Usage (called by fusion.py via _RetrieverRegistry)
───────────────────────────────────────────────────
    retriever = SupabaseVectorRetriever(
        user_id=user_id,
        supabase_client=sb,
    )
    results = retriever.retrieve_and_rerank("explain the SLA terms")
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import get_settings
from src.models import RetrievedChunk, RouteType
from vector_rag.embed import EmbeddingEngine, SupabaseVectorStore


class SupabaseVectorRetriever:
    """
    Dense retrieval from Supabase pgvector + optional cross-encoder reranking.

    Parameters
    ──────────
    user_id         : authenticated user's UUID (from Supabase auth)
    supabase_client : initialised supabase.Client instance
    """

    def __init__(self, user_id: str, supabase_client: Any):
        cfg             = get_settings()
        self.cfg_ret    = cfg.retrieval
        self.top_k      = cfg.retrieval.reranker_top_k
        self.embedder   = EmbeddingEngine()
        self.store      = SupabaseVectorStore(
            user_id=user_id,
            supabase_client=supabase_client,
        )
        self._reranker  = None   # lazy-loaded

    # ── Public API (matches the contract fusion.py expects) ───────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int  | None = None,
        metadata_filter: dict | None = None,  # unused — RLS handles isolation
    ) -> list[RetrievedChunk]:
        """
        Embed query → cosine similarity search in Supabase.
        Returns up to top_k chunks for this user.
        """
        top_k       = top_k or self.cfg_ret.reranker_top_k
        query_vec   = self.embedder.embed_query(query)
        results     = self.store.similarity_search(query_vec, top_k=top_k)

        if not results:
            logger.warning(
                f"Supabase vector search returned 0 results for user "
                f"{self.store.user_id[:8]}. "
                f"Has the user uploaded and ingested any documents?"
            )
        else:
            logger.debug(
                f"Supabase retrieval: {len(results)} results "
                f"for '{query[:60]}'"
            )
        return results

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve a wider candidate set, then rerank with a cross-encoder.
        Final list is trimmed to top_k after reranking.
        """
        final_k    = top_k or self.cfg_ret.reranker_top_k
        candidates = self.retrieve(query, top_k=final_k * 3)

        if not candidates:
            return []

        return self._rerank(query, candidates, final_k)

    # ── Internal 

    def _rerank(
        self,
        query:   str,
        chunks:  list[RetrievedChunk],
        top_k:   int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        reranker = self._get_reranker()
        pairs    = [(query, c.content) for c in chunks]
        scores   = reranker.predict(pairs)
        ranked   = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        for score, chunk in ranked[:top_k]:
            chunk.score = float(score)
        return [chunk for _, chunk in ranked[:top_k]]

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg_ret.reranker_model)
            logger.info(f"Loaded reranker: {self.cfg_ret.reranker_model}")
        return self._reranker


# ── Backwards-compatibility alias 
# fusion.py imports VectorRetriever — keep this alias so nothing else breaks
# until fusion.py is updated to pass user context.
class VectorRetriever(SupabaseVectorRetriever):
    """
    Alias kept for import compatibility.
    Prefer SupabaseVectorRetriever with explicit user_id in new code.
    """
    pass