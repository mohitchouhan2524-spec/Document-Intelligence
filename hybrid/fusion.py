"""
hybrid/fusion.py
Central orchestrator for the Hybrid-RAG Document Intelligence pipeline.

End-to-end flow
───────────────
    query
      │
      ▼
    QueryClassifier.classify()          ← rules.py → train.py
      │
      ├─ RouteType.VECTOR    ──────────▶ VectorRetriever.retrieve_and_rerank()
      │
      ├─ RouteType.VECTORLESS
      │       ├─ VectorlessMethod.BM25  ▶ BM25Retriever.search()
      │       ├─ VectorlessMethod.SQL   ▶ SQLRetriever.search()
      │       └─ VectorlessMethod.GRAPH ▶ KnowledgeGraphBuilder.search()
      │
      └─ RouteType.HYBRID   ──────────▶ vector + vectorless in parallel
                                          └─ RRFusion.fuse()
      │
      ▼
    ContextAssembler.build()            deduplicate · trim · format
      │
      ▼
    LLMGenerator.generate()             GROQ/GEMINI
      │
      ▼
    RAGResponse

Public API
──────────
    from hybrid.fusion import HybridPipeline

    pipeline = HybridPipeline()
    response = pipeline.query("compare SLA terms across all active vendors")

    response.answer           → str
    response.route_used       → RouteType
    response.retrieved_chunks → list[RetrievedChunk]
    response.latency_ms       → float

Config (configs/config.yaml)
────────────────────────────
    retrieval.fusion_method   : "rrf" | "linear"
    retrieval.rrf_k           : 60
    retrieval.reranker_top_k  : 5
    llm.provider              : "groq" | "gemini"
    llm.model                 : "qwen/qwen3.6-27b"
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from google.genai import types
from loguru import logger

from src.config import get_settings
from src.models import (
    QueryIntent,
    RAGResponse,
    RetrievedChunk,
    RouteType,
    VectorlessMethod,
)
from classifier.train import predict_intent


# ── Retriever registry 
# Lazy-loaded singletons: only initialised when actually needed.
# Avoids loading Qdrant/ES/spaCy clients if they're not used.

class _RetrieverRegistry:
    """Lazy singleton registry — imports and constructs retrievers on first use."""

    def __init__(self):
        self._vector:    Any = None
        self._bm25:      Any = None
        self._graph:     Any = None
        self._sql:       Any = None

    @property
    def vector(self):
        if self._vector is None:
            from vector_rag.retrieve import VectorRetriever
            self._vector = VectorRetriever()
            logger.debug("VectorRetriever initialised")
        return self._vector

    @property
    def bm25(self):
        if self._bm25 is None:
            try:
                from vectorless_rag.bm25 import ElasticsearchBM25
                self._bm25 = ElasticsearchBM25()
                logger.debug("ElasticsearchBM25 initialised")
            except Exception as e:
                logger.warning(f"Elasticsearch unavailable ({e}), falling back to InMemoryBM25")
                from vectorless_rag.bm25 import InMemoryBM25
                self._bm25 = InMemoryBM25()
        return self._bm25

    @property
    def graph(self):
        if self._graph is None:
            from vectorless_rag.tree_builder import KnowledgeGraphBuilder
            self._graph = KnowledgeGraphBuilder()
            logger.debug("KnowledgeGraphBuilder initialised")
        return self._graph

    @property
    def sql(self):
        if self._sql is None:
            try:
                from vectorless_rag.sql_retriever import SQLRetriever
                self._sql = SQLRetriever()
                logger.debug("SQLRetriever initialised")
            except ImportError:
                logger.warning("sql_retriever.py not yet written — SQL queries will return empty")
                self._sql = _NullRetriever("sql")
        return self._sql


class _NullRetriever:
    """Safe stub for retrievers that aren't implemented yet."""

    def __init__(self, name: str):
        self._name = name

    def search(self, query: str, top_k: int = 10, **_) -> list[RetrievedChunk]:
        logger.warning(f"NullRetriever({self._name}): returning empty — retriever not implemented")
        return []

    def retrieve(self, query: str, **_) -> list[RetrievedChunk]:
        return self.search(query)

    def retrieve_and_rerank(self, query: str, **_) -> list[RetrievedChunk]:
        return self.search(query)


# ── RRF fusion 

class RRFusion:
    """
    Reciprocal Rank Fusion — merges two ranked lists into one.

    Score formula:  RRF(d) = Σ  1 / (k + rank_i(d))
                          i ∈ {vector, vectorless}

    k = 60 (standard default, dampens the effect of very high ranks).
    Chunks appearing in both lists get a doubled contribution.
    Chunks unique to one list still appear, just with a lower score.

    Reference: Cormack, Clarke & Buettcher (SIGIR 2009).
    """

    def __init__(self, k: int | None = None):
        self.k = k or get_settings().retrieval.rrf_k

    def fuse(
        self,
        vector_chunks: list[RetrievedChunk],
        vectorless_chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or get_settings().retrieval.reranker_top_k

        # chunk_id → accumulated RRF score
        rrf_scores: dict[str, float] = defaultdict(float)
        # chunk_id → chunk object (keep the one with higher original score)
        chunk_map:  dict[str, RetrievedChunk] = {}
        for rank, chunk in enumerate(vector_chunks, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.k + rank)
            if chunk.chunk_id not in chunk_map or chunk.score > chunk_map[chunk.chunk_id].score:
                chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(vectorless_chunks, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.k + rank)
            if chunk.chunk_id not in chunk_map or chunk.score > chunk_map[chunk.chunk_id].score:
                chunk_map[chunk.chunk_id] = chunk

        ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        fused = []
        for cid in ranked_ids:
            chunk = chunk_map[cid]
            chunk.score = round(rrf_scores[cid], 6)
            fused.append(chunk)

        logger.debug(
            f"RRF fusion: {len(vector_chunks)} vector + {len(vectorless_chunks)} "
            f"vectorless → {len(fused)} fused (k={self.k})"
        )
        return fused


class LinearFusion:
    """
    Weighted linear combination of normalised scores.
    Useful when you want to bias toward one retrieval path.

    vector_weight + vectorless_weight should sum to 1.0.
    """

    def __init__(self, vector_weight: float = 0.5, vectorless_weight: float = 0.5):
        self.vw  = vector_weight
        self.vlw = vectorless_weight

    def _normalise(self, chunks: list[RetrievedChunk]) -> dict[str, float]:
        if not chunks:
            return {}
        scores = [c.score for c in chunks]
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s or 1.0
        return {c.chunk_id: (c.score - min_s) / span for c in chunks}

    def fuse(
        self,
        vector_chunks:     list[RetrievedChunk],
        vectorless_chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or get_settings().retrieval.reranker_top_k

        v_norm  = self._normalise(vector_chunks)
        vl_norm = self._normalise(vectorless_chunks)

        chunk_map: dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in vector_chunks + vectorless_chunks
        }
        combined: dict[str, float] = defaultdict(float)
        for cid, s in v_norm.items():
            combined[cid] += self.vw * s
        for cid, s in vl_norm.items():
            combined[cid] += self.vlw * s

        ranked = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        result = []
        for cid in ranked:
            chunk = chunk_map[cid]
            chunk.score = round(combined[cid], 6)
            result.append(chunk)
        return result


# ── Context assembler 

class ContextAssembler:
    """
    Takes a ranked list of RetrievedChunks and builds the context
    string that goes into the LLM prompt.

    Steps:
      1. Deduplicate by chunk_id (keep highest score)
      2. Re-sort by score descending
      3. Trim to reranker_top_k chunks
      4. Format each chunk with source metadata
    """

    # Approximate characters per token (conservative)
    _CHARS_PER_TOKEN = 4
    # Leave room for system prompt + query + answer
    _CONTEXT_TOKEN_BUDGET = 6_000

    def build(self, chunks: list[RetrievedChunk], max_chunks: int | None = None) -> tuple[str, list[RetrievedChunk]]:
        """
        Returns (context_string, deduplicated_chunks).
        context_string is ready to insert into the LLM prompt.
        """
        cfg = get_settings().retrieval
        max_chunks = max_chunks or cfg.reranker_top_k

        # 1. Deduplicate
        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
                seen[chunk.chunk_id] = chunk

        # 2. Sort + trim
        ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:max_chunks]

        # 3. Token budget guard — drop chunks that exceed budget
        budget_chars = self._CONTEXT_TOKEN_BUDGET * self._CHARS_PER_TOKEN
        kept: list[RetrievedChunk] = []
        total_chars = 0
        for chunk in ranked:
            chunk_len = len(chunk.content)
            if total_chars + chunk_len > budget_chars:
                logger.debug(f"Context budget hit at chunk {len(kept)+1}/{len(ranked)}, stopping")
                break
            kept.append(chunk)
            total_chars += chunk_len

        # 4. Format
        sections: list[str] = []
        for i, chunk in enumerate(kept, start=1):
            # 2. Add safety fallback (chunk.metadata or {}) in case metadata is None
            metadata = chunk.metadata or {}
            source   = metadata.get("filename") or metadata.get("source") or chunk.doc_id
    
    # Assumes chunk.source is an Enum. If it's just a string, remove .value
            retriever = chunk.source.value 
    
            header   = f"[{i}] Source: {source}  |  Retriever: {retriever}  |  Score: {chunk.score:.4f}"
            content  = chunk.content.strip()
    
            sections.append(f"{header}\n{content}")
            total_chars += len(content)  # 3. Update character count for the logger

            context = "\n\n---\n\n".join(sections) if sections else "No relevant context found."
            logger.debug(f"Context assembled: {len(kept)} chunks, ~{total_chars // getattr(self, '_CHARS_PER_TOKEN', 4)} tokens")
            return context, kept


# ── LLM generator 

class LLMGenerator:
    """
    Wraps GROQ and gemini generation.
    Provider and model are set in configs/config.yaml → llm section.
    """

    _PROMPT_TEMPLATE = """\
Use the following retrieved context to answer the question.
Cite source numbers [1], [2] etc. when referencing specific content.
If the context does not contain enough information, say so explicitly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    _SUPPORTED = {"google_genai", "groq"}

    def __init__(self):
        cfg = get_settings()
        self.llm_cfg = cfg.llm
        self._client: Any = None
        
        provider = self.llm_cfg.provider
        if provider not in self._SUPPORTED:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}\n"
                f"Supported providers: {sorted(self._SUPPORTED)}\n"
                f"Set llm.provider in configs/config.yaml"
            )

    def _get_client(self):
        """Initializes and returns the correct API client based on the provider."""
        if self._client is not None:
            return self._client
            
        if self.llm_cfg.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            self._client = Groq(api_key=api_key)
            
        elif self.llm_cfg.provider == "google_genai":
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            self._client = genai.Client(api_key=api_key)
            
        else:
            raise ValueError(f"Unknown provider: {self.llm_cfg.provider}")
            
        return self._client

    def _generate_groq(self, client: Any, user_content: str) -> str:
        system = self.llm_cfg.system_prompt or (
            "You are a Document Intelligence assistant. Answer questions accurately "
            "based solely on the provided context. If the context is insufficient, "
            "say so clearly. Never hallucinate facts not present in the context."
        )
        
        # Groq uses the OpenAI-compatible Chat Completions API
        resp = client.chat.completions.create(
            model=self.llm_cfg.model,
            max_tokens=self.llm_cfg.max_tokens,
            temperature=self.llm_cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content}
            ],
        )
        return resp.choices[0].message.content

    def _generate_genai(self, client: Any, user_content: str) -> str:
        system = self.llm_cfg.system_prompt or (
            "You are a Document Intelligence assistant. Answer only from provided context."
        )
        
        # Google GenAI uses the models.generate_content API
        resp = client.models.generate_content(
            model=self.llm_cfg.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.llm_cfg.temperature,
                max_output_tokens=self.llm_cfg.max_tokens,
            )
        )
        return resp.text

    def generate(self, query: str, context: str) -> str:
        """Routes the generation request to the correct configured LLM provider."""
        client = self._get_client()
        
        # Stitch the retrieved context and user query together into one prompt
        user_content = self._PROMPT_TEMPLATE.format(context=context, query=query)
        
        try:
            if self.llm_cfg.provider == "groq":
                return self._generate_groq(client, user_content)
            elif self.llm_cfg.provider == "google_genai":
                return self._generate_genai(client, user_content)
            else:
                raise ValueError(f"Generation not implemented for provider: {self.llm_cfg.provider}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[Generation error: {e}]"

# ── Main pipeline

class HybridPipeline:
    """
    End-to-end Hybrid-RAG pipeline for Document Intelligence.

    Usage
        pipeline = HybridPipeline()
        response = pipeline.query("compare SLA terms across all active vendors")

        print(response.answer)
        print(response.route_used)
        for chunk in response.retrieved_chunks:
            print(chunk.score, chunk.content[:80])

    Constructor kwargs
        fusion_method   : "rrf" | "linear"  (overrides config.yaml)
        vector_weight   : float  (only used when fusion_method="linear")
        vectorless_weight: float (only used when fusion_method="linear")
        generate        : bool   (set False to skip LLM — retrieval only)
    """

    def __init__(
        self,
        fusion_method:      str   | None = None,
        vector_weight:      float        = 0.5,
        vectorless_weight:  float        = 0.5,
        generate:           bool         = True,
    ):
        cfg = get_settings()
        method = fusion_method or cfg.retrieval.fusion_method

        self._registry   = _RetrieverRegistry()
        self._assembler  = ContextAssembler()
        self._generator  = LLMGenerator() if generate else None
        self._do_generate = generate

        if method == "linear":
            self._fuser: RRFusion | LinearFusion = LinearFusion(vector_weight, vectorless_weight)
        else:
            self._fuser = RRFusion(k=cfg.retrieval.rrf_k)

        logger.info(f"HybridPipeline ready — fusion: {method}, generate: {generate}")

    # ── Public entry point 

    def query(
        self,
        query: str,
        top_k:          int  | None = None,
        metadata_filter: dict | None = None,
        return_context: bool        = False,
    ) -> RAGResponse:
        """
        Run the full pipeline and return a RAGResponse.

        Parameters
        ──────────
        query           : the user's question
        top_k           : override number of final chunks (default from config)
        metadata_filter : pass-through filter to vector retriever
                          e.g. {"filetype": "pdf", "source": "contracts/"}
        return_context  : if True, raw context string is added to response.metadata
        """
        t_start = time.perf_counter()

        # 1. Classify
        intent = self._classify(query)
        logger.info(
            f"Route: {intent.route.value} "
            f"(method={intent.vectorless_method}, "
            f"confidence={intent.confidence:.2f}) | '{query[:70]}'"
        )

        # 2. Retrieve
        chunks = self._retrieve(query, intent, top_k, metadata_filter)

        # 3. Assemble context
        context, final_chunks = self._assembler.build(chunks, max_chunks=top_k)

        # 4. Generate
        answer = "[generation disabled]"
        if self._do_generate and self._generator:
            answer = self._generator.generate(query, context)

        latency_ms = (time.perf_counter() - t_start) * 1000

        extra: dict = {
            "intent_reasoning": intent.reasoning,
            "classifier_confidence": intent.confidence,
            "chunks_retrieved": len(chunks),
            "chunks_used": len(final_chunks),
        }
        if return_context:
            extra["context"] = context

        response = RAGResponse(
            query=query,
            route_used=intent.route,
            retrieved_chunks=final_chunks,
            answer=answer,
            latency_ms=round(latency_ms, 2),
            metadata=extra,
        )
        logger.info(f"Pipeline complete — {latency_ms:.0f}ms | route={intent.route.value} | chunks={len(final_chunks)}")
        return response
    # ── Retrieve-only shortcut (no LLM) ─

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks without calling the LLM. Useful for evaluation."""
        intent = self._classify(query)
        chunks = self._retrieve(query, intent, top_k, metadata_filter)
        _, final = self._assembler.build(chunks, max_chunks=top_k)
        return final

    # ── Internal steps

    def _classify(self, query: str) -> QueryIntent:
        try:
            return predict_intent(query)
        except Exception as e:
            logger.warning(f"Classifier failed ({e}), defaulting to HYBRID")
            return QueryIntent(
                route=RouteType.HYBRID,
                confidence=0.0,
                reasoning=f"Classifier exception: {e}",
            )

    def _retrieve(
        self,
        query: str,
        intent: QueryIntent,
        top_k: int | None,
        metadata_filter: dict | None,
    ) -> list[RetrievedChunk]:
        cfg   = get_settings().retrieval
        top_k = top_k or cfg.reranker_top_k

        if intent.route == RouteType.VECTOR:
            return self._retrieve_vector(query, top_k, metadata_filter)

        if intent.route == RouteType.VECTORLESS:
            return self._retrieve_vectorless(query, intent, top_k)

        # HYBRID — run both paths in parallel, fuse results
        return self._retrieve_hybrid(query, intent, top_k, metadata_filter)

    def _retrieve_vector(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[RetrievedChunk]:
        try:
            return self._registry.vector.retrieve_and_rerank(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def _retrieve_vectorless(
        self,
        query: str,
        intent: QueryIntent,
        top_k: int,
    ) -> list[RetrievedChunk]:
        method = intent.vectorless_method or VectorlessMethod.BM25

        try:
            if method == VectorlessMethod.BM25:
                return self._registry.bm25.search(query, top_k=top_k)
            if method == VectorlessMethod.SQL:
                return self._registry.sql.search(query, top_k=top_k)
            if method == VectorlessMethod.GRAPH:
                return self._registry.graph.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Vectorless retrieval ({method}) failed: {e}")

        return []

    def _retrieve_hybrid(
        self,
        query: str,
        intent: QueryIntent,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[RetrievedChunk]:
        """
        Run vector + vectorless in parallel using a thread pool,
        then fuse with RRF (or linear, depending on config).
        """
        vector_chunks:     list[RetrievedChunk] = []
        vectorless_chunks: list[RetrievedChunk] = []

        # Determine which vectorless retriever to use in the hybrid path.
        # Default to BM25 when method is not specified (most general-purpose).
        vl_method = intent.vectorless_method or VectorlessMethod.BM25

        def _run_vector():
            try:
                return self._registry.vector.retrieve_and_rerank(query, top_k=top_k * 2)
            except Exception as e:
                logger.error(f"Hybrid/vector path failed: {e}")
                return []

        def _run_vectorless():
            try:
                if vl_method == VectorlessMethod.BM25:
                    return self._registry.bm25.search(query, top_k=top_k * 2)
                if vl_method == VectorlessMethod.SQL:
                    return self._registry.sql.search(query, top_k=top_k * 2)
                if vl_method == VectorlessMethod.GRAPH:
                    return self._registry.graph.search(query, top_k=top_k * 2)
                return []
            except Exception as e:
                logger.error(f"Hybrid/vectorless path ({vl_method}) failed: {e}")
                return []

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vector     = pool.submit(_run_vector)
            f_vectorless = pool.submit(_run_vectorless)
            vector_chunks     = f_vector.result()
            vectorless_chunks = f_vectorless.result()

        logger.debug(
            f"Hybrid retrieval: {len(vector_chunks)} vector "
            f"+ {len(vectorless_chunks)} vectorless before fusion"
        )

        if not vector_chunks and not vectorless_chunks:
            logger.warning("Both retrieval paths returned empty results")
            return []

        # Fuse and return top_k
        return self._fuser.fuse(vector_chunks, vectorless_chunks, top_k=top_k)


# ── Convenience function 

def run_query(
    query: str,
    generate: bool = True,
    return_context: bool = False,
    **pipeline_kwargs,
) -> RAGResponse:
    """
    One-shot helper — constructs a pipeline and runs a single query.
    Use HybridPipeline() directly when running multiple queries
    to avoid re-initialising retrievers on every call.

    Example
    ───────
        from hybrid.fusion import run_query
        resp = run_query("what are the key risks in this filing?")
        print(resp.answer)
    """
    pipeline = HybridPipeline(generate=generate, **pipeline_kwargs)
    return pipeline.query(query, return_context=return_context)