"""
hybrid
──────
Central orchestrator for the Hybrid-RAG pipeline.

Imports the query classifier, routes to vector and/or vectorless
retrievers, fuses results, and returns a ranked context window
ready for LLM generation.

Exports (available after fusion.py is written)
──────────────────────────────────────────────
    HybridPipeline — end-to-end query → RAGResponse

Routing logic (implemented in fusion.py)
─────────────────────────────────────────
    RouteType.VECTOR      → VectorRetriever only
    RouteType.VECTORLESS  → ElasticsearchBM25 | SQLRetriever | KnowledgeGraphBuilder
    RouteType.HYBRID      → both paths in parallel → RRF fusion → rerank

Typical usage (once fusion.py exists)
──────────────────────────────────────
    from hybrid import HybridPipeline

    pipeline = HybridPipeline()
    response = pipeline.query("compare SLA terms across all active vendors")
    # response.route_used       → RouteType.HYBRID
    # response.answer           → "..."
    # response.retrieved_chunks → [RetrievedChunk, ...]

Status
──────
    fusion.py — not yet written (Phase 4).
    Import will raise ImportError until fusion.py exists.
"""
try:
    from hybrid.fusion import HybridPipeline
    __all__ = ["HybridPipeline"]
except ImportError:
    # fusion.py not yet written — safe to import this __init__ without error
    __all__ = []