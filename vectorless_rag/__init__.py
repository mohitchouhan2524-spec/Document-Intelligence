"""
vectorless_rag
──────────────
Structured / keyword retrieval path for Hybrid-RAG.
No embeddings — retrieves via BM25, SQL exact lookups, and graph traversal.

Exports
───────
    ElasticsearchBM25    — production BM25 via Elasticsearch
    InMemoryBM25         — dev-mode BM25 via rank_bm25 (no infra needed)
    KnowledgeGraphBuilder — NER-based entity graph + traversal search

Route mapping (set by classifier)
──────────────────────────────────
    VectorlessMethod.BM25   → ElasticsearchBM25  or  InMemoryBM25
    VectorlessMethod.GRAPH  → KnowledgeGraphBuilder
    VectorlessMethod.SQL    → sql_retriever.SQLRetriever  (coming in Phase 4)

Typical usage
─────────────
    from vectorless_rag import ElasticsearchBM25, KnowledgeGraphBuilder

    # BM25 indexing (run once after chunking)
    bm25 = ElasticsearchBM25()
    bm25.index_chunks(chunks)

    # BM25 retrieval
    results = bm25.search("contracts mentioning force majeure")

    # Graph build (run once after chunking)
    graph = KnowledgeGraphBuilder()
    graph.build(chunks)

    # Graph retrieval
    results = graph.search("who approved the Q3 budget")

Dev mode (no Elasticsearch)
────────────────────────────
    from vectorless_rag import InMemoryBM25
    bm25 = InMemoryBM25()
    bm25.index_chunks(chunks)
    results = bm25.search("force majeure")
"""
from vectorless_rag.bm25 import ElasticsearchBM25, InMemoryBM25
from vectorless_rag.tree_builder import KnowledgeGraphBuilder

__all__ = [
    "ElasticsearchBM25",
    "InMemoryBM25",
    "KnowledgeGraphBuilder",
]