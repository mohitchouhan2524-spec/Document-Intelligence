"""
vector_rag
──────────
Dense vector retrieval path for Hybrid-RAG.
Handles embedding generation, Qdrant indexing, and ANN search.

Exports
───────
    EmbeddingEngine  — wraps local (BAAI/bge-large) or OpenAI embeddings
    QdrantStore      — manages Qdrant collection: create, upsert, delete
    VectorIndexer    — orchestrates embed → upsert in batches
    VectorRetriever  — ANN search + optional cross-encoder reranking

Typical usage
─────────────
    from vector_rag import VectorIndexer, VectorRetriever

    # Indexing (run once after chunking)
    indexer = VectorIndexer()
    indexer.index(chunks)

    # Retrieval (called by hybrid/fusion.py at query time)
    retriever = VectorRetriever()
    results   = retriever.retrieve_and_rerank("explain the SLA terms")
    # results → list[RetrievedChunk], sorted by reranker score

Configuration
─────────────
    embedding.provider      : "local" | "openai"   (configs/config.yaml)
    embedding.local_model   : "BAAI/bge-large-en-v1.5"
    qdrant.collection_name  : "doc_intelligence"
    retrieval.reranker_model: "BAAI/bge-reranker-large"
"""
from vector_rag.embed import EmbeddingEngine, QdrantStore, VectorIndexer
from vector_rag.retrieve import VectorRetriever

__all__ = [
    "EmbeddingEngine",
    "QdrantStore",
    "VectorIndexer",
    "VectorRetriever",
]