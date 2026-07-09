"""
vector_rag/embed.py
Embedding engine + Supabase pgvector store for Hybrid-RAG.
Replaces supabase with Supabase pgvector so no local vector DB is needed.
Supabase schema required (run once in Supabase SQL editor)
    -- Enable pgvector extension
    create extension if not exists vector;
    -- Documents table (per-user, max 3 PDFs tracked)
    create table if not exists documents (
        id          uuid primary key default gen_random_uuid(),
        user_id     uuid not null references auth.users(id) on delete cascade,
        doc_id      text not null,
        filename    text,
        filetype    text,
        source      text,
        created_at  timestamptz default now(),
        unique(user_id, doc_id)
    );

    -- Chunks + embeddings table
    create table if not exists chunks (
        id          uuid primary key default gen_random_uuid(),
        user_id     uuid not null references auth.users(id) on delete cascade,
        doc_id      text not null,
        chunk_id    text not null,
        content     text not null,
        chunk_index integer,
        token_count integer,
        embedding   vector(1024),     -- adjust dim to match your model
        metadata    jsonb,
        created_at  timestamptz default now(),
        unique(user_id, chunk_id)
    );

    -- IVFFlat index for fast ANN search
    create index if not exists chunks_embedding_idx
        on chunks using ivfflat (embedding vector_cosine_ops)
        with (lists = 100);

    -- Row Level Security: users only see their own data
    alter table documents enable row level security;
    alter table chunks     enable row level security;

    create policy "user_documents" on documents
        for all using (auth.uid() = user_id);

    create policy "user_chunks" on chunks
        for all using (auth.uid() = user_id);

    -- Stored function for vector similarity search
    create or replace function match_chunks(
        query_embedding vector,
        match_user_id   uuid,
        match_count     int default 5
    )
    returns table (
        chunk_id    text,
        doc_id      text,
        content     text,
        chunk_index integer,
        token_count integer,
        metadata    jsonb,
        similarity  float
    )
    language sql stable
    as $$
        select
            chunk_id,
            doc_id,
            content,
            chunk_index,
            token_count,
            metadata,
            1 - (embedding <=> query_embedding) as similarity
        from chunks
        where user_id = match_user_id
        order by embedding <=> query_embedding
        limit match_count;
    $$;
Public API
    from vector_rag.embed import EmbeddingEngine, SupabaseVectorStore, VectorIndexer

    # Index (called from ingest pipeline or Streamlit upload)
    engine  = EmbeddingEngine()
    store   = SupabaseVectorStore(user_id="uuid-...", supabase_client=sb)
    indexer = VectorIndexer(store=store, engine=engine)
    indexer.index(chunks)

    # Retrieve (called from fusion.py)
    results = store.similarity_search(query_embedding, top_k=5)
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.models import Chunk, RetrievedChunk, RouteType
from src.config import get_settings


# ── Embedding engine (unchanged — local or OpenAI)

class EmbeddingEngine:
    """Wraps local (sentence-transformers) or OpenAI embedding model."""

    def __init__(self):
        cfg = get_settings()
        self.provider   = cfg.embedding.provider
        self.dimension  = cfg.embedding.dimension
        self.batch_size = cfg.embedding.batch_size

        if self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(cfg.embedding.local_model)
            logger.info(f"Loaded local embedding model: {cfg.embedding.local_model}")
        else:
            from openai import OpenAI
            import os
            self._client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            self._model_name = cfg.embedding.openai_model
            logger.info(f"Using OpenAI embedding model: {cfg.embedding.openai_model}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vecs.tolist()
        else:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp  = self._client.embeddings.create(
                    model=self._model_name, input=batch
                )
                all_embeddings.extend([e.embedding for e in resp.data])
            return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


# ── Supabase vector store
class SupabaseVectorStore:
    """
    pgvector-backed store on Supabase.

    All reads/writes are scoped to user_id so different users never
    see each other's data (enforced both in Python and via RLS policies).

    Per-user PDF history cap
    ────────────────────────
    Each user may have at most MAX_PDFS distinct documents stored.
    When a new document is uploaded and the cap is reached, the oldest
    document (by created_at) and all its chunks are deleted first.
    """

    MAX_PDFS = 3   # maximum documents retained per user

    def __init__(self, user_id: str, supabase_client: Any):
        self.user_id = user_id
        self._sb     = supabase_client
        logger.info(f"SupabaseVectorStore ready for user {user_id[:8]}…")

    # ── Document registry
    def register_document(
        self,
        doc_id:   str,
        filename: str,
        filetype: str,
        source:   str,
    ) -> None:
        """"Upsert a document record for this user.
        Enforces the MAX_PDFS cap by deleting the oldest doc if needed."""
        self._enforce_pdf_cap()
        result=self._sb.table("documents").upsert({
            "user_id":  self.user_id,
            "doc_id":   doc_id,
            "filename": filename,
            "filetype": filetype,
            "source":   source,
        }, on_conflict="user_id,doc_id").execute()
        print(result)
        logger.debug(f"Registered document: {filename} ({doc_id[:8]})")

    def _enforce_pdf_cap(self) -> None:
        """
        If the user already has MAX_PDFS documents, delete the oldest one
        (and cascade-delete its chunks via ON DELETE CASCADE on the FK).
        """
        resp = (
            self._sb.table("documents")
            .select("doc_id, filename, created_at")
            .eq("user_id", self.user_id)
            .order("created_at", desc=False)   # oldest first
            .execute()
        )
        docs = resp.data or []
        if len(docs) >= self.MAX_PDFS:
            oldest    = docs[0]
            oldest_id = oldest["doc_id"]
            # Delete chunks first (if no cascade), then document
            self._sb.table("chunks").delete().eq(
                "user_id", self.user_id
            ).eq("doc_id", oldest_id).execute()
            self._sb.table("documents").delete().eq(
                "user_id", self.user_id
            ).eq("doc_id", oldest_id).execute()
            logger.info(
                f"PDF cap ({self.MAX_PDFS}) reached — evicted oldest: "
                f"{oldest.get('filename', oldest_id)}"
            )

    def list_documents(self) -> list[dict]:
        """Return all documents for this user, newest first."""
        resp = (
            self._sb.table("documents")
            .select("doc_id, filename, filetype, source, created_at")
            .eq("user_id", self.user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return resp.data or []

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks for this user."""
        self._sb.table("chunks").delete().eq(
            "user_id", self.user_id
        ).eq("doc_id", doc_id).execute()
        self._sb.table("documents").delete().eq(
            "user_id", self.user_id
        ).eq("doc_id", doc_id).execute()
        logger.info(f"Deleted document {doc_id[:8]} for user {self.user_id[:8]}")

    # ── Chunk upsert
    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Upsert chunks (with embeddings) into Supabase.
        Each chunk must have chunk.embedding set before calling this.
        """
        if not chunks:
            return 0

        rows = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} has no embedding set")
            rows.append({
                "user_id":     self.user_id,
                "doc_id":      c.doc_id,
                "chunk_id":    c.chunk_id,
                "content":     c.content,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "embedding":   c.embedding,   # list[float] → pgvector
                "metadata":    json.dumps(c.metadata),
            })
        # Supabase upserts in batches of 50 to stay within request size limits
        batch_size = 50
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._sb.table("chunks").upsert(
                batch, on_conflict="user_id,chunk_id"
            ).execute()
            total += len(batch)

        logger.info(f"Upserted {total} chunks for user {self.user_id[:8]}")
        return total
    # ── Similarity search

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Call the match_chunks stored function via Supabase RPC.
        Returns top_k chunks ordered by cosine similarity.
        """
        try:
            resp = self._sb.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_user_id":   self.user_id,
                    "match_count":     top_k,
                },
            ).execute()
        except Exception as e:
            logger.error(f"Supabase similarity_search failed: {e}")
            return []

        rows = resp.data or []
        results = []
        for row in rows:
            meta = {}
            if row.get("metadata"):
                try:
                    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                except Exception:
                    meta = {}

            results.append(RetrievedChunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                content=row["content"],
                score=float(row.get("similarity", 0.0)),
                source=RouteType.VECTOR,
                metadata=meta,
            ))

        logger.debug(
            f"Supabase search: {len(results)} results for user {self.user_id[:8]}"
        )
        return results

    # ── Stats
    def stats(self) -> dict:
        """Return document + chunk counts for this user."""
        docs   = self.list_documents()
        chunks = (
            self._sb.table("chunks")
            .select("chunk_id", count="exact")
            .eq("user_id", self.user_id)
            .execute()
        )
        return {
            "documents": len(docs),
            "chunks":    chunks.count or 0,
            "doc_list":  docs,
            "max_pdfs":  self.MAX_PDFS,
        }
# ── Indexer — orchestrates embed → upsert 

class VectorIndexer:
    """
    Orchestrates embedding generation + Supabase upsert.

    Usage
    ─────
        indexer = VectorIndexer(store=store, engine=engine)
        indexer.index(chunks, doc=doc)   # registers doc + embeds chunks
    """

    def __init__(
        self,
        store:  SupabaseVectorStore | None = None,
        engine: EmbeddingEngine     | None = None,
    ):
        self.engine = engine or EmbeddingEngine()
        self.store  = store   # must be set before calling index()

    def index(
        self,
        chunks:   list[Chunk],
        doc:      Any   = None,       # ingestion.scraper.Document
        batch_size: int = 32,
    ) -> int:
        """
        Embed chunks in batches and upsert to Supabase.
        Optionally registers the parent document first.
        """
        if self.store is None:
            raise ValueError(
                "VectorIndexer.store is not set. "
                "Pass a SupabaseVectorStore when constructing VectorIndexer."
            )

        if doc is not None:
            import pathlib
            meta = doc.metadata or {}
            self.store.register_document(
                doc_id=doc.doc_id,
                filename=meta.get("filename", pathlib.Path(doc.source).name),
                filetype=meta.get("filetype", "unknown"),
                source=doc.source,
            )
        total = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch      = chunks[i : i + batch_size]
            texts      = [c.content for c in batch]
            embeddings = self.engine.embed_texts(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
            total += self.store.upsert_chunks(batch)

        logger.info(f"Indexed {total} chunks into Supabase")
        return total