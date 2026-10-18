from __future__ import annotations

from typing import Any


class RAGRuntime:
    def __init__(self):
        self.ready = False
        # TODO: construct shared Supabase client, classifier artifacts, scraper,
        # chunker, vector/vectorless indexers, and existing HybridPipeline here.

    async def startup(self) -> None:
        # TODO: validate settings, load model artifacts, check required services.
        self.ready = True

    async def shutdown(self) -> None:
        self.ready = False
        # TODO: close clients when supported.

    async def ingest_document(self, *, user_id: str, filename: str, content: bytes) -> dict[str, Any]:
        # Write bytes to a securely named temporary file; call DocumentScraper.load_file,
        # then DocumentChunker; index to each enabled backend with user_id scoping.
        # Persist document status and clean up the temporary file in a finally block.
        raise NotImplementedError("Connect DocumentScraper, DocumentChunker, and your indexers.")

    async def list_documents(self, *, user_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError("Implement user-scoped document listing.")

    async def get_document(self, *, user_id: str, doc_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("Implement user-scoped document lookup.")

    async def delete_document(self, *, user_id: str, doc_id: str) -> bool:
        # Remove document and chunks from vector, BM25, SQL, graph, and registry stores.
        raise NotImplementedError("Implement complete deletion across all enabled indexes.")

    async def classify(self, *, query: str) -> dict[str, Any]:
        # Call classifier.train.predict_intent(query, artifacts=loaded_artifacts).
        raise NotImplementedError("Connect classifier.train.predict_intent.")

    async def query(
        self, *, user_id: str, query: str, top_k: int, return_context: bool,
        route_override: str | None = None,
        vectorless_method_override: str | None = None,
    ) -> dict[str, Any]:
        # Call the existing fusion pipeline; normalize its result to QueryResponse.
        # Overrides must never bypass authorization/user filtering.
        raise NotImplementedError("Connect your existing hybrid/fusion pipeline and LLM generator.")

    async def index_stats(self, *, user_id: str) -> dict[str, Any]:
        raise NotImplementedError("Return user-scoped index counts.")


_runtime: RAGRuntime | None = None


def get_runtime() -> RAGRuntime:
    global _runtime
    if _runtime is None:
        _runtime = RAGRuntime()
    return _runtime
