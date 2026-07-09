# tests/test_retrieve.py

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.models import RetrievedChunk, RouteType
from vector_rag.retrieve import VectorRetriever


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Mock application configuration."""
    cfg = SimpleNamespace(
    supabase=SimpleNamespace(
        top_k=5,
    ),
    retrieval=SimpleNamespace(
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_k=2,
    ),
    )
    return cfg


@pytest.fixture
def retriever(mock_settings):
    """
    Create VectorRetriever with mocked dependencies.
    """
    with patch("vector_rag.retrieve.get_settings", return_value=mock_settings), \
         patch("vector_rag.retrieve.SupabaseClient"), \
         patch("vector_rag.retrieve.EmbeddingEngine"):

        r = VectorRetriever()
        r.client = MagicMock()
        r.embedder = MagicMock()

        return r


# -------------------------------------------------------------------
# retrieve()
# -------------------------------------------------------------------

def test_retrieve_returns_chunks(retriever):
    retriever.embedder.embed_query.return_value = [0.1, 0.2]

    fake_result = MagicMock()
    fake_result.id = "1"
    fake_result.score = 0.92
    fake_result.payload = {
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "content": "This is a chunk",
        "page": 4,
    }

    retriever.client.search.return_value = [fake_result]

    results = retriever.retrieve("What is AI?")

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "chunk-1"
    assert results[0].doc_id == "doc-1"
    assert results[0].content == "This is a chunk"
    assert results[0].score == 0.92
    assert results[0].source == RouteType.VECTOR
    assert results[0].metadata["page"] == 4


def test_retrieve_empty_results(retriever):
    retriever.embedder.embed_query.return_value = [0.3, 0.4]
    retriever.client.search.return_value = []

    results = retriever.retrieve("unknown query")

    assert results == []


def test_retrieve_uses_top_k(retriever):
    retriever.embedder.embed_query.return_value = [0.1]

    retriever.client.search.return_value = []

    retriever.retrieve("test", top_k=7)

    retriever.client.search.assert_called_once()

    kwargs = retriever.client.search.call_args.kwargs
    assert kwargs["limit"] == 7


def test_retrieve_with_metadata_filter(retriever):
    retriever.embedder.embed_query.return_value = [0.5]
    retriever.client.search.return_value = []

    retriever.retrieve(
        "policy",
        metadata_filter={"department": "HR"}
    )

    kwargs = retriever.client.search.call_args.kwargs

    assert kwargs["query_filter"] is not None


# -------------------------------------------------------------------
# rerank()
# -------------------------------------------------------------------

def test_rerank_orders_by_score(retriever):
    chunks = [
        RetrievedChunk(
            chunk_id="1",
            doc_id="a",
            content="A",
            score=0.1,
            source=RouteType.VECTOR,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="2",
            doc_id="b",
            content="B",
            score=0.2,
            source=RouteType.VECTOR,
            metadata={},
        ),
    ]

    reranker = MagicMock()
    reranker.predict.return_value = [0.3, 0.9]

    retriever._reranker = reranker

    ranked = retriever._rerank(
        "query",
        chunks,
        top_k=2,
    )

    assert ranked[0].chunk_id == "2"
    assert ranked[1].chunk_id == "1"

    assert ranked[0].score == pytest.approx(0.9)


def test_rerank_empty_input(retriever):
    result = retriever._rerank(
        "query",
        [],
        top_k=5,
    )

    assert result == []


# -------------------------------------------------------------------
# retrieve_and_rerank()
# -------------------------------------------------------------------

def test_retrieve_and_rerank_calls_both(retriever):
    fake_chunks = [
        RetrievedChunk(
            chunk_id="1",
            doc_id="d",
            content="hello",
            score=0.3,
            source=RouteType.VECTOR,
            metadata={},
        )
    ]

    retriever.retrieve = MagicMock(return_value=fake_chunks)
    retriever._rerank = MagicMock(return_value=fake_chunks)

    results = retriever.retrieve_and_rerank(
        "hello",
        top_k=1,
    )

    retriever.retrieve.assert_called_once()
    retriever._rerank.assert_called_once()

    assert results == fake_chunks


# -------------------------------------------------------------------
# lazy loading
# -------------------------------------------------------------------

def test_get_reranker_lazy_loading(retriever):
    fake_encoder = MagicMock()

    with patch(
        "sentence_transformers.CrossEncoder",
        return_value=fake_encoder,
    ):

        r1 = retriever._get_reranker()
        r2 = retriever._get_reranker()

        assert r1 is fake_encoder
        assert r2 is fake_encoder