from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
class RouteType(str, Enum):
    VECTOR = "vector"           # semantic / fuzzy
    VECTORLESS = "vectorless"   # exact / structured
    HYBRID = "hybrid"           # both paths fused


class VectorlessMethod(str, Enum):
    SQL = "sql"
    BM25 = "bm25"
    GRAPH = "graph"


class Document(BaseModel):
    doc_id: str
    source: str                   # file path or URL
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class QueryIntent(BaseModel):
    route: RouteType
    vectorless_method: VectorlessMethod | None = None
    confidence: float
    reasoning: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    source: RouteType
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    query: str
    route_used: RouteType
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)