from typing import Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

RouteName = Literal["vector", "vectorless", "hybrid"]
MethodName = Literal["sql", "bm25", "graph"]


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    filetype: str
    status: Literal["indexed", "failed", "processing"]
    chunk_count: int = 0
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)
    return_context: bool = True
    route_override: RouteName | None = None
    vectorless_method_override: MethodName | None = None


class SourceOut(BaseModel):
    chunk_id: str | None = None
    doc_id: str | None = None
    content: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    route_used: str
    vectorless_method: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    retrieved_chunks: list[SourceOut] = Field(default_factory=list)
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    route: str
    vectorless_method: str | None = None
    confidence: float
    reasoning: str


class IndexStatsOut(BaseModel):
    stats: dict[str, Any]
