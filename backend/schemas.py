from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class UserOut(BaseModel):
    id: str
    email: str


class SessionOut(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    user: UserOut


class MessageOut(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    engine_loaded: bool
    pipeline_ready: bool


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    fusion_method: Literal["rrf", "linear"] = "rrf"
    generate: bool = True


class ChunkOut(BaseModel):
    doc_id: str
    filename: str
    content: str
    score: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    latency_ms: float
    confidence: float
    reasoning: str
    chunks: list[ChunkOut]


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    filetype: str
    created_at: str


class DocumentLibrary(BaseModel):
    documents: list[DocumentOut]
    doc_count: int
    chunk_count: int
    max_pdfs: int


class FileResult(BaseModel):
    filename: str
    status: Literal["indexed", "failed"]
    documents: int = 0
    chunks: int = 0
    message: str = ""


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    files: list[FileResult]
    log: list[str]


class EvaluationResponse(BaseModel):
    metrics_summary: dict[str, Optional[float]]
    routing_accuracy: Optional[float]
    per_route: dict[str, dict[str, Optional[float]]]
    latency: dict[str, Optional[float]]
    report_name: Optional[str]


class ReportInfo(BaseModel):
    name: str
    metrics_summary: dict[str, Optional[float]]


class ConfigResponse(BaseModel):
    sections: dict[str, dict[str, Any]]
    is_admin: bool
    env: dict[str, bool]
    error: Optional[str] = None


class TrainResponse(BaseModel):
    message: str
    model_path: str


class ClassifierEvalResponse(BaseModel):
    macro_f1: float
    report: str
