"""
src
───
Shared foundations for the entire Hybrid-RAG project.

Exports
───────
    Models  — Document, Chunk, QueryIntent, RetrievedChunk, RAGResponse
    Enums   — RouteType, VectorlessMethod
    Config  — get_settings, Settings
"""
from src.models import (
    RouteType,
    VectorlessMethod,
    Document,
    Chunk,
    QueryIntent,
    RetrievedChunk,
    RAGResponse,
)
from src.config import get_settings, Settings

__all__ = [
    # enums
    "RouteType",
    "VectorlessMethod",
    # data models
    "Document",
    "Chunk",
    "QueryIntent",
    "RetrievedChunk",
    "RAGResponse",
    # config
    "get_settings",
    "Settings",
]