from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    local_model: str = "BAAI/bge-large-en-v1.5"
    openai_model: str = "text-embedding-3-large"
    google_model:str="models/text-embedding-004" # add to prevent crash
    dimension: int = 1024
    batch_size: int = 64


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "doc_intelligence"
    distance: str = "Cosine"
    top_k: int = 10


class ElasticsearchConfig(BaseModel):
    host: str = "localhost"
    port: int = 9200
    index_name: str = "doc_intelligence_bm25"
    top_k: int = 10


class ClassifierConfig(BaseModel):
    mode: str = "rules_first"
    model_path: str = "data/indexes/classifier_model.pkl"
    confidence_threshold: float = 0.75
    fallback: str = "hybrid"


class RetrievalConfig(BaseModel):
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_top_k: int = 5
    fusion_method: str = "rrf"
    rrf_k: int = 60


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 2048
    temperature: float = 0.1
    system_prompt: str = ""


class Settings(BaseSettings):
    # API keys (from .env)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")

    # Nested configs (loaded from yaml)
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    yaml_path = ROOT / "configs" / "config.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    settings = Settings(
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        qdrant=QdrantConfig(**raw.get("qdrant", {})),
        elasticsearch=ElasticsearchConfig(**raw.get("elasticsearch", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**raw.get("llm", {})),
    )
    return settings