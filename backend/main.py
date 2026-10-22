import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from backend import config
from backend.routers import admin, auth, chat, documents, evaluation
from backend.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("hybrid_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in your .env file"
        )
    app.state.supabase = create_client(
        config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY
    )
    app.state.engine = None
    app.state.pipeline_ready = False

    try:
        from vector_rag.embed import EmbeddingEngine

        app.state.engine = EmbeddingEngine()
        log.info("Embedding engine loaded")
    except Exception:
        log.exception("Embedding engine failed to load")
    try:
        from src.config import get_settings
        from vector_rag.embed import EmbeddingEngine, get_cross_encoder

        app.state.engine = EmbeddingEngine()
        get_cross_encoder(get_settings().retrieval.reranker_model)
        log.info("Embedding engine and reranker loaded")
    except Exception:
        log.exception("Model loading failed")
    try:
        from hybrid.fusion import HybridPipeline

        HybridPipeline(fusion_method="rrf", generate=False)
        app.state.pipeline_ready = True
        log.info("Hybrid pipeline warmed up")
    except Exception:
        log.exception("Hybrid pipeline failed to warm up")
    yield


app = FastAPI(title="Hybrid-RAG Document Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
app.include_router(admin.router, prefix="/settings", tags=["settings"])


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        engine_loaded=app.state.engine is not None,
        pipeline_ready=app.state.pipeline_ready,
    )
