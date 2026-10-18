from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.documents import router as documents_router
from api.routers.health import router as health_router
from api.routers.query import router as query_router
from api.services.runtime import get_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = get_runtime()
    await runtime.startup()
    app.state.rag_runtime = runtime
    yield
    await runtime.shutdown()


app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    description="Document ingestion and routed RAG query API",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(documents_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
