import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.limits import RateLimiter
from backend.schemas import ChunkOut, QueryRequest, QueryResponse

router = APIRouter()
log = logging.getLogger("hybrid_rag.chat")

query_limiter = RateLimiter(30, 60)


def _chunk_out(chunk) -> ChunkOut:
    source = getattr(chunk.source, "value", chunk.source)
    metadata = getattr(chunk, "metadata", None) or {}
    filename = metadata.get("filename") or metadata.get("source") or chunk.doc_id
    return ChunkOut(
        doc_id=str(chunk.doc_id),
        filename=str(filename),
        content=str(chunk.content),
        score=float(chunk.score or 0.0),
        source=str(source),
    )


@router.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    request: Request,
    user=Depends(query_limiter.by_user()),
):
    try:
        from hybrid.fusion import HybridPipeline

        pipeline = HybridPipeline(
            fusion_method=body.fusion_method,
            generate=body.generate,
            user_id=str(user.id),
            supabase_client=request.app.state.supabase,
        )
        response = pipeline.query(body.query, top_k=body.top_k, return_context=True)
    except Exception as exc:
        log.exception("Query failed for user %s", user.id)
        raise HTTPException(
            status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}"
        )

    metadata = response.metadata or {}
    return QueryResponse(
        answer=response.answer or "",
        route=str(response.route_used.value),
        latency_ms=float(response.latency_ms or 0.0),
        confidence=float(metadata.get("classifier_confidence", 0.0) or 0.0),
        reasoning=str(metadata.get("intent_reasoning", "") or ""),
        chunks=[_chunk_out(c) for c in (response.retrieved_chunks or [])],
    )
