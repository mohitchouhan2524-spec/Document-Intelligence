from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    runtime = getattr(request.app.state, "rag_runtime", None)
    if runtime is None or not runtime.ready:
        return {"status": "not_ready"}
    return {"status": "ready"}
