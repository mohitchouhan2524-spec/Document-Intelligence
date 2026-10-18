from fastapi import Header, HTTPException, Request


def get_runtime(request: Request):
    runtime = getattr(request.app.state, "rag_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="RAG runtime is not ready")
    return runtime


async def get_current_user_id(
    # DEVELOPMENT ONLY: replace with verified JWT authentication.
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
) -> str:
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Configure JWT authentication.",
        )
    return x_user_id.strip()
