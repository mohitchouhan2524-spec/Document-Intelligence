from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_current_user_id, get_runtime
from api.schemas import IndexStatsOut, QueryRequest, QueryResponse, RouteResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    try:
        result = await runtime.query(
            user_id=user_id,
            query=body.query,
            top_k=body.top_k,
            return_context=body.return_context,
            route_override=body.route_override,
            vectorless_method_override=body.vectorless_method_override,
        )
        return QueryResponse(**result)
    except NotImplementedError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/query/route", response_model=RouteResponse)
async def classify_query(
    body: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    result = await runtime.classify(query=body.query)
    return RouteResponse(**result)


@router.get("/indexes/stats", response_model=IndexStatsOut)
async def index_stats(
    user_id: str = Depends(get_current_user_id),
    runtime=Depends(get_runtime),
):
    return IndexStatsOut(stats=await runtime.index_stats(user_id=user_id))
