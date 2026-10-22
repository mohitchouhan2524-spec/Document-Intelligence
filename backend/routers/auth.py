import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from supabase import create_client

from backend import config
from backend.deps import bearer, current_user, forget
from backend.limits import RateLimiter
from backend.schemas import (
    LoginRequest,
    MessageOut,
    RefreshRequest,
    SessionOut,
    SignupRequest,
    UserOut,
)

router = APIRouter()
log = logging.getLogger("hybrid_rag.auth")

login_limiter = RateLimiter(10, 60)
signup_limiter = RateLimiter(5, 300)
refresh_limiter = RateLimiter(30, 60)


def _anon_client():
    return create_client(config.SUPABASE_URL, config.SUPABASE_ANON_KEY)


def _reason(exc: Exception) -> str:
    return getattr(exc, "message", None) or str(exc) or exc.__class__.__name__


def _session_out(resp) -> SessionOut:
    session = getattr(resp, "session", None)
    if session is None:
        raise HTTPException(status_code=401, detail="No session was returned")
    user = getattr(resp, "user", None) or session.user
    return SessionOut(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=int(session.expires_in or 3600),
        user=UserOut(id=str(user.id), email=user.email or ""),
    )


@router.post(
    "/signup",
    response_model=MessageOut,
    status_code=201,
    dependencies=[Depends(signup_limiter.by_ip())],
)
def signup(body: SignupRequest):
    try:
        _anon_client().auth.sign_up({"email": body.email, "password": body.password})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Sign-up failed: {_reason(exc)}")
    return MessageOut(
        message="Account created. Check your email to verify your address, then log in."
    )


@router.post(
    "/login",
    response_model=SessionOut,
    dependencies=[Depends(login_limiter.by_ip())],
)
def login(body: LoginRequest):
    try:
        resp = _anon_client().auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=_reason(exc))
    return _session_out(resp)


@router.post(
    "/refresh",
    response_model=SessionOut,
    dependencies=[Depends(refresh_limiter.by_ip())],
)
def refresh(body: RefreshRequest):
    try:
        resp = _anon_client().auth.refresh_session(body.refresh_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=_reason(exc))
    return _session_out(resp)


@router.post("/logout", response_model=MessageOut)
def logout(
    request: Request,
    user=Depends(current_user),
    cred=Depends(bearer),
):
    token = cred.credentials
    forget(token)
    try:
        request.app.state.supabase.auth.admin.sign_out(token)
    except Exception:
        log.warning("Supabase sign-out failed for user %s", user.id, exc_info=True)
    return MessageOut(message="Logged out")


@router.get("/me", response_model=UserOut)
def me(user=Depends(current_user)):
    return UserOut(id=str(user.id), email=user.email or "")
