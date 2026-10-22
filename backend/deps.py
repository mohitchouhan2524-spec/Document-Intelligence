import threading
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend import config

bearer = HTTPBearer(auto_error=False)

_cache: dict = {}
_guard = threading.Lock()


def forget(token: str) -> None:
    with _guard:
        _cache.pop(token, None)


def current_user(
    request: Request,
    cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
):
    if cred is None or not cred.credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = cred.credentials
    now = time.monotonic()
    with _guard:
        hit = _cache.get(token)
        if hit and hit[1] > now:
            return hit[0]
    try:
        user = request.app.state.supabase.auth.get_user(token).user
    except Exception:
        user = None
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    with _guard:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[token] = (user, now + config.TOKEN_CACHE_SECONDS)
    return user


def is_admin(user) -> bool:
    return (getattr(user, "email", "") or "").lower() in config.ADMIN_EMAILS


def require_admin(user=Depends(current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user
