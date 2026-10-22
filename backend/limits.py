import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

from fastapi import Depends, HTTPException, Request

from backend import config
from backend.deps import current_user

ingest_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_INGEST)
heavy_slots = threading.BoundedSemaphore(config.MAX_CONCURRENT_HEAVY)

_locks: dict = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


@contextmanager
def exclusive(key: str, message: str):
    lock = _lock_for(key)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=message)
    try:
        yield
    finally:
        lock.release()


@contextmanager
def acquire_slot(semaphore: threading.BoundedSemaphore, message: str):
    if not semaphore.acquire(timeout=config.QUEUE_TIMEOUT_SECONDS):
        raise HTTPException(
            status_code=503, detail=message, headers={"Retry-After": "15"}
        )
    try:
        yield
    finally:
        semaphore.release()


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits = defaultdict(deque)
        self._guard = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._guard:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Try again in {retry}s.",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)

    def by_user(self):
        def dependency(user=Depends(current_user)):
            self.check(str(user.id))
            return user

        return dependency

    def by_ip(self):
        def dependency(request: Request):
            host = request.client.host if request.client else "unknown"
            self.check(host)

        return dependency
