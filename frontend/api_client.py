import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

TIMEOUT_DEFAULT = 30
TIMEOUT_QUERY = 120
TIMEOUT_INGEST = 600
TIMEOUT_HEAVY = 1800


class ApiError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.message = message
        self.status = status


def _detail(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300] or resp.reason or "Request failed"
    detail = data.get("detail") if isinstance(data, dict) else data
    if isinstance(detail, list):
        return "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )
    return str(detail)


def _send(method: str, path: str, auth: bool, timeout: int, **kwargs) -> requests.Response:
    headers = {}
    if auth:
        headers["Authorization"] = f"Bearer {st.session_state.get('token') or ''}"
    try:
        return requests.request(
            method, f"{API_URL}{path}", headers=headers, timeout=timeout, **kwargs
        )
    except requests.ConnectionError:
        raise ApiError("Cannot reach the API server. Make sure it is running.")
    except requests.Timeout:
        raise ApiError("The request timed out. Please try again.")
    except requests.RequestException as exc:
        raise ApiError(str(exc))


def _store_session(data: dict) -> None:
    st.session_state.token = data["access_token"]
    st.session_state.refresh_token = data["refresh_token"]
    st.session_state.user = data["user"]


def clear_session() -> None:
    st.session_state.token = None
    st.session_state.refresh_token = None
    st.session_state.user = None
    st.session_state.messages = []


def _refresh() -> bool:
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False
    try:
        resp = _send(
            "POST",
            "/auth/refresh",
            auth=False,
            timeout=TIMEOUT_DEFAULT,
            json={"refresh_token": refresh_token},
        )
    except ApiError:
        return False
    if resp.status_code != 200:
        return False
    _store_session(resp.json())
    return True


def _request(
    method: str,
    path: str,
    auth: bool = True,
    timeout: int = TIMEOUT_DEFAULT,
    **kwargs,
) -> requests.Response:
    resp = _send(method, path, auth, timeout, **kwargs)
    if resp.status_code == 401 and auth and _refresh():
        resp = _send(method, path, auth, timeout, **kwargs)
    if resp.status_code == 401 and auth:
        clear_session()
        raise ApiError("Your session has expired. Please log in again.", 401)
    if resp.status_code >= 400:
        raise ApiError(_detail(resp), resp.status_code)
    return resp


def health() -> dict:
    return _request("GET", "/health", auth=False, timeout=5).json()


def signup(email: str, password: str) -> str:
    resp = _request(
        "POST", "/auth/signup", auth=False, json={"email": email, "password": password}
    )
    return resp.json()["message"]


def login(email: str, password: str) -> None:
    resp = _request(
        "POST", "/auth/login", auth=False, json={"email": email, "password": password}
    )
    _store_session(resp.json())


def logout() -> None:
    try:
        _send("POST", "/auth/logout", auth=True, timeout=10)
    except ApiError:
        pass
    clear_session()


def ask(query: str, top_k: int, fusion_method: str, generate: bool) -> dict:
    return _request(
        "POST",
        "/chat/query",
        timeout=TIMEOUT_QUERY,
        json={
            "query": query,
            "top_k": top_k,
            "fusion_method": fusion_method,
            "generate": generate,
        },
    ).json()


def list_documents() -> dict:
    return _request("GET", "/documents").json()


def delete_document(doc_id: str) -> str:
    return _request("DELETE", f"/documents/{doc_id}").json()["message"]


def ingest_documents(files: list, skip_bm25: bool, skip_sql: bool) -> dict:
    payload = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    return _request(
        "POST",
        "/documents/ingest",
        timeout=TIMEOUT_INGEST,
        files=payload,
        data={"skip_bm25": str(skip_bm25).lower(), "skip_sql": str(skip_sql).lower()},
    ).json()


def run_evaluation(
    file,
    sample: int,
    no_llm: bool,
    save_json: bool,
    fusion_method: str,
    generate: bool,
) -> dict:
    return _request(
        "POST",
        "/evaluation/run",
        timeout=TIMEOUT_HEAVY,
        files={"file": (file.name, file.getvalue(), "text/csv")},
        data={
            "sample": str(sample),
            "no_llm": str(no_llm).lower(),
            "save_json": str(save_json).lower(),
            "fusion_method": fusion_method,
            "generate": str(generate).lower(),
        },
    ).json()


def list_reports() -> list:
    return _request("GET", "/evaluation/reports").json()


def download_report(name: str) -> bytes:
    return _request("GET", f"/evaluation/reports/{name}").content


def get_config() -> dict:
    return _request("GET", "/settings/config").json()


def train_classifier() -> dict:
    return _request("POST", "/settings/classifier/train", timeout=TIMEOUT_HEAVY).json()


def evaluate_classifier() -> dict:
    return _request("GET", "/settings/classifier/evaluate", timeout=TIMEOUT_HEAVY).json()
