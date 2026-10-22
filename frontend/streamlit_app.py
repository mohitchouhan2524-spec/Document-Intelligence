from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend import api_client as api

st.set_page_config(
    page_title="Hybrid-RAG · Document Intelligence",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROUTE_COLORS = {"vector": "#2563EB", "vectorless": "#059669", "hybrid": "#7C3AED"}
ROUTE_ICONS = {"vector": "🔍", "vectorless": "🗄️", "hybrid": "⚡"}
ROUTE_LABELS = {
    "vector": "Semantic search",
    "vectorless": "Structured lookup",
    "hybrid": "Hybrid fusion",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0F1117; border-right: 1px solid #1E2130; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; }

.route-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 12px; border-radius: 9999px;
    font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
}
.badge-vector     { background:#DBEAFE; color:#1D4ED8; }
.badge-vectorless { background:#D1FAE5; color:#065F46; }
.badge-hybrid     { background:#EDE9FE; color:#5B21B6; }

.chat-user {
    background: #1E293B; border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0 4px auto; max-width: 80%;
    color: #E2E8F0; font-size: 15px;
}
.chat-assistant {
    background: #0F1117; border: 1px solid #1E2130;
    border-radius: 4px 12px 12px 12px; padding: 14px 18px;
    margin: 4px 0 8px 0; max-width: 92%;
    color: #E2E8F0; font-size: 15px; line-height: 1.65;
}
.meta-row {
    display: flex; align-items: center; gap: 10px; margin-top: 10px;
    font-size: 12px; color: #64748B; font-family: 'JetBrains Mono', monospace;
}

.chunk-card {
    background: #0F1117; border: 1px solid #1E2130; border-left: 3px solid;
    border-radius: 6px; padding: 10px 14px; margin: 6px 0;
    font-size: 13px; color: #CBD5E1;
    font-family: 'JetBrains Mono', monospace; line-height: 1.6;
}
.chunk-header { font-size: 11px; color: #64748B; margin-bottom: 6px; display: flex; gap: 12px; }

.conf-bar-wrap {
    background: #1E2130; border-radius: 4px; height: 5px; width: 120px;
    overflow: hidden; display: inline-block; vertical-align: middle;
}
.conf-bar-fill { height: 5px; border-radius: 4px; background: linear-gradient(90deg, #3B82F6, #8B5CF6); }

.stat-card {
    background: #0F1117; border: 1px solid #1E2130;
    border-radius: 10px; padding: 16px 20px; text-align: center;
}
.stat-value { font-size: 28px; font-weight: 600; color: #E2E8F0; }
.stat-label { font-size: 12px; color: #64748B; margin-top: 2px; }

.section-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: #64748B; margin-bottom: 6px;
}

.empty-state { text-align: center; padding: 48px 24px; color: #475569; }
.empty-state .icon { font-size: 40px; margin-bottom: 12px; }
.empty-state p { font-size: 15px; margin: 0; }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.dot-ok   { background: #10B981; }
.dot-warn { background: #F59E0B; }
.dot-err  { background: #EF4444; }
</style>
""",
    unsafe_allow_html=True,
)


def _init_state():
    defaults = {
        "user": None,
        "token": None,
        "refresh_token": None,
        "messages": [],
        "ingested_docs": 0,
        "ingested_chunks": 0,
        "ingest_report": None,
        "settings": {
            "fusion_method": "rrf",
            "top_k": 5,
            "generate": True,
            "show_chunks": True,
            "show_reasoning": False,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _fail(err: api.ApiError):
    if err.status == 401:
        st.rerun()
    st.error(err.message)


@st.cache_data(ttl=10, show_spinner=False)
def _api_health() -> dict:
    try:
        return api.health()
    except api.ApiError as err:
        return {"status": "offline", "error": err.message}


def _auth_gate() -> bool:
    if st.session_state.token and st.session_state.user:
        return True

    st.markdown(
        """
        <div style="max-width:420px;margin:0 auto;text-align:center">
            <div style="font-size:48px">🗂️</div>
            <h1 style="font-size:26px;font-weight:600;margin:12px 0 4px">Document Intelligence</h1>
            <p style="color:#64748B;font-size:14px;margin-bottom:32px">Sign in to access your workspace</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        tab_login, tab_signup = st.tabs(["🔑 Log In", "✨ Sign Up"])

        with tab_login:
            email = st.text_input("Email", key="login_email", placeholder="you@example.com")
            password = st.text_input(
                "Password", key="login_pass", type="password", placeholder="••••••••"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Log In", use_container_width=True, type="primary", key="btn_login"):
                if not email or not password:
                    st.warning("Please enter both email and password.")
                else:
                    try:
                        with st.spinner("Signing in…"):
                            api.login(email.strip(), password)
                        st.rerun()
                    except api.ApiError as err:
                        st.error(f"Login failed: {err.message}")

        with tab_signup:
            new_email = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            new_password = st.text_input(
                "Password", key="signup_pass", type="password", placeholder="Min 6 characters"
            )
            new_confirm = st.text_input(
                "Confirm password", key="signup_confirm", type="password", placeholder="••••••••"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "Create Account", use_container_width=True, type="primary", key="btn_signup"
            ):
                if not new_email or not new_password:
                    st.warning("Please fill in all fields.")
                elif new_password != new_confirm:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        with st.spinner("Creating account…"):
                            message = api.signup(new_email.strip(), new_password)
                        st.success(f"✅ {message}")
                    except api.ApiError as err:
                        st.error(f"Sign-up failed: {err.message}")
    return False


def _route_badge(route: str) -> str:
    icon = ROUTE_ICONS.get(route, "•")
    label = ROUTE_LABELS.get(route, route)
    return f'<span class="route-badge badge-{html.escape(route)}">{icon} {html.escape(label)}</span>'


def _conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    width = int(conf * 120)
    return (
        f'<span class="conf-bar-wrap">'
        f'<span class="conf-bar-fill" style="width:{width}px"></span>'
        f"</span> {pct}%"
    )


def _render_chunk_cards(chunks: list):
    for i, chunk in enumerate(chunks, 1):
        retriever = chunk.get("source", "")
        color = ROUTE_COLORS.get(retriever, "#64748B")
        content = chunk.get("content", "")
        preview = html.escape(content[:280]) + ("…" if len(content) > 280 else "")
        st.markdown(
            f"""
        <div class="chunk-card" style="border-left-color:{color}">
            <div class="chunk-header">
                <span>[{i}]</span>
                <span>{html.escape(str(chunk.get("filename", "")))}</span>
                <span>score={float(chunk.get("score", 0)):.4f}</span>
                <span>via {html.escape(str(retriever))}</span>
            </div>
            {preview}
        </div>
        """,
            unsafe_allow_html=True,
        )


def _sidebar():
    with st.sidebar:
        st.markdown("### 🗂️ Hybrid-RAG")
        st.markdown(
            '<p style="color:#475569;font-size:13px;margin-top:-8px">Document Intelligence</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown('<p class="section-eyebrow">Pipeline status</p>', unsafe_allow_html=True)
        status = _api_health()
        if status.get("status") != "ok":
            st.markdown('<span class="status-dot dot-err"></span>API offline', unsafe_allow_html=True)
            with st.expander("Error details"):
                st.code(status.get("error", "Unknown error"), language="text")
        elif not status.get("pipeline_ready"):
            st.markdown(
                '<span class="status-dot dot-warn"></span>API online · pipeline not warmed up',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="status-dot dot-ok"></span>Pipeline ready', unsafe_allow_html=True)
        st.divider()

        st.markdown('<p class="section-eyebrow">Settings</p>', unsafe_allow_html=True)
        current = st.session_state.settings
        fusion = st.selectbox(
            "Fusion method", ["rrf", "linear"], index=["rrf", "linear"].index(current["fusion_method"])
        )
        top_k = st.slider("Retrieved chunks", 1, 10, current["top_k"])
        generate = st.toggle("Generate answer (LLM)", current["generate"])
        show_chunks = st.toggle("Show source chunks", current["show_chunks"])
        show_reasoning = st.toggle("Show classifier reasoning", current["show_reasoning"])
        st.session_state.settings.update(
            {
                "fusion_method": fusion,
                "top_k": top_k,
                "generate": generate,
                "show_chunks": show_chunks,
                "show_reasoning": show_reasoning,
            }
        )
        st.divider()

        st.markdown('<p class="section-eyebrow">Session</p>', unsafe_allow_html=True)
        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"Queries this session: **{turns}**")
        st.markdown(f"Docs ingested: **{st.session_state.ingested_docs}**")
        st.markdown(f"Chunks indexed: **{st.session_state.ingested_chunks}**")
        st.divider()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        user = st.session_state.user or {}
        st.markdown(
            f'<p style="font-size:12px;color:#64748B;margin:8px 0">👤 {html.escape(user.get("email", ""))}</p>',
            unsafe_allow_html=True,
        )
        if st.button("🚪 Log Out", use_container_width=True, key="btn_logout"):
            api.logout()
            st.session_state.ingested_docs = 0
            st.session_state.ingested_chunks = 0
            st.rerun()


def _tab_chat():
    if not st.session_state.messages:
        st.markdown(
            """
        <div class="empty-state">
            <div class="icon">🗂️</div>
            <p>Ask anything about your documents.<br>
            Try: <em>"summarise the key findings"</em> or <em>"what is the total on invoice INV-004"</em></p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
                continue

            meta = msg.get("meta", {})
            route = meta.get("route", "vector")
            latency = meta.get("latency_ms", 0)
            conf = meta.get("confidence", 0)
            reasoning = meta.get("reasoning", "")
            chunks = meta.get("chunks", [])
            answer_html = html.escape(msg["content"]).replace("\n", "<br>")
            footer = (
                f'<div class="meta-row">{_route_badge(route)}'
                f"&nbsp;·&nbsp; {_conf_bar(conf)}"
                f"&nbsp;·&nbsp; {latency:.0f} ms</div>"
                if meta
                else ""
            )
            st.markdown(
                f'<div class="chat-assistant">{answer_html}{footer}</div>',
                unsafe_allow_html=True,
            )
            if st.session_state.settings["show_reasoning"] and reasoning:
                with st.expander("Classifier reasoning"):
                    st.caption(reasoning)
            if st.session_state.settings["show_chunks"] and chunks:
                with st.expander(f"Source chunks ({len(chunks)})"):
                    _render_chunk_cards(chunks)

    st.markdown("<br>", unsafe_allow_html=True)
    query = st.chat_input("Ask a question about your documents…")
    if query and query.strip():
        _handle_query(query.strip())


def _handle_query(query: str):
    settings = st.session_state.settings
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        with st.spinner("Retrieving…"):
            result = api.ask(
                query,
                top_k=settings["top_k"],
                fusion_method=settings["fusion_method"],
                generate=settings["generate"],
            )
        message = {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "route": result["route"],
                "latency_ms": result["latency_ms"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "chunks": result["chunks"],
            },
        }
    except api.ApiError as err:
        if err.status == 401:
            st.rerun()
        message = {"role": "assistant", "content": f"⚠️ {err.message}", "meta": {}}
    st.session_state.messages.append(message)
    st.rerun()


def _load_library():
    try:
        return api.list_documents(), None
    except api.ApiError as err:
        if err.status == 401:
            st.rerun()
        return None, err.message


def _show_ingest_report(report: dict):
    indexed = report["documents_indexed"]
    if indexed:
        st.success(
            f"✓ Ingested {indexed} document(s) · {report['chunks_indexed']} chunks · "
            "stored under your account"
        )
    else:
        st.error("No documents were indexed.")
    for item in report["files"]:
        if item["status"] == "failed":
            st.warning(f"{item['filename']}: {item['message']}")
    with st.expander("Ingest log"):
        st.code("\n".join(report["log"]), language="text")


def _tab_ingest():
    library, library_error = _load_library()
    max_pdfs = library["max_pdfs"] if library else 3

    st.markdown("### Upload and index documents")
    st.caption(
        "Supported: PDF, DOCX, TXT, HTML, CSV, XLSX  ·  "
        f"Each user stores at most **{max_pdfs} PDFs** — oldest is auto-removed when limit is reached."
    )

    report = st.session_state.ingest_report
    if report:
        _show_ingest_report(report)
        st.session_state.ingest_report = None

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "txt", "html", "htm", "csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    pdf_count = sum(1 for f in (uploaded or []) if f.name.lower().endswith(".pdf"))
    too_many = pdf_count > max_pdfs
    if too_many:
        st.warning(
            f"You selected {pdf_count} PDFs but the limit is {max_pdfs}. "
            f"Remove {pdf_count - max_pdfs} to continue."
        )

    col_a, col_b = st.columns(2)
    skip_bm25 = col_a.checkbox("Skip BM25 index", value=False)
    skip_sql = col_b.checkbox("Skip SQL index", value=False)

    if st.button("⬆️ Ingest documents", disabled=not uploaded or too_many, type="primary"):
        try:
            with st.spinner(f"Indexing {len(uploaded)} file(s)… this can take a minute."):
                result = api.ingest_documents(uploaded, skip_bm25, skip_sql)
            st.session_state.ingested_docs += result["documents_indexed"]
            st.session_state.ingested_chunks += result["chunks_indexed"]
            st.session_state.ingest_report = result
            st.rerun()
        except api.ApiError as err:
            _fail(err)

    st.divider()
    st.markdown("### Your document library")
    if library_error:
        st.warning(library_error)
    elif library:
        _render_user_docs(library)

    st.divider()
    st.markdown("### Index status")
    _render_index_stats(library)


def _render_user_docs(library: dict):
    docs = library["documents"]
    if not docs:
        st.info("No documents uploaded yet. Upload PDFs above to get started.")
        return

    st.caption(
        f"{library['doc_count']} / {library['max_pdfs']} documents stored  ·  "
        f"{library['chunk_count']} total chunks  ·  "
        "Oldest is auto-removed when limit is reached"
    )
    for doc in docs:
        col_name, col_date, col_del = st.columns([5, 3, 1])
        icon = "📄" if doc["filetype"] == "PDF" else "📝"
        col_name.markdown(f"{icon} **{doc['filename']}**  `{doc['filetype']}`")
        col_date.caption(f"Uploaded {doc['created_at'][:10]}")
        if col_del.button("🗑", key=f"del_{doc['doc_id']}", help=f"Delete {doc['filename']}"):
            try:
                api.delete_document(doc["doc_id"])
                st.rerun()
            except api.ApiError as err:
                _fail(err)


def _render_index_stats(library: dict | None):
    cols = st.columns(4)
    if not library:
        cols[0].info("Stats unavailable")
        return
    cols[0].markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{library['doc_count']}</div>
            <div class="stat-label">Your Documents</div>
        </div>""",
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f"""
        <div class="stat-card">
            <div class="stat-value">{library['chunk_count']}</div>
            <div class="stat-label">Your Chunks</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _tab_evaluation():
    st.markdown("### Evaluation")
    st.caption(
        "Upload a CSV with columns `query`, `reference` (and optionally `expected_route`) "
        "to run RAGAS + ROUGE-L metrics."
    )

    eval_file = st.file_uploader(
        "Evaluation dataset CSV", type=["csv"], label_visibility="collapsed"
    )
    col1, col2, col3 = st.columns(3)
    sample = col1.number_input("Max samples", 5, 500, 50, step=5)
    no_llm = col2.checkbox("ROUGE-L only (no API calls)", value=False)
    save_json = col3.checkbox("Save JSON report", value=True)

    if st.button("▶ Run evaluation", disabled=not eval_file, type="primary"):
        _run_evaluation(eval_file, int(sample), no_llm, save_json)

    _render_previous_reports()


def _run_evaluation(eval_file, sample: int, no_llm: bool, save_json: bool):
    settings = st.session_state.settings
    try:
        with st.spinner("Running evaluation… this can take several minutes."):
            result = api.run_evaluation(
                eval_file,
                sample,
                no_llm,
                save_json,
                settings["fusion_method"],
                settings["generate"],
            )
    except api.ApiError as err:
        _fail(err)
        return

    st.markdown("### Results")
    summary = result["metrics_summary"]
    metric_cols = st.columns(len(summary) or 1)
    for i, (name, score) in enumerate(sorted(summary.items())):
        metric_cols[i % len(metric_cols)].metric(
            name.replace("_", " ").title(), "–" if score is None else f"{score:.3f}"
        )

    if result["routing_accuracy"] is not None:
        st.metric("Routing accuracy", f"{result['routing_accuracy']:.3f}")

    if result["per_route"]:
        st.markdown("#### Per-route breakdown")
        frame = pd.DataFrame(result["per_route"]).T
        st.dataframe(frame.style.format("{:.3f}", na_rep="–"), use_container_width=True)

    st.markdown("#### Latency (ms)")
    latency = result["latency"]
    for col, key in zip(st.columns(4), ["mean", "p50", "p95", "p99"]):
        value = latency.get(key)
        col.metric(key.upper(), "–" if value is None else f"{value:.1f}")

    if result["report_name"]:
        try:
            data = api.download_report(result["report_name"])
            st.download_button(
                "⬇️ Download report JSON",
                data=data,
                file_name=result["report_name"],
                mime="application/json",
            )
        except api.ApiError as err:
            st.warning(f"Report saved but could not be downloaded: {err.message}")


def _render_previous_reports():
    try:
        reports = api.list_reports()
    except api.ApiError as err:
        if err.status == 401:
            st.rerun()
        return
    if not reports:
        return

    st.divider()
    st.markdown("### Previous reports")
    for report in reports:
        name = report["name"]
        scores = "  ".join(
            f"{k}={v:.3f}" for k, v in sorted(report["metrics_summary"].items()) if v is not None
        )
        col_label, col_action = st.columns([8, 1])
        col_label.markdown(f"**{name.removesuffix('.json')[:22]}  —  {scores}**")
        cache_key = f"report_bytes_{name}"
        if cache_key in st.session_state:
            col_action.download_button(
                "⬇",
                data=st.session_state[cache_key],
                file_name=name,
                mime="application/json",
                key=f"dl_{name}",
            )
        elif col_action.button("Get", key=f"get_{name}"):
            try:
                st.session_state[cache_key] = api.download_report(name)
                st.rerun()
            except api.ApiError as err:
                _fail(err)


def _tab_settings():
    st.markdown("### Settings")
    try:
        cfg = api.get_config()
    except api.ApiError as err:
        _fail(err)
        return

    if cfg.get("error"):
        st.warning(cfg["error"])

    sections = cfg["sections"]
    for title, key, expanded in [
        ("LLM configuration", "llm", True),
        ("Retrieval configuration", "retrieval", False),
        ("Classifier configuration", "classifier", False),
    ]:
        with st.expander(title, expanded=expanded):
            st.json(sections.get(key, {}))

    st.divider()
    st.markdown("### Classifier training")
    if not cfg["is_admin"]:
        st.info("Classifier training and environment details are restricted to administrators.")
        return

    st.caption("Retrain the route classifier with your current training data.")
    col1, col2 = st.columns(2)
    if col1.button("🔄 Train classifier", type="primary"):
        try:
            with st.spinner("Training…"):
                result = api.train_classifier()
            st.success(f"Classifier trained — model saved to {result['model_path']}")
        except api.ApiError as err:
            _fail(err)

    if col2.button("📋 Evaluate classifier"):
        try:
            with st.spinner("Evaluating…"):
                result = api.evaluate_classifier()
            st.metric("Macro F1", f"{result['macro_f1']:.4f}")
            st.code(result["report"], language="text")
        except api.ApiError as err:
            _fail(err)

    st.divider()
    st.markdown("### Environment")
    for key, present in cfg["env"].items():
        icon = "✅" if present else "❌"
        st.markdown(f"{icon}  `{key}` — {'set' if present else 'not set'}")


def main():
    if not _auth_gate():
        st.stop()

    _sidebar()

    st.markdown(
        "## 🗂️ Document Intelligence",
        help="Hybrid-RAG: routes each query to semantic search, structured lookup, or both.",
    )
    tab_chat, tab_ingest, tab_eval, tab_settings = st.tabs(
        ["💬 Chat", "📂 Ingest", "📊 Evaluation", "⚙️ Settings"]
    )

    with tab_chat:
        _tab_chat()
    with tab_ingest:
        _tab_ingest()
    with tab_eval:
        _tab_evaluation()
    with tab_settings:
        _tab_settings()


main()
