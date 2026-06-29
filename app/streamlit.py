"""
app/streamlit_app.py
────────────────────────────────────────────────────────────────────────────────
Streamlit front-end for Hybrid-RAG Document Intelligence.

Run with:
    streamlit run app/streamlit_app.py

Tabs
────
    💬 Chat          — ask questions, see route badge + source chunks
    📂 Ingest        — upload and index new documents
    📊 Evaluation    — run RAGAS + ROUGE-L on a labelled CSV
    ⚙️  Settings      — LLM provider, fusion method, top-k, etc.
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from typing import Any
import streamlit as st

# ── Project root on path 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) 
st.set_page_config(
    page_title="Hybrid-RAG · Document Intelligence",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens 
ROUTE_COLORS = {
    "vector":     "#2563EB",   # blue
    "vectorless": "#059669",   # green
    "hybrid":     "#7C3AED",   # purple
}
ROUTE_ICONS = {
    "vector":     "🔍",
    "vectorless": "🗄️",
    "hybrid":     "⚡",
}
ROUTE_LABELS = {
    "vector":     "Semantic search",
    "vectorless": "Structured lookup",
    "hybrid":     "Hybrid fusion",
}

# ── CSS 
st.markdown("""
<style>
/* ── Global ----*/
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar */
[data-testid="stSidebar"] {
    background: #0F1117;
    border-right: 1px solid #1E2130;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; }

/* ── Route badge  */
.route-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
}
.badge-vector     { background:#DBEAFE; color:#1D4ED8; }
.badge-vectorless { background:#D1FAE5; color:#065F46; }
.badge-hybrid     { background:#EDE9FE; color:#5B21B6; }

/* ── Chat messages  */
.chat-user {
    background: #1E293B;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 4px auto;
    max-width: 80%;
    color: #E2E8F0;
    font-size: 15px;
}
.chat-assistant {
    background: #0F1117;
    border: 1px solid #1E2130;
    border-radius: 4px 12px 12px 12px;
    padding: 14px 18px;
    margin: 4px 0 8px 0;
    max-width: 92%;
    color: #E2E8F0;
    font-size: 15px;
    line-height: 1.65;
}
.meta-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    font-size: 12px;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Source chunk card  */
.chunk-card {
    background: #0F1117;
    border: 1px solid #1E2130;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: #CBD5E1;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
}
.chunk-header {
    font-size: 11px;
    color: #64748B;
    margin-bottom: 6px;
    display: flex;
    gap: 12px;
}

/* ── Confidence bar  */
.conf-bar-wrap {
    background: #1E2130;
    border-radius: 4px;
    height: 5px;
    width: 120px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
}
.conf-bar-fill {
    height: 5px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
}

/* ── Stat card  */
.stat-card {
    background: #0F1117;
    border: 1px solid #1E2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value { font-size: 28px; font-weight: 600; color: #E2E8F0; }
.stat-label { font-size: 12px; color: #64748B; margin-top: 2px; }

/* ── Section headers  */
.section-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 6px;
}

/* ── Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #475569;
}
.empty-state .icon { font-size: 40px; margin-bottom: 12px; }
.empty-state p { font-size: 15px; margin: 0; }

/* ── Pipeline status dot */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-ok   { background: #10B981; }
.dot-warn { background: #F59E0B; }
.dot-err  { background: #EF4444; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation 
def _init_state():
    defaults = {
        "messages":       [],       # list[dict] — {role, content, meta}
        "pipeline":       None,     # HybridPipeline singleton
        "pipeline_error": None,     # str | None
        "ingested_docs":  0,
        "ingested_chunks": 0,
        "settings": {
            "fusion_method":    "rrf",
            "top_k":            5,
            "return_context":   True,
            "generate":         True,
            "show_chunks":      True,
            "show_reasoning":   False,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Pipeline loader 

@st.cache_resource(show_spinner="Loading pipeline…")
def _load_pipeline(fusion_method: str, generate: bool):
    """
    Cached singleton — only rebuilds when fusion_method or generate changes.
    Catching all exceptions so the app stays alive even if infra is down.
    """
    try:
        from hybrid.fusion import HybridPipeline
        return HybridPipeline(fusion_method=fusion_method, generate=generate), None
    except Exception as e:
        return None, str(e)


def _get_pipeline():
    s = st.session_state.settings
    pipeline, err = _load_pipeline(s["fusion_method"], s["generate"])
    st.session_state.pipeline       = pipeline
    st.session_state.pipeline_error = err
    return pipeline, err


# ── Helpers 

def _route_badge(route: str) -> str:
    icon  = ROUTE_ICONS.get(route, "•")
    label = ROUTE_LABELS.get(route, route)
    css   = f"badge-{route}"
    return f'<span class="route-badge {css}">{icon} {label}</span>'


def _conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    w   = int(conf * 120)
    return (
        f'<span class="conf-bar-wrap">'
        f'<span class="conf-bar-fill" style="width:{w}px"></span>'
        f'</span> {pct}%'
    )
def _chunk_color(route: str) -> str:
    return ROUTE_COLORS.get(route, "#64748B")


def _render_chunk_cards(chunks: list):
    for i, chunk in enumerate(chunks, 1):
        color    = _chunk_color(chunk.source.value if hasattr(chunk.source, "value") else str(chunk.source))
        source   = chunk.metadata.get("filename") or chunk.metadata.get("source") or chunk.doc_id
        retriever = chunk.source.value if hasattr(chunk.source, "value") else str(chunk.source)
        preview  = chunk.content[:280].replace("<", "&lt;").replace(">", "&gt;")
        if len(chunk.content) > 280:
            preview += "…"
        st.markdown(f"""
        <div class="chunk-card" style="border-left-color:{color}">
            <div class="chunk-header">
                <span>[{i}]</span>
                <span>{source}</span>
                <span>score={chunk.score:.4f}</span>
                <span>via {retriever}</span>
            </div>
            {preview}
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar

def _sidebar():
    with st.sidebar:
        st.markdown("### 🗂️ Hybrid-RAG")
        st.markdown('<p style="color:#475569;font-size:13px;margin-top:-8px">Document Intelligence</p>',
                    unsafe_allow_html=True)
        st.divider()

        # ── Pipeline status
        st.markdown('<p class="section-eyebrow">Pipeline status</p>', unsafe_allow_html=True)
        pipeline, err = _get_pipeline()
        if pipeline:
            st.markdown('<span class="status-dot dot-ok"></span>Pipeline ready',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-dot dot-err"></span>Pipeline offline',
                        unsafe_allow_html=True)
            with st.expander("Error details"):
                st.code(err or "Unknown error", language="text")

        st.divider()

        # ── Quick settings
        st.markdown('<p class="section-eyebrow">Settings</p>', unsafe_allow_html=True)

        fusion = st.selectbox(
            "Fusion method",
            ["rrf", "linear"],
            index=["rrf","linear"].index(st.session_state.settings["fusion_method"]),
        )
        top_k = st.slider("Retrieved chunks", 1, 10,
                          st.session_state.settings["top_k"])
        generate = st.toggle("Generate answer (LLM)",
                             st.session_state.settings["generate"])
        show_chunks = st.toggle("Show source chunks",
                                st.session_state.settings["show_chunks"])
        show_reasoning = st.toggle("Show classifier reasoning",
                                   st.session_state.settings["show_reasoning"])

        # Persist
        st.session_state.settings.update({
            "fusion_method":   fusion,
            "top_k":           top_k,
            "generate":        generate,
            "show_chunks":     show_chunks,
            "show_reasoning":  show_reasoning,
        })

        st.divider()

        # ── Stats 
        st.markdown('<p class="section-eyebrow">Session</p>', unsafe_allow_html=True)
        n_turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"Queries this session: **{n_turns}**")
        st.markdown(f"Docs ingested: **{st.session_state.ingested_docs}**")
        st.markdown(f"Chunks indexed: **{st.session_state.ingested_chunks}**")

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Tab: Chat ─

def _tab_chat():
    # ── Render history 
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🗂️</div>
            <p>Ask anything about your documents.<br>
            Try: <em>"summarise the key findings"</em> or <em>"what is the total on invoice INV-001"</em></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                meta     = msg.get("meta", {})
                route    = meta.get("route", "vector")
                latency  = meta.get("latency_ms", 0)
                conf     = meta.get("confidence", 0)
                reasoning = meta.get("reasoning", "")
                chunks   = meta.get("chunks", [])

                answer_html = msg["content"].replace("\n", "<br>")
                badge   = _route_badge(route)
                conf_bar = _conf_bar(conf)

                st.markdown(f"""
                <div class="chat-assistant">
                    {answer_html}
                    <div class="meta-row">
                        {badge}
                        &nbsp;·&nbsp; {conf_bar}
                        &nbsp;·&nbsp; {latency:.0f} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.session_state.settings["show_reasoning"] and reasoning:
                    with st.expander("Classifier reasoning"):
                        st.caption(reasoning)

                if st.session_state.settings["show_chunks"] and chunks:
                    with st.expander(f"Source chunks ({len(chunks)})"):
                        _render_chunk_cards(chunks)

    # ── Input 
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if st.session_state.question_count >= 10:
        st.warning("You have reached the 10-question limit for this session. Please clear the chat to start over!")
    # Passing disabled=True locks the input box
        query = st.chat_input("Limit reached", disabled=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        col_input, col_btn = st.columns([10, 1])
        with col_input:
            query = st.chat_input("Ask a question about your documents…")
            if query:
                st.session_state.question_count+=1
                _handle_query(query.strip())


def _handle_query(query: str):
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})

    pipeline, err = _get_pipeline()
    if not pipeline:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Pipeline unavailable: {err}",
            "meta": {},
        })
        st.rerun()
        return

    s = st.session_state.settings
    with st.spinner("Retrieving…"):
        try:
            response = pipeline.query(
                query,
                top_k=s["top_k"],
                return_context=True,
            )
            answer  = response.answer
            route   = response.route_used.value
            latency = response.latency_ms
            conf    = response.metadata.get("classifier_confidence", 0.0)
            reasoning = response.metadata.get("intent_reasoning", "")
            chunks  = response.retrieved_chunks

        except Exception as e:
            answer    = f"⚠️ Query failed: {type(e).__name__}: {e}"
            route     = "vector"
            latency   = 0.0
            conf      = 0.0
            reasoning = ""
            chunks    = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "route":     route,
            "latency_ms": latency,
            "confidence": conf,
            "reasoning":  reasoning,
            "chunks":     chunks,
        },
    })
    st.rerun()


# ── Tab: Ingest 
def _tab_ingest():
    st.markdown("### Upload and index documents")
    st.caption("Supported: PDF, DOCX, TXT, HTML, CSV, XLSX")

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "txt", "html", "htm", "csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns(3)
    skip_vector = col_a.checkbox("Skip vector index (Qdrant)",    value=False)
    skip_bm25   = col_b.checkbox("Skip BM25 index (Elasticsearch)", value=False)
    skip_sql    = col_c.checkbox("Skip SQL index (SQLite)",        value=False)

    if st.button("⬆️ Ingest documents", disabled=not uploaded, use_container_width=False,
                 type="primary"):
        _run_ingest(uploaded, skip_vector, skip_bm25, skip_sql)

    # ── DB stats 
    st.divider()
    st.markdown("### Index status")
    _render_index_stats()


def _run_ingest(uploaded_files, skip_vector: bool, skip_bm25: bool, skip_sql: bool):
    from ingestion.scraper import DocumentScraper
    from ingestion.chunker import DocumentChunker

    scraper = DocumentScraper()
    chunker = DocumentChunker()
    progress = st.progress(0, text="Preparing…")
    log      = st.empty()
    lines: list[str] = []

    def _log(msg: str):
        lines.append(msg)
        log.code("\n".join(lines[-20:]), language="text")

    # ── Write uploads to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for uf in uploaded_files:
            (tmp / uf.name).write_bytes(uf.read())
        _log(f"Saved {len(uploaded_files)} file(s) to temp directory")
        progress.progress(10, text="Loading…")

        docs = scraper.load_directory(tmp)
        _log(f"Loaded {len(docs)} document(s)")
        if not docs:
            st.error("No documents could be loaded. Check file formats.")
            return

        progress.progress(25, text="Chunking…")
        chunks = chunker.chunk_batch(docs)
        _log(f"Created {len(chunks)} chunks")
        progress.progress(40, text="Indexing…")

        # ── Vector
        if not skip_vector:
            try:
                from vector_rag.embed import VectorIndexer
                n = VectorIndexer().index(chunks)
                _log(f"✓ Qdrant: {n} chunks indexed")
            except Exception as e:
                _log(f"✗ Qdrant: {e}")
        progress.progress(60, text="BM25…")

        # ── BM25 
        if not skip_bm25:
            try:
                from vectorless_rag.bm25 import ElasticsearchBM25
                n = ElasticsearchBM25().index_chunks(chunks)
                _log(f"✓ Elasticsearch: {n} chunks indexed")
            except Exception:
                try:
                    from vectorless_rag.bm25 import InMemoryBM25
                    bm25 = InMemoryBM25()
                    bm25.index_chunks(chunks)
                    _log(f"✓ InMemoryBM25: {len(chunks)} chunks indexed")
                except Exception as e2:
                    _log(f"✗ BM25: {e2}")
        progress.progress(80, text="SQL…")

        # ── SQL
        if not skip_sql:
            try:
                from vectorless_rag.sql_retriever import SQLRetriever
                sql = SQLRetriever()
                nd  = sql.index_documents(docs)
                nc  = sql.index_chunks(chunks)
                sql.close()
                _log(f"✓ SQLite: {nd} docs, {nc} chunks")
            except Exception as e:
                _log(f"✗ SQLite: {e}")

    progress.progress(100, text="Done")
    st.session_state.ingested_docs   += len(docs)
    st.session_state.ingested_chunks += len(chunks)
    st.success(f"✓ Ingested {len(docs)} document(s), {len(chunks)} chunks")


def _render_index_stats():
    cols = st.columns(4)

    # SQLite stats
    try:
        from vectorless_rag.sql_retriever import SQLRetriever
        sql   = SQLRetriever()
        stats = sql.stats()
        sql.close()
        cols[0].markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['documents']}</div>
            <div class="stat-label">Documents (SQLite)</div>
        </div>""", unsafe_allow_html=True)
        cols[1].markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['chunks']}</div>
            <div class="stat-label">Chunks (SQLite)</div>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        cols[0].warning(f"SQLite: {e}")

    # Qdrant stats
    try:
        from qdrant_client import QdrantClient
        from src.config import get_settings
        cfg    = get_settings()
        client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        info   = client.get_collection(cfg.qdrant.collection_name)
        n_vecs = info.vectors_count or 0
        cols[2].markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{n_vecs}</div>
            <div class="stat-label">Vectors (Qdrant)</div>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        cols[2].warning(f"Qdrant: {e}")

    # Classifier model
    try:
        from src.config import get_settings
        model_path = Path(get_settings().classifier.model_path)
        if model_path.exists():
            kb = model_path.stat().st_size // 1024
            cols[3].markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{kb} KB</div>
                <div class="stat-label">Classifier model</div>
            </div>""", unsafe_allow_html=True)
        else:
            cols[3].markdown("""
            <div class="stat-card">
                <div class="stat-value" style="color:#EF4444">None</div>
                <div class="stat-label">Classifier model</div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        pass


# ── Tab: Evaluation

def _tab_evaluation():
    st.markdown("### Evaluation")
    st.caption(
        "Upload a CSV with columns `query`, `reference` (and optionally `expected_route`) "
        "to run RAGAS + ROUGE-L metrics."
    )

    eval_file = st.file_uploader(
        "Evaluation dataset CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns(3)
    sample    = col1.number_input("Max samples", 5, 500, 50, step=5)
    no_llm    = col2.checkbox("ROUGE-L only (no API calls)", value=False)
    save_json = col3.checkbox("Save JSON report", value=True)

    if st.button("▶ Run evaluation", disabled=not eval_file,
                 type="primary", use_container_width=False):
        _run_evaluation(eval_file, int(sample), no_llm, save_json)

    # ── Download last report 
    report_dir = ROOT / "evaluation" / "reports"
    reports    = sorted(report_dir.glob("report_*.json"), reverse=True) if report_dir.exists() else []
    if reports:
        st.divider()
        st.markdown("### Previous reports")
        for rp in reports[:5]:
            with open(rp) as f:
                import json
                data = json.load(f)
            macro = data.get("metrics_summary", {})
            label = f"{rp.stem}  —  " + "  ".join(
                f"{k}={v:.3f}" for k, v in sorted(macro.items())
            )
            col_l, col_dl = st.columns([8, 1])
            col_l.markdown(f"**{label}**")
            col_dl.download_button(
                "⬇",
                data=open(rp, "rb").read(),
                file_name=rp.name,
                mime="application/json",
                key=f"dl_{rp.name}",
            )


def _run_evaluation(eval_file, sample: int, no_llm: bool, save_json: bool):
    import tempfile, os

    pipeline, err = _get_pipeline()
    if not pipeline:
        st.error(f"Pipeline unavailable: {err}")
        return

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tf:
        tf.write(eval_file.read())
        tmp_path = tf.name

    progress = st.progress(0, text="Starting evaluation…")
    log      = st.empty()

    try:
        from evaluation.metrics import RAGEvaluator
        evaluator = RAGEvaluator(pipeline, use_llm_metrics=not no_llm)
        log.info("Evaluator initialised")
        progress.progress(10, text="Running samples…")

        report = evaluator.run(tmp_path, sample=sample)
        progress.progress(90, text="Aggregating…")

        # ── Show results 
        st.markdown("### Results")
        metric_cols = st.columns(len(report.metrics_summary) or 1)
        for i, (name, score) in enumerate(sorted(report.metrics_summary.items())):
            metric_cols[i % len(metric_cols)].metric(
                name.replace("_", " ").title(),
                f"{score:.3f}",
            )

        if report.routing_accuracy is not None:
            st.metric("Routing accuracy", f"{report.routing_accuracy:.3f}")

        st.markdown("#### Per-route breakdown")
        import pandas as pd
        df = pd.DataFrame(report.per_route).T
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### Latency (ms)")
        lat = report.latency_stats.get("overall", {})
        lat_cols = st.columns(4)
        for col, key in zip(lat_cols, ["mean", "p50", "p95", "p99"]):
            col.metric(key.upper(), f"{lat.get(key, 0):.1f}")
        from datetime import datetime, timezone
        if save_json:
            from datetime import datetime
            report_dir = ROOT / "evaluation" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = report_dir / f"report_{ts}.json"
            report.save(path)
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download report JSON",
                    data=f.read(),
                    file_name=path.name,
                    mime="application/json",
                )

        progress.progress(100, text="Done")

    except Exception as e:
        st.error(f"Evaluation failed: {type(e).__name__}: {e}")
    finally:
        os.unlink(tmp_path)


# ── Tab: Settings

def _tab_settings():
    st.markdown("### Settings")

    with st.expander("LLM configuration", expanded=True):
        try:
            from src.config import get_settings
            cfg = get_settings()
            st.json({
                "provider":    cfg.llm.provider,
                "model":       cfg.llm.model,
                "max_tokens":  cfg.llm.max_tokens,
                "temperature": cfg.llm.temperature,
            })
        except Exception as e:
            st.warning(f"Could not load config: {e}")

    with st.expander("Retrieval configuration"):
        try:
            from src.config import get_settings
            cfg = get_settings()
            st.json({
                "fusion_method":    cfg.retrieval.fusion_method,
                "rrf_k":            cfg.retrieval.rrf_k,
                "reranker_top_k":   cfg.retrieval.reranker_top_k,
                "reranker_model":   cfg.retrieval.reranker_model,
            })
        except Exception as e:
            st.warning(f"Could not load config: {e}")

    with st.expander("Classifier configuration"):
        try:
            from src.config import get_settings
            cfg = get_settings()
            st.json({
                "mode":                 cfg.classifier.mode,
                "confidence_threshold": cfg.classifier.confidence_threshold,
                "fallback":             cfg.classifier.fallback,
                "model_path":           cfg.classifier.model_path,
            })
        except Exception as e:
            st.warning(f"Could not load config: {e}")

    st.divider()
    st.markdown("### Classifier training")
    st.caption("Retrain the route classifier with your current training data.")

    col1, col2 = st.columns(2)
    if col1.button("🔄 Train classifier", type="primary"):
        with st.spinner("Training…"):
            try:
                from classifier.train import train
                arts = train()
                st.success(f"Classifier trained — model saved to {get_settings().classifier.model_path}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if col2.button("📋 Evaluate classifier"):
        test_csv = ROOT / "classifier" / "data" / "test.csv"
        if test_csv.exists():
            with st.spinner("Evaluating…"):
                try:
                    from classifier.train import evaluate
                    result = evaluate(str(test_csv))
                    st.metric("Macro F1", f"{result.get('macro_f1', 0):.4f}")
                    st.code(result.get("report", ""), language="text")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
        else:
            st.warning(f"Test CSV not found at {test_csv}")

    st.divider()
    st.markdown("### Environment")
    import os
    env_keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GOOGLE_API_KEY":    bool(os.getenv("GOOGLE_API_KEY")),
    }
    for key, present in env_keys.items():
        icon = "✅" if present else "❌"
        st.markdown(f"{icon}  `{key}` — {'set' if present else 'not set'}")


# ── Main layout 

def main():
    _sidebar()

    st.markdown(
        "## 🗂️ Document Intelligence",
        help="Hybrid-RAG: routes each query to semantic search, structured lookup, or both.",
    )

    tab_chat, tab_ingest, tab_eval, tab_settings = st.tabs([
        "💬 Chat",
        "📂 Ingest",
        "📊 Evaluation",
        "⚙️ Settings",
    ])

    with tab_chat:
        _tab_chat()

    with tab_ingest:
        _tab_ingest()

    with tab_eval:
        _tab_evaluation()

    with tab_settings:
        _tab_settings()


if __name__ == "__main__":
    main()