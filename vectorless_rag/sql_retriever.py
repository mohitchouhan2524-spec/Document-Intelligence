"""
vectorless_rag/sql_retriever.py
────────────────────────────────────────────────────────────────────────────────
SQLite-backed structured retriever for Hybrid-RAG Document Intelligence.

Handles queries routed to VectorlessMethod.SQL by the classifier — typically:
  • Exact ID lookups:   "what is the total on invoice INV-2041"
  • Field comparisons:  "list contracts expiring in Q1 2025"
  • Aggregations:       "how many invoices are overdue"
  • Status queries:     "get the status of contract CT-0034"
  • Date range scans:   "documents uploaded between 2024-01-01 and 2024-06-30"

Architecture
────────────
                    ┌───────────────────────────────┐
    Natural-language │   QueryParser                 │
    SQL query        │   extract IDs, dates,         │
         ──────────▶ │   fields, operators           │
                     └──────────┬────────────────────┘
                                │  ParsedQuery
                                ▼
                     ┌──────────────────────────────┐
                     │   SQLRetriever               │
                     │   build + execute SQL        │
                     │   SQLite via SQLAlchemy       │
                     └──────────┬───────────────────┘
                                │  list[RetrievedChunk]
                                ▼
                          fusion.py

Database schema
───────────────
    documents       — one row per ingested document
    chunks          — one row per chunk (FK → documents)

    documents
    ──────────
    doc_id          TEXT  PK
    source          TEXT        file path or URL
    filename        TEXT
    filetype        TEXT        pdf | docx | txt | html | csv
    doc_ref         TEXT        structured ID extracted on ingest (INV-001, PO-123…)
    doc_type        TEXT        invoice | contract | order | ticket | policy | report | other
    status          TEXT        ACTIVE | DRAFT | PENDING | EXPIRED | CLOSED
    amount          REAL        monetary value if present
    currency        TEXT        USD | EUR | GBP …
    vendor          TEXT
    department      TEXT
    owner           TEXT        author / assigned user
    created_at      TEXT        ISO-8601
    expiry_date     TEXT        ISO-8601
    approval_date   TEXT        ISO-8601
    page_count      INTEGER
    word_count      INTEGER
    version         TEXT
    tags            TEXT        comma-separated
    extra_meta      TEXT        JSON blob for anything else

    chunks
    ──────
    chunk_id        TEXT  PK
    doc_id          TEXT  FK → documents.doc_id
    content         TEXT
    chunk_index     INTEGER
    token_count     INTEGER

Public API
──────────
    from vectorless_rag.sql_retriever import SQLRetriever

    # ── Indexing (run once after ingestion) ───────────────────────────────
    retriever = SQLRetriever()
    retriever.index_documents(docs)   # list[Document]
    retriever.index_chunks(chunks)    # list[Chunk]

    # ── Retrieval (called by fusion.py at query time) ─────────────────────
    results = retriever.search("what is the total on invoice INV-2041")
    results = retriever.search("list contracts expiring in Q1 2025", top_k=20)

    # ── Direct structured query (bypass NL parsing) ───────────────────────
    results = retriever.query_structured(
        doc_type="invoice", status="PENDING", limit=10
    )

    # ── Raw SQL (power users / evaluation) ───────────────────────────────
    rows = retriever.raw_sql("SELECT doc_ref, amount FROM documents WHERE doc_type='invoice'")

CLI
───
    python -m vectorless_rag.sql_retriever index   --docs  data/raw/
    python -m vectorless_rag.sql_retriever search  "what is the status of PO-8812"
    python -m vectorless_rag.sql_retriever inspect
    python -m vectorless_rag.sql_retriever reset
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings
from src.models import Chunk, Document, RetrievedChunk, RouteType


# ── Schema DDL 
_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    filename      TEXT,
    filetype      TEXT,
    doc_ref       TEXT,          -- structured ID e.g. INV-2041, PO-8812
    doc_type      TEXT,          -- invoice | contract | order | ticket | policy | report | other
    status        TEXT,          -- ACTIVE | DRAFT | PENDING | EXPIRED | CLOSED
    amount        REAL,
    currency      TEXT,
    vendor        TEXT,
    department    TEXT,
    owner         TEXT,
    created_at    TEXT,
    expiry_date   TEXT,
    approval_date TEXT,
    page_count    INTEGER,
    word_count    INTEGER,
    version       TEXT,
    tags          TEXT,          -- comma-separated
    extra_meta    TEXT           -- JSON
);
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    chunk_index INTEGER,
    token_count INTEGER
);
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_doc_ref     ON documents(doc_ref);",
    "CREATE INDEX IF NOT EXISTS idx_doc_type    ON documents(doc_type);",
    "CREATE INDEX IF NOT EXISTS idx_status      ON documents(status);",
    "CREATE INDEX IF NOT EXISTS idx_vendor      ON documents(vendor);",
    "CREATE INDEX IF NOT EXISTS idx_department  ON documents(department);",
    "CREATE INDEX IF NOT EXISTS idx_created_at  ON documents(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_expiry_date ON documents(expiry_date);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);",
]


# ── Query parser 

# Regex bank — each pattern extracts one kind of signal from NL queries.
# Ordered from most-specific to least-specific.

# Structured document IDs: INV-2041, PO-8812, CT-0034, DOC-1193, ORD-9921 …
_RE_STRUCT_ID = re.compile(
    r"\b([A-Z]{2,6}-?\d{3,})\b"
)

# Bare numeric ID "invoice 4421" (no prefix)
_RE_BARE_NUM = re.compile(
    r"\b(invoice|order|contract|po|ticket|document|doc|case|ref|sku|project|agreement|licence|license)\s+#?(\d{3,})\b",
    re.IGNORECASE,
)

# Monetary amounts: "above 50000", "over 1M", "below 500k", "> 10000"
_RE_AMOUNT = re.compile(
    r"(above|over|below|under|greater than|less than|>=|<=|>|<)\s*\$?([\d,]+(?:\.\d+)?)\s*(k|m|million|thousand)?",
    re.IGNORECASE,
)

# Date patterns
_RE_FULL_DATE  = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b")
_RE_QUARTER    = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)
_RE_YEAR_ONLY  = re.compile(r"\b(20\d{2})\b")

# Status keywords
_RE_STATUS = re.compile(
    r"\b(active|draft|pending|expired|closed|overdue|auto|approved)\b",
    re.IGNORECASE,
)

# Doc-type keywords
_RE_DOCTYPE = re.compile(
    r"\b(invoices?|contracts?|orders?|tickets?|documents?|docs?|policies|policy|"
    r"reports?|agreements?|licen[sc]es?|templates?|proposals?|sow|briefs?)\b",
    re.IGNORECASE,
)

# Aggregation intent
_RE_AGGREGATE = re.compile(
    r"\b(how many|count|total number|number of)\b",
    re.IGNORECASE,
)

# Date field targets
_RE_EXPIRY     = re.compile(r"\b(expir|expire|expiry|expires|valid until|valid to)\b", re.IGNORECASE)
_RE_APPROVAL   = re.compile(r"\b(approval date|approved on|date approved)\b",          re.IGNORECASE)
_RE_CREATED    = re.compile(r"\b(created|uploaded|submitted|added)\b",                  re.IGNORECASE)

# Department / vendor cues
_RE_DEPT       = re.compile(r"\bdepartment\s+([A-Z][A-Z0-9_\- ]{1,30})\b",   re.IGNORECASE)
_RE_VENDOR     = re.compile(r"\bvendor\s+([A-Z][A-Za-z0-9_\- ]{1,30})\b",    re.IGNORECASE)
_RE_USER       = re.compile(r"\buser\s+(?:ID\s+)?([A-Z]{2,6}-\d{3,})\b",     re.IGNORECASE)


@dataclass
class ParsedQuery:
    """
    Structured intent extracted from a natural-language SQL query.
    Used by _QueryBuilder to assemble a parameterised SQL statement.
    """
    struct_ids:      list[str]         = field(default_factory=list)   # INV-2041, PO-8812
    bare_ids:        list[tuple[str,str]] = field(default_factory=list) # ("invoice","4421")
    doc_types:       list[str]         = field(default_factory=list)
    statuses:        list[str]         = field(default_factory=list)
    amount_filter:   tuple[str,float] | None = None                    # (operator, value)
    date_filters:    list[tuple[str,str,str]] = field(default_factory=list) # (field, op, iso_date)
    department:      str | None        = None
    vendor:          str | None        = None
    user_id:         str | None        = None
    is_aggregate:    bool              = False
    raw_query:       str               = ""


def _parse_query(query: str) -> ParsedQuery:
    """Extract structured intent from a natural-language query string."""
    pq = ParsedQuery(raw_query=query)

    # ── Structured IDs 
    pq.struct_ids = list(dict.fromkeys(_RE_STRUCT_ID.findall(query)))

    # ── Bare numeric IDs  "invoice 4421" 
    pq.bare_ids = [
        (m.group(1).lower(), m.group(2))
        for m in _RE_BARE_NUM.finditer(query)
    ]

    # ── Doc types 
    _DOCTYPE_MAP = {
        "invoice": "invoice", "invoices": "invoice",
        "contract": "contract", "contracts": "contract",
        "agreement": "contract", "agreements": "contract",
        "order": "order", "orders": "order",
        "ticket": "ticket", "tickets": "ticket",
        "document": None, "documents": None, "doc": None, "docs": None,
        "policy": "policy", "policies": "policy",
        "report": "report", "reports": "report",
        "licence": "licence", "license": "licence",
        "licences": "licence", "licenses": "licence",
        "template": "template", "templates": "template",
        "proposal": "proposal", "proposals": "proposal",
        "sow": "sow", "brief": "brief", "briefs": "brief",
    }
    found_types = set()
    for m in _RE_DOCTYPE.finditer(query):
        mapped = _DOCTYPE_MAP.get(m.group(1).lower())
        if mapped:
            found_types.add(mapped)
    pq.doc_types = list(found_types)

    # ── Status 
    _STATUS_MAP = {
        "active": "ACTIVE", "draft": "DRAFT", "pending": "PENDING",
        "expired": "EXPIRED", "closed": "CLOSED", "approved": "APPROVED",
        "auto": "AUTO",
    }
    # "overdue" → status PENDING + expiry < today
    if re.search(r"\boverdue\b", query, re.IGNORECASE):
        pq.statuses.append("PENDING")
        today = date.today().isoformat()
        pq.date_filters.append(("expiry_date", "<", today))
    else:
        for m in _RE_STATUS.finditer(query):
            mapped = _STATUS_MAP.get(m.group(1).lower())
            if mapped and mapped not in pq.statuses:
                pq.statuses.append(mapped)

    # ── Amount filter 
    m = _RE_AMOUNT.search(query)
    if m:
        op_raw  = m.group(1).lower().strip()
        val_str = m.group(2).replace(",", "")
        mult    = m.group(3) or ""
        value   = float(val_str)
        if mult.lower() in ("k", "thousand"):
            value *= 1_000
        elif mult.lower() in ("m", "million"):
            value *= 1_000_000
        op_map = {
            "above": ">", "over": ">", "greater than": ">", ">": ">",
            "below": "<", "under": "<", "less than": "<",  "<": "<",
            ">=": ">=", "<=": "<=",
        }
        pq.amount_filter = (op_map.get(op_raw, ">"), value)

    # ── Date filters 
    # Determine target date field from context
    if _RE_EXPIRY.search(query):
        date_field = "expiry_date"
    elif _RE_APPROVAL.search(query):
        date_field = "approval_date"
    else:
        date_field = "created_at"   # default

    for m in _RE_FULL_DATE.finditer(query):
        iso = m.group(1).replace("/", "-")
        # Single date: "on 2024-03-15" → exact day (use LIKE)
        pq.date_filters.append((date_field, "=", iso))

    for m in _RE_QUARTER.finditer(query):
        q_num, year = int(m.group(1)), int(m.group(2))
        month_start = (q_num - 1) * 3 + 1
        month_end   = q_num * 3
        start = f"{year}-{month_start:02d}-01"
        end   = f"{year}-{month_end:02d}-{'31' if month_end in (1,3,5,7,8,10,12) else '30'}"
        pq.date_filters.append((date_field, ">=", start))
        pq.date_filters.append((date_field, "<=", end))

    # ── Department / vendor / user 
    m = _RE_DEPT.search(query)
    if m:
        pq.department = m.group(1).strip()

    m = _RE_VENDOR.search(query)
    if m:
        pq.vendor = m.group(1).strip()

    m = _RE_USER.search(query)
    if m:
        pq.user_id = m.group(1).strip()

    # ── Aggregate intent
    pq.is_aggregate = bool(_RE_AGGREGATE.search(query))

    return pq


# ── SQL builder 

class _QueryBuilder:
    """
    Converts a ParsedQuery into a parameterised SQL SELECT statement.

    Always queries documents LEFT JOINed to chunks so the caller gets
    back content (chunk text) alongside document-level metadata.

    Returns (sql_string, params_tuple).
    """

    _SELECT_COLS = """
        d.doc_id, d.source, d.filename, d.filetype, d.doc_ref,
        d.doc_type, d.status, d.amount, d.currency, d.vendor,
        d.department, d.owner, d.created_at, d.expiry_date,
        d.approval_date, d.page_count, d.word_count, d.version,
        d.tags, d.extra_meta,
        c.chunk_id, c.content, c.chunk_index, c.token_count
    """

    def build(self, pq: ParsedQuery, limit: int = 10) -> tuple[str, tuple]:
        conditions: list[str] = []
        params:     list[Any] = []

        # ── Structured IDs → match doc_ref exactly 
        if pq.struct_ids:
            placeholders = ",".join("?" * len(pq.struct_ids))
            conditions.append(f"d.doc_ref IN ({placeholders})")
            params.extend(pq.struct_ids)

        # ── Bare numeric IDs → fuzzy match on doc_ref 
        for dtype, num in pq.bare_ids:
            conditions.append("d.doc_ref LIKE ?")
            params.append(f"%{num}%")

        # ── Doc type 
        if pq.doc_types:
            placeholders = ",".join("?" * len(pq.doc_types))
            conditions.append(f"d.doc_type IN ({placeholders})")
            params.extend(pq.doc_types)

        # ── Status 
        if pq.statuses:
            placeholders = ",".join("?" * len(pq.statuses))
            conditions.append(f"d.status IN ({placeholders})")
            params.extend(pq.statuses)

        # ── Amount 
        if pq.amount_filter:
            op, val = pq.amount_filter
            conditions.append(f"d.amount {op} ?")
            params.append(val)

        # ── Date filters 
        for dt_field, op, iso_val in pq.date_filters:
            if op == "=":
                # SQLite stores ISO strings, use LIKE for date-only match
                conditions.append(f"d.{dt_field} LIKE ?")
                params.append(f"{iso_val}%")
            else:
                conditions.append(f"d.{dt_field} {op} ?")
                params.append(iso_val)

        # ── Department / vendor / user 
        if pq.department:
            conditions.append("UPPER(d.department) = UPPER(?)")
            params.append(pq.department)

        if pq.vendor:
            conditions.append("d.vendor LIKE ?")
            params.append(f"%{pq.vendor}%")

        if pq.user_id:
            conditions.append("(d.owner LIKE ? OR d.doc_id LIKE ?)")
            params.extend([f"%{pq.user_id}%", f"%{pq.user_id}%"])

        # ── Fallback: FTS-style LIKE on doc_ref + filename 
        # If nothing matched structurally, do a broad keyword scan
        if not conditions:
            words = [w for w in re.split(r"\W+", pq.raw_query) if len(w) > 3][:5]
            if words:
                like_parts = []
                for w in words:
                    like_parts.append(
                        "(d.doc_ref LIKE ? OR d.filename LIKE ? OR c.content LIKE ?)"
                    )
                    params.extend([f"%{w}%", f"%{w}%", f"%{w}%"])
                conditions.append(f"({' OR '.join(like_parts)})")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if pq.is_aggregate:
            # COUNT query — returns a single row with a count
            sql = f"""
                SELECT COUNT(DISTINCT d.doc_id) AS count_result
                FROM documents d
                LEFT JOIN chunks c ON d.doc_id = c.doc_id
                {where_clause}
            """
        else:
            sql = f"""
                SELECT {self._SELECT_COLS}
                FROM documents d
                LEFT JOIN chunks c ON d.doc_id = c.doc_id
                {where_clause}
                ORDER BY d.created_at DESC
                LIMIT ?
            """
            params.append(limit)

        return sql.strip(), tuple(params)


# ── Document metadata extractor

# Maps common structured ID prefixes to doc_type
_ID_PREFIX_TO_TYPE: dict[str, str] = {
    "INV": "invoice",   "INVOICE": "invoice",
    "PO":  "order",     "ORD": "order",     "ORDER": "order",
    "CT":  "contract",  "CON": "contract",  "AGR": "contract",
    "DOC": "document",
    "TKT": "ticket",    "TICKET": "ticket",
    "LIC": "licence",
    "PR":  "project",
    "REF": "document",
    "CS":  "case",
    "USR": "user",
    "SKU": "product",
    "TPL": "template",
    "ATT": "attachment",
}

_RE_DOC_REF_EXTRACT = re.compile(r"\b([A-Z]{2,6}-?\d{3,})\b")


def _extract_doc_metadata(doc: Document) -> dict[str, Any]:
    """
    Heuristically extract structured fields from a Document's metadata
    and content. Looks for document IDs in filename and first 500 chars.
    """
    meta = dict(doc.metadata)
    result: dict[str, Any] = {}

    # ── doc_ref: look in filename first, then start of content 
    search_text = (meta.get("filename", "") + " " + doc.content[:500])
    refs = _RE_DOC_REF_EXTRACT.findall(search_text)
    result["doc_ref"] = refs[0] if refs else None

    # ── doc_type from ID prefix or filename
    doc_type = None
    if result["doc_ref"]:
        prefix = result["doc_ref"].split("-")[0].upper()
        doc_type = _ID_PREFIX_TO_TYPE.get(prefix)
    if not doc_type:
        fname = meta.get("filename", "").lower()
        for keyword, dtype in [
            ("invoice", "invoice"), ("contract", "contract"),
            ("order", "order"),    ("ticket", "ticket"),
            ("policy", "policy"),  ("report", "report"),
            ("licence", "licence"),("license", "licence"),
        ]:
            if keyword in fname:
                doc_type = dtype
                break
    result["doc_type"] = doc_type or "document"

    # ── status from content keywords 
    content_lower = doc.content[:300].lower()
    for status in ("draft", "pending", "expired", "closed", "approved", "active"):
        if status in content_lower:
            result["status"] = status.upper()
            break
    else:
        result["status"] = "ACTIVE"

    # ── amount: look for $12,450.00 or 12450 USD patterns
    amount_match = re.search(
        r"\$\s*([\d,]+(?:\.\d{2})?)|"
        r"([\d,]+(?:\.\d{2})?)\s*(USD|EUR|GBP|INR)",
        doc.content[:1000],
    )
    if amount_match:
        raw_amount = (amount_match.group(1) or amount_match.group(2) or "").replace(",", "")
        try:
            result["amount"] = float(raw_amount)
            result["currency"] = amount_match.group(3) or "USD"
        except ValueError:
            result["amount"] = None
            result["currency"] = None
    else:
        result["amount"]   = None
        result["currency"] = None

    # ── dates: look for ISO or common date formats 
    date_re  = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    all_dates = date_re.findall(doc.content[:2000])
    result["created_at"]    = meta.get("created_at", doc.created_at.isoformat())
    result["expiry_date"]   = all_dates[1] if len(all_dates) > 1 else None
    result["approval_date"] = all_dates[2] if len(all_dates) > 2 else None

    # ── page count 
    result["page_count"] = meta.get("page_count")
    result["word_count"] = len(doc.content.split())
    result["version"]    = meta.get("version")
    result["filename"]   = meta.get("filename")
    result["filetype"]   = meta.get("filetype")
    result["vendor"]     = meta.get("vendor")
    result["department"] = meta.get("department")
    result["owner"]      = meta.get("owner")
    result["tags"]       = meta.get("tags")

    # ── extra_meta: everything else that doesn't have a dedicated column 
    known_keys = {
        "filename", "filetype", "created_at", "page_count", "word_count",
        "version", "vendor", "department", "owner", "tags",
        "doc_ref", "doc_type", "status", "amount", "currency",
        "expiry_date", "approval_date",
    }
    extra = {k: v for k, v in meta.items() if k not in known_keys}
    result["extra_meta"] = json.dumps(extra) if extra else None

    return result


# ── Row → RetrievedChunk 
def _row_to_chunk(row: sqlite3.Row, score: float = 1.0) -> RetrievedChunk:
    """Convert a sqlite3.Row from the JOIN query into a RetrievedChunk."""
    row_dict = dict(row)
    content  = row_dict.get("content") or ""

    # If no chunk content (doc has no chunks yet), synthesise a summary row
    if not content:
        parts = []
        for field_name in ("doc_ref", "doc_type", "status", "amount",
                           "currency", "vendor", "expiry_date"):
            val = row_dict.get(field_name)
            if val is not None:
                parts.append(f"{field_name}: {val}")
        content = "  |  ".join(parts) if parts else f"Document: {row_dict.get('filename', row_dict['doc_id'])}"

    # Metadata for the chunk — everything except content itself
    metadata = {
        k: v for k, v in row_dict.items()
        if k not in ("content", "chunk_id", "doc_id") and v is not None
    }

    return RetrievedChunk(
        chunk_id=row_dict.get("chunk_id") or f"sql_{row_dict['doc_id']}",
        doc_id=row_dict["doc_id"],
        content=content,
        score=score,
        source=RouteType.VECTORLESS,
        metadata=metadata,
    )


def _aggregate_row_to_chunk(row: sqlite3.Row, query: str) -> RetrievedChunk:
    """Convert a COUNT(*) result row into a single RetrievedChunk."""
    count = dict(row).get("count_result", 0)
    content = f"Query result: {count} document(s) match your criteria."
    return RetrievedChunk(
        chunk_id="sql_aggregate_result",
        doc_id="aggregate",
        content=content,
        score=1.0,
        source=RouteType.VECTORLESS,
        metadata={"count": count, "query": query},
    )


# ── Main retriever 

class SQLRetriever:
    """
    SQLite-backed structured retriever for Document Intelligence.

    Translates natural-language SQL queries into parameterised SQLite
    queries, executes them, and returns RetrievedChunk objects compatible
    with the rest of the Hybrid-RAG pipeline.

    The same instance is reused across queries (connection pooling via
    SQLAlchemy check_same_thread=False for thread safety in Streamlit).
    """

    def __init__(self, db_path: str | None = None):
        import yaml
        from pathlib import Path as _Path
        cfg_path = _Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f)
        default_db = raw_cfg.get("sqlite", {}).get("db_path", "data/indexes/metadata.db")
        self.db_path = Path(db_path or default_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn   = self._connect()
        self._parser = _parse_query
        self._builder = _QueryBuilder()

        logger.info(f"SQLRetriever ready — db: {self.db_path}")

    # ── Connection 

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,    # safe for Streamlit / FastAPI
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")    # concurrent reads during writes
        conn.execute("PRAGMA foreign_keys=ON;")
        self._create_schema(conn)
        return conn

    def _create_schema(self, conn: sqlite3.Connection):
        conn.execute(_DDL_DOCUMENTS)
        conn.execute(_DDL_CHUNKS)
        for idx_sql in _DDL_INDEXES:
            conn.execute(idx_sql)
        conn.commit()
        logger.debug("SQLite schema initialised")

    # ── Indexing 

    def index_documents(self, docs: list[Document]) -> int:
        """
        Insert or replace Document objects into the documents table.
        Extracts structured metadata automatically.
        """
        if not docs:
            return 0

        rows = []
        for doc in docs:
            meta = _extract_doc_metadata(doc)
            rows.append((
                doc.doc_id,
                doc.source,
                meta.get("filename"),
                meta.get("filetype"),
                meta.get("doc_ref"),
                meta.get("doc_type"),
                meta.get("status"),
                meta.get("amount"),
                meta.get("currency"),
                meta.get("vendor"),
                meta.get("department"),
                meta.get("owner"),
                meta.get("created_at"),
                meta.get("expiry_date"),
                meta.get("approval_date"),
                meta.get("page_count"),
                meta.get("word_count"),
                meta.get("version"),
                meta.get("tags"),
                meta.get("extra_meta"),
            ))

        self._conn.executemany(
            """
            INSERT OR REPLACE INTO documents (
                doc_id, source, filename, filetype, doc_ref, doc_type,
                status, amount, currency, vendor, department, owner,
                created_at, expiry_date, approval_date, page_count,
                word_count, version, tags, extra_meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._conn.commit()
        logger.info(f"Indexed {len(rows)} documents into SQLite")
        return len(rows)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Insert or replace Chunk objects into the chunks table."""
        if not chunks:
            return 0

        rows = [
            (c.chunk_id, c.doc_id, c.content, c.chunk_index, c.token_count)
            for c in chunks
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, content, chunk_index, token_count)
            VALUES (?,?,?,?,?)
            """,
            rows,
        )
        self._conn.commit()
        logger.info(f"Indexed {len(rows)} chunks into SQLite")
        return len(rows)

    # ── Retrieval 

    def search(self, query: str, top_k: int = 10, **_) -> list[RetrievedChunk]:
        """
        Main retrieval entry point — called by fusion.py.
        Parses the natural-language query and returns matching chunks.
        """
        pq  = self._parser(query)
        sql, params = self._builder.build(pq, limit=top_k)

        logger.debug(f"SQL: {sql.strip()} | params={params}")

        try:
            cursor = self._conn.execute(sql, params)
            rows   = cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"SQLite query failed: {e}\nSQL: {sql}\nParams: {params}")
            return []

        if not rows:
            logger.debug(f"SQLRetriever: no results for '{query[:70]}'")
            return []

        if pq.is_aggregate:
            return [_aggregate_row_to_chunk(rows[0], query)]

        # Score: decay by position (1.0, 0.99, 0.98 …)
        results = [
            _row_to_chunk(row, score=round(1.0 - i * 0.01, 3))
            for i, row in enumerate(rows)
        ]
        logger.debug(f"SQLRetriever: {len(results)} results for '{query[:60]}'")
        return results

    # ── Structured query (bypass NL parsing)

    def query_structured(
        self,
        doc_type:   str | None = None,
        status:     str | None = None,
        vendor:     str | None = None,
        department: str | None = None,
        doc_ref:    str | None = None,
        limit:      int        = 10,
    ) -> list[RetrievedChunk]:
        """
        Direct structured query with explicit field values.
        Bypasses the NL parser — use when you already have structured filters.
        """
        conditions: list[str] = []
        params:     list[Any] = []
        if doc_type:
            conditions.append("d.doc_type = ?"); params.append(doc_type)
        if status:
            conditions.append("d.status = ?");   params.append(status.upper())
        if vendor:
            conditions.append("d.vendor LIKE ?"); params.append(f"%{vendor}%")
        if department:
            conditions.append("UPPER(d.department) = UPPER(?)"); params.append(department)
        if doc_ref:
            conditions.append("d.doc_ref LIKE ?"); params.append(f"%{doc_ref}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT {_QueryBuilder._SELECT_COLS}
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            {where}
            ORDER BY d.created_at DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as e:
            logger.error(f"query_structured failed: {e}")
            return []

        return [_row_to_chunk(r, score=round(1.0 - i * 0.01, 3)) for i, r in enumerate(rows)]

    # ── Raw SQL (evaluation / debugging) 

    def raw_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Execute arbitrary SQL and return raw row dicts.
        For evaluation, debugging, and reporting — not used by fusion.py.
        """
        try:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(f"raw_sql failed: {e}")
            return []

    # ── Inspect / admin ─

    def stats(self) -> dict[str, Any]:
        """Return database statistics — document count, chunk count, by doc_type."""
        doc_count   = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_type     = self._conn.execute(
            "SELECT doc_type, COUNT(*) as n FROM documents GROUP BY doc_type ORDER BY n DESC"
        ).fetchall()
        by_status   = self._conn.execute(
            "SELECT status, COUNT(*) as n FROM documents GROUP BY status ORDER BY n DESC"
        ).fetchall()
        return {
            "documents":  doc_count,
            "chunks":     chunk_count,
            "by_doc_type": {r["doc_type"] or "unknown": r["n"] for r in by_type},
            "by_status":   {r["status"]   or "unknown": r["n"] for r in by_status},
            "db_path":    str(self.db_path),
        }

    def reset(self):
        """Drop and recreate all tables. Use with caution."""
        self._conn.execute("DROP TABLE IF EXISTS chunks")
        self._conn.execute("DROP TABLE IF EXISTS documents")
        self._create_schema(self._conn)
        logger.warning("SQLite database reset — all data deleted")

    def close(self):
        self._conn.close()
        logger.debug("SQLite connection closed")


# ── CLI 

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectorless_rag.sql_retriever",
        description="Hybrid-RAG SQL retriever — index documents and search",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_idx = sub.add_parser("index", help="Index documents from a directory")
    p_idx.add_argument("--docs", required=True, help="Directory containing documents")

    # search
    p_srch = sub.add_parser("search", help="Search the SQL index")
    p_srch.add_argument("query", type=str)
    p_srch.add_argument("--top-k", type=int, default=10)

    # inspect
    sub.add_parser("inspect", help="Print database statistics")

    # reset
    sub.add_parser("reset", help="Drop and recreate all tables (DESTRUCTIVE)")

    return parser


def main():
    args = _build_parser().parse_args()
    retriever = SQLRetriever()

    if args.command == "index":
        from ingestion.scraper import DocumentScraper
        from ingestion.chunker import DocumentChunker
        docs   = DocumentScraper().load_directory(args.docs)
        chunks = DocumentChunker().chunk_batch(docs)
        n_docs   = retriever.index_documents(docs)
        n_chunks = retriever.index_chunks(chunks)
        print(f"Indexed {n_docs} documents, {n_chunks} chunks")

    elif args.command == "search":
        results = retriever.search(args.query, top_k=args.top_k)
        if not results:
            print("No results found.")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r.score:.3f}  doc_id={r.doc_id}")
            print(f"    {r.content[:200]}")

    elif args.command == "inspect":
        import json as _json
        print(_json.dumps(retriever.stats(), indent=2))

    elif args.command == "reset":
        confirm = input("This will delete all indexed data. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            retriever.reset()
            print("Database reset.")
        else:
            print("Aborted.")

    retriever.close()


if __name__ == "__main__":
    main()