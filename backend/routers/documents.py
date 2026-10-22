import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend import config
from backend.limits import RateLimiter, acquire_slot, exclusive, ingest_slots
from backend.schemas import (
    DocumentLibrary,
    DocumentOut,
    FileResult,
    IngestResponse,
    MessageOut,
)

router = APIRouter()
log = logging.getLogger("hybrid_rag.documents")

ingest_limiter = RateLimiter(6, 60)
read_limiter = RateLimiter(60, 60)

CHUNK_BYTES = 1024 * 1024


def _store(request: Request, user):
    from vector_rag.embed import SupabaseVectorStore

    return SupabaseVectorStore(
        user_id=str(user.id), supabase_client=request.app.state.supabase
    )


def _library(store) -> DocumentLibrary:
    stats = store.stats()
    docs = [
        DocumentOut(
            doc_id=str(d.get("doc_id", "")),
            filename=str(d.get("filename") or d.get("doc_id", "")),
            filetype=str(d.get("filetype") or "").upper(),
            created_at=str(d.get("created_at") or ""),
        )
        for d in stats.get("doc_list", [])
    ]
    return DocumentLibrary(
        documents=docs,
        doc_count=int(stats.get("documents", len(docs))),
        chunk_count=int(stats.get("chunks", 0)),
        max_pdfs=int(stats.get("max_pdfs", config.MAX_PDFS_PER_USER)),
    )


def _save_uploads(tmp: Path, files: List[UploadFile]) -> List[str]:
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    seen: set = set()
    saved: List[str] = []
    for upload in files:
        original = Path(upload.filename or "upload").name
        stem, suffix = Path(original).stem, Path(original).suffix
        name, n = original, 1
        while name.lower() in seen:
            n += 1
            name = f"{stem}_{n}{suffix}"
        seen.add(name.lower())

        written = 0
        with open(tmp / name, "wb") as out:
            while True:
                block = upload.file.read(CHUNK_BYTES)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{original} exceeds the {config.MAX_UPLOAD_MB} MB limit",
                    )
                out.write(block)
        if written == 0:
            raise HTTPException(status_code=400, detail=f"{original} is empty")
        saved.append(name)
    return saved


def _validate(request: Request, user, files: List[UploadFile]) -> None:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    if len(files) > config.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {config.MAX_FILES_PER_REQUEST} files at a time",
        )
    pdf_count = 0
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename or 'unnamed file'}",
            )
        if suffix == ".pdf":
            pdf_count += 1
    if pdf_count:
        try:
            max_pdfs = _library(_store(request, user)).max_pdfs
        except Exception:
            max_pdfs = config.MAX_PDFS_PER_USER
        if pdf_count > max_pdfs:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"You uploaded {pdf_count} PDFs but each account can store at most "
                    f"{max_pdfs}. Upload {max_pdfs} or fewer at a time."
                ),
            )


def _ingest(
    request: Request, user, files: List[UploadFile], skip_bm25: bool, skip_sql: bool
) -> IngestResponse:
    from ingestion.chunker import DocumentChunker
    from ingestion.scraper import DocumentScraper

    uid = str(user.id)
    lines: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        saved = _save_uploads(tmp, files)
        lines.append(f"Saved {len(saved)} file(s)")

        docs = DocumentScraper().load_directory(tmp)
        lines.append(f"Loaded {len(docs)} document(s)")
        if not docs:
            raise HTTPException(
                status_code=422,
                detail="No documents could be loaded. Check the file formats.",
            )

        chunks = DocumentChunker().chunk_batch(docs)
        lines.append(f"Created {len(chunks)} chunks")

        try:
            from vector_rag.embed import EmbeddingEngine, VectorIndexer

            store = _store(request, user)
            engine = request.app.state.engine or EmbeddingEngine()
            indexer = VectorIndexer(store=store, engine=engine)
        except Exception as exc:
            log.exception("Vector indexer setup failed")
            raise HTTPException(
                status_code=500, detail=f"Vector indexing unavailable: {exc}"
            )

        indexed_docs = []
        indexed_chunks = []
        doc_chunk_counts: dict = {}
        doc_errors: dict = {}

        for doc in docs:
            doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
            label = doc.metadata.get("filename", doc.doc_id)
            try:
                n = indexer.index(doc_chunks, doc=doc)
                n = n if isinstance(n, int) else len(doc_chunks)
                indexed_docs.append(doc)
                indexed_chunks.extend(doc_chunks)
                doc_chunk_counts[doc.doc_id] = n
                lines.append(f"✓ Vector: {label} → {n} chunks")
            except Exception as exc:
                log.exception("Indexing failed for %s", label)
                doc_errors[doc.doc_id] = str(exc)
                lines.append(f"✗ Vector: {label}: {exc}")

        if indexed_chunks and not skip_bm25:
            try:
                from vectorless_rag.bm25 import ElasticsearchBM25

                n = ElasticsearchBM25().index_chunks(indexed_chunks)
                lines.append(f"✓ Elasticsearch BM25: {n} chunks")
            except Exception:
                try:
                    from vectorless_rag.bm25 import InMemoryBM25

                    InMemoryBM25().index_chunks(indexed_chunks)
                    lines.append(f"✓ InMemoryBM25: {len(indexed_chunks)} chunks")
                except Exception as exc:
                    lines.append(f"✗ BM25: {exc}")

        if indexed_docs and not skip_sql:
            try:
                from vectorless_rag.sql_retriever import SQLRetriever

                sql = SQLRetriever()
                try:
                    nd = sql.index_documents(indexed_docs)
                    nc = sql.index_chunks(indexed_chunks)
                finally:
                    sql.close()
                lines.append(f"✓ SQLite: {nd} docs, {nc} chunks")
            except Exception as exc:
                lines.append(f"✗ SQLite: {exc}")

        results: List[FileResult] = []
        for name in saved:
            file_docs = [d for d in docs if d.metadata.get("filename") == name]
            if not file_docs:
                results.append(
                    FileResult(
                        filename=name,
                        status="failed",
                        message="The file could not be read",
                    )
                )
                continue
            ok = [d for d in file_docs if d.doc_id in doc_chunk_counts]
            if ok:
                results.append(
                    FileResult(
                        filename=name,
                        status="indexed",
                        documents=len(ok),
                        chunks=sum(doc_chunk_counts[d.doc_id] for d in ok),
                    )
                )
            else:
                message = next(
                    (doc_errors[d.doc_id] for d in file_docs if d.doc_id in doc_errors),
                    "Indexing failed",
                )
                results.append(FileResult(filename=name, status="failed", message=message))

    return IngestResponse(
        documents_indexed=len(indexed_docs),
        chunks_indexed=sum(doc_chunk_counts.values()),
        files=results,
        log=lines,
    )


@router.get("", response_model=DocumentLibrary)
def list_documents(request: Request, user=Depends(read_limiter.by_user())):
    try:
        return _library(_store(request, user))
    except Exception as exc:
        log.exception("Could not load library for user %s", user.id)
        raise HTTPException(
            status_code=500, detail=f"Could not load document library: {exc}"
        )


@router.delete("/{doc_id}", response_model=MessageOut)
def delete_document(
    doc_id: str, request: Request, user=Depends(read_limiter.by_user())
):
    with exclusive(
        f"ingest:{user.id}", "An upload is in progress. Try again when it finishes."
    ):
        store = _store(request, user)
        owned = {d.doc_id: d for d in _library(store).documents}
        if doc_id not in owned:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            store.delete_document(doc_id)
        except Exception as exc:
            log.exception("Delete failed for %s", doc_id)
            raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")
        return MessageOut(message=f"Deleted {owned[doc_id].filename}")


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: Request,
    files: List[UploadFile] = File(...),
    skip_bm25: bool = Form(False),
    skip_sql: bool = Form(False),
    user=Depends(ingest_limiter.by_user()),
):
    _validate(request, user, files)
    with exclusive(
        f"ingest:{user.id}", "An upload is already in progress for your account."
    ):
        with acquire_slot(
            ingest_slots, "The server is busy processing other uploads. Try again shortly."
        ):
            return _ingest(request, user, files, skip_bm25, skip_sql)
