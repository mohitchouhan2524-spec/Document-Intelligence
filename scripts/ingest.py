from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
# ── Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _banner(text: str):
    sep = "─" * 60
    print(f"\n{sep}\n  {text}\n{sep}")


def _step(n: int, total: int, label: str):
    print(f"\n[{n}/{total}] {label}")


def _ok(msg: str):
    print(f"  ✓  {msg}")


def _skip(msg: str):
    print(f"  ⊘  SKIPPED — {msg}")


def _fail(msg: str):
    print(f"  ✗  FAILED  — {msg}")


# ── Step runners

def step_load(docs_path: Path, verbose: bool):
    """Step 1 — Load documents from disk."""
    from ingestion.scraper import DocumentScraper
    scraper = DocumentScraper()
    docs = scraper.load_directory(docs_path)
    if not docs:
        print(f"\n[ERROR] No documents loaded from: {docs_path.resolve()}")
        print("        Make sure your PDF/DOCX files are inside that folder.")
        sys.exit(1)
    return docs


def step_chunk(docs, verbose: bool):
    """Step 2 — Chunk documents into overlapping token windows."""
    from ingestion.chunker import DocumentChunker
    chunker = DocumentChunker()
    chunks  = chunker.chunk_batch(docs)
    if not chunks:
        print("[ERROR] Chunking produced 0 chunks — documents may be empty after extraction.")
        sys.exit(1)
    if verbose:
        for c in chunks[:5]:
            print(f"    chunk_id={c.chunk_id}  tokens={c.token_count}  "
                  f"preview={c.content[:60].strip()!r}")
        if len(chunks) > 5:
            print(f"    ... and {len(chunks)-5} more")
    _ok(f"{len(chunks)} chunks from {len(docs)} documents")
    return chunks

def step_vector(chunks, reset: bool):
    """Step 3 — Generate embeddings and upload to Supabase."""
    try:
        from vector_rag.embed import VectorIndexer
        indexer = VectorIndexer()
        if reset:
            indexer.store.clear_all_chunks()
            _ok("Supabase vector table cleared")
        n = indexer.index(chunks)
        _ok(f"Supabase pgvector: {n} chunks indexed")

    except Exception as e:
        _fail(f"Supabase indexing: {type(e).__name__}: {e}")

def step_bm25(chunks, reset: bool):
    """Step 4 — Index chunks into Elasticsearch (fallback: InMemoryBM25)."""
    try:
        from vectorless_rag.bm25 import ElasticsearchBM25
        bm25 = ElasticsearchBM25()
        if reset:
            try:
                bm25.client.indices.delete(index=bm25.index)
                bm25._ensure_index()
                _ok("Elasticsearch index reset")
            except Exception:
                pass
        n = bm25.index_chunks(chunks)
        _ok(f"Elasticsearch BM25: {n} chunks indexed")
    except Exception as es_err:
        print(f"  ⚠  Elasticsearch unavailable ({type(es_err).__name__}), "
              f"falling back to InMemoryBM25")
        try:
            from vectorless_rag.bm25 import InMemoryBM25
            bm25 = InMemoryBM25()
            bm25.index_chunks(chunks)
            _ok(f"InMemoryBM25: {len(chunks)} chunks indexed "
                f"(saved to data/indexes/bm25_index.pkl)")
        except Exception as e2:
            _fail(f"InMemoryBM25: {type(e2).__name__}: {e2}")


def step_sql(docs, chunks, reset: bool):
    """Step 5 — Index metadata into SQLite."""
    try:
        from vectorless_rag.sql_retriever import SQLRetriever
        sql = SQLRetriever()
        if reset:
            sql.reset()
            _ok("SQLite database reset")
        n_docs   = sql.index_documents(docs)
        n_chunks = sql.index_chunks(chunks)
        stats    = sql.stats()
        _ok(f"SQLite: {n_docs} documents, {n_chunks} chunks indexed")
        _ok(f"SQLite doc types: {stats['by_doc_type']}")
        sql.close()
    except Exception as e:
        _fail(f"SQLite indexing: {type(e).__name__}: {e}")


def step_graph(chunks, reset: bool):
    """Step 6 — Build knowledge graph from entities."""
    try:
        from vectorless_rag.tree_builder import KnowledgeGraphBuilder
        graph = KnowledgeGraphBuilder()
        if reset and Path("data/indexes/knowledge_graph.json").exists():
            Path("data/indexes/knowledge_graph.json").unlink()
            _ok("Knowledge graph reset")
        graph.build(chunks)
        _ok(f"Knowledge graph: {graph.graph.number_of_nodes()} nodes, "
            f"{graph.graph.number_of_edges()} edges")
    except ImportError as e:
        _fail(f"Missing dependency: {e}\n"
              f"         Install: pip install spacy networkx && "
              f"python -m spacy download en_core_web_sm")
    except Exception as e:
        _fail(f"Graph build: {type(e).__name__}: {e}")


# ── Main 

def main():
    parser = argparse.ArgumentParser(
        prog="scripts/ingest.py",
        description="Hybrid-RAG Supabase ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--docs",         default="data/pdfs",
                        help="Folder containing documents (default: data/pdfs)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Load + chunk only — skip all indexing")
    parser.add_argument("--skip-vector",  action="store_true",
                        help="Skip Supabase vector indexing")
    parser.add_argument("--skip-es",      action="store_true",
                        help="Skip BM25 / Elasticsearch indexing")
    parser.add_argument("--skip-sql",     action="store_true",
                        help="Skip SQLite metadata indexing")
    parser.add_argument("--skip-graph",   action="store_true",
                        help="Skip knowledge graph build")
    parser.add_argument("--reset",        action="store_true",
                        help="Wipe existing indexes before ingesting")
    parser.add_argument("--verbose", "-v",action="store_true",
                        help="Show chunk previews")
    args = parser.parse_args()

    docs_path = Path(args.docs)
    t_start   = time.perf_counter()
    TOTAL_STEPS = 2 + sum([
        not args.skip_vector,
        not args.skip_es,
        not args.skip_sql,
        not args.skip_graph,
    ])

    _banner("Hybrid-RAG  ·  Ingestion Pipeline")
    print(f"  Source  : {docs_path.resolve()}")
    print(f"  Dry run : {args.dry_run}")
    print(f"  Reset   : {args.reset}")

    step_n = 1

    # ── Step 1: Load
    _step(step_n, TOTAL_STEPS, "Loading documents"); step_n += 1
    docs = step_load(docs_path, args.verbose)

    # ── Step 2: Chunk 
    _step(step_n, TOTAL_STEPS, "Chunking documents"); step_n += 1
    chunks = step_chunk(docs, args.verbose)

    if args.dry_run:
        _banner("Dry run complete — skipping all indexing")
        print(f"  Documents : {len(docs)}")
        print(f"  Chunks    : {len(chunks)}")
        return

    # ── Step 3: Generate embeddings and upload to supabase
    if not args.skip_vector:
        _step(step_n, TOTAL_STEPS, "Embedding + Supabase indexing"); step_n += 1
        step_vector(chunks, args.reset)
    else:
        _skip("Supabase vector indexing (--skip-vector)")

    # ── Step 4: BM25
    if not args.skip_es:
        _step(step_n, TOTAL_STEPS, "BM25 indexing (Elasticsearch / InMemory)"); step_n += 1
        step_bm25(chunks, args.reset)
    else:
        _skip("BM25 indexing (--skip-es)")

    # ── Step 5: SQL
    if not args.skip_sql:
        _step(step_n, TOTAL_STEPS, "SQLite metadata indexing"); step_n += 1
        step_sql(docs, chunks, args.reset)
    else:
        _skip("SQLite indexing (--skip-sql)")

    # ── Step 6: Graph
    if not args.skip_graph:
        _step(step_n, TOTAL_STEPS, "Building knowledge graph"); step_n += 1
        step_graph(chunks, args.reset)
    else:
        _skip("Knowledge graph (--skip-graph)")

    # ── Summary 
    elapsed = time.perf_counter() - t_start
    _banner(f"Ingestion complete  ·  {elapsed:.1f}s")
    print(f"  Documents ingested : {len(docs)}")
    print(f"  Chunks created     : {len(chunks)}")
    print(f"\nNext step — train the classifier:")
    print(f"  python -m classifier.train train")
    print(f"\nThen test a query end-to-end:")
    print(f"  python -c \"from hybrid.fusion import run_query; "
          f"print(run_query('summarise the documents').answer)\"")

if __name__ == "__main__":
    main()