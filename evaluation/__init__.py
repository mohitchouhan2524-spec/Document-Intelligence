"""
evaluation:
RAG evaluation metrics for Hybrid-RAG Document Intelligence.

Exports (available after metrics.py is written)
    RAGEvaluator — runs RAGAS + ROUGE-L + latency benchmarks

Metrics tracked:
    faithfulness        — answer grounded in retrieved context (RAGAS)
    answer_relevancy    — answer addresses the query (RAGAS)
    context_precision   — retrieved chunks contain the answer (RAGAS)
    rouge_l             — lexical overlap with reference answers
    latency_ms          — end-to-end query time per route

Typical usage (once metrics.py exists):
    from evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    report    = evaluator.run(pipeline, test_csv="classifier/data/test.csv")
    # report → dict with per-metric scores + per-route breakdown

Status:
    metrics.py — not yet written (Phase 4).
"""
try:
    from evaluation.metrics import RAGEvaluator
    __all__ = ["RAGEvaluator"]
except ImportError:
    __all__ = []