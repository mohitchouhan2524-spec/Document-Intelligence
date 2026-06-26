"""
classifier
──────────
Query-route classifier for Hybrid-RAG Document Intelligence.

Routes each incoming query to one of three retrieval paths:
    RouteType.VECTOR      → dense semantic search (Qdrant)
    RouteType.VECTORLESS  → BM25 / SQL / graph (Elasticsearch + SQLite)
    RouteType.HYBRID      → both paths, results fused via RRF

Primary API
───────────
    from classifier import predict_intent

    intent = predict_intent("who approved the Q3 budget")
    # intent.route            → RouteType.VECTORLESS
    # intent.vectorless_method → VectorlessMethod.GRAPH
    # intent.confidence       → 0.91

Pipeline
────────
    1. rules.classify()     — fast regex pre-filter (no model load)
    2. train.predict_intent()  — calibrated SGD + handcrafted features
    3. fallback to HYBRID if confidence < threshold (config.yaml)

Training
────────
    python -m classifier.train train
    python -m classifier.train evaluate --data classifier/data/test.csv
    python -m classifier.train predict "your query here"
    python -m classifier.train active-learn --pool unlabelled.txt
"""
from .rules import classify as rules_classify

__all__ = [
    "rules_classify",
]