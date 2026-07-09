"""
classifier/train.py
────────────────────────────────────────────────────────────────────────────────
ML classifier for Hybrid-RAG query routing in Document Intelligence.

Routes each user query to one of three retrieval paths:
    VECTOR      → dense semantic search via supabase pgvector
    VECTORLESS  → BM25 / SQL / graph via Elasticsearch + SQLite
    HYBRID      → both paths fused via RRF

Architecture
────────────
    ┌──────────────┐    ┌─────────────────────────┐    ┌──────────────┐
    │  Raw query   │───▶│  Feature engineering     │───▶│  Classifier  │
    └──────────────┘    │  (TF-IDF + handcrafted)  │    │  (SGD / LR)  │
                        └─────────────────────────┘    └──────┬───────┘
                                                               │
                        ┌──────────────────────────────────────┘
                        ▼
                ┌───────────────┐
                │  QueryIntent  │  → route + vectorless_method + confidence
                └───────────────┘

Training pipeline
─────────────────
1.  Build / load labelled dataset  (synthetic + user-supplied)
2.  Feature engineering            (TF-IDF unigrams+bigrams + 12 handcrafted)
3.  Train SGDClassifier            (log-loss → calibrated probabilities)
4.  Calibrate probabilities        (CalibratedClassifierCV, isotonic)
5.  Evaluate                       (macro-F1, per-class report, confusion matrix)
6.  Persist                        (pickle: model + vectoriser + label encoder)
7.  CLI                            (train / evaluate / predict / active-learn)

Usage
─────
    # Train from scratch (uses built-in synthetic data)
    python -m classifier.train train

    # Add your own labelled queries (CSV: query,label)
    python -m classifier.train train --data path/to/queries.csv

    # Evaluate saved model
    python -m classifier.train evaluate --data path/to/test.csv

    # Predict a single query
    python -m classifier.train predict "summarise the risk factors in Q3 report"

    # Active-learning loop (label uncertain predictions interactively)
    python -m classifier.train active-learn --pool path/to/unlabelled.txt
"""
from __future__ import annotations
import argparse
import csv
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix

from src.config import get_settings
from src.models import QueryIntent, RouteType, VectorlessMethod
from classifier.rules import classify as rules_classify

# ── Constants 

LABELS = [RouteType.VECTOR, RouteType.VECTORLESS, RouteType.HYBRID]
LABEL_NAMES = [l.value for l in LABELS]

# ── Synthetic training data 
# Format: (query_text, route_label, vectorless_method_or_None)
# ~40 examples per class — enough for a solid baseline; grows via active learning.

_SYNTHETIC: list[tuple[str, str, str | None]] = [

    # ── VECTOR (semantic / analytical) 
    ("summarise the key findings in the annual report",          "vector", None),
    ("what are the main risk factors described in this filing",  "vector", None),
    ("explain the product liability clause in plain English",    "vector", None),
    ("analyse the competitive landscape section",               "vector", None),
    ("what does the indemnification clause mean",               "vector", None),
    ("describe the data retention policy",                      "vector", None),
    ("what is the overall tone of this contract",               "vector", None),
    ("how does the pricing model work according to this doc",   "vector", None),
    ("what are the key obligations of the vendor",              "vector", None),
    ("give me an overview of the compliance requirements",      "vector", None),
    ("why was the merger blocked according to the report",      "vector", None),
    ("what are the implications of the new data privacy law",   "vector", None),
    ("explain the arbitration process described here",          "vector", None),
    ("what technical requirements are specified for the API",   "vector", None),
    ("how does the warranty coverage work",                     "vector", None),
    ("what are the termination conditions in this agreement",   "vector", None),
    ("compare the SLA terms across all service tiers",         "vector", None),
    ("what security controls are mandated by this policy",      "vector", None),
    ("summarize what changed in version 3 of the agreement",   "vector", None),
    ("what are the penalty clauses for late delivery",         "vector", None),
    ("identify the scope of work described in this SOW",       "vector", None),
    ("what payment terms does this contract specify",          "vector", None),
    ("what does the NDA cover in terms of information types",  "vector", None),
    ("how is intellectual property ownership handled here",    "vector", None),
    ("break down the revenue sharing structure",               "vector", None),
    ("what does this memo say about the restructuring plan",   "vector", None),
    ("what are the key deliverables mentioned in this brief",  "vector", None),
    ("explain the escalation procedure described in policy",   "vector", None),
    ("what standards does this document comply with",         "vector", None),
    ("what is the audit scope outlined in this engagement",   "vector", None),
    ("analyse the force majeure clause applicability",         "vector", None),
    ("what are the confidentiality obligations on both sides", "vector", None),
    ("how should disputes be resolved under this agreement",   "vector", None),
    ("what triggers the change control process",               "vector", None),
    ("describe the onboarding process in the service manual",  "vector", None),
    ("what does this whitepaper say about decentralisation",   "vector", None),
    ("what is the stated purpose of this regulatory document", "vector", None),
    ("explain the liability cap and exclusions in this deal",  "vector", None),
    ("what are the notice requirements for termination",       "vector", None),
    ("how are renewal terms structured in this subscription",  "vector", None),

    # ── VECTORLESS → SQL (exact / structured lookup) ──────────────────────────
    ("what is the total amount on invoice INV-2041",            "vectorless", "sql"),
    ("show me PO-8812 delivery date",                           "vectorless", "sql"),
    ("get the status of contract CT-0034",                      "vectorless", "sql"),
    ("what is the value of order ORD-9921",                     "vectorless", "sql"),
    ("fetch document ID DOC-1193 expiry date",                  "vectorless", "sql"),
    ("when was ticket TKT-5502 created",                        "vectorless", "sql"),
    ("what is the price of SKU AB-4401",                        "vectorless", "sql"),
    ("how many documents were uploaded on 2024-03-15",          "vectorless", "sql"),
    ("list all contracts expiring in Q1 2025",                  "vectorless", "sql"),
    ("what is the count of active vendor agreements",           "vectorless", "sql"),
    ("find record by ref number REF-0072",                      "vectorless", "sql"),
    ("what is the exact due date for invoice 4421",             "vectorless", "sql"),
    ("retrieve the document status for case CS-2210",           "vectorless", "sql"),
    ("how many invoices are overdue as of today",               "vectorless", "sql"),
    ("what is the total spend for vendor ACME Corp this year",  "vectorless", "sql"),
    ("look up project code PR-3309 budget",                     "vectorless", "sql"),
    ("get approval date for PO-1102",                           "vectorless", "sql"),
    ("what is the contract value for agreement AGR-0441",       "vectorless", "sql"),
    ("how many pages does document DOC-0091 have",              "vectorless", "sql"),
    ("find all documents uploaded by user ID USR-0042",         "vectorless", "sql"),
    ("what is the currency on invoice INV-9903",                "vectorless", "sql"),
    ("fetch the version number of template TPL-2201",           "vectorless", "sql"),
    ("what is the exact word count of document DOC-5510",       "vectorless", "sql"),
    ("show records where status is PENDING",                    "vectorless", "sql"),
    ("how many contracts are in DRAFT state",                   "vectorless", "sql"),
    ("what is the signatory name on agreement AGR-1122",        "vectorless", "sql"),
    ("get the file size of attachment ATT-0033",                "vectorless", "sql"),
    ("retrieve all documents with filetype PDF created in 2024","vectorless", "sql"),
    ("what department owns project PR-8801",                    "vectorless", "sql"),
    ("find the total number of clauses in contract CT-4420",    "vectorless", "sql"),
    ("when does licence LIC-0055 expire",                       "vectorless", "sql"),
    ("how many renewals has contract CT-0010 had",              "vectorless", "sql"),
    ("what is the discount percentage on order ORD-3304",       "vectorless", "sql"),
    ("get the tax rate applied to invoice INV-7701",            "vectorless", "sql"),
    ("show all documents tagged with category LEGAL",           "vectorless", "sql"),
    ("what is the payment method recorded for PO-2219",         "vectorless", "sql"),
    ("list documents uploaded between 2024-01-01 and 2024-06-30","vectorless","sql"),
    ("find documents where approval is NULL",                   "vectorless", "sql"),
    ("what is the maximum contract value in our system",        "vectorless", "sql"),
    ("how many unique vendors are in the database",             "vectorless", "sql"),

    # ── VECTORLESS → BM25 (keyword / phrase search) ───────────────────────────
    ("find all documents mentioning force majeure",             "vectorless", "bm25"),
    ("search for contracts containing the word indemnification","vectorless", "bm25"),
    ("find all files that mention GDPR",                        "vectorless", "bm25"),
    ("which documents contain the phrase data breach",          "vectorless", "bm25"),
    ("search for every record with the term arbitration",       "vectorless", "bm25"),
    ("find all contracts mentioning termination for cause",     "vectorless", "bm25"),
    ("which files include the keyword confidentiality",         "vectorless", "bm25"),
    ("search documents containing both NDA and exclusivity",    "vectorless", "bm25"),
    ("find all records that reference ISO 27001",               "vectorless", "bm25"),
    ("list documents that mention penalty clause",              "vectorless", "bm25"),
    ("which agreements contain the phrase limitation of liability","vectorless","bm25"),
    ("find every document that uses the term sub-contractor",   "vectorless", "bm25"),
    ("search for files mentioning the word escrow",             "vectorless", "bm25"),
    ("find all documents that reference SOC 2 compliance",      "vectorless", "bm25"),
    ("which contracts mention intellectual property assignment", "vectorless", "bm25"),
    ("search for all documents containing the phrase net 30",   "vectorless", "bm25"),
    ("find records that mention change order",                  "vectorless", "bm25"),
    ("which files contain the term representations and warranties","vectorless","bm25"),
    ("search for documents with the clause entire agreement",   "vectorless", "bm25"),
    ("find all policies that mention acceptable use",           "vectorless", "bm25"),

    # ── VECTORLESS → GRAPH (entity / relationship queries) ────────────────────
    ("who approved the Q3 budget document",                     "vectorless", "graph"),
    ("who authored the data governance policy",                 "vectorless", "graph"),
    ("who signed the vendor agreement with TechCorp",           "vectorless", "graph"),
    ("who is the owner of project Alpha documentation",         "vectorless", "graph"),
    ("which manager is responsible for this SLA",               "vectorless", "graph"),
    ("who created the compliance checklist last updated in May","vectorless", "graph"),
    ("who submitted the incident report IR-0042",               "vectorless", "graph"),
    ("what is the relationship between Acme Corp and vendor X", "vectorless", "graph"),
    ("which documents are linked to the Smith contract",        "vectorless", "graph"),
    ("who is associated with the legal review of this doc",     "vectorless", "graph"),
    ("find all entities mentioned alongside John Doe",          "vectorless", "graph"),
    ("which organisations are connected to this tender",        "vectorless", "graph"),
    ("what documents are related to the merger agreement",      "vectorless", "graph"),
    ("who wrote the technical specification for module B",      "vectorless", "graph"),
    ("which teams are mentioned in the escalation policy",      "vectorless", "graph"),

    # ── HYBRID (needs both semantic + structured) ─────────────────────────────
    ("how has our refund policy changed since 2022",            "hybrid", None),
    ("compare the SLA terms in the 2021 vs 2024 contracts",    "hybrid", None),
    ("show me how vendor pricing has evolved over the last 3 years","hybrid",None),
    ("what changed in the privacy policy between v1 and v2",   "hybrid", None),
    ("how have the termination clauses changed across renewals","hybrid", None),
    ("compare compliance requirements across all active vendors","hybrid",None),
    ("what differences exist between APAC and EMEA contracts",  "hybrid", None),
    ("which risk factors are new compared to last year report", "hybrid", None),
    ("show the trend in contract values over the past 5 years", "hybrid", None),
    ("how does the current data policy differ from 2020",       "hybrid", None),
    ("find all mentions of penalty clause and show their values","hybrid",None),
    ("list documents about liability and their approval dates",  "hybrid", None),
    ("which vendors mention GDPR and when did they first sign", "hybrid", None),
    ("how has the arbitration clause changed in each renewal",  "hybrid", None),
    ("compare indemnification scope in contracts above 1M value","hybrid",None),
    ("show me semantic context around clause 4.2 in all docs",  "hybrid", None),
    ("which documents mention data breach and are still active","hybrid", None),
    ("explain the payment terms for all overdue invoices",      "hybrid", None),
    ("find warranty clauses in contracts expiring this quarter","hybrid", None),
    ("compare the scope of NDA agreements signed after 2023",   "hybrid", None),
    ("what are the SLA penalties for vendor contracts over 500k","hybrid",None),
    ("list all force majeure clauses and their effective dates","hybrid", None),
    ("show pending documents that mention regulatory compliance","hybrid", None),
    ("find contracts by TechCorp and summarise their key terms","hybrid", None),
    ("which DRAFT agreements contain limitation of liability",   "hybrid", None),
]


# ── Feature engineering 

from classifier.features import HandcraftedFeatures

# ── Dataset helpers 

class LabelledQuery(NamedTuple):
    query: str
    route: str          # "vector" | "vectorless" | "hybrid"
    method: str | None  # "sql" | "bm25" | "graph" | None


def _build_synthetic_dataset() -> list[LabelledQuery]:
    return [LabelledQuery(q, r, m) for q, r, m in _SYNTHETIC]


def _load_csv_dataset(path: str | Path) -> list[LabelledQuery]:
    """
    Expects CSV with columns: query, label  (and optionally: method).
    label values: vector | vectorless | hybrid
    method values: sql | bm25 | graph  (optional, only relevant for vectorless)
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("query", "").strip()
            r = row.get("label", "").strip().lower()
            m = row.get("method", "").strip().lower() or None
            if q and r in LABEL_NAMES:
                rows.append(LabelledQuery(q, r, m))
    logger.info(f"Loaded {len(rows)} labelled examples from {path}")
    return rows


def _dataset_stats(dataset: list[LabelledQuery]) -> dict:
    from collections import Counter
    counts = Counter(d.route for d in dataset)
    return {"total": len(dataset), "by_class": dict(counts)}


# ── Model artifacts 

from classifier.artifacts import ClassifierArtifacts
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept any lookup from __main__ and redirect to classifier.train
        if module == "__main__":
            module = "classifier.train"
        return super().find_class(module, name)
def _save(artifacts: ClassifierArtifacts, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info(f"Model saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")


def load_artifacts(path: str | Path | None = None) -> ClassifierArtifacts:
    cfg = get_settings().classifier
    p   = Path(path or cfg.model_path)
    if not p.exists():
        raise FileNotFoundError(f"No classifier model found at {p}. Run: python -m classifier.train train")
    with open(p, "rb") as f:
        return _SafeUnpickler(f).load()


# ── HybridClassifier (module-level so pickle can serialise it) ────────────────
# Must live here — NOT inside train() — because pickle resolves classes by
# their fully-qualified module path at load time. A class defined inside a
# function becomes 'train.<locals>.HybridClassifier', which is unreachable
# on unpickle and raises AttributeError.

class HybridClassifier(BaseEstimator):
    """
    Fuses TF-IDF sparse matrix with dense handcrafted features,
    then delegates to a calibrated SGDClassifier.

    Kept at module level so pickle can find it as
    'classifier.train.HybridClassifier' when loading saved artifacts.
    """

    def __init__(self, tfidf, hc, clf):
        self.tfidf = tfidf
        self.hc    = hc
        self.clf   = clf

    def fit(self, X, y):
        X_tfidf = self.tfidf.fit_transform(X)
        X_hc    = csr_matrix(self.hc.fit_transform(X))
        X_all   = hstack([X_tfidf, X_hc])
        self.clf.fit(X_all, y)
        return self

    def predict(self, X):
        X_tfidf = self.tfidf.transform(X)
        X_hc    = csr_matrix(self.hc.transform(X))
        X_all   = hstack([X_tfidf, X_hc])
        return self.clf.predict(X_all)

    def predict_proba(self, X):
        X_tfidf = self.tfidf.transform(X)
        X_hc    = csr_matrix(self.hc.transform(X))
        X_all   = hstack([X_tfidf, X_hc])
        return self.clf.predict_proba(X_all)


# ── Training ──────────────────────────────────────────────────────────────────

def train(extra_data_path: str | None = None) -> ClassifierArtifacts:
    """
    Full training pipeline:
      1. Load synthetic + optional user-supplied data
      2. Build features (TF-IDF + handcrafted)
      3. Compute class weights (handles label imbalance automatically)
      4. Train SGDClassifier with log-loss (→ probabilities)
      5. Calibrate via isotonic regression (5-fold CV)
      6. Cross-validate and log macro-F1
      7. Refit on full data and persist
    """
    # 1. Dataset
    dataset = _build_synthetic_dataset()
    if extra_data_path:
        dataset += _load_csv_dataset(extra_data_path)
    stats = _dataset_stats(dataset)
    logger.info(f"Dataset: {stats}")

    queries = [d.query for d in dataset]
    labels  = [d.route for d in dataset]

    # Method hints — used at inference to populate vectorless_method
    # Map: route_label_str → most common method in training data
    from collections import Counter
    method_votes: dict[str, Counter] = {name: Counter() for name in LABEL_NAMES}
    for d in dataset:
        if d.method:
            method_votes[d.route][d.method] += 1
    method_map: dict[int, str | None] = {}

    le = LabelEncoder()
    le.fit(LABEL_NAMES)
    y = le.transform(labels)

    for idx, cls_name in enumerate(le.classes_):
        top = method_votes[cls_name].most_common(1)
        method_map[idx] = top[0][0] if top else None

    # 2. Feature union: TF-IDF (unigrams + bigrams) + handcrafted
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8_000,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b\w+\b",
        min_df=1,
    )
    hc = HandcraftedFeatures()

    # 3. Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = {i: w for i, w in enumerate(cw)}

    # 4. SGD with log-loss (logistic regression via SGD — fast, scalable)
    sgd = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.15,
        max_iter=1000,
        tol=1e-4,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1,
    )

    # 5. Calibration
    calibrated = CalibratedClassifierCV(sgd, cv=5, method="isotonic")

    # HybridClassifier is defined at module level (above train()) so that
    # pickle can serialise it by its fully-qualified path.
    clf = HybridClassifier(tfidf, hc, calibrated)

    # 6. Cross-validate (stratified 5-fold) — report macro-F1
    logger.info("Running 5-fold stratified cross-validation...")
    # For CV we need to re-fit tfidf each fold — use a wrapper that
    # evaluates via predict after manual fit
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(queries, labels)):
        q_train = [queries[i] for i in train_idx]
        y_train = [y[i]       for i in train_idx]
        q_val   = [queries[i] for i in val_idx]
        y_val   = [y[i]       for i in val_idx]

        fold_clf = HybridClassifier(
            TfidfVectorizer(ngram_range=(1, 2), max_features=8_000, sublinear_tf=True,
                            strip_accents="unicode", min_df=1, token_pattern=r"\b\w+\b"),
            HandcraftedFeatures(),
            CalibratedClassifierCV(
                SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4,
                              l1_ratio=0.15, max_iter=1000, random_state=42,
                              class_weight=class_weight_dict, n_jobs=-1),
                cv=3, method="isotonic",
            ),
        )
        fold_clf.fit(q_train, np.array(y_train))
        preds = fold_clf.predict(q_val)
        score = f1_score(y_val, preds, average="macro")
        cv_scores.append(score)
        logger.info(f"  Fold {fold + 1}/5 → macro-F1: {score:.4f}")

    logger.info(f"Cross-val macro-F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 7. Final fit on full dataset
    logger.info("Fitting final model on full dataset...")
    t0 = time.perf_counter()
    clf.fit(queries, y)
    elapsed = time.perf_counter() - t0
    logger.info(f"Training complete in {elapsed:.2f}s")

    artifacts = ClassifierArtifacts(
        pipeline=clf,
        handcrafted=hc,
        label_encoder=le,
        method_map=method_map,
    )
    cfg = get_settings().classifier
    _save(artifacts, Path(cfg.model_path))
    return artifacts


# ── Evaluation

def evaluate(test_data_path: str, model_path: str | None = None) -> dict:
    """
    Evaluate saved model on a held-out CSV test set.
    Prints per-class report + confusion matrix.
    Returns dict with macro-F1 and per-class scores.
    """
    arts = load_artifacts(model_path)
    dataset = _load_csv_dataset(test_data_path)
    if not dataset:
        logger.error("Test dataset is empty or malformed.")
        return {}

    queries = [d.query  for d in dataset]
    labels  = [d.route  for d in dataset]
    y_true  = arts.label_encoder.transform(labels)

    preds        = arts.pipeline.predict(queries)
    pred_labels  = arts.label_encoder.inverse_transform(preds)

    macro_f1 = f1_score(y_true, preds, average="macro")
    report   = classification_report(labels, pred_labels, target_names=LABEL_NAMES)
    cm       = confusion_matrix(y_true, preds)

    print("\n" + "─" * 60)
    print("CLASSIFICATION REPORT")
    print("─" * 60)
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {LABEL_NAMES}")
    print(cm)
    print(f"\nOverall macro-F1: {macro_f1:.4f}")
    print("─" * 60 + "\n")

    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm.tolist()}


# ── Inference 

def predict_intent(
    query: str,
    artifacts: ClassifierArtifacts | None = None,
    use_rules_first: bool = True,
) -> QueryIntent:
    """
    Main inference entry point used by hybrid/fusion.py.

    Pipeline:
      1. Rule-based classifier (fast, high-confidence patterns)
      2. ML classifier          (fires when rules return None or low confidence)
      3. Fallback               (HYBRID) if ML confidence < threshold
    """
    cfg = get_settings().classifier

    # Step 1: rules
    if use_rules_first and cfg.mode in ("rules_first", "hybrid"):
        rule_intent = rules_classify(query)
        if rule_intent and rule_intent.confidence >= cfg.confidence_threshold:
            logger.debug(f"Rule-based route: {rule_intent.route} ({rule_intent.confidence:.2f})")
            return rule_intent

    # Step 2: ML model
    if artifacts is None:
        try:
            artifacts = load_artifacts()
        except FileNotFoundError:
            logger.warning("No ML model found. Falling back to HYBRID route.")
            return QueryIntent(
                route=RouteType.HYBRID,
                confidence=0.5,
                reasoning="No ML model loaded — defaulting to hybrid retrieval",
            )

    proba  = artifacts.pipeline.predict_proba([query])[0]
    top_idx   = int(np.argmax(proba))
    top_prob  = float(proba[top_idx])
    top_label = artifacts.label_encoder.inverse_transform([top_idx])[0]

    # Step 3: confidence gate
    if top_prob < cfg.confidence_threshold:
        fallback_route = RouteType(cfg.fallback)
        logger.debug(f"Low confidence ({top_prob:.2f}) — falling back to {fallback_route}")
        return QueryIntent(
            route=fallback_route,
            confidence=top_prob,
            reasoning=f"ML confidence {top_prob:.2f} below threshold {cfg.confidence_threshold}",
        )

    route  = RouteType(top_label)
    method = None
    if route == RouteType.VECTORLESS:
        method_str = artifacts.method_map.get(top_idx)
        method = VectorlessMethod(method_str) if method_str else None

    return QueryIntent(
        route=route,
        vectorless_method=method,
        confidence=top_prob,
        reasoning=f"ML classifier: {top_label} (p={top_prob:.2f})",
    )


# ── Active learning

def active_learn(
    unlabelled_path: str,
    output_csv: str = "data/active_labels.csv",
    n_samples: int = 20,
    model_path: str | None = None,
):
    """
    Uncertainty sampling active-learning loop.
    Picks the queries the model is LEAST confident about and asks
    the user to label them interactively. Saves to CSV for re-training.

    Uncertainty = 1 - max(class_probabilities)
    """
    arts = load_artifacts(model_path)
    queries = Path(unlabelled_path).read_text(encoding="utf-8").splitlines()
    queries = [q.strip() for q in queries if q.strip()]

    if not queries:
        logger.error("Unlabelled pool is empty.")
        return

    proba = arts.pipeline.predict_proba(queries)
    uncertainty = 1.0 - proba.max(axis=1)  # higher → less certain
    ranked_idx  = np.argsort(uncertainty)[::-1][:n_samples]

    logger.info(f"Selected {len(ranked_idx)} most uncertain queries for labelling.")
    labelled = []

    print(f"\n{'─'*60}")
    print(f"ACTIVE LEARNING  — label {len(ranked_idx)} queries")
    print(f"Options: vector | vectorless | hybrid | skip")
    print(f"{'─'*60}\n")

    for rank, idx in enumerate(ranked_idx, 1):
        q     = queries[idx]
        u     = uncertainty[idx]
        preds = arts.label_encoder.inverse_transform([np.argmax(proba[idx])])[0]
        print(f"[{rank}/{len(ranked_idx)}] Uncertainty: {u:.3f}  |  Model guess: {preds}")
        print(f"  Query: {q}")
        label = input("  Your label: ").strip().lower()
        if label == "skip" or label == "":
            print()
            continue
        if label not in LABEL_NAMES:
            print(f"  ✗ Invalid label '{label}' — skipping.")
            continue
        method = ""
        if label == "vectorless":
            method = input("  Method (sql/bm25/graph): ").strip().lower()
        labelled.append({"query": q, "label": label, "method": method or ""})
        print(f"  ✓ Saved as '{label}'\n")

    if labelled:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query", "label", "method"]
        write_header = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(labelled)
        logger.info(f"Saved {len(labelled)} new labels → {out}")
        logger.info("Re-train with: python -m classifier.train train --data " + str(out))
    else:
        logger.info("No new labels added.")


# ── CLI 

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classifier.train",
        description="Hybrid-RAG query route classifier — train / evaluate / predict / active-learn",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train classifier (synthetic + optional CSV)")
    p_train.add_argument("--data", type=str, default=None,
                         help="Path to extra labelled CSV (query,label[,method])")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate saved model on test CSV")
    p_eval.add_argument("--data", type=str, required=True,
                        help="Test CSV path (query,label[,method])")
    p_eval.add_argument("--model", type=str, default=None, help="Model pickle path")

    # predict
    p_pred = sub.add_parser("predict", help="Predict route for a single query")
    p_pred.add_argument("query", type=str, help="Query string (quote it)")
    p_pred.add_argument("--model", type=str, default=None)
    p_pred.add_argument("--no-rules", action="store_true",
                        help="Skip rule-based pre-classifier")

    # active-learn
    p_al = sub.add_parser("active-learn", help="Interactive uncertainty-sampling loop")
    p_al.add_argument("--pool",    type=str, required=True, help="Unlabelled queries (.txt, one per line)")
    p_al.add_argument("--output",  type=str, default="data/active_labels.csv")
    p_al.add_argument("--n",       type=int, default=20, help="Samples to label")
    p_al.add_argument("--model",   type=str, default=None)

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        arts = train(extra_data_path=args.data)
        logger.info("Training complete. Run 'predict' to test the model.")

    elif args.command == "evaluate":
        evaluate(args.data, model_path=args.model)

    elif args.command == "predict":
        arts = None
        if args.model:
            arts = load_artifacts(args.model)
        intent = predict_intent(
            args.query,
            artifacts=arts,
            use_rules_first=not args.no_rules,
        )
        result = {
            "query":             args.query,
            "route":             intent.route.value,
            "vectorless_method": intent.vectorless_method.value if intent.vectorless_method else None,
            "confidence":        round(intent.confidence, 4),
            "reasoning":         intent.reasoning,
        }
        print(json.dumps(result, indent=2))

    elif args.command == "active-learn":
        active_learn(
            unlabelled_path=args.pool,
            output_csv=args.output,
            n_samples=args.n,
            model_path=args.model,
        )


if __name__ == "__main__":
    main()