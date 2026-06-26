"""
tests/test_classifier.py
────────────────────────────────────────────────────────────────────────────────
pytest suite for the Hybrid-RAG query classifier.

Coverage
────────
    rules.py
        TestRulesStructuredID     — ID patterns fire SQL route
        TestRulesExactLookup      — exact-lookup phrases fire SQL route
        TestRulesKeywordSearch    — keyword/phrase patterns fire BM25 route
        TestRulesEntityRelation   — who/authored-by patterns fire GRAPH route
        TestRulesTemporal         — change-over-time patterns fire HYBRID route
        TestRulesSemantic         — analytical patterns fire VECTOR route
        TestRulesNoFire           — ambiguous queries return None
        TestRulesConfidence       — confidence values are within spec
        TestRulesReturnType       — return type is QueryIntent or None

    train.py
        TestHandcraftedFeatures   — 12 binary/numeric feature dimensions
        TestHybridClassifier      — fit/predict/predict_proba contract
        TestTrainPipeline         — full train() produces valid artifacts
        TestPredictIntentRulesFirst — rules short-circuit ML when confident
        TestPredictIntentMLOnly   — ML path fires when rules return None
        TestPredictIntentFallback — low-confidence → HYBRID fallback
        TestPredictIntentNoModel  — missing model → HYBRID fallback
        TestActiveLearnSampling   — uncertainty sampling picks uncertain rows
        TestLoadArtifacts         — save/load round-trip is lossless
        TestEvaluatePipeline      — evaluate() on labelled CSV computes F1

Run
───
    pytest tests/test_classifier.py -v
    pytest tests/test_classifier.py -v -k "TestRules"       # only rules
    pytest tests/test_classifier.py -v -k "TestTrain"       # only ML
    pytest tests/test_classifier.py --tb=short --no-header
"""
from __future__ import annotations

import csv
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Project root 
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import QueryIntent, RouteType, VectorlessMethod
from classifier.rules import classify as rules_classify
from classifier.train import (
    HandcraftedFeatures,
    HybridClassifier,
    LabelledQuery,
    _build_synthetic_dataset,
    _dataset_stats,
    _load_csv_dataset,
    predict_intent,
    train,
    evaluate,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def trained_artifacts():
    """
    Train the full classifier once per module.
    Reused across all ML tests — expensive to rebuild per test.
    """
    return train()


@pytest.fixture
def tmp_csv(tmp_path):
    """Write a minimal labelled CSV and return its path."""
    def _make(rows: list[dict]) -> Path:
        p = tmp_path / "test_data.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query", "label", "method"])
            writer.writeheader()
            writer.writerows(rows)
        return p
    return _make


@pytest.fixture
def sample_chunks():
    """Representative queries for each route used in multiple test classes."""
    return {
        RouteType.VECTOR: [
            "summarise the key risk factors in the filing",
            "explain the indemnification clause in plain English",
            "what are the main findings of the annual report",
            "describe the data retention policy",
            "analyse the competitive landscape section",
        ],
        RouteType.VECTORLESS: [
            "what is the total on invoice INV-2041",
            "get the status of contract CT-0034",
            "find all documents mentioning force majeure",
            "who approved the Q3 budget document",
            "how many invoices are overdue as of today",
        ],
        RouteType.HYBRID: [
            "how has our refund policy changed since 2022",
            "compare the SLA terms in the 2021 vs 2024 contracts",
            "which risk factors are new compared to last year report",
            "how have the termination clauses changed across renewals",
            "show the trend in contract values over the past 5 years",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Rules tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRulesReturnType:
    """classify() returns QueryIntent | None with correct types."""

    def test_returns_query_intent_or_none(self):
        result = rules_classify("INV-2041 total amount")
        assert result is None or isinstance(result, QueryIntent)

    def test_query_intent_fields_valid(self):
        result = rules_classify("who approved the Q3 budget")
        if result is not None:
            assert isinstance(result.route, RouteType)
            assert isinstance(result.confidence, float)
            assert isinstance(result.reasoning, str)
            assert 0.0 <= result.confidence <= 1.0

    def test_none_for_ambiguous(self):
        # Short, completely ambiguous queries should return None
        assert rules_classify("help") is None
        assert rules_classify("document") is None


class TestRulesStructuredID:
    """Structured IDs (INV-xxx, PO-xxx …) route to VECTORLESS/SQL."""

    @pytest.mark.parametrize("query", [
        "what is the total amount on invoice INV-2041",
        "show me PO-8812 delivery date",
        "get the status of contract CT-0034",
        "fetch document ID DOC-1193 expiry date",
        "what is the value of order ORD-9921",
        "when was ticket TKT-5502 created",
        "find record by ref number REF-0072",
        "get approval date for PO-1102",
    ])
    def test_routes_to_vectorless(self, query):
        result = rules_classify(query)
        assert result is not None, f"Rule did not fire for: {query!r}"
        assert result.route == RouteType.VECTORLESS

    @pytest.mark.parametrize("query", [
        "what is the total amount on invoice INV-2041",
        "get the status of contract CT-0034",
        "look up project code PR-3309 budget",
    ])
    def test_method_is_sql(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.vectorless_method == VectorlessMethod.SQL

    def test_confidence_high_for_id(self):
        result = rules_classify("status of PO-8812")
        assert result is not None
        assert result.confidence >= 0.90

    @pytest.mark.parametrize("id_str", [
        "INV-001", "PO-8812", "CT-0034", "DOC-1193",
        "ORD-9921", "TKT-5502", "REF-0072", "AGR-0441",
    ])
    def test_id_pattern_coverage(self, id_str):
        result = rules_classify(f"what is the status of {id_str}")
        assert result is not None
        assert result.route == RouteType.VECTORLESS


class TestRulesExactLookup:
    """Exact-lookup phrases ('what is the total', 'get by id') → VECTORLESS/SQL."""

    @pytest.mark.parametrize("query", [
        "what is the total amount for this invoice",
        "what is the date of submission",
        "what is the count of active contracts",
        "show me the exact status of this document",
        "get this record by reference number",
    ])
    def test_routes_to_vectorless(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.route == RouteType.VECTORLESS

    def test_method_is_sql(self):
        result = rules_classify("what is the exact due date for this contract")
        assert result is not None
        assert result.vectorless_method == VectorlessMethod.SQL


class TestRulesKeywordSearch:
    """Keyword/phrase search patterns → VECTORLESS/BM25."""

    @pytest.mark.parametrize("query", [
        "find all documents mentioning force majeure",
        "search for contracts containing the word indemnification",
        "find all files that mention GDPR",
        "which documents contain the phrase data breach",
        "search for every record with the term arbitration",
        "find all contracts mentioning termination for cause",
        "which files include the keyword confidentiality",
    ])
    def test_routes_to_vectorless(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.route == RouteType.VECTORLESS

    @pytest.mark.parametrize("query", [
        "find all documents mentioning force majeure",
        "search for contracts containing indemnification",
    ])
    def test_method_is_bm25(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.vectorless_method == VectorlessMethod.BM25


class TestRulesEntityRelation:
    """Who/authored-by/signed-by patterns → VECTORLESS/GRAPH."""

    @pytest.mark.parametrize("query", [
        "who approved the Q3 budget document",
        "who authored the data governance policy",
        "who signed the vendor agreement with TechCorp",
        "who is the owner of project Alpha documentation",
        "who created the compliance checklist",
        "who submitted the incident report IR-0042",
        "document authored by the legal team",
        "contract signed by the CFO",
    ])
    def test_routes_to_vectorless(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.route == RouteType.VECTORLESS

    def test_method_is_graph(self):
        result = rules_classify("who approved the Q3 budget document")
        assert result is not None
        assert result.vectorless_method == VectorlessMethod.GRAPH


class TestRulesTemporal:
    """Change-over-time patterns → HYBRID."""

    @pytest.mark.parametrize("query", [
        "how has our refund policy changed since 2022",
        "how have the termination clauses changed across renewals",
        "what changed in the privacy policy between v1 and v2",
        "show the trend in contract values over the past 5 years",
        "compare the SLA terms between 2020 and 2024",
        "difference since last year in our compliance requirements",
    ])
    def test_routes_to_hybrid(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.route == RouteType.HYBRID


class TestRulesSemantic:
    """Analytical/semantic patterns → VECTOR."""

    @pytest.mark.parametrize("query", [
        "summarise the key findings in the annual report",
        "explain the product liability clause in plain English",
        "analyse the competitive landscape section",
        "describe the data retention policy",
        "what are the implications of the new data privacy law",
        "give me an overview of the compliance requirements",
    ])
    def test_routes_to_vector(self, query):
        result = rules_classify(query)
        assert result is not None
        assert result.route == RouteType.VECTOR


class TestRulesNoFire:
    """Queries with no strong signal return None."""

    @pytest.mark.parametrize("query", [
        "document",
        "hello",
        "what",
        "",
        "   ",
    ])
    def test_returns_none(self, query):
        result = rules_classify(query)
        assert result is None


class TestRulesConfidence:
    """Confidence values are in range and differentiated by pattern strength."""

    def test_id_confidence_highest(self):
        id_result   = rules_classify("status of INV-2041")
        sem_result  = rules_classify("summarise this document")
        if id_result and sem_result:
            assert id_result.confidence >= sem_result.confidence

    def test_confidence_in_range(self):
        for query in [
            "INV-2041 total", "find all documents mentioning GDPR",
            "who approved this", "summarise the report",
            "how has this changed since 2022",
        ]:
            result = rules_classify(query)
            if result is not None:
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Confidence {result.confidence} out of range for: {query!r}"
                )

    def test_reasoning_nonempty_when_fires(self):
        result = rules_classify("who approved the Q3 budget")
        if result is not None:
            assert len(result.reasoning) > 0


# ══════════════════════════════════════════════════════════════════════════════
# HandcraftedFeatures tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHandcraftedFeatures:
    """Verify each of the 12 feature dimensions fires correctly."""

    @pytest.fixture(autouse=True)
    def hc(self):
        self.hc = HandcraftedFeatures()
        self.hc.fit([])   # stateless — fit is a no-op

    def _feat(self, query: str) -> np.ndarray:
        return self.hc.transform([query])[0]

    def test_output_shape(self):
        out = self.hc.transform(["a", "b", "c"])
        assert out.shape == (3, 12)

    def test_output_dtype(self):
        out = self.hc.transform(["test query"])
        assert out.dtype == np.float32

    def test_feat0_structured_id(self):
        assert self._feat("status of INV-2041")[0] == 1.0
        assert self._feat("summarise this document")[0] == 0.0

    def test_feat1_date_ref(self):
        assert self._feat("contracts in Q1 2025")[1] == 1.0
        assert self._feat("explain the clause")[1] == 0.0

    def test_feat2_aggregation(self):
        assert self._feat("how many invoices are overdue")[2] == 1.0
        assert self._feat("explain the policy")[2] == 0.0

    def test_feat3_comparison(self):
        assert self._feat("compare the 2021 vs 2024 contracts")[3] == 1.0
        assert self._feat("what is the status")[3] == 0.0

    def test_feat4_entity_relation(self):
        assert self._feat("who approved the Q3 budget")[4] == 1.0
        assert self._feat("list all contracts")[4] == 0.0

    def test_feat5_keyword_signal(self):
        assert self._feat("find all documents containing force majeure")[5] == 1.0
        assert self._feat("describe the policy")[5] == 0.0

    def test_feat6_semantic_signal(self):
        assert self._feat("summarize the key findings")[6] == 1.0
        assert self._feat("get invoice INV-001 total")[6] == 0.0

    def test_feat7_temporal(self):
        assert self._feat("how has the policy changed since 2022")[7] == 1.0
        assert self._feat("get invoice total")[7] == 0.0

    def test_feat8_negation(self):
        assert self._feat("find contracts not containing arbitration")[8] == 1.0
        assert self._feat("find all contracts")[8] == 0.0

    def test_feat9_plural(self):
        assert self._feat("list all active contracts")[9] == 1.0
        assert self._feat("what is the total on invoice")[9] == 0.0

    def test_feat10_length_norm_capped(self):
        long_q  = " ".join(["word"] * 25)
        short_q = "hi"
        assert self._feat(long_q)[10]  == 1.0
        assert self._feat(short_q)[10] < 1.0

    def test_feat11_starts_with_who(self):
        assert self._feat("who signed the contract")[11] == 1.0
        assert self._feat("what is the total")[11] == 0.0

    def test_batch_consistency(self):
        queries = [
            "INV-2041 status",
            "summarise the report",
            "who approved this",
        ]
        batch_out = self.hc.transform(queries)
        for i, q in enumerate(queries):
            single_out = self.hc.transform([q])[0]
            np.testing.assert_array_equal(batch_out[i], single_out)

    def test_empty_string(self):
        # Should not crash on empty input
        out = self.hc.transform([""])
        assert out.shape == (1, 12)

    def test_all_zeros_for_neutral_query(self):
        # A query with no special signals should have mostly zeros
        out = self.hc.transform(["abc"])
        assert out.sum() >= 0   # just shouldn't crash or have NaN
        assert not np.any(np.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# HybridClassifier tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridClassifier:
    """HybridClassifier fit/predict/predict_proba contract."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, min_df=1)
        hc    = HandcraftedFeatures()
        sgd   = SGDClassifier(loss="log_loss", max_iter=200, random_state=42)
        cal   = CalibratedClassifierCV(sgd, cv=3, method="isotonic")
        self.clf = HybridClassifier(tfidf, hc, cal)

        self.X = [
            "summarise the annual report",
            "what is the total on invoice INV-001",
            "how has the policy changed since 2022",
            "explain the liability clause",
            "get status of contract CT-001",
            "compare 2021 vs 2024 terms",
        ]
        self.y = np.array([0, 1, 2, 0, 1, 2])  # vector=0, vectorless=1, hybrid=2

    def test_fit_returns_self(self):
        result = self.clf.fit(self.X, self.y)
        assert result is self.clf

    def test_predict_shape(self):
        self.clf.fit(self.X, self.y)
        preds = self.clf.predict(self.X)
        assert preds.shape == (len(self.X),)

    def test_predict_valid_labels(self):
        self.clf.fit(self.X, self.y)
        preds = self.clf.predict(self.X)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_shape(self):
        self.clf.fit(self.X, self.y)
        proba = self.clf.predict_proba(self.X)
        assert proba.shape == (len(self.X), 3)

    def test_predict_proba_sums_to_one(self):
        self.clf.fit(self.X, self.y)
        proba = self.clf.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_in_range(self):
        self.clf.fit(self.X, self.y)
        proba = self.clf.predict_proba(self.X)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_picklable(self, tmp_path):
        """HybridClassifier must be picklable at module level."""
        self.clf.fit(self.X, self.y)
        pkl_path = tmp_path / "clf.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.clf, f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        preds_orig   = self.clf.predict(self.X)
        preds_loaded = loaded.predict(self.X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_predict_single_sample(self):
        self.clf.fit(self.X, self.y)
        preds = self.clf.predict(["explain the SLA clause"])
        assert preds.shape == (1,)


# ══════════════════════════════════════════════════════════════════════════════
# Full training pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainPipeline:
    """train() produces valid ClassifierArtifacts."""

    def test_artifacts_not_none(self, trained_artifacts):
        assert trained_artifacts is not None

    def test_pipeline_attribute(self, trained_artifacts):
        assert hasattr(trained_artifacts, "pipeline")
        assert trained_artifacts.pipeline is not None

    def test_label_encoder_classes(self, trained_artifacts):
        le = trained_artifacts.label_encoder
        assert set(le.classes_) == {"vector", "vectorless", "hybrid"}

    def test_method_map_keys_valid(self, trained_artifacts):
        mm = trained_artifacts.method_map
        assert isinstance(mm, dict)
        for v in mm.values():
            assert v in (None, "sql", "bm25", "graph")

    def test_predict_on_new_query(self, trained_artifacts):
        preds = trained_artifacts.pipeline.predict(["summarise the annual report"])
        le    = trained_artifacts.label_encoder
        label = le.inverse_transform(preds)[0]
        assert label in {"vector", "vectorless", "hybrid"}

    def test_predict_proba_valid(self, trained_artifacts):
        proba = trained_artifacts.pipeline.predict_proba(
            ["what is the total on invoice INV-001"]
        )
        assert proba.shape == (1, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_train_with_extra_csv(self, tmp_csv, tmp_path):
        extra = tmp_csv([
            {"query": "list all expired licences",       "label": "vectorless", "method": "sql"},
            {"query": "explain the arbitration process", "label": "vector",     "method": ""},
            {"query": "compare v1 vs v2 of the policy",  "label": "hybrid",     "method": ""},
        ])
        # Redirect model save to tmp_path so we don't overwrite production model
        with patch("classifier.train.get_settings") as mock_cfg:
            mock_cfg.return_value.classifier.model_path = str(tmp_path / "model.pkl")
            mock_cfg.return_value.classifier.confidence_threshold = 0.75
            mock_cfg.return_value.classifier.fallback = "hybrid"
            mock_cfg.return_value.classifier.mode = "rules_first"
            arts = train(extra_data_path=str(extra))
        assert arts is not None

    def test_save_load_roundtrip(self, trained_artifacts, tmp_path):
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(trained_artifacts, f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        q = ["explain the warranty clause"]
        np.testing.assert_array_equal(
            trained_artifacts.pipeline.predict(q),
            loaded.pipeline.predict(q),
        )


# ══════════════════════════════════════════════════════════════════════════════
# predict_intent
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictIntentRulesFirst:
    """Rules short-circuit ML when confidence >= threshold."""

    def test_structured_id_returns_vectorless(self):
        intent = predict_intent("status of INV-2041", use_rules_first=True)
        assert intent.route == RouteType.VECTORLESS

    def test_who_query_returns_vectorless(self):
        intent = predict_intent("who approved the Q3 budget", use_rules_first=True)
        assert intent.route == RouteType.VECTORLESS

    def test_temporal_returns_hybrid(self):
        intent = predict_intent(
            "how has the policy changed since 2022", use_rules_first=True
        )
        assert intent.route == RouteType.HYBRID

    def test_semantic_returns_vector(self):
        intent = predict_intent(
            "summarise the key findings in the report", use_rules_first=True
        )
        assert intent.route == RouteType.VECTOR

    def test_rules_bypass_model(self, trained_artifacts):
        """When a rule fires with high confidence, ML is never called."""
        with patch.object(trained_artifacts.pipeline, "predict_proba") as mock_pp:
            predict_intent("status of INV-2041", artifacts=trained_artifacts,
                           use_rules_first=True)
            mock_pp.assert_not_called()


class TestPredictIntentMLOnly:
    """ML path fires correctly when rules return None."""

    def test_returns_query_intent(self, trained_artifacts):
        intent = predict_intent(
            "what obligations does the vendor have",
            artifacts=trained_artifacts,
            use_rules_first=False,
        )
        assert isinstance(intent, QueryIntent)
        assert intent.route in RouteType.__members__.values()

    def test_confidence_in_range(self, trained_artifacts):
        intent = predict_intent(
            "explain the scope of the NDA",
            artifacts=trained_artifacts,
            use_rules_first=False,
        )
        assert 0.0 <= intent.confidence <= 1.0

    def test_vectorless_has_method(self, trained_artifacts):
        intent = predict_intent(
            "what is the total on invoice INV-9999",
            artifacts=trained_artifacts,
            use_rules_first=False,
        )
        if intent.route == RouteType.VECTORLESS:
            assert intent.vectorless_method in VectorlessMethod.__members__.values()

    @pytest.mark.parametrize("query,expected_route", [
        ("summarise the annual report",               RouteType.VECTOR),
        ("what is the total on invoice INV-2041",     RouteType.VECTORLESS),
        ("find all documents mentioning force majeure",RouteType.VECTORLESS),
        ("who approved the Q3 budget document",       RouteType.VECTORLESS),
    ])
    def test_high_confidence_queries_correct(
        self, trained_artifacts, query, expected_route
    ):
        intent = predict_intent(query, artifacts=trained_artifacts,
                                use_rules_first=False)
        # High-confidence samples should be correct with a trained model
        assert intent.route == expected_route, (
            f"Expected {expected_route.value} for {query!r}, "
            f"got {intent.route.value} (conf={intent.confidence:.2f})"
        )


class TestPredictIntentFallback:
    """Low-confidence predictions fall back to HYBRID."""

    def test_low_confidence_falls_back(self, trained_artifacts):
        """Force low probability by patching predict_proba."""
        proba = np.array([[0.38, 0.33, 0.29]])  # max < 0.75 threshold

        with patch.object(trained_artifacts.pipeline, "predict_proba",
                          return_value=proba):
            intent = predict_intent(
                "some ambiguous query xyz",
                artifacts=trained_artifacts,
                use_rules_first=False,
            )
        assert intent.route == RouteType.HYBRID

    def test_fallback_confidence_preserved(self, trained_artifacts):
        proba = np.array([[0.40, 0.35, 0.25]])
        with patch.object(trained_artifacts.pipeline, "predict_proba",
                          return_value=proba):
            intent = predict_intent(
                "ambiguous query",
                artifacts=trained_artifacts,
                use_rules_first=False,
            )
        assert intent.confidence == pytest.approx(0.40, abs=0.01)


class TestPredictIntentNoModel:
    """Missing model file returns HYBRID fallback gracefully."""

    def test_missing_model_returns_hybrid(self, tmp_path):
        with patch("classifier.train.get_settings") as mock_cfg:
            mock_cfg.return_value.classifier.model_path = str(
                tmp_path / "nonexistent_model.pkl"
            )
            mock_cfg.return_value.classifier.confidence_threshold = 0.75
            mock_cfg.return_value.classifier.fallback = "hybrid"
            mock_cfg.return_value.classifier.mode = "rules_first"
            intent = predict_intent(
                "some query that needs a model",
                artifacts=None,
                use_rules_first=False,
            )
        assert intent.route == RouteType.HYBRID
        assert intent.confidence <= 0.75


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetHelpers:
    """_build_synthetic_dataset, _load_csv_dataset, _dataset_stats."""

    def test_synthetic_dataset_nonempty(self):
        ds = _build_synthetic_dataset()
        assert len(ds) > 0

    def test_synthetic_dataset_all_valid_labels(self):
        ds = _build_synthetic_dataset()
        for row in ds:
            assert row.route in {"vector", "vectorless", "hybrid"}

    def test_synthetic_dataset_all_three_classes(self):
        ds = _build_synthetic_dataset()
        labels = {row.route for row in ds}
        assert labels == {"vector", "vectorless", "hybrid"}

    def test_load_csv_valid(self, tmp_csv):
        rows = [
            {"query": "explain the NDA clause", "label": "vector",     "method": ""},
            {"query": "status of INV-001",       "label": "vectorless", "method": "sql"},
            {"query": "compare 2022 vs 2024",    "label": "hybrid",     "method": ""},
        ]
        path = tmp_csv(rows)
        ds = _load_csv_dataset(str(path))
        assert len(ds) == 3

    def test_load_csv_skips_invalid_labels(self, tmp_csv):
        rows = [
            {"query": "valid query",   "label": "vector",  "method": ""},
            {"query": "invalid query", "label": "unknown", "method": ""},
        ]
        path = tmp_csv(rows)
        ds = _load_csv_dataset(str(path))
        assert len(ds) == 1
        assert ds[0].route == "vector"

    def test_load_csv_missing_label_column_raises(self, tmp_path):
        p = tmp_path / "no_label.csv"
        p.write_text("query\nsome query\n")
        with pytest.raises(ValueError, match="label"):
            _load_csv_dataset(str(p))

    def test_dataset_stats_counts(self):
        ds = [
            LabelledQuery("a", "vector",     None),
            LabelledQuery("b", "vector",     None),
            LabelledQuery("c", "vectorless", "sql"),
            LabelledQuery("d", "hybrid",     None),
        ]
        stats = _dataset_stats(ds)
        assert stats["total"] == 4
        assert stats["by_class"]["vector"] == 2
        assert stats["by_class"]["vectorless"] == 1
        assert stats["by_class"]["hybrid"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluatePipeline:
    """evaluate() computes valid F1 on a labelled CSV."""

    def test_evaluate_returns_dict(self, trained_artifacts, tmp_csv, tmp_path):
        rows = [
            {"query": "summarise the annual report",            "label": "vector",     "method": ""},
            {"query": "what is the total on invoice INV-2041",  "label": "vectorless", "method": "sql"},
            {"query": "how has the policy changed since 2022",  "label": "hybrid",     "method": ""},
            {"query": "explain the indemnification clause",     "label": "vector",     "method": ""},
            {"query": "get status of contract CT-0034",         "label": "vectorless", "method": "sql"},
        ]
        path = tmp_csv(rows)
        # Point to our already-trained model
        model_path = tmp_path / "eval_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_artifacts, f)

        result = evaluate(str(path), model_path=str(model_path))
        assert isinstance(result, dict)
        assert "macro_f1" in result
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_evaluate_report_string(self, trained_artifacts, tmp_csv, tmp_path):
        rows = [
            {"query": "explain the SLA terms",      "label": "vector",     "method": ""},
            {"query": "status of INV-001",           "label": "vectorless", "method": "sql"},
            {"query": "compare 2021 vs 2024 terms",  "label": "hybrid",     "method": ""},
        ]
        path      = tmp_csv(rows)
        model_path = tmp_path / "m.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_artifacts, f)

        result = evaluate(str(path), model_path=str(model_path))
        assert "report" in result
        assert isinstance(result["report"], str)

    def test_evaluate_empty_csv_returns_empty(self, tmp_csv, tmp_path, trained_artifacts):
        path = tmp_csv([])
        model_path = tmp_path / "m.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_artifacts, f)
        result = evaluate(str(path), model_path=str(model_path))
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# Active learning
# ══════════════════════════════════════════════════════════════════════════════

class TestActiveLearnSampling:
    """Uncertainty sampling selects the least-confident rows."""

    def test_uncertain_samples_selected(self, trained_artifacts, tmp_path):
        """
        Queries with near-uniform class probabilities should rank higher
        than queries where one class probability dominates.
        """
        # Clear-cut queries — model should be very confident
        confident = [
            "summarise the annual report",                  # strong VECTOR signal
            "what is the total on invoice INV-2041",        # strong VECTORLESS signal
        ]
        # Ambiguous query — model should be less confident
        ambiguous = ["xyz abc def ghi jkl mno"]

        queries = ambiguous + confident

        proba = trained_artifacts.pipeline.predict_proba(queries)
        uncertainty = 1.0 - proba.max(axis=1)
        ranked = np.argsort(uncertainty)[::-1]

        # The ambiguous query should rank highest (most uncertain)
        assert ranked[0] == 0, (
            f"Ambiguous query should be most uncertain. "
            f"Uncertainties: {uncertainty}"
        )

    def test_uncertainty_score_range(self, trained_artifacts):
        queries = [
            "summarise the annual report",
            "what is the total on invoice INV-2041",
            "how has the policy changed since 2022",
        ]
        proba = trained_artifacts.pipeline.predict_proba(queries)
        uncertainty = 1.0 - proba.max(axis=1)
        assert (uncertainty >= 0.0).all()
        assert (uncertainty <= 1.0).all()