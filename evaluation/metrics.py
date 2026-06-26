"""
evaluation/metrics.py:
Evaluation suite for Hybrid-RAG Document Intelligence.

Computes four categories of metrics on a labelled test dataset:

    1. RAG quality   — faithfulness, answer relevancy, context precision (RAGAS)
    2. Lexical       — ROUGE-L  (F1 overlap between answer and reference)
    3. Routing       — classifier accuracy per route type
    4. Latency       — p50 / p95 / p99 per route, overall

Architecture:
    EvalSample          — one row of the evaluation dataset (query + expected)
    MetricResult        — scores for a single metric on a single sample
    EvalReport          — aggregated report across all samples

    RougeEvaluator      — self-contained, no external API
    FaithfulnessEvaluator   — LLM-as-judge via GROQ
    AnswerRelevancyEvaluator — LLM-as-judge via GROQ
    ContextPrecisionEvaluator — LLM-as-judge via GROQ
    RoutingEvaluator    — compares predicted vs expected route
    LatencyTracker      — collects and percentiles latency per route

    RAGEvaluator        — orchestrates all the above; main public API

CSV format expected (evaluation dataset):
    Required columns:
        query           — the question
        reference       — ground-truth answer (for ROUGE-L)

    Optional columns:
        expected_route  — vector | vectorless | hybrid  (for routing accuracy)

Public API:
    from evaluation.metrics import RAGEvaluator

    evaluator = RAGEvaluator(pipeline)
    report    = evaluator.run("evaluation/data/eval_set.csv")
    report.print_summary()
    report.save("evaluation/reports/report_2024.json")

CLI
    python -m evaluation.metrics \
        --data evaluation/data/eval_set.csv \
        --output evaluation/reports/report.json \
        --sample 50

Notes on RAGAS metrics:
    We implement a lightweight version of the three core RAGAS metrics
    using GROQ as the judge LLM, rather than importing the ragas library
    directly. This avoids the ragas → langchain → openai dependency chain
    and gives us full control over the prompts.

    Each metric uses a structured 0.0–1.0 JSON score returned by GROQ.
    Faithfulness    : is the answer supported by the retrieved context?
    Answer Relevancy: does the answer address the question?
    Context Precision: does the retrieved context contain what was needed?
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings
from src.models import RAGResponse, RouteType


# ── Data structures 

@dataclass
class EvalSample:
    """One row from the evaluation CSV."""
    query:          str
    reference:      str                 # ground-truth answer
    expected_route: RouteType | None = None  # optional routing ground-truth

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "EvalSample":
        route = None
        raw_route = row.get("expected_route", "").strip().lower()
        if raw_route in {r.value for r in RouteType}:
            route = RouteType(raw_route)
        return cls(
            query=row["query"].strip(),
            reference=row.get("reference", "").strip(),
            expected_route=route,
        )
 

@dataclass
class MetricResult:
    """Score for one metric on one sample."""
    name:       str
    score:      float           # 0.0 – 1.0
    reasoning:  str  = ""
    skipped:    bool = False    # True when metric couldn't be computed


@dataclass
class SampleResult:
    """All metric results for one (query, response) pair."""
    query:          str
    reference:      str
    answer:         str
    route_used:     str
    expected_route: str | None
    latency_ms:     float
    metrics:        list[MetricResult] = field(default_factory=list)

    def score(self, metric_name: str) -> float | None:
        for m in self.metrics:
            if m.name == metric_name and not m.skipped:
                return m.score
        return None


@dataclass
class EvalReport:
    """
    Aggregated evaluation report across all samples.
    Includes per-metric averages, per-route breakdowns, and latency stats.
    """
    total_samples:   int
    metrics_summary: dict[str, float]          # metric_name → mean score
    per_route:       dict[str, dict[str, float]]  # route → {metric → mean}
    routing_accuracy: float | None             # None if no expected_route labels
    latency_stats:   dict[str, Any]            # p50/p95/p99 overall + per route
    sample_results:  list[SampleResult] = field(default_factory=list)
    timestamp:       str = ""

    def print_summary(self):
        """Print a human-readable report to stdout."""
        sep = "─" * 64
        print(f"\n{sep}")
        print("  HYBRID-RAG EVALUATION REPORT")
        print(sep)
        print(f"  Samples evaluated : {self.total_samples}")
        if self.timestamp:
            print(f"  Timestamp         : {self.timestamp}")

        print(f"\n  METRIC AVERAGES")
        print(f"  {'Metric':<28} {'Score':>6}")
        print(f"  {'-'*28} {'-'*6}")
        for name, score in sorted(self.metrics_summary.items()):
            bar = "█" * int(score * 20)
            print(f"  {name:<28} {score:>6.3f}  {bar}")

        if self.routing_accuracy is not None:
            print(f"\n  Routing accuracy  : {self.routing_accuracy:.3f}")

        print(f"\n  PER-ROUTE BREAKDOWN")
        for route, scores in sorted(self.per_route.items()):
            print(f"  [{route}]")
            for mname, mscore in sorted(scores.items()):
                print(f"    {mname:<26} {mscore:.3f}")

        print(f"\n  LATENCY  (ms)")
        lat = self.latency_stats.get("overall", {})
        print(f"  {'p50':>6}  {'p95':>6}  {'p99':>6}  {'mean':>6}")
        print(f"  {lat.get('p50', 0):>6.1f}  {lat.get('p95', 0):>6.1f}  "
              f"{lat.get('p99', 0):>6.1f}  {lat.get('mean', 0):>6.1f}")
        for route, lstats in sorted(self.latency_stats.items()):
            if route == "overall":
                continue
            print(f"  [{route}]  p50={lstats.get('p50',0):.1f}  "
                  f"p95={lstats.get('p95',0):.1f}  mean={lstats.get('mean',0):.1f}")

        print(f"{sep}\n")

    def save(self, path: str | Path):
        """Persist full report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_samples":    self.total_samples,
            "timestamp":        self.timestamp,
            "metrics_summary":  self.metrics_summary,
            "routing_accuracy": self.routing_accuracy,
            "per_route":        self.per_route,
            "latency_stats":    self.latency_stats,
            "samples": [
                {
                    "query":          sr.query,
                    "reference":      sr.reference,
                    "answer":         sr.answer[:500],   # truncate for size
                    "route_used":     sr.route_used,
                    "expected_route": sr.expected_route,
                    "latency_ms":     sr.latency_ms,
                    "metrics":        [asdict(m) for m in sr.metrics],
                }
                for sr in self.sample_results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Report saved → {path}")


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

class RougeEvaluator:
    """
    Self-contained ROUGE-L F1 implementation.
    No external library required — uses LCS (longest common subsequence).

    ROUGE-L measures the longest common subsequence of tokens between
    the generated answer and the reference answer. F1 balances precision
    (how much of the answer is in the reference) and recall (how much of
    the reference is in the answer).
    """

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _lcs_length(a: list[str], b: list[str]) -> int:
        """Dynamic programming LCS length."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        # Space-optimised: only keep two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def score(self, prediction: str, reference: str) -> MetricResult:
        pred_tokens = self._tokenize(prediction)
        ref_tokens  = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return MetricResult(
                name="rouge_l", score=0.0,
                reasoning="Empty prediction or reference", skipped=True,
            )

        lcs = self._lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall    = lcs / len(ref_tokens)  if ref_tokens  else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return MetricResult(
            name="rouge_l",
            score=round(f1, 4),
            reasoning=f"LCS={lcs}, P={precision:.3f}, R={recall:.3f}",
        )


# ── LLM-as-judge base

class _LLMJudge:
    """
    Shared LLM judge infrastructure.
    Sends a structured prompt to Groq and parses a JSON score response.

    All LLM-based metrics inherit from this class.
    """

    _PARSE_INSTRUCTIONS = (
        "\n\nRespond ONLY with a JSON object with exactly two keys:\n"
        '  "score": float between 0.0 and 1.0\n'
        '  "reasoning": one sentence explaining the score\n'
        "No markdown, no extra text."
    )

    def __init__(self):
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(
                api_key=os.getenv("GROQ_API_KEY")
            )
            return self._client
    def _judge(self, prompt: str) -> tuple[float, str]:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + self._PARSE_INSTRUCTIONS,
                    }
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Clean up markdown code blocks if the LLM wraps the JSON
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json", "", 1).strip()
                data = json.loads(raw)
                # Clamp the score between 0.0 and 1.0
                score = max(0.0, min(1.0, float(data["score"])))
                reasoning = data.get("reasoning", "")
                return score, reasoning
        except Exception as e:
            logger.exception("LLM Judge failed")
            return 0.0, str(e)

# ── Faithfulness 

class FaithfulnessEvaluator(_LLMJudge):
    """
    Faithfulness: is every claim in the answer supported by the context?

    Score = fraction of answer statements that are grounded in context.
    1.0 = fully grounded, 0.0 = entirely hallucinated.
    """

    _PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the retrieved context.
A faithful answer only contains information that is explicitly stated or directly implied by the context.

CONTEXT:
{context}

QUESTION:
{query}

GENERATED ANSWER:
{answer}

Task: Score the faithfulness of the answer on a scale from 0.0 to 1.0.
- 1.0 = every claim in the answer is fully supported by the context
- 0.5 = some claims are supported, some are not or go beyond the context
- 0.0 = the answer contradicts or completely ignores the context"""

    def score(
        self,
        query:   str,
        answer:  str,
        context: str,
    ) -> MetricResult:
        if not answer.strip() or not context.strip():
            return MetricResult(
                name="faithfulness", score=0.0,
                reasoning="Empty answer or context", skipped=True,
            )
        prompt = self._PROMPT.format(query=query, answer=answer, context=context)
        sc, reasoning = self._judge(prompt)
        return MetricResult(name="faithfulness", score=sc, reasoning=reasoning)


# ── Answer relevancy 

class AnswerRelevancyEvaluator(_LLMJudge):
    """
    Answer relevancy: does the answer actually address the question?

    Score = degree to which the answer is on-topic and useful.
    1.0 = fully addresses the question, 0.0 = completely off-topic.
    """

    _PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the question asked.

QUESTION:
{query}

GENERATED ANSWER:
{answer}

Task: Score how well the answer addresses the question on a scale from 0.0 to 1.0.
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer is partially relevant but misses key aspects
- 0.0 = the answer does not address the question at all"""

    def score(self, query: str, answer: str) -> MetricResult:
        if not answer.strip():
            return MetricResult(
                name="answer_relevancy", score=0.0,
                reasoning="Empty answer", skipped=True,
            )
        prompt = self._PROMPT.format(query=query, answer=answer)
        sc, reasoning = self._judge(prompt)
        return MetricResult(name="answer_relevancy", score=sc, reasoning=reasoning)


# ── Context precision

class ContextPrecisionEvaluator(_LLMJudge):
    """
    Context precision: does the retrieved context contain the information
    needed to answer the question?

    Score = fraction of retrieved chunks that are actually relevant.
    1.0 = all retrieved context is useful, 0.0 = none of it is relevant.
    """

    _PROMPT = """\
You are evaluating whether the retrieved context is relevant and useful for answering a question.

QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

REFERENCE ANSWER (what the correct answer should contain):
{reference}

Task: Score how precisely the retrieved context supports answering the question on a scale from 0.0 to 1.0.
- 1.0 = the context contains exactly the information needed to answer correctly
- 0.5 = the context is partially relevant but contains significant noise or gaps
- 0.0 = the context is irrelevant or does not help answer the question"""

    def score(
        self,
        query:     str,
        context:   str,
        reference: str,
    ) -> MetricResult:
        if not context.strip():
            return MetricResult(
                name="context_precision", score=0.0,
                reasoning="Empty context", skipped=True,
            )
        prompt = self._PROMPT.format(
            query=query, context=context[:3000], reference=reference
        )
        sc, reasoning = self._judge(prompt)
        return MetricResult(name="context_precision", score=sc, reasoning=reasoning)


# ── Routing evaluator 
class RoutingEvaluator:
    """
    Compares the pipeline's chosen route against expected_route labels
    in the evaluation CSV.

    Returns per-class precision/recall and overall accuracy.
    """

    def score(
        self,
        predicted: RouteType,
        expected:  RouteType,
    ) -> MetricResult:
        correct = predicted == expected
        return MetricResult(
            name="routing_correct",
            score=1.0 if correct else 0.0,
            reasoning=f"predicted={predicted.value}, expected={expected.value}",
        )

    @staticmethod
    def aggregate(results: list[MetricResult]) -> dict[str, Any]:
        """Compute overall accuracy and per-class stats from a list of routing results."""
        if not results:
            return {}
        total   = len(results)
        correct = sum(1 for r in results if r.score == 1.0)

        # Per-class counts from reasoning strings
        per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for r in results:
            parts = dict(p.split("=") for p in r.reasoning.split(", "))
            pred = parts.get("predicted", "")
            exp  = parts.get("expected",  "")
            if r.score == 1.0:
                per_class[pred]["tp"] += 1
            else:
                per_class[pred]["fp"] += 1
                per_class[exp]["fn"]  += 1

        per_class_stats = {}
        for cls, counts in per_class.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_class_stats[cls] = {
                "precision": round(precision, 3),
                "recall":    round(recall,    3),
                "f1":        round(f1,        3),
                "support":   tp + fn,
            }

        return {
            "accuracy":  round(correct / total, 4),
            "correct":   correct,
            "total":     total,
            "per_class": per_class_stats,
        }


# ── Latency tracker 
class LatencyTracker:
    """Collects latency_ms per route and computes percentile stats."""

    def __init__(self):
        self._all:     list[float]                     = []
        self._by_route: dict[str, list[float]]         = defaultdict(list)

    def record(self, latency_ms: float, route: RouteType):
        self._all.append(latency_ms)
        self._by_route[route.value].append(latency_ms)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        def pct(p: float) -> float:
            idx = int(p / 100 * n)
            return round(s[min(idx, n - 1)], 2)
        return {
            "mean": round(statistics.mean(s), 2),
            "p50":  pct(50),
            "p95":  pct(95),
            "p99":  pct(99),
            "min":  round(s[0], 2),
            "max":  round(s[-1], 2),
            "n":    n,
        }

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"overall": self._stats(self._all)}
        for route, values in self._by_route.items():
            result[route] = self._stats(values)
        return result


# ── Dataset loader

def _load_eval_csv(path: str | Path) -> list[EvalSample]:
    """
    Load evaluation CSV.
    Required columns: query, reference
    Optional column:  expected_route
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}\n"
            f"Create a CSV with columns: query, reference [, expected_route]"
        )

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]

        if "query" not in headers:
            raise ValueError(f"CSV missing required 'query' column. Found: {reader.fieldnames}")
        if "reference" not in headers:
            raise ValueError(
                f"CSV missing required 'reference' column. Found: {reader.fieldnames}\n"
                f"Add a 'reference' column with ground-truth answers."
            )

        for i, row in enumerate(reader, start=2):
            q   = row.get("query",     "").strip()
            ref = row.get("reference", "").strip()
            if not q:
                logger.warning(f"Row {i}: empty query, skipping")
                continue
            rows.append(EvalSample.from_csv_row(row))

    logger.info(f"Loaded {len(rows)} eval samples from {path}")
    return rows


# ── Main evaluator 

class RAGEvaluator:
    """
    Orchestrates all metrics for the Hybrid-RAG pipeline.

    Usage
    ─────
        from hybrid.fusion import HybridPipeline
        from evaluation.metrics import RAGEvaluator

        pipeline  = HybridPipeline()
        evaluator = RAGEvaluator(pipeline)
        report    = evaluator.run("evaluation/data/eval_set.csv", sample=50)
        report.print_summary()
        report.save("evaluation/reports/latest.json")

    Metrics computed
    ────────────────
        rouge_l             — always (no API needed)
        faithfulness        — requires GROQ_API_KEY
        answer_relevancy    — requires GROQ_API_KEY
        context_precision   — requires GROQ_API_KEY
        routing_correct     — only when expected_route column present

    Parameters
    ──────────
        pipeline        : HybridPipeline instance
        metrics         : list of metric names to compute (default: all)
                          Options: "rouge_l", "faithfulness",
                                   "answer_relevancy", "context_precision"
        use_llm_metrics : set False to skip LLM-judge metrics (ROUGE only)
                          Useful for quick offline evaluation
    """

    _ALL_METRICS = ["rouge_l", "faithfulness", "answer_relevancy", "context_precision"]

    def __init__(
        self,
        pipeline:        Any,                          # HybridPipeline
        metrics:         list[str] | None = None,
        use_llm_metrics: bool             = True,
    ):
        self._pipeline = pipeline
        cfg_metrics    = get_settings().llm   # re-use llm cfg for api key check
        api_key_set    = bool(os.getenv("GROQ_API_KEY", ""))

        requested = metrics or self._ALL_METRICS
        if not use_llm_metrics or not api_key_set:
            if not api_key_set:
                logger.warning(
                    "GROQ_API_KEY not set — LLM-judge metrics "
                    "(faithfulness, answer_relevancy, context_precision) will be skipped. "
                    "Only ROUGE-L will be computed."
                )
            self._active_metrics = [m for m in requested if m == "rouge_l"]
        else:
            self._active_metrics = requested

        # Instantiate evaluators
        self._rouge     = RougeEvaluator()
        self._faith     = FaithfulnessEvaluator()    if "faithfulness"       in self._active_metrics else None
        self._relevancy = AnswerRelevancyEvaluator() if "answer_relevancy"   in self._active_metrics else None
        self._ctx_prec  = ContextPrecisionEvaluator()if "context_precision"  in self._active_metrics else None
        self._routing   = RoutingEvaluator()
        self._latency   = LatencyTracker()

        logger.info(f"RAGEvaluator ready — active metrics: {self._active_metrics}")
    def run(
        self,
        dataset_path: str | Path,
        sample:       int | None = None,
    ) -> EvalReport:
        """
        Evaluate the pipeline on every sample in the CSV.

        Parameters
        ──────────
        dataset_path : path to eval CSV (query, reference [, expected_route])
        sample       : if set, evaluate only the first N samples
        """
        samples = _load_eval_csv(dataset_path)
        if sample:
            samples = samples[:sample]
            logger.info(f"Evaluating on first {len(samples)} samples")

        sample_results: list[SampleResult] = []
        routing_results: list[MetricResult] = []

        for i, s in enumerate(samples, start=1):
            logger.info(f"[{i}/{len(samples)}] Evaluating: '{s.query[:70]}'")

            # ── Run pipeline
            try:
                response: RAGResponse = self._pipeline.query(
                    s.query, return_context=True
                )
            except Exception as e:
                logger.error(f"Pipeline failed on sample {i}: {e}")
                # Record a failed sample with zero scores
                sample_results.append(SampleResult(
                    query=s.query, reference=s.reference,
                    answer=f"[pipeline_error: {e}]",
                    route_used="error", expected_route=None,
                    latency_ms=0.0,
                    metrics=[
                        MetricResult(name=m, score=0.0,
                                     reasoning="pipeline_error", skipped=True)
                        for m in self._active_metrics
                    ],
                ))
                continue

            answer   = response.answer
            context  = response.metadata.get("context", "")
            route    = response.route_used
            latency  = response.latency_ms

            self._latency.record(latency, route)

            # ── Compute metrics
            metric_results: list[MetricResult] = []

            # ROUGE-L (always)
            if "rouge_l" in self._active_metrics:
                metric_results.append(self._rouge.score(answer, s.reference))

            # Faithfulness (LLM)
            if self._faith:
                metric_results.append(self._faith.score(s.query, answer, context))

            # Answer relevancy (LLM)
            if self._relevancy:
                metric_results.append(self._relevancy.score(s.query, answer))

            # Context precision (LLM)
            if self._ctx_prec:
                metric_results.append(
                    self._ctx_prec.score(s.query, context, s.reference)
                )

            # Routing accuracy
            if s.expected_route is not None:
                r = self._routing.score(route, s.expected_route)
                metric_results.append(r)
                routing_results.append(r)

            sample_results.append(SampleResult(
                query=s.query,
                reference=s.reference,
                answer=answer,
                route_used=route.value,
                expected_route=s.expected_route.value if s.expected_route else None,
                latency_ms=latency,
                metrics=metric_results,
            ))

            # Log per-sample scores
            scores_str = "  ".join(
                f"{m.name}={m.score:.3f}" for m in metric_results if not m.skipped
            )
            logger.info(f"  [{route.value}] {scores_str}  ({latency:.0f}ms)")

        # ── Aggregate 
        report = self._aggregate(sample_results, routing_results)
        return report

    def _aggregate(
        self,
        sample_results:  list[SampleResult],
        routing_results: list[MetricResult],
    ) -> EvalReport:
        from datetime import datetime
        if not sample_results:
            logger.warning("No sample results to aggregate")
            return EvalReport(
                total_samples=0,
                metrics_summary={},
                per_route={},
                routing_accuracy=None,
                latency_stats={},
            )

        # ── Per-metric averages (skip skipped samples) 
        metric_scores: dict[str, list[float]] = defaultdict(list)
        for sr in sample_results:
            for m in sr.metrics:
                if not m.skipped and m.name != "routing_correct":
                    metric_scores[m.name].append(m.score)

        metrics_summary = {
            name: round(statistics.mean(scores), 4)
            for name, scores in metric_scores.items()
            if scores
        }

        # ── Per-route breakdown 
        route_samples: dict[str, list[SampleResult]] = defaultdict(list)
        for sr in sample_results:
            route_samples[sr.route_used].append(sr)

        per_route: dict[str, dict[str, float]] = {}
        for route, rs in route_samples.items():
            route_metric_scores: dict[str, list[float]] = defaultdict(list)
            for sr in rs:
                for m in sr.metrics:
                    if not m.skipped and m.name != "routing_correct":
                        route_metric_scores[m.name].append(m.score)
            per_route[route] = {
                name: round(statistics.mean(vals), 4)
                for name, vals in route_metric_scores.items()
                if vals
            }
            per_route[route]["sample_count"] = len(rs)

        # ── Routing accuracy 
        routing_acc = None
        if routing_results:
            agg = RoutingEvaluator.aggregate(routing_results)
            routing_acc = agg.get("accuracy")

        # ── Latency
        latency_stats = self._latency.summary()

        return EvalReport(
            total_samples=len(sample_results),
            metrics_summary=metrics_summary,
            per_route=per_route,
            routing_accuracy=routing_acc,
            latency_stats=latency_stats,
            sample_results=sample_results,
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )


# ── CSV template generator 

def create_eval_template(output_path: str = "evaluation/data/eval_set.csv"):
    """
    Write a starter eval CSV with column headers and example rows.
    Edit with real queries + reference answers before running evaluation.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "query":          "what are the main risk factors in this filing",
            "reference":      "The main risk factors include market volatility, regulatory changes, and supply chain disruptions.",
            "expected_route": "vector",
        },
        {
            "query":          "what is the total amount on invoice INV-2041",
            "reference":      "The total amount on invoice INV-2041 is $12,450.00.",
            "expected_route": "vectorless",
        },
        {
            "query":          "how have SLA terms changed since 2022",
            "reference":      "SLA terms were updated in 2023 to reduce response time from 48h to 24h.",
            "expected_route": "hybrid",
        },
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["query", "reference", "expected_route"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Eval template written → {path}")
    print(f"Template created at {path}")
    print("Edit with your real queries and reference answers, then run:")
    print(f"  python -m evaluation.metrics --data {path}")


# ── CLI

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation.metrics",
        description="Hybrid-RAG evaluation — compute ROUGE-L, faithfulness, relevancy, latency",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Evaluate pipeline on a dataset CSV")
    p_run.add_argument("--data",   required=True, help="Eval CSV path (query,reference[,expected_route])")
    p_run.add_argument("--output", default=None,  help="Save JSON report to this path")
    p_run.add_argument("--sample", type=int, default=None, help="Evaluate only first N samples")
    p_run.add_argument("--no-llm", action="store_true",
                       help="Skip LLM-judge metrics (ROUGE-L only, no API key needed)")
    p_run.add_argument("--fusion", default=None,
                       choices=["rrf", "linear"], help="Override fusion method")

    # template
    p_tpl = sub.add_parser("template", help="Create a starter eval CSV template")
    p_tpl.add_argument("--output", default="evaluation/data/eval_set.csv")

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "template":
        create_eval_template(args.output)
        return

    if args.command == "run":
        from hybrid.fusion import HybridPipeline

        pipeline  = HybridPipeline(
            fusion_method=args.fusion,
            generate=not args.no_llm,
        )
        evaluator = RAGEvaluator(
            pipeline,
            use_llm_metrics=not args.no_llm,
        )
        report = evaluator.run(args.data, sample=args.sample)
        report.print_summary()

        if args.output:
            report.save(args.output)
        else:
            # Auto-save to evaluation/reports/
            from datetime import datetime
            ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            auto = Path("evaluation/reports") / f"report_{ts}.json"
            report.save(auto)


if __name__ == "__main__":
    main()