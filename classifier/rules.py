"""
classifier/rules.py
────────────────────
Fast rule-based query routing. Runs before the ML classifier.
Returns None if no rule fires (falls through to ML model).
"""
from __future__ import annotations
import re

from src.models import QueryIntent, RouteType, VectorlessMethod


# ── Pattern banks 

_EXACT_LOOKUP = re.compile(
    r"""
    \b(
      what\s+is\s+the\s+(total|amount|number|count|date|status|value|price|id)|
      (invoice|order|contract|document|ticket|ref|id|po|case)\s*[:\-#]?\s*\w+|
      (show|get|fetch|find|lookup|retrieve)\s+.{0,30}(by\s+(id|number|code|ref))|
      (exact|specific|precise)\s+
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_KEYWORD_SEARCH = re.compile(
    r"""
    \b(
      find\s+all|search\s+for|containing|mentioning|documents?\s+with|
      (all|every)\s+(contracts?|files?|documents?|records?)\s+(that|which|where)|
      keyword|phrase|term|clause
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ENTITY_RELATION = re.compile(
    r"""
    \b(
      who\s+(is|was|approved|signed|created|modified|owns?|wrote)|
      (approved|authored|signed|created|submitted)\s+by|
      (relationship|relation|between|linked\s+to|associated\s+with)|
      (manager|owner|author|assignee)\s+of
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEMANTIC = re.compile(
    r"""
    \b(
      summarize|explain|describe|analyse|analyze|understand|
      what\s+does\s+.+\s+mean|how\s+does|why\s+(is|was|did)|
      (key|main|important)\s+(points?|findings?|insights?|topics?)|
      overview|compare|contrast|implications?|impact\s+of
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TEMPORAL_CHANGE = re.compile(
    r"""
    \b(
      (has|have)\s+.+\s+changed|
      (change|diff|difference)\s+(since|between|from)|
      (evolution|history|timeline)\s+of|
      (before|after|compared\s+to)\s+(2\d{3}|last\s+year|this\s+year)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ID_PATTERN = re.compile(r"\b[A-Z]{2,}-?\d{3,}\b")  # e.g. PO-4421, INV-009


def classify(query: str) -> QueryIntent | None:
    """
    Returns a QueryIntent if a rule fires with high confidence,
    or None to signal fallback to ML classifier.
    """
    q = query.strip()

    # Structural IDs in query → very likely exact lookup
    if _ID_PATTERN.search(q):
        return QueryIntent(
            route=RouteType.VECTORLESS,
            vectorless_method=VectorlessMethod.SQL,
            confidence=0.95,
            reasoning="Query contains a structured ID pattern",
        )

    if _EXACT_LOOKUP.search(q):
        return QueryIntent(
            route=RouteType.VECTORLESS,
            vectorless_method=VectorlessMethod.SQL,
            confidence=0.88,
            reasoning="Exact lookup pattern matched",
        )

    if _KEYWORD_SEARCH.search(q):
        return QueryIntent(
            route=RouteType.VECTORLESS,
            vectorless_method=VectorlessMethod.BM25,
            confidence=0.85,
            reasoning="Keyword/phrase search pattern matched",
        )

    if _ENTITY_RELATION.search(q):
        return QueryIntent(
            route=RouteType.VECTORLESS,
            vectorless_method=VectorlessMethod.GRAPH,
            confidence=0.82,
            reasoning="Entity-relation pattern matched",
        )

    if _TEMPORAL_CHANGE.search(q):
        return QueryIntent(
            route=RouteType.HYBRID,
            confidence=0.80,
            reasoning="Temporal comparison — needs both structured dates and semantic content",
        )

    if _SEMANTIC.search(q):
        return QueryIntent(
            route=RouteType.VECTOR,
            confidence=0.83,
            reasoning="Semantic/analytical pattern matched",
        )

    # No rule fired
    return None