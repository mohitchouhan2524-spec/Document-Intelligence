import re
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """
    12 binary/numeric features capturing signal that TF-IDF misses:
      • Presence of structured IDs (regex)
      • Query length buckets
      • Wh-question type
      • Temporal signals
      • Aggregation signals
      • Comparison signals
      • Entity-relation signals
    Returns a dense (n_samples, 12) float32 array.
    """

    _ID       = re.compile(r"\b[A-Z]{2,}-?\d{3,}\b")
    _DATE     = re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|Q[1-4]\s+\d{4}|last\s+year|this\s+year)\b", re.I)
    _AGGR     = re.compile(r"\b(total|count|how\s+many|number\s+of|sum|average|max|min|list\s+all)\b", re.I)
    _COMPARE  = re.compile(r"\b(compare|vs|versus|differ|change|evolv|trend|between)\b", re.I)
    _ENTITY   = re.compile(r"\b(who\s+(is|was|approved|signed|created|owns?|wrote)|authored\s+by|signed\s+by)\b", re.I)
    _KEYWORD  = re.compile(r"\b(containing|mentioning|with\s+the\s+(word|phrase|term|clause)|search\s+for)\b", re.I)
    _SEMANTIC = re.compile(r"\b(summari[sz]e|explain|describe|analy[sz]e|what\s+does|overview|implications?|how\s+does)\b", re.I)
    _TEMPORAL = re.compile(r"\b(since|before|after|between|changed|history|timeline|evolv)\b", re.I)
    _NEGATION = re.compile(r"\b(not|no|without|excluding|except)\b", re.I)
    _PLURAL   = re.compile(r"\b(all|every|each|multiple|several|list)\b", re.I)

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        feats = np.zeros((len(X), 12), dtype=np.float32)
        for i, q in enumerate(X):
            toks   = q.split()
            n      = len(toks)
            feats[i, 0]  = float(bool(self._ID.search(q)))          # has_structured_id
            feats[i, 1]  = float(bool(self._DATE.search(q)))         # has_date_ref
            feats[i, 2]  = float(bool(self._AGGR.search(q)))         # has_aggregation
            feats[i, 3]  = float(bool(self._COMPARE.search(q)))      # has_comparison
            feats[i, 4]  = float(bool(self._ENTITY.search(q)))       # has_entity_relation
            feats[i, 5]  = float(bool(self._KEYWORD.search(q)))      # has_keyword_signal
            feats[i, 6]  = float(bool(self._SEMANTIC.search(q)))     # has_semantic_signal
            feats[i, 7]  = float(bool(self._TEMPORAL.search(q)))     # has_temporal_signal
            feats[i, 8]  = float(bool(self._NEGATION.search(q)))     # has_negation
            feats[i, 9]  = float(bool(self._PLURAL.search(q)))       # has_plural_intent
            feats[i, 10] = min(n / 20.0, 1.0)                        # query_length_norm
            feats[i, 11] = float(q.lower().startswith("who "))        # starts_with_who
        return feats
