"""Classical text representation: configurable normalisation, tokenisation and TF-IDF."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from thematicml.config import AnalysisConfig

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

NEGATIONS = frozenset(
    {
        "no", "not", "nor", "never", "none", "nobody", "nothing", "nowhere", "neither", "cannot",
        "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't", "aren't", "wasn't",
        "weren't", "shouldn't", "couldn't", "haven't", "hasn't", "hadn't", "without",
    }
)


class FeatureError(ValueError):
    pass


# Non-negation contractions; sklearn's list has "it" but not "it's". Negations stay (see NEGATIONS).
CONTRACTIONS = frozenset(
    {
        "it's", "i'm", "i've", "i'd", "i'll", "you're", "you've", "you'll", "you'd", "we're", "we've", "we'll", "we'd",
        "they're", "they've", "they'll", "they'd", "he's", "she's", "he'd", "she'd", "that's", "there's", "here's",
        "what's", "who's", "where's", "how's", "let's", "that'll", "it'll", "y'all",
    }
)


def build_stopwords(cfg: AnalysisConfig) -> set[str]:
    if not cfg.remove_stopwords:
        return {w.lower() for w in cfg.custom_stopwords}
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # imported on first use: slow to load in the browser edition

    words = set(ENGLISH_STOP_WORDS) | CONTRACTIONS
    if cfg.keep_negations:
        words -= NEGATIONS
    words |= {w.lower() for w in cfg.custom_stopwords}
    return words


def _synonym_pattern(synonyms: dict[str, str]):
    if not synonyms:
        return None
    keys = sorted((k for k in synonyms if k.strip()), key=len, reverse=True)
    if not keys:
        return None
    pat = re.compile(r"\b(" + "|".join(re.escape(k.lower()) for k in keys) + r")\b", re.IGNORECASE)
    lookup = {k.lower(): v.lower() for k, v in synonyms.items()}
    return pat, lookup


class TextNormaliser:
    """Picklable preprocessor + tokenizer shared by every vectorizer in a run."""

    def __init__(self, cfg: AnalysisConfig):
        self.lowercase = cfg.lowercase
        self.synonyms = dict(cfg.synonyms)
        self._syn = _synonym_pattern(self.synonyms)
        self.stopwords = build_stopwords(cfg)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_syn"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._syn = _synonym_pattern(self.synonyms)

    def preprocess(self, text: str) -> str:
        t = text.replace("’", "'").replace("‘", "'")
        if self.lowercase:
            t = t.lower()
        if self._syn:
            pat, lookup = self._syn
            t = pat.sub(lambda m: lookup[m.group(0).lower()], t)
        return t

    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in TOKEN_RE.findall(text) if tok not in self.stopwords and not tok.isdigit()]


@dataclass
class FeatureSet:
    matrix: sp.csr_matrix
    vectorizer: TfidfVectorizer
    normaliser: TextNormaliser
    vocabulary: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def summary(self, top_n: int = 25) -> dict:
        X = self.matrix
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        weight = np.asarray(X.sum(axis=0)).ravel()
        order = np.argsort(-weight)[:top_n]
        lengths = np.array([len(t.split()) for t in self.vocabulary])
        empty_rows = int((np.diff(X.indptr) == 0).sum())
        return {
            "n_documents": int(X.shape[0]),
            "vocabulary_size": int(X.shape[1]),
            "matrix_shape": [int(X.shape[0]), int(X.shape[1])],
            "nonzero": int(X.nnz),
            "density": float(X.nnz / (X.shape[0] * X.shape[1])) if X.shape[0] and X.shape[1] else 0.0,
            "ngram_counts": {str(n): int((lengths == n).sum()) for n in sorted(set(lengths.tolist()))},
            "empty_documents": empty_rows,
            "top_terms": [
                {"term": str(self.vocabulary[j]), "tfidf_sum": round(float(weight[j]), 4), "document_frequency": int(df[j])}
                for j in order
            ],
        }


def fit_tfidf(texts: list[str], cfg: AnalysisConfig) -> FeatureSet:
    n = len(texts)
    if n == 0:
        raise FeatureError("No text segments available for feature extraction.")
    norm = TextNormaliser(cfg)
    min_df = cfg.min_df if n >= cfg.min_df * 2 else 1
    max_df = cfg.max_df if n >= 10 else 1.0
    from sklearn.feature_extraction.text import TfidfVectorizer  # imported on first use: slow to load in the browser edition

    vec = TfidfVectorizer(
        preprocessor=norm.preprocess,
        tokenizer=norm.tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=cfg.max_features,
        sublinear_tf=True,
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError as exc:
        raise FeatureError(
            "No terms remained after preprocessing. Lower the minimum document frequency, "
            "disable stopword removal, or check that the selected text column contains text."
        ) from exc
    return FeatureSet(X.tocsr(), vec, norm, vec.get_feature_names_out())


def count_matrix(texts: list[str], features: FeatureSet) -> sp.csr_matrix:
    """Raw term counts over the same vocabulary (used by LDA and coherence)."""
    v = features.vectorizer
    from sklearn.feature_extraction.text import CountVectorizer  # imported on first use: slow to load in the browser edition

    cv = CountVectorizer(
        preprocessor=features.normaliser.preprocess,
        tokenizer=features.normaliser.tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=v.ngram_range,
        vocabulary=v.vocabulary_,
    )
    return cv.fit_transform(texts).tocsr()


def phrase_regex(term: str) -> re.Pattern:
    """Regex locating an extracted n-gram in original text, allowing up to two
    removed stopwords between its tokens (e.g. 'difficult navigate' matches
    'difficult to navigate')."""
    toks = [re.escape(t).replace("'", "['’‘]") for t in term.split()]
    gap = r"[\W_]+(?:[^\W_]+[\W_]+){0,2}?"
    return re.compile(r"(?<![^\W_])" + gap.join(toks) + r"(?![^\W_])", re.IGNORECASE)
