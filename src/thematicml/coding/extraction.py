"""Candidate codes from TF-IDF-ranked keywords and n-gram phrases."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from thematicml.features import NEGATIONS, FeatureSet, phrase_regex

CODE_COLUMNS = ["code_key", "label", "description", "method", "feature", "category"]
ASSIGN_COLUMNS = ["code_key", "segment_pos", "method", "score", "matched_text", "char_start", "char_end", "negated"]


def _label(term: str) -> str:
    return " ".join(w.capitalize() if w not in NEGATIONS else w for w in term.split())


def _negated_before(text: str, start: int) -> bool:
    window = text[max(0, start - 30):start].lower().replace("’", "'")
    toks = window.split()[-3:]
    return any(t.strip(".,;:!?") in NEGATIONS for t in toks)


def extract_phrase_codes(
    features: FeatureSet,
    segments: pd.DataFrame,
    n_codes: int,
    min_support: int,
    phrase_boost: float = 1.25,
    skip: Callable[[str], bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select keyword/phrase codes and assign them to the segments containing them.

    Ranking score = sum of TF-IDF weight over segments, multiplied by
    ``phrase_boost`` for every token beyond the first (favouring
    multi-word expressions). Terms whose segment set is near-identical to an
    already selected code (Jaccard >= 0.8), or unigrams mostly covered by a
    selected phrase containing them, are skipped as redundant, as are terms
    for which ``skip`` returns True (e.g. already covered by a codebook rule);
    the next-ranked term takes their place.
    """
    X = features.matrix.tocsc()
    vocab = features.vocabulary
    df = np.diff(X.indptr)
    weight = np.asarray(X.sum(axis=0)).ravel()
    ntok = np.array([len(t.split()) for t in vocab])
    score = weight * phrase_boost ** (ntok - 1)

    def is_content(term: str) -> bool:
        toks = term.split()
        return not all(t in NEGATIONS for t in toks) and any(len(t) > 2 for t in toks)

    # lazy: ``skip`` is only consulted for terms ranked above the cut-off
    eligible = (j for j in np.argsort(-score)
                if df[j] >= min_support and is_content(vocab[j]) and not (skip and skip(str(vocab[j]))))

    selected: list[int] = []
    seg_sets: dict[int, set[int]] = {}
    for j in eligible:
        if len(selected) >= n_codes:
            break
        rows = set(X.indices[X.indptr[j]:X.indptr[j + 1]].tolist())
        term = vocab[j]
        redundant = False
        for k in selected:
            other = seg_sets[k]
            inter = len(rows & other)
            if inter / len(rows | other) >= 0.8:
                redundant = True
                break
            # unigram mostly explained by a selected phrase containing it (or vice versa)
            if ntok[j] == 1 and ntok[k] > 1 and term in vocab[k].split() and inter / len(rows) >= 0.6:
                redundant = True
                break
        if redundant:
            continue
        selected.append(j)
        seg_sets[j] = rows

    codes, assigns = [], []
    seg_texts = segments["text"].tolist()
    seg_starts = segments["char_start"].tolist()
    for j in selected:
        term = str(vocab[j])
        key = f"kw:{term}"
        codes.append(
            {
                "code_key": key,
                "label": _label(term),
                "description": f"Passages containing the {'phrase' if ntok[j] > 1 else 'keyword'} “{term}” "
                f"(selected by TF-IDF ranking; present in {int(df[j])} segments).",
                "method": "tfidf-phrase" if ntok[j] > 1 else "tfidf-keyword",
                "feature": term,
                "category": None,
            }
        )
        rx = phrase_regex(term)
        col = X[:, j]
        for pos, val in zip(col.indices.tolist(), col.data.tolist()):
            text = seg_texts[pos]
            m = rx.search(text)
            if m:
                s, e = m.start(), m.end()
                assigns.append((key, pos, "tfidf", round(float(val), 4), text[s:e], seg_starts[pos] + s, seg_starts[pos] + e, _negated_before(text, s)))
            else:  # term formed after synonym mapping; segment-level evidence only
                assigns.append((key, pos, "tfidf", round(float(val), 4), None, None, None, False))
    return pd.DataFrame(codes, columns=CODE_COLUMNS), pd.DataFrame(assigns, columns=ASSIGN_COLUMNS)
