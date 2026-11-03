"""Topic discovery with NMF (on TF-IDF) or LDA (on term counts)."""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from thematicml.features import NEGATIONS


def umass_coherence(binary: sp.csc_matrix, term_ids: list[int]) -> float | None:
    """UMass coherence (Mimno et al., 2011) over document co-occurrence.

    Mean over ordered pairs of log((D(wi, wj) + 1) / D(wj)). Values closer to
    zero indicate more coherent topics. Computed on the analysed corpus itself.
    """
    if len(term_ids) < 2:
        return None
    cols = binary[:, term_ids]
    df = np.asarray(cols.sum(axis=0)).ravel()
    co = (cols.T @ cols).toarray()
    total, pairs = 0.0, 0
    for i in range(1, len(term_ids)):
        for j in range(i):
            if df[j] > 0:
                total += np.log((co[i, j] + 1.0) / df[j])
                pairs += 1
    return float(total / pairs) if pairs else None


def discover_topics(
    X_tfidf: sp.csr_matrix,
    X_counts: sp.csr_matrix,
    vocab,
    method: str = "nmf",
    n_topics: int = 8,
    n_top_terms: int = 10,
    min_weight: float = 0.3,
    random_state: int = 42,
) -> dict:
    t0 = time.perf_counter()
    # terms made only of negations ("not", "don't", "not not") are kept for coding but carry no topic content
    keep = np.array([not all(w in NEGATIONS for w in str(t).split()) for t in vocab], dtype=bool)
    if not keep.all() and keep.sum() >= 4:
        X_tfidf, X_counts = X_tfidf.tocsc()[:, keep].tocsr(), X_counts.tocsc()[:, keep].tocsr()
        vocab = np.asarray(vocab, dtype=object)[keep]
    k = max(2, min(n_topics, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1))
    if method == "nmf":
        from sklearn.decomposition import NMF  # imported on first use: slow to load in the browser edition

        model = NMF(n_components=k, init="nndsvda", random_state=random_state, max_iter=400)
        W = model.fit_transform(X_tfidf)
        objective = {"reconstruction_err": round(float(model.reconstruction_err_), 4)}
    else:
        from sklearn.decomposition import LatentDirichletAllocation  # imported on first use: slow to load in the browser edition

        model = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=random_state, max_iter=50)
        W = model.fit_transform(X_counts)
        objective = {"perplexity": round(float(model.perplexity(X_counts)), 2)}
    H = model.components_

    sums = W.sum(axis=1, keepdims=True)
    share = np.divide(W, sums, out=np.zeros_like(W), where=sums > 0)
    dominant = share.argmax(axis=1)
    dom_w = share.max(axis=1)
    # documents without any vocabulary terms get no topic (LDA would otherwise return its uniform prior)
    has_terms = np.diff(X_tfidf.tocsr().indptr) > 0
    dominant = np.where(has_terms & (sums.ravel() > 0) & (dom_w >= min_weight), dominant, -1)

    binary = (X_counts > 0).astype(np.float64).tocsc()
    topics, top_sets = [], []
    for t in range(k):
        order = np.argsort(-H[t])[:n_top_terms]
        top_sets.append(set(order.tolist()))
        topics.append(
            {
                "topic": t,
                "terms": [str(vocab[j]) for j in order],
                "weights": [round(float(H[t, j]), 4) for j in order],
                "size": int((dominant == t).sum()),
                "coherence_umass": None if (c := umass_coherence(binary, order.tolist())) is None else round(c, 4),
                "label": " / ".join(str(vocab[j]) for j in order[:3]),
            }
        )
    all_terms = [j for s in top_sets for j in s]
    coh = [t["coherence_umass"] for t in topics if t["coherence_umass"] is not None]
    metrics = {
        "method": method,
        "n_topics": k,
        "topic_diversity": round(len(set(all_terms)) / len(all_terms), 4) if all_terms else None,
        "mean_coherence_umass": round(float(np.mean(coh)), 4) if coh else None,
        "segments_without_dominant_topic": int((dominant == -1).sum()),
        "dominant_topic_rule": f"topic share >= {min_weight}",
        "runtime_s": round(time.perf_counter() - t0, 3),
        **objective,
    }
    return {"topics": topics, "dominant": dominant, "dominant_weight": dom_w, "doc_topic": share, "metrics": metrics}
