"""Pattern clustering of text segments (K-means or agglomerative) on LSA vectors."""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from thematicml.config import AnalysisConfig

NOISE = -1


class ClusteringError(ValueError):
    pass


def lsa_embed(X: sp.csr_matrix, n_components: int, random_state: int) -> np.ndarray:
    """Reduce TF-IDF with truncated SVD and L2-normalise (latent semantic analysis)."""
    k = max(1, min(n_components, X.shape[1] - 1, X.shape[0] - 1))
    if k < 2:
        return normalize(X.toarray())
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    return normalize(svd.fit_transform(X))


def _fit(method: str, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def _silhouette(Z, labels, seed) -> float | None:
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    n = len(labels)
    return float(silhouette_score(Z, labels, metric="cosine", sample_size=min(n, 2000), random_state=seed))


def cluster_segments(X: sp.csr_matrix, cfg: AnalysisConfig) -> dict:
    """Cluster rows of X. Rows with no features get label -1 (unassigned).

    Returns dict with labels (np.ndarray), embedding (np.ndarray, rows with
    features only), metrics, and k-selection diagnostics.
    """
    t0 = time.perf_counter()
    has_feat = np.diff(X.indptr) > 0
    idx = np.flatnonzero(has_feat)
    n = len(idx)
    if n < 3:
        raise ClusteringError("Clustering requires at least 3 segments with extracted features.")
    if cfg.clustering_method == "hierarchical" and n > cfg.max_hierarchical_segments:
        raise ClusteringError(
            f"Hierarchical clustering is limited to {cfg.max_hierarchical_segments} segments "
            f"(this run has {n}); use K-means or raise the limit in settings."
        )
    Z = lsa_embed(X[idx], cfg.svd_components, cfg.random_state)

    selection = []
    if cfg.n_clusters:
        k = min(cfg.n_clusters, n - 1)
        sub = _fit(cfg.clustering_method, Z, k, cfg.random_state)
    else:
        best = None
        for k_try in range(cfg.k_min, min(cfg.k_max, n - 1) + 1):
            lab = _fit(cfg.clustering_method, Z, k_try, cfg.random_state)
            s = _silhouette(Z, lab, cfg.random_state)
            selection.append({"k": k_try, "silhouette": None if s is None else round(s, 4)})
            if s is not None and (best is None or s > best[0]):
                best = (s, k_try, lab)
        if best is None:
            raise ClusteringError("Could not select a number of clusters.")
        _, k, sub = best

    labels = np.full(X.shape[0], NOISE, dtype=int)
    labels[idx] = sub
    sil = _silhouette(Z, sub, cfg.random_state)
    metrics = {
        "algorithm": cfg.clustering_method,
        "n_clusters": int(len(set(sub))),
        "k_selection": "fixed" if cfg.n_clusters else "max silhouette over k range",
        "k_candidates": selection,
        "representation": f"TF-IDF → truncated SVD ({Z.shape[1]} components) → L2 normalised",
        "silhouette_cosine": None if sil is None else round(sil, 4),
        "davies_bouldin": round(float(davies_bouldin_score(Z, sub)), 4) if len(set(sub)) > 1 else None,
        "calinski_harabasz": round(float(calinski_harabasz_score(Z, sub)), 2) if len(set(sub)) > 1 else None,
        "unassigned_segments": int((~has_feat).sum()),
        "unassigned_pct": round(100.0 * (~has_feat).sum() / X.shape[0], 2),
        "sizes": {int(c): int((sub == c).sum()) for c in sorted(set(sub))},
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    return {"labels": labels, "embedding": Z, "embedded_rows": idx, "metrics": metrics}


def describe_clusters(X: sp.csr_matrix, vocab, labels: np.ndarray, Z: np.ndarray, rows: np.ndarray, n_terms: int = 10, n_rep: int = 5) -> list[dict]:
    """Distinctive terms (mean TF-IDF in cluster minus outside) and the
    segments closest to each cluster centroid."""
    out = []
    pos_of = {r: i for i, r in enumerate(rows)}
    overall = np.asarray(X.mean(axis=0)).ravel()
    for c in sorted(set(labels.tolist()) - {NOISE}):
        members = np.flatnonzero(labels == c)
        inside = np.asarray(X[members].mean(axis=0)).ravel()
        n_out = X.shape[0] - len(members)
        outside = (overall * X.shape[0] - inside * len(members)) / max(n_out, 1)
        dist = inside - outside
        top = np.argsort(-dist)[:n_terms]
        zi = np.array([pos_of[m] for m in members])
        centroid = Z[zi].mean(axis=0)
        sims = Z[zi] @ centroid / (np.linalg.norm(centroid) or 1.0)
        order = np.argsort(-sims)
        out.append(
            {
                "cluster": int(c),
                "size": int(len(members)),
                "terms": [{"term": str(vocab[j]), "distinctiveness": round(float(dist[j]), 4)} for j in top if dist[j] > 0],
                "representative": [{"segment_pos": int(members[i]), "similarity": round(float(sims[i]), 4)} for i in order[:n_rep]],
                "cohesion": round(float(sims.mean()), 4),
            }
        )
    return out
