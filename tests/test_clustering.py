import numpy as np
import pytest

from thematicml.clustering import ClusteringError, cluster_segments, describe_clusters
from thematicml.config import AnalysisConfig
from thematicml.features import count_matrix, fit_tfidf
from thematicml.topics import discover_topics, umass_coherence

DOCS = (["slow loading page freezes performance"] * 10 + ["confusing menu navigation settings"] * 10
        + ["expensive price subscription plan"] * 10 + ["the and of"])


@pytest.fixture
def feats():
    return fit_tfidf(DOCS, AnalysisConfig(min_df=1, ngram_max=1))


def test_kmeans_fixed_k_and_unassigned(feats):
    res = cluster_segments(feats.matrix, AnalysisConfig(n_clusters=3))
    labels = res["labels"]
    assert labels[-1] == -1  # stopword-only document has no features
    assert len(set(labels[:10])) == 1 and len(set(labels[10:20])) == 1 and len(set(labels[20:30])) == 1
    assert res["metrics"]["n_clusters"] == 3
    assert res["metrics"]["unassigned_segments"] == 1
    d = describe_clusters(feats.matrix, feats.vocabulary, labels, res["embedding"], res["embedded_rows"])
    terms = {c["cluster"]: [t["term"] for t in c["terms"]] for c in d}
    assert any("navigation" in t for t in terms.values())


def test_auto_k_selects_by_silhouette(feats):
    res = cluster_segments(feats.matrix, AnalysisConfig(k_min=2, k_max=6))
    assert res["metrics"]["n_clusters"] == 3
    assert res["metrics"]["k_candidates"]


def test_hierarchical(feats):
    res = cluster_segments(feats.matrix, AnalysisConfig(clustering_method="hierarchical", n_clusters=3))
    assert res["metrics"]["algorithm"] == "hierarchical"


def test_hierarchical_limit(feats):
    cfg = AnalysisConfig(clustering_method="hierarchical", n_clusters=2).model_copy(update={"max_hierarchical_segments": 5})
    with pytest.raises(ClusteringError):
        cluster_segments(feats.matrix, cfg)


@pytest.mark.parametrize("method", ["nmf", "lda"])
def test_topics(feats, method):
    counts = count_matrix(DOCS, feats)
    res = discover_topics(feats.matrix, counts, feats.vocabulary, method, 3, 4, 0.3, 0)
    assert res["metrics"]["n_topics"] == 3
    assert len(res["dominant"]) == len(DOCS)
    assert res["dominant"][-1] == -1
    assert 0 < res["metrics"]["topic_diversity"] <= 1
    # UMass with +1 smoothing is <= log(2) when terms always co-occur
    assert all(t["coherence_umass"] is None or t["coherence_umass"] <= 0.7 for t in res["topics"])


def test_topics_ignore_negation_only_terms():
    docs = (["not slow loading page not"] * 6 + ["don't confusing menu navigation don't"] * 6
            + ["never expensive price plan never"] * 6)
    f = fit_tfidf(docs, AnalysisConfig(min_df=1, ngram_max=2))
    assert "not" in set(f.vocabulary)  # kept for coding
    res = discover_topics(f.matrix, count_matrix(docs, f), f.vocabulary, "nmf", 3, 8, 0.3, 0)
    terms = {t for topic in res["topics"] for t in topic["terms"]}
    assert not terms & {"not", "don't", "never"}
    assert "slow" in terms and "not slow" in terms  # negated content phrases stay


def test_umass_simple():
    import scipy.sparse as sp

    B = sp.csc_matrix(np.array([[1, 1], [1, 1], [1, 0]], dtype=float))
    # D(w1,w0)=2, D(w0)=3 -> log(3/3)=0
    assert umass_coherence(B, [0, 1]) == pytest.approx(0.0)


def test_nmf_start_is_passed_in_c_order(monkeypatch):
    """scikit-learn 1.8 (browser edition) returns a Fortran-ordered NNDSVDa start when passages < terms and then
    fails with 'ndarray is not C-contiguous'. Simulate that start: topics must still be found, identical to before."""
    import sklearn.decomposition._nmf as nmf_mod
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    from thematicml.topics import _fit_nmf

    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(120)]
    X = TfidfVectorizer().fit_transform([" ".join(rng.choice(words, 10)) for _ in range(6)])  # fewer passages than terms
    expected = NMF(n_components=4, init="nndsvda", random_state=7, max_iter=400).fit_transform(X)
    real = nmf_mod._initialize_nmf
    monkeypatch.setattr(nmf_mod, "_initialize_nmf", lambda *a, **k: tuple(np.asfortranarray(m) for m in real(*a, **k)))
    model, W = _fit_nmf(X, 4, 7)
    np.testing.assert_array_equal(W, expected)
    assert model.components_.shape == (4, X.shape[1])
