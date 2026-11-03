"""Classical link-prediction baseline for code relationships.

Holds out a fraction of observed co-occurrence edges, scores them against an
equal number of sampled non-edges with neighbourhood heuristics, and reports
ROC AUC. Any optional GNN must beat these baselines before it is preferred.
"""
from __future__ import annotations

import random

import networkx as nx


def link_prediction_baseline(G: nx.Graph, holdout: float = 0.2, seed: int = 42) -> dict:
    edges = list(G.edges())
    if len(edges) < 10 or G.number_of_nodes() < 5:
        return {"available": False, "reason": "Needs at least 10 edges and 5 nodes."}
    rng = random.Random(seed)
    rng.shuffle(edges)
    n_test = max(2, int(len(edges) * holdout))
    test_pos = edges[:n_test]
    train = G.copy()
    train.remove_edges_from(test_pos)
    nodes = list(G.nodes())
    non_edges = set()
    attempts = 0
    while len(non_edges) < n_test and attempts < n_test * 50:
        a, b = rng.sample(nodes, 2)
        attempts += 1
        if not G.has_edge(a, b):
            non_edges.add(tuple(sorted((a, b))))
    test_neg = list(non_edges)
    if not test_neg:
        return {"available": False, "reason": "Graph is too dense to sample non-edges."}
    pairs = test_pos + test_neg
    y = [1] * len(test_pos) + [0] * len(test_neg)
    results = {}
    for name, fn in (
        ("adamic_adar", nx.adamic_adar_index),
        ("jaccard", nx.jaccard_coefficient),
        ("preferential_attachment", nx.preferential_attachment),
    ):
        scores = [s for _, _, s in fn(train, pairs)]
        from sklearn.metrics import roc_auc_score  # imported on first use: slow to load in the browser edition

        results[name] = round(float(roc_auc_score(y, scores)), 4)
    return {
        "available": True,
        "task": "co-occurrence link prediction",
        "holdout_fraction": holdout,
        "test_positive": len(test_pos),
        "test_negative": len(test_neg),
        "roc_auc": results,
        "seed": seed,
    }
