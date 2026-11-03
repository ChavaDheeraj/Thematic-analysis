"""Parts of the GraphSAGE experiments shared by both backends: graph conversion, leakage-free splits,
negative sampling, metrics and the classical baselines every GNN result is reported next to.

No neural-network library is imported here; :mod:`thematicml.gnn.core` (PyTorch Geometric) and
:mod:`thematicml.gnn.numpy_backend` (NumPy) build on it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from thematicml.config import GNNConfig
from thematicml.gnn import MAX_DENSITY, MIN_EDGES, MIN_LABELLED, MIN_NODES, MIN_PER_CLASS


class GNNDataError(ValueError):
    """The graph or labels cannot support the requested task."""


# ---------------------------------------------------------------------- data


@dataclass
class GraphData:
    node_ids: list[int]
    labels: list[str]
    x: np.ndarray
    edges: np.ndarray  # (m, 2) node indices, u < v, unique, no self loops
    weights: np.ndarray
    feature_description: str
    version: str
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.node_ids)

    def index_of(self) -> dict[int, int]:
        return {nid: i for i, nid in enumerate(self.node_ids)}


def node_features(labels: list[str], frequency: list[float], methods: list[str], feature_set: str) -> tuple[np.ndarray, str]:
    n = len(labels)
    if feature_set == "identity":
        return np.eye(n, dtype=np.float32), "one-hot node identity (transductive; cannot score unseen codes)"
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=128, alternate_sign=False, norm="l2")
    parts = [hv.transform([str(x).lower() for x in labels]).toarray()]
    desc = ["128 hashed character 2–4-grams of the code label"]
    if feature_set == "label+frequency":
        f = np.log1p(np.asarray(frequency, dtype=float))
        f = (f - f.mean()) / (f.std() or 1.0)
        cats = sorted({str(m) for m in methods})
        onehot = np.array([[1.0 if str(m) == c else 0.0 for c in cats] for m in methods]).reshape(n, len(cats))
        parts += [f[:, None], onehot]
        desc += ["standardised log segment frequency", f"coding method one-hot ({', '.join(cats)})"]
    return np.hstack(parts).astype(np.float32), "; ".join(desc)


def build_graph_data(nodes: list[dict], edges: list[dict], feature_set: str = "label+frequency", params: dict | None = None) -> GraphData:
    """nodes: [{id, label, frequency, method}], edges: [{source, target, weight, type}] (node ids)."""
    if not nodes:
        raise GNNDataError("The graph has no nodes (no active codes).")
    ids = [int(n["id"]) for n in nodes]
    if len(set(ids)) != len(ids):
        raise GNNDataError("Duplicate node ids in the graph.")
    idx = {nid: i for i, nid in enumerate(ids)}
    seen: dict[tuple[int, int], float] = {}
    dropped = {"non_cooccurrence": 0, "self_loops": 0, "unknown_nodes": 0, "duplicates": 0}
    for e in edges:
        if e.get("type", "cooccurrence") != "cooccurrence":
            dropped["non_cooccurrence"] += 1
            continue
        try:
            a, b = idx[int(e["source"])], idx[int(e["target"])]
        except (KeyError, TypeError, ValueError):
            dropped["unknown_nodes"] += 1
            continue
        if a == b:
            dropped["self_loops"] += 1
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            dropped["duplicates"] += 1
            continue
        seen[key] = float(e.get("weight", 1.0))
    E = np.array(sorted(seen), dtype=np.int64).reshape(-1, 2)
    W = np.array([seen[tuple(p)] for p in E.tolist()], dtype=np.float32)
    x, desc = node_features([n["label"] for n in nodes], [n.get("frequency", 1) for n in nodes],
                            [n.get("method", "") for n in nodes], feature_set)
    h = hashlib.sha256(json.dumps({"nodes": ids, "edges": E.tolist(), "params": params or {}, "features": feature_set}).encode()).hexdigest()[:12]
    return GraphData(ids, [str(n["label"]) for n in nodes], x, E, W, desc, h,
                     {"nodes": len(ids), "edges": int(len(E)), "feature_dim": int(x.shape[1]), "dropped_edges": dropped,
                      "edge_type": "cooccurrence (observed)", "directed": False, "params": params or {}})

# ---------------------------------------------------------------------- split


@dataclass
class LinkSplit:
    train_mp: np.ndarray       # message-passing edges during training
    train_sup: np.ndarray      # supervision (target) edges during training
    val: np.ndarray
    val_neg: np.ndarray
    test: np.ndarray
    test_neg: np.ndarray

    @property
    def train(self) -> np.ndarray:
        return np.vstack([self.train_mp, self.train_sup]) if len(self.train_sup) and self.train_sup is not self.train_mp else self.train_mp

    def sizes(self) -> dict:
        return {"train_message_passing": int(len(self.train_mp)), "train_supervision": int(len(self.train_sup)),
                "val_pos": int(len(self.val)), "val_neg": int(len(self.val_neg)),
                "test_pos": int(len(self.test)), "test_neg": int(len(self.test_neg))}


def _pairs_set(a: np.ndarray) -> set[tuple[int, int]]:
    return {(int(u), int(v)) for u, v in a}


def sample_negatives(n_nodes: int, k: int, forbidden: set[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    """k distinct undirected non-edges (u < v) avoiding ``forbidden``."""
    capacity = n_nodes * (n_nodes - 1) // 2 - len(forbidden)
    if k > capacity:
        raise GNNDataError(f"The graph is too dense to sample {k} negative pairs ({capacity} non-edges available).")
    out: set[tuple[int, int]] = set()
    if capacity <= 4 * k:  # dense: enumerate the complement
        pool = [(u, v) for u in range(n_nodes) for v in range(u + 1, n_nodes) if (u, v) not in forbidden]
        choice = rng.choice(len(pool), size=k, replace=False)
        return np.array([pool[i] for i in choice], dtype=np.int64).reshape(-1, 2)
    while len(out) < k:
        u, v = rng.integers(0, n_nodes, size=2)
        if u == v:
            continue
        p = (int(min(u, v)), int(max(u, v)))
        if p not in forbidden and p not in out:
            out.add(p)
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)




def link_split(data: GraphData, seed: int, val_fraction: float = 0.15, test_fraction: float = 0.15,
               disjoint_train_ratio: float = 0.0) -> LinkSplit:
    m = len(data.edges)
    if data.n < MIN_NODES or m < MIN_EDGES:
        raise GNNDataError(f"Link prediction needs at least {MIN_NODES} codes and {MIN_EDGES} co-occurrence edges "
                           f"(this graph has {data.n} and {m}). Lower the minimum co-occurrence or add codes.")
    density = m / (data.n * (data.n - 1) / 2)
    if density > MAX_DENSITY:
        raise GNNDataError(f"The co-occurrence graph is too dense for link prediction: {density:.0%} of all code pairs are already linked "
                           f"(limit {MAX_DENSITY:.0%}), so there are too few unlinked pairs to learn from or evaluate. Raise the minimum "
                           "co-occurrence or minimum NPMI, use segment-level co-occurrence, or remove generic codes.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    n_test = max(2, round(m * test_fraction))
    n_val = max(2, round(m * val_fraction))
    test = data.edges[perm[:n_test]]
    val = data.edges[perm[n_test:n_test + n_val]]
    train = data.edges[perm[n_test + n_val:]]
    if disjoint_train_ratio > 0:
        k = max(1, round(len(train) * disjoint_train_ratio))
        tp = rng.permutation(len(train))
        train_sup, train_mp = train[tp[:k]], train[tp[k:]]
    else:
        train_sup = train_mp = train
    all_pos = _pairs_set(data.edges)
    test_neg = sample_negatives(data.n, len(test), all_pos, rng)
    val_neg = sample_negatives(data.n, len(val), all_pos | _pairs_set(test_neg), rng)
    return LinkSplit(train_mp, train_sup, val, val_neg, test, test_neg)


# ---------------------------------------------------------------------- metrics & baselines


def architecture(cfg: GNNConfig, in_dim: int, extra: dict | None = None) -> dict:
    return {"model": "GraphSAGE", "conv": "SAGEConv (mean aggregation; edge weights not used)", "in_dim": in_dim,
            "hidden_dim": cfg.hidden_dim, "out_dim": cfg.out_dim, "num_layers": cfg.num_layers, "dropout": cfg.dropout,
            "activation": "ReLU", "optimizer": "Adam", "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
            **(extra or {})}


# ---------------------------------------------------------------------- metrics & baselines


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    k = int(y.sum())
    order = np.argsort(-s, kind="stable")
    return {
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "average_precision": round(float(average_precision_score(y, s)), 4),
        "precision_at_k": round(float(y[order[:k]].mean()), 4) if k else None,
        "k": k,
    }


HEURISTICS = {
    "adamic_adar": lambda G, p: [s for *_, s in nx.adamic_adar_index(G, p)],
    "jaccard": lambda G, p: [s for *_, s in nx.jaccard_coefficient(G, p)],
    "common_neighbours": lambda G, p: [len(list(nx.common_neighbors(G, u, v))) for u, v in p],
    "preferential_attachment": lambda G, p: [s for *_, s in nx.preferential_attachment(G, p)],
}


def heuristic_baselines(n_nodes: int, mp_edges: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(map(tuple, mp_edges.tolist()))
    pairs = [tuple(p) for p in np.vstack([pos, neg]).tolist()]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return {name: ranking_metrics(y, np.asarray(fn(G, pairs), dtype=float)) for name, fn in HEURISTICS.items()}


# ---------------------------------------------------------------------- node classification




def node_split(labels: dict[int, str], seed: int, val_fraction: float, test_fraction: float) -> tuple[list[int], list[int], list[int], list[str]]:
    counts: dict[str, list[int]] = {}
    for i, c in labels.items():
        counts.setdefault(c, []).append(i)
    usable = {c: v for c, v in counts.items() if len(v) >= MIN_PER_CLASS}
    if len(usable) < 2 or sum(len(v) for v in usable.values()) < MIN_LABELLED:
        raise GNNDataError(
            f"Node classification needs researcher-assigned code categories: at least 2 categories with ≥{MIN_PER_CLASS} codes "
            f"each and ≥{MIN_LABELLED} labelled codes in total. Found: "
            + (", ".join(f"{c} ({len(v)})" for c, v in sorted(counts.items())) or "no categorised codes")
            + ". Assign categories to codes under Database → Thematic analysis; labels are never generated automatically.")
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c, members in sorted(usable.items()):
        m = list(rng.permutation(members))
        n_te = max(1, round(len(m) * test_fraction))
        n_va = max(1, round(len(m) * val_fraction))
        te += m[:n_te]; va += m[n_te:n_te + n_va]; tr += m[n_te + n_va:]
    return [int(i) for i in tr], [int(i) for i in va], [int(i) for i in te], sorted(usable)


def class_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(classes))), zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=list(range(len(classes))), zero_division=0)), 4),
        "per_class": [{"label": c, "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                       "support": int(s[i])} for i, c in enumerate(classes)],
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()},
    }


def classification_result(data: GraphData, labels: dict[int, str], split: tuple[list[int], list[int], list[int]], classes: list[str],
                          yt: np.ndarray, pred: np.ndarray, proba: np.ndarray, history: list[dict], best_epoch: int, epochs: int,
                          runtime: float, state: dict, device: str, seed: int) -> dict:
    """One node-classification repeat: GNN metrics per split and the baselines scored on the same test codes."""
    tr, va, te = split
    # baselines on the same split: majority class, and logistic regression on node features without the graph
    from sklearn.linear_model import LogisticRegression

    maj = int(np.bincount(yt[tr]).argmax())
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(data.x[tr], yt[tr])
    # graph only: each test node takes the category with the largest edge weight among its training-labelled neighbours
    votes = np.zeros((data.n, len(classes)))
    train_set = set(tr)
    for (u, v), w in zip(data.edges.tolist(), data.weights.tolist()):
        if v in train_set:
            votes[u, yt[v]] += w
        if u in train_set:
            votes[v, yt[u]] += w
    neighbour = [int(votes[i].argmax()) if votes[i].any() else maj for i in te]
    return {
        "seed": seed, "classes": classes,
        "split": {"train": len(tr), "val": len(va), "test": len(te), "labelled": len(labels), "split_rule": "stratified by category"},
        "history": history, "best_epoch": best_epoch, "epochs_run": len(history), "early_stopped": len(history) < epochs,
        "runtime_s": round(runtime, 3),
        "train": class_metrics(yt[tr].tolist(), pred[tr].tolist(), classes),
        "validation": class_metrics(yt[va].tolist(), pred[va].tolist(), classes),
        "test": class_metrics(yt[te].tolist(), pred[te].tolist(), classes),
        "baselines_test": {
            "majority_class": class_metrics(yt[te].tolist(), [maj] * len(te), classes),
            "logistic_regression_features_only": class_metrics(yt[te].tolist(), lr.predict(data.x[te]).tolist(), classes),
            "neighbour_vote_graph_only": class_metrics(yt[te].tolist(), neighbour, classes),
        },
        "predictions": [{"index": int(i), "predicted": classes[int(pred[i])], "confidence": round(float(proba[i].max()), 4),
                         "true": labels.get(int(i)), "split": "train" if i in tr else "val" if i in va else "test" if i in te else "unlabelled"}
                        for i in range(data.n)],
        "state": state,
        "device": device,
    }
