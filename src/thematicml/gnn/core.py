"""GraphSAGE on the code co-occurrence graph (PyTorch Geometric).

Neural and optional. Import this module only after
:func:`thematicml.graph.gnn.require_gnn` has succeeded.

Leakage rules this module follows
---------------------------------
* The graph is homogeneous: nodes are codes, edges are *observed* co-occurrence
  edges only. Similarity edges are excluded from the input because they are
  computed from the same segments that define co-occurrence.
* Node features never use corpus-derived text vectors (they would encode
  co-occurrence). They are hashed character n-grams of the code label, plus
  log frequency and coding method, or a one-hot identity.
* Link prediction: held-out validation and test edges are removed from the
  message-passing graph. Validation scores use training edges only; test
  scores use training + validation edges. Negatives never include a known
  edge, and validation/test negatives are fixed per split and never trained on.
* The checkpoint is selected on validation AUC; the test split is evaluated
  once per repeat, after selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.nn import SAGEConv

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


def device_for(pref: str) -> torch.device:
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise GNNDataError("CUDA was requested but no CUDA device is available; use device 'cpu' or 'auto'.")
        return torch.device("cuda")
    if pref == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _edge_index(pairs: np.ndarray, device) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    t = torch.as_tensor(pairs, dtype=torch.long, device=device).t()
    return torch.cat([t, t.flip(0)], dim=1)  # undirected: both directions


# ---------------------------------------------------------------------- model


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out]
        self.convs = torch.nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(layers))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LinkDecoder(torch.nn.Module):
    def __init__(self, dim: int, kind: str):
        super().__init__()
        self.kind = kind
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, 1)) if kind == "mlp" else None

    def forward(self, z, pairs: torch.Tensor):
        h = z[pairs[:, 0]] * z[pairs[:, 1]]  # symmetric in (u, v)
        return self.mlp(h).squeeze(-1) if self.mlp is not None else h.sum(-1)


class NodeClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, cfg: GNNConfig, n_classes: int):
        super().__init__()
        self.encoder = GraphSAGE(in_dim, cfg.hidden_dim, cfg.out_dim, cfg.num_layers, cfg.dropout)
        self.head = torch.nn.Linear(cfg.out_dim, n_classes)

    def forward(self, x, edge_index):
        return self.head(F.relu(self.encoder(x, edge_index)))


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


# ---------------------------------------------------------------------- link prediction


def train_link_predictor(data: GraphData, cfg: GNNConfig, seed: int, on_epoch=None) -> dict:
    seed_all(seed)
    dev = device_for(cfg.device)
    split = link_split(data, seed, cfg.val_fraction, cfg.test_fraction, cfg.disjoint_train_ratio)
    rng = np.random.default_rng(seed + 10_000)
    x = torch.as_tensor(data.x, device=dev)
    mp_train = _edge_index(split.train_mp, dev)
    mp_val = _edge_index(split.train, dev)
    mp_test = _edge_index(np.vstack([split.train, split.val]), dev)
    sup = torch.as_tensor(split.train_sup, dtype=torch.long, device=dev)
    held_out_neg = _pairs_set(split.val_neg) | _pairs_set(split.test_neg)
    train_forbidden = _pairs_set(split.train) | held_out_neg
    # one negative per supervision edge, capped by the unlinked pairs that remain available for training
    n_train_neg = min(len(split.train_sup), data.n * (data.n - 1) // 2 - len(train_forbidden))

    model = GraphSAGE(x.shape[1], cfg.hidden_dim, cfg.out_dim, cfg.num_layers, cfg.dropout).to(dev)
    dec = LinkDecoder(cfg.out_dim, cfg.decoder).to(dev)
    params = list(model.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def scores(mp, pos, neg):
        z = model(x, mp)
        p = torch.as_tensor(np.vstack([pos, neg]), dtype=torch.long, device=dev)
        logits = dec(z, p)
        y = torch.cat([torch.ones(len(pos), device=dev), torch.zeros(len(neg), device=dev)])
        return logits, y

    history, best = [], {"auc": -math.inf, "epoch": 0, "state": None}
    t0, bad = time.perf_counter(), 0
    for epoch in range(1, cfg.epochs + 1):
        model.train(); dec.train()
        opt.zero_grad()
        z = model(x, mp_train)
        neg = torch.as_tensor(sample_negatives(data.n, n_train_neg, train_forbidden, rng), dtype=torch.long, device=dev)
        logits = torch.cat([dec(z, sup), dec(z, neg)])
        y = torch.cat([torch.ones(len(sup), device=dev), torch.zeros(len(neg), device=dev)])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        model.eval(); dec.eval()
        with torch.no_grad():
            vl, vy = scores(mp_val, split.val, split.val_neg)
            val_loss = float(F.binary_cross_entropy_with_logits(vl, vy))
            val_auc = float(roc_auc_score(vy.cpu().numpy(), vl.cpu().numpy()))
        history.append({"epoch": epoch, "train_loss": round(loss.item(), 5), "val_loss": round(val_loss, 5), "val_auc": round(val_auc, 5)})
        if on_epoch:
            on_epoch(epoch, history[-1])
        if val_auc > best["auc"] + 1e-4:
            best = {"auc": val_auc, "epoch": epoch, "state": (copy.deepcopy(model.state_dict()), copy.deepcopy(dec.state_dict()))}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    runtime = time.perf_counter() - t0
    model.load_state_dict(best["state"][0]); dec.load_state_dict(best["state"][1])
    model.eval(); dec.eval()
    with torch.no_grad():
        tl, ty = scores(mp_test, split.test, split.test_neg)
        vl, vy = scores(mp_val, split.val, split.val_neg)
        z_tr = model(x, mp_train)
        tr_neg = sample_negatives(data.n, n_train_neg, train_forbidden, np.random.default_rng(seed + 20_000))
        trl = torch.cat([dec(z_tr, sup), dec(z_tr, torch.as_tensor(tr_neg, dtype=torch.long, device=dev))])
    tr_y = np.r_[np.ones(len(split.train_sup)), np.zeros(len(tr_neg))]
    return {
        "seed": seed,
        "split": split.sizes(),
        "history": history,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "early_stopped": len(history) < cfg.epochs,
        "runtime_s": round(runtime, 3),
        "train": ranking_metrics(tr_y, trl.cpu().numpy()),
        "validation": ranking_metrics(vy.cpu().numpy(), vl.cpu().numpy()),
        "test": ranking_metrics(ty.cpu().numpy(), tl.cpu().numpy()),
        "baselines_test": heuristic_baselines(data.n, np.vstack([split.train, split.val]), split.test, split.test_neg),
        "state": {"encoder": best["state"][0], "decoder": best["state"][1]},
        "device": str(dev),
    }


def embed(state: dict, data: GraphData, cfg: GNNConfig) -> np.ndarray:
    """Node embeddings using every observed edge for message passing."""
    dev = torch.device("cpu")
    model = GraphSAGE(data.x.shape[1], cfg.hidden_dim, cfg.out_dim, cfg.num_layers, cfg.dropout)
    model.load_state_dict(state["encoder"])
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(data.x), _edge_index(data.edges, dev)).numpy()


def predict_links(state: dict, data: GraphData, cfg: GNNConfig, top_k: int) -> list[dict]:
    """Score every unobserved pair; return the top_k as *predicted* (not observed) relationships."""
    if top_k <= 0 or data.n < 2:
        return []
    z = torch.as_tensor(embed(state, data, cfg))
    dec = LinkDecoder(cfg.out_dim, cfg.decoder)
    dec.load_state_dict(state["decoder"])
    dec.eval()
    observed = _pairs_set(data.edges)
    cand = np.array([(u, v) for u in range(data.n) for v in range(u + 1, data.n) if (u, v) not in observed], dtype=np.int64).reshape(-1, 2)
    if not len(cand):
        return []
    with torch.no_grad():
        s = torch.sigmoid(dec(z, torch.as_tensor(cand))).numpy()
    order = np.argsort(-s, kind="stable")[:top_k]
    G = nx.Graph()
    G.add_nodes_from(range(data.n))
    G.add_edges_from(map(tuple, data.edges.tolist()))
    chosen = [tuple(cand[i]) for i in order]
    aa = {(u, v): sc for u, v, sc in nx.adamic_adar_index(G, chosen)}
    return [{"source_index": int(cand[i][0]), "target_index": int(cand[i][1]), "score": round(float(s[i]), 5), "rank": r + 1,
             "adamic_adar": round(float(aa[tuple(cand[i])]), 4)} for r, i in enumerate(order)]


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


def train_node_classifier(data: GraphData, labels: dict[int, str], cfg: GNNConfig, seed: int, on_epoch=None) -> dict:
    seed_all(seed)
    dev = device_for(cfg.device)
    tr, va, te, classes = node_split(labels, seed, cfg.val_fraction, cfg.test_fraction)
    cidx = {c: i for i, c in enumerate(classes)}
    y_all = torch.full((data.n,), -1, dtype=torch.long)
    for i, c in labels.items():
        if c in cidx:
            y_all[i] = cidx[c]
    y_all = y_all.to(dev)
    x = torch.as_tensor(data.x, device=dev)
    ei = _edge_index(data.edges, dev)  # labels are not part of the edges or features
    counts = np.bincount([cidx[labels[i]] for i in tr], minlength=len(classes)).astype(float)
    weight = torch.as_tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    model = NodeClassifier(x.shape[1], cfg, len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    tr_t, va_t, te_t = (torch.as_tensor(v, dtype=torch.long, device=dev) for v in (tr, va, te))
    history, best, bad = [], {"score": -math.inf, "epoch": 0, "state": None}, 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[tr_t], y_all[tr_t], weight=weight)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
            vloss = float(F.cross_entropy(out[va_t], y_all[va_t], weight=weight))
            vpred = out[va_t].argmax(-1).cpu().tolist()
        vf1 = f1_score(y_all[va_t].cpu().tolist(), vpred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": round(loss.item(), 5), "val_loss": round(vloss, 5), "val_macro_f1": round(float(vf1), 5)})
        if on_epoch:
            on_epoch(epoch, history[-1])
        score = vf1 - 1e-3 * vloss  # macro-F1, ties broken by lower loss
        if score > best["score"] + 1e-6:
            best, bad = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        out = model(x, ei)
        pred = out.argmax(-1).cpu().numpy()
        proba = F.softmax(out, -1).cpu().numpy()
    yt = y_all.cpu().numpy()
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
        "history": history, "best_epoch": best["epoch"], "epochs_run": len(history), "early_stopped": len(history) < cfg.epochs,
        "runtime_s": round(time.perf_counter() - t0, 3),
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
        "state": {"classifier": best["state"], "classes": classes},
        "device": str(dev),
    }


# ---------------------------------------------------------------------- checkpoints


def save_checkpoint(path, state: dict) -> None:
    torch.save(state, path)


def load_checkpoint(path) -> dict:
    # weights_only=True refuses to unpickle arbitrary objects
    return torch.load(path, map_location="cpu", weights_only=True)
