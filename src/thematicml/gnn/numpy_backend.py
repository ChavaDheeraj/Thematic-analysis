"""GraphSAGE on the code co-occurrence graph in NumPy: the model of :mod:`thematicml.gnn.core` without PyTorch.

Used when PyTorch Geometric is not installed, which includes the browser edition (PyTorch does not run in a
browser). A code graph has tens to a few hundred nodes, so full-batch training with dense weights, a sparse
mean-aggregation matrix and hand-written gradients is fast.

The architecture, initialisation, optimiser, early stopping, splits, negative sampling and leakage rules
follow the PyTorch backend:

* each layer is PyG's ``SAGEConv`` with mean aggregation: ``W_neigh · mean(neighbours) + b + W_self · self``,
  ReLU and dropout between layers, no activation after the last;
* weights and biases start uniform in ±1/√fan_in, as ``torch.nn.Linear`` and PyG's ``Linear`` do;
* Adam with L2 weight decay added to the gradient (``torch.optim.Adam``), binary cross-entropy on logits for
  links and class-weighted cross-entropy for categories.

Random draws come from NumPy rather than torch, so for the same seed the two backends give similar but not
identical numbers. Gradients are checked against finite differences in ``tests/test_gnn_numpy.py``.
"""
from __future__ import annotations

import io
import json
import math
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score

from thematicml.config import GNNConfig
from thematicml.gnn.common import (  # noqa: F401  (re-exported: callers use backend.<name>)
    HEURISTICS,
    GNNDataError,
    GraphData,
    LinkSplit,
    _pairs_set,
    architecture,
    build_graph_data,
    class_metrics,
    classification_result,
    heuristic_baselines,
    link_split,
    node_features,
    node_split,
    ranking_metrics,
    sample_negatives,
)

BACKEND = "numpy"
CHECKPOINT_SUFFIX = ".npz"
DEVICE = "cpu (NumPy)"


def _check_device(pref: str) -> None:
    if pref == "cuda":
        raise GNNDataError("CUDA needs the PyTorch backend (pip install thematicml[gnn]); the built-in NumPy GraphSAGE runs on the CPU.")


# ---------------------------------------------------------------------- model


def mean_operator(n: int, pairs: np.ndarray) -> sp.csr_matrix:
    """Row-normalised adjacency of the undirected edges ``pairs``: row i averages i's neighbours (zeros if none)."""
    if len(pairs) == 0:
        return sp.csr_matrix((n, n))
    u, v = pairs[:, 0], pairs[:, 1]
    a = sp.csr_matrix((np.ones(2 * len(pairs)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    deg = np.asarray(a.sum(axis=1)).ravel()
    return sp.diags(np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)) @ a


def _uniform(rng: np.random.Generator, fan_in: int, shape: tuple[int, ...]) -> np.ndarray:
    bound = 1.0 / math.sqrt(fan_in)
    return rng.uniform(-bound, bound, size=shape)


def init_encoder(rng: np.random.Generator, in_dim: int, cfg: GNNConfig) -> dict[str, np.ndarray]:
    dims = [in_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.out_dim]
    p = {}
    for i in range(cfg.num_layers):
        a, b = dims[i], dims[i + 1]
        p[f"conv{i}.W_neigh"] = _uniform(rng, a, (a, b))
        p[f"conv{i}.b"] = _uniform(rng, a, (b,))
        p[f"conv{i}.W_self"] = _uniform(rng, a, (a, b))
    return p


def init_decoder(rng: np.random.Generator, dim: int, kind: str) -> dict[str, np.ndarray]:
    if kind != "mlp":
        return {}
    return {"mlp.W1": _uniform(rng, dim, (dim, dim)), "mlp.b1": _uniform(rng, dim, (dim,)),
            "mlp.W2": _uniform(rng, dim, (dim, 1)), "mlp.b2": _uniform(rng, dim, (1,))}


def encode(p: dict, x: np.ndarray, P: sp.csr_matrix, layers: int, dropout: float = 0.0, rng: np.random.Generator | None = None):
    """Node embeddings and the values the backward pass needs. Dropout only when ``rng`` is given (training)."""
    cache, h = [], x
    for i in range(layers):
        m = P @ h
        z = m @ p[f"conv{i}.W_neigh"] + p[f"conv{i}.b"] + h @ p[f"conv{i}.W_self"]
        mask = None
        if i < layers - 1:
            a = np.maximum(z, 0.0)
            if rng is not None and dropout > 0:
                mask = (rng.random(a.shape) >= dropout) / (1.0 - dropout)
                a = a * mask
            cache.append((h, m, z, mask))
            h = a
        else:
            cache.append((h, m, z, None))
            h = z
    return h, cache


def encode_backward(p: dict, P: sp.csr_matrix, cache: list, dh: np.ndarray, grads: dict) -> None:
    for i in reversed(range(len(cache))):
        h_in, m, z, mask = cache[i]
        dz = dh if i == len(cache) - 1 else dh * (z > 0) * (1.0 if mask is None else mask)
        grads[f"conv{i}.W_neigh"] = m.T @ dz
        grads[f"conv{i}.b"] = dz.sum(axis=0)
        grads[f"conv{i}.W_self"] = h_in.T @ dz
        dh = P.T @ (dz @ p[f"conv{i}.W_neigh"].T) + dz @ p[f"conv{i}.W_self"].T


def decode(p: dict, z: np.ndarray, pairs: np.ndarray, kind: str):
    """Link logits for code pairs; symmetric in (u, v)."""
    zu, zv = z[pairs[:, 0]], z[pairs[:, 1]]
    h = zu * zv
    if kind != "mlp":
        return h.sum(axis=1), (zu, zv, h, None)
    a = h @ p["mlp.W1"] + p["mlp.b1"]
    r = np.maximum(a, 0.0)
    return (r @ p["mlp.W2"] + p["mlp.b2"])[:, 0], (zu, zv, h, (a, r))


def decode_backward(p: dict, pairs: np.ndarray, cache: tuple, ds: np.ndarray, grads: dict, dz: np.ndarray) -> None:
    zu, zv, h, mlp = cache
    if mlp is None:
        dh = np.repeat(ds[:, None], h.shape[1], axis=1)
    else:
        a, r = mlp
        grads["mlp.W2"] = r.T @ ds[:, None]
        grads["mlp.b2"] = np.array([ds.sum()])
        da = (ds[:, None] @ p["mlp.W2"].T) * (a > 0)
        grads["mlp.W1"] = h.T @ da
        grads["mlp.b1"] = da.sum(axis=0)
        dh = da @ p["mlp.W1"].T
    np.add.at(dz, pairs[:, 0], dh * zv)
    np.add.at(dz, pairs[:, 1], dh * zu)


def bce_with_logits(s: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean binary cross-entropy on logits and its gradient."""
    loss = np.maximum(s, 0) - s * y + np.log1p(np.exp(-np.abs(s)))
    return float(loss.mean()), (1.0 / (1.0 + np.exp(-s)) - y) / len(s)


def weighted_cross_entropy(o: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """``torch.nn.functional.cross_entropy(o, y, weight=w)``: weighted mean of the negative log-likelihood, and its gradient."""
    shifted = o - o.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    wy = w[y]
    loss = float(-(wy * logp[np.arange(len(y)), y]).sum() / wy.sum())
    g = np.exp(logp)
    g[np.arange(len(y)), y] -= 1.0
    return loss, g * (wy / wy.sum())[:, None]


class Adam:
    """``torch.optim.Adam`` with L2 weight decay (added to the gradient, applied to every parameter)."""

    def __init__(self, params: dict, lr: float, weight_decay: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, self.wd, (self.b1, self.b2), self.eps, self.t = lr, weight_decay, betas, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, p in params.items():
            g = grads[k] + self.wd * p
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def _copy(p: dict) -> dict:
    return {k: v.copy() for k, v in p.items()}


def _split_state(p: dict) -> tuple[dict, dict]:
    return {k: v for k, v in p.items() if k.startswith("conv")}, {k: v for k, v in p.items() if not k.startswith("conv")}


# ---------------------------------------------------------------------- link prediction


def train_link_predictor(data: GraphData, cfg: GNNConfig, seed: int, on_epoch=None) -> dict:
    _check_device(cfg.device)
    split = link_split(data, seed, cfg.val_fraction, cfg.test_fraction, cfg.disjoint_train_ratio)
    rng = np.random.default_rng(seed + 10_000)
    drop = np.random.default_rng(seed + 30_000)
    x = data.x.astype(np.float64)
    P_train = mean_operator(data.n, split.train_mp)
    P_val = mean_operator(data.n, split.train)
    P_test = mean_operator(data.n, np.vstack([split.train, split.val]))
    sup = split.train_sup
    held_out_neg = _pairs_set(split.val_neg) | _pairs_set(split.test_neg)
    train_forbidden = _pairs_set(split.train) | held_out_neg
    # one negative per supervision edge, capped by the unlinked pairs that remain available for training
    n_train_neg = min(len(sup), data.n * (data.n - 1) // 2 - len(train_forbidden))

    init = np.random.default_rng(seed)
    params = {**init_encoder(init, x.shape[1], cfg), **init_decoder(init, cfg.out_dim, cfg.decoder)}
    opt = Adam(params, cfg.learning_rate, cfg.weight_decay)

    def scores(P, pos, neg, p=params):
        z, _ = encode(p, x, P, cfg.num_layers)
        s, _ = decode(p, z, np.vstack([pos, neg]), cfg.decoder)
        return s, np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    history, best = [], {"auc": -math.inf, "epoch": 0, "state": None}
    t0, bad = time.perf_counter(), 0
    for epoch in range(1, cfg.epochs + 1):
        z, cache = encode(params, x, P_train, cfg.num_layers, cfg.dropout, drop)
        neg = sample_negatives(data.n, n_train_neg, train_forbidden, rng)
        pairs = np.vstack([sup, neg])
        s, dcache = decode(params, z, pairs, cfg.decoder)
        loss, ds = bce_with_logits(s, np.r_[np.ones(len(sup)), np.zeros(len(neg))])
        grads, dz = {}, np.zeros_like(z)
        decode_backward(params, pairs, dcache, ds, grads, dz)
        encode_backward(params, P_train, cache, dz, grads)
        opt.step(params, grads)
        vl, vy = scores(P_val, split.val, split.val_neg)
        val_loss, _ = bce_with_logits(vl, vy)
        val_auc = float(roc_auc_score(vy, vl))
        history.append({"epoch": epoch, "train_loss": round(loss, 5), "val_loss": round(val_loss, 5), "val_auc": round(val_auc, 5)})
        if on_epoch:
            on_epoch(epoch, history[-1])
        if val_auc > best["auc"] + 1e-4:
            best = {"auc": val_auc, "epoch": epoch, "state": _copy(params)}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    runtime = time.perf_counter() - t0
    chosen = best["state"]
    tl, ty = scores(P_test, split.test, split.test_neg, chosen)
    vl, vy = scores(P_val, split.val, split.val_neg, chosen)
    tr_neg = sample_negatives(data.n, n_train_neg, train_forbidden, np.random.default_rng(seed + 20_000))
    trl, tr_y = scores(P_train, sup, tr_neg, chosen)
    encoder, decoder = _split_state(chosen)
    return {
        "seed": seed,
        "split": split.sizes(),
        "history": history,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "early_stopped": len(history) < cfg.epochs,
        "runtime_s": round(runtime, 3),
        "train": ranking_metrics(tr_y, trl),
        "validation": ranking_metrics(vy, vl),
        "test": ranking_metrics(ty, tl),
        "baselines_test": heuristic_baselines(data.n, np.vstack([split.train, split.val]), split.test, split.test_neg),
        "state": {"encoder": encoder, "decoder": decoder},
        "device": DEVICE,
    }


def embed(state: dict, data: GraphData, cfg: GNNConfig) -> np.ndarray:
    """Node embeddings using every observed edge for message passing."""
    z, _ = encode(state["encoder"], data.x.astype(np.float64), mean_operator(data.n, data.edges), cfg.num_layers)
    return z


def predict_links(state: dict, data: GraphData, cfg: GNNConfig, top_k: int) -> list[dict]:
    """Score every unobserved pair; return the top_k as *predicted* (not observed) relationships."""
    if top_k <= 0 or data.n < 2:
        return []
    z = embed(state, data, cfg)
    observed = _pairs_set(data.edges)
    cand = np.array([(u, v) for u in range(data.n) for v in range(u + 1, data.n) if (u, v) not in observed], dtype=np.int64).reshape(-1, 2)
    if not len(cand):
        return []
    logits, _ = decode(state["decoder"], z, cand, cfg.decoder)
    s = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(-s, kind="stable")[:top_k]
    G = nx.Graph()
    G.add_nodes_from(range(data.n))
    G.add_edges_from(map(tuple, data.edges.tolist()))
    chosen = [tuple(cand[i]) for i in order]
    aa = {(u, v): sc for u, v, sc in nx.adamic_adar_index(G, chosen)}
    return [{"source_index": int(cand[i][0]), "target_index": int(cand[i][1]), "score": round(float(s[i]), 5), "rank": r + 1,
             "adamic_adar": round(float(aa[tuple(cand[i])]), 4)} for r, i in enumerate(order)]


# ---------------------------------------------------------------------- node classification


def init_head(rng: np.random.Generator, dim: int, n_classes: int) -> dict[str, np.ndarray]:
    return {"head.W": _uniform(rng, dim, (dim, n_classes)), "head.b": _uniform(rng, dim, (n_classes,))}


def classify(p: dict, x: np.ndarray, P: sp.csr_matrix, cfg: GNNConfig, rng: np.random.Generator | None = None):
    """Class logits: a linear head on the ReLU of the encoder output, as ``core.NodeClassifier``."""
    e, cache = encode(p, x, P, cfg.num_layers, cfg.dropout, rng)
    r = np.maximum(e, 0.0)
    return r @ p["head.W"] + p["head.b"], (e, r, cache)


def train_node_classifier(data: GraphData, labels: dict[int, str], cfg: GNNConfig, seed: int, on_epoch=None) -> dict:
    _check_device(cfg.device)
    tr, va, te, classes = node_split(labels, seed, cfg.val_fraction, cfg.test_fraction)
    cidx = {c: i for i, c in enumerate(classes)}
    yt = np.full(data.n, -1, dtype=np.int64)
    for i, c in labels.items():
        if c in cidx:
            yt[i] = cidx[c]
    x = data.x.astype(np.float64)
    P = mean_operator(data.n, data.edges)  # labels are not part of the edges or features
    counts = np.bincount([cidx[labels[i]] for i in tr], minlength=len(classes)).astype(float)
    weight = counts.sum() / (len(classes) * np.maximum(counts, 1))
    init = np.random.default_rng(seed)
    params = {**init_encoder(init, x.shape[1], cfg), **init_head(init, cfg.out_dim, len(classes))}
    opt = Adam(params, cfg.learning_rate, cfg.weight_decay)
    drop = np.random.default_rng(seed + 30_000)
    tr_a, va_a, te_a = (np.asarray(v, dtype=np.int64) for v in (tr, va, te))
    history, best, bad = [], {"score": -math.inf, "epoch": 0, "state": None}, 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        o, (e, r, cache) = classify(params, x, P, cfg, drop)
        loss, g = weighted_cross_entropy(o[tr_a], yt[tr_a], weight)
        do = np.zeros_like(o)
        do[tr_a] = g
        grads = {"head.W": r.T @ do, "head.b": do.sum(axis=0)}
        encode_backward(params, P, cache, (do @ params["head.W"].T) * (e > 0), grads)
        opt.step(params, grads)
        o, _ = classify(params, x, P, cfg)
        vloss, _ = weighted_cross_entropy(o[va_a], yt[va_a], weight)
        vf1 = f1_score(yt[va_a].tolist(), o[va_a].argmax(axis=1).tolist(), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": round(loss, 5), "val_loss": round(vloss, 5), "val_macro_f1": round(float(vf1), 5)})
        if on_epoch:
            on_epoch(epoch, history[-1])
        score = vf1 - 1e-3 * vloss  # macro-F1, ties broken by lower loss
        if score > best["score"] + 1e-6:
            best, bad = {"score": score, "epoch": epoch, "state": _copy(params)}, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    o, _ = classify(best["state"], x, P, cfg)
    pred = o.argmax(axis=1)
    ex = np.exp(o - o.max(axis=1, keepdims=True))
    proba = ex / ex.sum(axis=1, keepdims=True)
    return classification_result(data, labels, (tr, va, te), classes, yt, pred, proba, history, best["epoch"], cfg.epochs,
                                 time.perf_counter() - t0, {"classifier": best["state"], "classes": classes}, DEVICE, seed)


def classifier_embeddings(state: dict, data: GraphData, cfg: GNNConfig) -> np.ndarray:
    """Encoder output of a trained node classifier, using every observed edge."""
    z, _ = encode(state["classifier"], data.x.astype(np.float64), mean_operator(data.n, data.edges), cfg.num_layers)
    return z


# ---------------------------------------------------------------------- checkpoints


def save_checkpoint(path, state: dict) -> None:
    """Arrays only (no pickle), so loading a checkpoint cannot run code."""
    flat = {f"{part}/{k}": v for part in ("encoder", "decoder", "classifier") for k, v in state["state"].get(part, {}).items()}
    if "classes" in state["state"]:
        flat["classes"] = np.array(state["state"]["classes"], dtype=str)
    flat["meta"] = np.array(state["meta"], dtype=str)
    buf = io.BytesIO()
    np.savez_compressed(buf, **flat)
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())


def load_checkpoint(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        state: dict = {}
        for key in z.files:
            if "/" in key:
                part, name = key.split("/", 1)
                state.setdefault(part, {})[name] = z[key]
        if "encoder" in state:
            state.setdefault("decoder", {})  # the dot-product decoder has no weights, so none were saved
        if "classes" in z.files:
            state["classes"] = [str(c) for c in z["classes"]]
        meta = str(z["meta"])
    json.loads(meta)  # refuse a checkpoint whose metadata is not JSON
    return {"state": state, "meta": meta}
