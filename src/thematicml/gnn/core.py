"""GraphSAGE on the code co-occurrence graph (PyTorch Geometric backend).

Neural and optional. Used when ``torch`` and ``torch_geometric`` are installed; otherwise
:mod:`thematicml.gnn.numpy_backend` runs the same model. Import through :func:`thematicml.gnn.backend`.

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
import math
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import SAGEConv

from thematicml.config import GNNConfig
from thematicml.gnn.common import (  # noqa: F401  (re-exported: callers use core.<name>)
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

BACKEND = "pytorch"
CHECKPOINT_SUFFIX = ".pt"


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
    return classification_result(data, labels, (tr, va, te), classes, yt, pred, proba, history, best["epoch"], cfg.epochs,
                                 time.perf_counter() - t0, {"classifier": best["state"], "classes": classes}, str(dev), seed)


def classifier_embeddings(state: dict, data: GraphData, cfg: GNNConfig) -> np.ndarray:
    """Encoder output of a trained node classifier, using every observed edge."""
    model = NodeClassifier(data.x.shape[1], cfg, len(state["classes"]))
    model.load_state_dict(state["classifier"])
    model.eval()
    with torch.no_grad():
        return model.encoder(torch.as_tensor(data.x), _edge_index(data.edges, torch.device("cpu"))).numpy()


# ---------------------------------------------------------------------- checkpoints


def save_checkpoint(path, state: dict) -> None:
    torch.save(state, path)


def load_checkpoint(path) -> dict:
    # weights_only=True refuses to unpickle arbitrary objects
    return torch.load(path, map_location="cpu", weights_only=True)
