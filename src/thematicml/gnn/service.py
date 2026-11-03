"""GNN experiments: create, validate, train, persist and query.

This module has no hard dependency on PyTorch; the training code in
:mod:`thematicml.gnn.core` is imported only when an experiment runs.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from thematicml.config import GNNConfig
from thematicml.gnn import backend
from thematicml.graph.gnn import OptionalDependencyError, gnn_status, require_gnn
from thematicml.storage.database import Store, new_id, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS gnn_experiments (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, project_id TEXT,
    task TEXT NOT NULL, status TEXT NOT NULL, config TEXT, graph_version TEXT, graph_info TEXT,
    progress REAL DEFAULT 0, history TEXT, metrics TEXT, baselines TEXT, repeats TEXT, device TEXT, versions TEXT,
    checkpoint_path TEXT, error TEXT, created_at TEXT, started_at TEXT, finished_at TEXT, duration_s REAL
);
CREATE TABLE IF NOT EXISTS gnn_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL REFERENCES gnn_experiments(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, source_code_id INTEGER, target_code_id INTEGER, source_label TEXT, target_label TEXT,
    score REAL, rank INTEGER, adamic_adar REAL, predicted_class TEXT, true_class TEXT, split TEXT,
    graph_version TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_gnn_pred ON gnn_predictions(experiment_id, rank);
"""


def ensure_schema(store: Store) -> None:
    with store.connect() as con:
        con.executescript(SCHEMA)


def software_versions() -> dict:
    out = {"python": platform.python_version(), "gnn_backend": gnn_status()["backend"], "numpy": np.__version__}
    try:
        import torch
        import torch_geometric

        out.update({"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                    "cuda_available": bool(torch.cuda.is_available())})
    except ImportError:
        pass
    import sklearn

    out["scikit_learn"] = sklearn.__version__
    return out


# ---------------------------------------------------------------------- graph from a run


def graph_inputs(store: Store, run_id: str, cfg: GNNConfig) -> tuple[list[dict], list[dict], dict]:
    """Nodes and observed co-occurrence edges from the run's *current* (researcher-edited) codes."""
    from thematicml.results import graph_view

    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    g = graph_view(store, run_id, level=cfg.cooccurrence_level, min_cooccurrence=cfg.min_cooccurrence,
                   similarity=False, min_npmi=cfg.min_npmi)
    nodes = [{"id": n["id"], "label": n["label"], "frequency": n["frequency"], "method": n["method"], "category": n.get("category")}
             for n in sorted(g["nodes"], key=lambda n: n["id"])]
    edges = [e for e in g["edges"] if e["type"] == "cooccurrence"]
    params = {"level": g["params"]["level"], "min_cooccurrence": g["params"]["min_cooccurrence"], "min_npmi": g["params"]["min_npmi"]}
    return nodes, edges, params


def validate(store: Store, run_id: str, cfg: GNNConfig) -> dict:
    """Explain whether the task can run on this run's graph, without training."""
    st = gnn_status()
    nodes, edges, params = graph_inputs(store, run_id, cfg)
    cats: dict[str, int] = {}
    for n in nodes:
        if n["category"]:
            cats[n["category"]] = cats.get(n["category"], 0) + 1
    out = {"gnn_installed": st["available"], "install_hint": st.get("reason"), "nodes": len(nodes), "edges": len(edges),
           "graph_params": params, "categories": cats, "task": cfg.task, "ok": True, "reasons": []}
    if not st["available"]:
        out["ok"] = False
        out["reasons"].append(st["reason"])
    if cfg.task == "link_prediction":
        from thematicml.gnn import MAX_DENSITY, MIN_EDGES, MIN_NODES

        n = len(nodes)
        density = len(edges) / (n * (n - 1) / 2) if n > 1 else 0.0
        out["density"] = round(density, 4)
        if n < MIN_NODES or len(edges) < MIN_EDGES:
            out["ok"] = False
            out["reasons"].append(f"Link prediction needs ≥{MIN_NODES} codes and ≥{MIN_EDGES} co-occurrence edges; this graph has "
                                  f"{n} and {len(edges)}.")
        elif density > MAX_DENSITY:
            out["ok"] = False
            out["reasons"].append(f"The graph is too dense ({density:.0%} of code pairs linked; limit {MAX_DENSITY:.0%}). Raise the minimum "
                                  "co-occurrence or minimum NPMI, or use segment-level co-occurrence.")
        out["labels_required"] = False
    else:
        from thematicml.gnn import MIN_LABELLED, MIN_PER_CLASS

        usable = {c: k for c, k in cats.items() if k >= MIN_PER_CLASS}
        out["labels_required"] = True
        if len(usable) < 2 or sum(usable.values()) < MIN_LABELLED:
            out["ok"] = False
            out["reasons"].append(f"Node classification needs ≥2 code categories with ≥{MIN_PER_CLASS} codes each (≥{MIN_LABELLED} "
                                  "labelled codes). Assign categories to codes on the Dashboard; labels are never generated.")
    return out


# ---------------------------------------------------------------------- lifecycle


def create_experiment(store: Store, run_id: str, cfg: GNNConfig) -> dict:
    ensure_schema(store)
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    if run["status"] != "completed":
        raise ValueError("GNN experiments need a completed analysis run.")
    require_gnn()
    eid = new_id("gnn")
    store.insert("gnn_experiments", {"id": eid, "run_id": run_id, "project_id": run["project_id"], "task": cfg.task, "status": "queued",
                                     "config": cfg.model_dump(), "created_at": now(), "versions": software_versions()})
    store.log(run["project_id"], "gnn_queued", {"experiment_id": eid, "run_id": run_id, "task": cfg.task})
    return get_experiment(store, eid)


def get_experiment(store: Store, eid: str) -> dict | None:
    ensure_schema(store)
    row = store.one("SELECT * FROM gnn_experiments WHERE id = ?", (eid,))
    if row:
        for k in ("history", "baselines", "repeats", "graph_info", "versions"):
            if isinstance(row.get(k), str):
                row[k] = json.loads(row[k])
    return row


def list_experiments(store: Store, run_id: str | None = None, project_id: str | None = None) -> list[dict]:
    ensure_schema(store)
    where, params = ("run_id = ?", (run_id,)) if run_id else ("project_id = ?", (project_id,))
    rows = store.query(f"SELECT id, run_id, project_id, task, status, config, graph_version, graph_info, progress, metrics, device, versions, "
                       f"checkpoint_path, error, created_at, started_at, finished_at, duration_s FROM gnn_experiments WHERE {where} ORDER BY created_at DESC", params)
    for r in rows:
        for k in ("graph_info", "versions"):
            if isinstance(r.get(k), str):
                r[k] = json.loads(r[k])
    return rows


def delete_experiment(store: Store, eid: str) -> None:
    import shutil

    exp = get_experiment(store, eid)
    if not exp:
        return
    if exp["status"] in ("queued", "running"):
        raise ValueError("Cannot delete an experiment that is in progress.")
    with store.connect() as con:
        con.execute("DELETE FROM gnn_experiments WHERE id = ?", (eid,))
    d = _exp_dir(store, exp, create=False)
    if d is not None:
        shutil.rmtree(d, ignore_errors=True)
    store.log(exp["project_id"], "gnn_deleted", {"experiment_id": eid})


def recover_interrupted(store: Store) -> int:
    ensure_schema(store)
    with store.connect() as con:
        return con.execute("UPDATE gnn_experiments SET status = 'interrupted', error = 'Application stopped during training; start a new experiment.' "
                           "WHERE status IN ('queued', 'running')").rowcount


def _exp_dir(store: Store, exp: dict, create: bool = True) -> Path | None:
    d = store.run_dir(exp["project_id"], exp["run_id"]) / "gnn" / exp["id"]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_file(store: Store, exp: dict) -> Path:
    """Resolve a stored checkpoint path and refuse anything outside the workspace."""
    rel = exp.get("checkpoint_path")
    if not rel:
        raise ValueError("This experiment has no checkpoint.")
    root = store.root.resolve()
    p = (root / rel).resolve()
    if root not in p.parents or p.suffix not in CHECKPOINT_BACKENDS:
        raise ValueError("Checkpoint path is outside the workspace or not a checkpoint file (.pt or .npz); refusing to load it.")
    if not p.exists():
        raise ValueError("Checkpoint file is missing.")
    return p


CHECKPOINT_BACKENDS = {".pt": "pytorch", ".npz": "numpy"}


def checkpoint_backend(path: Path):
    """The backend that wrote a checkpoint: PyTorch checkpoints need PyTorch to be read back."""
    from thematicml.graph.gnn import _torch

    name = CHECKPOINT_BACKENDS[path.suffix]
    if name == "pytorch" and _torch() is None:
        raise ValueError("This experiment was trained with PyTorch, which is not installed here, so its checkpoint cannot be "
                         "loaded. Start a new experiment to train it with the built-in GraphSAGE.")
    return backend(name)


def _agg(values: list[float | None]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0, "n": len(v),
            "min": round(float(min(v)), 4), "max": round(float(max(v)), 4)}


def execute_experiment(store: Store, eid: str) -> dict:
    for _ in experiment_steps(store, eid):
        pass
    return get_experiment(store, eid)


def experiment_steps(store: Store, eid: str):
    """Train an experiment, yielding after each repeat (the browser edition hands the page a turn there)."""
    exp = get_experiment(store, eid)
    if exp is None:
        raise KeyError(eid)
    cfg = GNNConfig(**exp["config"])
    t0 = time.perf_counter()
    store.update("gnn_experiments", {"id": eid}, {"status": "running", "started_at": now(), "error": None})
    try:
        core = backend()
        nodes, edges, params = graph_inputs(store, exp["run_id"], cfg)
        data = core.build_graph_data(nodes, edges, cfg.feature_set, params)
        store.update("gnn_experiments", {"id": eid}, {"graph_version": data.version,
                                                      "graph_info": {**data.info, "features": data.feature_description}})
        total = cfg.repeats * cfg.epochs
        done = [0]

        def tick(epoch, _row):
            done[0] += 1
            if epoch % 10 == 0:
                store.update("gnn_experiments", {"id": eid}, {"progress": round(min(0.99, done[0] / total), 3)})

        results = []
        for r in range(cfg.repeats):
            seed = cfg.seed + r
            if cfg.task == "link_prediction":
                res = core.train_link_predictor(data, cfg, seed, tick)
            else:
                labels = {i: n["category"] for i, n in enumerate(nodes) if n["category"]}
                res = core.train_node_classifier(data, labels, cfg, seed, tick)
            results.append(res)
            done[0] = (r + 1) * cfg.epochs
            store.update("gnn_experiments", {"id": eid}, {"progress": round(min(0.99, done[0] / total), 3)})
            yield r

        first = results[0]
        d = _exp_dir(store, exp)
        ckpt = d / f"model{core.CHECKPOINT_SUFFIX}"
        meta = {"config": cfg.model_dump(), "graph_version": data.version, "node_ids": data.node_ids,
                "feature_description": data.feature_description, "in_dim": int(data.x.shape[1]), "seed": first["seed"]}
        core.save_checkpoint(ckpt, {"state": first["state"], "meta": json.dumps(meta)})

        if cfg.task == "link_prediction":
            metrics, baselines = _link_summary(results)
            preds = core.predict_links(first["state"], data, cfg, cfg.top_k_predictions)
            emb = core.embed(first["state"], data, cfg)
            pred_rows = [{"experiment_id": eid, "kind": "link", "source_code_id": data.node_ids[p["source_index"]],
                          "target_code_id": data.node_ids[p["target_index"]], "source_label": data.labels[p["source_index"]],
                          "target_label": data.labels[p["target_index"]], "score": p["score"], "rank": p["rank"],
                          "adamic_adar": p["adamic_adar"], "graph_version": data.version, "created_at": now()} for p in preds]
        else:
            metrics, baselines = _clf_summary(results)
            pred_rows = [{"experiment_id": eid, "kind": "node_class", "source_code_id": data.node_ids[p["index"]],
                          "source_label": data.labels[p["index"]], "predicted_class": p["predicted"], "score": p["confidence"],
                          "true_class": p["true"], "split": p["split"], "graph_version": data.version, "created_at": now()}
                         for p in first["predictions"]]
            emb = core.classifier_embeddings(first["state"], data, cfg)
        _save_embeddings(d, data, emb, nodes)
        with store.connect() as con:
            con.execute("DELETE FROM gnn_predictions WHERE experiment_id = ?", (eid,))
            for row in pred_rows:
                store.insert("gnn_predictions", row, con)
        repeats = [{k: v for k, v in res.items() if k not in ("state", "history", "predictions")} for res in results]
        store.update("gnn_experiments", {"id": eid}, {
            "status": "completed", "progress": 1.0, "finished_at": now(), "duration_s": round(time.perf_counter() - t0, 3),
            "history": first["history"], "metrics": metrics, "baselines": baselines, "repeats": repeats,
            "device": first["device"], "checkpoint_path": str(ckpt.resolve().relative_to(store.root.resolve())),
        })
        store.log(exp["project_id"], "gnn_completed", {"experiment_id": eid, "task": cfg.task, "duration_s": round(time.perf_counter() - t0, 1)})
    except Exception as exc:  # record the failure on the experiment instead of losing it
        msg = str(exc) if isinstance(exc, (ValueError, OptionalDependencyError, RuntimeError)) else f"{type(exc).__name__}: {exc}"
        store.update("gnn_experiments", {"id": eid}, {"status": "failed", "error": msg, "finished_at": now(),
                                                      "duration_s": round(time.perf_counter() - t0, 3)})
        store.log(exp["project_id"], "gnn_failed", {"experiment_id": eid, "error": msg})


execute_experiment.steps = experiment_steps


def _link_summary(results: list[dict]) -> tuple[dict, dict]:
    metric_names = ("roc_auc", "average_precision", "precision_at_k")
    gnn = {split: {m: _agg([r[split][m] for r in results]) for m in metric_names} for split in ("train", "validation", "test")}
    names = results[0]["baselines_test"].keys()
    base = {b: {m: _agg([r["baselines_test"][b][m] for r in results]) for m in metric_names} for b in names}
    best_base = max(base, key=lambda b: base[b]["roc_auc"]["mean"] or -1)
    # baselines are scored on the same test edges in every repeat, so the comparison is paired over repeats
    paired = [r["test"]["roc_auc"] - r["baselines_test"][best_base]["roc_auc"] for r in results
              if r["test"]["roc_auc"] is not None and r["baselines_test"][best_base]["roc_auc"] is not None]
    comparison = None
    if len(paired) > 1:
        d, se = float(np.mean(paired)), float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
        higher = sum(x > 0 for x in paired)
        comparison = {"baseline": best_base, "metric": "roc_auc", "mean_difference": round(d, 4), "standard_error": round(se, 4),
                      "gnn_higher_in": higher, "repeats": len(paired),
                      "rule": "paired over repeats on the same test edges; a difference counts when mean ± 2 SE excludes 0"}
        tally = f"higher in {higher} of {len(paired)} repeats, mean ROC-AUC difference {d:+.3f}"
        verdict = (f"GNN higher than the best baseline on held-out edges ({tally})" if d - 2 * se > 0 else
                   f"GNN lower than the best baseline on held-out edges ({tally})" if d + 2 * se < 0 else
                   f"No clear difference between the GNN and the best baseline ({tally})")
    else:
        verdict = "Single repeat: no conclusion about which method is better."
    metrics = {
        "task": "link_prediction", "repeats": len(results), "gnn": gnn, "best_baseline": best_base, "verdict": verdict, "paired_comparison": comparison,
        "split": results[0]["split"], "best_epoch": [r["best_epoch"] for r in results], "epochs_run": [r["epochs_run"] for r in results],
        "training_runtime_s": round(sum(r["runtime_s"] for r in results), 3),
        "note": "train = supervision edges seen in training; validation selects the checkpoint; test = held-out edges evaluated once per repeat. "
                "Baselines are scored on the same test edges using the same message-passing graph (train + validation edges).",
    }
    return metrics, base


def _clf_summary(results: list[dict]) -> tuple[dict, dict]:
    names = ("accuracy", "macro_f1", "weighted_f1")
    gnn = {split: {m: _agg([r[split][m] for r in results]) for m in names} for split in ("train", "validation", "test")}
    base = {b: {m: _agg([r["baselines_test"][b][m] for r in results]) for m in names} for b in results[0]["baselines_test"]}
    best_base = max(base, key=lambda b: base[b]["macro_f1"]["mean"] or -1)
    paired = [r["test"]["macro_f1"] - r["baselines_test"][best_base]["macro_f1"] for r in results]  # same test nodes
    comparison, verdict = None, "Single repeat: no conclusion about which method is better."
    if len(paired) > 1:
        d, se = float(np.mean(paired)), float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
        higher = sum(x > 0 for x in paired)
        comparison = {"baseline": best_base, "metric": "macro_f1", "mean_difference": round(d, 4), "standard_error": round(se, 4),
                      "gnn_higher_in": higher, "repeats": len(paired),
                      "rule": "paired over repeats on the same test nodes; a difference counts when mean ± 2 SE excludes 0"}
        tally = f"higher in {higher} of {len(paired)} repeats, mean macro-F1 difference {d:+.3f}"
        verdict = (f"GNN higher than the best baseline on held-out codes ({tally})" if d - 2 * se > 0 else
                   f"GNN lower than the best baseline on held-out codes ({tally})" if d + 2 * se < 0 else
                   f"No clear difference between the GNN and the best baseline ({tally})")
    metrics = {"task": "node_classification", "repeats": len(results), "gnn": gnn, "classes": results[0]["classes"],
               "best_baseline": best_base, "verdict": verdict, "paired_comparison": comparison,
               "split": results[0]["split"], "test_detail_first_repeat": results[0]["test"],
               "best_epoch": [r["best_epoch"] for r in results], "training_runtime_s": round(sum(r["runtime_s"] for r in results), 3),
               "note": "Labels are researcher-assigned code categories. Splits are stratified by category; test is evaluated once per repeat."}
    return metrics, base


def _save_embeddings(d: Path, data, emb: np.ndarray, nodes: list[dict]) -> None:
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(emb) if emb.shape[0] >= 2 else np.zeros((emb.shape[0], 2))
    out = [{"id": nid, "label": lab, "x": round(float(a), 4), "y": round(float(b), 4), "frequency": n["frequency"], "category": n["category"]}
           for nid, lab, (a, b), n in zip(data.node_ids, data.labels, xy, nodes)]
    (d / "embeddings.json").write_text(json.dumps({"method": "PCA of final-layer GraphSAGE embeddings (all observed edges)", "points": out}), encoding="utf-8")


def embeddings(store: Store, eid: str) -> dict:
    exp = get_experiment(store, eid)
    if exp is None:
        raise KeyError(eid)
    p = _exp_dir(store, exp, create=False) / "embeddings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"points": []}


def predictions(store: Store, eid: str, limit: int = 200) -> list[dict]:
    ensure_schema(store)
    return store.query("SELECT * FROM gnn_predictions WHERE experiment_id = ? ORDER BY CASE WHEN rank IS NULL THEN 1 ELSE 0 END, rank, id LIMIT ?",
                       (eid, limit))


def rescore(store: Store, eid: str, top_k: int = 50) -> dict:
    """Load the saved checkpoint and score the run's *current* graph (inference)."""
    exp = get_experiment(store, eid)
    if exp is None:
        raise KeyError(eid)
    if exp["status"] != "completed":
        raise ValueError("Only completed experiments have a checkpoint.")
    if exp["task"] != "link_prediction":
        raise ValueError("Re-scoring is implemented for link prediction experiments.")
    path = checkpoint_file(store, exp)
    core = checkpoint_backend(path)
    ck = core.load_checkpoint(path)
    meta = json.loads(ck["meta"])
    cfg = GNNConfig(**meta["config"])
    nodes, edges, params = graph_inputs(store, exp["run_id"], cfg)
    data = core.build_graph_data(nodes, edges, cfg.feature_set, params)
    if cfg.feature_set == "identity" and data.node_ids != meta["node_ids"]:
        raise ValueError("This checkpoint uses identity features and the set of codes has changed; it cannot score the new graph.")
    if data.x.shape[1] != meta["in_dim"]:
        raise ValueError("Feature dimensions differ from the checkpoint (for example the set of coding methods changed).")
    preds = core.predict_links(ck["state"], data, cfg, top_k)
    return {"graph_version": data.version, "trained_on_graph_version": meta["graph_version"],
            "graph_changed": data.version != meta["graph_version"],
            "predictions": [{"source_code_id": data.node_ids[p["source_index"]], "target_code_id": data.node_ids[p["target_index"]],
                             "source_label": data.labels[p["source_index"]], "target_label": data.labels[p["target_index"]],
                             "score": p["score"], "rank": p["rank"], "adamic_adar": p["adamic_adar"]} for p in preds]}
