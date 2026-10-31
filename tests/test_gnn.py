"""GNN tests. Graphs here are SYNTHETIC test fixtures with planted structure.

They verify the mechanics (conversion, splits, leakage rules, training,
checkpoints, inference). They say nothing about performance on real data.
"""
import json

import numpy as np
import pytest

from thematicml.config import GNNConfig

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from thematicml.gnn import core  # noqa: E402


def planted_graph(seed=0, groups=("alpha", "beta", "gamma"), size=12, p_in=0.5, p_out=0.03):
    """Stochastic block model; labels share a prefix within a group (SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    nodes, members = [], []
    for g in groups:
        for i in range(size):
            nid = 100 + len(nodes)
            nodes.append({"id": nid, "label": f"{g} topic {i}", "frequency": int(rng.integers(3, 40)), "method": "tfidf", "category": g})
            members.append(g)
    edges = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            if rng.random() < (p_in if members[a] == members[b] else p_out):
                edges.append({"source": nodes[a]["id"], "target": nodes[b]["id"], "type": "cooccurrence", "weight": int(rng.integers(2, 9))})
    return nodes, edges


FAST = dict(epochs=120, patience=40, repeats=1, hidden_dim=32, out_dim=16)


def test_conversion_ids_types_and_dedup():
    nodes, edges = planted_graph()
    extra = [{"source": edges[0]["target"], "target": edges[0]["source"], "type": "cooccurrence"},  # reverse duplicate
             {"source": nodes[0]["id"], "target": nodes[0]["id"], "type": "cooccurrence"},  # self loop
             {"source": nodes[0]["id"], "target": nodes[5]["id"], "type": "similarity", "weight": 0.9},  # other edge type
             {"source": 999999, "target": nodes[1]["id"], "type": "cooccurrence"}]  # unknown node
    d = core.build_graph_data(nodes, edges + extra)
    assert d.node_ids == [n["id"] for n in nodes]
    assert len(d.edges) == len(edges)
    assert (d.edges[:, 0] < d.edges[:, 1]).all()
    assert d.info["dropped_edges"] == {"non_cooccurrence": 1, "self_loops": 1, "unknown_nodes": 1, "duplicates": 1}
    assert d.x.shape[0] == len(nodes) and d.x.dtype == np.float32
    assert core.build_graph_data(nodes, edges).version == d.version  # dropped edges do not change the version
    assert core.build_graph_data(nodes, edges[:-1]).version != d.version


def test_empty_and_malformed_graphs():
    with pytest.raises(core.GNNDataError):
        core.build_graph_data([], [])
    with pytest.raises(core.GNNDataError):
        core.build_graph_data([{"id": 1, "label": "a"}, {"id": 1, "label": "b"}], [])
    tiny = core.build_graph_data([{"id": i, "label": f"c{i}"} for i in range(4)], [{"source": 0, "target": 1}])
    with pytest.raises(core.GNNDataError, match="at least"):
        core.link_split(tiny, 0)


def test_link_split_is_disjoint_and_leak_free():
    nodes, edges = planted_graph()
    d = core.build_graph_data(nodes, edges)
    s = core.link_split(d, seed=3, disjoint_train_ratio=0.3)
    sets = {k: core._pairs_set(getattr(s, k)) for k in ("train_mp", "train_sup", "val", "test", "val_neg", "test_neg")}
    pos = core._pairs_set(d.edges)
    # positives partition the edge set
    assert sets["train_mp"] | sets["train_sup"] | sets["val"] | sets["test"] == pos
    assert not (sets["test"] & (sets["train_mp"] | sets["train_sup"] | sets["val"]))
    assert not (sets["val"] & (sets["train_mp"] | sets["train_sup"]))
    assert not (sets["train_mp"] & sets["train_sup"])
    # negatives are true non-edges, disjoint from each other
    assert not ((sets["val_neg"] | sets["test_neg"]) & pos)
    assert not (sets["val_neg"] & sets["test_neg"])
    assert len(s.test_neg) == len(s.test) and len(s.val_neg) == len(s.val)
    # same seed -> same split
    assert core._pairs_set(core.link_split(d, seed=3, disjoint_train_ratio=0.3).test) == sets["test"]
    # default (PyG-style): training edges are both message passing and supervision; held-out sets stay excluded
    s0 = core.link_split(d, seed=3)
    assert core._pairs_set(s0.train_mp) == core._pairs_set(s0.train_sup) == core._pairs_set(s0.train)
    assert not (core._pairs_set(s0.train) & (core._pairs_set(s0.val) | core._pairs_set(s0.test)))


def test_dense_graph_is_refused():
    nodes = [{"id": i, "label": f"code {i}"} for i in range(10)]
    edges = [{"source": a, "target": b} for a in range(10) for b in range(a + 1, 10) if (a + b) % 5]  # ~78% dense
    d = core.build_graph_data(nodes, edges)
    with pytest.raises(core.GNNDataError, match="too dense"):
        core.link_split(d, 0)


def test_negative_sampling_never_returns_positives():
    rng = np.random.default_rng(0)
    forbidden = {(0, 1), (0, 2), (1, 2)}
    neg = core.sample_negatives(5, 7, forbidden, rng)
    assert len(neg) == 7 and not (core._pairs_set(neg) & forbidden)
    with pytest.raises(core.GNNDataError, match="dense"):
        core.sample_negatives(4, 10, forbidden, rng)


def test_graphsage_link_prediction_learns_planted_structure():
    nodes, edges = planted_graph()
    d = core.build_graph_data(nodes, edges)
    runs = [core.train_link_predictor(d, GNNConfig(**FAST), seed=s) for s in (1, 2, 3)]
    res = runs[0]
    assert res["device"] == "cpu" and res["epochs_run"] >= 1
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    # planted communities are learnable; a 36-node graph has only ~16 test edges, so average over seeds
    assert np.mean([r["test"]["roc_auc"] for r in runs]) > 0.75
    assert set(res["baselines_test"]) == {"adamic_adar", "jaccard", "common_neighbours", "preferential_attachment"}
    assert res["split"]["test_pos"] == res["split"]["test_neg"]


def test_test_edges_not_in_message_passing(monkeypatch):
    """Spy on every edge_index the model sees and check held-out edges are absent."""
    nodes, edges = planted_graph()
    d = core.build_graph_data(nodes, edges)
    split = core.link_split(d, seed=5)
    held_test = core._pairs_set(split.test)
    held_val = core._pairs_set(split.val)
    seen = []
    orig = core.GraphSAGE.forward

    def spy(self, x, ei):
        seen.append({(min(a, b), max(a, b)) for a, b in ei.t().tolist()})
        return orig(self, x, ei)

    monkeypatch.setattr(core.GraphSAGE, "forward", spy)
    core.train_link_predictor(d, GNNConfig(epochs=5, patience=10, repeats=1), seed=5)
    assert seen
    assert all(not (s & held_test) for s in seen)  # test edges never enter message passing
    with_val = [s for s in seen if s & held_val]
    assert len(with_val) == 1  # validation edges are added only for the single final test pass
    assert with_val[0] == core._pairs_set(split.train) | held_val


def test_node_classification_with_labels_and_without():
    nodes, edges = planted_graph()
    d = core.build_graph_data(nodes, edges)
    labels = {i: n["category"] for i, n in enumerate(nodes)}
    res = core.train_node_classifier(d, labels, GNNConfig(task="node_classification", **FAST), seed=0)
    assert res["classes"] == ["alpha", "beta", "gamma"]
    assert res["test"]["macro_f1"] >= 0.6
    assert {"majority_class", "logistic_regression_features_only", "neighbour_vote_graph_only"} == set(res["baselines_test"])
    assert res["baselines_test"]["neighbour_vote_graph_only"]["macro_f1"] >= 0.6  # planted communities follow categories
    assert sum(r["support"] for r in res["test"]["per_class"]) == res["split"]["test"]
    with pytest.raises(core.GNNDataError, match="categories"):
        core.train_node_classifier(d, {0: "alpha", 1: "beta"}, GNNConfig(task="node_classification"), seed=0)


def test_checkpoint_roundtrip_and_inference(tmp_path):
    nodes, edges = planted_graph()
    d = core.build_graph_data(nodes, edges)
    cfg = GNNConfig(**FAST)
    res = core.train_link_predictor(d, cfg, seed=2)
    p = tmp_path / "m.pt"
    core.save_checkpoint(p, {"state": res["state"], "meta": json.dumps({"x": 1})})
    ck = core.load_checkpoint(p)
    a = core.predict_links(res["state"], d, cfg, 10)
    b = core.predict_links(ck["state"], d, cfg, 10)
    assert a == b and len(a) == 10
    observed = core._pairs_set(d.edges)
    assert all((p["source_index"], p["target_index"]) not in observed for p in a)  # predictions are never observed edges
    assert [p["rank"] for p in a] == list(range(1, 11))


def test_missing_dependency_is_reported(monkeypatch):
    import builtins

    from thematicml.graph import gnn as gmod

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("torch_geometric"):
            raise ImportError("no pyg")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    st = gmod.gnn_status()
    assert st["available"] is False and "pip install" in st["reason"]
    with pytest.raises(gmod.OptionalDependencyError):
        gmod.require_gnn()


def test_experiment_persistence_and_rescore(tmp_path, sample_csv):
    """End to end on the synthetic sample CSV fixture: train, persist, reload checkpoint."""
    from thematicml import AnalysisConfig, Project
    from thematicml.gnn import service

    p = Project("gnn", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", sentiment=False, temporal=False, n_codes=30))
    exp = res.train_gnn(GNNConfig(epochs=40, patience=20, repeats=2, top_k_predictions=10))
    assert exp["status"] == "completed", exp["error"]
    m = exp["metrics"]
    assert m["repeats"] == 2 and m["gnn"]["test"]["roc_auc"]["n"] == 2
    assert set(exp["baselines"]) >= {"adamic_adar", "preferential_attachment"}
    assert exp["graph_version"] and exp["versions"]["torch"]
    assert len(exp["history"]) >= 1 and exp["device"] == "cpu"
    preds = service.predictions(p.store, exp["id"])
    assert len(preds) == 10 and all(r["kind"] == "link" for r in preds)
    ckpt = service.checkpoint_file(p.store, exp)
    assert ckpt.exists() and p.store.root.resolve() in ckpt.parents
    again = service.rescore(p.store, exp["id"], top_k=10)
    assert again["graph_changed"] is False
    assert [x["score"] for x in again["predictions"]] == [r["score"] for r in preds]
    # reopening the workspace keeps the experiment
    from thematicml.storage import Store

    assert service.get_experiment(Store(tmp_path), exp["id"])["status"] == "completed"
    # tampered checkpoint path is refused
    p.store.update("gnn_experiments", {"id": exp["id"]}, {"checkpoint_path": "../../outside.pt"})
    with pytest.raises(ValueError, match="outside"):
        service.checkpoint_file(p.store, service.get_experiment(p.store, exp["id"]))
    # node classification without researcher categories is refused with a reason, not faked
    bad = res.train_gnn(GNNConfig(task="node_classification", epochs=5))
    assert bad["status"] == "failed" and "categor" in bad["error"]


def test_node_classification_on_a_run_mixing_categorised_rules_and_tfidf_codes(tmp_path, sample_csv):
    """Uncategorised TF-IDF codes read back from storage must count as unlabelled, not as a NaN category."""
    from thematicml import AnalysisConfig, Project
    from thematicml.config import CodebookRule
    from thematicml.gnn import service

    rules = {"Usability": ["menu*", "navigat*", "search*"], "Performance": ["slow*", "freez*", "load*", "crash*"],
             "Value": ["price*", "expensive", "support*"]}
    codebook = [CodebookRule(label=kw.rstrip("*").title(), keywords=[kw], category=cat) for cat, kws in rules.items() for kw in kws]
    p = Project("gnn-nc", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", sentiment=False, temporal=False,
                                   coding_method="both", codebook=codebook, n_codes=10, cooccurrence_level="segment"))
    # the API serialises with strict JSON: a NaN category or note would make these pages fail
    import json

    from thematicml.reporting import build_dashboard
    from thematicml.results import code_table

    rows = code_table(p.store, res.run_id)
    json.dumps(rows, allow_nan=False)
    json.dumps(build_dashboard(p.store, res.run_id), allow_nan=False, default=str)
    assert {r["category"] for r in rows} <= {None, *rules}
    nodes, _, _ = service.graph_inputs(p.store, res.run_id, GNNConfig(task="node_classification"))
    assert {n["category"] for n in nodes} <= {None, *rules}
    assert any(n["category"] is None for n in nodes)  # the TF-IDF codes
    v = service.validate(p.store, res.run_id, GNNConfig(task="node_classification"))
    assert set(v["categories"]) <= set(rules)
    exp = res.train_gnn(GNNConfig(task="node_classification", epochs=10, patience=5, repeats=1))
    assert exp["status"] == "completed", exp["error"]
    assert "neighbour_vote_graph_only" in exp["baselines"]


def test_gnn_http_api(tmp_path, sample_csv, monkeypatch):
    import importlib
    import time

    from fastapi.testclient import TestClient

    monkeypatch.setenv("THEMATICML_HOME", str(tmp_path / "ws"))
    import thematicml_api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        st = c.get("/api/gnn/status").json()
        assert st["available"] and st["model"] == "GraphSAGE" and "torch" in st["versions"]
        pid = c.post("/api/projects", json={"name": "g"}).json()["id"]
        with open(sample_csv, "rb") as fh:
            did = c.post(f"/api/projects/{pid}/uploads", files=[("files", ("s.csv", fh, "text/csv"))]).json()[0]["id"]
        c.post(f"/api/datasets/{did}/mapping", json={"text_col": "response_text", "id_col": "respondent_id"})
        cfg = {**c.get("/api/config/defaults").json()["config"], "sentiment": False, "n_codes": 30,
               "gnn": {"epochs": 30, "patience": 15, "repeats": 1, "top_k_predictions": 5}}
        rid = c.post(f"/api/projects/{pid}/runs", json={"dataset_ids": [did], "config": cfg}).json()["id"]

        def wait(pred, timeout=180):
            t0 = time.time()
            while time.time() - t0 < timeout:
                v = pred()
                if v:
                    return v
                time.sleep(0.3)
            raise TimeoutError

        # the experiment requested at upload time is queued after the run completes
        auto = wait(lambda: [e for e in c.get(f"/api/runs/{rid}/gnn").json() if e["status"] in ("completed", "failed")])
        assert auto[0]["status"] == "completed", auto[0]["error"]
        v = c.post(f"/api/runs/{rid}/gnn/validate", json={"task": "node_classification"}).json()
        assert v["ok"] is False and v["labels_required"] and v["reasons"]
        assert c.post(f"/api/runs/{rid}/gnn", json={"task": "node_classification"}).status_code == 400
        assert c.post(f"/api/runs/{rid}/gnn", json={"epochs": 0}).status_code == 422
        eid = c.post(f"/api/runs/{rid}/gnn", json={"epochs": 20, "patience": 10, "repeats": 1, "decoder": "mlp", "top_k_predictions": 5}).json()["id"]
        exp = wait(lambda: (e := c.get(f"/api/gnn/{eid}").json())["status"] in ("completed", "failed") and e)
        assert exp["status"] == "completed", exp["error"]
        assert len(c.get(f"/api/gnn/{eid}/predictions").json()) == 5
        assert c.get(f"/api/gnn/{eid}/embeddings").json()["points"]
        assert c.post(f"/api/gnn/{eid}/rescore?top_k=5").json()["graph_changed"] is False
        csv = c.get(f"/api/gnn/{eid}/export")
        assert csv.status_code == 200 and "source_label" in csv.text
        assert c.get(f"/api/gnn/{eid}/export?what=experiment").json()["id"] == eid
        assert len(c.get(f"/api/projects/{pid}/gnn").json()) == 2
        assert c.delete(f"/api/gnn/{eid}").status_code == 200
        assert c.get(f"/api/gnn/{eid}").status_code == 404


def test_verdict_is_paired_over_repeats():
    from thematicml.gnn.service import _link_summary

    def rep(g, b):
        m = {"roc_auc": g, "average_precision": g, "precision_at_k": g}
        return {"train": m, "validation": m, "test": m, "split": {}, "best_epoch": 1, "epochs_run": 1, "runtime_s": 0.0,
                "baselines_test": {"adamic_adar": {"roc_auc": b, "average_precision": b, "precision_at_k": b}}}

    # the GNN loses by 0.03 on every split while both vary a lot between splits: unpaired SDs overlap, the paired test does not
    results = [rep(b - 0.03, b) for b in (0.70, 0.80, 0.90, 0.75, 0.85)]
    metrics, _ = _link_summary(results)
    assert metrics["verdict"].startswith("GNN lower") and "0 of 5" in metrics["verdict"]
    assert metrics["paired_comparison"]["mean_difference"] == -0.03
    mixed = [rep(b + d, b) for b, d in ((0.7, 0.02), (0.8, -0.02), (0.9, 0.01), (0.75, -0.01))]
    assert _link_summary(mixed)[0]["verdict"].startswith("No clear difference")
