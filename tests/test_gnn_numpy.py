"""The built-in NumPy GraphSAGE (used without PyTorch, and in the browser edition).

Graphs here are SYNTHETIC test fixtures with planted structure, as in test_gnn.py. These tests run with or
without PyTorch installed: they check the gradients, that training learns planted structure, the leakage
rules, checkpoints, and the service end to end with the NumPy backend forced.
"""
import json

import numpy as np
import pytest

from thematicml.config import GNNConfig
from thematicml.gnn import numpy_backend as nb

from gnn_fixtures import FAST, planted_graph


@pytest.fixture
def numpy_only(monkeypatch):
    monkeypatch.setenv("THEMATICML_GNN_BACKEND", "numpy")


def _numeric_grad(f, params, key, eps=1e-6):
    g = np.zeros_like(params[key])
    it = np.nditer(params[key], flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = params[key][i]
        params[key][i] = old + eps
        up = f()
        params[key][i] = old - eps
        down = f()
        params[key][i] = old
        g[i] = (up - down) / (2 * eps)
    return g


@pytest.mark.parametrize("decoder,layers", [("dot", 2), ("mlp", 2), ("dot", 3)])
def test_link_gradients_match_finite_differences(decoder, layers):
    rng = np.random.default_rng(0)
    n, d = 7, 5
    cfg = GNNConfig(hidden_dim=4, out_dim=3, num_layers=layers, decoder=decoder, dropout=0.0)
    x = rng.normal(size=(n, d))
    P = nb.mean_operator(n, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5]]))  # node 6 has no neighbours
    params = {**nb.init_encoder(rng, d, cfg), **nb.init_decoder(rng, cfg.out_dim, decoder)}
    pairs = np.array([[0, 2], [1, 3], [4, 6], [5, 2]])
    y = np.array([1.0, 0.0, 1.0, 0.0])

    def loss():
        z, _ = nb.encode(params, x, P, layers)
        return nb.bce_with_logits(nb.decode(params, z, pairs, decoder)[0], y)[0]

    z, cache = nb.encode(params, x, P, layers)
    s, dcache = nb.decode(params, z, pairs, decoder)
    _, ds = nb.bce_with_logits(s, y)
    grads, dz = {}, np.zeros_like(z)
    nb.decode_backward(params, pairs, dcache, ds, grads, dz)
    nb.encode_backward(params, P, cache, dz, grads)
    for key in params:
        np.testing.assert_allclose(grads[key], _numeric_grad(loss, params, key), rtol=1e-4, atol=1e-7, err_msg=key)


def test_classifier_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    n, d = 8, 4
    cfg = GNNConfig(hidden_dim=5, out_dim=3, num_layers=2, dropout=0.0)
    x = rng.normal(size=(n, d))
    P = nb.mean_operator(n, np.array([[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [2, 6]]))
    params = {**nb.init_encoder(rng, d, cfg), **nb.init_head(rng, cfg.out_dim, 3)}
    idx, y, w = np.array([0, 2, 3, 5, 7]), np.array([0, 1, 2, 1, 0]), np.array([0.7, 1.4, 2.0])

    def loss():
        return nb.weighted_cross_entropy(nb.classify(params, x, P, cfg)[0][idx], y, w)[0]

    o, (e, r, cache) = nb.classify(params, x, P, cfg)
    _, g = nb.weighted_cross_entropy(o[idx], y, w)
    do = np.zeros_like(o)
    do[idx] = g
    grads = {"head.W": r.T @ do, "head.b": do.sum(axis=0)}
    nb.encode_backward(params, P, cache, (do @ params["head.W"].T) * (e > 0), grads)
    for key in params:
        np.testing.assert_allclose(grads[key], _numeric_grad(loss, params, key), rtol=1e-4, atol=1e-7, err_msg=key)


def test_weighted_cross_entropy_matches_the_definition():
    o = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    y, w = np.array([0, 2]), np.array([1.0, 2.0, 3.0])
    p = np.exp(o) / np.exp(o).sum(1, keepdims=True)
    expected = -(1.0 * np.log(p[0, 0]) + 3.0 * np.log(p[1, 2])) / 4.0
    assert nb.weighted_cross_entropy(o, y, w)[0] == pytest.approx(expected)


def test_link_prediction_learns_planted_structure():
    d = nb.build_graph_data(*planted_graph())
    runs = [nb.train_link_predictor(d, GNNConfig(**FAST), seed=s) for s in (1, 2, 3)]
    res = runs[0]
    assert res["device"] == nb.DEVICE and res["epochs_run"] >= 1
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    assert np.mean([r["test"]["roc_auc"] for r in runs]) > 0.75
    assert set(res["baselines_test"]) == {"adamic_adar", "jaccard", "common_neighbours", "preferential_attachment"}
    assert res["split"]["test_pos"] == res["split"]["test_neg"]
    # same seed, same numbers
    assert nb.train_link_predictor(d, GNNConfig(**FAST), seed=1)["test"] == res["test"]


def test_held_out_edges_never_enter_message_passing(monkeypatch):
    d = nb.build_graph_data(*planted_graph())
    split = nb.link_split(d, seed=5)
    held_test, held_val = nb._pairs_set(split.test), nb._pairs_set(split.val)
    seen = []
    orig = nb.mean_operator

    def spy(n, pairs):
        seen.append(nb._pairs_set(pairs))
        return orig(n, pairs)

    monkeypatch.setattr(nb, "mean_operator", spy)
    nb.train_link_predictor(d, GNNConfig(epochs=5, patience=10, repeats=1), seed=5)
    assert seen and all(not (s & held_test) for s in seen[:2])  # training and validation graphs
    assert seen[1] == nb._pairs_set(split.train) and not (seen[0] & held_val)
    assert seen[2] == nb._pairs_set(split.train) | held_val  # validation edges join only for the test pass


def test_node_classification_learns_categories():
    nodes, edges = planted_graph()
    d = nb.build_graph_data(nodes, edges)
    labels = {i: n["category"] for i, n in enumerate(nodes)}
    res = nb.train_node_classifier(d, labels, GNNConfig(task="node_classification", **FAST), seed=0)
    assert res["classes"] == ["alpha", "beta", "gamma"]
    assert res["test"]["macro_f1"] >= 0.6
    assert {"majority_class", "logistic_regression_features_only", "neighbour_vote_graph_only"} == set(res["baselines_test"])
    with pytest.raises(nb.GNNDataError, match="categories"):
        nb.train_node_classifier(d, {0: "alpha", 1: "beta"}, GNNConfig(task="node_classification"), seed=0)


def test_cuda_is_refused_with_a_reason():
    d = nb.build_graph_data(*planted_graph())
    with pytest.raises(nb.GNNDataError, match="PyTorch"):
        nb.train_link_predictor(d, GNNConfig(device="cuda", **FAST), seed=0)


def test_checkpoint_roundtrip_without_pickle(tmp_path):
    d = nb.build_graph_data(*planted_graph())
    cfg = GNNConfig(**{**FAST, "decoder": "mlp"})
    res = nb.train_link_predictor(d, cfg, seed=2)
    p = tmp_path / "m.npz"
    nb.save_checkpoint(p, {"state": res["state"], "meta": json.dumps({"x": 1})})
    ck = nb.load_checkpoint(p)
    assert json.loads(ck["meta"]) == {"x": 1}
    a, b = nb.predict_links(res["state"], d, cfg, 10), nb.predict_links(ck["state"], d, cfg, 10)
    assert a == b and [x["rank"] for x in a] == list(range(1, 11))
    assert all((x["source_index"], x["target_index"]) not in nb._pairs_set(d.edges) for x in a)
    with np.load(p, allow_pickle=False) as z:  # nothing in the file needs unpickling
        assert all(z[k].dtype != object for k in z.files)


def test_status_and_backend_choice(numpy_only):
    from thematicml.gnn import backend, gnn_status

    st = gnn_status()
    assert st["available"] and st["backend"] == "numpy" and st["devices"] == ["cpu"] and "pip install" in st["note"]
    assert backend() is nb


def test_service_end_to_end_with_numpy(numpy_only, tmp_path, sample_csv):
    from thematicml import AnalysisConfig, Project
    from thematicml.gnn import service

    p = Project("gnn-np", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", sentiment=False, temporal=False, n_codes=30))
    exp = res.train_gnn(GNNConfig(epochs=40, patience=20, repeats=2, top_k_predictions=10))
    assert exp["status"] == "completed", exp["error"]
    assert exp["device"] == nb.DEVICE and exp["versions"]["gnn_backend"] == "numpy"
    assert exp["metrics"]["gnn"]["test"]["roc_auc"]["n"] == 2 and exp["metrics"]["verdict"]
    ckpt = service.checkpoint_file(p.store, exp)
    assert ckpt.suffix == ".npz" and p.store.root.resolve() in ckpt.parents
    preds = service.predictions(p.store, exp["id"])
    again = service.rescore(p.store, exp["id"], top_k=10)
    assert again["graph_changed"] is False and [x["score"] for x in again["predictions"]] == [r["score"] for r in preds]
    assert service.embeddings(p.store, exp["id"])["points"]
    bad = res.train_gnn(GNNConfig(task="node_classification", epochs=5))
    assert bad["status"] == "failed" and "categor" in bad["error"]


def test_a_pytorch_checkpoint_without_pytorch_is_explained(numpy_only, tmp_path, monkeypatch):
    from thematicml.gnn import service
    from thematicml.graph import gnn as gmod

    monkeypatch.setattr(gmod, "_torch", lambda: None)
    p = tmp_path / "model.pt"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="trained with PyTorch"):
        service.checkpoint_backend(p)
