import networkx as nx
import pandas as pd
import pytest

from thematicml.graph import (
    build_code_graph,
    cooccurrence_edges,
    detect_communities,
    graph_metrics,
    leiden_available,
    link_prediction_baseline,
    node_metrics,
)


def test_cooccurrence_counts_and_npmi():
    units = {"a": {1, 2, 3}, "b": {1, 2}, "c": {3, 4}}
    edges = {(e["source"], e["target"]): e for e in cooccurrence_edges(units, 10, 1)}
    assert edges[("a", "b")]["weight"] == 2
    assert edges[("a", "b")]["jaccard"] == pytest.approx(2 / 3, abs=1e-4)
    assert edges[("a", "c")]["weight"] == 1
    assert ("b", "c") not in edges
    assert all(-1 <= e["npmi"] <= 1 for e in edges.values())
    assert cooccurrence_edges(units, 10, 2, None) and len(cooccurrence_edges(units, 10, 2, None)) == 1


def test_npmi_filter_drops_chance_pairs():
    # x and y each appear in half the units, overlapping exactly at chance level
    units = {"x": set(range(0, 50)), "y": set(range(25, 75))}
    assert cooccurrence_edges(units, 100, 1, None)
    assert cooccurrence_edges(units, 100, 1, 0.0) == []


def test_build_graph_record_level():
    segments = pd.DataFrame({"record_id": [1, 1, 2, 3], "record_key": ["A", "A", "B", "C"]})
    codes = pd.DataFrame({"code_key": ["x", "y", "z"], "label": ["X", "Y", "Z"], "method": "m", "category": None})
    assigns = pd.DataFrame({"code_key": ["x", "y", "x", "y", "z"], "segment_pos": [0, 1, 2, 2, 3]})
    G, typed = build_code_graph(codes, assigns, segments, None, "record", 1, False, 0.3, None)
    assert G["x"]["y"]["weight"] == 2  # records 1 and 2
    assert G.degree("z") == 0
    assert G.nodes["x"]["records"] == 2 and G.nodes["x"]["respondents"] == 2
    Gs, _ = build_code_graph(codes, assigns, segments, None, "segment", 1, False, 0.3, None)
    assert Gs["x"]["y"]["weight"] == 1  # only segment 2


def _two_cliques():
    G = nx.Graph()
    for grp in (["a", "b", "c", "d"], ["e", "f", "g", "h"]):
        for i, u in enumerate(grp):
            for v in grp[i + 1:]:
                G.add_edge(u, v, weight=5)
    G.add_edge("d", "e", weight=1)
    G.add_node("iso")
    return G


@pytest.mark.parametrize("method", ["louvain", "greedy", "label_propagation"] + (["leiden"] if leiden_available() else []))
def test_communities(method):
    G = _two_cliques()
    m, info = detect_communities(G, method, 1.0, 1)
    assert m["a"] == m["b"] == m["c"]
    assert m["f"] == m["g"] == m["h"]
    assert m["a"] != m["h"]
    assert info["n_multi_node_communities"] == 2
    assert info["sizes"][-1] == 1  # isolated node is a singleton
    assert info["modularity"] > 0.3


def test_metrics_bridge_has_high_betweenness():
    G = _two_cliques()
    m, _ = detect_communities(G, "louvain", 1.0, 1)
    nm = node_metrics(G)
    assert nm["d"]["betweenness"] > nm["a"]["betweenness"]
    gm = graph_metrics(G, m)
    assert gm["nodes"] == 9 and gm["isolated_nodes"] == 1 and gm["cross_community_edges"] == 1 and gm["connected_components"] == 2


def test_link_prediction_baseline():
    G = nx.connected_caveman_graph(4, 5)
    r = link_prediction_baseline(G, seed=1)
    assert r["available"] and r["roc_auc"]["adamic_adar"] > 0.5
    assert link_prediction_baseline(nx.path_graph(3))["available"] is False
