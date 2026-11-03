"""SYNTHETIC graph fixtures shared by the GNN tests of both backends."""
import numpy as np


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
