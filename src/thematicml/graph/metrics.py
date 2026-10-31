"""Network statistics. Centrality describes structural position in the code
graph; it is not evidence that a concept is the most important finding."""
from __future__ import annotations

import networkx as nx


def node_metrics(G: nx.Graph) -> dict[str, dict]:
    if G.number_of_nodes() == 0:
        return {}
    # betweenness treats weights as distances, so invert co-occurrence strength
    D = G.copy()
    for a, b, d in D.edges(data=True):
        d["distance"] = 1.0 / d.get("weight", 1.0)
    betw = nx.betweenness_centrality(D, weight="distance", normalized=True)
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {n: 1.0 / G.number_of_nodes() for n in G}
    out = {}
    for n in G:
        out[n] = {
            "degree": int(G.degree(n)),
            "weighted_degree": float(G.degree(n, weight="weight")),
            "betweenness": round(float(betw[n]), 5),
            "pagerank": round(float(pr[n]), 5),
        }
    return out


def graph_metrics(G: nx.Graph, membership: dict) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = list(nx.connected_components(G)) if n else []
    cross = sum(1 for a, b in G.edges() if membership.get(a) != membership.get(b))
    return {
        "nodes": n,
        "edges": m,
        "density": round(nx.density(G), 5) if n > 1 else 0.0,
        "average_degree": round(2 * m / n, 3) if n else 0.0,
        "connected_components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "isolated_nodes": sum(1 for x in G if G.degree(x) == 0),
        "cross_community_edges": cross,
        "average_clustering": round(nx.average_clustering(G, weight="weight"), 4) if n else 0.0,
    }
