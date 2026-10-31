"""Community detection on code graphs."""
from __future__ import annotations

import networkx as nx


def leiden_available() -> bool:
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError:
        return False
    return True


def detect_communities(G: nx.Graph, method: str = "louvain", resolution: float = 1.0, seed: int = 42) -> tuple[dict, dict]:
    """Return (node -> community id, info). Isolated nodes get singleton communities.

    Community ids are ordered by community size (0 = largest).
    """
    info: dict = {"method": method, "resolution": resolution, "seed": seed}
    if G.number_of_nodes() == 0:
        return {}, {**info, "n_communities": 0, "modularity": None}
    H = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
    groups: list[set] = []
    if H.number_of_edges() > 0:
        if method == "leiden":
            if not leiden_available():
                raise RuntimeError("Leiden requires the optional packages python-igraph and leidenalg (pip install thematicml[leiden]).")
            import igraph as ig
            import leidenalg

            nodes = list(H.nodes())
            pos = {n: i for i, n in enumerate(nodes)}
            g = ig.Graph(n=len(nodes), edges=[(pos[a], pos[b]) for a, b in H.edges()])
            g.es["weight"] = [H[a][b].get("weight", 1.0) for a, b in H.edges()]
            part = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                resolution_parameter=resolution, seed=seed,
            )
            groups = [{nodes[i] for i in comm} for comm in part]
        elif method == "greedy":
            groups = [set(c) for c in nx.community.greedy_modularity_communities(H, weight="weight", resolution=resolution)]
        elif method == "label_propagation":
            groups = [set(c) for c in nx.community.asyn_lpa_communities(H, weight="weight", seed=seed)]
        else:
            groups = [set(c) for c in nx.community.louvain_communities(H, weight="weight", resolution=resolution, seed=seed)]
    modularity = nx.community.modularity(H, groups, weight="weight") if groups else None
    isolated = [n for n in G if G.degree(n) == 0]
    groups = sorted(groups, key=lambda s: (-len(s), sorted(s)[0])) + [{n} for n in sorted(isolated)]
    membership = {n: i for i, comm in enumerate(groups) for n in comm}
    info.update(
        {
            "n_communities": len(groups),
            "n_multi_node_communities": sum(1 for g in groups if len(g) > 1),
            "modularity": None if modularity is None else round(float(modularity), 4),
            "sizes": [len(g) for g in groups],
        }
    )
    return membership, info
