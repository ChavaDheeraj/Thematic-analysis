"""Build code-relationship graphs.

Two edge types are kept distinct:

* ``cooccurrence`` – observed: the number of records (or segments) in which
  both codes were assigned. Also stores Jaccard and NPMI. By default only
  pairs with NPMI > 0 (co-occurring more often than chance) are kept.
* ``similarity`` – representational: cosine similarity between the mean
  TF-IDF vectors of the segments assigned to each code.

Community detection and centrality use co-occurrence edges only.
"""
from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


def cooccurrence_edges(unit_sets: dict[str, set], n_units: int, min_count: int, min_npmi: float | None = None) -> list[dict]:
    """unit_sets maps code_key -> set of unit ids (records or segments)."""
    keys = sorted(unit_sets)
    # invert for efficiency: unit -> codes
    by_unit: dict = {}
    for k in keys:
        for u in unit_sets[k]:
            by_unit.setdefault(u, []).append(k)
    counts: dict[tuple[str, str], int] = {}
    for codes in by_unit.values():
        for a, b in itertools.combinations(sorted(codes), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    edges = []
    for (a, b), c in counts.items():
        if c < min_count:
            continue
        na, nb = len(unit_sets[a]), len(unit_sets[b])
        union = na + nb - c
        p_ab, p_a, p_b = c / n_units, na / n_units, nb / n_units
        if p_ab >= 1.0:
            npmi = 1.0
        else:
            npmi = math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)
        if min_npmi is not None and npmi <= min_npmi:
            continue
        edges.append({"source": a, "target": b, "type": "cooccurrence", "weight": c,
                      "jaccard": round(c / union, 4), "npmi": round(npmi, 4)})
    return edges


def similarity_edges(X: sp.csr_matrix, seg_sets: dict[str, set[int]], threshold: float) -> list[dict]:
    from sklearn.preprocessing import normalize  # imported here: scikit-learn is slow to load in the browser edition

    keys = sorted(seg_sets)
    if len(keys) < 2:
        return []
    rows = []
    for k in keys:
        idx = sorted(seg_sets[k])
        rows.append(np.asarray(X[idx].mean(axis=0)).ravel())
    C = normalize(np.vstack(rows))
    S = C @ C.T
    edges = []
    for i, j in zip(*np.triu_indices(len(keys), k=1)):
        if S[i, j] >= threshold:
            edges.append({"source": keys[i], "target": keys[j], "type": "similarity", "weight": round(float(S[i, j]), 4)})
    return edges


def build_code_graph(
    codes: pd.DataFrame,
    assignments: pd.DataFrame,
    segments: pd.DataFrame,
    X: sp.csr_matrix | None,
    level: str = "record",
    min_cooccurrence: int = 2,
    similarity: bool = True,
    similarity_threshold: float = 0.3,
    min_npmi: float | None = 0.0,
) -> tuple[nx.Graph, list[dict]]:
    """Return (analysis graph with co-occurrence edges, list of all typed edges).

    ``assignments`` needs code_key and segment_pos; ``segments`` needs record_id
    and record_key (respondent).
    """
    seg_record = segments["record_id"].to_numpy()
    seg_resp = segments["record_key"].to_numpy() if "record_key" in segments else seg_record
    seg_sets: dict[str, set[int]] = {}
    for key, grp in assignments.groupby("code_key"):
        seg_sets[key] = set(grp["segment_pos"].astype(int).tolist())

    G = nx.Graph()
    for row in codes.itertuples():
        ss = seg_sets.get(row.code_key, set())
        if not ss:
            continue
        G.add_node(
            row.code_key,
            label=row.label,
            method=row.method,
            # uncategorised codes read back from storage carry NaN; keep None so category counts and JSON stay clean
            category=row.category if isinstance(row.category, str) and row.category.strip() else None,
            frequency=len(ss),
            records=len({seg_record[i] for i in ss}),
            respondents=len({seg_resp[i] for i in ss}),
        )
    present = {k: v for k, v in seg_sets.items() if k in G}

    if level == "record":
        units = {k: {seg_record[i] for i in v} for k, v in present.items()}
        n_units = len(set(seg_record.tolist()))
    else:
        units = present
        n_units = len(segments)
    co = cooccurrence_edges(units, max(n_units, 1), min_cooccurrence, min_npmi)
    for e in co:
        G.add_edge(e["source"], e["target"], weight=e["weight"], jaccard=e["jaccard"], npmi=e["npmi"])
    sim = similarity_edges(X, present, similarity_threshold) if (similarity and X is not None) else []
    return G, co + sim
