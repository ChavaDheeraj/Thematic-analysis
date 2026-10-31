"""Temporal analysis: per-period prevalence, trends, change points and dynamic graphs.

Only uses timestamps that were present in the data. Records without a
parseable timestamp are excluded from temporal results and counted.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from thematicml.graph.builder import cooccurrence_edges
from thematicml.graph.communities import detect_communities
from thematicml.temporal.trends import (
    arima_forecast,
    change_points,
    exponential_smoothing,
    mann_kendall,
    rolling_mean,
    seasonal_decomposition,
)

PANDAS_FREQ = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS"}


def choose_frequency(ts: pd.Series) -> str:
    span = (ts.max() - ts.min()).days
    if span <= 45:
        return "D"
    if span <= 270:
        return "W"
    if span <= 365 * 4:
        return "M"
    return "Q"


def _period_label(p: pd.Timestamp, freq: str) -> str:
    if freq == "M":
        return p.strftime("%Y-%m")
    if freq == "Q":
        return f"{p.year}-Q{(p.month - 1) // 3 + 1}"
    return p.strftime("%Y-%m-%d")


def temporal_analysis(
    segments: pd.DataFrame,
    assignments: pd.DataFrame,
    code_labels: dict,
    themes: list[dict] | None = None,
    freq: str = "auto",
    top_codes: int = 10,
    n_windows: int = 4,
    min_cooccurrence: int = 2,
    seed: int = 42,
    sentiment: np.ndarray | None = None,
) -> dict:
    """segments needs record_id and timestamp (ISO string or None)."""
    ts = pd.to_datetime(segments["timestamp"], errors="coerce")
    has_ts = ts.notna().to_numpy()
    n_rec_total = segments["record_id"].nunique()
    n_rec_ts = segments.loc[has_ts, "record_id"].nunique()
    if n_rec_ts == 0:
        return {"available": False, "reason": "No parseable timestamps in the analysed records. Temporal analysis requires ordered data."}
    if ts[has_ts].nunique() < 2:
        return {"available": False, "reason": "All timestamps are identical; there is no ordering to analyse."}
    f = choose_frequency(ts[has_ts]) if freq == "auto" else freq
    period = ts.dt.to_period({"D": "D", "W": "W-SUN", "M": "M", "Q": "Q"}[f]).dt.start_time
    full_index = pd.date_range(period[has_ts].min(), period[has_ts].max(), freq=PANDAS_FREQ[f])
    if f == "W":
        full_index = pd.date_range(period[has_ts].min(), period[has_ts].max(), freq="7D")
    labels = [_period_label(p, f) for p in full_index]
    seg_period = period.to_numpy()

    seg_counts = pd.Series(seg_period[has_ts]).value_counts().reindex(full_index, fill_value=0)
    rec_frame = pd.DataFrame({"record_id": segments["record_id"].to_numpy()[has_ts], "p": seg_period[has_ts]}).drop_duplicates("record_id")
    rec_counts = rec_frame["p"].value_counts().reindex(full_index, fill_value=0)

    a_pos = assignments["segment_pos"].astype(int).to_numpy()
    a_ok = has_ts[a_pos]
    a_df = pd.DataFrame({"code": assignments["code_key"].to_numpy()[a_ok], "p": seg_period[a_pos[a_ok]], "pos": a_pos[a_ok]}).drop_duplicates(["code", "pos"])
    totals = a_df["code"].value_counts()
    chosen = totals.index[:top_codes].tolist()

    def series_info(counts: pd.Series) -> dict:
        share = [round(100 * c / s, 3) if s else 0.0 for c, s in zip(counts.tolist(), seg_counts.tolist())]
        cps = change_points(share)
        return {
            "counts": [int(v) for v in counts.tolist()],
            "share_pct": share,
            "rolling_mean_share": rolling_mean(share, 3),
            "smoothed_share": exponential_smoothing(share, 0.3),
            "trend": mann_kendall(share),
            "change_points": [labels[i] for i in cps],
        }

    code_series = []
    for key in chosen:
        counts = a_df[a_df["code"] == key]["p"].value_counts().reindex(full_index, fill_value=0)
        code_series.append({"code_key": key, "label": code_labels.get(key, key), **series_info(counts)})

    theme_series = []
    for th in themes or []:
        keys = set(th["code_keys"])
        sub = a_df[a_df["code"].isin(keys)].drop_duplicates("pos")
        counts = sub["p"].value_counts().reindex(full_index, fill_value=0)
        theme_series.append({"theme_id": th.get("id"), "label": th["label"], **series_info(counts)})

    sentiment_series = None
    if sentiment is not None:
        sdf = pd.DataFrame({"p": seg_period[has_ts], "s": sentiment[has_ts]})
        means = sdf.groupby("p")["s"].mean().reindex(full_index)
        sentiment_series = [None if pd.isna(v) else round(float(v), 4) for v in means.tolist()]

    volume = [int(v) for v in seg_counts.tolist()]
    out = {
        "available": True,
        "frequency": f,
        "periods": labels,
        "records_with_timestamp": n_rec_ts,
        "records_without_timestamp": n_rec_total - n_rec_ts,
        "coverage_pct": round(100 * n_rec_ts / n_rec_total, 2),
        "segments_per_period": volume,
        "records_per_period": [int(v) for v in rec_counts.tolist()],
        "volume_trend": mann_kendall(volume),
        "volume_change_points": [labels[i] for i in change_points(volume)],
        "seasonal_decomposition": seasonal_decomposition(volume, f),
        "arima": arima_forecast(volume),
        "codes": code_series,
        "themes": theme_series,
        "sentiment_mean": sentiment_series,
        "share_definition": "segments with the code in the period / all analysed segments in the period × 100",
        "warnings": [],
    }
    if len(labels) < 4:
        out["warnings"].append("Fewer than 4 periods: trend tests are not reported.")
    out["dynamic_graph"] = dynamic_graph(segments, assignments, code_labels, ts, has_ts, n_windows, min_cooccurrence, seed)
    return out


def dynamic_graph(segments, assignments, code_labels, ts, has_ts, n_windows, min_cooccurrence, seed) -> dict:
    """Build a co-occurrence graph per equal-length time window and compare consecutive windows."""
    t = ts[has_ts]
    lo, hi = t.min(), t.max()
    edges_t = pd.date_range(lo, hi, periods=n_windows + 1)
    seg_rec = segments["record_id"].to_numpy()
    a_pos = assignments["segment_pos"].astype(int).to_numpy()
    windows = []
    for w in range(n_windows):
        start, end = edges_t[w], edges_t[w + 1]
        in_w = has_ts & (ts >= start).to_numpy() & ((ts <= end) if w == n_windows - 1 else (ts < end)).to_numpy()
        sel = in_w[a_pos]
        sub = assignments[sel]
        units: dict = {}
        for key, pos in zip(sub["code_key"], sub["segment_pos"].astype(int)):
            units.setdefault(key, set()).add(seg_rec[pos])
        n_units = len(set(seg_rec[in_w].tolist()))
        G = nx.Graph()
        for k, v in units.items():
            G.add_node(k, frequency=len(v))
        for e in cooccurrence_edges(units, max(n_units, 1), min_cooccurrence, 0.0):
            G.add_edge(e["source"], e["target"], weight=e["weight"])
        membership, info = detect_communities(G, "louvain", 1.0, seed) if G.number_of_nodes() else ({}, {"n_communities": 0})
        windows.append({
            "label": f"{start:%Y-%m-%d} – {end:%Y-%m-%d}",
            "records": n_units,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": info.get("n_multi_node_communities", 0),
            "largest_community": max(info.get("sizes", [0]) or [0]),
            "_graph": G,
            "_membership": membership,
        })
    comparisons = []
    for i in range(1, len(windows)):
        A, B = windows[i - 1]["_graph"], windows[i]["_graph"]
        ea = {tuple(sorted(e)): A.edges[e]["weight"] for e in A.edges()}
        eb = {tuple(sorted(e)): B.edges[e]["weight"] for e in B.edges()}
        new = sorted(((p, w) for p, w in eb.items() if p not in ea), key=lambda x: -x[1])
        gone = sorted(((p, w) for p, w in ea.items() if p not in eb), key=lambda x: -x[1])
        changed = sorted(((p, eb[p] - ea[p]) for p in set(ea) & set(eb)), key=lambda x: -abs(x[1]))
        freq_change = []
        for k in set(A.nodes()) | set(B.nodes()):
            fa = A.nodes[k]["frequency"] if k in A else 0
            fb = B.nodes[k]["frequency"] if k in B else 0
            freq_change.append({"code": code_labels.get(k, k), "before": fa, "after": fb, "change": fb - fa})
        freq_change.sort(key=lambda r: -abs(r["change"]))

        def lab(p):
            return [code_labels.get(p[0], p[0]), code_labels.get(p[1], p[1])]

        comparisons.append({
            "from": windows[i - 1]["label"],
            "to": windows[i]["label"],
            "new_edges": [{"pair": lab(p), "weight": w} for p, w in new[:20]],
            "disappeared_edges": [{"pair": lab(p), "weight": w} for p, w in gone[:20]],
            "weight_changes": [{"pair": lab(p), "change": d} for p, d in changed[:20] if d != 0],
            "node_frequency_changes": freq_change[:20],
            "n_new_edges": len(new),
            "n_disappeared_edges": len(gone),
        })
    for w in windows:
        w.pop("_graph")
        w.pop("_membership")
    return {
        "windows": windows,
        "comparisons": comparisons,
        "window_rule": f"{n_windows} equal-length time windows between the first and last timestamp",
        "caveat": "Windows contain different numbers of records; an edge missing in a window may reflect sample size.",
    }
