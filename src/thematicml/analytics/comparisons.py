"""Group comparisons within a run, and comparisons between runs/uploads."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def group_comparison(assignments: pd.DataFrame, segments: pd.DataFrame, labels: dict, code_keys: list[str], min_group_size: int = 5) -> dict:
    """Record-level prevalence of each code across groups, with a chi-square
    test of independence (Fisher exact for 2x2 tables with small expected counts).

    p-values are adjusted with Benjamini-Hochberg across the tested codes.
    """
    if "group_value" not in segments or segments["group_value"].isna().all():
        return {"available": False, "reason": "No group field was mapped for this dataset."}
    rec = segments.drop_duplicates("record_id")[["record_id", "group_value"]].dropna()
    sizes = rec["group_value"].value_counts()
    keep = sizes[sizes >= min_group_size].index.tolist()
    if len(keep) < 2:
        return {"available": False, "reason": f"Need at least two groups with >= {min_group_size} records."}
    rec = rec[rec["group_value"].isin(keep)]
    seg_rec = segments["record_id"].to_numpy()
    results = []
    for key in code_keys:
        pos = assignments.loc[assignments["code_key"] == key, "segment_pos"].astype(int).to_numpy()
        has = set(seg_rec[pos].tolist())
        rec_has = rec["record_id"].isin(has)
        table = pd.crosstab(rec["group_value"], rec_has).reindex(columns=[False, True], fill_value=0)
        obs = table.to_numpy()
        row = {
            "code_key": key,
            "label": labels.get(key, key),
            "groups": {
                str(g): {"records": int(obs[i].sum()), "with_code": int(obs[i, 1]),
                         "pct": round(100 * obs[i, 1] / obs[i].sum(), 2) if obs[i].sum() else None}
                for i, g in enumerate(table.index)
            },
        }
        if obs[:, 1].sum() == 0 or obs[:, 0].sum() == 0:
            row.update({"test": None, "p_value": None, "note": "Code absent or universal; not tested."})
        else:
            chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
            low = (expected < 5).mean()
            if obs.shape == (2, 2) and low > 0:
                _, p = stats.fisher_exact(obs)
                row.update({"test": "fisher_exact", "p_value": float(p)})
            else:
                row.update({"test": "chi_square", "statistic": round(float(chi2), 4), "dof": int(dof), "p_value": float(p)})
                if low > 0.2:
                    row["note"] = "More than 20% of expected counts are below 5; chi-square approximation may be unreliable."
            n = obs.sum()
            k = min(obs.shape) - 1
            row["cramers_v"] = round(math.sqrt(stats.chi2_contingency(obs, correction=False)[0] / (n * k)), 4) if k > 0 else None
        results.append(row)
    tested = [r for r in results if r.get("p_value") is not None]
    for r, adj in zip(tested, benjamini_hochberg([r["p_value"] for r in tested])):
        r["p_adjusted_bh"] = round(adj, 6)
        r["p_value"] = round(r["p_value"], 6)
    return {
        "available": True,
        "unit": "record",
        "groups": {str(g): int(sizes[g]) for g in keep},
        "excluded_small_groups": [str(g) for g in sizes.index if g not in keep],
        "results": results,
        "caveat": "Associations are descriptive; they do not establish causes. Tests assume independent records.",
    }


def two_proportion(k1: int, n1: int, k2: int, n2: int) -> dict:
    if not n1 or not n2:
        return {"diff_pct": None, "ci95": None, "p_value": None}
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pooled = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    p = float(2 * stats.norm.sf(abs(diff) / se0)) if se0 > 0 else None
    return {
        "diff_pct": round(100 * diff, 2),
        "ci95": [round(100 * (diff - 1.96 * se), 2), round(100 * (diff + 1.96 * se), 2)],
        "p_value": None if p is None else round(p, 6),
    }


def _norm(label: str) -> str:
    return " ".join(str(label).lower().split())


def compare_runs(a: dict, b: dict) -> dict:
    """Compare two analysis snapshots (from Store.run_snapshot).

    Each snapshot provides codes (label, records), n_records, edges (label
    pairs with weight), communities (label -> community), themes (label,
    code_labels, status) and config.
    """
    ca = {_norm(c["label"]): c for c in a["codes"]}
    cb = {_norm(c["label"]): c for c in b["codes"]}
    shared = sorted(set(ca) & set(cb))
    freq = []
    for k in shared:
        x, y = ca[k], cb[k]
        tp = two_proportion(x["records"], a["n_records"], y["records"], b["n_records"])
        freq.append({
            "code": x["label"],
            "a_records": x["records"], "a_pct": round(100 * x["records"] / a["n_records"], 2) if a["n_records"] else None,
            "b_records": y["records"], "b_pct": round(100 * y["records"] / b["n_records"], 2) if b["n_records"] else None,
            "absolute_change": y["records"] - x["records"],
            **tp,
        })
    freq.sort(key=lambda r: -abs(r["diff_pct"] or 0))

    def edge_set(s):
        return {tuple(sorted((_norm(e["a"]), _norm(e["b"])))): e["weight"] for e in s["edges"]}

    ea, eb = edge_set(a), edge_set(b)
    shared_nodes = set(shared)
    new_edges = [{"a": p[0], "b": p[1], "weight": eb[p]} for p in eb if p not in ea and set(p) <= shared_nodes]
    gone_edges = [{"a": p[0], "b": p[1], "weight": ea[p]} for p in ea if p not in eb and set(p) <= shared_nodes]

    comm_a = {_norm(k): v for k, v in a["communities"].items()}
    comm_b = {_norm(k): v for k, v in b["communities"].items()}
    both = [k for k in shared if k in comm_a and k in comm_b]
    community = {"compared_codes": len(both)}
    if len(both) >= 2:
        from sklearn.metrics import adjusted_rand_score

        community["adjusted_rand_index"] = round(float(adjusted_rand_score([comm_a[k] for k in both], [comm_b[k] for k in both])), 4)
        moved_together, split_apart = [], []
        for i, x in enumerate(both):
            for y in both[i + 1:]:
                ta, tb = comm_a[x] == comm_a[y], comm_b[x] == comm_b[y]
                if ta and not tb:
                    split_apart.append([ca[x]["label"], ca[y]["label"]])
                elif tb and not ta:
                    moved_together.append([ca[x]["label"], ca[y]["label"]])
        community["pairs_split_apart"] = split_apart[:50]
        community["pairs_joined"] = moved_together[:50]

    themes = []
    for ta in a["themes"]:
        sa = {_norm(x) for x in ta["code_labels"]}
        best = None
        for tb in b["themes"]:
            sb = {_norm(x) for x in tb["code_labels"]}
            j = len(sa & sb) / len(sa | sb) if sa | sb else 0
            if j >= 0.3 and (best is None or j > best[0]):
                best = (j, tb)
        if best:
            themes.append({"a_theme": ta["label"], "a_status": ta["status"], "b_theme": best[1]["label"],
                           "b_status": best[1]["status"], "code_overlap_jaccard": round(best[0], 3)})

    cfg_a, cfg_b = a.get("config", {}), b.get("config", {})
    config_diff = [{"parameter": k, "a": cfg_a.get(k), "b": cfg_b.get(k)} for k in sorted(set(cfg_a) | set(cfg_b)) if cfg_a.get(k) != cfg_b.get(k)]

    return {
        "a": {"run_id": a["run_id"], "label": a.get("label"), "n_records": a["n_records"], "n_codes": len(ca)},
        "b": {"run_id": b["run_id"], "label": b.get("label"), "n_records": b["n_records"], "n_codes": len(cb)},
        "shared_codes": [ca[k]["label"] for k in shared],
        "only_in_a": sorted(ca[k]["label"] for k in set(ca) - set(cb)),
        "only_in_b": sorted(cb[k]["label"] for k in set(cb) - set(ca)),
        "frequency_changes": freq,
        "new_relationships": sorted(new_edges, key=lambda e: -e["weight"])[:100],
        "disappeared_relationships": sorted(gone_edges, key=lambda e: -e["weight"])[:100],
        "community_changes": community,
        "shared_themes": themes,
        "model_differences": config_diff,
        "caveat": "A pattern absent from one dataset has not necessarily disappeared in the population; it may not appear in that sample. "
        "Compare dataset sizes and confidence intervals. Differences may also be caused by different configurations (see model_differences).",
    }
