"""Code frequencies and prevalence at assignment, segment, record and respondent level."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from thematicml.analytics.percentages import percentage, wilson_ci


def code_frequency_table(codes: pd.DataFrame, assignments: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """One row per code.

    Denominators: segment_pct uses all analysed segments; record_pct uses all
    analysed records; respondent_pct uses all distinct respondents (record_key).
    """
    n_seg = len(segments)
    n_rec = segments["record_id"].nunique()
    n_resp = segments["record_key"].nunique()
    seg_rec = segments["record_id"].to_numpy()
    seg_resp = segments["record_key"].to_numpy()
    rows = []
    grouped = {k: g for k, g in assignments.groupby("code_key")}
    for c in codes.itertuples():
        g = grouped.get(c.code_key)
        if g is None:
            n_a = n_s = n_r = n_p = 0
        else:
            pos = g["segment_pos"].astype(int).to_numpy()
            n_a, n_s = len(g), len(set(pos.tolist()))
            n_r = len(set(seg_rec[pos].tolist()))
            n_p = len(set(seg_resp[pos].tolist()))
        ci = wilson_ci(n_p, n_resp)
        rows.append(
            {
                "code_key": c.code_key,
                "label": c.label,
                "method": c.method,
                "assignments": n_a,
                "segments": n_s,
                "segment_pct": round(100 * n_s / n_seg, 2) if n_seg else None,
                "records": n_r,
                "record_pct": round(100 * n_r / n_rec, 2) if n_rec else None,
                "respondents": n_p,
                "respondent_pct": round(100 * n_p / n_resp, 2) if n_resp else None,
                "respondent_ci_low": ci[0] if ci else None,
                "respondent_ci_high": ci[1] if ci else None,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["segments", "label"], ascending=[False, True]).reset_index(drop=True)
    df.attrs["denominators"] = {"segments": n_seg, "records": n_rec, "respondents": n_resp}
    return df


def coverage_metrics(assignments: pd.DataFrame, segments: pd.DataFrame) -> dict:
    n_seg = len(segments)
    coded_pos = set(assignments["segment_pos"].astype(int).tolist()) if len(assignments) else set()
    rec_of = segments["record_id"].to_numpy()
    resp_of = segments["record_key"].to_numpy()
    coded_records = {rec_of[p] for p in coded_pos}
    coded_resp = {resp_of[p] for p in coded_pos}
    n_rec, n_resp = segments["record_id"].nunique(), segments["record_key"].nunique()
    per_resp = pd.Series(0, index=pd.Index(sorted(set(resp_of.tolist()))))
    if len(assignments):
        tmp = pd.DataFrame({"resp": resp_of[assignments["segment_pos"].astype(int).to_numpy()], "code": assignments["code_key"].to_numpy()})
        counts = tmp.drop_duplicates().groupby("resp").size()
        per_resp.loc[counts.index] = counts.values
    return {
        "coded_segments": percentage(len(coded_pos), n_seg, "segments with >= 1 code / analysed segments", "segment"),
        "uncoded_segments": percentage(n_seg - len(coded_pos), n_seg, "segments with no code / analysed segments", "segment"),
        "coded_records": percentage(len(coded_records), n_rec, "records with >= 1 coded segment / analysed records", "record"),
        "coded_respondents": percentage(len(coded_resp), n_resp, "respondents with >= 1 code / analysed respondents", "respondent"),
        "total_assignments": int(len(assignments)),
        "codes_per_respondent": {
            "mean": round(float(per_resp.mean()), 3) if len(per_resp) else 0.0,
            "median": float(per_resp.median()) if len(per_resp) else 0.0,
            "max": int(per_resp.max()) if len(per_resp) else 0,
            "definition": "distinct codes per respondent",
        },
    }


def cooccurrence_table(assignments: pd.DataFrame, segments: pd.DataFrame, labels: dict, level: str = "record", top: int = 50) -> list[dict]:
    if assignments.empty:
        return []
    unit = segments["record_id"].to_numpy() if level == "record" else np.arange(len(segments))
    tmp = pd.DataFrame({"u": unit[assignments["segment_pos"].astype(int).to_numpy()], "c": assignments["code_key"].to_numpy()}).drop_duplicates()
    counts: dict = {}
    for _, g in tmp.groupby("u"):
        for a, b in itertools.combinations(sorted(g["c"].tolist()), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rows = [{"code_a": labels.get(a, a), "code_b": labels.get(b, b), "key_a": a, "key_b": b, "count": c} for (a, b), c in counts.items()]
    rows.sort(key=lambda r: -r["count"])
    return rows[:top]
