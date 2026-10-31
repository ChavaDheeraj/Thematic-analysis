"""Candidate theme discovery without generative models.

Candidate themes are graph communities of co-occurring codes. Labels come from
a configurable template filled with the community's most frequent code labels.
Representative quotations are extracted verbatim. Every theme starts as a
*candidate*; only a researcher can approve it.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def template_label(template: str, terms: list[str]) -> str:
    """Fill ``{1}``, ``{2}``, ... with ranked terms; drop placeholders that have no term."""
    def repl(m):
        i = int(m.group(1)) - 1
        return terms[i] if 0 <= i < len(terms) else ""

    label = re.sub(r"\{(\d+)\}", repl, template)
    label = re.sub(r"\s*(&|and|/|,)\s*$", "", label.strip())
    label = re.sub(r"^\s*(&|and|/|,)\s*", "", label)
    return re.sub(r"\s{2,}", " ", label).strip() or (terms[0] if terms else "Untitled theme")


def discover_themes(
    codes: pd.DataFrame,
    assignments: pd.DataFrame,
    segments: pd.DataFrame,
    membership: dict,
    X,
    template: str = "{1} & {2}",
    min_codes: int = 2,
    n_rep: int = 5,
    sentiment: np.ndarray | None = None,
    dominant_topic: np.ndarray | None = None,
) -> list[dict]:
    labels = dict(zip(codes["code_key"], codes["label"]))
    seg_sets = {k: set(g["segment_pos"].astype(int).tolist()) for k, g in assignments.groupby("code_key")}
    comms: dict[int, list[str]] = {}
    for k, c in membership.items():
        comms.setdefault(c, []).append(k)
    n_seg = len(segments)
    n_rec = segments["record_id"].nunique()
    n_resp = segments["record_key"].nunique()
    seg_rec = segments["record_id"].to_numpy()
    seg_resp = segments["record_key"].to_numpy()
    Xn = normalize(X) if X is not None else None

    themes = []
    for cid in sorted(comms):
        keys = comms[cid]
        if len(keys) < min_codes:
            continue
        keys = sorted(keys, key=lambda k: (-len(seg_sets.get(k, ())), labels.get(k, k)))
        segs = set().union(*(seg_sets.get(k, set()) for k in keys))
        if not segs:
            continue
        seg_list = np.array(sorted(segs))
        hits = np.array([sum(1 for k in keys if p in seg_sets.get(k, ())) for p in seg_list])
        if Xn is not None:
            centroid = np.asarray(Xn[seg_list].mean(axis=0)).ravel()
            norm = np.linalg.norm(centroid) or 1.0
            sims = np.asarray(Xn[seg_list] @ centroid).ravel() / norm
        else:
            sims = np.zeros(len(seg_list))
        order = np.lexsort((-sims, -hits))
        representative = [{"segment_pos": int(seg_list[i]), "codes_matched": int(hits[i]), "similarity": round(float(sims[i]), 4)} for i in order[:n_rep]]

        contrasting, polarity = [], None
        if sentiment is not None:
            s = sentiment[seg_list]
            pos, neg = int((s >= 0.05).sum()), int((s <= -0.05).sum())
            if pos + neg:
                polarity = "positive" if pos >= neg else "negative"
                opp = np.flatnonzero(s <= -0.05) if polarity == "positive" else np.flatnonzero(s >= 0.05)
                opp = opp[np.argsort(-np.abs(s[opp]))][:3]
                contrasting = [{"segment_pos": int(seg_list[i]), "compound": round(float(s[i]), 4)} for i in opp]

        topic = None
        if dominant_topic is not None:
            dt = dominant_topic[seg_list]
            dt = dt[dt >= 0]
            if len(dt):
                vals, cnt = np.unique(dt, return_counts=True)
                topic = {"topic": int(vals[cnt.argmax()]), "share_pct": round(100 * cnt.max() / len(seg_list), 2)}

        recs = {seg_rec[p] for p in segs}
        resp = {seg_resp[p] for p in segs}
        top_labels = [labels.get(k, k) for k in keys]
        themes.append(
            {
                "community": int(cid),
                "label": template_label(template, top_labels),
                "definition": f"Candidate theme grouping {len(keys)} co-occurring codes: {', '.join(top_labels[:6])}"
                + ("…" if len(keys) > 6 else "") + ". Template-generated; awaiting researcher definition.",
                "code_keys": keys,
                "segments": len(segs),
                "segment_pct": round(100 * len(segs) / n_seg, 2) if n_seg else None,
                "records": len(recs),
                "record_pct": round(100 * len(recs) / n_rec, 2) if n_rec else None,
                "respondents": len(resp),
                "respondent_pct": round(100 * len(resp) / n_resp, 2) if n_resp else None,
                "representative": representative,
                "contradictory": contrasting,
                "majority_polarity": polarity,
                "associated_topic": topic,
                "source": "graph-community",
            }
        )
    themes.sort(key=lambda t: -t["segments"])
    return themes
