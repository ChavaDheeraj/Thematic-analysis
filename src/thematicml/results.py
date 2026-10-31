"""Read analysis results back from storage as live, researcher-edit-aware views.

The pipeline stores a snapshot of the metrics computed at run time
(``runs.metrics``). The functions here recompute prevalence, coverage and
graph structure from the *current* codes and assignments, so that merges,
splits, rejections and approvals made by the researcher are reflected.
"""
from __future__ import annotations

import functools
import json

import joblib
import pandas as pd

from thematicml.analytics import code_frequency_table, cooccurrence_table, coverage_metrics, group_comparison
from thematicml.analytics.percentages import percentage
from thematicml.graph import build_code_graph, detect_communities, graph_metrics, node_metrics
from thematicml.storage.database import Store

ACTIVE = ("candidate", "approved")


class NotFound(KeyError):
    pass


def get_run_or_404(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise NotFound(f"Run {run_id} not found")
    return run


def load_segments(store: Store, run_id: str) -> pd.DataFrame:
    run = get_run_or_404(store, run_id)
    rows = store.query(
        "SELECT s.id AS segment_id, s.pos, s.record_id, s.seg_index, s.text, s.char_start, s.char_end, s.cluster, s.topic, "
        "s.topic_weight, s.sentiment, s.sentiment_label, r.record_key, r.dataset_id, r.timestamp, r.group_value, d.filename "
        "FROM segments s JOIN records r ON r.id = s.record_id JOIN datasets d ON d.id = r.dataset_id "
        "WHERE s.run_id = ? ORDER BY s.pos",
        (run_id,),
    )
    df = pd.DataFrame(rows, columns=["segment_id", "pos", "record_id", "seg_index", "text", "char_start", "char_end", "cluster", "topic",
                                     "topic_weight", "sentiment", "sentiment_label", "record_key", "dataset_id", "timestamp", "group_value", "filename"])
    if len(run["dataset_ids"]) > 1 and len(df):
        gen = df["record_key"].str.startswith("row-")
        df.loc[gen, "record_key"] = df.loc[gen, "dataset_id"] + ":" + df.loc[gen, "record_key"]
    return df


def load_codes(store: Store, run_id: str, statuses=ACTIVE) -> pd.DataFrame:
    q = ",".join("?" * len(statuses))
    rows = store.query(f"SELECT * FROM codes WHERE run_id = ? AND status IN ({q}) ORDER BY id", (run_id, *statuses))
    df = pd.DataFrame(rows, columns=["id", "run_id", "code_key", "label", "description", "method", "feature", "category", "status",
                                     "origin", "merged_into", "notes", "updated_at"])
    # pandas infers NaN for NULLs in a column that also holds text (or float for ids); keep None and int ids so rows stay JSON-safe
    for col in ("description", "feature", "category", "origin", "notes", "updated_at"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    df["merged_into"] = pd.Series([None if pd.isna(v) else int(v) for v in df["merged_into"]], index=df.index, dtype=object)
    return df


def load_assignments(store: Store, run_id: str, statuses=ACTIVE) -> pd.DataFrame:
    q = ",".join("?" * len(statuses))
    rows = store.query(
        f"SELECT a.id, a.code_id, a.segment_id, s.pos AS segment_pos, a.method, a.score, a.matched_text, a.char_start, a.char_end, a.negated "
        f"FROM assignments a JOIN segments s ON s.id = a.segment_id JOIN codes c ON c.id = a.code_id "
        f"WHERE a.run_id = ? AND c.status IN ({q})",
        (run_id, *statuses),
    )
    df = pd.DataFrame(rows, columns=["id", "code_id", "segment_id", "segment_pos", "method", "score", "matched_text", "char_start", "char_end", "negated"])
    df["code_key"] = df["code_id"]
    return df


def _codes_keyed(codes: pd.DataFrame) -> pd.DataFrame:
    out = codes.rename(columns={"code_key": "machine_key"}).copy()
    out["code_key"] = out["id"]
    return out


@functools.lru_cache(maxsize=8)
def _features_cached(path: str, mtime: float):
    return joblib.load(path)


def load_features(store: Store, run: dict):
    p = store.run_dir(run["project_id"], run["id"]) / "features.joblib"
    if not p.exists():
        return None
    return _features_cached(str(p), p.stat().st_mtime)


def feature_matrix_by_pos(store: Store, run: dict, segments: pd.DataFrame):
    art = load_features(store, run)
    if art is None:
        return None
    return art["features"].matrix


# ---------------------------------------------------------------------- codes

def code_table(store: Store, run_id: str, include_inactive: bool = True) -> list[dict]:
    segs = load_segments(store, run_id)
    codes_all = load_codes(store, run_id, ("candidate", "approved", "rejected", "merged") if include_inactive else ACTIVE)
    assigns = load_assignments(store, run_id, ("candidate", "approved", "rejected"))
    freq = code_frequency_table(_codes_keyed(codes_all), assigns, segs)
    by_id = {r["code_key"]: r for r in freq.to_dict(orient="records")} if len(freq) else {}
    out = []
    for c in codes_all.to_dict(orient="records"):
        f = by_id.get(c["id"], {})
        out.append({
            "id": c["id"], "label": c["label"], "description": c["description"], "method": c["method"], "feature": c["feature"],
            "category": c["category"], "status": c["status"], "origin": c["origin"], "merged_into": c["merged_into"], "notes": c["notes"],
            "assignments": f.get("assignments", 0), "segments": f.get("segments", 0), "segment_pct": f.get("segment_pct"),
            "records": f.get("records", 0), "record_pct": f.get("record_pct"), "respondents": f.get("respondents", 0),
            "respondent_pct": f.get("respondent_pct"), "respondent_ci95": [f.get("respondent_ci_low"), f.get("respondent_ci_high")],
        })
    out.sort(key=lambda r: (r["status"] in ("merged", "rejected"), -r["segments"], r["label"]))
    return out


# ---------------------------------------------------------------------- themes

def theme_table(store: Store, run_id: str) -> list[dict]:
    segs = load_segments(store, run_id)
    assigns = load_assignments(store, run_id)
    codes = load_codes(store, run_id, ("candidate", "approved", "rejected", "merged"))
    label = dict(zip(codes["id"], codes["label"]))
    status = dict(zip(codes["id"], codes["status"]))
    merged_into = dict(zip(codes["id"], codes["merged_into"]))
    n_seg, n_rec, n_resp = len(segs), segs["record_id"].nunique(), segs["record_key"].nunique()
    seg_rec = dict(zip(segs["pos"], segs["record_id"]))
    seg_resp = dict(zip(segs["pos"], segs["record_key"]))
    by_code = {k: set(g["segment_pos"].tolist()) for k, g in assigns.groupby("code_id")}
    themes = store.query("SELECT * FROM themes WHERE run_id = ? ORDER BY id", (run_id,))
    out = []
    for t in themes:
        ids = []
        for cid in t["code_ids"] or []:
            # follow merges so themes keep pointing at the surviving code
            seen = set()
            while status.get(cid) == "merged" and merged_into.get(cid) and cid not in seen:
                seen.add(cid)
                cid = int(merged_into[cid])
            if status.get(cid) in ACTIVE and cid not in ids:
                ids.append(cid)
        segs_t = set().union(*(by_code.get(c, set()) for c in ids)) if ids else set()
        recs = {seg_rec[p] for p in segs_t}
        resp = {seg_resp[p] for p in segs_t}
        out.append({
            **t,
            "code_ids": ids,
            "codes": [{"id": c, "label": label.get(c, str(c))} for c in ids],
            "segments": len(segs_t),
            "prevalence": {
                "segment": percentage(len(segs_t), n_seg, "segments with any theme code / analysed segments", "segment"),
                "record": percentage(len(recs), n_rec, "records with any theme code / analysed records", "record"),
                "respondent": percentage(len(resp), n_resp, "respondents with any theme code / analysed respondents", "respondent", ci=True),
            },
            "supporting_quotations": len(t["representative"] or []),
            "contradictory_count": len(t["contradictory"] or []),
        })
    return out


# ---------------------------------------------------------------------- graph

def graph_view(
    store: Store,
    run_id: str,
    level: str | None = None,
    min_cooccurrence: int | None = None,
    similarity: bool = True,
    similarity_threshold: float | None = None,
    method: str | None = None,
    resolution: float | None = None,
    include_isolated: bool = True,
    min_npmi: float | None = None,
) -> dict:
    """Recompute the code graph from current codes with the given parameters."""
    run = get_run_or_404(store, run_id)
    cfg = run["config"]
    level = level or cfg["cooccurrence_level"]
    min_cooccurrence = min_cooccurrence or cfg["min_cooccurrence"]
    similarity_threshold = cfg["similarity_threshold"] if similarity_threshold is None else similarity_threshold
    method = method or cfg["graph_method"]
    resolution = resolution or cfg["resolution"]
    min_npmi = cfg.get("min_npmi", 0.0) if min_npmi is None else min_npmi

    segs = load_segments(store, run_id)
    codes = _codes_keyed(load_codes(store, run_id))
    assigns = load_assignments(store, run_id)
    X = feature_matrix_by_pos(store, run, segs) if similarity else None
    G, typed = build_code_graph(codes, assigns, segs, X, level, min_cooccurrence, similarity, similarity_threshold, min_npmi)
    membership, cinfo = detect_communities(G, method, resolution, cfg["random_state"])
    nm = node_metrics(G)
    themes = theme_table(store, run_id)
    theme_of: dict = {}
    for t in themes:
        for c in t["code_ids"]:
            theme_of.setdefault(c, []).append({"id": t["id"], "label": t["label"], "status": t["status"]})
    n_rec, n_resp = segs["record_id"].nunique(), segs["record_key"].nunique()
    status = dict(zip(codes["id"], codes["status"]))
    nodes = []
    for n, d in G.nodes(data=True):
        if not include_isolated and G.degree(n) == 0:
            continue
        nodes.append({
            "id": int(n), "label": d["label"], "type": "code", "method": d["method"], "category": d["category"],
            "status": status.get(n), "frequency": d["frequency"], "records": d["records"],
            "record_pct": round(100 * d["records"] / n_rec, 2) if n_rec else None,
            "respondents": d["respondents"], "respondent_pct": round(100 * d["respondents"] / n_resp, 2) if n_resp else None,
            "community": membership.get(n, -1), "themes": theme_of.get(int(n), []), **nm.get(n, {}),
        })
    keep = {x["id"] for x in nodes}
    edges = [{**e, "source": int(e["source"]), "target": int(e["target"])} for e in typed if int(e["source"]) in keep and int(e["target"]) in keep]
    communities = []
    by_comm: dict = {}
    for x in nodes:
        by_comm.setdefault(x["community"], []).append(x)
    for cid, members in sorted(by_comm.items()):
        members.sort(key=lambda x: -x["frequency"])
        internal = sum(1 for e in edges if e["type"] == "cooccurrence" and membership.get(e["source"]) == cid and membership.get(e["target"]) == cid)
        communities.append({"community": cid, "size": len(members), "internal_edges": internal,
                            "members": [{"id": m["id"], "label": m["label"]} for m in members]})
    stats = graph_metrics(G, membership)
    rank = sorted(nodes, key=lambda x: -x.get("degree", 0))
    return {
        "params": {"level": level, "min_cooccurrence": min_cooccurrence, "min_npmi": min_npmi, "similarity": similarity,
                   "similarity_threshold": similarity_threshold, "method": method, "resolution": resolution},
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "community_info": cinfo,
        "stats": stats,
        "top": {
            "degree": [{"id": x["id"], "label": x["label"], "value": x.get("degree")} for x in rank[:10]],
            "betweenness": [{"id": x["id"], "label": x["label"], "value": x.get("betweenness")} for x in sorted(nodes, key=lambda x: -x.get("betweenness", 0))[:10]],
            "pagerank": [{"id": x["id"], "label": x["label"], "value": x.get("pagerank")} for x in sorted(nodes, key=lambda x: -x.get("pagerank", 0))[:10]],
        },
        "isolated": [{"id": x["id"], "label": x["label"]} for x in nodes if x.get("degree", 0) == 0],
        "note": "Centrality describes structural position in the code graph, not research importance. "
                "Co-occurrence edges are observed; similarity edges are derived from TF-IDF representations.",
    }


# ---------------------------------------------------------------------- evidence

def _segment_rows(store: Store, segment_ids: list[int]) -> dict[int, dict]:
    if not segment_ids:
        return {}
    q = ",".join("?" * len(segment_ids))
    rows = store.query(
        f"SELECT s.id, s.text, s.char_start, s.char_end, s.record_id, s.sentiment, s.sentiment_label, r.record_key, r.timestamp, "
        f"r.group_value, d.filename, d.id AS dataset_id FROM segments s JOIN records r ON r.id = s.record_id "
        f"JOIN datasets d ON d.id = r.dataset_id WHERE s.id IN ({q})",
        segment_ids,
    )
    return {r["id"]: r for r in rows}


def quotation(store: Store, seg: dict) -> dict:
    """Return the exact source quotation and verify it against the stored record text."""
    rec = store.one("SELECT text FROM records WHERE id = ?", (seg["record_id"],))
    source = rec["text"] if rec else ""
    exact = source[seg["char_start"]:seg["char_end"]] if rec else None
    return {
        "segment_id": seg["id"], "text": seg["text"], "record_id": seg["record_id"], "record_key": seg["record_key"],
        "source_file": seg["filename"], "dataset_id": seg["dataset_id"], "char_start": seg["char_start"], "char_end": seg["char_end"],
        "timestamp": seg["timestamp"], "group": seg["group_value"], "sentiment": seg["sentiment"], "sentiment_label": seg["sentiment_label"],
        "verified": exact == seg["text"],
        "context_before": source[max(0, seg["char_start"] - 160):seg["char_start"]],
        "context_after": source[seg["char_end"]:seg["char_end"] + 160],
    }


def code_evidence(store: Store, code_id: int, limit: int = 200, offset: int = 0) -> dict:
    code = store.one("SELECT * FROM codes WHERE id = ?", (code_id,))
    if not code:
        raise NotFound(f"Code {code_id} not found")
    rows = store.query("SELECT * FROM assignments WHERE code_id = ? ORDER BY score DESC, id LIMIT ? OFFSET ?", (code_id, limit, offset))
    total = store.scalar("SELECT COUNT(*) FROM assignments WHERE code_id = ?", (code_id,))
    segs = _segment_rows(store, [r["segment_id"] for r in rows])
    items = []
    for a in rows:
        q = quotation(store, segs[a["segment_id"]])
        q.update({"assignment_id": a["id"], "method": a["method"], "score": a["score"], "matched_text": a["matched_text"],
                  "match_start": a["char_start"], "match_end": a["char_end"], "negated": bool(a["negated"])})
        items.append(q)
    return {"code": code, "total": total, "items": items}


def theme_evidence(store: Store, theme_id: int) -> dict:
    t = store.one("SELECT * FROM themes WHERE id = ?", (theme_id,))
    if not t:
        raise NotFound(f"Theme {theme_id} not found")
    rep_ids = [r["segment_id"] for r in t["representative"] or []]
    con_ids = [r["segment_id"] for r in t["contradictory"] or []]
    segs = _segment_rows(store, rep_ids + con_ids)
    notes = store.query("SELECT * FROM annotations WHERE theme_id = ? ORDER BY id", (theme_id,))
    verified = {n["segment_id"]: n for n in notes if n["verified"] is not None}
    def pack(ids, extra):
        out = []
        for i, sid in enumerate(ids):
            if sid in segs:
                q = quotation(store, segs[sid])
                q.update(extra[i])
                q["researcher_verified"] = bool(verified[sid]["verified"]) if sid in verified else None
                out.append(q)
        return out
    return {
        "theme": t,
        "representative": pack(rep_ids, t["representative"] or []),
        "contradictory": pack(con_ids, t["contradictory"] or []),
        "contradictory_rule": "Passages assigned to the theme's codes whose lexicon sentiment polarity is opposite to the theme's majority polarity. "
                              "These are candidates for contradictory evidence and need researcher review.",
        "annotations": notes,
    }


def record_detail(store: Store, record_id: int, run_id: str | None = None) -> dict:
    rec = store.one("SELECT r.*, d.filename FROM records r JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?", (record_id,))
    if not rec:
        raise NotFound(f"Record {record_id} not found")
    out = {"record": rec, "segments": []}
    if run_id:
        segs = store.query("SELECT * FROM segments WHERE run_id = ? AND record_id = ? ORDER BY seg_index", (run_id, record_id))
        for s in segs:
            s["codes"] = store.query(
                "SELECT c.id, c.label, c.status, a.method, a.score, a.matched_text, a.char_start, a.char_end, a.negated "
                "FROM assignments a JOIN codes c ON c.id = a.code_id WHERE a.segment_id = ?", (s["id"],))
            s["verified"] = rec["text"][s["char_start"]:s["char_end"]] == s["text"]
        out["segments"] = segs
        code_ids = {c["id"] for s in segs for c in s["codes"] if c["status"] in ACTIVE}
        out["themes"] = [{"id": t["id"], "label": t["label"], "status": t["status"]} for t in theme_table(store, run_id) if code_ids & set(t["code_ids"])]
    return out


# ---------------------------------------------------------------------- clusters & topics

def cluster_view(store: Store, run_id: str) -> dict:
    run = get_run_or_404(store, run_id)
    segs = load_segments(store, run_id)
    assigns = load_assignments(store, run_id)
    codes = load_codes(store, run_id)
    label = dict(zip(codes["id"], codes["label"]))
    clusters = store.query("SELECT * FROM clusters WHERE run_id = ? ORDER BY cluster", (run_id,))
    code_by_pos = assigns.groupby("segment_pos")["code_id"].apply(list).to_dict() if len(assigns) else {}
    rep_ids = [r["segment_id"] for c in clusters for r in (c["representative"] or [])]
    seg_rows = _segment_rows(store, rep_ids)
    out = []
    for c in clusters:
        members = segs[segs["cluster"] == c["cluster"]]
        code_counts: dict = {}
        for p in members["pos"]:
            for cid in code_by_pos.get(p, []):
                code_counts[cid] = code_counts.get(cid, 0) + 1
        top_codes = sorted(code_counts.items(), key=lambda x: -x[1])[:10]
        out.append({
            **c,
            "passages": int(len(members)),
            "records": int(members["record_id"].nunique()),
            "respondents": int(members["record_key"].nunique()),
            "n_codes": len(code_counts),
            "top_codes": [{"id": k, "label": label.get(k, str(k)), "segments": v} for k, v in top_codes],
            "representative_passages": [quotation(store, seg_rows[r["segment_id"]]) | {"similarity": r["similarity"]}
                                        for r in (c["representative"] or []) if r["segment_id"] in seg_rows],
            "mean_sentiment": None if members["sentiment"].isna().all() else round(float(members["sentiment"].mean()), 4),
        })
    unassigned = int((segs["cluster"] == -1).sum()) if len(segs) else 0
    return {
        "metrics": (run.get("metrics") or {}).get("clustering", {}),
        "clusters": out,
        "unassigned": percentage(unassigned, len(segs), "segments without extracted features (not clustered) / analysed segments", "segment"),
    }


def topic_view(store: Store, run_id: str) -> dict:
    run = get_run_or_404(store, run_id)
    topics = store.query("SELECT * FROM topics WHERE run_id = ? ORDER BY topic", (run_id,))
    segs = load_segments(store, run_id)
    n = len(segs)
    for t in topics:
        t["share"] = percentage(int((segs["topic"] == t["topic"]).sum()), n, "segments whose dominant topic is this topic / analysed segments", "segment")
        top = segs[segs["topic"] == t["topic"]].sort_values("topic_weight", ascending=False).head(3)
        t["examples"] = [{"segment_id": int(r.segment_id), "text": r.text, "weight": r.topic_weight} for r in top.itertuples()]
    return {"metrics": (run.get("metrics") or {}).get("topics", {}), "topics": topics,
            "note": "Topics are statistical word distributions, not researcher-approved themes."}


def temporal_view(store: Store, run_id: str) -> dict:
    run = get_run_or_404(store, run_id)
    p = store.run_dir(run["project_id"], run_id) / "temporal.json"
    if not p.exists():
        return {"available": False, "reason": "Temporal results are not available for this run."}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- snapshot & statistics

def live_statistics(store: Store, run_id: str) -> dict:
    segs = load_segments(store, run_id)
    codes = _codes_keyed(load_codes(store, run_id))
    assigns = load_assignments(store, run_id)
    run = get_run_or_404(store, run_id)
    labels = dict(zip(codes["code_key"], codes["label"]))
    freq = code_frequency_table(codes, assigns, segs)
    top = freq["code_key"].head(15).tolist() if len(freq) else []
    return {
        "coverage": coverage_metrics(assigns, segs),
        "cooccurrence_top": cooccurrence_table(assigns, segs, labels, run["config"]["cooccurrence_level"], 30),
        "group_comparison": group_comparison(assigns, segs, labels, top),
    }


def run_snapshot(store: Store, run_id: str) -> dict:
    run = get_run_or_404(store, run_id)
    segs = load_segments(store, run_id)
    codes = code_table(store, run_id, include_inactive=False)
    g = graph_view(store, run_id, similarity=False)
    lab = {n["id"]: n["label"] for n in g["nodes"]}
    names = ", ".join(d["filename"] for d in (store.get_dataset(x) for x in run["dataset_ids"]) if d)
    return {
        "run_id": run_id,
        "label": f"{run['label']} · {names}",
        "n_records": int(segs["record_id"].nunique()),
        "codes": [{"label": c["label"], "records": c["records"]} for c in codes],
        "edges": [{"a": lab[e["source"]], "b": lab[e["target"]], "weight": e["weight"]} for e in g["edges"] if e["type"] == "cooccurrence"],
        "communities": {n["label"]: n["community"] for n in g["nodes"]},
        "themes": [{"label": t["label"], "status": t["status"], "code_labels": [c["label"] for c in t["codes"]]} for t in theme_table(store, run_id)],
        "config": {k: v for k, v in run["config"].items() if k not in ("codebook", "synonyms", "custom_stopwords")},
    }
