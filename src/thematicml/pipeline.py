"""The analysis pipeline.

Stages run in order and the full intermediate state is checkpointed to disk
after each one, so a failed or interrupted run can be resumed from the stage
that failed without recomputing earlier stages.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Callable, Generator

import joblib
import numpy as np
import pandas as pd

from thematicml import __version__
from thematicml.analytics import code_frequency_table, cooccurrence_table, coverage_metrics, group_comparison
from thematicml.analytics.percentages import percentage
from thematicml.analytics.sentiment import DEFINITIONS as SENTIMENT_DEFINITIONS
from thematicml.analytics.sentiment import LABELS as SENTIMENT_LABELS
from thematicml.analytics.sentiment import score_texts, sentiment_available
from thematicml.clustering import ClusteringError, cluster_segments, describe_clusters
from thematicml.coding import (ASSIGN_COLUMNS, CODE_COLUMNS, apply_codebook, extract_phrase_codes, keyword_regex,
                               train_record_classifier)
from thematicml.config import AnalysisConfig
from thematicml.data import segment_records
from thematicml.features import count_matrix, fit_tfidf
from thematicml.graph import (
    build_code_graph,
    detect_communities,
    gnn_status,
    graph_metrics,
    link_prediction_baseline,
    node_metrics,
)
from thematicml.storage.database import Store, new_id, now
from thematicml.temporal import temporal_analysis
from thematicml.themes import discover_themes
from thematicml.topics import discover_topics

STAGES: list[tuple[str, str]] = [
    ("preprocess", "Preprocessing & segmentation"),
    ("features", "Feature extraction"),
    ("coding", "Initial coding"),
    ("clustering", "Pattern clustering"),
    ("topics", "Topic discovery"),
    ("statistics", "Statistical analysis"),
    ("graph", "Graph construction"),
    ("themes", "Theme discovery"),
    ("temporal", "Temporal analysis"),
    ("persist", "Saving results"),
]
STAGE_NAMES = [s for s, _ in STAGES]


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (MB). Includes everything else the process holds."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            # the process handle is pointer-sized; without these types ctypes truncates it to 32 bits and the call fails
            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) or not pmc.PeakWorkingSetSize:
                return None
            return round(pmc.PeakWorkingSetSize / 2**20, 1)
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)
    except Exception:
        return None


def repeated_passages(records: pd.DataFrame, min_words: int = 8) -> pd.Series:
    """True for records whose (respondent, text) pair already appeared in an earlier dataset of the run.

    The same page or file uploaded twice would otherwise be counted twice in every prevalence
    and co-occurrence figure. Only the same respondent giving the same text counts, so two
    sources quoting one passage both stay; repeats inside one dataset are left to its duplicate
    policy; and short texts are never repeats: identical short answers ("Yes", "Not sure") in two
    survey waves are legitimate.
    """
    norm = records["text"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    keys = records["record_key"] if "record_key" in records else pd.Series([""] * len(records), index=records.index)
    first: dict[tuple, str] = {}
    out = []
    for key, text, ds, words in zip(keys, norm, records["dataset_id"], records["word_count"]):
        if words < min_words:
            out.append(False)
            continue
        out.append(first.setdefault((key, text), ds) != ds)
    return pd.Series(out, index=records.index)


# ---------------------------------------------------------------------- stages

def _stage_preprocess(store: Store, run: dict, cfg: AnalysisConfig, st: dict) -> None:
    ids = run["dataset_ids"]
    q = ",".join("?" * len(ids))
    rows = store.query(
        f"SELECT r.id AS record_id, r.dataset_id, r.record_key, r.text, r.timestamp, r.group_value, r.label, r.word_count "
        f"FROM records r WHERE r.dataset_id IN ({q}) AND r.is_valid = 1 ORDER BY r.dataset_id, r.row_index",
        ids,
    )
    if not rows:
        raise PipelineError("preprocess", "No valid records found. Map the dataset columns and validate before analysing.")
    records = pd.DataFrame(rows)
    repeats = 0
    if len(ids) > 1:
        # generated row keys are only unique within one upload
        gen = records["record_key"].str.startswith("row-")
        records.loc[gen, "record_key"] = records.loc[gen, "dataset_id"] + ":" + records.loc[gen, "record_key"]
        if cfg.dedupe_across_datasets:
            rep = repeated_passages(records)
            repeats = int(rep.sum())
            records = records[~rep].reset_index(drop=True)
            if repeats:
                st["warnings"].append(f"{repeats} passage(s) repeated from another upload in this run (for example the same page "
                                      "saved twice) were analysed once.")
    segs = segment_records(records, cfg.segmentation, cfg.min_segment_words)
    if segs.empty:
        raise PipelineError("preprocess", "Segmentation produced no passages.")
    segs = segs.merge(records[["record_id", "dataset_id", "record_key", "timestamp", "group_value"]], on="record_id", how="left")
    st["records"] = records
    st["segments"] = segs
    st["summary"]["dataset"] = {
        "analysed_records": int(len(records)),
        "respondents": int(records["record_key"].nunique()),
        "segments": int(len(segs)),
        "total_words": int(records["word_count"].sum()),
        "mean_words_per_record": round(float(records["word_count"].mean()), 2),
        "median_words_per_record": float(records["word_count"].median()),
        "segmentation": cfg.segmentation,
        "records_with_timestamp": int(records["timestamp"].notna().sum()),
        "cross_dataset_duplicates": repeats,
    }


def _stage_features(store, run, cfg, st) -> None:
    feats = fit_tfidf(st["segments"]["text"].tolist(), cfg)
    st["features"] = feats
    st["counts"] = count_matrix(st["segments"]["text"].tolist(), feats)
    st["summary"]["features"] = {
        **feats.summary(),
        "settings": {
            "lowercase": cfg.lowercase, "remove_stopwords": cfg.remove_stopwords, "keep_negations": cfg.keep_negations,
            "ngram_range": [cfg.ngram_min, cfg.ngram_max], "min_df": cfg.min_df, "max_df": cfg.max_df,
            "max_features": cfg.max_features, "sublinear_tf": True, "synonyms": len(cfg.synonyms),
        },
        # what was used: short texts relax the frequency filters instead of losing every term
        "used": {"min_df": feats.vectorizer.min_df, "max_df": feats.vectorizer.max_df, "relaxed": feats.relaxed},
    }


def _stage_coding(store, run, cfg, st) -> None:
    segs = st["segments"]
    codes, assigns = [pd.DataFrame(columns=CODE_COLUMNS)], [pd.DataFrame(columns=ASSIGN_COLUMNS)]
    skipped: list[str] = []
    support = cfg.min_code_support
    if cfg.coding_method in ("tfidf", "both"):
        skip = None
        if cfg.coding_method == "both" and cfg.complement_codebook and cfg.codebook:
            patterns = [keyword_regex(k) for r in cfg.codebook for k in r.keywords if k.strip()]

            def skip(term: str) -> bool:
                if any(p.search(term) for p in patterns):
                    skipped.append(term)
                    return True
                return False

        # a code must recur, but a short text cannot repeat a phrase in three passages: scale the support down
        support = min(cfg.min_code_support, max(1, len(segs) // 10))
        c, a = extract_phrase_codes(st["features"], segs, cfg.n_codes, support, cfg.phrase_boost, skip)
        if c.empty and support > 1:
            support = 1
            c, a = extract_phrase_codes(st["features"], segs, cfg.n_codes, support, cfg.phrase_boost, skip)
        codes.append(c)
        assigns.append(a)
    if cfg.coding_method in ("rules", "both"):
        if not cfg.codebook:
            st["warnings"].append("Rule-based coding was selected but the codebook is empty.")
        c, a = apply_codebook(cfg.codebook, segs)
        codes.append(c)
        assigns.append(a)
    supervised = {"available": False, "reason": "Supervised coding not enabled."}
    if cfg.supervised:
        c, a, supervised = train_record_classifier(st["records"], segs, st["features"], cfg.random_state)
        codes.append(c)
        assigns.append(a)
        if not supervised.get("available"):
            st["warnings"].append(f"Supervised coding skipped: {supervised.get('reason')}")
    codes_df = pd.concat([c for c in codes if len(c)] or [codes[0]], ignore_index=True).drop_duplicates("code_key")
    assign_df = pd.concat([a for a in assigns if len(a)] or [assigns[0]], ignore_index=True)
    if codes_df.empty:
        st["warnings"].append("This text has no recurring words or phrases to code. Add codebook rules, or analyse it together with other files.")
    st["codes"] = codes_df.reset_index(drop=True)
    st["assignments"] = assign_df.reset_index(drop=True)
    st["summary"]["coding"] = {
        "method": cfg.coding_method,
        "n_codes": int(len(codes_df)),
        "by_method": codes_df["method"].value_counts().to_dict() if len(codes_df) else {},
        "assignments": int(len(assign_df)),
        "supervised": supervised,
        "selection": f"top {cfg.n_codes} TF-IDF terms with support >= {support if cfg.coding_method != 'rules' else cfg.min_code_support} segments, phrase boost {cfg.phrase_boost}",
    }
    if skipped:
        st["summary"]["coding"]["tfidf_terms_left_to_codebook"] = {"count": len(skipped), "top": skipped[:30]}


def _stage_clustering(store, run, cfg, st) -> None:
    X = st["features"].matrix
    try:
        res = cluster_segments(X, cfg)
    except ClusteringError as exc:
        st["clusters"] = None
        st["summary"]["clustering"] = {"available": False, "reason": str(exc)}
        st["warnings"].append(f"Clustering unavailable: {exc}")
        return
    st["clusters"] = {
        "labels": res["labels"],
        "described": describe_clusters(X, st["features"].vocabulary, res["labels"], res["embedding"], res["embedded_rows"], 10, cfg.n_representative),
    }
    st["summary"]["clustering"] = {"available": True, **res["metrics"]}


def _stage_topics(store, run, cfg, st) -> None:
    X = st["features"].matrix
    if X.shape[0] < 4 or X.shape[1] < 4:
        st["topics"] = None
        st["summary"]["topics"] = {"available": False, "reason": "Topic modelling needs at least 4 segments and 4 terms."}
        return
    res = discover_topics(X, st["counts"], st["features"].vocabulary, cfg.topic_method, cfg.n_topics, cfg.n_top_terms,
                          cfg.topic_min_weight, cfg.random_state)
    st["topics"] = res
    st["summary"]["topics"] = {"available": True, **res["metrics"]}


def _stage_statistics(store, run, cfg, st) -> None:
    segs, codes, assigns = st["segments"], st["codes"], st["assignments"]
    if cfg.sentiment and sentiment_available():
        scores = score_texts(segs["text"].tolist())
        st["sentiment"] = scores
        labels = pd.Series([s["label"] for s in scores])
        counts = {lab: int((labels == lab).sum()) for lab in SENTIMENT_LABELS}
        st["summary"]["sentiment"] = {
            "available": True, "unit": "segment", "counts": counts,
            "percentages": {k: percentage(v, len(scores), f"{k} segments / analysed segments", "segment") for k, v in counts.items()},
            "mean_compound": round(float(np.mean([s["compound"] for s in scores])), 4),
            **SENTIMENT_DEFINITIONS,
        }
    else:
        st["sentiment"] = None
        reason = "Sentiment analysis disabled in the configuration." if not cfg.sentiment else \
            "Optional dependency vaderSentiment is not installed (pip install thematicml[sentiment])."
        st["summary"]["sentiment"] = {"available": False, "reason": reason}

    freq = code_frequency_table(codes, assigns, segs)
    st["frequency"] = freq
    labels = dict(zip(codes["code_key"], codes["label"]))
    top = freq["code_key"].head(15).tolist() if len(freq) else []
    st["summary"]["statistics"] = {
        "coverage": coverage_metrics(assigns, segs),
        "denominators": freq.attrs.get("denominators", {}),
        "cooccurrence_top": cooccurrence_table(assigns, segs, labels, cfg.cooccurrence_level, 30),
        "group_comparison": group_comparison(assigns, segs, labels, top),
    }


def _stage_graph(store, run, cfg, st) -> None:
    G, typed = build_code_graph(st["codes"], st["assignments"], st["segments"], st["features"].matrix,
                                cfg.cooccurrence_level, cfg.min_cooccurrence, cfg.similarity_edges, cfg.similarity_threshold, cfg.min_npmi)
    membership, cinfo = detect_communities(G, cfg.graph_method, cfg.resolution, cfg.random_state)
    nm = node_metrics(G)
    st["graph"] = {"membership": membership, "node_metrics": nm, "edges": typed}
    st["summary"]["graph"] = {
        **graph_metrics(G, membership),
        "communities": cinfo,
        "edge_types": {"cooccurrence": sum(1 for e in typed if e["type"] == "cooccurrence"),
                       "similarity": sum(1 for e in typed if e["type"] == "similarity")},
        "construction": {"nodes": "codes", "cooccurrence_level": cfg.cooccurrence_level, "min_cooccurrence": cfg.min_cooccurrence,
                         "min_npmi": cfg.min_npmi, "similarity_edges": cfg.similarity_edges, "similarity_threshold": cfg.similarity_threshold,
                         "community_edges": "cooccurrence"},
        "link_prediction_baseline": link_prediction_baseline(G, seed=cfg.random_state) if cfg.link_prediction_baseline else {"available": False, "reason": "disabled"},
        "gnn": gnn_status(),
    }
    st["nx_graph"] = G


def _stage_themes(store, run, cfg, st) -> None:
    sent = np.array([s["compound"] for s in st["sentiment"]]) if st.get("sentiment") else None
    dom = st["topics"]["dominant"] if st.get("topics") else None
    themes = discover_themes(st["codes"], st["assignments"], st["segments"], st["graph"]["membership"], st["features"].matrix,
                             cfg.theme_label_template, cfg.min_theme_codes, cfg.n_representative, sent, dom)
    st["themes"] = themes
    st["summary"]["themes"] = {"candidate_themes": len(themes), "source": "graph communities with >= %d codes" % cfg.min_theme_codes,
                               "label_template": cfg.theme_label_template}


def _stage_temporal(store, run, cfg, st) -> None:
    if not cfg.temporal:
        st["temporal"] = {"available": False, "reason": "Temporal analysis disabled in the configuration."}
        return
    if st["segments"]["timestamp"].isna().all():
        st["temporal"] = {"available": False, "reason": "No timestamp column was mapped or no timestamps could be parsed. Temporal analysis requires ordered data."}
        return
    labels = dict(zip(st["codes"]["code_key"], st["codes"]["label"]))
    sent = np.array([s["compound"] for s in st["sentiment"]]) if st.get("sentiment") else None
    st["temporal"] = temporal_analysis(st["segments"], st["assignments"], labels,
                                       [{"label": t["label"], "code_keys": t["code_keys"]} for t in st["themes"]],
                                       cfg.temporal_freq, cfg.temporal_top_codes, cfg.n_graph_windows, cfg.min_cooccurrence,
                                       cfg.random_state, sent)


def _stage_persist(store: Store, run: dict, cfg, st) -> None:
    rid = run["id"]
    segs, codes, assigns = st["segments"], st["codes"], st["assignments"]
    labels = st["clusters"]["labels"] if st.get("clusters") else None
    topics = st.get("topics")
    sent = st.get("sentiment")
    with store.connect() as con:
        for t in ("assignments", "codes", "themes", "clusters", "topics", "segments", "annotations"):
            con.execute(f"DELETE FROM {t} WHERE run_id = ?", (rid,))
        seg_rows = []
        for i, s in enumerate(segs.itertuples()):
            seg_rows.append((
                rid, i, int(s.record_id), int(s.seg_index), s.text, int(s.char_start), int(s.char_end),
                None if labels is None else int(labels[i]),
                None if topics is None else int(topics["dominant"][i]),
                None if topics is None else round(float(topics["dominant_weight"][i]), 4),
                None if sent is None else sent[i]["compound"],
                None if sent is None else sent[i]["label"],
            ))
        con.executemany(
            "INSERT INTO segments (run_id, pos, record_id, seg_index, text, char_start, char_end, cluster, topic, topic_weight, sentiment, sentiment_label) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", seg_rows)
        seg_ids = [r[0] for r in con.execute("SELECT id FROM segments WHERE run_id = ? ORDER BY pos", (rid,)).fetchall()]
        code_id = {}
        for c in codes.itertuples():
            code_id[c.code_key] = store.insert("codes", {
                "run_id": rid, "code_key": c.code_key, "label": c.label, "description": c.description, "method": c.method,
                "feature": c.feature, "category": c.category, "status": "candidate", "origin": "machine", "updated_at": now(),
            }, con)
        con.executemany(
            "INSERT INTO assignments (run_id, code_id, segment_id, method, score, matched_text, char_start, char_end, negated) VALUES (?,?,?,?,?,?,?,?,?)",
            [(rid, code_id[a.code_key], seg_ids[int(a.segment_pos)], a.method, None if pd.isna(a.score) else float(a.score),
              a.matched_text if isinstance(a.matched_text, str) else None,
              None if pd.isna(a.char_start) else int(a.char_start), None if pd.isna(a.char_end) else int(a.char_end),
              int(bool(a.negated))) for a in assigns.itertuples()],
        )
        if st.get("clusters"):
            for c in st["clusters"]["described"]:
                top_terms = [t["term"] for t in c["terms"][:3]]
                store.insert("clusters", {
                    "run_id": rid, "cluster": c["cluster"], "label": " / ".join(top_terms) or f"Cluster {c['cluster']}",
                    "size": c["size"], "terms": c["terms"],
                    "representative": [{"segment_id": seg_ids[r["segment_pos"]], "similarity": r["similarity"]} for r in c["representative"]],
                    "cohesion": c["cohesion"],
                }, con)
        if topics:
            for t in topics["topics"]:
                store.insert("topics", {"run_id": rid, "topic": t["topic"], "label": t["label"], "terms": t["terms"],
                                        "weights": t["weights"], "size": t["size"], "coherence": t["coherence_umass"]}, con)
        for th in st["themes"]:
            store.insert("themes", {
                "run_id": rid, "label": th["label"], "definition": th["definition"],
                "code_ids": [code_id[k] for k in th["code_keys"]], "status": "candidate", "source": th["source"],
                "community": th["community"],
                "representative": [{"segment_id": seg_ids[r["segment_pos"]], "codes_matched": r["codes_matched"], "similarity": r["similarity"]} for r in th["representative"]],
                "contradictory": [{"segment_id": seg_ids[r["segment_pos"]], "compound": r["compound"]} for r in th["contradictory"]],
                "majority_polarity": th["majority_polarity"], "associated_topic": th["associated_topic"],
                "origin": "machine", "updated_at": now(),
            }, con)

    # graph snapshot (node ids are code ids so it lines up with the database)
    rdir = store.run_dir(run["project_id"], rid)
    G = st["nx_graph"]
    import networkx as nx

    H = nx.relabel_nodes(G, {k: int(code_id[k]) for k in G.nodes()})
    key_of = {v: k for k, v in code_id.items()}
    for n, d in H.nodes(data=True):
        d["community"] = st["graph"]["membership"].get(key_of[n], -1)
        for k, v in list(d.items()):
            if v is None:
                d[k] = ""
    nx.write_graphml(H, rdir / "graph.graphml")
    joblib.dump({"features": st["features"], "segment_ids": seg_ids}, rdir / "features.joblib")
    (rdir / "temporal.json").write_text(json.dumps(_jsonable(st["temporal"])), encoding="utf-8")
    st["summary"]["temporal"] = _temporal_summary(st["temporal"])


def _temporal_summary(t: dict) -> dict:
    if not t.get("available"):
        return t
    return {
        "available": True, "frequency": t["frequency"], "n_periods": len(t["periods"]),
        "coverage_pct": t["coverage_pct"], "volume_trend": t["volume_trend"],
        "increasing_codes": [c["label"] for c in t["codes"] if c["trend"].get("trend") == "increasing"],
        "decreasing_codes": [c["label"] for c in t["codes"] if c["trend"].get("trend") == "decreasing"],
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


STAGE_FUNCS: dict[str, Callable] = {
    "preprocess": _stage_preprocess, "features": _stage_features, "coding": _stage_coding,
    "clustering": _stage_clustering, "topics": _stage_topics, "statistics": _stage_statistics,
    "graph": _stage_graph, "themes": _stage_themes, "temporal": _stage_temporal, "persist": _stage_persist,
}


# ---------------------------------------------------------------------- runner

def create_run(store: Store, project_id: str, dataset_ids: list[str], cfg: AnalysisConfig, label: str | None = None) -> dict:
    if not dataset_ids:
        raise ValueError("Select at least one dataset to analyse.")
    for did in dataset_ids:
        ds = store.get_dataset(did)
        if ds is None or ds["project_id"] != project_id:
            raise ValueError(f"Dataset {did} does not belong to this project.")
        if ds["status"] not in ("ready", "analysed"):
            raise ValueError(f"Dataset '{ds['filename']}' is not ready (status: {ds['status']}). Map and validate its columns first.")
    rid = new_id("run")
    n = (store.scalar("SELECT COUNT(*) FROM runs WHERE project_id = ?", (project_id,)) or 0) + 1
    store.insert("runs", {
        "id": rid, "project_id": project_id, "dataset_ids": dataset_ids, "label": label or f"Run {n}",
        "config": cfg.model_dump(), "status": "queued", "created_at": now(), "package_version": __version__,
        "completed_stages": [], "stage_timings": {}, "warnings": [],
    })
    store.log(project_id, "analysis_queued", {"run_id": rid, "datasets": dataset_ids})
    return store.get_run(rid)


def run_stages(store: Store, run_id: str, on_progress: Callable[[dict], None] | None = None) -> Generator[str, None, dict]:
    """Run the pipeline one stage at a time, yielding each stage's name once it is saved; returns the final run.

    ``execute_run`` drives it to the end. The browser edition yields to the page between stages,
    because a browser tab has no threads to run an analysis in the background.
    """
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    cfg = AnalysisConfig(**run["config"])
    rdir = store.run_dir(run["project_id"], run_id)
    ckpt = rdir / "checkpoint.joblib"
    completed: list[str] = list(run.get("completed_stages") or [])
    if completed and ckpt.exists():
        st = joblib.load(ckpt)
    else:
        completed = []
        st = {"summary": {}, "warnings": []}
    timings = dict(run.get("stage_timings") or {})
    started = time.perf_counter()
    store.update("runs", {"id": run_id}, {"status": "running", "started_at": run.get("started_at") or now(), "error": None, "failed_stage": None})

    for i, name in enumerate(STAGE_NAMES):
        if name in completed:
            continue
        store.update("runs", {"id": run_id}, {"stage": name, "progress": round(len(completed) / len(STAGES), 3)})
        if on_progress:
            on_progress({"run_id": run_id, "stage": name, "progress": len(completed) / len(STAGES)})
        t0 = time.perf_counter()
        try:
            STAGE_FUNCS[name](store, run, cfg, st)
        except Exception as exc:
            msg = str(exc) if isinstance(exc, (PipelineError, ValueError, RuntimeError)) else f"{type(exc).__name__}: {exc}"
            (rdir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            store.update("runs", {"id": run_id}, {"status": "failed", "failed_stage": name, "error": msg,
                                                   "stage_timings": timings, "warnings": st["warnings"]})
            store.log(run["project_id"], "analysis_failed", {"run_id": run_id, "stage": name, "error": msg})
            return store.get_run(run_id)
        timings[name] = round(time.perf_counter() - t0, 3)
        completed.append(name)
        if name != "persist":
            joblib.dump(st, ckpt)
        store.update("runs", {"id": run_id}, {"completed_stages": completed, "stage_timings": timings,
                                               "progress": round(len(completed) / len(STAGES), 3)})
        yield name

    duration = round(sum(timings.values()), 3)
    summary = _jsonable(st["summary"])
    summary["runtime"] = {"stage_timings_s": timings, "total_s": duration, "peak_memory_mb": peak_memory_mb(),
                          "memory_note": "peak resident memory of the analysis process", "package_version": __version__,
                          "random_state": cfg.random_state}
    store.update("runs", {"id": run_id}, {
        "status": "completed", "stage": None, "progress": 1.0, "finished_at": now(), "duration_s": duration,
        "peak_memory_mb": summary["runtime"]["peak_memory_mb"], "metrics": summary, "warnings": st["warnings"],
    })
    for did in run["dataset_ids"]:
        store.update("datasets", {"id": did}, {"status": "analysed"})
    if ckpt.exists():
        ckpt.unlink()
    store.log(run["project_id"], "analysis_completed", {"run_id": run_id, "duration_s": duration, "wall_s": round(time.perf_counter() - started, 3)})
    return store.get_run(run_id)


def execute_run(store: Store, run_id: str, on_progress: Callable[[dict], None] | None = None) -> dict:
    stages = run_stages(store, run_id, on_progress)
    while True:
        try:
            next(stages)
        except StopIteration as finished:
            return finished.value
