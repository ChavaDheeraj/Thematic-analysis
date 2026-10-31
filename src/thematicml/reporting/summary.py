"""Assemble the numerical dashboard for one analysis run (sections A–H)."""
from __future__ import annotations

import itertools

import pandas as pd

from thematicml.analytics.percentages import percentage
from thematicml.analytics.sentiment import LABELS as SENTIMENT_LABELS
from thematicml.results import (
    code_table,
    get_run_or_404,
    live_statistics,
    load_assignments,
    load_segments,
    temporal_view,
    theme_table,
)
from thematicml.storage.database import Store

EXPORT_FORMATS = [
    {"format": "pdf", "label": "Complete report (PDF)"},
    {"format": "docx", "label": "Editable report (DOCX)"},
    {"format": "html", "label": "Report (HTML)"},
    {"format": "xlsx", "label": "Metrics & tables (XLSX)"},
    {"format": "cleaned_csv", "label": "Cleaned dataset (CSV)"},
    {"format": "assignments_csv", "label": "Initial coding: statements and codes (CSV)"},
    {"format": "codebook_csv", "label": "Codebook with quotations (CSV)"},
    {"format": "categories_csv", "label": "Categories (CSV)"},
    {"format": "themes_csv", "label": "Themes and sub-themes (CSV)"},
    {"format": "theme_links_csv", "label": "Thematic map links (CSV)"},
    {"format": "codes_csv", "label": "Codes (CSV)"},
    {"format": "results_json", "label": "Full analysis results (JSON)"},
    {"format": "metrics_json", "label": "Run metrics (JSON)"},
]


def build_dashboard(store: Store, run_id: str) -> dict:
    run = get_run_or_404(store, run_id)
    pid = run["project_id"]
    m = run.get("metrics") or {}
    datasets = [store.get_dataset(d) for d in run["dataset_ids"]]
    datasets = [d for d in datasets if d]

    total = sum((d.get("quality") or {}).get("total_records", 0) for d in datasets)
    valid = sum((d.get("quality") or {}).get("valid_records", 0) for d in datasets)
    missing = sum((d.get("quality") or {}).get("missing_text", 0) for d in datasets)
    dups = sum((d.get("quality") or {}).get("duplicate_records", 0) for d in datasets)
    excluded = sum((d.get("quality") or {}).get("excluded_records", 0) for d in datasets)
    bad_ts = sum((d.get("quality") or {}).get("invalid_timestamps", 0) for d in datasets)
    other_lang = sum((d.get("quality") or {}).get("non_english_records", 0) for d in datasets)

    segs = load_segments(store, run_id)
    codes = code_table(store, run_id)
    active = [c for c in codes if c["status"] in ("candidate", "approved")]
    themes = theme_table(store, run_id)
    main_themes = [t for t in themes if not t.get("parent_id")]
    stats = live_statistics(store, run_id)
    ds_meta = m.get("dataset", {})

    project_runs = store.list_runs(pid)
    completed_runs = [r for r in project_runs if r["status"] == "completed"]
    analysed_ids = {d for r in completed_runs for d in r["dataset_ids"]}
    not_analysed = f"AND r.dataset_id NOT IN ({','.join('?' * len(analysed_ids))})" if analysed_ids else ""
    awaiting = store.scalar(
        "SELECT COUNT(*) FROM records r JOIN datasets d ON d.id = r.dataset_id WHERE d.project_id = ? AND r.is_valid = 1 " + not_analysed,
        (pid, *analysed_ids),
    ) or 0

    # ---------------- A
    overview = {
        "total_records": total,
        "valid_records": percentage(valid, total, "valid records / parsed records", "record"),
        "unique_codes": len(active),
        "candidate_themes": sum(1 for t in main_themes if t["status"] == "candidate"),
        "approved_themes": sum(1 for t in main_themes if t["status"] == "approved"),
        "total_themes": len(main_themes),
        "total_uploads": store.scalar("SELECT COUNT(*) FROM datasets WHERE project_id = ?", (pid,)),
        "uploads_in_run": len(datasets),
        "respondents": ds_meta.get("respondents"),
        "total_words": ds_meta.get("total_words"),
        "segments": len(segs),
        "mean_words_per_record": ds_meta.get("mean_words_per_record"),
        "median_words_per_record": ds_meta.get("median_words_per_record"),
        "categories": len({c["category"] for c in active if c["category"]}),
        "total_runs": len(project_runs),
        "last_analysis": max((r["finished_at"] for r in completed_runs if r["finished_at"]), default=None),
        "processing_s": run.get("duration_s"),
        "datasets": [{"id": d["id"], "filename": d["filename"]} for d in datasets],
    }

    # ---------------- B
    coverage = stats["coverage"]
    quality = {
        "valid": percentage(valid, total, "valid records / parsed records", "record"),
        "missing_text": percentage(missing, total, "records with empty text / parsed records", "record"),
        "duplicates": percentage(dups, total, "repeat (record ID, non-empty text) pairs / parsed records; empty rows count as missing", "record"),
        "excluded": percentage(excluded, total, "records excluded from analysis / parsed records", "record"),
        "invalid_metadata": percentage(bad_ts, total, "records with unparseable timestamps / parsed records", "record"),
        "non_english": percentage(other_lang, total, "records clearly in a language other than English / parsed records", "record"),
        "language_policy": sorted({(d.get("quality") or {}).get("language_policy", "keep") for d in datasets}),
        "cross_dataset_duplicates": ds_meta.get("cross_dataset_duplicates", 0),
        "awaiting_analysis": int(awaiting),
        "uncoded_segments": coverage["uncoded_segments"],
        "duplicate_policy": sorted({(d.get("quality") or {}).get("duplicate_policy", "flag") for d in datasets}),
        "warnings": [w for d in datasets for w in (d.get("warnings") or []) + ((d.get("quality") or {}).get("warnings") or [])] + (run.get("warnings") or []),
    }

    # ---------------- C
    sorted_active = sorted(active, key=lambda c: -c["segments"])
    coding = {
        "unique_codes": len(active),
        "total_assignments": coverage["total_assignments"],
        "coded_segments": coverage["coded_segments"],
        "coded_records": coverage["coded_records"],
        "coded_respondents": coverage["coded_respondents"],
        "codes_per_respondent": coverage["codes_per_respondent"],
        "unassigned_segments": coverage["uncoded_segments"],
        "awaiting_review": sum(1 for c in active if c["status"] == "candidate"),
        "approved": sum(1 for c in active if c["status"] == "approved"),
        "rejected": sum(1 for c in codes if c["status"] == "rejected"),
        "merged": sum(1 for c in codes if c["status"] == "merged"),
        "split": sum(1 for c in codes if c["method"] == "researcher" and c["origin"] == "researcher"),
        "most_frequent": sorted_active[:10],
        "least_frequent": [c for c in sorted_active[::-1] if c["segments"] > 0][:10],
        "table": codes,
        "review_progress": percentage(sum(1 for c in codes if c["status"] != "candidate"), len(codes), "codes reviewed (approved, rejected or merged) / all codes", "code"),
        "denominators": {"segments": len(segs), "records": int(segs["record_id"].nunique()), "respondents": int(segs["record_key"].nunique())},
    }

    # ---------------- D
    topics_m = m.get("topics", {})
    themes_d = {
        "topics": topics_m.get("n_topics") if topics_m.get("available") else None,
        "topic_metrics": topics_m,
        "candidate": overview["candidate_themes"],
        "approved": overview["approved_themes"],
        "subthemes": sum(1 for t in themes if t.get("parent_id")),
        "table": themes,
        "codes_in_approved_themes": percentage(
            len({c for t in themes if t["status"] == "approved" for c in t["code_ids"]}), len(active),
            "active codes included in an approved theme / active codes", "code"),
        "distinction": "Topics are statistical outputs of the topic model; themes are candidate groupings that a researcher must review and approve.",
    }

    # ---------------- E
    sentiment = {"available": False, "reason": (m.get("sentiment") or {}).get("reason", "Sentiment not computed.")}
    if len(segs) and segs["sentiment_label"].notna().any():
        counts = segs["sentiment_label"].value_counts().to_dict()
        assigns = load_assignments(store, run_id)
        pos_label = dict(zip(segs["pos"], segs["sentiment_label"]))
        by_theme = []
        for t in themes:
            ps = set(assigns[assigns["code_id"].isin(t["code_ids"])]["segment_pos"].tolist())
            c = pd.Series([pos_label[p] for p in ps]).value_counts().to_dict() if ps else {}
            by_theme.append({"theme_id": t["id"], "label": t["label"], "segments": len(ps),
                             **{lab: int(c.get(lab, 0)) for lab in SENTIMENT_LABELS}})
        sentiment = {
            "available": True,
            "counts": {lab: int(counts.get(lab, 0)) for lab in SENTIMENT_LABELS},
            "percentages": {lab: percentage(int(counts.get(lab, 0)), len(segs), f"{lab} segments / analysed segments", "segment") for lab in SENTIMENT_LABELS},
            "by_theme": by_theme,
            "definitions": {k: v for k, v in (m.get("sentiment") or {}).items() if k in ("method", "labels", "limitations")},
        }
    theme_co = []
    if len(themes) > 1:
        assigns = load_assignments(store, run_id)
        rec_of = dict(zip(segs["pos"], segs["record_id"]))
        rec_sets = {t["id"]: {rec_of[p] for p in assigns[assigns["code_id"].isin(t["code_ids"])]["segment_pos"]} for t in themes}
        lab = {t["id"]: t["label"] for t in themes}
        for a, b in itertools.combinations(rec_sets, 2):
            n = len(rec_sets[a] & rec_sets[b])
            if n:
                theme_co.append({"theme_a": lab[a], "theme_b": lab[b], "records": n})
        theme_co.sort(key=lambda r: -r["records"])
    temporal = temporal_view(store, run_id)
    statistics = {
        "sentiment": sentiment,
        "code_cooccurrence": stats["cooccurrence_top"],
        "theme_cooccurrence": theme_co[:30],
        "group_comparison": stats["group_comparison"],
        "temporal": {k: temporal.get(k) for k in ("available", "reason", "frequency", "periods", "segments_per_period", "volume_trend", "coverage_pct")}
                    | {"codes": [{"label": c["label"], "trend": c["trend"], "share_pct": c["share_pct"], "change_points": c["change_points"]} for c in temporal.get("codes", [])[:8]]},
    }

    # ---------------- F
    same_data = [r for r in completed_runs if sorted(r["dataset_ids"]) == sorted(run["dataset_ids"])]
    experiments = [{
        "run_id": r["id"], "label": r["label"], "finished_at": r["finished_at"], "duration_s": r["duration_s"],
        "clustering": (r["config"] or {}).get("clustering_method"), "topics": (r["config"] or {}).get("topic_method"),
        "graph": (r["config"] or {}).get("graph_method"),
        "n_clusters": ((r["metrics"] or {}).get("clustering") or {}).get("n_clusters"),
        "silhouette": ((r["metrics"] or {}).get("clustering") or {}).get("silhouette_cosine"),
        "coherence": ((r["metrics"] or {}).get("topics") or {}).get("mean_coherence_umass"),
        "communities": (((r["metrics"] or {}).get("graph") or {}).get("communities") or {}).get("n_multi_node_communities"),
        "modularity": (((r["metrics"] or {}).get("graph") or {}).get("communities") or {}).get("modularity"),
        "current": r["id"] == run_id,
    } for r in same_data]
    ml = {
        "features": m.get("features", {}),
        "coding": m.get("coding", {}),
        "clustering": m.get("clustering", {}),
        "topics": topics_m,
        "graph": m.get("graph", {}),
        "runtime": m.get("runtime", {}),
        "config": run["config"],
        "package_version": run.get("package_version"),
        "last_successful_run": overview["last_analysis"],
        "experiments": experiments,
        "accuracy_note": "No single accuracy figure is reported for the unsupervised pipeline. Classification metrics appear only "
                         "when supervised coding is enabled and labelled records allow a held-out evaluation.",
    }

    # ---------------- G
    activity = store.activity(pid, 25)

    return {
        "run": {k: run[k] for k in ("id", "label", "status", "created_at", "finished_at", "duration_s", "dataset_ids", "warnings")},
        "overview": overview,
        "quality": quality,
        "coding": coding,
        "themes": themes_d,
        "statistics": statistics,
        "ml": ml,
        "activity": activity,
        "exports": EXPORT_FORMATS,
    }
