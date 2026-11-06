"""Export analysis results and reports.

Reports (PDF, DOCX, HTML) are rendered from one report model so that their
content is identical. Machine-generated candidates and researcher-approved
findings are labelled throughout.
"""
from __future__ import annotations

import html
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from thematicml import __version__
from thematicml.reporting.summary import build_dashboard
from thematicml.results import (
    cluster_view,
    graph_view,
    load_segments,
    temporal_view,
    theme_evidence,
    topic_view,
)
from thematicml.storage.database import Store, now
from thematicml.thematic import cleaned_frame, cleaned_view, codebook, notes_view, recurring_ideas, thematic_structure

TABULAR = {"codes_csv", "assignments_csv", "themes_csv", "clusters_csv", "graph_nodes_csv", "graph_edges_csv", "cleaned_csv", "codebook_csv",
           "categories_csv", "theme_links_csv"}
ALL_FORMATS = TABULAR | {"results_json", "metrics_json", "temporal_json", "xlsx", "docx", "pdf", "html", "graphml"}

STATUS_LABEL = {"approved": "Researcher-approved", "candidate": "Machine-generated candidate", "rejected": "Rejected by researcher",
                "needs_revision": "Needs revision", "merged": "Merged"}


def _fmt_pct(p: dict | None) -> str:
    if not p or p.get("value") is None:
        return "n/a"
    return f"{p['value']:.2f}% ({p['numerator']}/{p['denominator']})"


# ---------------------------------------------------------------------- tables

def codes_frame(dash: dict) -> pd.DataFrame:
    rows = [{
        "code_id": c["id"], "label": c["label"], "status": STATUS_LABEL.get(c["status"], c["status"]), "method": c["method"],
        "category": c["category"], "description": c["description"], "assignments": c["assignments"], "segments": c["segments"],
        "segment_pct": c["segment_pct"], "records": c["records"], "record_pct": c["record_pct"], "respondents": c["respondents"],
        "respondent_pct": c["respondent_pct"], "respondent_ci95_low": c["respondent_ci95"][0], "respondent_ci95_high": c["respondent_ci95"][1],
    } for c in dash["coding"]["table"]]
    return pd.DataFrame(rows)


def themes_frame(dash: dict, structure: dict) -> pd.DataFrame:
    subs = {t["id"]: "; ".join(s["label"] for s in t["subthemes"]) for t in structure["themes"]}
    parent = {s["id"]: t["label"] for t in structure["themes"] for s in t["subthemes"] if s["id"]}
    rows = [{
        "theme_id": t["id"], "label": t["label"], "status": STATUS_LABEL.get(t["status"], t["status"]), "origin": t["origin"],
        "definition": t["definition"], "sub_theme_of": parent.get(t["id"]), "sub_themes": subs.get(t["id"]),
        "codes": "; ".join(c["label"] for c in t["codes"]), "segments": t["segments"],
        "segment_pct": t["prevalence"]["segment"]["value"], "record_pct": t["prevalence"]["record"]["value"],
        "respondents": t["prevalence"]["respondent"]["numerator"], "respondent_pct": t["prevalence"]["respondent"]["value"],
        "supporting_quotations": t["supporting_quotations"], "contradictory_examples": t["contradictory_count"],
        "notes": t.get("notes"),
    } for t in dash["themes"]["table"]]
    return pd.DataFrame(rows)


def codebook_frame(store: Store, run_id: str) -> pd.DataFrame:
    rows = []
    for c in codebook(store, run_id):
        q = c["quotations"]
        rows.append({
            "code": c["label"], "definition": c["description"], "category": c["category"], "suggested_category": c["suggested_category"],
            "status": STATUS_LABEL.get(c["status"], c["status"]), "passages": c["segments"], "records": c["records"],
            "quotation_1": q[0]["text"] if q else None, "source_1": q[0]["source_file"] if q else None,
            "quotation_2": q[1]["text"] if len(q) > 1 else None, "source_2": q[1]["source_file"] if len(q) > 1 else None,
        })
    return pd.DataFrame(rows)


def categories_frame(structure: dict) -> pd.DataFrame:
    rows = [{"category": c["name"], "origin": "suggested" if c["suggested"] else "codebook or researcher", "codes": "; ".join(x["label"] for x in c["codes"]),
             "suggested_codes": "; ".join(x["label"] for x in c["codes"] if x["suggested"]), "passages": c["segments"], "passage_pct": c["share"]["value"]}
            for c in structure["categories"]]
    if structure["uncategorised"]:
        rows.append({"category": "(not yet categorised)", "origin": "", "codes": "; ".join(x["label"] for x in structure["uncategorised"]),
                     "suggested_codes": "", "passages": None, "passage_pct": None})
    return pd.DataFrame(rows)


def theme_links_frame(structure: dict) -> pd.DataFrame:
    unit = structure["links"]["unit"]
    return pd.DataFrame([{
        "theme_a": r["a_label"], "theme_b": r["b_label"], "relation": r["relation"], f"shared_{unit}s": r["shared"], "expected_by_chance": r["expected"],
        "lift": r["lift"], "p_adjusted_bh": r["p_adjusted"], "strongest_code_pair": " + ".join(r.get("via") or []),
        "quotation": (r.get("quote") or {}).get("text"),
    } for r in structure["links"]["rows"]])


def assignments_frame(store: Store, run_id: str) -> pd.DataFrame:
    rows = store.query(
        "SELECT c.id AS code_id, c.label AS code, c.status AS code_status, a.method, a.score, a.negated, s.id AS segment_id, "
        "s.text AS quotation, s.char_start, s.char_end, a.matched_text, r.id AS record_id, r.row_index + 1 AS source_row, "
        "r.record_key AS respondent, "
        "r.timestamp, r.group_value, d.filename AS source_file FROM assignments a JOIN codes c ON c.id = a.code_id "
        "JOIN segments s ON s.id = a.segment_id JOIN records r ON r.id = s.record_id JOIN datasets d ON d.id = r.dataset_id "
        "WHERE a.run_id = ? ORDER BY c.id, s.pos", (run_id,))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- report model

def report_model(store: Store, run_id: str) -> dict:
    dash = build_dashboard(store, run_id)
    run = store.get_run(run_id)
    project = store.get_project(run["project_id"])
    cfg = run["config"]
    ml = dash["ml"]
    sections: list[tuple[str, list]] = []

    ov, q = dash["overview"], dash["quality"]
    sections.append(("Dataset profile", [
        ("table", ["Metric", "Value"], [
            ["Source files", ", ".join(d["filename"] for d in ov["datasets"])],
            ["Parsed records", ov["total_records"]], ["Valid records", _fmt_pct(q["valid"])],
            ["Missing text", _fmt_pct(q["missing_text"])], ["Duplicates", _fmt_pct(q["duplicates"])],
            ["Excluded", _fmt_pct(q["excluded"])], ["Respondents", ov["respondents"]], ["Text segments", ov["segments"]],
            ["Total words", ov["total_words"]], ["Mean / median words per record", f"{ov['mean_words_per_record']} / {ov['median_words_per_record']}"],
        ]),
    ]))
    if len(ov["datasets"]) > 1:
        segs = load_segments(store, run_id)
        per = segs.groupby("filename").agg(records=("record_id", "nunique"), passages=("segment_id", "size")).sort_index()
        sections.append(("Files analysed together", [
            ("p", f"One thematic analysis of {len(ov['datasets'])} files: the codes, categories, themes, thematic map and statistics in this "
                  "report cover all of them. Passages that the same source repeated in another file are analysed once."),
            ("table", ["File", "Records analysed", "Passages"], [[f, int(r.records), int(r.passages)] for f, r in per.iterrows()]),
        ]))
    sections.append(("Research questions", [("p", project.get("description") or "No research questions were recorded for this project.")]))
    cleaned = cleaned_view(store, run_id, size=1)
    sections.append(("Cleaned dataset", [
        ("p", "Cleaning removes whole records and page furniture; it does not rewrite anyone's words, so quotations stay verbatim."),
        ("table", ["Step", "Records"], [["Read from the files", cleaned["parsed"]]] + [[f"Removed: {r['label']}", r["records"]] for r in cleaned["removed"]]
         + [["Cleaned records analysed", cleaned["kept"]], ["Flagged duplicates kept in the analysis", cleaned["flagged_duplicates"]], ["Passages", cleaned["segments"]]]),
    ]))
    structure = thematic_structure(store, run_id)
    ideas, notes = recurring_ideas(store, run_id), notes_view(store, run_id)
    note_blocks = [("table", ["Recurring idea (repeated words)", "Passages", "Files"],
                    [[", ".join(i["terms"]), _fmt_pct(i["share"]), i["files"]] for i in ideas["ideas"]])] if ideas["ideas"] else []
    note_blocks.append(("p", ideas["note"]))
    note_blocks += [("p", f"Researcher memo: {m['note']}") for m in reversed(notes["memos"])]
    note_blocks += [("p", f"Note on code {c['label']}: {c['notes']}") for c in notes["codes"]]
    note_blocks += [("p", f"Note on theme {t['label']}: {t['notes']}") for t in notes["themes"]]
    sections.append(("Notes and recurring ideas", note_blocks))
    sections.append(("Preprocessing settings", [("table", ["Setting", "Value"], [
        ["Segmentation", f"{cfg['segmentation']} (merge passages under {cfg['min_segment_words']} words)"],
        ["Lowercase", cfg["lowercase"]], ["Stopword removal", cfg["remove_stopwords"]], ["Negations kept", cfg["keep_negations"]],
        ["N-gram range", f"{cfg['ngram_min']}–{cfg['ngram_max']}"], ["Min / max document frequency", f"{cfg['min_df']} / {cfg['max_df']}"],
        ["Vocabulary size", ml["features"].get("vocabulary_size")], ["Feature matrix", " × ".join(map(str, ml["features"].get("matrix_shape", [])))],
        ["Custom stopwords", len(cfg["custom_stopwords"])], ["Synonym mappings", len(cfg["synonyms"])],
    ])]))
    cod = dash["coding"]
    sections.append(("Coding methodology", [
        ("p", f"Coding method: {cfg['coding_method']}. {ml['coding'].get('selection', '')}. Codebook rules: {len(cfg['codebook'])}. "
              "Codes are machine-generated candidates until a researcher approves them."),
        ("table", ["Metric", "Value"], [
            ["Active codes", cod["unique_codes"]], ["Total assignments", cod["total_assignments"]],
            ["Segments coded", _fmt_pct(cod["coded_segments"])], ["Records coded", _fmt_pct(cod["coded_records"])],
            ["Respondents coded", _fmt_pct(cod["coded_respondents"])], ["Codes awaiting review", cod["awaiting_review"]],
            ["Approved / rejected / merged", f"{cod['approved']} / {cod['rejected']} / {cod['merged']}"],
        ]),
        ("table", ["Code", "Status", "Segments", "Segment %", "Respondents", "Respondent %"],
         [[c["label"], STATUS_LABEL.get(c["status"], c["status"]), c["segments"], c["segment_pct"], c["respondents"], c["respondent_pct"]]
          for c in cod["table"] if c["status"] in ("candidate", "approved")][:60]),
    ]))
    sections.append(("Codebook", [("table", ["Code", "Category", "Definition", "Quotation"],
                                   [[c["label"], c["category"] or (f"{c['suggested_category']} (suggested)" if c["suggested_category"] else None), c["description"],
                                     c["quotations"][0]["text"] if c["quotations"] else None] for c in codebook(store, run_id)][:100])]))
    sections.append(("Categories", [
        ("p", "Categories group similar codes. Those marked suggested were matched by text similarity and need researcher review."),
        ("table", ["Category", "Origin", "Passages", "Codes"],
         [[c["name"], "suggested" if c["suggested"] else "codebook or researcher", _fmt_pct(c["share"]), ", ".join(x["label"] for x in c["codes"])] for c in structure["categories"]]),
    ] + ([("p", "Not yet categorised: " + ", ".join(x["label"] for x in structure["uncategorised"]))] if structure["uncategorised"] else [])))
    cl = ml["clustering"]
    if cl.get("available"):
        cv = cluster_view(store, run_id)
        sections.append(("Clustering methodology", [
            ("p", f"Algorithm: {cl['algorithm']}; {cl['representation']}; k = {cl['n_clusters']} ({cl['k_selection']}). "
                  f"Silhouette (cosine) = {cl['silhouette_cosine']}; Davies–Bouldin = {cl['davies_bouldin']}. "
                  f"Unassigned segments: {cl['unassigned_segments']} ({cl['unassigned_pct']}%)."),
            ("table", ["Cluster", "Label", "Passages", "Respondents", "Cohesion", "Distinctive terms"],
             [[c["cluster"], c["label"], c["passages"], c["respondents"], c["cohesion"], ", ".join(t["term"] for t in (c["terms"] or [])[:6])] for c in cv["clusters"]]),
        ]))
    else:
        sections.append(("Clustering methodology", [("p", f"Clustering unavailable: {cl.get('reason')}")]))
    tv = topic_view(store, run_id)
    tm = tv["metrics"]
    if tm.get("available"):
        sections.append(("Topic-modelling results", [
            ("p", f"Method: {tm['method'].upper()} with {tm['n_topics']} topics. Mean UMass coherence {tm['mean_coherence_umass']}; "
                  f"topic diversity {tm['topic_diversity']}. {tv['note']}"),
            ("table", ["Topic", "Top terms", "Segments (dominant)", "Coherence (UMass)"],
             [[t["topic"], ", ".join(t["terms"][:8]), _fmt_pct(t["share"]), t["coherence"]] for t in tv["topics"]]),
        ]))
    g = graph_view(store, run_id)
    gs = g["stats"]
    lp = ml["graph"].get("link_prediction_baseline", {})
    sections.append(("Graph analysis", [
        ("p", f"Nodes are codes; co-occurrence edges count {g['params']['level']}s containing both codes (minimum {g['params']['min_cooccurrence']}); "
              f"similarity edges join codes whose mean TF-IDF vectors have cosine ≥ {g['params']['similarity_threshold']}. "
              f"Communities: {g['params']['method']} on co-occurrence edges (modularity {g['community_info'].get('modularity')}). {g['note']}"),
        ("table", ["Metric", "Value"], [[k.replace("_", " ").capitalize(), v] for k, v in gs.items()] +
         [["Communities (multi-node)", g["community_info"].get("n_multi_node_communities")]]),
        ("table", ["Highest degree", "Degree", "Highest betweenness", "Betweenness"],
         [[a["label"], a["value"], b["label"], b["value"]] for a, b in zip(g["top"]["degree"], g["top"]["betweenness"])]),
    ] + ([("p", "Link-prediction baseline ROC AUC: " + ", ".join(f"{k} {v}" for k, v in lp["roc_auc"].items()))] if lp.get("available") else [])))
    tmp = temporal_view(store, run_id)
    if tmp.get("available"):
        sections.append(("Temporal analysis", [
            ("p", f"Frequency: {tmp['frequency']}; {len(tmp['periods'])} periods ({tmp['periods'][0]} – {tmp['periods'][-1]}). "
                  f"{tmp['coverage_pct']}% of records have a timestamp. Share = {tmp['share_definition']}. "
                  "Mann–Kendall assumes serially independent observations."),
            ("table", ["Code", "Trend", "p-value", "Sen's slope (pp/period)", "Change points"],
             [[c["label"], c["trend"].get("trend", c["trend"].get("reason")), c["trend"].get("p_value"), c["trend"].get("sens_slope"),
               ", ".join(c["change_points"])] for c in tmp["codes"]]),
        ]))
    else:
        sections.append(("Temporal analysis", [("p", f"Not applicable: {tmp.get('reason')}")]))
    st = dash["statistics"]
    blocks = []
    if st["sentiment"]["available"]:
        blocks.append(("table", ["Sentiment (lexicon)", "Segments"], [[k, _fmt_pct(v)] for k, v in st["sentiment"]["percentages"].items()]))
        blocks.append(("p", st["sentiment"]["definitions"].get("limitations", "")))
    if st["code_cooccurrence"]:
        blocks.append(("table", ["Code A", "Code B", "Co-occurring units"], [[r["code_a"], r["code_b"], r["count"]] for r in st["code_cooccurrence"][:15]]))
    gc = st["group_comparison"]
    if gc.get("available"):
        blocks.append(("table", ["Code", "Test", "p", "p (BH-adjusted)", "Cramér's V"],
                       [[r["label"], r.get("test"), r.get("p_value"), r.get("p_adjusted_bh"), r.get("cramers_v")] for r in gc["results"]]))
        blocks.append(("p", gc["caveat"]))
    sections.append(("Statistical results", blocks or [("p", "No statistical results available.")]))

    theme_blocks = []
    sub_of = {t["id"]: t["subthemes"] for t in structure["themes"]}
    for t in dash["themes"]["table"]:
        ev = theme_evidence(store, t["id"])
        theme_blocks.append(("h3", f"{t['label']} — {STATUS_LABEL.get(t['status'], t['status'])}"))
        theme_blocks.append(("p", f"{t['definition'] or ''}"))
        if sub_of.get(t["id"]):
            theme_blocks.append(("p", "Sub-themes: " + "; ".join(f"{s['label']} ({', '.join(c['label'] for c in s['codes'])})" for s in sub_of[t["id"]]) + "."))
        theme_blocks.append(("p", f"Codes: {', '.join(c['label'] for c in t['codes'])}. Respondent coverage {_fmt_pct(t['prevalence']['respondent'])}; "
                                  f"supporting passages {t['segments']}."))
        for qd in ev["representative"]:
            theme_blocks.append(("quote", qd["text"], f"{qd['source_file']} · respondent {qd['record_key']} · chars {qd['char_start']}–{qd['char_end']}"
                                 + (" · verified against source" if qd["verified"] else " · NOT VERIFIED")))
        if ev["contradictory"]:
            theme_blocks.append(("p", "Contradictory evidence candidates (opposite sentiment polarity):"))
            for qd in ev["contradictory"]:
                theme_blocks.append(("quote", qd["text"], f"{qd['source_file']} · respondent {qd['record_key']}"))
    sections.append(("Candidate and approved themes", theme_blocks or [("p", "No themes were identified.")]))
    links = structure["links"]
    sections.append(("Thematic map", [
        ("p", links["rule"] + " The map itself can be saved as PNG or SVG from Graphs & Patterns."),
        ("table", ["Theme A", "Theme B", "Relation", f"Shared {links['unit']}s", "Expected", "Lift", "Strongest code pair"],
         [[r["a_label"], r["b_label"], {"together": "more than chance", "apart": "less than chance", "chance": "about chance"}[r["relation"]],
           r["shared"], r["expected"], r["lift"], " + ".join(r.get("via") or [])] for r in links["rows"]]),
    ]))

    sup = ml["coding"].get("supervised", {})
    eval_blocks = [("table", ["Metric", "Value"], [
        ["Clustering silhouette (cosine)", cl.get("silhouette_cosine")], ["Clustering Davies–Bouldin", cl.get("davies_bouldin")],
        ["Topic coherence (UMass)", tm.get("mean_coherence_umass")], ["Topic diversity", tm.get("topic_diversity")],
        ["Graph modularity", g["community_info"].get("modularity")],
    ])]
    if sup.get("available") and "accuracy" in sup:
        eval_blocks.append(("table", ["Supervised metric", "Value"], [
            ["Accuracy", sup["accuracy"]], ["Majority baseline accuracy", sup["majority_baseline_accuracy"]],
            ["Macro precision", sup["macro_precision"]], ["Macro recall", sup["macro_recall"]], ["Macro F1", sup["macro_f1"]],
            ["Train / test records", f"{sup['split']['train']} / {sup['split']['test']}"]]))
    eval_blocks.append(("p", ml["accuracy_note"]))
    sections.append(("Model evaluation", eval_blocks))
    sections.append(("Limitations", [
        ("p", "Codes, clusters, topics and candidate themes are produced by classical statistical methods that detect lexical patterns; "
              "they do not understand meaning or context. Candidate labels are template-generated from frequent terms. "
              "Lexicon sentiment ignores sarcasm and domain usage. Centrality reflects graph structure, not importance. "
              "Prevalence figures describe this sample and do not establish causes. All findings require researcher validation."),
    ]))
    th = dash["themes"]["table"]
    sections.append(("Researcher validation", [("table", ["Item", "Value"], [
        ["Code review progress", _fmt_pct(cod["review_progress"])],
        ["Approved themes", sum(1 for t in th if t["status"] == "approved")],
        ["Candidate themes awaiting review", sum(1 for t in th if t["status"] == "candidate")],
        ["Codes in approved themes", _fmt_pct(dash["themes"]["codes_in_approved_themes"])],
    ])]))
    sections.append(("Reproducibility metadata", [
        ("table", ["Item", "Value"], [
            ["Run ID", run_id], ["Run label", run["label"]], ["ThematicML version", run.get("package_version")],
            ["Random seed", cfg["random_state"]], ["Started", run.get("started_at")], ["Finished", run.get("finished_at")],
            ["Processing time (s)", run.get("duration_s")], ["Peak memory (MB)", run.get("peak_memory_mb")],
        ]),
        ("p", "Full configuration: " + json.dumps({k: v for k, v in cfg.items() if k != "codebook"}, default=str)),
    ]))
    return {
        "title": f"ThematicML report — {project['name']}",
        "subtitle": f"{run['label']} · generated {datetime.now():%Y-%m-%d %H:%M} · no generative AI was used",
        "sections": sections,
        "dashboard": dash,
    }


# ---------------------------------------------------------------------- renderers

def render_html(model: dict) -> str:
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(model['title'])}</title>",
           "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1f2937}"
           "table{border-collapse:collapse;width:100%;margin:.75rem 0;font-size:.9rem}td,th{border:1px solid #d1d5db;padding:.3rem .5rem;text-align:left}"
           "th{background:#f3f4f6}blockquote{border-left:3px solid #6366f1;margin:.5rem 0;padding:.25rem .75rem;background:#f9fafb}"
           "blockquote small{display:block;color:#6b7280;margin-top:.25rem}h2{margin-top:2rem;border-bottom:1px solid #e5e7eb}</style></head><body>",
           f"<h1>{html.escape(model['title'])}</h1><p><em>{html.escape(model['subtitle'])}</em></p>"]
    for heading, blocks in model["sections"]:
        out.append(f"<h2>{html.escape(heading)}</h2>")
        for b in blocks:
            if b[0] == "p":
                out.append(f"<p>{html.escape(str(b[1]))}</p>")
            elif b[0] == "h3":
                out.append(f"<h3>{html.escape(b[1])}</h3>")
            elif b[0] == "quote":
                out.append(f"<blockquote>“{html.escape(b[1])}”<small>{html.escape(b[2])}</small></blockquote>")
            elif b[0] == "table":
                out.append("<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in b[1]) + "</tr>")
                for row in b[2]:
                    out.append("<tr>" + "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in row) + "</tr>")
                out.append("</table>")
    out.append("</body></html>")
    return "".join(out)


def render_docx(model: dict, path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_heading(model["title"], 0)
    d.add_paragraph(model["subtitle"]).italic = True
    for heading, blocks in model["sections"]:
        d.add_heading(heading, 1)
        for b in blocks:
            if b[0] == "p":
                d.add_paragraph(str(b[1]))
            elif b[0] == "h3":
                d.add_heading(b[1], 2)
            elif b[0] == "quote":
                p = d.add_paragraph(style="Intense Quote")
                p.add_run(f"“{b[1]}”")
                d.add_paragraph(b[2]).runs[0].italic = True
            elif b[0] == "table":
                t = d.add_table(rows=1, cols=len(b[1]))
                t.style = "Light Grid Accent 1"
                for i, h in enumerate(b[1]):
                    t.rows[0].cells[i].text = str(h)
                for row in b[2]:
                    cells = t.add_row().cells
                    for i, v in enumerate(row):
                        cells[i].text = "" if v is None else str(v)
    d.save(str(path))


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def render_pdf(model: dict, path: Path) -> None:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as exc:
        raise RuntimeError("PDF export requires the optional package fpdf2 (pip install thematicml[reports]).") from exc
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 15)
    font = next((f for f in _FONT_CANDIDATES if Path(f).exists()), None)
    if font:
        pdf.add_font("body", "", font)
        bold = font.replace("arial.ttf", "arialbd.ttf").replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
        pdf.add_font("body", "B", bold if Path(bold).exists() else font)
        pdf.add_font("body", "I", font)
        family = "body"

        def clean(s):
            return str(s)
    else:
        family = "helvetica"

        def clean(s):
            return str(s).encode("latin-1", "replace").decode("latin-1")
    pdf.add_page()
    nl = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
    pdf.set_font(family, "B", 16)
    pdf.multi_cell(0, 8, clean(model["title"]), **nl)
    pdf.set_font(family, "I", 9)
    pdf.multi_cell(0, 5, clean(model["subtitle"]), **nl)
    for heading, blocks in model["sections"]:
        pdf.ln(3)
        pdf.set_font(family, "B", 13)
        pdf.multi_cell(0, 7, clean(heading), **nl)
        for b in blocks:
            if b[0] == "p":
                pdf.set_font(family, "", 9)
                pdf.multi_cell(0, 4.5, clean(b[1]), **nl)
                pdf.ln(1)
            elif b[0] == "h3":
                pdf.set_font(family, "B", 10.5)
                pdf.multi_cell(0, 5.5, clean(b[1]), **nl)
            elif b[0] == "quote":
                pdf.set_font(family, "", 9)
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(pdf.epw - 5, 4.5, clean(f"“{b[1]}”"), **nl)
                pdf.set_font(family, "I", 7.5)
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(pdf.epw - 5, 4, clean(b[2]), **nl)
                pdf.ln(1)
            elif b[0] == "table":
                pdf.set_font(family, "", 8)
                with pdf.table(text_align="LEFT", line_height=4.5) as table:
                    header = table.row()
                    for h in b[1]:
                        header.cell(clean(h))
                    for row in b[2]:
                        r = table.row()
                        for v in row:
                            r.cell(clean("" if v is None else v)[:300])
                pdf.ln(2)
    pdf.output(str(path))


# ---------------------------------------------------------------------- entry points

def export_run(store: Store, run_id: str, fmt: str, out_dir: Path | None = None) -> Path:
    if fmt not in ALL_FORMATS:
        raise ValueError(f"Unknown export format '{fmt}'. Options: {sorted(ALL_FORMATS)}")
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    if run["status"] != "completed":
        raise ValueError("Only completed runs can be exported.")
    out_dir = Path(out_dir) if out_dir else store.export_dir(run["project_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"{run_id}_{stamp}"
    ext = {"xlsx": "xlsx", "docx": "docx", "pdf": "pdf", "html": "html", "graphml": "graphml"}.get(fmt, "json" if fmt.endswith("json") else "csv")
    path = base.with_name(f"{base.name}_{fmt.replace('_csv', '').replace('_json', '')}.{ext}")

    if fmt in ("docx", "pdf", "html"):
        model = report_model(store, run_id)
        if fmt == "html":
            path.write_text(render_html(model), encoding="utf-8")
        elif fmt == "docx":
            render_docx(model, path)
        else:
            render_pdf(model, path)
    else:
        dash = build_dashboard(store, run_id)
        if fmt == "codes_csv":
            codes_frame(dash).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "themes_csv":
            themes_frame(dash, thematic_structure(store, run_id)).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "assignments_csv":
            assignments_frame(store, run_id).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "cleaned_csv":
            cleaned_frame(store, run_id).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "codebook_csv":
            codebook_frame(store, run_id).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "categories_csv":
            categories_frame(thematic_structure(store, run_id)).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "theme_links_csv":
            theme_links_frame(thematic_structure(store, run_id)).to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt == "clusters_csv":
            seg = load_segments(store, run_id)[["segment_id", "record_id", "record_key", "cluster", "topic", "topic_weight", "text"]]
            seg.to_csv(path, index=False, encoding="utf-8-sig")
        elif fmt in ("graph_nodes_csv", "graph_edges_csv", "graphml"):
            g = graph_view(store, run_id)
            if fmt == "graph_nodes_csv":
                pd.DataFrame([{k: v for k, v in n.items() if k != "themes"} for n in g["nodes"]]).to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "graph_edges_csv":
                pd.DataFrame(g["edges"]).to_csv(path, index=False, encoding="utf-8-sig")
            else:
                import networkx as nx

                G = nx.MultiGraph()
                for n in g["nodes"]:
                    G.add_node(n["id"], **{k: ("" if v is None else v) for k, v in n.items() if k not in ("id", "themes")})
                for e in g["edges"]:
                    G.add_edge(e["source"], e["target"], **{k: v for k, v in e.items() if k not in ("source", "target")})
                nx.write_graphml(G, path)
        elif fmt == "metrics_json":
            path.write_text(json.dumps(run["metrics"], indent=2, default=str), encoding="utf-8")
        elif fmt == "temporal_json":
            path.write_text(json.dumps(temporal_view(store, run_id), indent=2, default=str), encoding="utf-8")
        elif fmt == "results_json":
            payload = {
                "generator": f"thematicml {__version__}", "exported_at": now(), "run": {k: run[k] for k in ("id", "label", "config", "dataset_ids", "finished_at")},
                "dashboard": dash, "clusters": cluster_view(store, run_id), "topics": topic_view(store, run_id),
                "graph": graph_view(store, run_id), "temporal": temporal_view(store, run_id),
                "assignments": assignments_frame(store, run_id).to_dict(orient="records"),
            }
            path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        elif fmt == "xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                ov = dash["overview"]
                pd.DataFrame([{"metric": k, "value": json.dumps(v, default=str) if isinstance(v, (dict, list)) else v} for k, v in ov.items()]).to_excel(xw, sheet_name="Overview", index=False)
                pd.DataFrame([{"metric": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in dash["quality"].items() if k != "warnings"]).to_excel(xw, sheet_name="Data quality", index=False)
                structure = thematic_structure(store, run_id)
                codes_frame(dash).to_excel(xw, sheet_name="Codes", index=False)
                codebook_frame(store, run_id).to_excel(xw, sheet_name="Codebook", index=False)
                categories_frame(structure).to_excel(xw, sheet_name="Categories", index=False)
                themes_frame(dash, structure).to_excel(xw, sheet_name="Themes", index=False)
                theme_links_frame(structure).to_excel(xw, sheet_name="Theme links", index=False)
                pd.DataFrame(dash["statistics"]["code_cooccurrence"]).to_excel(xw, sheet_name="Co-occurrence", index=False)
                if dash["statistics"]["sentiment"]["available"]:
                    pd.DataFrame([{"label": k, **v} for k, v in dash["statistics"]["sentiment"]["percentages"].items()]).to_excel(xw, sheet_name="Sentiment", index=False)
                cv = cluster_view(store, run_id)
                pd.DataFrame([{k: (", ".join(t["term"] for t in v[:8]) if k == "terms" else v) for k, v in c.items()
                               if k not in ("representative", "representative_passages", "top_codes")} for c in cv["clusters"]]).to_excel(xw, sheet_name="Clusters", index=False)
                tv = topic_view(store, run_id)
                pd.DataFrame([{"topic": t["topic"], "terms": ", ".join(t["terms"]), "segments": t["share"]["numerator"],
                               "pct": t["share"]["value"], "coherence_umass": t["coherence"]} for t in tv["topics"]]).to_excel(xw, sheet_name="Topics", index=False)
                g = graph_view(store, run_id)
                pd.DataFrame([{k: v for k, v in n.items() if k != "themes"} for n in g["nodes"]]).to_excel(xw, sheet_name="Graph nodes", index=False)
                pd.DataFrame(g["edges"]).to_excel(xw, sheet_name="Graph edges", index=False)
                pd.DataFrame(dash["ml"]["experiments"]).to_excel(xw, sheet_name="Experiments", index=False)

    store.insert("exports", {"project_id": run["project_id"], "run_id": run_id, "format": fmt, "path": str(path),
                             "size_bytes": path.stat().st_size, "created_at": now()})
    store.log(run["project_id"], "report_exported", {"run_id": run_id, "format": fmt, "file": path.name})
    return path


def export_project_zip(store: Store, project_id: str, out_dir: Path | None = None) -> Path:
    """Back up a project: database rows (as JSON) plus original files and run artifacts."""
    project = store.get_project(project_id)
    if project is None:
        raise KeyError(project_id)
    out_dir = Path(out_dir) if out_dir else store.export_dir(project_id)
    path = out_dir / f"{project_id}_backup_{datetime.now():%Y%m%d-%H%M%S}.zip"
    run_ids = [r["id"] for r in store.list_runs(project_id)]
    ds_ids = [d["id"] for d in store.list_datasets(project_id)]

    def rows(sql, ids):
        if not ids:
            return []
        return store.query(sql % ",".join("?" * len(ids)), ids)

    dump = {
        "format": "thematicml-project-backup", "version": 1, "generator": f"thematicml {__version__}", "created_at": now(),
        "project": project, "datasets": store.list_datasets(project_id), "runs": store.list_runs(project_id),
        "records": rows("SELECT * FROM records WHERE dataset_id IN (%s)", ds_ids),
        "segments": rows("SELECT * FROM segments WHERE run_id IN (%s)", run_ids),
        "codes": rows("SELECT * FROM codes WHERE run_id IN (%s)", run_ids),
        "assignments": rows("SELECT * FROM assignments WHERE run_id IN (%s)", run_ids),
        "themes": rows("SELECT * FROM themes WHERE run_id IN (%s)", run_ids),
        "clusters": rows("SELECT * FROM clusters WHERE run_id IN (%s)", run_ids),
        "topics": rows("SELECT * FROM topics WHERE run_id IN (%s)", run_ids),
        "annotations": rows("SELECT * FROM annotations WHERE run_id IN (%s)", run_ids),
        "activity": store.activity(project_id, 100000),
    }
    base = store.project_dir(project_id)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(dump, default=str))
        for sub in ("datasets", "runs"):
            for f in (base / sub).rglob("*"):
                if f.is_file() and f.name != "checkpoint.joblib":
                    z.write(f, f.relative_to(base).as_posix())
    store.insert("exports", {"project_id": project_id, "run_id": None, "format": "project_zip", "path": str(path),
                             "size_bytes": path.stat().st_size, "created_at": now()})
    store.log(project_id, "project_exported", {"file": path.name})
    return path
