"""Integration tests: the complete workflow through the Python API."""
import json
import zipfile

import pandas as pd
import pytest

from thematicml import AnalysisConfig, CodebookRule, Project
from thematicml import pipeline as pl
from thematicml import review
from thematicml.reporting import ALL_FORMATS, build_dashboard, export_project_zip
from thematicml.results import code_evidence, code_table, graph_view, record_detail, theme_evidence, theme_table
from thematicml.storage import Store


def test_full_run_metrics(analysed):
    _, result = analysed
    m = result.metrics
    assert m["dataset"]["analysed_records"] == 396  # 401 rows - 5 empty; the duplicate is flagged but kept
    assert m["features"]["vocabulary_size"] > 50
    assert m["coding"]["n_codes"] == 30
    assert m["clustering"]["available"] and m["clustering"]["silhouette_cosine"] is not None
    assert m["topics"]["available"] and m["topics"]["n_topics"] == 8
    assert m["graph"]["nodes"] > 0 and m["graph"]["communities"]["n_communities"] > 0
    assert m["themes"]["candidate_themes"] >= 1
    assert m["temporal"]["available"] and m["temporal"]["frequency"] == "M"
    assert m["sentiment"]["available"]
    assert sum(m["sentiment"]["counts"].values()) == m["dataset"]["segments"]
    assert m["statistics"]["group_comparison"]["available"]
    assert "accuracy" not in json.dumps(m["clustering"])  # no fabricated accuracy for unsupervised steps


def test_quotations_are_verbatim(analysed):
    p, result = analysed
    store = p.store
    codes = code_table(store, result.run_id)
    ev = code_evidence(store, codes[0]["id"], limit=50)
    assert ev["items"] and all(item["verified"] for item in ev["items"])
    for item in ev["items"]:
        if item["matched_text"]:
            rec = store.one("SELECT text FROM records WHERE id = ?", (item["record_id"],))["text"]
            assert rec[item["match_start"]:item["match_end"]] == item["matched_text"]
    t = theme_table(store, result.run_id)[0]
    tev = theme_evidence(store, t["id"])
    assert tev["representative"] and all(q["verified"] for q in tev["representative"])


def test_dashboard_percentages_have_denominators(analysed):
    p, result = analysed
    d = build_dashboard(p.store, result.run_id)
    assert d["overview"]["total_records"] == 401
    v = d["quality"]["valid"]
    assert v["numerator"] == 396
    assert v["denominator"] == 401
    assert d["quality"]["missing_text"]["numerator"] == 5
    assert d["quality"]["duplicates"]["numerator"] == 1
    assert d["coding"]["coded_segments"]["denominator"] == d["overview"]["segments"]
    assert d["statistics"]["temporal"]["available"]
    assert d["ml"]["experiments"][0]["current"]


def test_persistence_across_restart(analysed):
    p, result = analysed
    reopened = Project("shared", workspace=p.store.root)
    assert reopened.id == p.id
    again = reopened.result(result.run_id)
    assert again.status == "completed"
    assert len(again.codes) == len(result.codes)


def test_review_workflow(tmp_path, sample_csv):
    p = Project("review", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", n_codes=15, sentiment=False, temporal=False,
                                   coding_method="both", codebook=[CodebookRule(label="Navigation", keywords=["navigat*", "menu*"])]))
    s, rid = p.store, res.run_id
    codes = [c for c in code_table(s, rid) if c["segments"] > 0]
    assert any(c["method"] == "rule" for c in codes)
    a, b = codes[0], codes[1]
    before = a["segments"] + b["segments"]
    merged = review.merge_codes(s, [a["id"], b["id"]], label="Merged code")
    assert merged["label"] == "Merged code"
    after = {c["id"]: c for c in code_table(s, rid)}
    assert after[b["id"]]["status"] == "merged" and after[b["id"]]["merged_into"] == a["id"]
    assert after[a["id"]]["segments"] <= before  # overlapping segments are not double counted

    ev = code_evidence(s, a["id"], limit=3)
    new = review.split_code(s, a["id"], [i["assignment_id"] for i in ev["items"]], "Split part")
    assert code_table(s, rid) and {c["id"]: c for c in code_table(s, rid)}[new["id"]]["assignments"] == 3

    review.update_code(s, codes[2]["id"], status="rejected")
    assert codes[2]["id"] not in {n["id"] for n in graph_view(s, rid)["nodes"]}

    themes = theme_table(s, rid)
    if themes:
        review.update_theme(s, themes[0]["id"], status="approved", label="Researcher label", definition="Defined by me")
        t = {t["id"]: t for t in theme_table(s, rid)}[themes[0]["id"]]
        assert t["status"] == "approved" and t["origin"] == "researcher"
    nt = review.create_theme(s, rid, "Manual theme", [a["id"]], "Manual definition")
    assert {t["id"] for t in theme_table(s, rid)} >= {nt["id"]}

    rec = s.one("SELECT record_id FROM segments WHERE run_id = ? LIMIT 1", (rid,))["record_id"]
    detail = record_detail(s, rec, rid)
    assert detail["segments"] and all(seg["verified"] for seg in detail["segments"])

    clusters = s.query("SELECT cluster, size FROM clusters WHERE run_id = ? ORDER BY size DESC", (rid,))
    big = clusters[0]["cluster"]
    parts = review.split_cluster(s, rid, big, 2)
    assert len(parts) == 2 and sum(x["size"] for x in parts) == clusters[0]["size"]
    review.merge_clusters(s, rid, [x["cluster"] for x in parts], "Recombined")
    acts = {x["kind"] for x in s.activity(p.id, 100)}
    assert {"codes_merged", "code_split", "cluster_split", "clusters_merged"} <= acts


def test_exports_all_formats(analysed, tmp_path):
    p, result = analysed
    for fmt in sorted(ALL_FORMATS):
        path = result.export(fmt, tmp_path)
        assert path.exists() and path.stat().st_size > 0, fmt
    z = export_project_zip(p.store, p.id, tmp_path)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert "project.json" in names and any(n.endswith("original.csv") for n in names)
        dump = json.loads(zf.read("project.json"))
        assert dump["records"] and dump["codes"]


def test_retry_resumes_from_failed_stage(tmp_path, sample_csv, monkeypatch):
    p = Project("retry", workspace=tmp_path)
    did = p.load(sample_csv)
    p.map_columns({"text_col": "response_text", "id_col": "respondent_id"})
    run = pl.create_run(p.store, p.id, [did], AnalysisConfig(temporal=False, sentiment=False))
    calls = []
    original = pl.STAGE_FUNCS["features"]

    def counting(*a):
        calls.append(1)
        return original(*a)

    def boom(*a):
        raise RuntimeError("simulated failure")

    monkeypatch.setitem(pl.STAGE_FUNCS, "features", counting)
    monkeypatch.setitem(pl.STAGE_FUNCS, "graph", boom)
    r = pl.execute_run(p.store, run["id"])
    assert r["status"] == "failed" and r["failed_stage"] == "graph" and "simulated" in r["error"]
    assert "topics" in r["completed_stages"]

    monkeypatch.setitem(pl.STAGE_FUNCS, "graph", pl._stage_graph)
    r = pl.execute_run(p.store, run["id"])
    assert r["status"] == "completed"
    assert len(calls) == 1  # features were not recomputed


def test_interrupted_runs_recovered(tmp_path, sample_csv):
    p = Project("interrupt", workspace=tmp_path)
    did = p.load(sample_csv)
    p.map_columns({"text_col": "response_text"})
    run = pl.create_run(p.store, p.id, [did], AnalysisConfig())
    p.store.update("runs", {"id": run["id"]}, {"status": "running", "stage": "coding"})
    assert Store(tmp_path).recover_interrupted_runs() == 1
    r = p.store.get_run(run["id"])
    assert r["status"] == "interrupted" and r["failed_stage"] == "coding"


def test_no_timestamps_disables_temporal(tmp_path, sample_csv):
    p = Project("notime", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", sentiment=False))
    assert res.status == "completed"
    assert res.metrics["temporal"]["available"] is False
    assert "timestamp" in res.metrics["temporal"]["reason"].lower()


def test_reproducible(tmp_path, sample_csv):
    cfg = AnalysisConfig(text_col="response_text", id_col="respondent_id", sentiment=False, temporal=False, random_state=3)
    out = []
    for name in ("r1", "r2"):
        p = Project(name, workspace=tmp_path / name)
        p.load(sample_csv)
        res = p.analyze(cfg)
        out.append((res.codes[["label", "segments"]].values.tolist(), res.metrics["clustering"]["sizes"],
                    res.metrics["graph"]["communities"]["sizes"]))
    assert out[0] == out[1]


def test_multi_dataset_comparison(tmp_path, sample_csv):
    from thematicml.analytics import compare_runs
    from thematicml.results import run_snapshot

    p = Project("multi", workspace=tmp_path)
    d1 = p.load(sample_csv)
    d2 = p.load(sample_csv)
    cfg = AnalysisConfig(text_col="response_text", id_col="respondent_id", sentiment=False, temporal=False)
    r1 = p.analyze(cfg, [d1])
    r2 = p.analyze(cfg.model_copy(update={"n_codes": 20}), [d2])
    # the same file twice stands in for two waves here, so keep the copy's passages
    both = p.analyze(cfg.model_copy(update={"dedupe_across_datasets": False}), [d1, d2])
    assert both.metrics["dataset"]["analysed_records"] == 2 * r1.metrics["dataset"]["analysed_records"]
    once = p.analyze(cfg, [d1, d2])  # by default a file uploaded twice is analysed once (short answers excepted)
    assert once.metrics["dataset"]["cross_dataset_duplicates"] > 0
    assert once.metrics["dataset"]["analysed_records"] < both.metrics["dataset"]["analysed_records"]
    cmp = compare_runs(run_snapshot(p.store, r1.run_id), run_snapshot(p.store, r2.run_id))
    assert cmp["shared_codes"] and not cmp["only_in_b"]
    assert {"parameter": "n_codes", "a": 40, "b": 20} in cmp["model_differences"]
    assert all(f["diff_pct"] == 0 for f in cmp["frequency_changes"])  # identical data
    assert "not necessarily disappeared" in cmp["caveat"]


def test_dataset_without_valid_records_rejected(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("id,text\n1,\n2, \n")
    p = Project("bad", workspace=tmp_path)
    did = p.load(f)
    q = p.map_columns({"text_col": "text"})
    assert q["valid_records"] == 0
    with pytest.raises(ValueError):
        pl.create_run(p.store, p.id, [did], AnalysisConfig())


def test_peak_memory_is_measured():
    from thematicml.pipeline import peak_memory_mb

    mb = peak_memory_mb()
    assert mb is None or mb > 1  # a real reading or explicitly unavailable, never a fake zero
    import sys

    if sys.platform == "win32":
        assert mb is not None and mb > 1


def test_workspace_can_be_moved_and_deletion_removes_files(tmp_path, sample_csv):
    """Backup/restore = copy the workspace folder; it must work from a new location."""
    import shutil

    from thematicml.storage import Store
    from thematicml.storage.datasets import parsed_frame

    old = tmp_path / "ws_old"
    p = Project("move", workspace=old)
    did = p.load(sample_csv)
    ds = p.store.get_dataset(did)
    assert not ds["original_path"].startswith(str(old))  # stored relative to the workspace
    res = p.analyze(AnalysisConfig(text_col="response_text", sentiment=False, temporal=False))
    # simulate a database written by an earlier version that stored absolute paths
    p.store.update("datasets", {"id": did}, {"original_path": str(p.store.resolve(ds["original_path"]).resolve())})
    new = tmp_path / "ws_restored"
    shutil.copytree(old, new)
    shutil.rmtree(old)
    s = Store(new)
    assert s.resolve(s.get_dataset(did)["original_path"]).exists()
    assert len(parsed_frame(s, did)) == 401
    assert code_table(s, res.run_id)
    # deleting a dataset removes its runs and files on disk
    ddir = new / "projects" / p.id / "datasets" / did
    rdir = new / "projects" / p.id / "runs" / res.run_id
    assert ddir.exists() and rdir.exists()
    s.delete_dataset(did)
    assert not ddir.exists() and not rdir.exists()
    assert s.get_run(res.run_id) is None
    assert s.scalar("SELECT COUNT(*) FROM records WHERE dataset_id = ?", (did,)) == 0


def test_review_rejects_ids_from_other_runs(tmp_path, sample_csv):
    p = Project("iso", workspace=tmp_path)
    p.load(sample_csv)
    cfg = AnalysisConfig(text_col="response_text", sentiment=False, temporal=False, n_codes=10)
    a, b = p.analyze(cfg), p.analyze(cfg)
    s = p.store
    code_b = s.one("SELECT id FROM codes WHERE run_id = ?", (b.run_id,))["id"]
    seg_b = s.one("SELECT id FROM segments WHERE run_id = ?", (b.run_id,))["id"]
    with pytest.raises(ValueError, match="do not belong"):
        review.create_theme(s, a.run_id, "x", [code_b])
    with pytest.raises(ValueError, match="do not belong"):
        review.annotate(s, a.run_id, seg_b, verified=True)
    with pytest.raises(ValueError, match="do not belong"):
        review.create_code(s, a.run_id, "x", [seg_b])
    t = review.create_theme(s, a.run_id, "ok", [])
    with pytest.raises(ValueError, match="do not belong"):
        review.update_theme(s, t["id"], code_ids=[code_b])


def test_supervised_evaluation_is_leak_free(tmp_path, sample_csv):
    p = Project("sup", workspace=tmp_path)
    p.load(sample_csv)
    res = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", label_col="primary_topic", supervised=True,
                                   sentiment=False, temporal=False))
    sup = res.metrics["coding"]["supervised"]
    assert sup["available"] and sup["split"]["vectorizer"] == "re-fitted on training records only"
    cv = sup["cross_validation"]
    assert cv["folds"] >= 3 and len(cv["fold_accuracy"]) == cv["folds"]
    assert sup["accuracy"] > sup["majority_baseline_accuracy"]


def test_passages_repeated_across_uploads_are_analysed_once(tmp_path):
    page = "\n\n".join(f"Paragraph {i}: walking alone at night in a new city made me plan every route and share my location." for i in range(12))
    for name in ("a.txt", "copy.txt"):
        (tmp_path / name).write_text(page, encoding="utf-8")
    (tmp_path / "other.txt").write_text("Yes.\n\nA different story about meeting other women in a hostel kitchen and cooking together.", encoding="utf-8")
    p = Project("dedupe", workspace=tmp_path / "ws")
    ids = [p.load(tmp_path / n) for n in ("a.txt", "copy.txt", "other.txt")]
    for did in ids:
        p.validate(dataset_id=did)
    res = p.analyze(AnalysisConfig(temporal=False, sentiment=False, n_codes=5, min_code_support=1), dataset_ids=ids)
    m = res.metrics
    assert m["dataset"]["cross_dataset_duplicates"] == 12 and m["dataset"]["analysed_records"] == 14
    assert any("repeated from another upload" in w for w in res.run["warnings"])
    text = "The same long answer of more than eight words appears in both waves."
    rep = pl.repeated_passages(pd.DataFrame({"text": ["Yes.", "Yes.", text, text], "record_key": ["r1", "r1", "r1", "r2"],
                                             "dataset_id": ["d1", "d2", "d1", "d2"], "word_count": [1, 1, 13, 13]}))
    assert not rep.any()  # short answers, and the same text from a different respondent, are not repeats
