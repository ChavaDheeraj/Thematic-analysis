"""The seven thematic-analysis steps read back from a stored run."""
import pytest

from thematicml import AnalysisConfig, CodebookRule, Project, review
from thematicml.reporting import report_model
from thematicml.results import NotFound, code_table, load_segments
from thematicml.thematic import (
    _subthemes,
    _theme_links,
    cleaned_frame,
    cleaned_view,
    coded_statements,
    codebook,
    notes_view,
    recurring_ideas,
    suggest_categories,
    thematic_structure,
)


@pytest.fixture
def run(tmp_path, sample_csv):
    """A project of its own, for tests that change codes, themes or notes."""
    p = Project("thematic", workspace=tmp_path)
    p.load(sample_csv)
    r = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", n_codes=30, random_state=7))
    assert r.status == "completed", r.run
    return p.store, r.run_id


def test_cleaned_dataset_accounts_for_every_record(analysed):
    p, result = analysed
    c = cleaned_view(p.store, result.run_id, size=10)
    assert c["parsed"] == 401 and c["parsed"] == c["kept"] + c["removed_total"]
    assert {r["label"] for r in c["removed"]} == {"Empty text"} and c["total"] == c["kept"]
    assert len(c["rows"]) == 10 and all(r["passages"] >= 1 for r in c["rows"])
    frame = cleaned_frame(p.store, result.run_id)
    assert len(frame) == c["kept"] and frame["text"].str.strip().ne("").all()


def test_cleaned_dataset_search_narrows_the_rows_but_not_the_counts(analysed):
    p, result = analysed
    full = cleaned_view(p.store, result.run_id)
    hit = cleaned_view(p.store, result.run_id, search="support")
    assert 0 < hit["total"] < full["total"] and hit["kept"] == full["kept"]
    assert all("support" in r["text"].lower() or "support" in r["record_key"].lower() for r in hit["rows"])


def test_recurring_ideas_quote_real_passages_largest_first(analysed):
    p, result = analysed
    out = recurring_ideas(p.store, result.run_id)
    assert out["available"] and out["ideas"]
    shares = [i["share"]["numerator"] for i in out["ideas"]]
    assert shares == sorted(shares, reverse=True)
    quoted = [i["quote"] for i in out["ideas"] if i["quote"]]
    assert quoted and all(q["verified"] for q in quoted)


def test_notes_are_stored_listed_and_reported(run):
    store, rid = run
    with pytest.raises(ValueError):
        review.add_note(store, rid, "   ")
    note = review.add_note(store, rid, "Support comes up before price.")
    assert [n["note"] for n in notes_view(store, rid)["memos"]] == ["Support comes up before price."]
    code = code_table(store, rid, include_inactive=False)[0]
    review.update_code(store, code["id"], notes="Check this one")
    assert [c["id"] for c in notes_view(store, rid)["codes"]] == [code["id"]]
    assert thematic_structure(store, rid)["path"]["notes"] == 2
    text = str(report_model(store, rid)["sections"])
    assert "Support comes up before price." in text and "Check this one" in text
    review.delete_note(store, note["id"])
    assert notes_view(store, rid)["memos"] == []
    with pytest.raises(NotFound):
        review.delete_note(store, note["id"])


def test_delete_note_leaves_quotation_annotations_alone(run):
    store, rid = run
    seg = load_segments(store, rid)["segment_id"].iloc[0]
    a = review.annotate(store, rid, int(seg), note="Verified", verified=True)
    with pytest.raises(NotFound):
        review.delete_note(store, a["id"])
    assert store.scalar("SELECT COUNT(*) FROM annotations WHERE id = ?", (a["id"],)) == 1


def test_initial_coding_lists_statements_with_their_codes(analysed):
    p, result = analysed
    s = coded_statements(p.store, result.run_id, size=5)
    assert 0 < s["coded"] <= s["segments"] and s["total"] == s["coded"]
    assert len(s["rows"]) == 5 and all(r["codes"] for r in s["rows"])
    code = code_table(p.store, result.run_id, include_inactive=False)[0]
    one = coded_statements(p.store, result.run_id, code_id=code["id"], size=5)
    assert one["total"] == code["segments"] and all(code["id"] in [c["id"] for c in r["codes"]] for r in one["rows"])
    second = coded_statements(p.store, result.run_id, page=2, size=5)
    assert not {r["id"] for r in s["rows"]} & {r["id"] for r in second["rows"]}


def test_rejected_codes_leave_initial_coding_and_the_codebook(run):
    store, rid = run
    code = code_table(store, rid, include_inactive=False)[0]
    review.update_code(store, code["id"], status="rejected")
    assert code["id"] not in [c["id"] for c in codebook(store, rid)]
    rows = coded_statements(store, rid, size=50)["rows"]
    assert all(code["id"] not in [c["id"] for c in r["codes"]] for r in rows)


def test_codebook_quotes_are_verbatim_and_from_the_code(analysed):
    p, result = analysed
    book = codebook(p.store, result.run_id)
    assert book and all(c["quotations"] for c in book if c["segments"])
    for c in book:
        assert len(c["quotations"]) <= 2
        assert len({q["record_id"] for q in c["quotations"]}) == len(c["quotations"])
        for q in c["quotations"]:
            assert q["verified"]
            assert p.store.scalar("SELECT COUNT(*) FROM assignments WHERE code_id = ? AND segment_id = ?", (c["id"], q["segment_id"])) == 1


def test_codes_without_a_codebook_get_suggested_groups_that_are_not_stored(analysed):
    p, result = analysed
    run = p.store.get_run(result.run_id)
    codes = code_table(p.store, result.run_id, include_inactive=False)
    assert not any(c["category"] for c in codes)
    suggested = suggest_categories(p.store, run, codes)
    assert suggested
    s = thematic_structure(p.store, result.run_id)
    assert s["categories"] and all(c["suggested"] and c["n_suggested"] == len(c["codes"]) for c in s["categories"])
    assert sum(len(c["codes"]) for c in s["categories"]) + len(s["uncategorised"]) == len(codes)
    assert not any(c["category"] for c in code_table(p.store, result.run_id))


def test_suggestions_join_a_category_only_when_close_enough(run):
    store, rid = run
    codes = code_table(store, rid, include_inactive=False)
    review.update_code(store, codes[0]["id"], category="Service")
    codes = code_table(store, rid, include_inactive=False)
    suggested = suggest_categories(store, store.get_run(rid), codes)
    assert codes[0]["id"] not in suggested  # an assigned category is never second-guessed
    cats = thematic_structure(store, rid)["categories"]
    service = next(c for c in cats if c["name"] == "Service")
    assert not service["suggested"]
    assert next(x for x in service["codes"] if x["id"] == codes[0]["id"])["suggested"] is False


def test_accepting_a_suggestion_stores_it(run):
    store, rid = run
    cat = next(c for c in thematic_structure(store, rid)["categories"] if c["n_suggested"])
    code = cat["codes"][0]
    assert code["suggested"]
    review.update_code(store, code["id"], category=cat["name"])
    assert store.scalar("SELECT category FROM codes WHERE id = ?", (code["id"],)) == cat["name"]
    after = next(c for c in thematic_structure(store, rid)["categories"] if c["name"] == cat["name"])
    assert next(x for x in after["codes"] if x["id"] == code["id"])["suggested"] is False


def test_subthemes_need_two_codes_and_more_than_one_group():
    codes = {i: {"label": f"c{i}", "segments": 10 - i} for i in range(1, 7)}
    theme = {"code_ids": [1, 2, 3, 4, 5], "status": "candidate"}
    subs, other = _subthemes(theme, [], {1: "A", 2: "A", 3: "B", 4: "B", 5: "C"}, codes, {})
    assert [s["label"] for s in subs] == ["A", "B"] and [c["id"] for c in other] == [5]
    subs, other = _subthemes(theme, [], {i: "A" for i in range(1, 6)}, codes, {})
    assert subs == [] and len(other) == 5  # one category covering the whole theme adds nothing


def test_a_researcher_subtheme_takes_its_codes_out_of_the_category_groups():
    codes = {i: {"label": f"c{i}", "segments": 1} for i in range(1, 6)}
    theme = {"code_ids": [1, 2, 3, 4], "status": "candidate"}
    child = {"id": 9, "label": "Mine", "status": "candidate", "code_ids": [1, 2]}
    subs, other = _subthemes(theme, [child], {1: "A", 2: "A", 3: "A", 4: "A"}, codes, {})
    assert [(s["label"], s["source"]) for s in subs] == [("Mine", "researcher"), ("A", "category")]
    assert [c["id"] for c in subs[1]["codes"]] == [3, 4] and other == []


def test_a_rejected_subtheme_gives_its_codes_back_but_stays_reviewable():
    codes = {i: {"label": f"c{i}", "segments": 1} for i in range(1, 5)}
    theme = {"code_ids": [1, 2, 3, 4], "status": "candidate"}
    child = {"id": 9, "label": "Mine", "status": "rejected", "code_ids": [1, 2]}
    subs, other = _subthemes(theme, [child], {1: "A", 2: "A", 3: "B", 4: "B"}, codes, {})
    assert [(s["label"], s["status"]) for s in subs][0] == ("Mine", "rejected")
    assert [s["label"] for s in subs[1:]] == ["A", "B"] and other == []


def test_rejected_subthemes_are_not_counted(run):
    store, rid = run
    parent = thematic_structure(store, rid)["themes"][0]
    child = review.create_theme(store, rid, "Nested", [])
    review.update_theme(store, child["id"], parent_id=parent["id"])
    counted = thematic_structure(store, rid)["path"]["subthemes"]
    review.update_theme(store, child["id"], status="rejected")
    after = thematic_structure(store, rid)
    assert after["path"]["subthemes"] == counted - 1
    assert child["id"] in [x["id"] for t in after["themes"] for x in t["subthemes"]]


def test_a_researcher_subtheme_may_share_a_category_name():
    codes = {i: {"label": f"c{i}", "segments": 1} for i in range(1, 6)}
    theme = {"code_ids": [1, 2, 3, 4], "status": "candidate"}
    child = {"id": 9, "label": "A", "status": "candidate", "code_ids": [1, 2]}
    subs, _ = _subthemes(theme, [child], {1: "A", 2: "A", 3: "A", 4: "A"}, codes, {})
    assert [s["label"] for s in subs] == ["A", "A"] and [s["id"] for s in subs] == [9, None]


def test_sub_theme_position_is_validated(run):
    store, rid = run
    a = review.create_theme(store, rid, "Parent", [])
    b = review.create_theme(store, rid, "Child", [])
    c = review.create_theme(store, rid, "Third", [])
    for bad in (a["id"], 10**6):
        with pytest.raises(ValueError):
            review.update_theme(store, a["id"], parent_id=bad)
    assert review.update_theme(store, b["id"], parent_id=a["id"])["parent_id"] == a["id"]
    with pytest.raises(ValueError):
        review.update_theme(store, c["id"], parent_id=b["id"])  # no sub-sub-themes
    with pytest.raises(ValueError):
        review.update_theme(store, a["id"], parent_id=c["id"])  # a parent cannot become a child
    s = thematic_structure(store, rid)
    assert b["id"] not in [t["id"] for t in s["themes"]]
    assert b["id"] in [x["id"] for t in s["themes"] if t["id"] == a["id"] for x in t["subthemes"]]
    assert review.update_theme(store, b["id"], parent_id=0)["parent_id"] is None
    assert b["id"] in [t["id"] for t in thematic_structure(store, rid)["themes"]]


def test_other_run_cannot_be_a_parent(tmp_path, sample_csv):
    p = Project("two", workspace=tmp_path)
    p.load(sample_csv)
    cfg = AnalysisConfig(text_col="response_text", id_col="respondent_id", n_codes=15, random_state=7)
    r1, r2 = p.analyze(cfg), p.analyze(cfg)
    a = review.create_theme(p.store, r1.run_id, "A", [])
    b = review.create_theme(p.store, r2.run_id, "B", [])
    with pytest.raises(ValueError):
        review.update_theme(p.store, a["id"], parent_id=b["id"])


def _links(store, rid, sets):
    """Link two made-up themes; positions are real passages so quotations resolve."""
    segs = load_segments(store, rid)
    codes = {i: {"label": f"code {i}"} for i in sets}
    themes = [{"id": i, "label": f"T{i}", "code_ids": [i]} for i in sets]
    seg_record = dict(zip(segs["pos"].tolist(), segs["record_id"].tolist()))
    run = {"id": rid, "config": {"cooccurrence_level": "segment"}}
    return _theme_links(store, run, themes, codes, sets, seg_record), len(segs)


def test_map_links_only_what_differs_from_chance(analysed):
    p, result = analysed
    n = len(load_segments(p.store, result.run_id))
    half = n // 2
    together, _ = _links(p.store, result.run_id, {1: set(range(half)), 2: set(range(half // 2, half + half // 2))})
    assert together["rows"][0]["relation"] == "chance"  # half of each overlaps: exactly what chance gives
    linked, _ = _links(p.store, result.run_id, {1: set(range(100)), 2: set(range(100))})
    r = linked["rows"][0]
    assert r["relation"] == "together" and r["lift"] > 1 and r["shared"] == 100 and r["via"] == ["code 1", "code 2"]
    assert r["quote"]["verified"]
    apart, _ = _links(p.store, result.run_id, {1: set(range(half)), 2: set(range(half, n))})
    assert apart["rows"][0]["relation"] == "apart" and apart["rows"][0]["shared"] == 0


def test_rejected_themes_leave_the_map_but_stay_reviewable(run):
    store, rid = run
    s = thematic_structure(store, rid)
    t = s["themes"][0]
    review.update_theme(store, t["id"], status="rejected")
    after = thematic_structure(store, rid)
    assert t["id"] in [x["id"] for x in after["themes"]]
    assert after["path"]["themes"] == s["path"]["themes"] - 1
    assert all(t["id"] not in (r["a"], r["b"]) for r in after["links"]["rows"])


def test_approving_a_theme_makes_it_final(run):
    store, rid = run
    t = thematic_structure(store, rid)["themes"][0]
    assert thematic_structure(store, rid)["path"]["final"] == 0
    review.update_theme(store, t["id"], status="approved")
    assert thematic_structure(store, rid)["path"]["final"] == 1


def test_only_real_edits_mark_a_theme_as_researcher_written(run):
    store, rid = run
    t = thematic_structure(store, rid)["themes"][0]
    # the review form always sends the label and definition, changed or not
    saved = review.update_theme(store, t["id"], label=t["label"], definition=t["definition"], status="approved")
    assert saved["origin"] == "machine"
    assert review.update_theme(store, t["id"], definition="Written by the researcher.")["origin"] == "researcher"


def test_a_sub_theme_is_not_counted_as_a_main_theme(run):
    store, rid = run
    from thematicml.reporting import build_dashboard

    before = build_dashboard(store, rid)["overview"]["total_themes"]
    a = review.create_theme(store, rid, "Main", [])
    b = review.create_theme(store, rid, "Nested", [])
    review.update_theme(store, b["id"], parent_id=a["id"])
    assert build_dashboard(store, rid)["overview"]["total_themes"] == before + 1


def test_codebook_categories_become_subthemes(tmp_path, sample_csv):
    p = Project("rules", workspace=tmp_path)
    p.load(sample_csv)
    rules = [CodebookRule(label=k.title(), keywords=[k], category="Help" if k in ("support", "service") else "Price")
             for k in ("support", "service", "price", "cost")]
    r = p.analyze(AnalysisConfig(text_col="response_text", id_col="respondent_id", coding_method="both", codebook=rules, n_codes=20, random_state=7))
    assert r.status == "completed", r.run
    s = thematic_structure(p.store, r.run_id)
    assert {"Help", "Price"} <= {c["name"] for c in s["categories"] if not c["suggested"]}
    assert s["path"]["categories"] == len(s["categories"])
