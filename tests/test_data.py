import json

import pandas as pd
import pytest

from thematicml.config import ColumnMapping
from thematicml.data import LoadError, load_file, normalise_records, segment_records, segment_text, suggest_mapping


def test_csv_load_and_suggest(sample_csv):
    lf = load_file(sample_csv)
    assert lf.format == "csv" and lf.encoding == "utf-8"
    assert len(lf.frame) == 401
    s = suggest_mapping(lf.frame)
    assert s["text_col"] == "response_text"
    assert s["id_col"] == "respondent_id"
    assert s["time_col"] == "created_at"
    assert s["group_col"] == "region"


def test_cp1252_fallback_warns(tmp_path):
    p = tmp_path / "w.csv"
    p.write_bytes("id,text\n1,caf\xe9 is good\n".encode("cp1252"))
    lf = load_file(p)
    assert lf.encoding == "cp1252"
    assert lf.warnings
    assert lf.frame.loc[0, "text"] == "café is good"


def test_json_jsonl_txt(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"records": [{"id": 1, "text": "hello world"}, {"id": 2, "text": None}]}))
    df = load_file(tmp_path / "a.json").frame
    assert list(df.columns) == ["id", "text"] and df.loc[1, "text"] is None
    (tmp_path / "b.jsonl").write_text('{"t": "one"}\n\n{"t": "two"}\n')
    assert load_file(tmp_path / "b.jsonl").frame["t"].tolist() == ["one", "two"]
    (tmp_path / "c.txt").write_text("First paragraph\nstill first.\n\nSecond one.\n")
    t = load_file(tmp_path / "c.txt")
    assert not t.structured
    assert t.frame["text"].tolist() == ["First paragraph still first.", "Second one."]


def test_docx_and_pdf(tmp_path):
    import docx
    from fpdf import FPDF

    d = docx.Document()
    d.add_paragraph("The menus are confusing.")
    d.add_paragraph("")
    d.add_paragraph("Support was quick.")
    d.save(tmp_path / "i.docx")
    f = load_file(tmp_path / "i.docx").frame
    assert f["text"].tolist() == ["The menus are confusing.", "Support was quick."]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 6, "Interview transcript line one.")
    pdf.output(str(tmp_path / "t.pdf"))
    pf = load_file(tmp_path / "t.pdf").frame
    assert "Interview transcript" in pf["text"].iloc[0]
    assert pf["source_page"].iloc[0] == 1


def test_unsupported_format(tmp_path):
    (tmp_path / "x.xls").write_text("x")
    with pytest.raises(LoadError):
        load_file(tmp_path / "x.xls")


def test_validation_counts():
    df = pd.DataFrame({
        "id": ["a", "b", "b", None, "c"],
        "text": ["Hello there", "Same text", "same  TEXT", "Missing id row", "   "],
        "when": ["2024-01-01", "not a date", "2024-02-01", "", "2024-03-01"],
    })
    recs, rep = normalise_records(df, ColumnMapping(text_col="text", id_col="id", time_col="when"))
    assert rep.total_records == 5
    assert rep.missing_text == 1
    assert rep.duplicate_records == 1  # (b, same text) normalised
    assert rep.invalid_timestamps == 1
    assert rep.missing_ids == 1
    assert rep.valid_records == 4  # duplicates are flagged, not excluded, by default
    assert recs.loc[3, "record_key"] == "row-4"
    assert recs.loc[0, "text"] == "Hello there"  # original text preserved
    d = rep.to_dict()["percentages"]["missing_text"]
    assert d["numerator"] == 1 and d["denominator"] == 5 and d["value"] == 20.0

    _, rep2 = normalise_records(df, ColumnMapping(text_col="text", id_col="id", duplicate_policy="exclude"))
    assert rep2.valid_records == 3 and rep2.excluded_records == 2


def test_segment_offsets_exact():
    text = 'I cannot find the settings. "The menus are confusing!" Ok.\nNew line here please.'
    spans = segment_text(text, "sentence", min_words=3)
    assert all(text[s:e].strip() == text[s:e] for s, e in spans)
    joined = [text[s:e] for s, e in spans]
    assert joined[0] == "I cannot find the settings."
    assert any(j.startswith('"The menus') for j in joined)
    # "Ok." has < 3 words and is merged into the following span
    assert not any(j == "Ok." for j in joined)
    assert segment_text(text, "record") == [(0, len(text))]


def test_segment_records_roundtrip():
    recs = pd.DataFrame({"record_id": [1, 2], "text": ["One two three. Four five six seven.", "Single passage only here"]})
    segs = segment_records(recs, "sentence", 3)
    for r in segs.itertuples():
        src = recs.set_index("record_id").loc[r.record_id, "text"]
        assert src[r.char_start:r.char_end] == r.text
    assert len(segs) == 3


def test_xlsx_roundtrip(tmp_path):
    import pandas as pd

    p = tmp_path / "s.xlsx"
    pd.DataFrame({"id": ["a", "b"], "answer": ["The menu is confusing.", None]}).to_excel(p, index=False)
    lf = load_file(p)
    assert lf.frame["answer"].tolist() == ["The menu is confusing.", None]
    assert suggest_mapping(lf.frame)["text_col"] == "answer"


HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Travelling alone | Example Blog</title>
<meta property="article:published_time" content="2024-03-02T10:00:00+00:00"><script>var x = "<p>not text</p>";</script></head>
<body><nav><ul><li>Home page link with several words</li></ul></nav>
<div class="cs-site-content cs-sidebar-enabled"><div class="entry-content">
<h2>Before you go</h2>
<p>Travelling alone taught me to trust my own judgement in new places.</p>
<p>It’s normal to feel nervous on the first night in a new city.</p>
<div class="sharedaddy share-buttons"><p>Share this post on your favourite network today</p></div>
</div><aside class="widget"><p>Subscribe to my newsletter for weekly travel tips</p></aside></div>
<ol class="commentlist"><li class="comment" id="comment-11"><div class="comment-author">Jane Private Name</div>
<div class="comment-content"><p>This helped me plan my first solo trip to Lisbon last spring.</p><p>Second paragraph of the same comment here.</p></div>
<time datetime="2024-03-05T09:00:00">5 March</time></li>
<li class="comment" id="comment-12"><div class="comment-content"><p>I disagree, hostels felt safer to me than hotels.</p></div></li></ol>
<div id="respond" class="comment-respond"><h3 class="comment-reply-title">Leave a Comment Cancel Reply</h3>
<form><p>Save my name, email, and website in this browser for next time.</p></form></div>
<footer><p>Copyright notice and other footer text here</p></footer></body></html>"""


def test_html_article_and_comments(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(HTML_PAGE, encoding="utf-8")
    lf = load_file(p)
    df = lf.frame
    arts = df[df["source_kind"] == "article"]
    coms = df[df["source_kind"] == "comment"]
    assert arts["text"].tolist() == ["Travelling alone taught me to trust my own judgement in new places.",
                                     "It’s normal to feel nervous on the first night in a new city."]  # exact text, curly apostrophe kept
    assert arts["section"].tolist() == ["Before you go", "Before you go"]
    assert set(arts["respondent_id"]) == {"travelling-alone-example-blog:author"}
    assert coms["respondent_id"].tolist() == ["travelling-alone-example-blog:comment-1", "travelling-alone-example-blog:comment-2"]
    assert "Second paragraph of the same comment here." in coms.iloc[0]["text"]  # one record per comment
    assert coms.iloc[0]["published"] == "2024-03-05T09:00:00"
    joined = " ".join(df["text"])
    for leaked in ("Jane Private Name", "Share this post", "Subscribe", "Leave a Comment", "Save my name", "Copyright", "Home page", "not text"):
        assert leaked not in joined
    assert lf.structured is False and lf.warnings  # under 300 article words -> warning
    m = suggest_mapping(df)
    assert m["text_col"] == "text" and m["id_col"] == "respondent_id" and m["group_col"] == "source_kind"


def test_language_policy_flags_or_excludes_non_english():
    df = pd.DataFrame({"id": ["a", "b", "c"], "text": [
        "I felt safe walking back to the hostel after dinner, even though it was dark.",
        "Observou-se que o medo é um sentimento experienciado pelas mulheres durante as viagens realizadas no Brasil.",
        "Muito obrigada!",  # too short to judge: kept
    ]})
    recs, rep = normalise_records(df, ColumnMapping(text_col="text", id_col="id"))
    assert rep.non_english_records == 1 and rep.valid_records == 3 and any("language other than English" in w for w in rep.warnings)
    recs, rep = normalise_records(df, ColumnMapping(text_col="text", id_col="id", language_policy="exclude_non_english"))
    assert [None if pd.isna(r) else r for r in recs["exclusion_reason"]] == [None, "non_english", None]
    assert rep.valid_records == 2 and rep.to_dict()["percentages"]["non_english"]["numerator"] == 1
