import textwrap
from collections import Counter

import pytest

from thematicml.data.loaders import load_file
from thematicml.data.pdf import _repair, _table_residue, paragraphs_from_pages

WIDTH = 70  # characters per full line, as in a single-column paper


def wrap(text: str) -> list[str]:
    """Lines of one paragraph as a PDF text layer gives them: full lines, then a shorter last line."""
    return textwrap.wrap(text, WIDTH)


def page(n: int, *blocks: list[str]) -> list[str]:
    return ["Journal of Qualitative Travel Studies", str(n), *[ln for b in blocks for ln in b], f"Page {n} of 4"]


ONE = ("Travelling alone taught me to trust my own judgement and to plan ahead with care before every "
       "single trip, and it changed how I think about risk, freedom and my own abilities.")
TWO = ("Most participants described careful research before departure and the motivations behind it; the "
       "self-actualization they hoped for shaped the whole journey that")
TWO_END = "followed, from first booking to last night."
THREE = ("The interviews also covered how women negotiated harassment, judgement from relatives and the quiet "
         "pleasure of eating alone in a busy restaurant.")


def test_paragraphs_rejoin_wrapped_lines_and_drop_running_headers():
    two = wrap(TWO)
    pages = [
        page(1, wrap(ONE), ["Safety and planning"], two),
        page(2, wrap(TWO_END), wrap(THREE)),
        page(3, wrap("Women reported more freedom and confidence after their first solo trip than before it."),
             ["References"], ["Smith, J. (2020). A study of travel. Journal of Travel, 1(2), 3-4."]),
        page(4, ["Appendix A: Interview guide with questions that must not be imported."]),
    ]
    paras, stats = paragraphs_from_pages(pages)
    texts = [p["text"] for p in paras]
    assert texts[0] == ONE
    # a paragraph that does not end a sentence continues on the next page, after the running header
    assert texts[1] == f"{TWO} {TWO_END}" and paras[1]["page"] == 1 and paras[1]["section"] == "Safety and planning"
    assert texts[2] == THREE and paras[2]["page"] == 2
    assert not any("Journal of Qualitative" in t or "Page " in t for t in texts)
    assert stats["references_page"] == 3 and not any("Smith" in t or "Appendix" in t for t in texts)


def test_hyphenated_line_breaks():
    lines = ["Participants listed their motiva-", "tions in detail and described the self-",
             "actualization they hoped for, while self-actualization and motivations were discussed at length",
             "by every one of the interviewed women in the study."]
    paras, _ = paragraphs_from_pages([lines])
    text = paras[0]["text"]
    # "motivations" occurs unhyphenated elsewhere -> rejoined; "self-actualization" occurs hyphenated -> kept
    assert "their motivations in detail" in text and "the self-actualization they hoped" in text


def test_tables_captions_contents_and_back_matter_are_dropped():
    body = wrap("Participants described feeling watched in crowded markets at night while travelling alone, "
                "especially in places where few women were out after dark.")
    pages = [[*body, "Introduction .......... 3", "Findings .......... 9",
              "Table 2: Destinations by number of reviews",
              "1. Monkey Forest 35,237", "2. Rice Terrace 16,815", "Connect with locals Y Y N",
              "Funding", *wrap("The author received no financial support for the research, authorship or publication of this article."),
              "Findings", *wrap("Women also described joy and freedom when travelling at their own pace, without "
                                "anyone else's plans to follow or anyone to look after.")]]
    paras, stats = paragraphs_from_pages(pages)
    texts = [p["text"] for p in paras]
    assert len(texts) == 2 and texts[0].startswith("Participants described") and texts[1].startswith("Women also described")
    assert paras[1]["section"] == "Findings"
    assert stats["table_lines"] == 3 and stats["back_matter_lines"] >= 1 and stats["captions"] == 1 and stats["noise_lines"] == 2


def test_repair_spacing_artefacts_from_document_vocabulary():
    counts = Counter({"participants": 3, "tourists": 2, "part": 4})
    assert _repair("All part icipants agreed , in fact .", counts, set()) == "All participants agreed, in fact."
    assert _repair("in -depth talks with Khoo -Lattimore", counts, set()) == "in-depth talks with Khoo-Lattimore"
    assert _repair("staff who meet tour - ists", counts, set()) == "staff who meet tourists"
    # a real spaced dash, a negative number and ordinary word pairs stay
    assert _repair("safety - that was the point, from -8 to 4", counts, set()) == "safety - that was the point, from -8 to 4"
    assert _repair("a round table with part time staff", counts, set()) == "a round table with part time staff"


def test_pdf_file_one_respondent_per_document(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.set_font("helvetica", size=11)
    stories = ["she planned every route in advance and shared it with her sister before leaving home.",
               "a guide gave her the front seat and explained how to photograph the canyon walls.",
               "she ate dinner at the bar with a book, and the bartender kept an eye on her."]
    for n, story in enumerate(stories, start=1):
        pdf.add_page()
        pdf.cell(0, 6, "Women Travel Review", new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, f"On day {n} {story}")
        pdf.cell(0, 6, str(n), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(tmp_path / "review.pdf"))
    lf = load_file(tmp_path / "review.pdf")
    df = lf.frame
    assert df["text"].tolist() == [f"On day {n} {s}" for n, s in enumerate(stories, start=1)]
    assert set(df["respondent_id"]) == {"review:author"} and set(df["source_kind"]) == {"document"}
    assert df["source_page"].tolist() == [1, 2, 3]
    assert not lf.structured and not any("Women Travel Review" in t for t in df["text"])


def test_uploaded_pdfs_are_named_after_the_upload_not_the_stored_copy(tmp_path):
    fpdf = pytest.importorskip("fpdf")
    from thematicml.storage.database import Store
    from thematicml.storage.datasets import import_file, parsed_frame

    store = Store(tmp_path / "ws")
    pid = store.create_project("p")["id"]
    keys = []
    for name in ("Liu_Risks.pdf", "Final_Thesis.pdf"):
        pdf = fpdf.FPDF()
        pdf.set_font("helvetica", size=11)
        pdf.add_page()
        pdf.multi_cell(0, 6, f"This paper about {name} describes how women plan routes and share them with family.")
        pdf.output(str(tmp_path / name))
        ds = import_file(store, pid, tmp_path / name, name)
        keys.append(set(parsed_frame(store, ds["id"])["respondent_id"]))
    assert keys == [{"liu-risks:author"}, {"final-thesis:author"}]  # the stored copies are both "original.pdf"


def test_word_tables_flattened_into_a_paragraph_are_residue():
    cells = ("Age Too young to handle the potential risks Race Perceived racism and bias Country Mistakenly recognised "
             "Psychological risk Emotion Terror, despair Feelings Embarrassment, stressful Moods Homesick Physical risk "
             "Healthy and illness Sickness affects the travel plan Personal safety Sexual harassment Environmental "
             "Keep cautious of surroundings Social tension Tourist image Tourist identity Afraid to be recognised "
             "Local people's attitude Hostility from local people Trust Lack of trust Family Friends Culture shock")
    assert _table_residue(cells)
    prose = ("Thematic analysis is used in this study to examine the gathered data, which is a popular technique for "
             "finding, examining and interpreting patterns in qualitative data (Braun and Clarke 2006), and this "
             "strategy was selected because it permits a flexible yet rigorous reading of the forty interviews that "
             "Nepal's female visitors gave about their routes, their lodgings and the people they met on the way")
    assert len(prose.split()) >= 60 and not _table_residue(prose)  # long sentence, few capitals
