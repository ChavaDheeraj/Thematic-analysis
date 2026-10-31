from thematicml.data.html import extract_html

P = "This is a long paragraph of article prose about travelling alone that easily passes the length threshold for prose {}."

SPLIT = f"""<html><head><title>Split article</title></head><body>
<div class="page"><div class="content">
  <div class="block"><p>{P.format(1)}</p><p>{P.format(2)}</p><p>{P.format(3)}</p></div>
  <div class="block"><p>{P.format(4)}</p><p>{P.format(5)}</p></div>
  <div class="related-posts"><p>{P.format("related")}</p></div>
</div></div>
<footer><p>Copyright footer text that should never be imported here.</p></footer></body></html>"""

TRANSCRIPT = """<html><head><title>Episode 6</title></head><body><article>
<p>In this episode three women talk about travelling alone and what it taught them.</p>
<p><b>Amber:</b> [00:00:00] Welcome to the show, today we are talking about solo travel and fear.</p>
<p><b>Becky:</b> Thanks for having me, I started travelling alone only a year or two ago.</p>
<p>It was late in life, but it changed how I see myself completely.</p>
<p><b>Kristen:</b> I backpacked across Australia for two months on my own when I was younger.</p>
<p><b>Amber:</b> What were you most afraid of before your first trip on your own?</p>
<p><b>Becky:</b> The language barrier, honestly, but hand gestures and a little English go far.</p>
<p><b>Kristen:</b> Eating alone at the bar of a restaurant is my favourite way to have dinner.</p>
</article></body></html>"""

TOC = """<html><head><title>Tips page</title><link rel="canonical" href="https://example.org/tips"></head><body><main>
<ul><li><a href="#trust-your-gut">Trust your gut when something feels off</a></li>
<li><a href="#arrive-early">Arrive in daylight whenever you possibly can</a></li></ul>
<h2 id="trust-your-gut">1. Trust Your Gut When Something Feels Off</h2>
<p>If a situation feels wrong, leave it, even if it means being rude to someone who seems friendly.</p>
<h2 id="arrive-early">2. Arrive in Daylight Whenever You Possibly Can</h2>
<p>Arriving while it is light out makes the trip from the airport to your accommodation much easier.</p>
</main></body></html>"""


def test_article_split_over_sibling_containers_is_read_whole():
    rows, _, _ = extract_html(SPLIT.encode(), "split")
    texts = [r["text"] for r in rows]
    assert texts == [P.format(i) for i in range(1, 6)]  # both blocks, not the related posts or the footer


def test_transcript_speakers_are_pseudonymous_and_labels_removed():
    rows, warnings, info = extract_html(TRANSCRIPT.encode(), "ep")
    turns = [r for r in rows if r["source_kind"] == "transcript"]
    assert [r["respondent_id"] for r in turns] == [
        "episode-6:speaker-1", "episode-6:speaker-2", "episode-6:speaker-2", "episode-6:speaker-3",
        "episode-6:speaker-1", "episode-6:speaker-2", "episode-6:speaker-3"]
    assert turns[0]["text"] == "Welcome to the show, today we are talking about solo travel and fear."
    assert turns[2]["text"].startswith("It was late in life")  # unlabelled paragraph continues the speaker
    joined = " ".join(r["text"] for r in rows)
    for name in ("Amber:", "Becky:", "Kristen:", "[00:00:00]"):
        assert name not in joined
    assert info["speakers"] == 3 and any("transcript" in w for w in warnings)
    assert rows[0]["source_kind"] == "article"  # the intro before the first label


def test_bold_lead_ins_in_a_listicle_are_not_speakers():
    page = "<html><head><title>Tips</title></head><body><article>" + "".join(
        f"<p><b>Tip {name}:</b> {P.format(name)}</p>" for name in ("Pack", "Plan", "Share", "Rest", "Walk", "Eat")) + "</article></body></html>"
    rows, _, info = extract_html(page.encode(), "tips")
    assert info["speakers"] == 0 and {r["source_kind"] for r in rows} == {"article"}


def test_contents_list_mirroring_headings_is_dropped():
    rows, _, _ = extract_html(TOC.encode(), "toc")
    texts = [r["text"] for r in rows]
    assert len(texts) == 2 and texts[0].startswith("If a situation feels wrong")
    assert rows[0]["section"] == "1. Trust Your Gut When Something Feels Off"
