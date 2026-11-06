"""Rebuild paragraphs from the text layer of a PDF (papers, theses, reports, transcripts).

pypdf returns each page as lines with no paragraph breaks, plus running
headers and footers, page numbers, hyphenated line breaks, contents pages,
table cells and the reference list. Read naively, a whole page becomes one
"paragraph". Here lines are rejoined into paragraphs:

* a paragraph ends at a blank line, at a heading, or after a line that ends a
  sentence and is clearly shorter than the page's typical line; one that does
  not end a sentence continues on the next page;
* lines recurring at the top or bottom of many pages (after masking digits),
  page numbers, DOI/URL-only lines and copyright lines are dropped;
* a word hyphenated across a line break is rejoined without the hyphen when the
  unhyphenated form occurs elsewhere in the document, kept hyphenated when the
  hyphenated form does ("self-actualization"), and otherwise rejoined, which is
  how typesetters break words; text-layer spacing artefacts ("tour - ists",
  "part icipants", "fact ,") are repaired the same way, from words the document
  itself uses;
* contents lines (dot leaders), figure/table captions, table rows (short lines
  ending in numbers or Y/N cells) and paragraphs that are mostly numbers are
  dropped; so are back-matter sections (funding, author contributions, ORCID,
  conflict-of-interest statements), and nothing after a References /
  Bibliography heading is imported (what follows is the reference list and, in
  theses, appendices of questionnaires and consent forms).

One document is one respondent (``<slug>:author``), as for a web article.
Scanned pages have no text layer; OCR is not included.
"""
from __future__ import annotations

import re
from collections import Counter
from io import BytesIO
from pathlib import Path


from thematicml.data.html import page_slug

MIN_WORDS = 4
EDGE_LINES = 3  # header/footer candidates: this many non-empty lines at the top and bottom of a page
SENTENCE_END = re.compile(r"[.!?:;…][\"'”’)\]]*$")
REFERENCES = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:references?|bibliography|works cited|reference list|literature cited|"
                        r"referências|referencias|referências bibliográficas|bibliografía|bibliografia)$", re.IGNORECASE)
BACK_MATTER = re.compile(r"^(?:acknowledge?ments?|funding|author contributions?|declaration of conflicting interests?|"
                         r"conflicts? of interests?|competing interests?|orcid ids?|data availability statement|"
                         r"consent to participate|consent for publication|ethic(?:s|al) (?:approval|statement)|"
                         r"supplementa(?:l|ry) material|author biograph(?:y|ies)|about the authors?)$", re.IGNORECASE)
CAPTION = re.compile(r"^(?:(?:figure|fig\.|table|tabela|tabla|chart|graph|gráfico|image|photo|source|fonte|fuente|note)\s*\d+[a-z]?\b"
                     r"|(?:source|fonte|fuente)s?\s*:)", re.IGNORECASE)
NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[ivxlc]+\.|[A-Z]\.)\s+\S")
PAGE_NUMBER = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
LINK_ONLY = re.compile(r"^(?:https?://\S+|www\.\S+|doi:?\s*\S+|\S+@\S+\.\w+)$", re.IGNORECASE)
DOT_LEADER = re.compile(r"(?:\.\s?){4,}")
SENTENCE_STOP = re.compile(r"[.!?…][\"'”’)\]]*(?:\s|$)")  # a sentence end anywhere in a paragraph
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBER = re.compile(r"^[(\[]?[-+]?\d[\d.,]*%?[)\]]?$")


def _norm(line: str) -> str:
    """Line identity for header/footer detection: digits masked, case and spacing ignored."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def _boilerplate_forms(pages: list[list[str]]) -> set[str]:
    """Normalised lines recurring at the top or bottom of at least max(3, 30%) of pages."""
    counts: Counter = Counter()
    for lines in pages:
        content = [ln for ln in lines if ln.strip()]
        # running heads are short; a long line recurring at a page edge is more likely repeated text than a header
        counts.update({_norm(ln) for ln in content[:EDGE_LINES] + content[-EDGE_LINES:] if len(ln.split()) <= 15})
    need = max(3, 0.3 * len(pages))
    return {form for form, c in counts.items() if c >= need and form}


def _is_noise_line(line: str, forms: set[str]) -> bool:
    return (_norm(line) in forms or bool(PAGE_NUMBER.match(line)) or bool(LINK_ONLY.match(line)) or line.startswith("©")
            or bool(DOT_LEADER.search(line)))


def _is_number(tok: str) -> bool:
    return bool(NUMBER.match(tok))


def _is_table_row(line: str, width: float) -> bool:
    """A short line without sentence punctuation that ends in a number or in single-letter cells (Y Y N)."""
    toks = line.split()
    if not toks or SENTENCE_END.search(line) or len(line) >= 0.75 * width:
        return False
    cells = len(toks) >= 2 and all(len(t) == 1 and t not in ("a", "I") for t in toks[-2:])
    return _is_number(toks[-1]) or cells


def _is_heading(line: str, width: float) -> bool:
    """A short standalone line without sentence punctuation that starts like a title."""
    words = line.split()
    if not words or len(words) > 12 or len(line) > 0.7 * width or SENTENCE_END.search(line) or line.endswith((",", "-")):
        return False
    if not WORD.search(line) or _is_number(words[-1]) or sum(_is_number(w) for w in words) >= 2:
        return False  # page ranges ("1 -20") and table totals are not headings
    return line[0].isupper() or bool(NUMBERING.match(line))


def _join(prev: str, nxt: str, vocab: set[str], hyphenated: set[str]) -> str:
    """Append a wrapped line, rejoining a word hyphenated across the break."""
    m = re.search(r"([^\W\d_]+) ?-$", prev)
    tail = re.match(r"[^\W\d_]+", nxt)
    if m and tail:
        head, stem = m.group(1), prev[:m.start()] + m.group(1)
        if tail.group(0)[0].isupper():  # "Anglo-" + "Saxon": a real compound
            return f"{stem}-{nxt}"
        joined, compound = (head + tail.group(0)).lower(), f"{head}-{tail.group(0)}".lower()
        if compound in hyphenated and joined not in vocab:
            return f"{stem}-{nxt}"
        return stem + nxt
    return f"{prev} {nxt}"


def _table_residue(text: str) -> bool:
    """Table content flattened into text: many numbers or single-letter cells, or few real words, or a long
    run of capitalised cell text with almost no sentence ends (a word table: "Age Too young… Race Perceived…")."""
    toks = text.split()
    numbers = sum(_is_number(t.strip(".,;:")) for t in toks)
    cells = sum(1 for t in toks if len(t) == 1 and t not in ("a", "A", "I", "&"))
    alpha = sum(1 for t in toks if WORD.fullmatch(t.strip(".,;:()!?\"'“”’‘")))
    word_table = (len(toks) >= 60 and len(SENTENCE_STOP.findall(text)) * 40 < len(toks)
                  and sum(t[:1].isupper() for t in toks) >= 0.25 * len(toks))
    return ((numbers >= 4 and numbers >= 0.25 * len(toks)) or (cells >= 4 and cells >= 0.2 * len(toks))
            or alpha < 0.5 * len(toks) or word_table)


def _repair(text: str, counts: Counter, hyphenated: set[str]) -> str:
    """Undo spacing artefacts of the text layer using the document's own vocabulary."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # imported on first use: slow to load in the browser edition

    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)  # "In fact ," -> "In fact,"

    def hyphen(m):  # "in -depth", "tour - ists", "self - actualization"; a spaced dash between words stays
        a, b = m.group(1), m.group(2)
        if f"{a}-{b}".lower() in hyphenated:
            return f"{a}-{b}"
        if counts[(a + b).lower()] >= 1 and counts[b.lower()] <= 1:
            return a + b
        return f"{a}-{b}" if m.group(0).count(" ") == 1 else m.group(0)

    text = re.sub(r"\b([^\W\d_]+)(?: -|- | - )([a-z][^\W\d_]*)\b", hyphen, text)
    # "Khoo -Lattimore", "non -Western", "25 -34": a space before a joining hyphen is never typography
    # (a word followed by " -8" is a negative number and stays)
    text = re.sub(r"\b([^\W_]+) -(?=[^\W_])(?!\d)|\b(\d+) -(?=\d)", lambda m: (m.group(1) or m.group(2)) + "-", text)

    # "part icipants": rejoin when the second piece never occurs as a word of its own in this document
    # and the joined word does; function words on either side are left alone ("a round", "to gether")
    toks, out = re.sub(r"\s+", " ", text).strip().split(" "), []
    i = 0
    while i < len(toks):
        a = toks[i]
        b = re.match(r"([a-z]{3,})(\W*)$", toks[i + 1]) if i + 1 < len(toks) else None
        if (b and len(a) >= 2 and WORD.fullmatch(a) and not (len(a) <= 3 and a.lower() in ENGLISH_STOP_WORDS)
                and b.group(1) not in ENGLISH_STOP_WORDS and counts[b.group(1)] <= 1 and counts[(a + b.group(1)).lower()] >= 2):
            out.append(a + toks[i + 1])
            i += 2
            continue
        out.append(a)
        i += 1
    return " ".join(out)


def _title(reader, pages: list[list[str]], fallback: str) -> str:
    """Document title from the PDF metadata when it appears on the first pages (templates leave junk titles)."""
    try:
        meta = reader.metadata.title if reader.metadata else None
    except Exception:  # malformed metadata dictionaries are common
        meta = None
    if meta and meta.strip():
        head = re.sub(r"\s+", " ", " ".join(" ".join(p) for p in pages[:2]).lower())
        probe = re.sub(r"\s+", " ", meta.strip().lower())[:30]
        if probe and probe in head:
            return meta.strip()
    return fallback


def _median(values: list[int], default: float) -> float:
    return sorted(values)[len(values) // 2] if values else default


def paragraphs_from_pages(pages: list[list[str]]) -> tuple[list[dict], dict]:
    """Rejoin page lines into paragraphs. Returns (paragraphs, statistics).

    Each paragraph: text, page (1-based page where it starts), section (latest heading or None).
    """
    forms = _boilerplate_forms(pages)
    full = " ".join(" ".join(p) for p in pages)
    vocab = {w.lower() for w in WORD.findall(full)}
    hyphenated = {h.lower() for h in re.findall(r"[^\W\d_]+-[^\W\d_]+", full)}
    doc_width = _median([len(ln.strip()) for p in pages for ln in p if len(ln.strip()) >= 20], 80)

    paras: list[dict] = []
    stats = {"noise_lines": 0, "table_lines": 0, "headings": 0, "captions": 0, "back_matter_lines": 0,
             "references_page": None, "pages_after_references": 0}
    cur: dict | None = None
    section, back_matter = None, False

    def flush():
        nonlocal cur
        if cur is not None:
            paras.append(cur)
        cur = None

    for page_no, lines in enumerate(pages, start=1):
        page_lengths = [len(ln.strip()) for ln in lines if len(ln.strip()) >= 20]
        width = _median(page_lengths, doc_width) if len(page_lengths) >= 5 else doc_width
        if width < 0.8 * doc_width:  # a page of table rows or a list: its median line is not the text width
            width = doc_width
        for raw in lines:
            line = raw.strip()
            if not line:
                if cur is not None and SENTENCE_END.search(cur["text"]):
                    flush()
                continue
            if _is_noise_line(line, forms):
                stats["noise_lines"] += 1
                continue
            ends = cur is None or bool(SENTENCE_END.search(cur["text"]))
            if _is_table_row(line, width):  # before headings: a row such as "Connect with locals Y Y N" looks like one
                stats["table_lines"] += 1
                continue
            if ends and _is_heading(line, width):
                flush()
                if REFERENCES.match(line):
                    stats["references_page"] = page_no
                    stats["pages_after_references"] = len(pages) - page_no
                    return paras, stats
                if CAPTION.match(line):
                    stats["captions"] += 1
                else:
                    section, back_matter = line, bool(BACK_MATTER.match(line))
                    stats["headings"] += 1
                continue
            if back_matter:
                stats["back_matter_lines"] += 1
                continue
            if CAPTION.match(line) and ends:
                flush()
                stats["captions"] += 1
                continue
            if cur is None:
                cur = {"text": line, "page": page_no, "section": section}
            else:
                cur["text"] = _join(cur["text"], line, vocab, hyphenated)
            if SENTENCE_END.search(line) and len(line) < 0.85 * width:
                flush()
        # a page that ends a sentence usually ends the paragraph too; splitting one at a sentence
        # boundary is harmless, merging two unrelated paragraphs is not
        if cur is not None and SENTENCE_END.search(cur["text"]):
            flush()
    flush()
    return paras, stats


def extract_pdf(source, fallback_name: str = "document") -> tuple[list[dict], list[str], dict]:
    """Return (rows, warnings, info), like :func:`thematicml.data.html.extract_html`.

    ``source`` is a path or the file's bytes. Rows: respondent_id, text, source_kind
    ("document"), section, source_page, paragraph, page_title.
    """
    from pypdf import PdfReader

    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("The PDF is password-protected.")
    pages, empty = [], 0
    for page in reader.pages:
        text = page.extract_text() or ""
        if not text.strip():
            empty += 1
        pages.append(text.replace("\r\n", "\n").split("\n"))
    title = _title(reader, pages, fallback_name)
    paras, stats = paragraphs_from_pages(pages)
    counts = Counter(w.lower() for p in pages for ln in p for w in WORD.findall(ln))
    hyphenated = {h.lower() for p in pages for ln in p for h in re.findall(r"[^\W\d_]+-[^\W\d_]+", ln)}

    rows, dropped = [], {"short": 0, "table": 0}
    slug = page_slug(title, fallback_name)
    for p in paras:
        text = _repair(p["text"], counts, hyphenated)
        if len(text.split()) < MIN_WORDS:
            dropped["short"] += 1
            continue
        if _table_residue(text):
            dropped["table"] += 1
            continue
        rows.append({"respondent_id": f"{slug}:author", "text": text, "source_kind": "document", "section": p["section"],
                     "source_page": p["page"], "paragraph": len(rows) + 1, "page_title": title})

    warnings = []
    if empty:
        warnings.append(f"{empty} page(s) contained no extractable text (scanned images need OCR, which is not included).")
    if stats["references_page"]:
        warnings.append(f"Text from the References heading on page {stats['references_page']} to the end "
                        f"({stats['pages_after_references']} further page(s)) was not imported: reference lists and appendices.")
    tables = stats["table_lines"] + dropped["table"]
    if stats["noise_lines"] or tables or stats["back_matter_lines"]:
        warnings.append(f"Removed {stats['noise_lines']} running header/footer, page-number, link or contents line(s), "
                        f"{tables} table row(s) or fragment(s) and {stats['back_matter_lines']} line(s) of back matter "
                        "(funding, author contributions, conflict-of-interest statements).")
    info = {"title": title, "pages": len(pages), "empty_pages": empty, "paragraphs": len(rows), **stats,
            **{f"dropped_{k}": v for k, v in dropped.items()}}
    return rows, warnings, info
