"""Extract qualitative text from saved web pages (articles, blog posts, forums, transcripts).

A page becomes one record per article paragraph plus one record per reader
comment. Navigation, sidebars, sharing widgets and similar boilerplate are
dropped. Respondent IDs are pseudonymous: article paragraphs share
``<page>:author`` and each comment gets ``<page>:comment-<n>``. Commenter
names are never stored.

Podcast and interview transcripts, whose paragraphs open with a speaker label
(``<b>Amber:</b> ...``), become one record per paragraph with
``<page>:speaker-<n>``, numbered by first appearance. The label and a leading
timestamp are cut from the text, so speaker names are not stored either; names
spoken inside the conversation stay, they are part of what was said.

Pages with little or no article text (index and home pages, or sites that build their text with
JavaScript) are still read: the rest of the page's text blocks, and prose kept in its data scripts
(JSON-LD ``articleBody``, a framework's page data), become records with ``source_kind`` "page".

Only the HTML markup is parsed; no scripts are executed and nothing is fetched.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import NamedTuple

BLOCKS = ("p", "li", "blockquote", "h2", "h3", "h4", "dd", "td")
HEADINGS = ("h2", "h3", "h4")
DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "button", "select", "input", "textarea", "nav", "header", "footer", "form")
NOISE = {
    "sidebar", "widget", "share", "sharing", "sharedaddy", "social", "related", "newsletter", "subscribe", "subscription",
    "menu", "breadcrumb", "breadcrumbs", "cookie", "cookies", "popup", "modal", "advert", "ads", "ad", "promo", "signup",
    "footer", "navigation", "navbar", "nav", "toc", "pagination", "jp", "author", "bio", "disclosure",
}
COMMENT = {"comment", "comments", "comment-content", "comment-body", "commentlist", "comment-list"}
# comment forms and their notices are not comments
RESPOND = {"respond", "comment-respond", "comment-reply-title", "comment-form", "commentform", "logged-in-as", "comment-notes", "reply-title"}
MIN_WORDS = 4
# below this many words of article text the rest of the page is read as well
THIN_ARTICLE = 50
# below this many words, also after reading the page's other paragraphs, every block of visible text is read
NO_ARTICLE = 20
# Main-content widening (see _widen): a paragraph this long, mostly not link text, is article prose rather than a
# caption, teaser or notice; a wider container must add this share of new prose to the prose already held.
PROSE_WORDS = 20
WIDEN_SHARE = 0.5
# Transcripts (see _speaker_label / _is_transcript)
MIN_TURNS = 5
TIMESTAMP = r"(?:\[\d{1,2}(?::\d{2}){1,2}\]|\(\d{1,2}(?::\d{2}){1,2}\)|\d{1,2}:\d{2}:\d{2})"
SPEAKER_NAME = r"[A-Z][\w'’.-]*(?:\s+(?:[A-Z][\w'’.-]*|\d+)){0,2}"
# broadcast transcripts add a role after the name ("Heather Gibson, University of Florida:"); the role is not kept
SPEAKER_ROLE = r"[A-Z][\w'’.&-]*(?:\s+[\w'’.&-]+){0,5}"
SPEAKER = re.compile(rf"(?:{TIMESTAMP}\s*)?(?P<label>{SPEAKER_NAME}(?:,\s*{SPEAKER_ROLE})?)\s*:(?:\s+|$)")
BOLD_LABEL = re.compile(r"[A-Z][\w'’.-]*(?:\s+[\w'’.-]+){0,3}")
LEADING_TIMESTAMP = re.compile(rf"^\s*{TIMESTAMP}\s*")
INLINE_TIMESTAMP = re.compile(r"\s*\[\d{1,2}(?::\d{2}){1,2}\]")
# lead-ins and field names that open paragraphs in listicles, guides and FAQs but are not people
NOT_SPEAKERS = {
    "note", "notes", "nb", "ps", "tip", "tips", "pro tip", "top tip", "travel tip", "update", "updated", "edit", "source",
    "sources", "photo", "photos", "image", "credit", "credits", "caption", "warning", "caution", "important",
    "disclaimer", "disclosure", "example", "bonus", "step", "day", "week", "part", "chapter", "option", "price",
    "prices", "cost", "costs", "budget", "address", "location", "hours", "website", "phone", "email", "contact",
    "related", "read more", "see also", "also read", "q", "a", "question", "answer", "summary", "conclusion",
    "verdict", "rating", "pros", "cons", "best for", "highlights", "why", "how", "what", "when", "where", "who",
    "duration", "distance", "difficulty", "getting there", "accommodation", "takeaway", "key takeaways", "fun fact",
    "reminder", "faq", "bottom line", "transcript", "intro", "introduction", "outro", "music", "sponsor", "ad",
}
# Third-party post embeds. A saved page keeps only the embed's placeholder and its "A post shared by ..." line.
EMBEDS = {"instagram-media", "twitter-tweet", "twitter-timeline", "tiktok-embed", "fb-post", "fb-video", "bluesky-embed",
          "reddit-embed-bq", "text-post-media"}
EMBED_TEXT = re.compile(r"(?:A post shared by .{1,120}|View (?:this post|more) on (?:Instagram|TikTok|Threads))", re.I)
# Recirculation blocks inside the article container: a section under one of these headings links to other posts.
RELATED_HEADING = re.compile(
    r"(?:more inspiration|you (?:might|may) also (?:like|enjoy)|related(?: (?:posts?|articles?|stories|reading|content|guides?))?"
    r"|read (?:more|next)|recommended(?: (?:reading|posts|articles|for you))?|more (?:from|stories|to read|like this)\b.*"
    r"|popular (?:posts|articles|stories)|trending(?: now| stories\b.*)?|keep reading|further reading|also read|see also)"
    r"\s*[:.!…]*", re.I)
READ_MORE = re.compile(r"(?:read (?:more|next|also)|related|see also|also read)\s*:\s*", re.I)
LINK_SHARE = 0.9
# Podcast and newsletter sign-offs. A paragraph with two or more different cues is a call to action, not content;
# one cue alone ("sign up for a day trip") is ordinary advice.
CTA = re.compile(r"follow us on|subscrib(?:e|ing)\b|rate and review|leave (?:us )?a review|drop us a line|email us at"
                 r"|a production of|produced by|music (?:composition )?by|apple podcasts|google podcasts|spotify"
                 r"|wherever you (?:get|listen to) (?:your )?podcasts|sign up for (?:our|the) newsletter", re.I)
CTA_CUES = 2
EMAIL_PLACEHOLDER = re.compile(r"\[email\s*protected\]", re.I)


class _Leaf(NamedTuple):
    tag: object
    text: str
    words: int
    comment: object  # element holding the whole comment, or None for page content


def _tokens(tag) -> set[str]:
    attrs = []
    for key in ("id", "class", "role"):
        v = tag.get(key) if tag.attrs else None
        if isinstance(v, list):
            attrs.extend(v)
        elif v:
            attrs.append(v)
    out = set()
    for a in attrs:
        a = a.lower()
        out.add(a)
        out.update(t for t in re.split(r"[-_\s]+", a) if t)
    return out


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _charset(raw: bytes) -> str:
    m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", raw[:8000], re.I)
    if m:
        enc = m.group(1).decode("ascii", "ignore").lower()
        try:
            "".encode(enc)
            return enc
        except LookupError:
            pass
    return "utf-8"


def _ancestors(tag) -> list:
    """Ancestors below <body>, innermost first."""
    out = []
    for a in tag.parents:
        if a.name in ("body", "html", "[document]") or a.name is None:
            break
        out.append(a)
    return out


def _link_words(tag) -> int:
    return sum(len(a.get_text(" ", strip=True).split()) for a in tag.find_all("a"))


def _widen(main, prose: list[_Leaf]):
    """Climb from the densest container to the one holding the whole article.

    CMS modules, columns and per-section wrappers split one article over sibling
    containers, so the densest container may hold only part of it. Walking up,
    an ancestor is taken when the prose it adds (outside the container held so
    far and not under a boilerplate marker) is at least WIDEN_SHARE of the prose
    already held. Wrappers that add nothing are passed through; the climb stops
    at the first ancestor that adds only a little (a sidebar or teaser rail next
    to the article) and at boilerplate-marked wrappers, above which the text held
    would count as inner noise.
    """
    chains = []
    for lf in prose:
        anc = _ancestors(lf.tag)
        noise_at = next((i for i, a in enumerate(anc) if _tokens(a) & NOISE), len(anc))
        chains.append(({id(a): i for i, a in enumerate(anc)}, noise_at, lf.words))

    def prose_under(node, held=None) -> int:
        total = 0
        for pos, noise_at, words in chains:
            i = pos.get(id(node))
            if i is None or i > noise_at or (held is not None and id(held) in pos):
                continue
            total += words
        return total

    if _tokens(main) & NOISE:
        return main
    best, held = main, prose_under(main)
    for anc in _ancestors(main):
        new = prose_under(anc, best)
        if new and new >= max(PROSE_WORDS, WIDEN_SHARE * held):
            best, held = anc, held + new
        elif new:
            break
        if _tokens(anc) & NOISE:
            break
    return best


def _main_container(leaves: list[_Leaf]):
    """Main content container, readability-style, or None when the page has no paragraphs.

    Each paragraph credits its parent fully and its grandparent by half; comments
    do not count. The best-scoring container is then widened (see _widen).
    """
    score: Counter = Counter()
    nodes = {}
    for lf in leaves:
        if lf.comment is not None or lf.tag.name != "p" or lf.words < MIN_WORDS:
            continue
        parent = lf.tag.parent
        if parent is not None:
            score[id(parent)] += lf.words
            nodes[id(parent)] = parent
            if parent.parent is not None and parent.parent.name not in ("body", "html", "[document]"):
                score[id(parent.parent)] += lf.words / 2
                nodes[id(parent.parent)] = parent.parent
    if not score:
        return None
    prose = [lf for lf in leaves if lf.comment is None and lf.tag.name == "p" and lf.words >= PROSE_WORDS
             and not _tokens(lf.tag) & NOISE and _link_words(lf.tag) < lf.words / 2]
    return _widen(nodes[score.most_common(1)[0][0]], prose)


def _heading_key(text: str) -> str:
    """Heading text without numbering, case or punctuation, to spot a contents list that mirrors the headings."""
    t = re.sub(r"^\s*(?:\d+|[ivx]+)\s*[.):-]\s*", "", text.lower())
    return " ".join(re.findall(r"[^\W_]+", t))


def _is_toc_entry(block, words: int, page_url: str | None) -> bool:
    """A list entry that is (almost) all links to anchors on the same page: a table of contents."""
    here = (page_url or "").split("#")[0].rstrip("/")
    linked = 0
    for a in block.find_all("a", href=True):
        target, _, frag = a["href"].partition("#")
        if frag and (not target or (here and target.rstrip("/") == here)):
            linked += len(a.get_text(" ", strip=True).split())
    return linked > 0 and linked >= 0.8 * words


def _is_link_entry(block, text: str, words: int) -> bool:
    """A list entry, or a "Read more: ..." line, that is (almost) all links to other pages: a link to another post."""
    m = READ_MORE.match(text)
    if block.name != "li" and not m:
        return False
    linked = sum(len(a.get_text(" ", strip=True).split()) for a in block.find_all("a", href=True) if not a["href"].startswith("#"))
    rest = words - (len(m.group(0).split()) if m else 0)
    return rest > 0 and linked >= LINK_SHARE * rest


def _is_prompt(block, words: int) -> bool:
    """A paragraph set entirely in bold: in a written Q&A, the interviewer's question, not the last speaker's turn."""
    bold = sum(len(x.get_text(" ", strip=True).split()) for x in block.find_all(["b", "strong"]))
    return words > 0 and bold >= LINK_SHARE * words


def _is_call_to_action(text: str) -> bool:
    return len({m.group(0).lower() for m in CTA.finditer(text)}) >= CTA_CUES


def _flatten_tables(soup) -> None:
    """Read a data table one row at a time: each body row becomes one paragraph, cells joined by '; '.

    A row is one statement ("Small European cities; Walkable, connected, easy to explore; First-time solo
    travelers"); cell by cell it would be split into fragments that mean little alone. Header rows are left out,
    since repeating them in every row would turn column names into frequent phrases. Layout tables, whose cells
    hold paragraphs or lists, are left alone and read block by block.
    """
    for table in soup.find_all("table"):
        cells = table.find_all(["td", "th"])
        if not cells or table.find("table") or any(c.find(BLOCKS) for c in cells):
            continue
        holder = soup.new_tag("div")
        for tr in table.find_all("tr"):
            if tr.find("td") is None:  # header row
                continue
            parts = [t for c in tr.find_all(["td", "th"]) if (t := _clean(c.get_text(" ", strip=True)))]
            if parts:
                p = soup.new_tag("p")
                p.string = "; ".join(parts)
                holder.append(p)
        table.replace_with(holder)


def _speaker_label(block, text: str) -> tuple[str, str] | None:
    """(speaker key, text after the label) when a paragraph opens with a speaker label, else None.

    A label is plain text of 1-3 capitalised words followed by ':' (``Amber: ...``,
    ``Speaker 2: ...``), or a leading <b>/<strong> name followed by ':'; a timestamp
    such as ``[00:01:23]`` may come before or after it. A full name may carry a role
    (``Molly Furey, Traveler: ...``); the speaker is the name alone, so a person keeps
    one key whether or not the role is repeated.
    """
    label = rest = None
    m = SPEAKER.match(text)
    if m:
        label, rest = m.group("label"), text[m.end():]
    else:
        bold = block.find(["b", "strong"])
        btxt = _clean(bold.get_text(" ", strip=True)) if bold is not None else ""
        if btxt and text.startswith(btxt):
            after = text[len(btxt):].lstrip()
            if btxt.endswith(":"):
                label, rest = btxt[:-1].strip(), after
            elif after.startswith(":"):
                label, rest = btxt, after[1:]
            if label is not None and (len(label) > 80 or not BOLD_LABEL.fullmatch(label.partition(",")[0].strip())):
                label = None
    if label is None:
        return None
    name, comma, role = (x.strip() for x in label.partition(","))
    if comma and (len(name.split()) < 2 or not role or len(role.split()) > 6):
        return None  # "However, in my view: ..." is a sentence; a role follows a full name
    key = " ".join(name.lower().rstrip(".").split())
    if key in NOT_SPEAKERS or key.split()[0] in NOT_SPEAKERS:
        return None
    return key, LEADING_TIMESTAMP.sub("", rest)


def _is_transcript(labels: list[str], paragraphs: int) -> bool:
    """Enough labelled paragraphs, by recurring speakers, to read the page as a conversation.

    Speakers take several turns each; bold lead-ins in a listicle ("Pack light:")
    are all different, and a few quoted people in a long article are a small share.
    """
    counts = Counter(labels)
    return (len(labels) >= MIN_TURNS and len(counts) >= 2 and len(labels) >= 0.2 * paragraphs
            and sum(c for c in counts.values() if c > 1) >= len(labels) / 2)


def page_slug(title: str, fallback: str) -> str:
    base = title or fallback
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    return (slug[:40].rstrip("-") or "page")


def _prose(t: str) -> bool:
    return len(t.split()) >= MIN_WORDS and not t.startswith(("http://", "https://")) and \
        sum(ch.isalpha() for ch in t) >= 0.6 * len(t.replace(" ", ""))


# elements that start a new line of text; inline elements (span, a, b, em ...) join the line they sit in
LINE_BLOCKS = {"body", "main", "article", "section", "div", "p", "li", "dd", "dt", "td", "th", "blockquote", "pre", "figcaption",
               "h1", "h2", "h3", "h4", "h5", "h6"}


def _visible_blocks(soup) -> list[str]:
    """The page's visible text, one entry per block element, for pages that keep their prose in plain <div>s:
    sidebars, sharing widgets, comment forms and comments (read separately) are left out.

    A box named like boilerplate ("widget", "sidebar") counts as boilerplate only when it holds under 30% of the
    page's text: some templates (Blogger) wrap the post itself in a "widget" inside a "has-sidebar" layout.
    """
    root = soup.body or soup
    total = len(root.get_text(" ", strip=True)) or 1
    sizes: dict[int, int] = {}

    def small(tag) -> bool:
        if id(tag) not in sizes:
            sizes[id(tag)] = len(tag.get_text(" ", strip=True))
        return sizes[id(tag)] < 0.3 * total

    parts: dict[int, list[str]] = {}
    order: list[int] = []
    for string in root.find_all(string=True):
        if type(string).__name__ != "NavigableString" or not string.strip():
            continue  # HTML comments, doctype, CDATA
        block, skip = None, False
        for anc in string.parents:
            tokens = _tokens(anc)
            if anc.name == "aside" or tokens & (RESPOND | COMMENT) or (tokens & NOISE and small(anc)):
                skip = True
                break
            if block is None and anc.name in LINE_BLOCKS:
                block = anc
        if skip or block is None:
            continue
        key = id(block)
        if key not in parts:
            parts[key] = []
            order.append(key)
        parts[key].append(string.strip())
    return [t for t in (_clean(EMAIL_PLACEHOLDER.sub(" ", " ".join(parts[k]))) for k in order) if _prose(t)]


def _embedded_texts(soup) -> list[str]:
    """Prose kept in a page's data scripts: sites that build their text with JavaScript often leave it there."""
    from bs4 import BeautifulSoup

    out: list[str] = []

    def walk(v, depth=0):
        if depth > 40:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, str) and len(v.split()) >= 12:
            text = BeautifulSoup(v, "html.parser").get_text("\n") if "<" in v else v
            out.extend(t for t in (_clean(part) for part in re.split(r"\n+", text)) if _prose(t))

    for script in soup.find_all("script"):
        if "json" not in (script.get("type") or "").lower() and script.get("id") != "__NEXT_DATA__":
            continue
        try:
            walk(json.loads(script.string or script.get_text() or ""))
        except (ValueError, RecursionError):
            continue
    return out


def extract_html(raw: bytes, fallback_name: str = "page") -> tuple[list[dict], list[str], dict]:
    """Return (rows, warnings, page_info). Each row has text and provenance fields."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency is required by the package
        raise RuntimeError("HTML import needs beautifulsoup4 (pip install beautifulsoup4).") from exc

    enc = _charset(raw)
    soup = BeautifulSoup(raw.decode(enc, errors="replace"), "html.parser")
    meta = lambda prop: (soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop}) or {}).get("content")  # noqa: E731
    title = _clean(meta("og:title") or (soup.title.get_text() if soup.title else "") or fallback_name)
    published = meta("article:published_time") or meta("datePublished")
    site = meta("og:site_name")
    canonical = (soup.find("link", rel="canonical") or {}).get("href") or meta("og:url")
    embedded = _embedded_texts(soup)
    for t in soup(DROP_TAGS):
        t.decompose()
    for t in [t for t in soup.find_all(True) if t.attrs and _tokens(t) & EMBEDS]:
        if not t.decomposed:
            t.decompose()
    _flatten_tables(soup)

    def comment_container(block):
        """The element holding one whole comment (nested replies stay separate), or None."""
        fallback = None
        for anc in _ancestors(block):
            tok = _tokens(anc)
            if tok & RESPOND:
                return None
            single = "comment" in tok or any(x.startswith("comment-") and x[8:].isdigit() for x in tok)
            if anc.name in ("li", "article") and single:
                return anc
            if fallback is None and tok & COMMENT:
                fallback = anc
        return fallback

    # Innermost blocks only, so text is not duplicated. Short paragraphs are kept here because a
    # transcript turn such as "Nikki: OK." still changes who is speaking; they never become records.
    leaves = []
    for b in soup.find_all(BLOCKS):
        if b.find(BLOCKS):
            continue
        text = _clean(EMAIL_PLACEHOLDER.sub(" ", b.get_text(" ", strip=True)))
        if not text or EMBED_TEXT.fullmatch(text) or any(a.name == "aside" or (_tokens(a) & RESPOND) for a in _ancestors(b)):
            continue
        leaves.append(_Leaf(b, text, len(text.split()), comment_container(b)))

    main = _main_container(leaves)

    def inner_noise(block) -> bool:
        """Boilerplate inside the main container (share buttons, related posts, ...)."""
        for anc in [block, *block.parents]:
            if anc is main or anc.name in ("body", "html", "[document]") or anc.name is None:
                return False
            if _tokens(anc) & NOISE:
                return True
        return False

    content = []
    if main is not None:
        content = [lf for lf in leaves if lf.comment is None and any(a is main for a in lf.tag.parents) and not inner_noise(lf.tag)]
    # a contents list repeats the headings; the headings already label each record's section
    heading_keys = {_heading_key(lf.text) for lf in content if lf.tag.name in HEADINGS}
    content = [lf for lf in content if lf.tag.name in HEADINGS
               or not (_heading_key(lf.text) in heading_keys or _is_toc_entry(lf.tag, lf.words, canonical)
                       or _is_link_entry(lf.tag, lf.text, lf.words))]
    paragraphs = [lf for lf in content if lf.tag.name == "p"]
    labels = {id(lf.tag): lab for lf in paragraphs if (lab := _speaker_label(lf.tag, lf.text))}
    transcript = _is_transcript([lab[0] for lab in labels.values()], len(paragraphs))
    keep = {id(lf.tag) for lf in content}

    rows, comment_ids, speakers = [], {}, {}
    slug = page_slug(title, fallback_name)
    section = speaker = related = None  # related: level of the recirculation heading being skipped
    n_block = main_words = transcript_rows = 0
    for lf in leaves:
        b, text = lf.tag, lf.text
        if lf.comment is None:
            if id(b) not in keep:
                continue
            if b.name in HEADINGS:  # headings are short but name sections
                level = HEADINGS.index(b.name)
                if related is not None and level > related:
                    continue  # a card title inside the recirculation block
                section, speaker = text, None
                related = level if RELATED_HEADING.fullmatch(text) else None
                continue
            if related is not None:
                continue
            rid, kind = f"{slug}:author", "article"
            if transcript and b.name == "p":  # an unlabelled paragraph continues the previous speaker
                if id(b) in labels:
                    key, text = labels[id(b)]
                    speaker = speakers.setdefault(key, len(speakers) + 1)
                elif _is_prompt(b, lf.words):
                    speaker = None
                if speaker is not None:
                    rid, kind = f"{slug}:speaker-{speaker}", "transcript"
                    text = _clean(INLINE_TIMESTAMP.sub("", text))
            words = len(text.split())
            if words < MIN_WORDS or _is_call_to_action(text):
                continue
            n_block += 1
            main_words += words
            transcript_rows += kind == "transcript"
            rows.append({"respondent_id": rid, "text": text, "source_kind": kind, "section": section,
                         "block": n_block, "published": published, "page_title": title})
            continue
        if lf.words < MIN_WORDS:
            continue
        key = id(lf.comment)
        rid = f"{slug}:comment-{comment_ids.setdefault(key, len(comment_ids) + 1)}"
        existing = next((r for r in reversed(rows) if r["respondent_id"] == rid), None)
        if existing is not None:  # further paragraphs of the same comment stay in one record
            existing["text"] += "\n\n" + text
            continue
        tm = lf.comment.find("time")
        rows.append({"respondent_id": rid, "text": text, "source_kind": "comment", "section": "Reader comments", "block": None,
                     "published": tm.get("datetime") if tm is not None and tm.get("datetime") else None, "page_title": title})

    # Little or no article text (an index or home page, prose kept in plain <div>s, or text that loads with
    # JavaScript): read the page's other paragraphs and its data scripts; if that is still thin, every block of
    # visible text; if there is nothing at all, its description and title.
    page_words = 0
    if main_words < THIN_ARTICLE:
        seen = {r["text"] for r in rows}
        found = [lf.text for lf in leaves if lf.comment is None and lf.tag.name not in HEADINGS and _prose(lf.text)
                 and not (_is_call_to_action(lf.text) or _is_toc_entry(lf.tag, lf.words, canonical) or _is_link_entry(lf.tag, lf.text, lf.words)
                          or any(_tokens(a) & NOISE for a in [lf.tag, *lf.tag.parents]))] + embedded
        if main_words + sum(len(t.split()) for t in found if t not in seen) < NO_ARTICLE:
            found += _visible_blocks(soup)
        if not found and not rows:
            found = [t for t in (_clean(meta("og:description") or meta("description") or ""), title) if t]
        for text in found:
            if text in seen:
                continue
            seen.add(text)
            n_block += 1
            page_words += len(text.split())
            rows.append({"respondent_id": f"{slug}:author", "text": text, "source_kind": "page", "section": None,
                         "block": n_block, "published": published, "page_title": title})

    warnings = []
    n_speakers = len(speakers) if transcript_rows else 0
    if n_speakers:
        warnings.append(
            f"Read as a transcript: {transcript_rows} paragraphs from {n_speakers} speakers were imported as "
            f"{slug}:speaker-1..{n_speakers} (numbered by first appearance); speaker labels and timestamps were "
            "removed from the text."
        )
    n_comments = len(comment_ids)
    info = {"title": title, "site": site, "url": canonical, "published": published, "article_words": main_words, "page_words": page_words,
            "comments": n_comments, "speakers": n_speakers, "encoding": enc}
    return rows, warnings, info
