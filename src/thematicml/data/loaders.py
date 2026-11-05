"""Load qualitative datasets from common file formats into a DataFrame.

Any file that holds readable text is accepted, whatever its extension; audio and video are refused
(upload a transcript instead). The kind of file is decided from its content as well as its name, so a
mislabelled or extension-less file still gets the right reader.

Tabular formats (CSV, TSV, XLSX, JSON, JSONL) keep their columns. Text documents (TXT and any other
plain text, Markdown, RTF, DOCX, ODT, XML, EPUB) become one record per paragraph with a generated
``record_id``; PowerPoint (PPTX) one record per slide; subtitles (SRT, WebVTT) one record per passage of
consecutive captions. PDFs have their paragraphs rebuilt from the text layer (see ``thematicml.data.pdf``)
and, like a web article, one pseudonymous respondent per document, with ``source_page`` and ``section``
provenance. Saved web pages (HTML) become one record per article paragraph and per reader comment, with
pseudonymous respondent IDs (see ``thematicml.data.html``). Older Office files (DOC, XLS, PPT) have their
text recovered on a best-effort basis. The original file is never modified.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# extension -> reader; a reader's own name maps to itself because the stored copy is called original.<reader>
READERS = {
    "csv": "csv", "tsv": "tsv", "tab": "tsv", "xlsx": "xlsx", "xlsm": "xlsx", "json": "json", "jsonl": "jsonl", "ndjson": "jsonl",
    "txt": "txt", "text": "txt", "md": "txt", "markdown": "txt", "rst": "txt", "log": "txt", "tex": "txt",
    "docx": "docx", "odt": "odt", "pptx": "pptx", "epub": "epub", "pdf": "pdf", "rtf": "rtf", "xml": "xml",
    "html": "html", "htm": "html", "xhtml": "html", "shtml": "html", "srt": "srt", "vtt": "vtt",
    "doc": "legacy", "xls": "legacy", "ppt": "legacy", "legacy": "legacy",
}
SUPPORTED_FORMATS = tuple(sorted(set(READERS.values())))
AUDIO_VIDEO = {
    "mp3", "wav", "m4a", "aac", "flac", "ogg", "oga", "opus", "wma", "aif", "aiff", "aifc", "amr", "mid", "midi", "ape", "alac",
    "mp4", "m4v", "mov", "avi", "mkv", "webm", "wmv", "flv", "mpg", "mpeg", "mpe", "3gp", "3g2", "m2ts", "mts", "vob", "ogv", "f4v", "asf",
}
# formats with no text to read: images (no OCR), programs, archives, fonts, databases
NOT_TEXT = {
    "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif", "ico", "psd", "raw", "cr2", "nef", "avif",
    "exe", "dll", "so", "dylib", "bin", "msi", "apk", "app", "dmg", "iso", "class", "jar", "pyc", "o", "obj",
    "zip", "rar", "7z", "gz", "tgz", "bz2", "xz", "tar", "ttf", "otf", "woff", "woff2", "sqlite", "db", "mdb", "accdb",
}
MEDIA_MESSAGE = "Audio and video files are not accepted. Upload a transcript of the recording instead (for example .txt, .docx, .srt or .vtt)."


class LoadError(ValueError):
    pass


@dataclass
class LoadedFile:
    frame: pd.DataFrame
    format: str
    encoding: str | None = None
    warnings: list[str] = field(default_factory=list)
    structured: bool = True


def _utf16(raw: bytes) -> str | None:
    """UTF-16 text, with or without a byte-order mark (Windows "Unicode" text files)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    sample = raw[:4096]
    if len(sample) >= 4:
        odd, even = sample[1::2].count(0), sample[0::2].count(0)
        if odd > 0.4 * (len(sample) // 2) and even < 0.05 * len(sample):
            return raw.decode("utf-16-le", errors="replace")
        if even > 0.4 * (len(sample) // 2) and odd < 0.05 * len(sample):
            return raw.decode("utf-16-be", errors="replace")
    return None


def _text_sample(head: bytes) -> str | None:
    """The start of the file as text, or None when it is binary."""
    if not head:
        return ""
    wide = _utf16(head)
    if wide is not None:
        return wide
    if b"\x00" in head:
        return None
    text = head.decode("utf-8", errors="replace")
    if text.count("\ufffd") > 3:
        text = head.decode("cp1252", errors="replace")
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r\f\v\x1a")
    return None if controls > 0.02 * len(text) else text


def _sniff(head: bytes) -> str:
    """What the first bytes say the file is: media, pdf, zip, ole, binary, rtf, vtt, html, xml or text."""
    if head.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")):
        pass  # byte-order marks: text, decided below
    elif (head.startswith((b"ID3", b"fLaC", b"OggS", b"#!AMR", b"FLV\x01", b".snd", b"MThd", b"MAC "))
          or head[:4] in (b"\x1aE\xdf\xa3", b"0&\xb2u", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")
          or (head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI ", b"CDXA", b"RMID"))
          or (head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"))
          or (len(head) > 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06 in (0x02, 0x04, 0x06, 0x00) and head[1] != 0xFF)):
        return "media"
    if head[4:8] == b"ftyp":
        return "binary" if head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"avif") else "media"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"
    if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*", b"\x7fELF", b"\xca\xfe\xba\xbe",
                        b"\x1f\x8b", b"Rar!", b"7z\xbc\xaf", b"BZh", b"\xfd7zXZ", b"wOFF", b"wOF2", b"SQLite format")):
        return "binary"
    text = _text_sample(head)
    if text is None:
        return "binary"
    start = text.lstrip("\ufeff \t\r\n")[:2000].lower()
    if start.startswith("{\\rtf"):
        return "rtf"
    if start.startswith("webvtt"):
        return "vtt"
    if start.startswith(("<!doctype html", "<html")) or re.search(r"<(html|body|p|div|article)[\s>]", start):
        return "html"
    if start.startswith("<?xml") or re.match(r"<[a-z_][\w:.-]*[\s>/]", start):
        return "xml"
    return "text"


def _zip_reader(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except zipfile.BadZipFile:
        return None
    if "word/document.xml" in names:
        return "docx"
    if any(n.startswith("ppt/slides/") for n in names):
        return "pptx"
    if any(n.startswith("xl/") for n in names):
        return "xlsx"
    if "META-INF/container.xml" in names:
        return "epub"
    if "content.xml" in names:
        return "odt"
    return None


def detect_format(path: str | Path, name: str | None = None) -> str:
    """The reader for a file, from its extension and its content. Raises LoadError for audio, video and files
    without text. ``name`` is the uploaded file name when ``path`` is a temporary or stored copy."""
    path = Path(path)
    ext = Path(name or path.name).suffix.lower().lstrip(".")
    with open(path, "rb") as fh:
        head = fh.read(65536)
    kind = _sniff(head)
    if ext in AUDIO_VIDEO or kind == "media":
        raise LoadError(MEDIA_MESSAGE)
    if ext in NOT_TEXT:
        raise LoadError(f"This is not a text file (.{ext}), so there is nothing to analyse. Upload documents, spreadsheets, "
                        "web pages, transcripts or any other file that contains text.")
    by_content = {"pdf": "pdf", "ole": "legacy", "rtf": "rtf", "vtt": "vtt", "html": "html", "xml": "xml", "text": "txt"}
    if kind == "zip":
        by_content["zip"] = _zip_reader(path)
    reader = READERS.get(ext)
    if reader is not None:
        # a file whose content plainly belongs to another reader (an .xls that is really a web page, a .txt that is
        # really a PDF) is read by its content; binary content under a text extension has nothing to read
        if reader in ("txt", "csv", "tsv", "json", "jsonl", "html", "xml", "srt", "vtt", "rtf") and kind in ("binary", "zip", "ole", "pdf"):
            if by_content.get(kind):
                return by_content[kind]
            raise LoadError("This file does not contain readable text.")
        if reader == "legacy" and kind != "ole":
            return by_content.get(kind) or reader
        if reader in ("docx", "xlsx", "pptx", "odt", "epub") and kind != "zip":
            return by_content.get(kind) or reader
        if reader == "pdf" and kind != "pdf":
            return by_content.get(kind) or reader
        return reader
    found = by_content.get(kind)
    if found is None:
        raise LoadError(f"No readable text was found in this file{f' (.{ext})' if ext else ''}. Upload documents, spreadsheets, "
                        "web pages, transcripts or any other file that contains text.")
    return found


def _read_text(path: Path) -> tuple[str, str, list[str]]:
    raw = path.read_bytes()
    warnings: list[str] = []
    wide = _utf16(raw)
    if wide is not None:
        return wide.lstrip("\ufeff"), "utf-16", warnings
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc), "utf-8", warnings
        except UnicodeDecodeError:
            continue
    try:
        text = raw.decode("cp1252")
        warnings.append("File is not valid UTF-8; decoded as Windows-1252. Check special characters.")
        return text, "cp1252", warnings
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        warnings.append("File is not valid UTF-8; decoded as Latin-1. Check special characters.")
        return text, "latin-1", warnings


def _paragraph_frame(paragraphs: list[tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for i, (text, extra) in enumerate(paragraphs, start=1):
        rows.append({"record_id": f"P{i:05d}", "text": text, **extra})
    if not rows:
        return pd.DataFrame(columns=["record_id", "text"])
    return pd.DataFrame(rows)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    out = []
    for p in parts:
        p = re.sub(r"[ \t]*\n[ \t]*", " ", p).strip()
        if p:
            out.append(p)
    return out


SUBTITLE_TIMING = re.compile(r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}")
SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")


def _subtitle_passages(text: str) -> list[tuple[str, dict]]:
    """Captions (SRT, WebVTT) joined into passages: a caption is often half a sentence, so consecutive captions
    are joined until a sentence ends after at least 25 words (or 80 words in all)."""
    cues, cur, start, skipping = [], [], None, False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") + [""]:
        s = line.strip()
        if not s:
            if cur:
                cues.append((start, " ".join(cur)))
            cur, skipping = [], False
            continue
        if skipping:
            continue
        if SUBTITLE_TIMING.match(s):
            start = s.split("-->")[0].strip()
            continue
        if not cur and (s.isdigit() or s.upper().startswith("WEBVTT")):
            continue
        if not cur and s.split(" ")[0].upper() in ("NOTE", "STYLE", "REGION"):
            skipping = True
            continue
        clean = re.sub(r"<[^>]+>", "", s).strip()
        if clean:
            cur.append(clean)
    passages, buf, buf_start = [], [], None
    for cue_start, t in cues:
        buf_start = buf_start or cue_start
        buf.append(t)
        n = len(" ".join(buf).split())
        if (n >= 25 and SENTENCE_END.search(t)) or n >= 80:
            passages.append((" ".join(buf), {"cue_start": buf_start}))
            buf, buf_start = [], None
    if buf:
        passages.append((" ".join(buf), {"cue_start": buf_start}))
    return passages


RTF_SKIP = {
    "fonttbl", "colortbl", "datastore", "themedata", "stylesheet", "info", "pict", "object", "header", "footer", "headerl",
    "headerr", "headerf", "footerl", "footerr", "footerf", "listtable", "listoverridetable", "rsidtbl", "generator",
    "xmlnstbl", "mmathpr", "latentstyles", "pgdsctbl", "revtbl", "filetbl", "fldinst", "shppict", "nonshppict", "bkmkstart",
    "bkmkend", "objdata", "blipuid", "falt", "panose", "operator", "author", "title", "company", "creatim", "revtim",
}
RTF_CHARS = {"emdash": "—", "endash": "–", "bullet": "•", "lquote": "‘", "rquote": "’", "ldblquote": "“", "rdblquote": "”"}
RTF_TOKEN = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})?[ ]?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I | re.S)


def _rtf_text(rtf: str) -> str:
    """Plain text of an RTF document: control words and hidden destinations (fonts, styles, pictures) removed."""
    stack, ignorable, ucskip, curskip, out = [], False, 1, 0, []
    for word, arg, hexcode, char, brace, tchar in RTF_TOKEN.findall(rtf):
        if brace:
            curskip = 0
            if brace == "{":
                stack.append((ucskip, ignorable))
            elif stack:
                ucskip, ignorable = stack.pop()
        elif char:
            curskip = 0
            if char == "*":
                ignorable = True
            elif not ignorable and char in "{}\\~":
                out.append("\xa0" if char == "~" else char)
        elif word:
            curskip = 0
            w = word.lower()
            if w in RTF_SKIP:
                ignorable = True
            elif ignorable:
                continue
            elif w in ("par", "sect", "page"):
                out.append("\n\n")
            elif w in ("line", "row"):
                out.append("\n")
            elif w in ("tab", "cell"):
                out.append("\t")
            elif w == "uc":
                ucskip = int(arg or 1)
            elif w == "u" and arg:
                code = int(arg)
                out.append(chr(code + 65536 if code < 0 else code))
                curskip = ucskip
            elif w in RTF_CHARS:
                out.append(RTF_CHARS[w])
        elif hexcode:
            if curskip:
                curskip -= 1
            elif not ignorable:
                out.append(bytes([int(hexcode, 16)]).decode("cp1252", errors="replace"))
        elif tchar:
            if curskip:
                curskip -= 1
            elif not ignorable:
                out.append(tchar)
    return "".join(out)


def _xml_root(data: bytes):
    import xml.etree.ElementTree as ET

    return ET.fromstring(data)


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _odt_paragraphs(path: Path) -> list[tuple[str, dict]]:
    with zipfile.ZipFile(path) as z:
        root = _xml_root(z.read("content.xml"))
    out = []
    for el in root.iter():
        if _local(el.tag) in ("p", "h"):
            t = re.sub(r"\s+", " ", "".join(el.itertext())).strip()
            if t:
                out.append((t, {"paragraph": len(out) + 1}))
    return out


def _pptx_slides(path: Path) -> list[tuple[str, dict]]:
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
        out = []
        for n in slides:
            lines = []
            for el in _xml_root(z.read(n)).iter():
                if _local(el.tag) == "p":
                    t = "".join(x.text or "" for x in el.iter() if _local(x.tag) == "t").strip()
                    if t:
                        lines.append(t)
            if lines:
                out.append(("\n".join(lines), {"slide": int(re.findall(r"\d+", n)[-1])}))
    return out


def _html_blocks(markup: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for t in soup(("script", "style", "nav", "header", "footer")):
        t.decompose()
    blocks = [b for b in soup.find_all(("p", "li", "blockquote", "h1", "h2", "h3", "h4", "dd", "td")) if not b.find(("p", "li", "blockquote", "td"))]
    texts = [re.sub(r"\s+", " ", b.get_text(" ", strip=True)) for b in blocks]
    texts = [t for t in texts if t]
    return texts or [t for t in (x.strip() for x in soup.get_text("\n").split("\n")) if t]


def _epub_paragraphs(path: Path) -> list[tuple[str, dict]]:
    from posixpath import dirname, join, normpath

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        order = []
        try:
            opf = next(el.get("full-path") for el in _xml_root(z.read("META-INF/container.xml")).iter() if _local(el.tag) == "rootfile")
            root = _xml_root(z.read(opf))
            items = {el.get("id"): el.get("href") for el in root.iter() if _local(el.tag) == "item"}
            order = [normpath(join(dirname(opf), items[el.get("idref")])) for el in root.iter()
                     if _local(el.tag) == "itemref" and el.get("idref") in items]
        except (KeyError, StopIteration, SyntaxError):
            pass
        order = [n for n in order if n in names] or sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        out = []
        for chapter, n in enumerate(order, start=1):
            for t in _html_blocks(z.read(n).decode("utf-8", errors="replace")):
                out.append((t, {"chapter": chapter}))
    return out


def _xml_paragraphs(text: str) -> list[tuple[str, dict]]:
    """Text of an XML document: each element holding text of its own (mixed content keeps its inline markup's text)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "xml")
    out, taken = [], set()
    for el in soup.find_all(True):
        if any(id(a) in taken for a in el.parents):
            continue
        if any(s.strip() for s in el.find_all(string=True, recursive=False)):
            taken.add(id(el))
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if t:
                out.append((t, {"element": el.name}))
    return out


def _legacy_paragraphs(raw: bytes) -> list[tuple[str, dict]]:
    """Text recovered from an older binary Office file (Word 97–2003 .doc, Excel .xls, PowerPoint .ppt).

    These formats store their text as runs of Windows-1252 or UTF-16 characters between binary structures.
    Runs are split into paragraphs and kept when they read as prose: at least four words, mostly letters.
    """
    runs = [m.group().decode("cp1252", errors="replace") for m in re.finditer(rb"[\x09\x0a\x0d\x20-\x7e\x91-\x97\xa0-\xff]{20,}", raw)]
    runs += [m.group().decode("utf-16-le", errors="replace") for m in re.finditer(rb"(?:[\x09\x0a\x0d\x20-\x7e\xa0-\xff]\x00|[\x18-\x22\x13\x14]\x20){12,}", raw)]
    out, seen = [], set()
    for run in runs:
        for part in re.split(r"[\r\n\x0b\x0c\x07]+", run):
            t = re.sub(r"\s+", " ", part).strip()
            letters = sum(ch.isalpha() for ch in t)
            if len(t.split()) >= 4 and letters >= 0.6 * len(t.replace(" ", "")) and t not in seen:
                seen.add(t)
                out.append((t, {"paragraph": len(out) + 1}))
    return out


def _reason(exc: Exception, path: Path) -> str:
    """Parser error text without the local file path (it points into the workspace or a temp folder)."""
    msg = str(exc) or type(exc).__name__
    for p in {str(path), str(path.resolve()), path.as_posix()}:
        msg = msg.replace(p, path.name)
    return msg


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Cells as strings, missing cells as None, whatever the source format."""
    from thematicml.data.validation import clean_cell

    for col in df.columns:
        df[col] = pd.Series([clean_cell(v) for v in df[col]], index=df.index, dtype=object)
    return df


def load_file(path: str | Path, name: str | None = None) -> LoadedFile:
    """Parse one file. ``name`` is the uploaded file name (the stored copy is called ``original.<ext>``);
    documents and pages without a title of their own take their pseudonymous respondent IDs from it."""
    path = Path(path)
    stem = Path(name).stem if name else path.stem
    if not path.exists():
        raise LoadError(f"File not found: {path}")
    fmt = detect_format(path, None if path.stem == "original" else name)

    if fmt == "csv":
        text, enc, warnings = _read_text(path)
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
        except pd.errors.ParserError as exc:
            raise LoadError(f"Could not parse CSV: {exc}") from exc
        return LoadedFile(_tidy(df), fmt, enc, warnings)

    if fmt == "tsv":
        text, enc, warnings = _read_text(path)
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False, na_values=[""])
        except pd.errors.ParserError as exc:
            raise LoadError(f"Could not parse the tab-separated file: {exc}") from exc
        return LoadedFile(_tidy(df), fmt, enc, warnings)

    if fmt == "xlsx":
        try:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
        except Exception as exc:  # openpyxl raises many types
            raise LoadError(f"Could not read workbook: {_reason(exc, path)}") from exc
        return LoadedFile(_tidy(df), fmt, None, [])

    if fmt in ("json", "jsonl"):
        text, enc, warnings = _read_text(path)
        try:
            if fmt == "jsonl":
                data = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LoadError(f"Invalid JSON: {exc}") from exc
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise LoadError("JSON object must contain exactly one list of records.")
            data = lists[0]
        if not isinstance(data, list) or (data and not all(isinstance(r, dict) for r in data)):
            raise LoadError("JSON must be a list of objects (records).")
        return LoadedFile(_tidy(pd.json_normalize(data)), fmt, enc, warnings)

    if fmt == "txt":
        text, enc, warnings = _read_text(path)
        paras = [(p, {"paragraph": i}) for i, p in enumerate(_split_paragraphs(text), start=1)]
        return LoadedFile(_paragraph_frame(paras), fmt, enc, warnings, structured=False)

    if fmt in ("srt", "vtt"):
        text, enc, warnings = _read_text(path)
        return LoadedFile(_paragraph_frame(_subtitle_passages(text)), fmt, enc, warnings, structured=False)

    if fmt == "rtf":
        text, enc, warnings = _read_text(path)
        paras = [(p, {"paragraph": i}) for i, p in enumerate(_split_paragraphs(_rtf_text(text)), start=1)]
        return LoadedFile(_paragraph_frame(paras), fmt, enc, warnings, structured=False)

    if fmt == "xml":
        text, enc, warnings = _read_text(path)
        return LoadedFile(_paragraph_frame(_xml_paragraphs(text)), fmt, enc, warnings, structured=False)

    if fmt in ("odt", "pptx", "epub"):
        reader = {"odt": _odt_paragraphs, "pptx": _pptx_slides, "epub": _epub_paragraphs}[fmt]
        try:
            paras = reader(path)
        except Exception as exc:  # damaged archives and XML raise many types
            raise LoadError(f"Could not read the {fmt.upper()} file: {_reason(exc, path)}") from exc
        return LoadedFile(_paragraph_frame(paras), fmt, None, [], structured=False)

    if fmt == "legacy":
        paras = _legacy_paragraphs(path.read_bytes())
        ext = Path(name or path.name).suffix.lower().lstrip(".")
        kind = f".{ext} " if ext in ("doc", "xls", "ppt") else ""
        warnings = [f"Text was recovered from an older Office file {kind}on a best-effort basis; formatting and tables are lost. "
                    "Saving it as .docx, .xlsx or .pptx gives the most faithful import."]
        return LoadedFile(_paragraph_frame(paras), fmt, None, warnings, structured=False)

    if fmt == "docx":
        import docx

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise LoadError(f"Could not read DOCX: {_reason(exc, path)}") from exc
        paras = []
        for i, p in enumerate(document.paragraphs, start=1):
            t = p.text.strip()
            if t:
                paras.append((t, {"paragraph": i}))
        warnings = []
        if document.tables:
            warnings.append(f"{len(document.tables)} table(s) in the document were not imported; only paragraphs are read.")
        return LoadedFile(_paragraph_frame(paras), fmt, None, warnings, structured=False)

    if fmt in ("html", "htm"):
        from thematicml.data.html import extract_html

        rows, warnings, info = extract_html(path.read_bytes(), stem)
        cols = ["respondent_id", "text", "source_kind", "section", "block", "published", "page_title"]
        df = pd.DataFrame(rows, columns=cols)
        return LoadedFile(_tidy(df), fmt, info["encoding"], warnings, structured=False)

    # pdf
    from thematicml.data.pdf import extract_pdf

    try:
        rows, warnings, _info = extract_pdf(path, stem)
    except Exception as exc:  # pypdf raises many types for damaged files
        raise LoadError(f"Could not read PDF: {_reason(exc, path)}") from exc
    cols = ["respondent_id", "text", "source_kind", "section", "source_page", "paragraph", "page_title"]
    return LoadedFile(pd.DataFrame(rows, columns=cols), fmt, None, warnings, structured=False)


def suggest_mapping(df: pd.DataFrame) -> dict:
    """Heuristically suggest which columns hold text, IDs, timestamps and groups."""
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}
    sample = df.head(200)

    def avg_len(c):
        s = sample[c].dropna().astype(str)
        return s.str.len().mean() if len(s) else 0

    text_col = None
    named = [c for c in cols if any(k in lower[c] for k in ("text", "response", "comment", "answer", "feedback", "transcript", "content"))]
    if named:
        text_col = max(named, key=avg_len)
    elif cols:
        text_col = max(cols, key=avg_len)

    def first(keys, exclude=()):
        for c in cols:
            if c in exclude:
                continue
            if any(k in lower[c] for k in keys):
                return c
        return None

    id_col = first(("respondent", "record_id", "participant", "_id", "id"), exclude=(text_col,))
    time_col = first(("date", "time", "created", "timestamp", "wave", "month", "published"), exclude=(text_col, id_col))
    if time_col is not None:
        try:  # utc=True also accepts a mix of timezone-aware and naive values
            parsed = pd.to_datetime(sample[time_col], errors="coerce", format="mixed", utc=True)
        except (ValueError, TypeError):
            parsed = pd.Series([pd.NaT] * len(sample))
        if parsed.notna().mean() < 0.5:
            time_col = None
    group_col = first(("region", "group", "segment", "country", "department", "team", "category", "source_kind", "source"), exclude=(text_col, id_col, time_col))
    return {"text_col": text_col, "id_col": id_col, "time_col": time_col, "group_col": group_col, "label_col": None}
