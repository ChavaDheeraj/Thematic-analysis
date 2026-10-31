"""Load qualitative datasets from common file formats into a DataFrame.

Tabular formats (CSV, XLSX, JSON, JSONL) keep their columns. TXT and DOCX
documents become one record per paragraph with a generated ``record_id`` plus
a ``paragraph`` column. PDFs have their paragraphs rebuilt from the text layer
(see ``thematicml.data.pdf``) and, like a web article, one pseudonymous
respondent per document, with ``source_page`` and ``section`` provenance.
Saved web pages (HTML) become one record per article paragraph and per reader
comment, with pseudonymous respondent IDs (see ``thematicml.data.html``).
The original file is never modified.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SUPPORTED_FORMATS = ("csv", "xlsx", "txt", "docx", "pdf", "json", "jsonl", "html", "htm")


class LoadError(ValueError):
    pass


@dataclass
class LoadedFile:
    frame: pd.DataFrame
    format: str
    encoding: str | None = None
    warnings: list[str] = field(default_factory=list)
    structured: bool = True


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "xls":
        raise LoadError("Legacy .xls is not supported; save the workbook as .xlsx.")
    if ext not in SUPPORTED_FORMATS:
        raise LoadError(f"Unsupported file format '.{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}")
    return ext


def _read_text(path: Path) -> tuple[str, str, list[str]]:
    raw = path.read_bytes()
    warnings: list[str] = []
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
    fmt = detect_format(path)

    if fmt == "csv":
        text, enc, warnings = _read_text(path)
        from io import StringIO

        try:
            df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, na_values=[""])
        except pd.errors.ParserError as exc:
            raise LoadError(f"Could not parse CSV: {exc}") from exc
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
