"""Split record text into passages while keeping exact character offsets.

Every segment satisfies ``record_text[char_start:char_end] == segment_text``,
so quotations can always be traced back to the unchanged source.
"""
from __future__ import annotations

import re

import pandas as pd

from thematicml.data.validation import word_count

# Sentence boundary: terminal punctuation (optionally followed by quotes/brackets)
# then whitespace, or one or more line breaks.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_PARA_END = re.compile(r"\n\s*\n")


def _spans(text: str, mode: str) -> list[tuple[int, int]]:
    if mode == "record":
        boundaries: list[tuple[int, int]] = [(0, len(text))]
    else:
        pattern = _SENT_END if mode == "sentence" else _PARA_END
        boundaries = []
        start = 0
        for m in pattern.finditer(text):
            boundaries.append((start, m.start() + len(m.group(0).rstrip()) if mode == "sentence" else m.start()))
            start = m.end()
        boundaries.append((start, len(text)))
    out = []
    for s, e in boundaries:
        # trim whitespace while keeping offsets exact
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def segment_text(text: str, mode: str = "sentence", min_words: int = 3) -> list[tuple[int, int]]:
    """Return (start, end) spans. Spans shorter than ``min_words`` are merged
    into the following span (or the previous one at the end of a record)."""
    spans = _spans(text, mode)
    if mode == "record" or len(spans) <= 1:
        return spans
    merged: list[list[int]] = []
    pending: list[int] | None = None
    for s, e in spans:
        if pending is not None:
            s = pending[0]
            pending = None
        if word_count(text[s:e]) < min_words:
            pending = [s, e]
            continue
        merged.append([s, e])
    if pending is not None:
        if merged:
            merged[-1][1] = pending[1]
        else:
            merged.append(pending)
    return [(s, e) for s, e in merged]


def segment_records(records: pd.DataFrame, mode: str = "sentence", min_words: int = 3) -> pd.DataFrame:
    """Segment valid records. ``records`` needs columns record_id and text.

    Returns columns: record_id, seg_index, text, char_start, char_end.
    """
    rows = []
    for rec_id, text in zip(records["record_id"], records["text"]):
        for i, (s, e) in enumerate(segment_text(text, mode, min_words)):
            rows.append((rec_id, i, text[s:e], s, e))
    return pd.DataFrame(rows, columns=["record_id", "seg_index", "text", "char_start", "char_end"])
