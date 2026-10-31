"""Rule-based coding from a researcher-defined codebook."""
from __future__ import annotations

import re

import pandas as pd

from thematicml.coding.extraction import ASSIGN_COLUMNS, CODE_COLUMNS, _negated_before
from thematicml.config import CodebookRule


def keyword_regex(keyword: str) -> re.Pattern:
    kw = keyword.strip()
    prefix = kw.endswith("*")
    kw = kw.rstrip("*")
    parts = [re.escape(p) for p in kw.split()]
    body = r"\s+".join(parts)
    tail = r"[^\W_]*" if prefix else ""
    return re.compile(r"(?<![^\W_])" + body + tail + r"(?![^\W_])", re.IGNORECASE)


def apply_codebook(rules: list[CodebookRule], segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes, assigns = [], []
    texts = segments["text"].tolist()
    starts = segments["char_start"].tolist()
    for rule in rules:
        kws = [k for k in rule.keywords if k.strip()]
        if not kws:
            continue
        key = f"rule:{rule.label}"
        codes.append(
            {
                "code_key": key,
                "label": rule.label,
                "description": rule.description or f"Codebook rule matching: {', '.join(kws)}",
                "method": "rule",
                "feature": " | ".join(kws),
                "category": rule.category,
            }
        )
        patterns = [keyword_regex(k) for k in kws]
        for pos, text in enumerate(texts):
            best = None
            for p in patterns:
                m = p.search(text)
                if m and (best is None or m.start() < best.start()):
                    best = m
            if best:
                s, e = best.start(), best.end()
                assigns.append((key, pos, "rule", 1.0, text[s:e], starts[pos] + s, starts[pos] + e, _negated_before(text, s)))
    return pd.DataFrame(codes, columns=CODE_COLUMNS), pd.DataFrame(assigns, columns=ASSIGN_COLUMNS)
