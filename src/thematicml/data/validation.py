"""Validate a mapped dataset and produce normalised records plus a quality report."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from thematicml.config import ColumnMapping
from thematicml.data.language import is_non_english

_WS = re.compile(r"\s+")


def clean_cell(v) -> str | None:
    """Return a cell as a string, or None when missing (handles None, NaN and pd.NA)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def clean_stripped(v) -> str | None:
    s = clean_cell(v)
    return None if s is None or not s.strip() else s.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


@dataclass
class QualityReport:
    total_records: int
    valid_records: int
    missing_text: int
    duplicate_records: int
    excluded_records: int
    invalid_timestamps: int
    missing_ids: int
    duplicate_policy: str
    non_english_records: int = 0
    language_policy: str = "keep"
    warnings: list[str] = field(default_factory=list)

    def pct(self, n: int) -> float:
        return round(100.0 * n / self.total_records, 2) if self.total_records else 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["percentages"] = {
            "valid": {"numerator": self.valid_records, "denominator": self.total_records, "value": self.pct(self.valid_records),
                      "definition": "valid records / all parsed records"},
            "missing_text": {"numerator": self.missing_text, "denominator": self.total_records, "value": self.pct(self.missing_text),
                             "definition": "records with empty or missing text / all parsed records"},
            "duplicates": {"numerator": self.duplicate_records, "denominator": self.total_records, "value": self.pct(self.duplicate_records),
                           "definition": "repeat occurrences of an identical (record ID, non-empty text) pair / all parsed records; empty rows count as missing, not duplicate"},
            "excluded": {"numerator": self.excluded_records, "denominator": self.total_records, "value": self.pct(self.excluded_records),
                         "definition": "records excluded from analysis / all parsed records"},
            "invalid_metadata": {"numerator": self.invalid_timestamps, "denominator": self.total_records, "value": self.pct(self.invalid_timestamps),
                                 "definition": "records whose timestamp could not be parsed / all parsed records"},
            "non_english": {"numerator": self.non_english_records, "denominator": self.total_records, "value": self.pct(self.non_english_records),
                            "definition": "records clearly in a language other than English (function-word profile) / all parsed records"},
        }
        return d


def normalise_records(df: pd.DataFrame, mapping: ColumnMapping) -> tuple[pd.DataFrame, QualityReport]:
    """Return one normalised row per source row, with validity flags.

    Output columns: row_index, record_key, text, timestamp, group_value, label,
    metadata (dict), word_count, is_valid, is_duplicate, exclusion_reason, invalid_timestamp.
    The original text is kept exactly as uploaded.
    """
    if mapping.text_col not in df.columns:
        raise ValueError(f"Text column '{mapping.text_col}' not found. Columns: {list(df.columns)}")
    for col in (mapping.id_col, mapping.time_col, mapping.group_col, mapping.label_col, *mapping.metadata_cols):
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    n = len(df)
    text = pd.Series([clean_cell(v) for v in df[mapping.text_col]], index=df.index, dtype=object)
    missing = text.map(lambda v: v is None or not v.strip())

    if mapping.id_col:
        keys = pd.Series([clean_stripped(v) for v in df[mapping.id_col]], index=df.index, dtype=object)
    else:
        keys = pd.Series([None] * n, index=df.index)
    missing_ids = int(keys.isna().sum()) if mapping.id_col else 0
    keys = pd.Series([k if k is not None else f"row-{i + 1}" for i, k in enumerate(keys)], index=df.index)

    invalid_ts = 0
    if mapping.time_col:
        raw_ts = df[mapping.time_col]
        ts = pd.to_datetime(raw_ts, errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
        invalid_ts = int((ts.isna() & raw_ts.notna() & (raw_ts.astype(str).str.strip() != "")).sum())
        ts_out = pd.Series([None if pd.isna(t) else t.isoformat() for t in ts], index=df.index, dtype=object)
    else:
        ts_out = pd.Series([None] * n, index=df.index)

    norm_text = text.map(lambda v: _WS.sub(" ", v.strip().lower()) if v else "")
    dup_key = keys.astype(str) + "\u0000" + norm_text
    is_dup = dup_key.duplicated(keep="first") & ~missing
    non_english = text.map(lambda v: bool(v) and is_non_english(v)) & ~missing

    def col_or_none(c):
        if not c:
            return pd.Series([None] * n, index=df.index)
        return pd.Series([clean_stripped(v) for v in df[c]], index=df.index, dtype=object)

    meta_cols = [c for c in df.columns if c in mapping.metadata_cols]
    records = pd.DataFrame(
        {
            "row_index": range(n),
            "record_key": keys.values,
            "text": text.map(lambda v: v if v is not None else "").values,
            "timestamp": ts_out.values,
            "group_value": col_or_none(mapping.group_col).values,
            "label": col_or_none(mapping.label_col).values,
        }
    )
    records["metadata"] = [
        dict(zip(meta_cols, vals)) for vals in zip(*[[clean_cell(v) for v in df[c]] for c in meta_cols])
    ] if meta_cols else [{} for _ in range(n)]
    records["word_count"] = records["text"].map(word_count)
    records["is_duplicate"] = is_dup.values
    reasons = []
    for m, d, ne in zip(missing.values, is_dup.values, non_english.values):
        if m:
            reasons.append("missing_text")
        elif d and mapping.duplicate_policy == "exclude":
            reasons.append("duplicate")
        elif ne and mapping.language_policy == "exclude_non_english":
            reasons.append("non_english")
        else:
            reasons.append(None)
    records["exclusion_reason"] = reasons
    records["is_valid"] = records["exclusion_reason"].isna()
    records["invalid_timestamp"] = False
    if mapping.time_col:
        raw = df[mapping.time_col].astype(str).str.strip()
        records["invalid_timestamp"] = (records["timestamp"].isna() & (raw != "") & df[mapping.time_col].notna()).values

    warnings = []
    if missing_ids:
        warnings.append(f"{missing_ids} record(s) have no ID; row numbers were used instead.")
    if invalid_ts:
        warnings.append(f"{invalid_ts} timestamp(s) could not be parsed; those records are kept but excluded from temporal analysis.")
    if mapping.time_col and records["timestamp"].notna().sum() == 0:
        warnings.append("No timestamps could be parsed; temporal analysis will be unavailable.")
    n_other = int(non_english.sum())
    if n_other and mapping.language_policy == "keep":
        warnings.append(f"{n_other} record(s) appear to be in a language other than English. Text features, stopwords and "
                        "sentiment are English-only; set the language policy to exclude them from analysis.")
    elif n_other:
        warnings.append(f"{n_other} record(s) in a language other than English were excluded from analysis.")

    report = QualityReport(
        total_records=n,
        valid_records=int(records["is_valid"].sum()),
        missing_text=int(missing.sum()),
        duplicate_records=int(is_dup.sum()),
        excluded_records=int((~records["is_valid"]).sum()),
        invalid_timestamps=invalid_ts,
        missing_ids=missing_ids,
        duplicate_policy=mapping.duplicate_policy,
        non_english_records=n_other,
        language_policy=mapping.language_policy,
        warnings=warnings,
    )
    return records, report
