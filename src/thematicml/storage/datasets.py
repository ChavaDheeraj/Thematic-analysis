"""Import uploaded files and apply column mappings."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import joblib
import pandas as pd

from thematicml.config import ColumnMapping
from thematicml.data import load_file, normalise_records, suggest_mapping
from thematicml.data.loaders import LoadError, detect_format
from thematicml.storage.database import Store, new_id, now


class DuplicateFileError(LoadError):
    """The project already holds this file (same content, or same name)."""


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate(store: Store, project_id: str, filename: str, digest: str) -> tuple[dict, str] | None:
    """An upload in the project with the same content, or else the same name, and which of the two matched.
    Failed uploads do not count: they hold no records, so the file can be uploaded again."""
    rows = store.query("SELECT id, filename, sha256, original_path FROM datasets WHERE project_id = ? AND status != 'failed' "
                       "ORDER BY uploaded_at", (project_id,))
    for ds in rows:
        if ds["sha256"] is None:  # uploaded before fingerprints were stored
            original = store.resolve(ds["original_path"])
            if not original.exists():
                continue
            ds["sha256"] = file_digest(original)
            store.update("datasets", {"id": ds["id"]}, {"sha256": ds["sha256"]})
        if ds["sha256"] == digest:
            return ds, "content"
    for ds in rows:
        if ds["filename"].casefold() == filename.casefold():
            return ds, "name"
    return None


def import_file(store: Store, project_id: str, path: str | Path, filename: str | None = None, allow_duplicate: bool = False) -> dict:
    """Copy the original file into the workspace, parse it and register the dataset.

    The stored original is never modified. Parsing failures are recorded on the
    dataset (status 'failed') rather than discarding the upload. A file the project
    already holds (same name or same content) is refused with DuplicateFileError unless
    ``allow_duplicate`` is set, for example to load the same survey export as two waves.
    """
    path = Path(path)
    # display name only: drop any directory parts a browser or client sent
    filename = Path((filename or path.name).replace("\\", "/")).name or path.name
    digest = file_digest(path)
    if not allow_duplicate and (dup := find_duplicate(store, project_id, filename, digest)):
        existing, how = dup
        if how == "name":
            raise DuplicateFileError("A file with this name already exists in this project. Rename the file, or delete the "
                                     "existing one under Database → Files & records, to upload this version.")
        if existing["filename"].casefold() == filename.casefold():
            raise DuplicateFileError("This file already exists in this project. It was not added again.")
        raise DuplicateFileError(f"This file already exists in this project (uploaded as “{existing['filename']}”). It was not added again.")
    fmt = detect_format(path, filename)
    did = new_id("ds")
    ddir = store.dataset_dir(project_id, did)
    original = ddir / f"original.{fmt}"
    shutil.copyfile(path, original)
    row = {
        "id": did, "project_id": project_id, "filename": filename, "format": fmt,
        "size_bytes": original.stat().st_size, "uploaded_at": now(), "original_path": store.relative(original),
        "status": "uploaded", "sha256": digest,
    }
    try:
        loaded = load_file(original, filename)
    except Exception as exc:
        row.update({"status": "failed", "error": str(exc)})
        store.insert("datasets", row)
        store.log(project_id, "upload_failed", {"dataset_id": did, "filename": filename, "error": str(exc)})
        return store.get_dataset(did)
    joblib.dump(loaded.frame, ddir / "parsed.joblib")
    suggestion = suggest_mapping(loaded.frame)
    row.update({
        "encoding": loaded.encoding, "structured": int(loaded.structured), "columns": [str(c) for c in loaded.frame.columns],
        "n_rows": int(len(loaded.frame)), "warnings": loaded.warnings, "mapping": {"suggested": suggestion},
    })
    if len(loaded.frame) == 0:
        row.update({"status": "failed", "error": "No readable text was found in this file."})
    store.insert("datasets", row)
    store.log(project_id, "dataset_uploaded", {"dataset_id": did, "filename": filename, "format": fmt, "rows": int(len(loaded.frame))})
    return store.get_dataset(did)


def parsed_frame(store: Store, dataset_id: str) -> pd.DataFrame:
    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise KeyError(dataset_id)
    original = store.resolve(ds["original_path"])
    p = original.parent / "parsed.joblib"
    if p.exists():
        return joblib.load(p)
    return load_file(original, ds["filename"]).frame


def preview(store: Store, dataset_id: str, rows: int = 20) -> dict:
    df = parsed_frame(store, dataset_id)
    from thematicml.data.validation import clean_cell

    head = df.head(rows).copy()
    head = pd.DataFrame({c: [clean_cell(v) for v in head[c]] for c in head.columns}, dtype=object)
    profile = []
    for c in df.columns:
        s = df[c]
        nonnull = s.dropna().astype(str)
        profile.append({
            "column": str(c),
            "non_empty": int((nonnull.str.strip() != "").sum()),
            "missing": int(len(s) - (nonnull.str.strip() != "").sum()),
            "unique": int(nonnull.nunique()),
            "avg_length": round(float(nonnull.str.len().mean()), 1) if len(nonnull) else 0.0,
        })
    return {"columns": [str(c) for c in df.columns], "rows": head.to_dict(orient="records"), "total_rows": int(len(df)), "profile": profile}


def apply_mapping(store: Store, dataset_id: str, mapping: ColumnMapping) -> dict:
    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise KeyError(dataset_id)
    df = parsed_frame(store, dataset_id)
    records, report = normalise_records(df, mapping)
    store.replace_records(dataset_id, records)
    current = ds.get("mapping") or {}
    current["applied"] = mapping.model_dump()
    store.update("datasets", {"id": dataset_id}, {
        "mapping": current,
        "quality": report.to_dict(),
        "status": "ready" if report.valid_records > 0 else "failed",
        "error": None if report.valid_records > 0 else "No valid records after validation.",
    })
    store.log(ds["project_id"], "dataset_validated", {"dataset_id": dataset_id, "valid": report.valid_records, "total": report.total_records})
    return store.get_dataset(dataset_id)
