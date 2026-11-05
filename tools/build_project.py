"""Build or update a project from a folder of files with a saved configuration (reproducible runs).

    python tools/build_project.py --project "Solo female travel (t.data, tuned)" --config configs/solo-female-travel.json \
        --files "t.data/*.html" "t.data/New folder/*" --label "t.data: tuned configuration"

The configuration file holds three sections::

    {"mapping": {...ColumnMapping...}, "analysis": {...AnalysisConfig...}, "gnn": [{...GNNConfig...}, ...]}

Files already in the project (same bytes) and byte-identical copies among the inputs are skipped,
so re-running the command only adds new files. Optional mapping columns a file does not have
(for example no timestamp in a PDF) are left unmapped for that file. The analysis runs on every
dataset in the project that validated successfully.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

from thematicml import AnalysisConfig, Project
from thematicml.config import ColumnMapping, GNNConfig
from thematicml.storage.datasets import apply_mapping, parsed_frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="files or glob patterns")
    ap.add_argument("--label", default=None, help="run label")
    ap.add_argument("--description", default="")
    ap.add_argument("--workspace", default=None, help="defaults to THEMATICML_HOME or ~/.thematicml")
    ap.add_argument("--no-run", action="store_true", help="import and validate only")
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mapping = ColumnMapping(**spec["mapping"])
    cfg = AnalysisConfig(**spec.get("analysis", {}))
    project = Project(args.project, args.workspace, args.description)
    store = project.store

    known = {}
    for ds in store.list_datasets(project.id):
        try:
            known[_sha256(store.resolve(ds["original_path"]))] = ds["filename"]
        except OSError:
            continue
    paths = []
    for pat in args.files:
        paths += [Path(p) for p in sorted(glob.glob(pat)) if Path(p).is_file()]
    for path in paths:
        digest = _sha256(path)
        if digest in known:
            print(f"skip  {path.name} (same file as {known[digest]})")
            continue
        try:
            did = project.load(path, allow_duplicate=True)  # identical files are skipped above; same-named files in different folders are different pages
        except ValueError as exc:
            print(f"FAIL  {path.name}: {exc}")
            continue
        known[digest] = path.name
        cols = parsed_frame(store, did).columns
        absent = {k: None for k in ("id_col", "time_col", "group_col", "label_col") if getattr(mapping, k) not in (None, *cols)}
        m = mapping.model_copy(update={**absent, "metadata_cols": [c for c in mapping.metadata_cols if c in cols]})
        ds = apply_mapping(store, did, m)
        q = ds["quality"] or {}
        print(f"added {path.name}: {q.get('valid_records', 0)}/{q.get('total_records', 0)} valid records ({ds['status']})")

    if args.no_run:
        return 0
    ids = [d["id"] for d in store.list_datasets(project.id) if d["status"] in ("ready", "analysed")]
    result = project.analyze(cfg, dataset_ids=ids, label=args.label)
    result.summary()
    if result.status != "completed":
        return 1
    for g in spec.get("gnn", []):
        exp = result.train_gnn(GNNConfig(**g))
        m = exp.get("metrics") or {}
        print(f"GNN {g.get('task', 'link_prediction')}: {exp['status']}" + (f" - {exp.get('error')}" if exp.get("error") else
              f" - {m.get('verdict', '')}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
