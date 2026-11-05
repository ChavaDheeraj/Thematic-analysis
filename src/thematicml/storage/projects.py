"""High-level Python API: ``Project`` and ``AnalysisResult``.

>>> from thematicml import Project, AnalysisConfig
>>> project = Project("customer-research")
>>> project.load("responses.csv")
>>> cfg = AnalysisConfig(text_col="response", id_col="respondent_id", time_col="created_at")
>>> result = project.analyze(cfg)
>>> result.codes.to_csv("codes.csv", index=False)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thematicml.config import AnalysisConfig, ColumnMapping
from thematicml.storage.database import Store
from thematicml.storage.datasets import apply_mapping, import_file


class Metrics(dict):
    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self, indent=indent, default=str)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text


class AnalysisResult:
    def __init__(self, store: Store, run_id: str):
        self.store = store
        self.run_id = run_id

    @property
    def run(self) -> dict:
        return self.store.get_run(self.run_id)

    @property
    def status(self) -> str:
        return self.run["status"]

    def _require_completed(self):
        r = self.run
        if r["status"] != "completed":
            raise RuntimeError(f"Run {self.run_id} is {r['status']}" + (f" (failed at {r['failed_stage']}: {r['error']})" if r.get("error") else ""))

    @property
    def codes(self) -> pd.DataFrame:
        from thematicml.results import code_table

        self._require_completed()
        return pd.DataFrame(code_table(self.store, self.run_id))

    @property
    def themes(self) -> pd.DataFrame:
        from thematicml.reporting.exports import themes_frame
        from thematicml.reporting.summary import build_dashboard

        self._require_completed()
        return themes_frame(build_dashboard(self.store, self.run_id))

    @property
    def assignments(self) -> pd.DataFrame:
        from thematicml.reporting.exports import assignments_frame

        self._require_completed()
        return assignments_frame(self.store, self.run_id)

    @property
    def segments(self) -> pd.DataFrame:
        from thematicml.results import load_segments

        return load_segments(self.store, self.run_id)

    @property
    def metrics(self) -> Metrics:
        self._require_completed()
        return Metrics(self.run["metrics"] or {})

    def graph(self, **params) -> dict:
        from thematicml.results import graph_view

        return graph_view(self.store, self.run_id, **params)

    def temporal(self) -> dict:
        from thematicml.results import temporal_view

        return temporal_view(self.store, self.run_id)

    def dashboard(self) -> dict:
        from thematicml.reporting.summary import build_dashboard

        return build_dashboard(self.store, self.run_id)

    def export(self, fmt: str, out_dir: str | Path | None = None) -> Path:
        from thematicml.reporting.exports import export_run

        return export_run(self.store, self.run_id, fmt, Path(out_dir) if out_dir else None)

    def train_gnn(self, cfg=None) -> dict:
        """Train and evaluate an optional GraphSAGE experiment on this run's code graph (blocking)."""
        from thematicml.config import GNNConfig
        from thematicml.gnn.service import create_experiment, execute_experiment

        self._require_completed()
        exp = create_experiment(self.store, self.run_id, cfg or GNNConfig())
        return execute_experiment(self.store, exp["id"])

    def gnn_experiments(self) -> list[dict]:
        from thematicml.gnn.service import list_experiments

        return list_experiments(self.store, run_id=self.run_id)

    def summary(self, print_out: bool = True) -> str:
        r = self.run
        if r["status"] != "completed":
            text = f"Run {self.run_id}: {r['status']} (stage {r.get('failed_stage') or r.get('stage')}): {r.get('error')}"
        else:
            m = r["metrics"]
            d, c, g = m["dataset"], m["coding"], m["graph"]
            cl = m["clustering"]
            text = "\n".join([
                f"ThematicML run {self.run_id} ({r['label']}) — completed in {r['duration_s']} s",
                f"  records analysed: {d['analysed_records']}  respondents: {d['respondents']}  segments: {d['segments']}",
                f"  vocabulary: {m['features']['vocabulary_size']} terms  codes: {c['n_codes']}  assignments: {c['assignments']}",
                f"  coded segments: {m['statistics']['coverage']['coded_segments']['value']}%",
                f"  clusters: {cl.get('n_clusters', 'n/a')} (silhouette {cl.get('silhouette_cosine', 'n/a')})",
                f"  topics: {m['topics'].get('n_topics', 'n/a')}  graph: {g['nodes']} nodes / {g['edges']} edges / "
                f"{g['communities']['n_multi_node_communities']} communities",
                f"  candidate themes: {m['themes']['candidate_themes']}  temporal: {'yes' if m['temporal'].get('available') else 'no'}",
            ] + [f"  warning: {w}" for w in r.get("warnings") or []])
        if print_out:
            print(text)
        return text


class Project:
    def __init__(self, name: str, workspace: str | Path | None = None, description: str = ""):
        self.store = Store(workspace)
        existing = self.store.project_by_name(name)
        self.info = existing or self.store.create_project(name, description)
        self.id = self.info["id"]
        self.dataset_ids: list[str] = []

    def load(self, path: str | Path, allow_duplicate: bool = False) -> str:
        """Upload a file. A file the project already holds is refused unless ``allow_duplicate`` (two waves of one export)."""
        ds = import_file(self.store, self.id, path, allow_duplicate=allow_duplicate)
        if ds["status"] == "failed":
            raise ValueError(f"Could not load {path}: {ds['error']}")
        self.dataset_ids.append(ds["id"])
        return ds["id"]

    def map_columns(self, mapping: ColumnMapping | dict, dataset_id: str | None = None) -> dict:
        if isinstance(mapping, dict):
            mapping = ColumnMapping(**mapping)
        did = dataset_id or self._last()
        ds = apply_mapping(self.store, did, mapping)
        return ds["quality"]

    def validate(self, mapping: ColumnMapping | dict | None = None, dataset_id: str | None = None) -> dict:
        """Validate with an explicit mapping, or the heuristically suggested one."""
        did = dataset_id or self._last()
        if mapping is None:
            ds = self.store.get_dataset(did)
            applied = (ds.get("mapping") or {}).get("applied")
            sug = applied or (ds.get("mapping") or {}).get("suggested") or {}
            if not sug.get("text_col"):
                raise ValueError("Could not infer the text column; pass a mapping.")
            mapping = ColumnMapping(**{k: v for k, v in sug.items() if k in ColumnMapping.model_fields})
        return self.map_columns(mapping, did)

    def analyze(self, cfg: AnalysisConfig | None = None, dataset_ids: list[str] | None = None, label: str | None = None) -> AnalysisResult:
        from thematicml.pipeline import create_run, execute_run

        cfg = cfg or AnalysisConfig()
        ids = dataset_ids or self.dataset_ids or [d["id"] for d in self.store.list_datasets(self.id)]
        if not ids:
            raise ValueError("No datasets loaded.")
        for did in ids:
            ds = self.store.get_dataset(did)
            if cfg.text_col:
                applied = (ds.get("mapping") or {}).get("applied") or {}
                wanted = cfg.mapping().model_dump()
                if any(applied.get(k) != wanted[k] for k in ("text_col", "id_col", "time_col", "group_col", "label_col")):
                    apply_mapping(self.store, did, cfg.mapping())
            elif ds["status"] == "uploaded":
                self.validate(dataset_id=did)
        run = create_run(self.store, self.id, ids, cfg, label)
        execute_run(self.store, run["id"])
        return AnalysisResult(self.store, run["id"])

    def runs(self) -> list[dict]:
        return self.store.list_runs(self.id)

    def result(self, run_id: str) -> AnalysisResult:
        return AnalysisResult(self.store, run_id)

    def _last(self) -> str:
        if self.dataset_ids:
            return self.dataset_ids[-1]
        ds = self.store.list_datasets(self.id)
        if not ds:
            raise ValueError("No datasets loaded.")
        return ds[0]["id"]
