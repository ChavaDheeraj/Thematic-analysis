"""Local SQLite persistence for projects, datasets, runs and researcher edits."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, format TEXT NOT NULL, size_bytes INTEGER, uploaded_at TEXT NOT NULL,
    original_path TEXT NOT NULL, encoding TEXT, structured INTEGER DEFAULT 1, columns TEXT, n_rows INTEGER,
    mapping TEXT, status TEXT NOT NULL, quality TEXT, warnings TEXT, error TEXT, sha256 TEXT
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_index INTEGER, record_key TEXT, text TEXT, timestamp TEXT, group_value TEXT, label TEXT,
    metadata TEXT, word_count INTEGER, is_valid INTEGER, is_duplicate INTEGER, exclusion_reason TEXT,
    invalid_timestamp INTEGER
);
CREATE INDEX IF NOT EXISTS ix_records_dataset ON records(dataset_id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    dataset_ids TEXT NOT NULL, label TEXT, config TEXT NOT NULL, status TEXT NOT NULL, stage TEXT,
    progress REAL DEFAULT 0, completed_stages TEXT DEFAULT '[]', error TEXT, failed_stage TEXT,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, duration_s REAL, peak_memory_mb REAL,
    package_version TEXT, metrics TEXT, stage_timings TEXT, warnings TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    pos INTEGER NOT NULL, record_id INTEGER NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    seg_index INTEGER, text TEXT, char_start INTEGER, char_end INTEGER,
    cluster INTEGER, topic INTEGER, topic_weight REAL, sentiment REAL, sentiment_label TEXT
);
CREATE INDEX IF NOT EXISTS ix_segments_run ON segments(run_id, pos);
CREATE INDEX IF NOT EXISTS ix_segments_record ON segments(record_id);
CREATE TABLE IF NOT EXISTS codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    code_key TEXT NOT NULL, label TEXT NOT NULL, description TEXT, method TEXT, feature TEXT, category TEXT,
    status TEXT DEFAULT 'candidate', origin TEXT DEFAULT 'machine', merged_into INTEGER, notes TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_codes_run ON codes(run_id);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    code_id INTEGER NOT NULL REFERENCES codes(id) ON DELETE CASCADE, segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    method TEXT, score REAL, matched_text TEXT, char_start INTEGER, char_end INTEGER, negated INTEGER DEFAULT 0,
    reviewed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_assign_run ON assignments(run_id, code_id);
CREATE INDEX IF NOT EXISTS ix_assign_segment ON assignments(segment_id);
CREATE TABLE IF NOT EXISTS clusters (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, cluster INTEGER NOT NULL, label TEXT, size INTEGER,
    terms TEXT, representative TEXT, cohesion REAL, theme_id INTEGER, category TEXT, PRIMARY KEY (run_id, cluster)
);
CREATE TABLE IF NOT EXISTS topics (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, topic INTEGER NOT NULL, label TEXT, terms TEXT,
    weights TEXT, size INTEGER, coherence REAL, PRIMARY KEY (run_id, topic)
);
CREATE TABLE IF NOT EXISTS themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label TEXT NOT NULL, definition TEXT, code_ids TEXT, status TEXT DEFAULT 'candidate', source TEXT,
    community INTEGER, representative TEXT, contradictory TEXT, majority_polarity TEXT, associated_topic TEXT,
    parent_id INTEGER, notes TEXT, origin TEXT DEFAULT 'machine', updated_at TEXT
);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    segment_id INTEGER REFERENCES segments(id) ON DELETE CASCADE, theme_id INTEGER, code_id INTEGER,
    note TEXT, verified INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, run_id TEXT, format TEXT, path TEXT, size_bytes INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_FIELDS = {
    "columns", "mapping", "quality", "warnings", "dataset_ids", "config", "completed_stages", "metrics",
    "stage_timings", "terms", "weights", "representative", "contradictory", "code_ids", "associated_topic",
    "detail", "metadata", "value", "history", "baselines", "repeats", "graph_info", "versions",
}


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def default_workspace() -> Path:
    return Path(os.environ.get("THEMATICML_HOME", Path.home() / ".thematicml"))


def _decode(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if k in JSON_FIELDS and isinstance(v, str):
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return d


def _enc(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    return v


class Store:
    def __init__(self, workspace: str | Path | None = None):
        self.root = Path(workspace) if workspace else default_workspace()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "thematicml.sqlite3"
        with self.connect() as con:
            con.executescript(SCHEMA)
            con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            # columns added after the first release; workspaces created earlier get them here
            if "sha256" not in {r[1] for r in con.execute("PRAGMA table_info(datasets)")}:
                con.execute("ALTER TABLE datasets ADD COLUMN sha256 TEXT")
        from thematicml.gnn.service import ensure_schema

        ensure_schema(self)

    # ------------------------------------------------------------------ basics
    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def query(self, sql: str, params=()) -> list[dict]:
        with self.connect() as con:
            return [_decode(r) for r in con.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()) -> dict | None:
        with self.connect() as con:
            return _decode(con.execute(sql, params).fetchone())

    def scalar(self, sql: str, params=()):
        with self.connect() as con:
            r = con.execute(sql, params).fetchone()
            return r[0] if r else None

    def insert(self, table: str, row: dict, con=None) -> int:
        cols = list(row)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        vals = [_enc(row[c]) for c in cols]
        if con is not None:
            return con.execute(sql, vals).lastrowid
        with self.connect() as c:
            return c.execute(sql, vals).lastrowid

    def update(self, table: str, key: dict, values: dict, con=None) -> None:
        if not values:
            return
        sets = ", ".join(f"{k} = ?" for k in values)
        where = " AND ".join(f"{k} = ?" for k in key)
        params = [_enc(v) for v in values.values()] + list(key.values())
        sql = f"UPDATE {table} SET {sets} WHERE {where}"
        if con is not None:
            con.execute(sql, params)
            return
        with self.connect() as c:
            c.execute(sql, params)

    # ------------------------------------------------------------------ paths
    def project_dir(self, project_id: str) -> Path:
        p = self.root / "projects" / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dataset_dir(self, project_id: str, dataset_id: str) -> Path:
        p = self.project_dir(project_id) / "datasets" / dataset_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_dir(self, project_id: str, run_id: str) -> Path:
        p = self.project_dir(project_id) / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def relative(self, p: str | Path) -> str:
        """Path stored in the database: relative to the workspace, so the workspace can be moved or restored."""
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()

    def resolve(self, stored: str | Path) -> Path:
        """Turn a stored path into a real one. Also re-roots absolute paths written by earlier versions
        (``.../projects/<id>/...``) when the workspace has since been moved."""
        p = Path(stored)
        if not p.is_absolute():
            return self.root / p
        if p.exists():
            return p
        parts = p.parts
        if "projects" in parts:
            i = len(parts) - 1 - parts[::-1].index("projects")
            moved = self.root.joinpath(*parts[i:])
            if moved.exists():
                return moved
        return p

    def export_dir(self, project_id: str) -> Path:
        p = self.project_dir(project_id) / "exports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ projects
    def create_project(self, name: str, description: str = "") -> dict:
        pid = new_id("prj")
        self.insert("projects", {"id": pid, "name": name, "description": description, "created_at": now()})
        self.log(pid, "project_created", {"name": name})
        return self.get_project(pid)

    def get_project(self, project_id: str) -> dict | None:
        return self.one("SELECT * FROM projects WHERE id = ?", (project_id,))

    def project_by_name(self, name: str) -> dict | None:
        return self.one("SELECT * FROM projects WHERE name = ?", (name,))

    def list_projects(self) -> list[dict]:
        return self.query(
            "SELECT p.*, (SELECT COUNT(*) FROM datasets d WHERE d.project_id = p.id) AS n_datasets, "
            "(SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS n_runs FROM projects p ORDER BY created_at"
        )

    def delete_project(self, project_id: str) -> None:
        import shutil

        with self.connect() as con:
            con.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            con.execute("DELETE FROM activity WHERE project_id = ?", (project_id,))
            con.execute("DELETE FROM exports WHERE project_id = ?", (project_id,))
        shutil.rmtree(self.root / "projects" / project_id, ignore_errors=True)

    # ------------------------------------------------------------------ activity
    def log(self, project_id: str | None, kind: str, detail: dict | None = None) -> None:
        self.insert("activity", {"project_id": project_id, "ts": now(), "kind": kind, "detail": detail or {}})

    def activity(self, project_id: str, limit: int = 50) -> list[dict]:
        return self.query("SELECT * FROM activity WHERE project_id = ? ORDER BY id DESC LIMIT ?", (project_id, limit))

    # ------------------------------------------------------------------ datasets
    def get_dataset(self, dataset_id: str) -> dict | None:
        return self.one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))

    def list_datasets(self, project_id: str) -> list[dict]:
        return self.query(
            "SELECT d.*, (SELECT COUNT(*) FROM runs r WHERE r.dataset_ids LIKE '%' || d.id || '%') AS n_runs "
            "FROM datasets d WHERE project_id = ? ORDER BY uploaded_at DESC",
            (project_id,),
        )

    def delete_dataset(self, dataset_id: str) -> None:
        import shutil

        ds = self.get_dataset(dataset_id)
        if not ds:
            return
        runs = [r["id"] for r in self.query("SELECT id, dataset_ids FROM runs WHERE project_id = ?", (ds["project_id"],))
                if dataset_id in r["dataset_ids"]]
        for rid in runs:
            self.delete_run(rid)
        with self.connect() as con:
            con.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        shutil.rmtree(self.root / "projects" / ds["project_id"] / "datasets" / dataset_id, ignore_errors=True)
        self.log(ds["project_id"], "dataset_deleted", {"dataset_id": dataset_id, "filename": ds["filename"], "runs_deleted": len(runs)})

    def replace_records(self, dataset_id: str, records) -> None:
        cols = ["row_index", "record_key", "text", "timestamp", "group_value", "label", "metadata", "word_count",
                "is_valid", "is_duplicate", "exclusion_reason", "invalid_timestamp"]
        rows = [
            (dataset_id, int(r.row_index), r.record_key, r.text, r.timestamp, r.group_value, r.label,
             json.dumps(r.metadata or {}), int(r.word_count), int(bool(r.is_valid)), int(bool(r.is_duplicate)),
             r.exclusion_reason, int(bool(r.invalid_timestamp)))
            for r in records.itertuples()
        ]
        with self.connect() as con:
            con.execute("DELETE FROM records WHERE dataset_id = ?", (dataset_id,))
            con.executemany(
                f"INSERT INTO records (dataset_id, {', '.join(cols)}) VALUES ({', '.join('?' * (len(cols) + 1))})", rows
            )

    # ------------------------------------------------------------------ runs
    def get_run(self, run_id: str) -> dict | None:
        return self.one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self, project_id: str) -> list[dict]:
        return self.query("SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC", (project_id,))

    def delete_run(self, run_id: str) -> None:
        import shutil

        run = self.get_run(run_id)
        if not run:
            return
        with self.connect() as con:
            con.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        shutil.rmtree(self.root / "projects" / run["project_id"] / "runs" / run_id, ignore_errors=True)

    def recover_interrupted_runs(self) -> int:
        """Mark runs left 'running'/'queued' by a crash or restart as interrupted (resumable)."""
        with self.connect() as con:
            cur = con.execute(
                "UPDATE runs SET status = 'interrupted', error = 'Application stopped before the run finished; retry to resume from the last completed stage.', "
                "failed_stage = stage WHERE status IN ('running', 'queued')"
            )
            return cur.rowcount

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> dict:
        defaults = {"max_upload_mb": 200, "max_hierarchical_segments": 8000, "worker_threads": 1}
        for r in self.query("SELECT key, value FROM settings"):
            defaults[r["key"]] = r["value"]
        return defaults

    def set_settings(self, values: dict) -> dict:
        with self.connect() as con:
            for k, v in values.items():
                con.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (k, json.dumps(v)))
        return self.get_settings()

    def storage_usage(self, project_id: str) -> dict:
        base = self.root / "projects" / project_id

        def size(p: Path) -> int:
            if not p.exists():
                return 0
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        return {
            "original_files_bytes": size(base / "datasets"),
            "model_artifacts_bytes": size(base / "runs"),
            "exports_bytes": size(base / "exports"),
            "database_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "database_note": "The SQLite database is shared by all projects in this workspace.",
        }
