"""ThematicML local web API.

Run with:  uvicorn thematicml_api.main:app --app-dir backend --port 8000
All analysis logic lives in the ``thematicml`` package; this module only
exposes it over HTTP and runs analyses in a background worker thread.
"""
from __future__ import annotations

import base64
import binascii
import os
import secrets
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import thematicml
from thematicml import review
from thematicml.analytics import compare_runs
from thematicml.analytics.sentiment import sentiment_available
from thematicml.config import AnalysisConfig, ColumnMapping, GNNConfig
from thematicml.gnn import service as gnn
from thematicml.data import LoadError
from thematicml.graph import gnn_status, leiden_available
from thematicml.pipeline import STAGES, create_run, run_stages
from thematicml.reporting import ALL_FORMATS, build_dashboard, export_project_zip, export_run
from thematicml.results import (
    NotFound,
    cluster_view,
    code_evidence,
    code_table,
    graph_view,
    record_detail,
    run_snapshot,
    temporal_view,
    theme_evidence,
    theme_table,
    topic_view,
)
from thematicml.storage import Store, apply_mapping, import_file, preview
from thematicml.thematic import cleaned_view, coded_statements, codebook, notes_view, recurring_ideas, thematic_structure

store = Store(os.environ.get("THEMATICML_HOME") or None)
store.recover_interrupted_runs()
gnn.recover_interrupted(store)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thematicml-run")

app = FastAPI(title="ThematicML", version=thematicml.__version__,
              description="Local machine-learning platform for qualitative research. No LLMs are used.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# A password protects the app once other devices can reach it (start.ps1 -Network, Docker, hosting).
# Unset, the app stays open as before, which is only safe on 127.0.0.1.
_USER = os.environ.get("THEMATICML_USER") or "researcher"
_PASSWORD = os.environ.get("THEMATICML_PASSWORD") or None
if os.environ.get("THEMATICML_REQUIRE_PASSWORD") == "1" and not _PASSWORD:
    raise RuntimeError("Set THEMATICML_PASSWORD: this server can be reached from other devices and must not run without a password.")


def _signed_in(header: str | None) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:], validate=True).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # compare both, always, so the answer takes the same time whichever part is wrong
    return secrets.compare_digest(user.encode(), _USER.encode()) & secrets.compare_digest(password.encode(), _PASSWORD.encode())


@app.middleware("http")
async def _password(request: Request, call_next):
    # the health check stays open so a hosting service can tell the app is up
    if _PASSWORD and request.url.path != "/api/health" and not _signed_in(request.headers.get("authorization")):
        return JSONResponse({"detail": "Sign in required"}, status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="ThematicML", charset="UTF-8"'})
    return await call_next(request)


@app.exception_handler(NotFound)
async def _nf(_: Request, exc: NotFound):
    return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)


@app.exception_handler(ValueError)
async def _ve(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


def _need(obj, what: str):
    if obj is None:
        raise HTTPException(404, f"{what} not found")
    return obj


def _submit(run_id: str):
    executor.submit(_safe_execute, run_id)


def _execute_steps(run_id: str):
    """The analysis one stage at a time, then the GNN experiment requested at upload time."""
    try:
        run = yield from run_stages(store, run_id)
    except Exception as exc:  # run_stages records stage failures; this guards unexpected errors
        store.update("runs", {"id": run_id}, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return
    gcfg = (run.get("config") or {}).get("gnn")
    if run["status"] == "completed" and gcfg:
        # optional GNN experiment requested at upload time; failures are recorded on the experiment
        try:
            exp = gnn.create_experiment(store, run_id, GNNConfig(**gcfg))
            executor.submit(gnn.execute_experiment, store, exp["id"])
        except Exception as exc:
            store.log(run["project_id"], "gnn_failed", {"run_id": run_id, "error": str(exc)})


def _safe_execute(run_id: str):
    for _ in _execute_steps(run_id):
        pass


_safe_execute.steps = _execute_steps  # the browser edition runs it a stage at a time (thematicml_api/browser.py)


# ---------------------------------------------------------------------- system

@app.get("/api/health")
def health():
    return {"status": "ok", "version": thematicml.__version__}


@app.get("/api/system")
def system():
    import importlib.util

    def has(m):
        return importlib.util.find_spec(m) is not None

    return {
        "version": thematicml.__version__,
        # "browser" when the API runs inside the page (thematicml_api/browser.py); the workspace is then the browser's storage
        "edition": os.environ.get("THEMATICML_EDITION", "server"),
        "workspace": str(store.root),
        "database": str(store.db_path),
        "llm_dependency": False,
        "optional": {
            "leiden": leiden_available(),
            "sentiment_vader": sentiment_available(),
            "statsmodels": has("statsmodels"),
            "pdf_reports": has("fpdf"),
            "gnn": gnn_status(),
        },
        "stages": [{"id": s, "label": label} for s, label in STAGES],
    }


@app.get("/api/settings")
def get_settings():
    return store.get_settings()


@app.put("/api/settings")
def put_settings(values: dict):
    allowed = {"max_upload_mb", "max_hierarchical_segments", "default_config"}
    return store.set_settings({k: v for k, v in values.items() if k in allowed})


@app.get("/api/config/defaults")
def config_defaults():
    s = store.get_settings()
    cfg = AnalysisConfig(**(s.get("default_config") or {})).model_dump()
    cfg["max_hierarchical_segments"] = s.get("max_hierarchical_segments", cfg["max_hierarchical_segments"])
    return {"config": cfg, "schema": AnalysisConfig.model_json_schema()}


# ---------------------------------------------------------------------- projects

class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


@app.get("/api/projects")
def list_projects():
    return store.list_projects()


def _project_name(p: ProjectIn, pid: str | None = None) -> str:
    name = p.name.strip()
    if not name:
        raise HTTPException(400, "The project needs a name")
    other = store.project_by_name(name)
    if other and other["id"] != pid:
        raise HTTPException(409, "A project with this name already exists")
    return name


@app.post("/api/projects")
def create_project(p: ProjectIn):
    return store.create_project(_project_name(p), p.description)


@app.patch("/api/projects/{pid}")
def update_project(pid: str, p: ProjectIn):
    _need(store.get_project(pid), "Project")
    store.update("projects", {"id": pid}, {"name": _project_name(p, pid), "description": p.description})
    return store.get_project(pid)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    _need(store.get_project(pid), "Project")
    store.delete_project(pid)
    return {"deleted": pid}


@app.get("/api/projects/{pid}/backup")
def backup_project(pid: str):
    _need(store.get_project(pid), "Project")
    path = export_project_zip(store, pid)
    return FileResponse(path, filename=path.name, media_type="application/zip")


@app.get("/api/projects/{pid}/activity")
def activity(pid: str, limit: int = 50):
    return store.activity(pid, limit)


# ---------------------------------------------------------------------- datasets

@app.post("/api/projects/{pid}/uploads")
async def upload(pid: str, files: list[UploadFile] = File(...)):
    _need(store.get_project(pid), "Project")
    limit = int(store.get_settings().get("max_upload_mb", 200)) * 2**20
    out = []
    for f in files:
        tmpdir = tempfile.mkdtemp(prefix="tml-upload-")
        try:
            tmp = Path(tmpdir) / Path(f.filename or "upload").name
            size = 0
            with tmp.open("wb") as fh:
                while chunk := await f.read(2**20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"{f.filename}: file exceeds the {limit // 2**20} MB upload limit (change it in Settings).")
                    fh.write(chunk)
            try:
                ds = import_file(store, pid, tmp, f.filename)
            except LoadError as exc:
                out.append({"filename": f.filename, "status": "failed", "error": str(exc)})
                continue
            out.append(ds)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return out


@app.get("/api/projects/{pid}/datasets")
def list_datasets(pid: str):
    return store.list_datasets(pid)


@app.get("/api/datasets/{did}")
def get_dataset(did: str):
    return _need(store.get_dataset(did), "Dataset")


@app.get("/api/datasets/{did}/preview")
def dataset_preview(did: str, rows: int = Query(20, le=200)):
    _need(store.get_dataset(did), "Dataset")
    return preview(store, did, rows)


@app.post("/api/datasets/{did}/mapping")
def map_dataset(did: str, mapping: ColumnMapping):
    _need(store.get_dataset(did), "Dataset")
    return apply_mapping(store, did, mapping)


@app.delete("/api/datasets/{did}")
def delete_dataset(did: str):
    _need(store.get_dataset(did), "Dataset")
    store.delete_dataset(did)
    return {"deleted": did}


@app.get("/api/datasets/{did}/original")
def download_original(did: str):
    ds = _need(store.get_dataset(did), "Dataset")
    path = store.resolve(ds["original_path"])
    if not path.exists():
        raise HTTPException(404, "The original file is missing from the workspace.")
    return FileResponse(path, filename=ds["filename"])


def _record_filters(did, search, validity, min_words, max_words, date_from, date_to, group):
    where, params = ["r.dataset_id = ?"], [did]
    if search:
        where.append("(r.text LIKE ? OR r.record_key LIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
    if validity == "valid":
        where.append("r.is_valid = 1")
    elif validity == "excluded":
        where.append("r.is_valid = 0")
    elif validity == "duplicate":
        where.append("r.is_duplicate = 1")
    if min_words is not None:
        where.append("r.word_count >= ?")
        params.append(min_words)
    if max_words is not None:
        where.append("r.word_count <= ?")
        params.append(max_words)
    if date_from:
        where.append("r.timestamp >= ?")
        params.append(date_from)
    if date_to:
        where.append("r.timestamp <= ?")
        params.append(date_to + "T23:59:59" if len(date_to) == 10 else date_to)
    if group:
        where.append("r.group_value = ?")
        params.append(group)
    return " AND ".join(where), params


@app.get("/api/datasets/{did}/records")
def records(did: str, page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500), search: str = "", validity: str = "all",
            min_words: int | None = None, max_words: int | None = None, date_from: str | None = None,
            date_to: str | None = None, group: str | None = None, run_id: str | None = None,
            coded: str = "all", sort: str = "row_index", order: str = "asc"):
    ds = _need(store.get_dataset(did), "Dataset")
    where, params = _record_filters(did, search, validity, min_words, max_words, date_from, date_to, group)
    if run_id and coded in ("coded", "uncoded"):
        op = "IN" if coded == "coded" else "NOT IN"
        where += f" AND r.id {op} (SELECT s.record_id FROM segments s JOIN assignments a ON a.segment_id = s.id JOIN codes c ON c.id = a.code_id WHERE s.run_id = ? AND c.status IN ('candidate','approved'))"
        params.append(run_id)
    sort_col = {"row_index": "r.row_index", "word_count": "r.word_count", "timestamp": "r.timestamp", "record_key": "r.record_key"}.get(sort, "r.row_index")
    direction = "DESC" if order == "desc" else "ASC"
    total = store.scalar(f"SELECT COUNT(*) FROM records r WHERE {where}", params)
    rows = store.query(
        f"SELECT r.id, r.row_index, r.record_key, r.text, r.timestamp, r.group_value, r.label, r.word_count, r.is_valid, "
        f"r.is_duplicate, r.exclusion_reason, r.invalid_timestamp FROM records r WHERE {where} ORDER BY {sort_col} {direction} LIMIT ? OFFSET ?",
        params + [size, (page - 1) * size],
    )
    if run_id and rows:
        ids = [r["id"] for r in rows]
        q = ",".join("?" * len(ids))
        counts = store.query(
            f"SELECT s.record_id, COUNT(DISTINCT s.id) AS segments, COUNT(DISTINCT CASE WHEN c.status IN ('candidate','approved') THEN a.code_id END) AS codes, "
            f"AVG(s.sentiment) AS sentiment FROM segments s LEFT JOIN assignments a ON a.segment_id = s.id LEFT JOIN codes c ON c.id = a.code_id "
            f"WHERE s.run_id = ? AND s.record_id IN ({q}) GROUP BY s.record_id", [run_id, *ids])
        codes_of = {}
        for r in store.query(
            f"SELECT DISTINCT s.record_id, a.code_id FROM segments s JOIN assignments a ON a.segment_id = s.id JOIN codes c ON c.id = a.code_id "
            f"WHERE s.run_id = ? AND s.record_id IN ({q}) AND c.status IN ('candidate','approved')", [run_id, *ids]):
            codes_of.setdefault(r["record_id"], set()).add(r["code_id"])
        themes = theme_table(store, run_id)
        by = {c["record_id"]: c for c in counts}
        for r in rows:
            c = by.get(r["id"], {})
            r["segments"] = c.get("segments", 0)
            r["codes_assigned"] = c.get("codes", 0)
            r["sentiment"] = None if c.get("sentiment") is None else round(c["sentiment"], 3)
            r["themes_assigned"] = sum(1 for t in themes if codes_of.get(r["id"], set()) & set(t["code_ids"]))
    groups = [g["group_value"] for g in store.query("SELECT DISTINCT group_value FROM records WHERE dataset_id = ? AND group_value IS NOT NULL ORDER BY 1", (did,))]
    return {"dataset": {"id": ds["id"], "filename": ds["filename"]}, "total": total, "page": page, "size": size, "rows": rows, "groups": groups[:200]}


@app.get("/api/datasets/{did}/records.csv")
def records_csv(did: str, search: str = "", validity: str = "all", min_words: int | None = None, max_words: int | None = None,
                date_from: str | None = None, date_to: str | None = None, group: str | None = None):
    import csv
    import io

    ds = _need(store.get_dataset(did), "Dataset")
    where, params = _record_filters(did, search, validity, min_words, max_words, date_from, date_to, group)
    rows = store.query(f"SELECT r.id, r.row_index, r.record_key, r.text, r.timestamp, r.group_value, r.label, r.word_count, r.is_valid, "
                       f"r.is_duplicate, r.exclusion_reason FROM records r WHERE {where} ORDER BY r.row_index", params)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["id"])
    w.writeheader()
    w.writerows(rows)
    store.log(ds["project_id"], "records_exported", {"dataset_id": did, "rows": len(rows)})
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{Path(ds["filename"]).stem}_records.csv"'})


@app.get("/api/records/{rid}")
def get_record(rid: int, run_id: str | None = None):
    return record_detail(store, rid, run_id)


# ---------------------------------------------------------------------- runs

class RunIn(BaseModel):
    dataset_ids: list[str]
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    label: str | None = None


@app.post("/api/projects/{pid}/runs")
def start_run(pid: str, body: RunIn):
    _need(store.get_project(pid), "Project")
    run = create_run(store, pid, body.dataset_ids, body.config, body.label)
    _submit(run["id"])
    return run


@app.get("/api/projects/{pid}/runs")
def list_runs(pid: str):
    return store.list_runs(pid)


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    return _need(store.get_run(rid), "Run")


@app.post("/api/runs/{rid}/retry")
def retry_run(rid: str):
    run = _need(store.get_run(rid), "Run")
    if run["status"] not in ("failed", "interrupted"):
        raise HTTPException(400, f"Only failed or interrupted runs can be retried (status: {run['status']}).")
    store.update("runs", {"id": rid}, {"status": "queued", "error": None})
    store.log(run["project_id"], "analysis_retried", {"run_id": rid, "from_stage": run.get("failed_stage")})
    _submit(rid)
    return store.get_run(rid)


@app.patch("/api/runs/{rid}")
def rename_run(rid: str, body: dict):
    _need(store.get_run(rid), "Run")
    if body.get("label"):
        store.update("runs", {"id": rid}, {"label": str(body["label"])[:120]})
    return store.get_run(rid)


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    run = _need(store.get_run(rid), "Run")
    if run["status"] in ("running", "queued"):
        raise HTTPException(400, "Cannot delete a run that is in progress.")
    store.delete_run(rid)
    store.log(run["project_id"], "run_deleted", {"run_id": rid})
    return {"deleted": rid}


def _completed(rid: str) -> dict:
    run = _need(store.get_run(rid), "Run")
    if run["status"] != "completed":
        raise HTTPException(409, f"Run is {run['status']}; results are available once it completes.")
    return run


@app.get("/api/runs/{rid}/dashboard")
def dashboard(rid: str):
    _completed(rid)
    return build_dashboard(store, rid)


@app.get("/api/runs/{rid}/codes")
def codes(rid: str):
    _completed(rid)
    return code_table(store, rid)


@app.get("/api/runs/{rid}/themes")
def themes(rid: str):
    _completed(rid)
    return theme_table(store, rid)


@app.get("/api/runs/{rid}/graph")
def graph(rid: str, level: str | None = None, min_cooccurrence: int | None = None, min_npmi: float | None = None,
          similarity: bool = True, similarity_threshold: float | None = None, method: str | None = None,
          resolution: float | None = None, include_isolated: bool = True):
    _completed(rid)
    return graph_view(store, rid, level, min_cooccurrence, similarity, similarity_threshold, method, resolution, include_isolated, min_npmi)


@app.get("/api/runs/{rid}/clusters")
def clusters(rid: str):
    _completed(rid)
    return cluster_view(store, rid)


@app.get("/api/runs/{rid}/topics")
def topics(rid: str):
    _completed(rid)
    return topic_view(store, rid)


@app.get("/api/runs/{rid}/temporal")
def temporal(rid: str):
    _completed(rid)
    return temporal_view(store, rid)


@app.get("/api/runs/{rid}/thematic")
def thematic(rid: str):
    _completed(rid)
    return thematic_structure(store, rid)


@app.get("/api/runs/{rid}/cleaned")
def cleaned(rid: str, page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=250), search: str = ""):
    _completed(rid)
    return cleaned_view(store, rid, page, size, search)


@app.get("/api/runs/{rid}/ideas")
def ideas(rid: str):
    _completed(rid)
    return recurring_ideas(store, rid)


@app.get("/api/runs/{rid}/statements")
def statements(rid: str, page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=250), search: str = "", code_id: int | None = None):
    _completed(rid)
    return coded_statements(store, rid, page, size, search, code_id)


@app.get("/api/runs/{rid}/codebook")
def run_codebook(rid: str):
    _completed(rid)
    return codebook(store, rid)


class NoteIn(BaseModel):
    text: str


@app.get("/api/runs/{rid}/notes")
def notes(rid: str):
    return notes_view(store, rid)


@app.post("/api/runs/{rid}/notes")
def add_note(rid: str, body: NoteIn):
    return review.add_note(store, rid, body.text)


@app.delete("/api/notes/{nid}")
def delete_note(nid: int):
    review.delete_note(store, nid)
    return {"deleted": nid}


@app.get("/api/runs/{rid}/export/{fmt}")
def export(rid: str, fmt: str):
    _completed(rid)
    if fmt not in ALL_FORMATS:
        raise HTTPException(400, f"Unknown format. Options: {sorted(ALL_FORMATS)}")
    try:
        path = export_run(store, rid, fmt)
    except RuntimeError as exc:
        raise HTTPException(501, str(exc))
    return FileResponse(path, filename=path.name)


@app.get("/api/compare")
def compare(a: str, b: str):
    _completed(a)
    _completed(b)
    return compare_runs(run_snapshot(store, a), run_snapshot(store, b))


# ---------------------------------------------------------------------- review: codes

class CodePatch(BaseModel):
    label: str | None = None
    description: str | None = None
    category: str | None = None
    status: str | None = None
    notes: str | None = None


class MergeIn(BaseModel):
    code_ids: list[int]
    label: str | None = None
    target_id: int | None = None


class SplitIn(BaseModel):
    assignment_ids: list[int]
    new_label: str


class NewCode(BaseModel):
    label: str
    segment_ids: list[int]
    description: str = ""


@app.patch("/api/codes/{cid}")
def patch_code(cid: int, body: CodePatch):
    return review.update_code(store, cid, **body.model_dump())


@app.post("/api/codes/merge")
def merge(body: MergeIn):
    return review.merge_codes(store, body.code_ids, body.label, body.target_id)


@app.post("/api/codes/{cid}/split")
def split(cid: int, body: SplitIn):
    return review.split_code(store, cid, body.assignment_ids, body.new_label)


@app.post("/api/runs/{rid}/codes")
def new_code(rid: str, body: NewCode):
    return review.create_code(store, rid, body.label, body.segment_ids, body.description)


@app.get("/api/codes/{cid}/evidence")
def evidence(cid: int, limit: int = Query(100, le=1000), offset: int = 0):
    return code_evidence(store, cid, limit, offset)


# ---------------------------------------------------------------------- review: themes

class ThemePatch(BaseModel):
    label: str | None = None
    definition: str | None = None
    status: str | None = None
    notes: str | None = None
    code_ids: list[int] | None = None
    parent_id: int | None = None


class NewTheme(BaseModel):
    label: str
    code_ids: list[int]
    definition: str = ""
    parent_id: int | None = None


class AnnotationIn(BaseModel):
    segment_id: int
    note: str = ""
    verified: bool | None = None
    theme_id: int | None = None
    code_id: int | None = None


@app.patch("/api/themes/{tid}")
def patch_theme(tid: int, body: ThemePatch):
    return review.update_theme(store, tid, **body.model_dump())


@app.post("/api/runs/{rid}/themes")
def new_theme(rid: str, body: NewTheme):
    return review.create_theme(store, rid, body.label, body.code_ids, body.definition, body.parent_id)


@app.get("/api/themes/{tid}/evidence")
def t_evidence(tid: int):
    return theme_evidence(store, tid)


@app.post("/api/runs/{rid}/annotations")
def annotate(rid: str, body: AnnotationIn):
    return review.annotate(store, rid, body.segment_id, body.note, body.verified, body.theme_id, body.code_id)


# ---------------------------------------------------------------------- review: clusters

class ClusterPatch(BaseModel):
    label: str | None = None
    theme_id: int | None = None
    category: str | None = None


class ClusterMerge(BaseModel):
    clusters: list[int]
    label: str | None = None


@app.patch("/api/runs/{rid}/clusters/{c}")
def patch_cluster(rid: str, c: int, body: ClusterPatch):
    return review.update_cluster(store, rid, c, body.label, body.theme_id, body.category)


@app.post("/api/runs/{rid}/clusters/merge")
def merge_clusters(rid: str, body: ClusterMerge):
    return review.merge_clusters(store, rid, body.clusters, body.label)


@app.post("/api/runs/{rid}/clusters/{c}/split")
def split_cluster(rid: str, c: int, k: int = 2):
    return review.split_cluster(store, rid, c, k)


# ---------------------------------------------------------------------- optional GNN

@app.get("/api/gnn/status")
def gnn_status_ep():
    return {**gnn_status(), "versions": gnn.software_versions(), "defaults": GNNConfig().model_dump()}


@app.post("/api/runs/{rid}/gnn/validate")
def gnn_validate(rid: str, cfg: GNNConfig):
    _completed(rid)
    return gnn.validate(store, rid, cfg)


@app.post("/api/runs/{rid}/gnn")
def gnn_create(rid: str, cfg: GNNConfig):
    _completed(rid)
    check = gnn.validate(store, rid, cfg)
    if not check["ok"]:
        raise HTTPException(400, " ".join(check["reasons"]))
    exp = gnn.create_experiment(store, rid, cfg)
    executor.submit(gnn.execute_experiment, store, exp["id"])
    return exp


@app.get("/api/runs/{rid}/gnn")
def gnn_list_run(rid: str):
    _need(store.get_run(rid), "Run")
    return gnn.list_experiments(store, run_id=rid)


@app.get("/api/projects/{pid}/gnn")
def gnn_list_project(pid: str):
    _need(store.get_project(pid), "Project")
    return gnn.list_experiments(store, project_id=pid)


@app.get("/api/gnn/{eid}")
def gnn_get(eid: str):
    return _need(gnn.get_experiment(store, eid), "Experiment")


@app.get("/api/gnn/{eid}/predictions")
def gnn_predictions(eid: str, limit: int = Query(200, le=5000)):
    _need(gnn.get_experiment(store, eid), "Experiment")
    return gnn.predictions(store, eid, limit)


@app.get("/api/gnn/{eid}/embeddings")
def gnn_embeddings(eid: str):
    _need(gnn.get_experiment(store, eid), "Experiment")
    return gnn.embeddings(store, eid)


@app.post("/api/gnn/{eid}/rescore")
def gnn_rescore(eid: str, top_k: int = Query(50, ge=1, le=1000)):
    _need(gnn.get_experiment(store, eid), "Experiment")
    return gnn.rescore(store, eid, top_k)


@app.get("/api/gnn/{eid}/export")
def gnn_export(eid: str, what: str = "predictions"):
    import csv
    import io
    import json as _json

    exp = _need(gnn.get_experiment(store, eid), "Experiment")
    if what == "experiment":
        body = _json.dumps({k: v for k, v in exp.items()}, indent=2, default=str)
        return StreamingResponse(iter([body]), media_type="application/json",
                                 headers={"Content-Disposition": f'attachment; filename="{eid}.json"'})
    rows = gnn.predictions(store, eid, 100000)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["experiment_id"])
    w.writeheader()
    w.writerows(rows)
    store.log(exp["project_id"], "gnn_exported", {"experiment_id": eid})
    return StreamingResponse(iter(["﻿" + buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{eid}_predictions.csv"'})


@app.delete("/api/gnn/{eid}")
def gnn_delete(eid: str):
    _need(gnn.get_experiment(store, eid), "Experiment")
    gnn.delete_experiment(store, eid)
    return {"deleted": eid}


# ---------------------------------------------------------------------- database page

@app.get("/api/projects/{pid}/database")
def database_summary(pid: str):
    _need(store.get_project(pid), "Project")
    runs = store.list_runs(pid)
    run_ids = [r["id"] for r in runs]
    q = ",".join("?" * len(run_ids)) or "NULL"
    one = lambda sql, p=(): store.scalar(sql, p) or 0  # noqa: E731
    usage = store.storage_usage(pid)
    datasets = store.list_datasets(pid)
    return {
        "total_uploads": len(datasets),
        "source_files": len({d["filename"] for d in datasets}),
        "total_records": one("SELECT COUNT(*) FROM records r JOIN datasets d ON d.id = r.dataset_id WHERE d.project_id = ?", (pid,)),
        "processed_records": one("SELECT COUNT(*) FROM records r JOIN datasets d ON d.id = r.dataset_id WHERE d.project_id = ? AND r.is_valid = 1", (pid,)),
        "total_segments": one(f"SELECT COUNT(*) FROM segments WHERE run_id IN ({q})", run_ids),
        "total_code_assignments": one(f"SELECT COUNT(*) FROM assignments WHERE run_id IN ({q})", run_ids),
        "unique_codes": one(f"SELECT COUNT(DISTINCT lower(label)) FROM codes WHERE run_id IN ({q}) AND status IN ('candidate','approved')", run_ids),
        "categories": one(f"SELECT COUNT(DISTINCT category) FROM codes WHERE run_id IN ({q}) AND category IS NOT NULL AND category != ''", run_ids),
        "candidate_themes": one(f"SELECT COUNT(*) FROM themes WHERE run_id IN ({q}) AND status = 'candidate'", run_ids),
        "approved_themes": one(f"SELECT COUNT(*) FROM themes WHERE run_id IN ({q}) AND status = 'approved'", run_ids),
        "analysis_runs": len(runs),
        "failed_runs": sum(1 for r in runs if r["status"] in ("failed", "interrupted")),
        "exported_reports": one("SELECT COUNT(*) FROM exports WHERE project_id = ?", (pid,)),
        "storage": usage,
        "storage_total_bytes": usage["original_files_bytes"] + usage["model_artifacts_bytes"] + usage["exports_bytes"],
        "definitions": {
            "total_segments": "text segments stored across all analysis runs (each run segments its data)",
            "total_code_assignments": "code-to-segment assignments across all runs",
            "unique_codes": "distinct active code labels across all runs",
        },
        "exports": store.query("SELECT * FROM exports WHERE project_id = ? ORDER BY id DESC LIMIT 100", (pid,)),
    }


@app.get("/api/search")
def search(project_id: str, q: str = Query(..., min_length=1)):
    like = f"%{q}%"
    runs = [r["id"] for r in store.list_runs(project_id) if r["status"] == "completed"]
    rq = ",".join("?" * len(runs)) or "NULL"
    return {
        "datasets": store.query("SELECT id, filename, status FROM datasets WHERE project_id = ? AND filename LIKE ? LIMIT 10", (project_id, like)),
        "codes": store.query(f"SELECT id, run_id, label, status FROM codes WHERE run_id IN ({rq}) AND label LIKE ? AND status IN ('candidate','approved') LIMIT 15", (*runs, like)),
        "themes": store.query(f"SELECT id, run_id, label, status FROM themes WHERE run_id IN ({rq}) AND label LIKE ? LIMIT 15", (*runs, like)),
        "records": store.query(
            "SELECT r.id, r.dataset_id, r.record_key, substr(r.text, 1, 160) AS snippet FROM records r JOIN datasets d ON d.id = r.dataset_id "
            "WHERE d.project_id = ? AND r.text LIKE ? LIMIT 15", (project_id, like)),
    }


# ---------------------------------------------------------------------- frontend

_dist = Path(os.environ.get("THEMATICML_FRONTEND", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        candidate = (_dist / full_path).resolve()
        # a crafted URL (encoded ../ or an absolute path) must not reach files outside the build folder
        if full_path and candidate.is_relative_to(_dist.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
