"""End-to-end test of the HTTP API used by the web frontend."""
import importlib
import time

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    os.environ["THEMATICML_HOME"] = str(tmp_path_factory.mktemp("api_ws"))
    import thematicml_api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def wait(client, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("completed", "failed"):
            return r
        time.sleep(0.25)
    raise TimeoutError(run_id)


def test_full_http_workflow(client, sample_csv):
    assert client.get("/api/health").json()["status"] == "ok"
    sysinfo = client.get("/api/system").json()
    assert sysinfo["llm_dependency"] is False

    pid = client.post("/api/projects", json={"name": "API project", "description": "RQ1"}).json()["id"]
    assert client.post("/api/projects", json={"name": "API project"}).status_code == 409
    assert client.post("/api/projects", json={"name": "   "}).status_code == 400  # spaces only would become an empty name
    other = client.post("/api/projects", json={"name": "Second project"}).json()["id"]
    assert client.patch(f"/api/projects/{other}", json={"name": "API project"}).status_code == 409
    assert client.patch(f"/api/projects/{other}", json={"name": " "}).status_code == 400
    assert client.patch(f"/api/projects/{other}", json={"name": "Second project", "description": "kept"}).json()["description"] == "kept"
    assert client.delete(f"/api/projects/{other}").status_code == 200

    with open(sample_csv, "rb") as fh:
        up = client.post(f"/api/projects/{pid}/uploads", files=[("files", ("feedback.csv", fh, "text/csv"))]).json()
    ds = up[0]
    assert ds["status"] == "uploaded" and ds["mapping"]["suggested"]["text_col"] == "response_text"

    bad = client.post(f"/api/projects/{pid}/uploads", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).json()
    assert bad[0]["status"] == "failed"

    pv = client.get(f"/api/datasets/{ds['id']}/preview?rows=5").json()
    assert len(pv["rows"]) == 5 and pv["total_rows"] == 401

    mapped = client.post(f"/api/datasets/{ds['id']}/mapping", json={
        "text_col": "response_text", "id_col": "respondent_id", "time_col": "created_at", "group_col": "region"}).json()
    assert mapped["status"] == "ready" and mapped["quality"]["valid_records"] == 396

    cfg = client.get("/api/config/defaults").json()["config"]
    cfg.update({"n_codes": 20})
    run = client.post(f"/api/projects/{pid}/runs", json={"dataset_ids": [ds["id"]], "config": cfg, "label": "First"}).json()
    done = wait(client, run["id"])
    assert done["status"] == "completed", done
    rid = run["id"]

    dash = client.get(f"/api/runs/{rid}/dashboard").json()
    assert dash["overview"]["total_records"] == 401
    assert dash["coding"]["unique_codes"] == 20

    g = client.get(f"/api/runs/{rid}/graph?min_cooccurrence=3&similarity=false").json()
    assert g["params"]["min_cooccurrence"] == 3 and all(e["type"] == "cooccurrence" for e in g["edges"])
    assert client.get(f"/api/runs/{rid}/clusters").json()["clusters"]
    assert client.get(f"/api/runs/{rid}/topics").json()["topics"]
    assert client.get(f"/api/runs/{rid}/temporal").json()["available"]

    code = dash["coding"]["table"][0]
    ev = client.get(f"/api/codes/{code['id']}/evidence?limit=5").json()
    assert ev["items"][0]["verified"]
    assert client.patch(f"/api/codes/{code['id']}", json={"status": "approved"}).json()["status"] == "approved"
    assert client.patch(f"/api/codes/{code['id']}", json={"status": "bogus"}).status_code == 400

    th = client.get(f"/api/runs/{rid}/themes").json()[0]
    assert client.patch(f"/api/themes/{th['id']}", json={"status": "approved", "definition": "x"}).json()["status"] == "approved"
    tev = client.get(f"/api/themes/{th['id']}/evidence").json()
    seg = tev["representative"][0]["segment_id"]
    assert client.post(f"/api/runs/{rid}/annotations", json={"segment_id": seg, "verified": True, "theme_id": th["id"]}).status_code == 200

    recs = client.get(f"/api/datasets/{ds['id']}/records?run_id={rid}&size=10&validity=valid").json()
    assert recs["total"] == 396 and "codes_assigned" in recs["rows"][0]
    excl = client.get(f"/api/datasets/{ds['id']}/records?validity=excluded").json()
    assert excl["total"] == 5
    detail = client.get(f"/api/records/{recs['rows'][0]['id']}?run_id={rid}").json()
    assert detail["segments"]
    assert client.get(f"/api/datasets/{ds['id']}/records.csv").status_code == 200

    t = client.get(f"/api/runs/{rid}/thematic").json()
    assert t["path"]["records"] == 396 and t["path"]["codes"] == 20 and t["themes"] and "rule" in t["links"]
    cl = client.get(f"/api/runs/{rid}/cleaned?size=5").json()
    assert cl["parsed"] == 401 and cl["kept"] == 396 and len(cl["rows"]) == 5
    assert client.get(f"/api/runs/{rid}/ideas").json()["ideas"]
    st = client.get(f"/api/runs/{rid}/statements?size=5").json()
    assert len(st["rows"]) == 5 and st["rows"][0]["codes"]
    # a negative LIMIT would return every row at once
    for url in (f"/api/runs/{rid}/cleaned?size=-1", f"/api/runs/{rid}/statements?page=0", f"/api/datasets/{ds['id']}/records?size=-1"):
        assert client.get(url).status_code == 422, url
    book = client.get(f"/api/runs/{rid}/codebook").json()
    assert len(book) == 20 and book[0]["quotations"][0]["verified"]
    assert client.post(f"/api/runs/{rid}/notes", json={"text": " "}).status_code == 400
    note = client.post(f"/api/runs/{rid}/notes", json={"text": "First impressions"}).json()
    assert client.get(f"/api/runs/{rid}/notes").json()["memos"][0]["note"] == "First impressions"
    assert client.delete(f"/api/notes/{note['id']}").status_code == 200
    assert client.delete(f"/api/notes/{note['id']}").status_code == 404
    assert client.get("/api/runs/run_missing/notes").status_code == 404
    themes = client.get(f"/api/runs/{rid}/themes").json()
    assert client.patch(f"/api/themes/{themes[0]['id']}", json={"parent_id": themes[0]["id"]}).status_code == 400

    for fmt in ("codes_csv", "xlsx", "html", "pdf", "docx", "graphml", "cleaned_csv", "codebook_csv", "categories_csv", "theme_links_csv", "themes_csv"):
        r = client.get(f"/api/runs/{rid}/export/{fmt}")
        assert r.status_code == 200 and len(r.content) > 100, fmt

    db = client.get(f"/api/projects/{pid}/database").json()
    assert db["total_uploads"] == 1 and db["total_records"] == 401 and db["exported_reports"] >= 6

    s = client.get(f"/api/search?project_id={pid}&q=support").json()
    assert s["records"]

    run2 = client.post(f"/api/projects/{pid}/runs", json={"dataset_ids": [ds["id"]], "config": {**cfg, "clustering_method": "hierarchical", "topic_method": "lda"}}).json()
    assert wait(client, run2["id"])["status"] == "completed"
    cmp = client.get(f"/api/compare?a={rid}&b={run2['id']}").json()
    assert any(d["parameter"] == "clustering_method" for d in cmp["model_differences"])

    assert client.get(f"/api/projects/{pid}/backup").status_code == 200
    assert client.delete(f"/api/runs/{run2['id']}").status_code == 200
    assert client.get(f"/api/runs/{run2['id']}").status_code == 404


def test_duplicate_uploads_are_refused(client, sample_csv, tmp_path):
    pid = client.post("/api/projects", json={"name": "Duplicates"}).json()["id"]
    up = lambda name, data: client.post(f"/api/projects/{pid}/uploads", files=[("files", (name, data, "text/plain"))]).json()[0]  # noqa: E731
    data = sample_csv.read_bytes()
    first = up("feedback.csv", data)
    assert first["status"] == "uploaded"
    again = up("feedback.csv", data)
    assert "id" not in again and again["status"] == "failed" and again["error"] == "This file already exists in this project. It was not added again."
    renamed = up("copy of feedback.csv", data)
    assert "id" not in renamed and "already exists in this project (uploaded as “feedback.csv”)" in renamed["error"]
    same_name = up("FEEDBACK.csv", b"id,text\n1,A different file that happens to have the same name as the first.\n")
    assert "id" not in same_name and "same name" not in same_name["error"] and "name already exists" in same_name["error"]
    assert len(client.get(f"/api/projects/{pid}/datasets").json()) == 1
    # a failed upload holds no records, so the file can be sent again
    broken = up("notes.json", b"{not json")
    assert broken["status"] == "failed" and broken["id"]
    assert up("notes.json", b'[{"text": "Fixed file with a proper list of records in it."}]')["status"] == "uploaded"
    # another project may hold the same file
    other = client.post("/api/projects", json={"name": "Duplicates elsewhere"}).json()["id"]
    assert client.post(f"/api/projects/{other}/uploads", files=[("files", ("feedback.csv", data, "text/csv"))]).json()[0]["status"] == "uploaded"
    # uploads from before fingerprints were stored are still recognised
    import thematicml_api.main as main

    main.store.update("datasets", {"id": first["id"]}, {"sha256": None})
    assert "already exists" in up("third name.csv", data)["error"]


def test_audio_video_and_any_text_type(client):
    pid = client.post("/api/projects", json={"name": "Formats"}).json()["id"]
    up = lambda name, data: client.post(f"/api/projects/{pid}/uploads", files=[("files", (name, data, "application/octet-stream"))]).json()[0]  # noqa: E731
    assert "Audio and video" in up("interview.mp3", b"ID3\x03\x00\x00\x00rest")["error"]
    assert "Audio and video" in up("interview.mp4", b"\x00\x00\x00\x18ftypisom")["error"]
    ok = up("field notes.md", "# Day one\n\nTravelling alone taught me to trust my own judgement in every new city.".encode())
    assert ok["status"] == "uploaded" and ok["n_rows"] == 2
    assert up("captions.vtt", b"WEBVTT\n\n00:01.000 --> 00:04.000\nWelcome to the show about travelling alone.\n")["status"] == "uploaded"
