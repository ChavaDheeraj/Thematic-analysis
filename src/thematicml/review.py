"""Researcher review actions. Every change is logged in the activity table."""
from __future__ import annotations

import numpy as np

from thematicml.results import NotFound, feature_matrix_by_pos, get_run_or_404, load_features, load_segments
from thematicml.storage.database import Store, now

CODE_STATUSES = ("candidate", "approved", "rejected")
THEME_STATUSES = ("candidate", "approved", "rejected", "needs_revision")


def _code(store: Store, code_id: int) -> dict:
    c = store.one("SELECT * FROM codes WHERE id = ?", (code_id,))
    if not c:
        raise NotFound(f"Code {code_id} not found")
    return c


def _project_of_run(store: Store, run_id: str) -> str:
    return get_run_or_404(store, run_id)["project_id"]


def update_code(store: Store, code_id: int, **fields) -> dict:
    c = _code(store, code_id)
    allowed = {k: v for k, v in fields.items() if k in ("label", "description", "category", "status", "notes") and v is not None}
    if "status" in allowed and allowed["status"] not in CODE_STATUSES:
        raise ValueError(f"status must be one of {CODE_STATUSES}")
    if "label" in allowed and not str(allowed["label"]).strip():
        raise ValueError("label cannot be empty")
    allowed["updated_at"] = now()
    store.update("codes", {"id": code_id}, allowed)
    with store.connect() as con:
        con.execute("UPDATE assignments SET reviewed = 1 WHERE code_id = ?", (code_id,))
    store.log(_project_of_run(store, c["run_id"]), "code_reviewed", {"code_id": code_id, "label": c["label"], "changes": list(allowed)})
    return _code(store, code_id)


def merge_codes(store: Store, code_ids: list[int], label: str | None = None, target_id: int | None = None) -> dict:
    """Merge codes into ``target_id`` (default: the first). Assignments move to the
    target (duplicates on the same segment are dropped); merged codes keep a pointer."""
    if len(code_ids) < 2:
        raise ValueError("Select at least two codes to merge.")
    codes = [_code(store, i) for i in code_ids]
    if len({c["run_id"] for c in codes}) != 1:
        raise ValueError("Codes must belong to the same analysis run.")
    target = target_id or code_ids[0]
    if target not in code_ids:
        raise ValueError("Target must be one of the merged codes.")
    others = [i for i in code_ids if i != target]
    q = ",".join("?" * len(others))
    with store.connect() as con:
        con.execute(
            f"UPDATE assignments SET code_id = ?, reviewed = 1 WHERE code_id IN ({q}) AND segment_id NOT IN "
            f"(SELECT segment_id FROM assignments WHERE code_id = ?)", (target, *others, target))
        con.execute(f"DELETE FROM assignments WHERE code_id IN ({q})", others)
        con.execute(f"UPDATE codes SET status = 'merged', merged_into = ?, updated_at = ? WHERE id IN ({q})", (target, now(), *others))
        fields = {"updated_at": now(), "origin": "researcher"}
        if label:
            fields["label"] = label
        merged_labels = [c["label"] for c in codes]
        fields["description"] = f"Merged by researcher from: {', '.join(merged_labels)}."
        store.update("codes", {"id": target}, fields, con)
    store.log(_project_of_run(store, codes[0]["run_id"]), "codes_merged", {"target": target, "merged": others, "labels": [c["label"] for c in codes]})
    return _code(store, target)


def split_code(store: Store, code_id: int, assignment_ids: list[int], new_label: str) -> dict:
    """Move the selected assignments of a code into a new researcher-created code."""
    c = _code(store, code_id)
    if not assignment_ids:
        raise ValueError("Select passages to move into the new code.")
    if not new_label.strip():
        raise ValueError("The new code needs a label.")
    q = ",".join("?" * len(assignment_ids))
    owned = store.scalar(f"SELECT COUNT(*) FROM assignments WHERE code_id = ? AND id IN ({q})", (code_id, *assignment_ids))
    if owned != len(assignment_ids):
        raise ValueError("Some selected passages do not belong to this code.")
    with store.connect() as con:
        new_id = store.insert("codes", {
            "run_id": c["run_id"], "code_key": f"split:{code_id}:{new_label}", "label": new_label.strip(),
            "description": f"Split by researcher from “{c['label']}”.", "method": "researcher", "feature": c["feature"],
            "category": c["category"], "status": "candidate", "origin": "researcher", "updated_at": now(),
        }, con)
        con.execute(f"UPDATE assignments SET code_id = ?, reviewed = 1 WHERE id IN ({q})", (new_id, *assignment_ids))
    # themes containing the original code also get the new one
    for t in store.query("SELECT id, code_ids FROM themes WHERE run_id = ?", (c["run_id"],)):
        if code_id in (t["code_ids"] or []):
            store.update("themes", {"id": t["id"]}, {"code_ids": t["code_ids"] + [new_id]})
    store.log(_project_of_run(store, c["run_id"]), "code_split", {"from": code_id, "new": new_id, "moved": len(assignment_ids)})
    return _code(store, new_id)


def create_code(store: Store, run_id: str, label: str, segment_ids: list[int], description: str = "") -> dict:
    get_run_or_404(store, run_id)
    _require_in_run(store, run_id, "segments", segment_ids, "segments")
    with store.connect() as con:
        cid = store.insert("codes", {"run_id": run_id, "code_key": f"manual:{label}", "label": label, "description": description or "Created by researcher.",
                                     "method": "researcher", "status": "approved", "origin": "researcher", "updated_at": now()}, con)
        con.executemany("INSERT INTO assignments (run_id, code_id, segment_id, method, score, reviewed) VALUES (?,?,?,?,?,1)",
                        [(run_id, cid, s, "researcher", 1.0) for s in segment_ids])
    store.log(_project_of_run(store, run_id), "code_created", {"code_id": cid, "label": label})
    return _code(store, cid)


# ---------------------------------------------------------------------- themes

def update_theme(store: Store, theme_id: int, **fields) -> dict:
    t = store.one("SELECT * FROM themes WHERE id = ?", (theme_id,))
    if not t:
        raise NotFound(f"Theme {theme_id} not found")
    allowed = {k: v for k, v in fields.items() if k in ("label", "definition", "status", "notes", "code_ids", "parent_id") and v is not None}
    if "code_ids" in allowed:
        _require_in_run(store, t["run_id"], "codes", allowed["code_ids"], "codes")
    if "status" in allowed and allowed["status"] not in THEME_STATUSES:
        raise ValueError(f"status must be one of {THEME_STATUSES}")
    if "label" in allowed and not str(allowed["label"]).strip():
        raise ValueError("label cannot be empty")
    # the form sends every field, so only real changes mark the text as the researcher's own
    if any(k in allowed and allowed[k] != t[k] for k in ("label", "definition")):
        allowed["origin"] = "researcher"
    if "parent_id" in allowed:
        allowed["parent_id"] = _check_parent(store, t, allowed["parent_id"])
    allowed["updated_at"] = now()
    store.update("themes", {"id": theme_id}, allowed)
    pid = _project_of_run(store, t["run_id"])
    if allowed.get("status") == "approved":
        store.log(pid, "theme_approved", {"theme_id": theme_id, "label": allowed.get("label", t["label"])})
    else:
        store.log(pid, "theme_reviewed", {"theme_id": theme_id, "changes": list(allowed)})
    return store.one("SELECT * FROM themes WHERE id = ?", (theme_id,))


def _check_parent(store: Store, theme: dict, parent_id: int) -> int | None:
    """Sub-themes go one level deep. A parent of 0 makes the theme top-level again."""
    if not parent_id:
        return None
    parent = store.one("SELECT id, parent_id FROM themes WHERE id = ? AND run_id = ?", (parent_id, theme["run_id"]))
    if not parent or parent_id == theme["id"]:
        raise ValueError("A sub-theme needs another theme of the same analysis as its parent.")
    if parent["parent_id"]:
        raise ValueError("That theme is already a sub-theme; sub-themes cannot be nested further.")
    if store.scalar("SELECT COUNT(*) FROM themes WHERE parent_id = ?", (theme["id"],)):
        raise ValueError("This theme has sub-themes of its own; move those first.")
    return parent_id


def _require_in_run(store: Store, run_id: str, table: str, ids: list[int], what: str) -> None:
    if not ids:
        return
    q = ",".join("?" * len(ids))
    n = store.scalar(f"SELECT COUNT(*) FROM {table} WHERE run_id = ? AND id IN ({q})", (run_id, *ids))
    if n != len(set(ids)):
        raise ValueError(f"Some {what} do not belong to this analysis run.")


def create_theme(store: Store, run_id: str, label: str, code_ids: list[int], definition: str = "", parent_id: int | None = None) -> dict:
    get_run_or_404(store, run_id)
    _require_in_run(store, run_id, "codes", code_ids, "codes")
    tid = store.insert("themes", {"run_id": run_id, "label": label, "definition": definition, "code_ids": code_ids,
                                  "status": "candidate", "source": "researcher", "representative": [], "contradictory": [],
                                  "parent_id": parent_id, "origin": "researcher", "updated_at": now()})
    store.log(_project_of_run(store, run_id), "theme_created", {"theme_id": tid, "label": label})
    return store.one("SELECT * FROM themes WHERE id = ?", (tid,))


def annotate(store: Store, run_id: str, segment_id: int, note: str = "", verified: bool | None = None,
             theme_id: int | None = None, code_id: int | None = None) -> dict:
    get_run_or_404(store, run_id)
    _require_in_run(store, run_id, "segments", [segment_id], "segments")
    if theme_id is not None:
        _require_in_run(store, run_id, "themes", [theme_id], "themes")
    aid = store.insert("annotations", {"run_id": run_id, "segment_id": segment_id, "theme_id": theme_id, "code_id": code_id,
                                       "note": note, "verified": None if verified is None else int(verified), "created_at": now()})
    store.log(_project_of_run(store, run_id), "quotation_verified" if verified else "annotation_added", {"segment_id": segment_id, "theme_id": theme_id})
    return store.one("SELECT * FROM annotations WHERE id = ?", (aid,))


def add_note(store: Store, run_id: str, text: str) -> dict:
    """A free-standing researcher memo on this analysis (no passage, code or theme attached)."""
    get_run_or_404(store, run_id)
    if not text.strip():
        raise ValueError("A note cannot be empty.")
    nid = store.insert("annotations", {"run_id": run_id, "note": text.strip(), "created_at": now()})
    store.log(_project_of_run(store, run_id), "annotation_added", {"note_id": nid})
    return store.one("SELECT id, note, created_at FROM annotations WHERE id = ?", (nid,))


def delete_note(store: Store, note_id: int) -> None:
    n = store.one("SELECT run_id FROM annotations WHERE id = ? AND segment_id IS NULL AND theme_id IS NULL AND code_id IS NULL", (note_id,))
    if not n:
        raise NotFound(f"Note {note_id} not found")
    with store.connect() as con:
        con.execute("DELETE FROM annotations WHERE id = ?", (note_id,))
    store.log(_project_of_run(store, n["run_id"]), "note_deleted", {"note_id": note_id})


# ---------------------------------------------------------------------- clusters

def update_cluster(store: Store, run_id: str, cluster: int, label: str | None = None, theme_id: int | None = None, category: str | None = None) -> dict:
    vals = {k: v for k, v in {"label": label, "theme_id": theme_id, "category": category}.items() if v is not None}
    store.update("clusters", {"run_id": run_id, "cluster": cluster}, vals)
    store.log(_project_of_run(store, run_id), "cluster_updated", {"run_id": run_id, "cluster": cluster, **vals})
    return store.one("SELECT * FROM clusters WHERE run_id = ? AND cluster = ?", (run_id, cluster))


def merge_clusters(store: Store, run_id: str, clusters: list[int], label: str | None = None) -> dict:
    if len(clusters) < 2:
        raise ValueError("Select at least two clusters.")
    target, others = clusters[0], clusters[1:]
    q = ",".join("?" * len(others))
    with store.connect() as con:
        con.execute(f"UPDATE segments SET cluster = ? WHERE run_id = ? AND cluster IN ({q})", (target, run_id, *others))
        size = con.execute("SELECT COUNT(*) FROM segments WHERE run_id = ? AND cluster = ?", (run_id, target)).fetchone()[0]
        reps = []
        for row in con.execute(f"SELECT representative FROM clusters WHERE run_id = ? AND cluster IN ({','.join('?' * len(clusters))})", (run_id, *clusters)):
            import json

            reps += json.loads(row[0] or "[]")
        reps.sort(key=lambda r: -r["similarity"])
        con.execute(f"DELETE FROM clusters WHERE run_id = ? AND cluster IN ({q})", (run_id, *others))
        store.update("clusters", {"run_id": run_id, "cluster": target}, {"size": size, "representative": reps[:5], **({"label": label} if label else {})}, con)
    store.log(_project_of_run(store, run_id), "clusters_merged", {"run_id": run_id, "clusters": clusters})
    return store.one("SELECT * FROM clusters WHERE run_id = ? AND cluster = ?", (run_id, target))


def split_cluster(store: Store, run_id: str, cluster: int, k: int = 2) -> list[dict]:
    """Re-cluster one cluster's passages into k sub-clusters with K-means."""
    from sklearn.cluster import KMeans

    from thematicml.clustering import describe_clusters, lsa_embed

    run = get_run_or_404(store, run_id)
    segs = load_segments(store, run_id)
    X = feature_matrix_by_pos(store, run, segs)
    if X is None:
        raise ValueError("Feature artifacts for this run are missing; re-run the analysis to split clusters.")
    members = segs[segs["cluster"] == cluster]
    if len(members) < k * 2:
        raise ValueError("Cluster is too small to split.")
    pos = members["pos"].to_numpy()
    Xs = X[pos]
    Z = lsa_embed(Xs, run["config"]["svd_components"], run["config"]["random_state"])
    sub = KMeans(n_clusters=k, n_init=10, random_state=run["config"]["random_state"]).fit_predict(Z)
    next_id = (store.scalar("SELECT MAX(cluster) FROM clusters WHERE run_id = ?", (run_id,)) or 0) + 1
    new_ids = [cluster] + [next_id + i for i in range(k - 1)]
    mapped = np.array([new_ids[s] for s in sub])
    vocab = load_features(store, run)["features"].vocabulary
    described = describe_clusters(Xs, vocab, mapped, Z, np.arange(len(pos)), 10, 5)
    seg_ids = members["segment_id"].to_numpy()
    with store.connect() as con:
        con.executemany("UPDATE segments SET cluster = ? WHERE id = ?", [(int(c), int(s)) for c, s in zip(mapped, seg_ids)])
        con.execute("DELETE FROM clusters WHERE run_id = ? AND cluster = ?", (run_id, cluster))
        for d in described:
            store.insert("clusters", {
                "run_id": run_id, "cluster": d["cluster"], "label": " / ".join(t["term"] for t in d["terms"][:3]) or f"Cluster {d['cluster']}",
                "size": d["size"], "terms": d["terms"],
                "representative": [{"segment_id": int(seg_ids[r["segment_pos"]]), "similarity": r["similarity"]} for r in d["representative"]],
                "cohesion": d["cohesion"],
            }, con)
    store.log(run["project_id"], "cluster_split", {"run_id": run_id, "cluster": cluster, "into": new_ids})
    return store.query("SELECT * FROM clusters WHERE run_id = ? AND cluster IN (%s)" % ",".join("?" * len(new_ids)), (run_id, *new_ids))
