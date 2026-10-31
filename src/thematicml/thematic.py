"""The seven steps of a thematic analysis, read back from one stored run.

1 cleaned dataset · 2 notes and recurring ideas · 3 initial coding · 4 codebook ·
5 categories · 6 themes and sub-themes · 7 thematic map

Like ``results``, these are live views: researcher edits show up at once. What
the machine proposes here (recurring ideas, category suggestions, sub-themes,
links between themes) is labelled as a proposal and is only stored when the
researcher accepts it.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from thematicml.analytics import benjamini_hochberg
from thematicml.analytics.percentages import percentage
from thematicml.results import (
    _segment_rows,
    code_table,
    feature_matrix_by_pos,
    get_run_or_404,
    load_assignments,
    load_segments,
    quotation,
    theme_table,
    topic_view,
)
from thematicml.storage.database import Store
from thematicml.themes import template_label

# a code joins its nearest category only above this cosine similarity; on the t.data corpus the
# matches below it read as guesses (Home -> Safety strategies, Work -> Motivations)
MIN_CATEGORY_SIMILARITY = 0.30
# codes left over are grouped with each other at this average cosine distance
GROUP_DISTANCE = 0.8
# theme pairs must differ from chance by this much as well as being significant, so large files do not turn tiny effects into links
MIN_LIFT_CHANGE = 0.05
ALPHA = 0.05
QUOTE_WORDS = (12, 60)


# ---------------------------------------------------------------------- 1 cleaned dataset

def _cleaned_counts(store: Store, run: dict) -> dict:
    """How the uploaded records became the records this run analysed."""
    ids = run["dataset_ids"]
    q = ",".join("?" * len(ids))
    by_reason = {r["reason"]: r["n"] for r in store.query(
        f"SELECT exclusion_reason AS reason, COUNT(*) AS n FROM records WHERE dataset_id IN ({q}) GROUP BY exclusion_reason", ids)}
    kept = store.scalar("SELECT COUNT(DISTINCT record_id) FROM segments WHERE run_id = ?", (run["id"],)) or 0
    flagged = store.scalar(
        "SELECT COUNT(*) FROM records WHERE is_duplicate = 1 AND id IN (SELECT record_id FROM segments WHERE run_id = ?)", (run["id"],)) or 0
    repeated = ((run.get("metrics") or {}).get("dataset") or {}).get("cross_dataset_duplicates", 0)
    no_passage = max(0, by_reason.get(None, 0) - repeated - kept)
    removed = [
        {"label": "Empty text", "records": by_reason.get("missing_text", 0)},
        {"label": "Duplicates", "records": by_reason.get("duplicate", 0)},
        {"label": "Other language", "records": by_reason.get("non_english", 0)},
        {"label": "Repeated in another file", "records": repeated},
        {"label": "No usable passage", "records": no_passage},
    ]
    return {"parsed": sum(by_reason.values()), "kept": kept, "flagged_duplicates": flagged,
            "removed": [r for r in removed if r["records"]], "removed_total": sum(r["records"] for r in removed)}


def cleaned_view(store: Store, run_id: str, page: int = 1, size: int = 25, search: str = "") -> dict:
    run = get_run_or_404(store, run_id)
    where, params = "", [run_id]
    if search:
        where = "WHERE r.text LIKE ? OR r.record_key LIKE ?"
        params += [f"%{search}%", f"%{search}%"]
    join = ("FROM records r JOIN datasets d ON d.id = r.dataset_id JOIN "
            "(SELECT record_id, COUNT(*) AS passages FROM segments WHERE run_id = ? GROUP BY record_id) s ON s.record_id = r.id")
    total = store.scalar(f"SELECT COUNT(*) {join} {where}", params)
    rows = store.query(
        f"SELECT r.id, r.record_key, r.text, r.word_count, r.is_duplicate, d.filename AS source_file, s.passages {join} {where} "
        f"ORDER BY r.dataset_id, r.row_index LIMIT ? OFFSET ?", params + [size, (page - 1) * size])
    segments = store.scalar("SELECT COUNT(*) FROM segments WHERE run_id = ?", (run_id,)) or 0
    return {**_cleaned_counts(store, run), "segments": segments, "total": total, "page": page, "size": size, "rows": rows}


def cleaned_frame(store: Store, run_id: str) -> pd.DataFrame:
    """Exactly the records the analysis read, in source order."""
    return pd.DataFrame(store.query(
        "SELECT d.filename AS source_file, r.row_index + 1 AS source_row, r.record_key, r.text, r.word_count, r.timestamp, r.group_value, "
        "r.is_duplicate AS flagged_duplicate FROM records r JOIN datasets d ON d.id = r.dataset_id "
        "WHERE r.id IN (SELECT record_id FROM segments WHERE run_id = ?) ORDER BY r.dataset_id, r.row_index", (run_id,)))


# ---------------------------------------------------------------------- 2 notes

def _idea_passage(members: pd.DataFrame, terms: list[str]):
    """A readable passage for a topic: the one using most of its terms among its strongest passages."""
    best, best_hits = None, -1
    for row in members.sort_values("topic_weight", ascending=False).head(60).itertuples():
        if not QUOTE_WORDS[0] <= len(row.text.split()) <= QUOTE_WORDS[1]:
            continue
        low = row.text.lower()
        hits = sum(t in low for t in terms)
        if hits > best_hits:
            best, best_hits = row, hits
    return best


def recurring_ideas(store: Store, run_id: str) -> dict:
    """Topics, read as a first list of ideas that keep coming back in the data."""
    tv = topic_view(store, run_id)
    segs = load_segments(store, run_id)
    picks = {}
    for t in tv["topics"]:
        row = _idea_passage(segs[segs["topic"] == t["topic"]], t["terms"][:8])
        if row is not None:
            picks[t["topic"]] = int(row.segment_id)
    rows = _segment_rows(store, list(picks.values()))
    ideas = []
    for t in sorted(tv["topics"], key=lambda t: -t["share"]["numerator"]):
        members = segs[segs["topic"] == t["topic"]]
        if members.empty:
            continue
        ideas.append({
            "topic": t["topic"], "terms": t["terms"][:6], "share": t["share"], "files": int(members["filename"].nunique()),
            "quote": quotation(store, rows[picks[t["topic"]]]) if t["topic"] in picks else None,
        })
    return {"available": bool(tv["metrics"].get("available")), "reason": tv["metrics"].get("reason"), "ideas": ideas,
            "n_files": int(segs["filename"].nunique()),
            "note": "Drafted by the topic model from repeated words. Read them against the data; they are prompts for notes, not findings."}


def notes_view(store: Store, run_id: str) -> dict:
    get_run_or_404(store, run_id)
    return {
        "memos": store.query("SELECT id, note, created_at FROM annotations WHERE run_id = ? AND segment_id IS NULL AND theme_id IS NULL "
                             "AND code_id IS NULL ORDER BY id DESC", (run_id,)),
        "codes": store.query("SELECT id, label, notes FROM codes WHERE run_id = ? AND notes IS NOT NULL AND notes != '' ORDER BY label", (run_id,)),
        "themes": store.query("SELECT id, label, notes FROM themes WHERE run_id = ? AND notes IS NOT NULL AND notes != '' ORDER BY label", (run_id,)),
    }


# ---------------------------------------------------------------------- 3 initial coding

def coded_statements(store: Store, run_id: str, page: int = 1, size: int = 25, search: str = "", code_id: int | None = None) -> dict:
    """Meaningful statements (passages holding at least one code) with the codes assigned to each."""
    get_run_or_404(store, run_id)
    join = ("FROM segments s JOIN assignments a ON a.segment_id = s.id JOIN codes c ON c.id = a.code_id "
            "JOIN records r ON r.id = s.record_id JOIN datasets d ON d.id = r.dataset_id")
    where, params = ["s.run_id = ?", "c.status IN ('candidate', 'approved')"], [run_id]
    if search:
        where.append("s.text LIKE ?")
        params.append(f"%{search}%")
    if code_id is not None:
        where.append("a.code_id = ?")
        params.append(code_id)
    cond = " AND ".join(where)
    total = store.scalar(f"SELECT COUNT(DISTINCT s.id) {join} WHERE {cond}", params)
    rows = store.query(
        f"SELECT DISTINCT s.id, s.pos, s.text, r.record_key, d.filename AS source_file {join} WHERE {cond} ORDER BY s.pos LIMIT ? OFFSET ?",
        params + [size, (page - 1) * size])
    by_seg: dict[int, list[dict]] = {}
    if rows:
        q = ",".join("?" * len(rows))
        for a in store.query(
            f"SELECT a.segment_id, c.id, c.label, a.matched_text, a.negated FROM assignments a JOIN codes c ON c.id = a.code_id "
            f"WHERE c.status IN ('candidate', 'approved') AND a.segment_id IN ({q}) ORDER BY c.label", [r["id"] for r in rows]):
            by_seg.setdefault(a.pop("segment_id"), []).append(a)
    return {
        "total": total, "page": page, "size": size,
        "coded": store.scalar("SELECT COUNT(DISTINCT s.id) FROM segments s JOIN assignments a ON a.segment_id = s.id JOIN codes c ON c.id = a.code_id "
                              "WHERE s.run_id = ? AND c.status IN ('candidate', 'approved')", (run_id,)),
        "segments": store.scalar("SELECT COUNT(*) FROM segments WHERE run_id = ?", (run_id,)),
        "rows": [{**r, "codes": by_seg.get(r["id"], [])} for r in rows],
    }


# ---------------------------------------------------------------------- 4 codebook

def _quotations(store: Store, run: dict, per_code: int = 2) -> dict[int, list[dict]]:
    """Verbatim example quotations per code: its most typical readable passages, from different records."""
    segs = load_segments(store, run["id"])
    X = feature_matrix_by_pos(store, run, segs)
    Xn = normalize(X) if X is not None else None
    seg_ids = segs["segment_id"].to_numpy()
    records = segs["record_id"].to_numpy()
    words = segs["text"].str.split().str.len().to_numpy()
    chosen: dict[int, list[int]] = {}
    for cid, g in load_assignments(store, run["id"]).groupby("code_id"):
        pos = g["segment_pos"].to_numpy()
        typical = np.zeros(len(pos))
        if Xn is not None:
            centroid = normalize(np.asarray(Xn[pos].mean(axis=0)))
            typical = np.asarray(Xn[pos] @ centroid.T).ravel()
        unreadable = (words[pos] < QUOTE_WORDS[0]) | (words[pos] > QUOTE_WORDS[1])
        # a negation nearby can reverse what the code says, so those passages come last
        picks, seen = [], set()
        for i in np.lexsort((-typical, unreadable, g["negated"].to_numpy(dtype=bool))):
            if records[pos[i]] not in seen:
                seen.add(records[pos[i]])
                picks.append(int(seg_ids[pos[i]]))
            if len(picks) == per_code:
                break
        chosen[int(cid)] = picks
    rows = _segment_rows(store, [s for ids in chosen.values() for s in ids])
    return {cid: [quotation(store, rows[s]) for s in ids if s in rows] for cid, ids in chosen.items()}


def codebook(store: Store, run_id: str) -> list[dict]:
    run = get_run_or_404(store, run_id)
    codes = code_table(store, run_id, include_inactive=False)
    quotes = _quotations(store, run)
    suggested = suggest_categories(store, run, codes)
    return [{**c, "quotations": quotes.get(c["id"], []), "suggested_category": suggested.get(c["id"])} for c in codes]


# ---------------------------------------------------------------------- 5 categories

def suggest_categories(store: Store, run: dict, codes: list[dict], segs: pd.DataFrame | None = None, assigns: pd.DataFrame | None = None) -> dict[int, str]:
    """Propose a category for each code that has none. Nothing is stored here.

    Codes are compared by the mean TF-IDF vector of their passages. A code joins the
    nearest existing category when it is close enough; the rest are grouped with each
    other under a template name. Codes too unlike anything stay uncategorised.
    """
    todo = [c for c in codes if not c["category"]]
    if not todo:
        return {}
    segs = load_segments(store, run["id"]) if segs is None else segs
    X = feature_matrix_by_pos(store, run, segs)
    if X is None:
        return {}
    assigns = load_assignments(store, run["id"]) if assigns is None else assigns
    positions = {k: g["segment_pos"].unique() for k, g in assigns.groupby("code_id")}
    Xn = normalize(X)
    vec = {c["id"]: normalize(np.asarray(Xn[positions[c["id"]]].mean(axis=0)))[0] for c in codes if c["id"] in positions}

    out: dict[int, str] = {}
    members: dict[str, list] = {}
    for c in codes:
        if c["category"] and c["id"] in vec:
            members.setdefault(c["category"], []).append(vec[c["id"]])
    names = sorted(members)
    if names:
        C = normalize(np.vstack([np.mean(members[n], axis=0) for n in names]))
        for c in todo:
            if c["id"] in vec:
                sims = C @ vec[c["id"]]
                if sims.max() >= MIN_CATEGORY_SIMILARITY:
                    out[c["id"]] = names[int(sims.argmax())]

    rest = [c for c in todo if c["id"] not in out and c["id"] in vec]
    if len(rest) >= 2:
        M = np.vstack([vec[c["id"]] for c in rest])
        dist = np.clip(1 - M @ M.T, 0, 2)
        np.fill_diagonal(dist, 0)
        groups = AgglomerativeClustering(n_clusters=None, distance_threshold=GROUP_DISTANCE, metric="precomputed", linkage="average").fit_predict(dist)
        for g in set(groups.tolist()):
            group = sorted((c for c, x in zip(rest, groups) if x == g), key=lambda c: -c["segments"])
            if len(group) >= 2:
                name = template_label("{1} & {2}", [c["label"] for c in group])
                out.update({c["id"]: name for c in group})
    return out


def _category_rows(codes: list[dict], suggested: dict[int, str], pos_of: dict, n_seg: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str | None, list[dict]] = {}
    for c in codes:
        groups.setdefault(c["category"] or suggested.get(c["id"]), []).append(c)
    def brief(c):
        return {"id": c["id"], "label": c["label"], "segments": c["segments"], "suggested": not c["category"]}

    rows = []
    for name, members in groups.items():
        if name is None:
            continue
        members = sorted(members, key=lambda c: -c["segments"])
        pos = set().union(*(pos_of.get(c["id"], set()) for c in members))
        rows.append({
            "name": name, "suggested": all(not c["category"] for c in members), "codes": [brief(c) for c in members],
            "n_suggested": sum(1 for c in members if not c["category"]), "segments": len(pos),
            "share": percentage(len(pos), n_seg, "passages with any code in this category / analysed passages", "segment"),
        })
    rows.sort(key=lambda r: -r["segments"])
    return rows, [brief(c) for c in sorted(groups.get(None, []), key=lambda c: -c["segments"])]


# ---------------------------------------------------------------------- 6 themes and sub-themes

def _subthemes(theme: dict, children: list[dict], category_of: dict, codes: dict, pos_of: dict) -> tuple[list[dict], list[dict]]:
    """Sub-themes of a theme, and its codes left outside them.

    A researcher's nested themes come first. The other codes form one candidate
    sub-theme per category with at least two codes in this theme.
    """
    def pack(ids):
        rows = [{"id": c, "label": codes[c]["label"], "segments": codes[c]["segments"]} for c in ids if c in codes]
        return sorted(rows, key=lambda c: -c["segments"])

    def reach(ids):
        return len(set().union(*(pos_of.get(c, set()) for c in ids))) if ids else 0

    subs = [{"id": ch["id"], "label": ch["label"], "status": ch["status"], "source": "researcher", "codes": pack(ch["code_ids"]),
             "segments": reach(ch["code_ids"])} for ch in children]
    taken = {c for ch in children if ch["status"] != "rejected" for c in ch["code_ids"]}
    groups: dict[str | None, list[int]] = {}
    for cid in theme["code_ids"]:
        if cid not in taken:
            groups.setdefault(category_of.get(cid), []).append(cid)
    other: list[int] = []
    derived = []
    for name, ids in groups.items():
        if name and len(ids) >= 2:
            derived.append({"id": None, "label": name, "status": theme["status"], "source": "category", "codes": pack(ids), "segments": reach(ids)})
        else:
            other += ids
    derived.sort(key=lambda s: -s["segments"])
    if not subs and len(derived) == 1 and not other:
        # one category covering the whole theme adds nothing to the theme itself
        return [], derived[0]["codes"]
    return subs + derived, pack(other)


# ---------------------------------------------------------------------- 7 thematic map

def _theme_links(store: Store, run: dict, themes: list[dict], codes: dict, pos_of: dict, seg_record: dict) -> dict:
    """How often each pair of themes meets, against what chance alone would give.

    Two themes are linked when they share more passages (or records, following the
    run's co-occurrence unit) than expected from their sizes: two-sided hypergeometric
    test, Benjamini–Hochberg adjusted across the pairs. Passages of one document are
    not independent, so this is a guide to reading the map, not an inferential claim.
    """
    level = run["config"]["cooccurrence_level"]
    to_unit = (lambda p: seg_record[p]) if level == "record" else (lambda p: p)
    n_units = len(set(seg_record.values())) if level == "record" else len(seg_record)
    pos = {t["id"]: set().union(*(pos_of.get(c, set()) for c in t["code_ids"])) if t["code_ids"] else set() for t in themes}
    units = {k: {to_unit(p) for p in v} for k, v in pos.items()}
    rows = []
    for a, b in itertools.combinations(themes, 2):
        A, B = units[a["id"]], units[b["id"]]
        if not A or not B:
            continue
        n = len(A & B)
        expected = len(A) * len(B) / n_units
        p = min(1.0, 2 * min(float(hypergeom.sf(n - 1, n_units, len(A), len(B))), float(hypergeom.cdf(n, n_units, len(A), len(B)))))
        rows.append({"a": a["id"], "b": b["id"], "a_label": a["label"], "b_label": b["label"], "shared": n, "expected": round(expected, 1),
                     "lift": round(n / expected, 2) if expected else None, "p": p})
    for r, adj in zip(rows, benjamini_hochberg([r["p"] for r in rows])):
        r["p_adjusted"] = round(adj, 5)
        del r["p"]
        lift = r["lift"] or 0
        r["relation"] = "together" if adj < ALPHA and lift >= 1 + MIN_LIFT_CHANGE else "apart" if adj < ALPHA and lift <= 1 - MIN_LIFT_CHANGE else "chance"
    by_id = {t["id"]: t for t in themes}
    linked = [r for r in rows if r["relation"] == "together"]
    picks = {}
    for r in linked:
        ids_a, ids_b = by_id[r["a"]]["code_ids"], by_id[r["b"]]["code_ids"]
        pairs = [(len(pos_of.get(x, set()) & pos_of.get(y, set())), x, y) for x in ids_a for y in ids_b if x != y and x in codes and y in codes]
        top = max(pairs, key=lambda t: t[0], default=(0, None, None))
        r["via"] = [codes[top[1]]["label"], codes[top[2]]["label"]] if top[0] else None
        hits = {}
        for code_ids, side in ((ids_a, 0), (ids_b, 1)):
            for c in code_ids:
                for p in pos_of.get(c, set()):
                    hits.setdefault(p, [0, 0])[side] += 1
        both = sorted((p for p, h in hits.items() if h[0] and h[1]), key=lambda p: (-min(hits[p]), -sum(hits[p]), p))
        picks[(r["a"], r["b"])] = both[:40]
    wanted = {p for ps in picks.values() for p in ps}
    id_of = {row["pos"]: row["id"] for row in store.query("SELECT id, pos FROM segments WHERE run_id = ?", (run["id"],)) if row["pos"] in wanted}
    seg_rows = _segment_rows(store, list(id_of.values()))
    for r in linked:
        cands = [seg_rows[id_of[p]] for p in picks[(r["a"], r["b"])] if p in id_of and id_of[p] in seg_rows]
        fit = [s for s in cands if QUOTE_WORDS[0] <= len(s["text"].split()) <= QUOTE_WORDS[1]] or cands
        r["quote"] = quotation(store, fit[0]) if fit else None
    rows.sort(key=lambda r: -(r["lift"] or 0))
    return {"unit": "passage" if level != "record" else "record", "n_units": n_units, "rows": rows,
            "rule": f"Lines join themes that share more {'records' if level == 'record' else 'passages'} than their sizes would give by chance "
                    f"(two-sided hypergeometric test, Benjamini–Hochberg p < {ALPHA}, at least {round(100 * MIN_LIFT_CHANGE)}% from chance). "
                    "Passages of one document are not independent, so read this as a guide, not a statistical claim."}


def thematic_structure(store: Store, run_id: str) -> dict:
    """Categories, themes with sub-themes, links between themes, and the count for each step."""
    run = get_run_or_404(store, run_id)
    codes = code_table(store, run_id, include_inactive=False)
    code_by_id = {c["id"]: c for c in codes}
    assigns = load_assignments(store, run_id)
    segs = load_segments(store, run_id)
    pos_of = {k: set(g["segment_pos"].tolist()) for k, g in assigns.groupby("code_id")}
    seg_record = dict(zip(segs["pos"].tolist(), segs["record_id"].tolist()))
    suggested = suggest_categories(store, run, codes, segs, assigns)
    category_of = {c["id"]: c["category"] or suggested.get(c["id"]) for c in codes}
    categories, uncategorised = _category_rows(codes, suggested, pos_of, len(segs))

    all_themes = theme_table(store, run_id)
    children: dict[int, list[dict]] = {}
    for t in all_themes:
        if t.get("parent_id"):
            children.setdefault(t["parent_id"], []).append(t)
    top = sorted((t for t in all_themes if not t.get("parent_id")), key=lambda t: -t["segments"])
    themes = []
    for rank, t in enumerate(top):
        subs, other = _subthemes(t, children.get(t["id"], []), category_of, code_by_id, pos_of)
        themes.append({**t, "rank": rank, "subthemes": subs, "other_codes": other})
    live = [t for t in themes if t["status"] != "rejected"]
    links = _theme_links(store, run, live, code_by_id, pos_of, seg_record)
    notes = notes_view(store, run_id)
    tv = (run.get("metrics") or {}).get("topics") or {}
    return {
        "path": {
            "records": _cleaned_counts(store, run)["kept"],
            "ideas": tv.get("n_topics", 0) if tv.get("available") else 0,
            "notes": len(notes["memos"]) + len(notes["codes"]) + len(notes["themes"]),
            "statements": int(assigns["segment_pos"].nunique()),
            "codes": len(codes),
            "categories": len(categories),
            "themes": len(live),
            "final": sum(1 for t in live if t["status"] == "approved"),
            "subthemes": sum(1 for t in live for x in t["subthemes"] if x["status"] != "rejected"),
            "links": sum(1 for r in links["rows"] if r["relation"] == "together"),
        },
        "categories": categories,
        "uncategorised": uncategorised,
        "themes": themes,
        "links": links,
    }
