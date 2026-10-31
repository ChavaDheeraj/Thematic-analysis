"""Label-free hyperparameter sweeps on one corpus (optional; never touches the workspace).

Every variant runs the analysis stages in memory exactly as the pipeline does
(segmentation, TF-IDF, coding, clustering, topics, statistics, graph, themes)
and reports quality measures that need no ground truth:

* topics: NPMI coherence of each topic's top terms counted over records
  (Lau et al., 2014), topic diversity (Dieng et al., 2020), their product
  ("topic quality"), and stability as the best-match Jaccard overlap of top
  terms between seeds;
* clusters: cosine silhouette and the adjusted Rand index between seeds;
* code graph: density, modularity, communities, largest-community share,
  community adjusted Rand index between seeds, and the link-prediction baselines;
* codes: coverage and the code list itself, for reading.

Use it to choose settings; then run the chosen configuration through the app or
``Project.analyze`` so results are stored and traceable.

    python tools/tune.py spec.json --out results.jsonl --workers 4

``spec.json``::

    {"files": ["t.data/*.html", "t.data/New folder/*"], "exclude": ["t.data/New folder/download (12).htm"],
     "mapping": {"text_col": "text", "id_col": "respondent_id", "group_col": "source_kind"},
     "base": {"n_topics": 10, "graph_method": "leiden"},
     "variants": [{"name": "topics-12", "overrides": {"n_topics": 12}}],
     "seeds": [42, 43, 44]}
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from thematicml import pipeline as P
from thematicml.clustering import ClusteringError, cluster_segments
from thematicml.config import AnalysisConfig, ColumnMapping
from thematicml.data import load_file, normalise_records, segment_records
from thematicml.graph import detect_communities
from thematicml.topics import discover_topics


# ---------------------------------------------------------------------- corpus

def resolve_files(patterns: list[str], exclude: list[str] = ()) -> list[Path]:
    skip = {Path(e).resolve() for e in exclude}
    out: list[Path] = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            p = Path(f)
            if p.is_file() and p.resolve() not in skip and p not in out:
                out.append(p)
    return out


def build_corpus(files: list[Path], mapping: ColumnMapping) -> tuple[pd.DataFrame, list[dict]]:
    """Records of all files as the pipeline sees them (valid rows only), plus a per-file report.

    Files whose parsed text is identical to an earlier file are skipped (saved twice).
    """
    seen: dict[str, str] = {}
    frames, report = [], []
    for i, path in enumerate(files):
        loaded = load_file(path)
        df = loaded.frame
        digest = hashlib.sha1("\u0000".join(df[mapping.text_col].fillna("").astype(str)).encode("utf-8")).hexdigest()
        entry = {"file": path.as_posix(), "rows": int(len(df)), "warnings": loaded.warnings}
        if digest in seen:
            report.append({**entry, "skipped": f"identical to {seen[digest]}"})
            continue
        seen[digest] = path.as_posix()
        # optional columns a file does not have (e.g. no timestamp on a PDF) are left unmapped for that file
        absent = {k: None for k in ("id_col", "time_col", "group_col", "label_col") if getattr(mapping, k) not in (None, *df.columns)}
        m = mapping.model_copy(update={**absent, "metadata_cols": [c for c in mapping.metadata_cols if c in df.columns]})
        recs, quality = normalise_records(df, m)
        recs = recs[recs["is_valid"]].copy()
        recs["dataset_id"] = f"ds{i:03d}"
        recs["filename"] = path.name
        # generated keys are only unique within one file, as in the pipeline
        gen = recs["record_key"].str.startswith("row-")
        recs.loc[gen, "record_key"] = recs.loc[gen, "dataset_id"] + ":" + recs.loc[gen, "record_key"]
        frames.append(recs)
        report.append({**entry, "valid": int(len(recs)), "quality": {k: v for k, v in quality.to_dict().items() if k != "percentages"}})
    records = pd.concat(frames, ignore_index=True)
    repeated = P.repeated_passages(records)  # as the pipeline does for a run over several uploads
    records = records[~repeated].reset_index(drop=True)
    records.insert(0, "record_id", np.arange(1, len(records) + 1))
    report.append({"cross_file_repeats_removed": int(repeated.sum())})
    return records, report


def segments_for(records: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    segs = segment_records(records, cfg.segmentation, cfg.min_segment_words)
    return segs.merge(records[["record_id", "dataset_id", "record_key", "timestamp", "group_value"]], on="record_id", how="left")


# ---------------------------------------------------------------------- measures

def record_binary(segments: pd.DataFrame, counts: sp.csr_matrix) -> sp.csc_matrix:
    """Record x term presence matrix aggregated from segment counts."""
    codes, uniques = pd.factorize(segments["record_id"])
    A = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniques), len(codes)))
    return ((A @ (counts > 0).astype(np.float64)) > 0).astype(np.float64).tocsc()


def npmi(B: sp.csc_matrix, term_ids: list[int]) -> float | None:
    """Mean normalised PMI over term pairs; a pair that never co-occurs scores -1."""
    if len(term_ids) < 2:
        return None
    D = B.shape[0]
    cols = B[:, term_ids]
    df = np.asarray(cols.sum(axis=0)).ravel()
    co = (cols.T @ cols).toarray()
    vals = []
    for i, j in itertools.combinations(range(len(term_ids)), 2):
        if co[i, j] == 0 or df[i] == 0 or df[j] == 0:
            vals.append(-1.0)
            continue
        p_ij = co[i, j] / D
        if p_ij >= 1.0:
            vals.append(1.0)
            continue
        vals.append(math.log(p_ij / ((df[i] / D) * (df[j] / D))) / -math.log(p_ij))
    return float(np.mean(vals))


def matched_jaccard(a: list[set], b: list[set]) -> float:
    """Mean Jaccard overlap of topics after optimal one-to-one matching."""
    M = np.array([[len(x & y) / len(x | y) if x | y else 1.0 for y in b] for x in a])
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].mean())


def mean_pairwise(values: list, fn) -> float | None:
    pairs = list(itertools.combinations(values, 2))
    return round(float(np.mean([fn(x, y) for x, y in pairs])), 4) if pairs else None


def _topic_sets(res: dict, vocab) -> list[set]:
    index = {t: i for i, t in enumerate(vocab)}
    return [{index[t] for t in topic["terms"]} for topic in res["topics"]]


def evaluate(records: pd.DataFrame, cfg: AnalysisConfig, seeds: list[int] = (), stages=P.STAGE_NAMES) -> dict:
    t0 = time.perf_counter()
    st: dict = {"summary": {}, "warnings": [], "records": records, "segments": segments_for(records, cfg)}
    out: dict = {"records": int(len(records)), "respondents": int(records["record_key"].nunique()), "segments": int(len(st["segments"]))}
    P._stage_features(None, None, cfg, st)
    P._stage_coding(None, None, cfg, st)
    feats, segs = st["features"], st["segments"]
    out["vocabulary"] = int(feats.matrix.shape[1])
    codes, assigns = st["codes"], st["assignments"]
    support = assigns.groupby("code_key")["segment_pos"].nunique() if len(assigns) else pd.Series(dtype=int)
    out["codes"] = {
        "n": int(len(codes)), "by_method": codes["method"].value_counts().to_dict() if len(codes) else {},
        "coverage_pct": round(100.0 * assigns["segment_pos"].nunique() / max(len(segs), 1), 2) if len(assigns) else 0.0,
        "list": [[str(lab), int(support.get(k, 0)), str(m)] for k, lab, m in zip(codes["code_key"], codes["label"], codes["method"])],
    }

    if "clustering" in stages:
        P._stage_clustering(None, None, cfg, st)
        cl = st["summary"]["clustering"]
        out["clustering"] = {k: cl.get(k) for k in ("available", "n_clusters", "silhouette_cosine", "davies_bouldin", "calinski_harabasz", "k_candidates", "reason")}
        if seeds and cl.get("available"):
            labels = []
            for s in seeds:
                try:
                    labels.append(cluster_segments(feats.matrix, cfg.model_copy(update={"random_state": s}))["labels"])
                except ClusteringError:
                    pass
            out["clustering"]["seed_ari"] = mean_pairwise(labels, adjusted_rand_score)

    if "topics" in stages:
        P._stage_topics(None, None, cfg, st)
        res = st["topics"]
        if res:
            B = record_binary(segs, st["counts"])
            sets = _topic_sets(res, feats.vocabulary)
            per_topic = [npmi(B, sorted(s)) for s in sets]
            mean_npmi = float(np.mean([v for v in per_topic if v is not None]))
            m = res["metrics"]
            out["topics"] = {"n": m["n_topics"], "diversity": m["topic_diversity"], "umass": m["mean_coherence_umass"],
                             "npmi": round(mean_npmi, 4), "quality": round(mean_npmi * (m["topic_diversity"] or 0), 4),
                             "no_topic_segments": m["segments_without_dominant_topic"],
                             "list": [[t["label"], t["size"], round(v, 3) if v is not None else None] for t, v in zip(res["topics"], per_topic)]}
            if seeds:
                runs = [_topic_sets(discover_topics(feats.matrix, st["counts"], feats.vocabulary, cfg.topic_method, cfg.n_topics,
                                                    cfg.n_top_terms, cfg.topic_min_weight, s), feats.vocabulary) for s in seeds]
                out["topics"]["seed_jaccard"] = mean_pairwise(runs, matched_jaccard)

    if "graph" in stages:
        if "statistics" in stages:
            P._stage_statistics(None, None, cfg, st)
        else:  # the graph stage does not need statistics; keep the sweep fast
            st["sentiment"] = None
        P._stage_graph(None, None, cfg, st)
        g = st["summary"]["graph"]
        sizes = g["communities"].get("sizes") or []
        out["graph"] = {"nodes": g["nodes"], "edges": g["edges"], "density": g["density"],
                        "modularity": g["communities"].get("modularity"),
                        "communities_2plus": g["communities"].get("n_multi_node_communities"),
                        "largest_share": round(sizes[0] / g["nodes"], 3) if sizes and g["nodes"] else None,
                        "link_baseline_auc": (g.get("link_prediction_baseline") or {}).get("roc_auc")}
        if seeds:
            G = st["nx_graph"]
            nodes = sorted(G.nodes())
            mems = [detect_communities(G, cfg.graph_method, cfg.resolution, s)[0] for s in seeds]
            out["graph"]["seed_ari"] = mean_pairwise([[m[n] for n in nodes] for m in mems], adjusted_rand_score)
        if "themes" in stages:
            if st.get("sentiment") is None:
                st["sentiment"] = None
            P._stage_themes(None, None, cfg, st)
            out["themes"] = [[t["label"], len(t["code_keys"])] for t in st["themes"]]
    out["warnings"] = st["warnings"]
    out["runtime_s"] = round(time.perf_counter() - t0, 2)
    return out


# ---------------------------------------------------------------------- driver

def _run_variant(args) -> dict:
    corpus_path, base, variant, seeds, stages = args
    records = joblib.load(corpus_path)
    cfg = AnalysisConfig(**{**base, **variant.get("overrides", {})})
    try:
        res = evaluate(records, cfg, seeds, stages)
        return {"name": variant["name"], "overrides": variant.get("overrides", {}), **res}
    except Exception as exc:  # report the failure and keep the sweep going
        return {"name": variant["name"], "overrides": variant.get("overrides", {}), "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True, help="results as JSON lines (one variant per line)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = out.with_suffix(".corpus.joblib")
    if spec.get("corpus"):
        corpus_path = Path(spec["corpus"])
    else:
        files = resolve_files(spec["files"], spec.get("exclude", []))
        records, report = build_corpus(files, ColumnMapping(**spec["mapping"]))
        joblib.dump(records, corpus_path)
        out.with_suffix(".corpus.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"corpus: {len(records)} records from {sum('file' in r and 'skipped' not in r for r in report)} files -> {corpus_path}", file=sys.stderr)
    stages = tuple(spec.get("stages") or P.STAGE_NAMES)
    jobs = [(corpus_path, spec.get("base", {}), v, spec.get("seeds", []), stages) for v in spec["variants"]]
    with out.open("w", encoding="utf-8") as fh:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for row in pool.map(_run_variant, jobs):
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                    print(f"{row['name']}: {'error' if 'error' in row else 'ok'} ({row.get('runtime_s', '-')} s)", file=sys.stderr)
        else:
            for job in jobs:
                row = _run_variant(job)
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                print(f"{row['name']}: {'error' if 'error' in row else 'ok'} ({row.get('runtime_s', '-')} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
