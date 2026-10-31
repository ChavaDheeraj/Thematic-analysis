/* eslint-disable @typescript-eslint/no-explicit-any */
// Detailed tables and numbers for one analysis, shown on the Database page (the Dashboard keeps the visual summary).
import { useMemo, useState } from "react";
import { api, download, fmt } from "../api";
import { chrome, series } from "../colors";
import { useApp, useFetch } from "../state";
import { Chart, axis, baseOption, lineStyle } from "./Chart";
import { GnnDashboardCard } from "./Gnn";
import { Badge, Button, Card, ErrorBox, PctStat, SectionTitle, Stat, Table, Unavailable, cx, inputBase } from "./ui";

// ---------------------------------------------------------------- B
export function Quality({ d }: { d: any }) {
  const q = d.quality;
  return (
    <>
      <SectionTitle id="quality" title="Data quality" desc={`Record-level, relative to all parsed records. Duplicate policy: ${q.duplicate_policy.join(", ")} · language policy: ${(q.language_policy ?? ["keep"]).join(", ").replace("exclude_non_english", "exclude non-English")}${q.cross_dataset_duplicates ? ` · ${fmt.n(q.cross_dataset_duplicates)} passages repeated across uploads analysed once` : ""}`} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <PctStat label="Valid records" p={q.valid} />
        <PctStat label="Missing text" p={q.missing_text} />
        <PctStat label="Duplicate records" p={q.duplicates} />
        <PctStat label="Excluded records" p={q.excluded} />
        <PctStat label="Invalid metadata" p={q.invalid_metadata} />
        {q.non_english && <PctStat label="Non-English" p={q.non_english} />}
        <Stat label="Awaiting analysis" value={fmt.n(q.awaiting_analysis)} detail="valid records in never-analysed uploads" />
        <PctStat label="Uncoded segments" p={q.uncoded_segments} />
      </div>
      {q.warnings.length > 0 && <ul className="mt-2 list-disc pl-5 text-xs text-warn">{q.warnings.slice(0, 6).map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
    </>
  );
}

// ---------------------------------------------------------------- C
export function CodingStats({ d }: { d: any }) {
  const c = d.coding;
  return (
    <>
      <SectionTitle id="coding" title="Initial coding" desc="Meaningful statements found in the data and the codes given to them: TF-IDF phrases, codebook rules and (optionally) a supervised classifier" />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <Stat label="Unique codes" value={fmt.n(c.unique_codes)} />
        <Stat label="Code assignments" value={fmt.n(c.total_assignments)} detail="code × segment" />
        <PctStat label="Records with ≥1 code" p={c.coded_records} />
        <PctStat label="Segments with ≥1 code" p={c.coded_segments} />
        <Stat label="Codes per respondent" value={fmt.n(c.codes_per_respondent?.mean, 2)} detail={`median ${fmt.n(c.codes_per_respondent?.median)} · max ${fmt.n(c.codes_per_respondent?.max)}`} />
        <Stat label="Awaiting review" value={fmt.n(c.awaiting_review)} detail={`${c.approved} approved · ${c.rejected} rejected`} />
        <Stat label="Merged / split" value={`${c.merged} / ${c.split}`} />
        <PctStat label="Review progress" p={c.review_progress} />
      </div>
    </>
  );
}

export function Codebook({ d, runId, onOpen, reload }: { d: any; runId: string; onOpen: (id: number) => void; reload: () => void }) {
  const { toast } = useApp();
  const c = d.coding;
  const [book, bookErr] = useFetch<any[]>(`/api/runs/${runId}/codebook`, [d]);
  const extra = useMemo(() => new Map((book ?? []).map((b) => [b.id, b])), [book]);
  const [filter, setFilter] = useState("active");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<number[]>([]);
  const [mergeLabel, setMergeLabel] = useState("");
  const rows = c.table.filter((r: any) =>
    (filter === "all" || (filter === "active" ? ["candidate", "approved"].includes(r.status) : r.status === filter)) &&
    (!q || r.label.toLowerCase().includes(q.toLowerCase())));

  const act = async (fn: () => Promise<unknown>, msg: string) => {
    try { await fn(); toast(msg, "success"); setSel([]); setMergeLabel(""); reload(); } catch (e: any) { toast(e.message, "error"); }
  };
  const bulk = (status: string) => act(() => Promise.all(sel.map((id) => api.patch(`/api/codes/${id}`, { status }))), `${sel.length} codes ${status}`);

  return (
    <>
      <SectionTitle id="codebook" title="Codebook" desc="Every code with its definition, category and a verbatim quotation. The quotation is the code's most typical readable passage." />
      <Card title={`${fmt.n(c.unique_codes)} codes`} subtitle="Click a row to read all its passages, edit, approve, reject or split"
        actions={<>
          <input className={cx(inputBase, "w-40")} placeholder="Filter codes" value={q} onChange={(e) => setQ(e.target.value)} />
          <select aria-label="Code status" className={cx(inputBase, "w-32")} value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="active">Active</option><option value="candidate">Candidate</option><option value="approved">Approved</option>
            <option value="rejected">Rejected</option><option value="merged">Merged</option><option value="all">All</option>
          </select>
          <Button small onClick={() => download(`/api/runs/${runId}/export/codebook_csv`).catch((e) => toast(e.message, "error"))}>Codebook (CSV)</Button>
        </>}>
        {bookErr && <div className="mb-2"><ErrorBox error={bookErr} /></div>}
        {sel.length > 0 && (
          <div className="mb-2 flex flex-wrap items-center gap-2 rounded-md bg-raised p-2 text-sm">
            <span>{sel.length} selected</span>
            <Button small kind="primary" onClick={() => bulk("approved")}>✓ Approve</Button>
            <Button small kind="danger" onClick={() => bulk("rejected")}>✕ Reject</Button>
            <input className={cx(inputBase, "w-44")} placeholder="Merged label (optional)" value={mergeLabel} onChange={(e) => setMergeLabel(e.target.value)} />
            <Button small disabled={sel.length < 2} onClick={() => act(() => api.post("/api/codes/merge", { code_ids: sel, label: mergeLabel || undefined }), "Codes merged")}>Merge into first</Button>
            <Button small kind="ghost" onClick={() => setSel([])}>Clear</Button>
          </div>
        )}
        <Table<any>
          rows={rows} rowKey={(r) => r.id} maxHeight={600} onRow={(r) => onOpen(r.id)} initialSort={{ key: "segments", dir: -1 }}
          cols={[
            { key: "sel", label: "", width: "28px", render: (r) => <input type="checkbox" aria-label={`Select ${r.label}`} checked={sel.includes(r.id)} onClick={(e) => e.stopPropagation()} onChange={(e) => setSel(e.target.checked ? [...sel, r.id] : sel.filter((x) => x !== r.id))} /> },
            { key: "label", label: "Code", render: (r) => <div className="max-w-[230px]"><div className="font-medium">{r.label}</div>{r.description && <div className="line-clamp-2 text-[11px] leading-snug text-muted" title={r.description}>{r.description}</div>}</div> },
            { key: "category", label: "Category", render: (r) => {
              const s = extra.get(r.id)?.suggested_category;
              return r.category ? <span className="text-xs text-ink2">{r.category}</span> : s ? <span className="text-xs italic text-muted" title="Suggested by text similarity, not yet assigned">{s}?</span> : <span className="text-muted">—</span>;
            } },
            { key: "quote", label: "Quotation", sort: (r) => r.label, render: (r) => {
              const q = extra.get(r.id)?.quotations?.[0];
              return q ? <div className="line-clamp-3 min-w-[260px] max-w-md text-xs italic leading-relaxed text-ink2" title={q.source_file}>“{q.text}”</div> : <span className="text-muted">{book || bookErr ? "—" : "…"}</span>;
            } },
            { key: "segments", label: "Passages", num: true, render: (r) => fmt.n(r.segments) },
            { key: "respondent_pct", label: "% respondents", num: true, render: (r) => <span title={r.respondent_ci95?.[0] !== null ? `95% CI ${r.respondent_ci95?.join("–")}%` : ""}>{fmt.pct(r.respondent_pct)}</span> },
            { key: "status", label: "Status", render: (r) => <Badge status={r.status} /> },
          ]}
        />
        <p className="mt-2 text-[11px] text-muted">% respondents = respondents with code / {fmt.n(c.denominators.respondents)} respondents (Wilson 95% CI on hover). A category ending in ? is a suggestion.</p>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- D
export function ThemeStats({ d, s, runId, reload }: { d: any; s: any; runId: string; reload: () => void }) {
  const { toast } = useApp();
  const t = d.themes;
  const [newLabel, setNewLabel] = useState("");
  const create = async () => {
    try { await api.post(`/api/runs/${runId}/themes`, { label: newLabel, code_ids: [] }); setNewLabel(""); toast("Theme created — open it to add codes", "success"); reload(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const p = s.path;
  return (
    <>
      <SectionTitle id="themes" title="Final themes and sub-themes" desc="A theme is a candidate until you read its evidence and approve it; then it is final. Candidate sub-themes group a theme's codes by category. For your own, add a theme and set its position to sub-theme." />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Final themes" value={`${fmt.n(p.final)} of ${fmt.n(p.themes)}`} detail="researcher-approved" />
        <Stat label="Candidate themes" value={fmt.n(t.candidate)} detail="machine-generated, awaiting review" />
        <Stat label="Sub-themes" value={fmt.n(p.subthemes)} detail="in themes that are not rejected" />
        <PctStat label="Codes in final themes" p={t.codes_in_approved_themes} />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input className={cx(inputBase, "w-56")} placeholder="New theme label" value={newLabel} onChange={(e) => setNewLabel(e.target.value)} />
        <Button small disabled={!newLabel.trim()} onClick={create}>＋ Add theme</Button>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- E
export function Statistics({ d }: { d: any }) {
  const s = d.statistics;
  const tmp = s.temporal;
  const tOpt = useMemo(() => () => ({
    ...baseOption(),
    grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" },
    legend: { ...baseOption().legend, type: "scroll" },
    xAxis: axis("category", { data: tmp.periods ?? [], boundaryGap: false }),
    yAxis: axis("value", { axisLabel: { color: chrome().muted, formatter: "{value}%" } }),
    series: (tmp.codes ?? []).slice(0, 6).map((c: any, i: number) => ({ ...lineStyle(series(i)), name: c.label, data: c.share_pct })),
  }), [tmp]);
  const gc = s.group_comparison;
  return (
    <>
      <SectionTitle id="statistics" title="Co-occurrence, groups and time" desc="Descriptive associations; none of these establish causes" />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Temporal trends" subtitle={tmp.available ? `Share of segments per ${({ D: "day", W: "week", M: "month", Q: "quarter" } as any)[tmp.frequency]} · ${fmt.pct(tmp.coverage_pct)} of records timestamped` : undefined}>
          {!tmp.available ? <Unavailable reason={tmp.reason} /> : (
            <>
              <Chart option={tOpt} height={240} name="temporal-codes" />
              <Table<any> rows={tmp.codes} rowKey={(r) => r.label} maxHeight={200}
                cols={[
                  { key: "label", label: "Code" },
                  { key: "trend", label: "Mann–Kendall", render: (r) => r.trend.available === false ? <span className="text-muted">—</span> : <span>{r.trend.trend}{r.trend.p_value !== undefined ? <span className="text-muted"> (p={r.trend.p_value})</span> : null}</span> },
                  { key: "slope", label: "Sen's slope", num: true, render: (r) => fmt.n(r.trend.sens_slope, 3) },
                  { key: "cp", label: "Change points", render: (r) => r.change_points.join(", ") || "—" },
                ]} />
            </>
          )}
        </Card>
        <Card title="Code co-occurrence" subtitle={`Top pairs counted per ${d.ml.config.cooccurrence_level}`}>
          <Table<any> rows={s.code_cooccurrence} rowKey={(r) => `${r.key_a}-${r.key_b}`} maxHeight={300} initialSort={{ key: "count", dir: -1 }}
            cols={[{ key: "code_a", label: "Code A" }, { key: "code_b", label: "Code B" }, { key: "count", label: "Co-occurrences", num: true }]} />
        </Card>
        <Card title="Theme co-occurrence" subtitle="Records containing codes from both themes">
          {s.theme_cooccurrence.length ? (
            <Table<any> rows={s.theme_cooccurrence} rowKey={(r) => `${r.theme_a}-${r.theme_b}`} maxHeight={300}
              cols={[{ key: "theme_a", label: "Theme A" }, { key: "theme_b", label: "Theme B" }, { key: "records", label: "Records", num: true }]} />
          ) : <Unavailable reason="Needs at least two themes." />}
        </Card>
      </div>
      <Card className="mt-4" title="Group comparisons" subtitle={gc.available ? `Code prevalence by group (record level). ${gc.caveat}` : undefined}>
        {!gc.available ? <Unavailable reason={gc.reason} /> : (
          <>
            <div className="mb-2 text-xs text-ink2">Groups: {Object.entries(gc.groups).map(([g, n]) => `${g} (n=${n})`).join(" · ")}{gc.excluded_small_groups.length ? ` · excluded small groups: ${gc.excluded_small_groups.join(", ")}` : ""}</div>
            <Table<any> rows={gc.results} rowKey={(r) => String(r.code_key)} maxHeight={360} initialSort={{ key: "p_adjusted_bh", dir: 1 }}
              cols={[
                { key: "label", label: "Code" },
                ...Object.keys(gc.groups).map((g) => ({ key: `g-${g}`, label: g, num: true, sort: (r: any) => r.groups[g]?.pct ?? -1, render: (r: any) => r.groups[g] ? <span title={`${r.groups[g].with_code}/${r.groups[g].records}`}>{fmt.pct(r.groups[g].pct)}</span> : "—" })),
                { key: "test", label: "Test", render: (r) => r.test ?? "—" },
                { key: "p_value", label: "p", num: true, render: (r) => r.p_value ?? "—" },
                { key: "p_adjusted_bh", label: "p (BH)", num: true, render: (r) => r.p_adjusted_bh !== undefined ? <span className={cx(r.p_adjusted_bh < 0.05 && "font-semibold")}>{r.p_adjusted_bh}</span> : "—" },
                { key: "cramers_v", label: "Cramér's V", num: true, render: (r) => r.cramers_v ?? "—" },
                { key: "note", label: "Assumptions", render: (r) => <span className="text-[11px] text-muted">{r.note ?? ""}</span> },
              ]} />
            <p className="mt-2 text-[11px] text-muted">p (BH) = Benjamini–Hochberg adjusted across the codes tested. Fisher's exact test is used for 2×2 tables with small expected counts.</p>
          </>
        )}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- F
export function MLMetrics({ d }: { d: any }) {
  const m = d.ml;
  const f = m.features, cl = m.clustering, tp = m.topics, g = m.graph, rt = m.runtime, sup = m.coding?.supervised;
  const kOpt = useMemo(() => () => ({
    ...baseOption(),
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" },
    xAxis: axis("category", { data: (cl.k_candidates ?? []).map((k: any) => `k=${k.k}`) }),
    yAxis: axis("value", { name: "silhouette", nameTextStyle: { color: chrome().muted, fontSize: 11 } }),
    series: [{ ...lineStyle(series(0)), showSymbol: true, data: (cl.k_candidates ?? []).map((k: any) => k.silhouette) }],
  }), [cl]);
  const row = (k: string, v: React.ReactNode) => <div className="flex items-baseline justify-between gap-4"><dt className="text-muted">{k}</dt><dd className="text-right tabular-nums">{v}</dd></div>;
  return (
    <>
      <SectionTitle id="ml" title="ML performance and experiment metrics" desc={m.accuracy_note} />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Features (fitted TF-IDF)">
          <dl className="space-y-1 text-sm">
            {row("Vocabulary size", fmt.n(f.vocabulary_size))}
            {row("Matrix", f.matrix_shape ? `${fmt.n(f.matrix_shape[0])} × ${fmt.n(f.matrix_shape[1])}` : "—")}
            {row("Density", f.density !== undefined ? fmt.pct(100 * f.density, 2) : "—")}
            {row("N-grams", f.settings?.ngram_range?.join("–"))}
            {row("Empty segments", fmt.n(f.empty_documents))}
          </dl>
        </Card>
        <Card title="Clustering (unsupervised)">
          {!cl.available ? <Unavailable reason={cl.reason} /> : (
            <dl className="space-y-1 text-sm">
              {row("Algorithm", cl.algorithm)}
              {row("Clusters", fmt.n(cl.n_clusters))}
              {row("k selection", cl.k_selection)}
              {row("Silhouette (cosine)", fmt.n(cl.silhouette_cosine, 3))}
              {row("Davies–Bouldin", fmt.n(cl.davies_bouldin, 3))}
              {row("Unassigned", `${fmt.n(cl.unassigned_segments)} (${fmt.pct(cl.unassigned_pct)})`)}
            </dl>
          )}
        </Card>
        <Card title="Topic model">
          {!tp.available ? <Unavailable reason={tp.reason} /> : (
            <dl className="space-y-1 text-sm">
              {row("Method", tp.method?.toUpperCase())}
              {row("Topics", fmt.n(tp.n_topics))}
              {row("Mean UMass coherence", fmt.n(tp.mean_coherence_umass, 3))}
              {row("Topic diversity", fmt.n(tp.topic_diversity, 3))}
              {tp.reconstruction_err !== undefined && row("Reconstruction error", fmt.n(tp.reconstruction_err, 3))}
              {tp.perplexity !== undefined && row("Perplexity", fmt.n(tp.perplexity, 1))}
              {row("No dominant topic", fmt.n(tp.segments_without_dominant_topic))}
            </dl>
          )}
        </Card>
        <Card title="Graph & runtime">
          <dl className="space-y-1 text-sm">
            {row("Nodes / edges", `${fmt.n(g.nodes)} / ${fmt.n(g.edges)}`)}
            {row("Communities", `${fmt.n(g.communities?.n_multi_node_communities)} (${g.communities?.method})`)}
            {row("Modularity", fmt.n(g.communities?.modularity, 3))}
            {row("Link-pred. AUC (Adamic–Adar)", g.link_prediction_baseline?.available ? fmt.n(g.link_prediction_baseline.roc_auc.adamic_adar, 3) : "—")}
            {row("Processing time", fmt.sec(rt.total_s))}
            {row("Peak memory", rt.peak_memory_mb ? `${fmt.n(rt.peak_memory_mb)} MB` : "—")}
            {row("Version / seed", `${m.package_version} / ${rt.random_state}`)}
          </dl>
        </Card>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Supervised classification" subtitle="Shown only with researcher labels and a held-out test split">
          {!sup?.available || sup.accuracy === undefined ? <Unavailable reason={sup?.available ? sup.note : sup?.reason} /> : (
            <>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                <Stat label="Accuracy (test)" value={fmt.n(sup.accuracy, 3)} detail={`majority-class baseline ${fmt.n(sup.majority_baseline_accuracy, 3)}`} />
                <Stat label="Macro precision" value={fmt.n(sup.macro_precision, 3)} />
                <Stat label="Macro recall" value={fmt.n(sup.macro_recall, 3)} />
                <Stat label="Macro F1" value={fmt.n(sup.macro_f1, 3)} />
              </div>
              <p className="mt-2 text-xs text-muted">{sup.n_labelled} labelled records · single hold-out split: train {sup.split.train} / test {sup.split.test}, grouped by {sup.split.grouped_by} to prevent leakage; TF-IDF re-fitted on training records only.</p>
              {sup.cross_validation && (
                <p className="mt-1 text-sm"><span className="font-medium">{sup.cross_validation.folds}-fold grouped cross-validation:</span> accuracy {fmt.n(sup.cross_validation.accuracy_mean, 3)} ± {fmt.n(sup.cross_validation.accuracy_sd, 3)} · macro F1 {fmt.n(sup.cross_validation.macro_f1_mean, 3)} ± {fmt.n(sup.cross_validation.macro_f1_sd, 3)} <span className="text-xs text-muted">(more reliable than one split)</span></p>
              )}
              {sup.per_class && (
                <Table<any> rows={sup.per_class} rowKey={(r) => r.label}
                  cols={[{ key: "label", label: "Class" }, { key: "precision", label: "Precision", num: true }, { key: "recall", label: "Recall", num: true }, { key: "f1", label: "F1", num: true }, { key: "support", label: "Support", num: true }]} />
              )}
              {sup.confusion_matrix && (
                <div className="mt-3 overflow-x-auto text-xs">
                  <div className="mb-1 font-medium text-ink2">Confusion matrix (rows = true, columns = predicted)</div>
                  <table className="tabular-nums"><thead><tr><th />{sup.confusion_matrix.labels.map((l: string) => <th key={l} className="px-2 text-right font-medium text-ink2">{l}</th>)}</tr></thead>
                    <tbody>{sup.confusion_matrix.matrix.map((r: number[], i: number) => <tr key={i} className="border-t border-line"><th className="pr-2 text-left font-medium text-ink2">{sup.confusion_matrix.labels[i]}</th>{r.map((v, j) => <td key={j} className={cx("px-2 text-right", i === j && "font-semibold")}>{v}</td>)}</tr>)}</tbody></table>
                </div>
              )}
            </>
          )}
        </Card>
        <Card title="Cluster count selection" subtitle={cl.k_selection === "fixed" ? "k was fixed in the configuration" : "Silhouette for each candidate k; the highest was selected"}>
          {cl.available && (cl.k_candidates?.length ?? 0) > 1 ? <Chart option={kOpt} height={200} name="k-selection" /> : <Unavailable reason={cl.available ? "k was fixed, so no candidates were compared." : cl.reason} />}
          <div className="mt-2 text-xs text-ink2">Representation: {cl.representation ?? "—"}</div>
        </Card>
      </div>
      <div className="mt-4"><GnnDashboardCard runId={d.run.id} /></div>
      <Card className="mt-4" title="Experiment comparison" subtitle="Completed analyses of the same file. More clusters or higher modularity does not by itself mean a better model.">
        <Table<any> rows={m.experiments} rowKey={(r) => r.run_id}
          cols={[
            { key: "label", label: "Run", render: (r) => <span className={cx(r.current && "font-semibold")}>{r.label}{r.current ? " (current)" : ""}</span> },
            { key: "clustering", label: "Clustering" }, { key: "topics", label: "Topics" }, { key: "graph", label: "Graph" },
            { key: "n_clusters", label: "Clusters", num: true, render: (r) => fmt.n(r.n_clusters) },
            { key: "silhouette", label: "Silhouette", num: true, render: (r) => fmt.n(r.silhouette, 3) },
            { key: "coherence", label: "UMass", num: true, render: (r) => fmt.n(r.coherence, 3) },
            { key: "communities", label: "Communities", num: true, render: (r) => fmt.n(r.communities) },
            { key: "modularity", label: "Modularity", num: true, render: (r) => fmt.n(r.modularity, 3) },
            { key: "duration_s", label: "Runtime", num: true, render: (r) => fmt.sec(r.duration_s) },
          ]} />
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- G
const KIND: Record<string, string> = {
  dataset_uploaded: "Dataset uploaded", upload_failed: "Upload failed", dataset_validated: "Dataset validated", analysis_queued: "Analysis queued",
  analysis_completed: "Analysis completed", analysis_failed: "Analysis failed", analysis_retried: "Analysis retried", code_reviewed: "Code reviewed",
  codes_merged: "Codes merged", code_split: "Code split", code_created: "Code created", theme_approved: "Theme approved", theme_reviewed: "Theme reviewed",
  theme_created: "Theme created", report_exported: "Report exported", quotation_verified: "Quotation verified", annotation_added: "Annotation added",
  cluster_updated: "Cluster updated", clusters_merged: "Clusters merged", cluster_split: "Cluster split", dataset_deleted: "Dataset deleted",
  run_deleted: "Run deleted", project_created: "Project created", records_exported: "Records exported", note_deleted: "Note deleted",
};
function describe(a: any): string {
  const x = a.detail ?? {};
  switch (a.kind) {
    case "dataset_uploaded": return `${x.filename} · ${x.rows} rows`;
    case "analysis_completed": return `${x.run_id} · ${fmt.sec(x.duration_s)}`;
    case "analysis_failed": return `${x.run_id} · stage ${x.stage}: ${x.error}`;
    case "report_exported": return `${x.format?.toUpperCase()} · ${x.run_id}`;
    case "theme_approved": return x.label ?? "";
    case "codes_merged": return (x.labels ?? []).join(" + ");
    default: return x.filename ?? x.label ?? x.run_id ?? "";
  }
}
export function Activity({ d }: { d: any }) {
  return (
    <>
      <SectionTitle id="activity" title="Recent activity" />
      <Card>
        <Table<any> rows={d.activity} rowKey={(r) => r.id} maxHeight={300}
          cols={[
            { key: "kind", label: "Activity", render: (r) => <span className={cx(r.kind.includes("failed") && "text-bad")}>{KIND[r.kind] ?? r.kind}</span> },
            { key: "info", label: "Information", render: (r) => <span className="text-ink2">{describe(r)}</span> },
            { key: "ts", label: "When", render: (r) => fmt.date(r.ts) },
          ]} />
      </Card>
    </>
  );
}
