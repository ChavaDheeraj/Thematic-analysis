/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, download, fmt, qs, type Dataset, type Run } from "../api";
import { CodeDrawer, ThemeDrawer } from "../components/Evidence";
import { GnnDatabaseSection } from "../components/Gnn";
import { Activity, MLMetrics, Quality, Statistics } from "../components/RunDetails";
import { STEPS, ThematicAnalysis, type Step } from "../components/Thematic";
import { DataToolbar } from "../components/Toolbar";
import { Badge, Button, Card, Empty, ErrorBox, Field, Loading, Modal, PageHeader, SectionTitle, Stat, Table, Tabs, Unavailable, cx, inputBase, inputCls } from "../components/ui";
import { useApp, useFetch } from "../state";

const TABS = [
  { id: "files", label: "Files & records" },
  { id: "codes", label: "Thematic analysis" },
  { id: "quality", label: "Quality & statistics" },
  { id: "model", label: "Model metrics" },
  { id: "history", label: "History & storage" },
] as const;
type Tab = (typeof TABS)[number]["id"];
const RUN_TABS: Tab[] = ["codes", "quality", "model"];

export default function DatabasePage() {
  const { project } = useApp();
  if (!project) return <Empty title="No project selected">Create or select a project first.</Empty>;
  return <Database />;
}

function Database() {
  const { project, datasets, dataset, datasetId, allFiles, allRuns, run, refreshData, toast, setDatasetId } = useApp();
  // with all files analysed together, a row of the file list opens that file's records without leaving the combined analysis
  const [viewId, setViewId] = useState<string | null>(null);
  const shown = dataset ?? datasets.find((x) => x.id === viewId) ?? null;
  const [params, setParams] = useSearchParams();
  const tab: Tab = TABS.find((t) => t.id === params.get("tab"))?.id ?? "files";
  const step: Step = STEPS.find((s) => s.id === params.get("step") && s.id !== "map")?.id ?? "cleaned";
  const [recordId, setRecordId] = useState<number | null>(null);
  const [codeId, setCodeId] = useState<number | null>(null);
  const [themeId, setThemeId] = useState<number | null>(null);
  // links from search: ?dataset=…&record=…, ?code=…, ?theme=…
  useEffect(() => {
    const d = params.get("dataset"), r = params.get("record"), c = params.get("code"), t = params.get("theme");
    if (d && d !== datasetId) setDatasetId(d);
    if (r) setRecordId(Number(r));
    if (c) setCodeId(Number(c));
    if (t) setThemeId(Number(t));
  }, [params]); // eslint-disable-line react-hooks/exhaustive-deps

  const ok = run?.status === "completed";
  const [d, dErr, dLoading, reloadD] = useFetch<any>(ok && RUN_TABS.includes(tab) ? `/api/runs/${run!.id}/dashboard` : null);
  const [structure, sErr, , reloadS] = useFetch<any>(ok && tab === "codes" ? `/api/runs/${run!.id}/thematic` : null);
  const reload = () => { reloadD(); reloadS(); };
  const del = async (x: Dataset) => {
    if (!confirm(`Delete “${x.filename}”, its records and every analysis that used it? This cannot be undone.`)) return;
    try { await api.del(`/api/datasets/${x.id}`); await refreshData(); toast("File deleted"); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const closeDrawers = () => { setCodeId(null); setThemeId(null); setParams(tab === "codes" ? { tab, step } : { tab }); };
  const theme = themeId !== null ? d?.themes.table.find((t: any) => t.id === themeId) ?? null : null;

  const runBody = (render: (d: any) => ReactNode) => {
    if (!ok) return <div className="mt-6"><Unavailable reason={run ? `${run.label} is ${run.status}.` : allFiles
      ? "Your files have not been analysed together yet. Use “Analyse all files together” on the Upload & analyse page."
      : "Choose a file with a completed analysis in the bar above."} /></div>;
    if (dErr) return <div className="mt-6"><ErrorBox error={dErr} /></div>;
    if (dLoading && !d) return <Loading />;
    return d ? render(d) : null;
  };

  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader title="Database" desc="Choose all files together or one file, and an analysis; then explore every stored record, code, theme and metric in detail, and export them." />
      <DataToolbar />
      <Tabs tabs={[...TABS]} value={tab} onChange={(t) => setParams({ tab: t })} />
      <div className="[&>:first-child]:mt-6">
        {tab === "files" && (
          <>
            <Files datasets={datasets} active={shown?.id ?? null} allFiles={allFiles} onPick={(id) => (allFiles ? setViewId(id) : setDatasetId(id))} />
            {shown && <Records key={shown.id} ds={shown} onRecord={setRecordId} onDelete={del} />}
          </>
        )}
        {tab === "codes" && runBody((d) => <ThematicAnalysis d={d} structure={structure} structureError={sErr} runId={run!.id} step={step} onCode={setCodeId} onTheme={setThemeId} reload={reload} />)}
        {tab === "quality" && runBody((d) => <><Quality d={d} /><Statistics d={d} /></>)}
        {tab === "model" && (
          <>
            {runBody((d) => <MLMetrics d={d} />)}
            <GnnDatabaseSection projectId={project!.id} />
          </>
        )}
        {tab === "history" && <HistoryTab projectId={project!.id} runs={allRuns} />}
      </div>
      <RecordDrawer recordId={recordId} onClose={() => { setRecordId(null); if (params.get("record")) setParams({ tab }); }} />
      {/* while the file or analysis is switching, d still holds the previous analysis for one render */}
      {ok && d && <CodeDrawer codeId={codeId} onClose={closeDrawers} onChanged={reload} />}
      {ok && d && <ThemeDrawer theme={theme} runId={run!.id} onClose={closeDrawers} onChanged={reload} codes={d.coding.table} themes={d.themes.table} />}
    </div>
  );
}

function Files({ datasets, active, allFiles, onPick }: { datasets: Dataset[]; active: string | null; allFiles: boolean; onPick: (id: string) => void }) {
  return (
    <>
      <SectionTitle id="files" title="Files" desc={allFiles ? `All ${datasets.length} files are analysed together. Click a row to open its records.`
        : "Click a row to open its records; it also becomes the file shown on the Dashboard"} />
      <Card>
        <Table<Dataset> rows={datasets} rowKey={(d) => d.id} onRow={(d) => onPick(d.id)} initialSort={{ key: "uploaded_at", dir: -1 }}
          empty="No files yet."
          cols={[
            { key: "filename", label: "File", render: (d) => (
              <div className="max-w-[260px]">
                <div className={cx("truncate font-medium", d.id === active && "text-accent")} title={d.filename}>{d.filename}</div>
                <div className="font-mono text-[11px] text-muted">{d.id}</div>
              </div>
            ) },
            { key: "format", label: "Format", render: (d) => d.format.toUpperCase() },
            { key: "size_bytes", label: "Size", num: true, render: (d) => <span className="whitespace-nowrap">{fmt.bytes(d.size_bytes)}</span> },
            { key: "uploaded_at", label: "Uploaded at", render: (d) => <span className="whitespace-nowrap">{fmt.date(d.uploaded_at)}</span> },
            { key: "valid", label: "Valid / total", num: true, sort: (d) => d.quality?.valid_records ?? -1, render: (d) => <span className="whitespace-nowrap">{fmt.n(d.quality?.valid_records)} / {fmt.n(d.quality?.total_records ?? d.n_rows)}</span> },
            { key: "missing", label: "Missing", num: true, sort: (d) => d.quality?.missing_text ?? -1, render: (d) => fmt.n(d.quality?.missing_text) },
            { key: "dups", label: "Duplicates", num: true, sort: (d) => d.quality?.duplicate_records ?? -1, render: (d) => fmt.n(d.quality?.duplicate_records) },
            { key: "status", label: "Status", render: (d) => <Badge status={d.status} /> },
            { key: "n_runs", label: "Analyses", num: true },
          ]} />
      </Card>
    </>
  );
}

function HistoryTab({ projectId, runs }: { projectId: string; runs: Run[] }) {
  const [summary, sErr] = useFetch<any>(`/api/projects/${projectId}/database`, [runs.length]);
  const [activity] = useFetch<any[]>(`/api/projects/${projectId}/activity?limit=100`, [runs.length]);
  const s = summary;
  return (
    <>
      <History runs={runs} />
      {activity && <Activity d={{ activity }} />}
      <ExportLog exports={s?.exports ?? []} />
      <SectionTitle id="storage" title="Storage" desc="Counts from the local SQLite database and workspace files" />
      {sErr && <ErrorBox error={sErr} />}
      {!s ? <Loading /> : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Files" value={fmt.n(s.total_uploads)} detail={`${fmt.n(s.source_files)} distinct source files`} />
            <Stat label="Records" value={fmt.n(s.total_records)} detail={`${fmt.n(s.processed_records)} valid for analysis`} />
            <Stat label="Analyses" value={fmt.n(s.analysis_runs)} detail={`${fmt.n(s.failed_runs)} failed · ${fmt.n(s.exported_reports)} reports exported`} />
            <Stat label="Storage used" value={fmt.bytes(s.storage_total_bytes)} detail={`+ shared database ${fmt.bytes(s.storage.database_bytes)}`} />
          </div>
          <div className="mt-2 text-xs text-muted">
            Original files {fmt.bytes(s.storage.original_files_bytes)} · processed data & model artifacts {fmt.bytes(s.storage.model_artifacts_bytes)} · exports {fmt.bytes(s.storage.exports_bytes)}. {s.storage.database_note}
          </div>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------- C
function Records({ ds, onRecord, onDelete }: { ds: Dataset; onRecord: (id: number) => void; onDelete: (d: Dataset) => void }) {
  const { allRuns, dateFrom, dateTo, toast, setDatasetId } = useApp();
  const nav = useNavigate();
  const dsRuns = allRuns.filter((r) => r.status === "completed" && r.dataset_ids.includes(ds.id));
  const [runId, setRunId] = useState<string>(dsRuns[0]?.id ?? "");
  const [f, setF] = useState({ search: "", validity: "all", min_words: "", max_words: "", coded: "all", group: "", sort: "row_index", order: "asc" });
  const [page, setPage] = useState(1);
  const [size, setSize] = useState(50);
  const [search, setSearch] = useState("");
  useEffect(() => { const t = setTimeout(() => { setF((x) => ({ ...x, search })); setPage(1); }, 300); return () => clearTimeout(t); }, [search]);
  const params = { ...f, page, size, run_id: runId || null, date_from: dateFrom || null, date_to: dateTo || null };
  const [data, err, loading] = useFetch<any>(`/api/datasets/${ds.id}/records${qs(params as any)}`);
  const pages = data ? Math.max(1, Math.ceil(data.total / size)) : 1;
  const set = (k: string, v: string) => { setF({ ...f, [k]: v }); setPage(1); };
  const exportUrl = `/api/datasets/${ds.id}/records.csv${qs({ search: f.search, validity: f.validity, min_words: f.min_words, max_words: f.max_words, group: f.group, date_from: dateFrom, date_to: dateTo })}`;
  return (
    <>
      <SectionTitle id="records" title="Records" desc="Original text, exactly as uploaded. Counts of codes and themes come from the selected analysis." />
      <Card actions={<>
        <Button small onClick={() => { setDatasetId(ds.id); nav("/upload"); }} title="Re-map columns or run a new analysis of this file">Analyse this file</Button>
        <Button small onClick={() => download(`/api/datasets/${ds.id}/original`).catch((e) => toast(e.message, "error"))} title="The original file, unchanged">Download original</Button>
        <Button small kind="danger" onClick={() => onDelete(ds)}>Delete file</Button>
      </>} title={ds.filename} subtitle={`${ds.format.toUpperCase()} · ${fmt.bytes(ds.size_bytes)} · uploaded ${fmt.date(ds.uploaded_at)}`}>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
          <div className="col-span-2"><Field label="Search original text or record ID"><input className={inputCls} value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search…" /></Field></div>
          <Field label="Validity"><select className={inputCls} value={f.validity} onChange={(e) => set("validity", e.target.value)}>
            <option value="all">All</option><option value="valid">Valid</option><option value="excluded">Excluded</option><option value="duplicate">Duplicates</option></select></Field>
          <Field label="Min words"><input type="number" min={0} className={inputCls} value={f.min_words} onChange={(e) => set("min_words", e.target.value)} /></Field>
          <Field label="Max words"><input type="number" min={0} className={inputCls} value={f.max_words} onChange={(e) => set("max_words", e.target.value)} /></Field>
          <Field label="Analysis"><select className={inputCls} value={runId} onChange={(e) => { setRunId(e.target.value); setPage(1); }}>
            <option value="">— none —</option>{dsRuns.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}</select></Field>
          <Field label="Coding"><select className={inputCls} value={f.coded} disabled={!runId} onChange={(e) => set("coded", e.target.value)}>
            <option value="all">All</option><option value="coded">Coded</option><option value="uncoded">Uncoded</option></select></Field>
          <Field label="Group"><select className={inputCls} value={f.group} onChange={(e) => set("group", e.target.value)}>
            <option value="">All</option>{(data?.groups ?? []).map((g: string) => <option key={g} value={g}>{g}</option>)}</select></Field>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <Field label="Sort"><select className={cx(inputBase, "w-36")} value={f.sort} onChange={(e) => set("sort", e.target.value)}>
            <option value="row_index">Source order</option><option value="word_count">Word count</option><option value="timestamp">Timestamp</option><option value="record_key">Record ID</option></select></Field>
          <Field label="Order"><select className={cx(inputBase, "w-28")} value={f.order} onChange={(e) => set("order", e.target.value)}><option value="asc">Ascending</option><option value="desc">Descending</option></select></Field>
          <span className="ml-auto self-end text-xs text-muted">{dateFrom || dateTo ? `Date range ${dateFrom || "…"} – ${dateTo || "…"} applied · ` : ""}{data ? `${fmt.n(data.total)} matching records` : ""}</span>
          <div className="self-end"><Button small onClick={() => download(exportUrl).catch((e) => toast(e.message, "error"))}>Export filtered (CSV)</Button></div>
        </div>
        <div className="mt-3">
          {err && <ErrorBox error={err} />}
          {loading && !data && <Loading />}
          {data && (
            <Table<any> rows={data.rows} rowKey={(r) => r.id} onRow={(r) => onRecord(r.id)} maxHeight={560}
              cols={[
                { key: "record_key", label: "Record ID", render: (r) => <span className="block max-w-[200px] truncate font-mono text-xs" title={r.record_key}>{r.record_key}</span> },
                { key: "text", label: "Text", render: (r) => <div className="max-w-[400px] truncate" title={r.text ?? ""}>{r.text || <span className="text-muted">∅ empty</span>}</div> },
                { key: "word_count", label: "Words", num: true },
                ...(runId ? [
                  { key: "segments", label: "Segments", num: true },
                  { key: "codes_assigned", label: "Codes", num: true },
                  { key: "themes_assigned", label: "Themes", num: true },
                  { key: "sentiment", label: "Sentiment", num: true, render: (r: any) => r.sentiment === null || r.sentiment === undefined ? "—" : r.sentiment.toFixed(2) },
                ] : []),
                { key: "timestamp", label: "Timestamp", render: (r) => r.timestamp ? fmt.day(r.timestamp) : r.invalid_timestamp ? <span className="text-warn">invalid</span> : "—" },
                { key: "group_value", label: "Group", render: (r) => r.group_value ?? "—" },
                { key: "is_valid", label: "Status", render: (r) => r.is_valid ? (r.is_duplicate ? <Badge status="needs_revision">duplicate</Badge> : <Badge status="ready">valid</Badge>) : <Badge status="failed">{r.exclusion_reason ?? "excluded"}</Badge> },
              ]} />
          )}
          {data && (
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted">Rows per page</span>
                <select className={cx(inputBase, "w-20")} value={size} onChange={(e) => { setSize(Number(e.target.value)); setPage(1); }}>{[25, 50, 100, 250].map((n) => <option key={n}>{n}</option>)}</select>
              </div>
              <div className="flex items-center gap-2">
                <Button small disabled={page <= 1} onClick={() => setPage(1)}>«</Button>
                <Button small disabled={page <= 1} onClick={() => setPage(page - 1)}>‹ Prev</Button>
                <span className="tabular-nums">Page {page} / {pages}</span>
                <Button small disabled={page >= pages} onClick={() => setPage(page + 1)}>Next ›</Button>
                <Button small disabled={page >= pages} onClick={() => setPage(pages)}>»</Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </>
  );
}

function RecordDrawer({ recordId, onClose }: { recordId: number | null; onClose: () => void }) {
  const { allRuns } = useApp();
  const [runId, setRunId] = useState<string>("");
  const [d, err, loading] = useFetch<any>(recordId ? `/api/records/${recordId}${qs({ run_id: runId || null })}` : null);
  const usable = d ? allRuns.filter((r) => r.status === "completed" && r.dataset_ids.includes(d.record.dataset_id)) : [];
  useEffect(() => { if (d && !runId && usable.length) setRunId(usable[0].id); }, [d]); // eslint-disable-line react-hooks/exhaustive-deps
  const rec = d?.record;
  return (
    <Modal open={recordId !== null} onClose={onClose} title={rec ? `Record ${rec.record_key}` : "Record"} wide>
      {loading && <Loading />}
      {err && <ErrorBox error={err} />}
      {rec && (
        <div className="space-y-4 text-sm">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs md:grid-cols-4">
            <div><dt className="text-muted">Source</dt><dd>{rec.filename} · row {rec.row_index + 1}</dd></div>
            <div><dt className="text-muted">Words</dt><dd>{rec.word_count}</dd></div>
            <div><dt className="text-muted">Timestamp</dt><dd>{rec.timestamp ?? (rec.invalid_timestamp ? "invalid" : "—")}</dd></div>
            <div><dt className="text-muted">Group / label</dt><dd>{rec.group_value ?? "—"} / {rec.label ?? "—"}</dd></div>
            <div><dt className="text-muted">Status</dt><dd>{rec.is_valid ? "valid" : `excluded: ${rec.exclusion_reason}`}{rec.is_duplicate ? " · duplicate" : ""}</dd></div>
            {Object.entries(rec.metadata ?? {}).map(([k, v]) => <div key={k}><dt className="text-muted">{k}</dt><dd>{String(v ?? "—")}</dd></div>)}
          </dl>
          <div>
            <div className="mb-1 text-xs font-medium text-ink2">Original text (unchanged)</div>
            <div className="whitespace-pre-wrap rounded-md border border-line bg-surface p-3 leading-relaxed">{rec.text || <span className="text-muted">∅ empty</span>}</div>
          </div>
          <Field label="Readings from analysis">
            <select className={cx(inputCls, "max-w-sm")} value={runId} onChange={(e) => setRunId(e.target.value)}>
              <option value="">— none —</option>{usable.map((r) => <option key={r.id} value={r.id}>{r.label}</option>)}</select>
          </Field>
          {runId && d.segments.length === 0 && <Unavailable reason="This record was not analysed in the selected run (for example, it was excluded)." />}
          {d.segments.length > 0 && (
            <>
              {d.themes?.length > 0 && <div className="text-xs"><span className="font-medium text-ink2">Themes: </span>{d.themes.map((t: any) => <span key={t.id} className="mr-2">{t.label} <Badge status={t.status} /></span>)}</div>}
              <table className="w-full text-xs">
                <thead className="text-left text-ink2"><tr><th className="py-1 pr-2 font-medium">#</th><th className="pr-2 font-medium">Segment</th><th className="pr-2 font-medium">Chars</th><th className="pr-2 font-medium">Cluster</th><th className="pr-2 font-medium">Topic</th><th className="pr-2 font-medium">Sentiment</th><th className="font-medium">Codes</th></tr></thead>
                <tbody>{d.segments.map((s: any) => (
                  <tr key={s.id} className="border-t border-line align-top">
                    <td className="py-1.5 pr-2 tabular-nums">{s.seg_index + 1}</td>
                    <td className="max-w-md pr-2">{s.text}<div className={s.verified ? "text-good" : "text-bad"}>{s.verified ? "✓ verbatim" : "✕ mismatch"}</div></td>
                    <td className="pr-2 tabular-nums text-muted">{s.char_start}–{s.char_end}</td>
                    <td className="pr-2 tabular-nums">{s.cluster ?? "—"}</td>
                    <td className="pr-2 tabular-nums">{s.topic === null || s.topic < 0 ? "—" : `${s.topic} (${s.topic_weight})`}</td>
                    <td className="pr-2">{s.sentiment_label ? `${s.sentiment_label} (${s.sentiment?.toFixed(2)})` : "—"}</td>
                    <td>{s.codes.length ? s.codes.map((c: any) => <div key={`${c.id}-${c.char_start}`}><span className={cx(c.status === "rejected" && "line-through text-muted")}>{c.label}</span> <span className="text-muted">· {c.method}{c.matched_text ? ` · “${c.matched_text}”` : ""}{c.negated ? " · negation" : ""}</span></div>) : <span className="text-muted">uncoded</span>}</td>
                  </tr>))}</tbody>
              </table>
            </>
          )}
        </div>
      )}
    </Modal>
  );
}

// ---------------------------------------------------------------- D
function History({ runs }: { runs: Run[] }) {
  const { datasets, refreshData, toast, openRun } = useApp();
  const nav = useNavigate();
  const retry = async (r: Run) => { try { await api.post(`/api/runs/${r.id}/retry`); await refreshData(); toast("Retry queued"); } catch (e: any) { toast(e.message, "error"); } };
  const del = async (r: Run) => {
    if (!confirm(`Delete analysis “${r.label}” and its results? Uploaded data is kept.`)) return;
    try { await api.del(`/api/runs/${r.id}`); await refreshData(); toast("Analysis deleted"); } catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <>
      <SectionTitle id="history" title="Analysis history" desc="Every analysis in this project with its configuration, status and timings" />
      <Card>
        <Table<Run> rows={runs} rowKey={(r) => r.id} empty="No analyses yet." initialSort={{ key: "created_at", dir: -1 }}
          cols={[
            { key: "id", label: "Run ID", render: (r) => <span className="font-mono text-xs">{r.id}</span> },
            { key: "label", label: "Label" },
            { key: "datasets", label: "File", render: (r) => <span className="text-xs">{r.dataset_ids.length > 1 ? <span title={r.dataset_ids.map((id) => datasets.find((d) => d.id === id)?.filename ?? id).join(", ")}>Combined · {r.dataset_ids.length} files</span> : datasets.find((d) => d.id === r.dataset_ids[0])?.filename ?? r.dataset_ids[0]}</span> },
            { key: "config", label: "Configuration", render: (r) => <span className="text-xs text-ink2">{r.config.segmentation} · {r.config.coding_method}/{r.config.n_codes} · {r.config.clustering_method}{r.config.n_clusters ? `(${r.config.n_clusters})` : "(auto)"} · {r.config.topic_method}({r.config.n_topics}) · {r.config.graph_method} · seed {r.config.random_state}</span> },
            { key: "status", label: "Status", render: (r) => <div><Badge status={r.status} />{r.failed_stage && <div className="text-[11px] text-bad">at {r.failed_stage}</div>}</div> },
            { key: "created_at", label: "Started", render: (r) => fmt.date(r.created_at) },
            { key: "duration_s", label: "Duration", num: true, render: (r) => fmt.sec(r.duration_s) },
            { key: "peak_memory_mb", label: "Peak MB", num: true, render: (r) => fmt.n(r.peak_memory_mb) },
            {
              key: "actions", label: "Actions", render: (r) => (
                <div className="flex gap-1">
                  {r.status === "completed" && <Button small kind="ghost" onClick={() => { openRun(r.id); nav("/dashboard"); }}>Open</Button>}
                  {(r.status === "failed" || r.status === "interrupted") && <Button small onClick={() => retry(r)}>Retry</Button>}
                  {!["queued", "running"].includes(r.status) && <Button small kind="danger" onClick={() => del(r)}>Delete</Button>}
                </div>
              ),
            },
          ]} />
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- F
function ExportLog({ exports }: { exports: any[] }) {
  return (
    <>
      <SectionTitle id="exports" title="Exported reports" desc="Files written to the project's exports folder" />
      <Card>
        <Table<any> rows={exports} rowKey={(r) => r.id} maxHeight={300} empty="Nothing exported yet."
          cols={[
            { key: "created_at", label: "Exported", render: (r) => fmt.date(r.created_at) },
            { key: "format", label: "Format", render: (r) => r.format.toUpperCase() },
            { key: "run_id", label: "Run", render: (r) => <span className="font-mono text-xs">{r.run_id ?? "project"}</span> },
            { key: "size_bytes", label: "Size", num: true, render: (r) => fmt.bytes(r.size_bytes) },
            { key: "path", label: "Path", render: (r) => <span className="break-all font-mono text-[11px] text-muted">{r.path}</span> },
          ]} />
      </Card>
    </>
  );
}
