import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, download, fmt } from "../api";
import { ALL_FILES, useApp } from "../state";
import { Icon, ICONS } from "./icons";
import { Badge, Button, cx, inputBase } from "./ui";

/** Database controls: which file and analysis to show, the date filter, search and exports. */
export function DataToolbar() {
  const { project, datasets, datasetId, setDatasetId, runs, run, setRunId, dateFrom, dateTo, setDateRange, activeRuns } = useApp();
  return (
    <div className="mb-6 rounded-xl border border-line bg-surface px-4 py-3" style={{ boxShadow: "var(--shadow-sm)" }}>
      <div className="flex flex-wrap items-end gap-3">
        <Ctl label="File">
          <select aria-label="File" className={cx(inputBase, "w-56 py-1.5")} value={datasetId ?? ""} disabled={!project}
            onChange={(e) => setDatasetId(e.target.value)}>
            {!datasets.length && <option value="">No files yet</option>}
            {datasets.length > 1 && <option value={ALL_FILES}>All files ({datasets.length})</option>}
            {datasets.length > 1 && <option disabled>──────────</option>}
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.filename}</option>)}
          </select>
        </Ctl>
        <Ctl label="Analysis">
          <select aria-label="Analysis" className={cx(inputBase, "w-56 py-1.5")} value={run?.id ?? ""} onChange={(e) => setRunId(e.target.value || null)} disabled={!runs.length}>
            {!runs.length && <option value="">{datasetId === ALL_FILES ? "No analysis of all files together" : "Not analysed yet"}</option>}
            {runs.map((r) => <option key={r.id} value={r.id}>{r.label}{r.dataset_ids.length > 1 ? ` · ${r.dataset_ids.length} files` : ""} · {r.status}{r.finished_at ? ` · ${fmt.day(r.finished_at)}` : ""}</option>)}
          </select>
        </Ctl>
        <Ctl label="Date range">
          <div className="flex items-center gap-1.5">
            <input type="date" className={cx(inputBase, "w-[8.5rem] py-1.5")} value={dateFrom} onChange={(e) => setDateRange(e.target.value, dateTo)} title="Filters records and temporal views (timestamped data only)" />
            <span className="text-xs text-muted">–</span>
            <input type="date" className={cx(inputBase, "w-[8.5rem] py-1.5")} value={dateTo} onChange={(e) => setDateRange(dateFrom, e.target.value)} />
            {(dateFrom || dateTo) && <Button small kind="ghost" onClick={() => setDateRange("", "")}>Clear</Button>}
          </div>
        </Ctl>
        <div className="ml-auto flex flex-wrap items-end gap-2">
          <Search />
          {activeRuns.length > 0 && <div className="flex h-[34px] items-center"><Badge status="running">{activeRuns.length} running</Badge></div>}
          <ExportMenu />
        </div>
      </div>
    </div>
  );
}

function Ctl({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-muted">{label}</div>
      {children}
    </div>
  );
}

function Search() {
  const { project, openRun, setDatasetId } = useApp();
  const [q, setQ] = useState("");
  const [res, setRes] = useState<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [open, setOpen] = useState(false);
  const nav = useNavigate();
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!project || q.trim().length < 2) { setRes(null); return; }
    const t = setTimeout(() => api.get(`/api/search?project_id=${project.id}&q=${encodeURIComponent(q.trim())}`).then(setRes).catch(() => setRes(null)), 250);
    return () => clearTimeout(t);
  }, [q, project]);
  useEffect(() => {
    const h = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const go = (path: string, runId?: string, datasetId?: string) => {
    if (runId) openRun(runId);
    else if (datasetId) setDatasetId(datasetId);
    setOpen(false);
    nav(path);
  };
  const total = res ? res.datasets.length + res.codes.length + res.themes.length + res.records.length : 0;
  return (
    <div className="relative" ref={box}>
      <Ctl label="Search">
        <div className="relative">
          <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted">
            <Icon path={ICONS.search} size={13} />
          </span>
          <input
            className={cx(inputBase, "w-44 py-1.5 pl-8")}
            placeholder="Files, codes, themes…"
            value={q}
            onChange={(e) => { setQ(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            disabled={!project}
          />
        </div>
      </Ctl>
      {open && res && (
        <div className="absolute right-0 z-40 mt-1.5 max-h-[70vh] w-[28rem] max-w-[calc(100vw-2.5rem)] overflow-y-auto rounded-xl border border-line bg-surface p-2 text-sm" style={{ boxShadow: "var(--shadow-xl)" }}>
          {total === 0 && <div className="px-3 py-6 text-center text-muted">No matches found.</div>}
          {(["datasets", "codes", "themes", "records"] as const).map((k) => {
            const title = k === "datasets" ? "Files" : k.charAt(0).toUpperCase() + k.slice(1);
            return res[k].length ? (
              <div key={k} className="mb-1">
                <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted">{title}</div>
                {res[k].map((x: any) => ( // eslint-disable-line @typescript-eslint/no-explicit-any
                  <button key={`${k}-${x.id}`}
                    className="block w-full rounded-lg px-3 py-2 text-left transition-colors hover:bg-raised"
                    onClick={() => k === "datasets" ? go(`/database?tab=files&dataset=${x.id}`, undefined, x.id)
                      : k === "records" ? go(`/database?tab=files&dataset=${x.dataset_id}&record=${x.id}`, undefined, x.dataset_id)
                      : go(`/database?tab=codes&${k === "codes" ? "step=codebook&code" : "step=themes&theme"}=${x.id}`, x.run_id)}>
                    {k === "records"
                      ? <><span className="text-muted">{x.record_key}</span> {x.snippet}</>
                      : <>{x.label ?? x.filename} <span className="text-xs text-muted">{x.status}</span></>}
                  </button>
                ))}
              </div>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
}

function ExportMenu() {
  const { run, project, toast, refreshData } = useApp();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const items = [
    ["pdf", "Complete report (PDF)"], ["docx", "Editable report (DOCX)"], ["html", "Report (HTML)"], ["xlsx", "Metrics & tables (XLSX)"],
    ["cleaned_csv", "Cleaned dataset (CSV)"], ["assignments_csv", "Initial coding (CSV)"], ["codebook_csv", "Codebook with quotations (CSV)"],
    ["categories_csv", "Categories (CSV)"], ["themes_csv", "Themes and sub-themes (CSV)"], ["theme_links_csv", "Thematic map links (CSV)"], ["codes_csv", "Codes (CSV)"],
    ["results_json", "Analysis results (JSON)"], ["clusters_csv", "Cluster assignments (CSV)"], ["graphml", "Graph (GraphML)"],
    ["graph_nodes_csv", "Graph nodes (CSV)"], ["graph_edges_csv", "Graph edges (CSV)"], ["temporal_json", "Temporal results (JSON)"],
  ];
  const run_ = async (url: string, key: string) => {
    setBusy(key);
    try { await download(url); toast("Export ready", "success"); refreshData(); } catch (e: any) { toast(e.message, "error"); } // eslint-disable-line @typescript-eslint/no-explicit-any
    finally { setBusy(null); }
  };
  return (
    <div className="relative">
      <Ctl label={<>&nbsp;</>}>
        <button onClick={() => setOpen((o) => !o)} disabled={!project}
          className="flex h-[34px] items-center gap-2 rounded-lg border border-line bg-surface px-3 text-sm font-medium text-ink2 transition-colors hover:bg-raised hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          style={{ boxShadow: "var(--shadow-sm)" }}>
          <Icon path={ICONS.export} size={14} />
          <span>Export</span>
        </button>
      </Ctl>
      {open && (
        <div className="absolute right-0 z-40 mt-1.5 max-h-[70vh] w-72 overflow-y-auto rounded-xl border border-line bg-surface p-1.5" style={{ boxShadow: "var(--shadow-xl)" }} onMouseLeave={() => setOpen(false)}>
          <div className="px-3 py-1.5 text-[11px] text-muted">
            {run?.status === "completed" ? `Selected: ${run.label}` : "Select a completed analysis to export"}
          </div>
          <div className="my-1 border-t border-line" />
          {items.map(([f, label]) => (
            <button key={f} disabled={run?.status !== "completed" || busy !== null}
              className="block w-full rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-raised disabled:opacity-40"
              onClick={() => run_(`/api/runs/${run!.id}/export/${f}`, f)}>
              {busy === f ? <span className="text-muted">Preparing…</span> : label}
            </button>
          ))}
          <div className="my-1 border-t border-line" />
          <button disabled={busy !== null} className="block w-full rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-raised"
            onClick={() => run_(`/api/projects/${project!.id}/backup`, "zip")}>
            {busy === "zip" ? <span className="text-muted">Preparing…</span> : "Project data & backup (ZIP)"}
          </button>
          <div className="px-3 py-1.5 text-[11px] text-muted">Hover any chart for PNG / SVG download.</div>
        </div>
      )}
    </div>
  );
}
