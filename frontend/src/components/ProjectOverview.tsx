/* eslint-disable @typescript-eslint/no-explicit-any */
// The project at a glance: how many files it holds, each file's own analysis, and the average of those analyses.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmt, type Dataset, type Run } from "../api";
import { sentimentColor } from "../colors";
import { useApp, useFetch } from "../state";
import { Icon, ICONS } from "./icons";
import { Badge, Button, Card, ErrorBox, Loading, SectionTitle, Stat, cx } from "./ui";

const usable = (d: Dataset) => d.status === "ready" || d.status === "analysed";

/** Files with no completed analysis of their own. */
export function unanalysed(files: Dataset[], runs: Run[]): Dataset[] {
  return files.filter((f) => usable(f) && !runs.some((r) => r.status === "completed" && r.dataset_ids.length === 1 && r.dataset_ids[0] === f.id));
}

/** Queue one analysis per file (they run one after another); returns the queued runs. */
export async function analyseEach(projectId: string, files: Dataset[], config?: any): Promise<Run[]> {
  const cfg = config ?? (await api.get<any>("/api/config/defaults")).config;
  const runs: Run[] = [];
  for (const f of files) runs.push(await api.post<Run>(`/api/projects/${projectId}/runs`, { dataset_ids: [f.id], config: cfg, label: f.filename }));
  return runs;
}

const COLUMNS: [string, string, (v: any) => string][] = [
  ["records", "Records", (v) => fmt.n(v)], ["passages", "Passages", (v) => fmt.n(v)], ["codes", "Codes", (v) => fmt.n(v, 1)],
  ["coded_pct", "Coded", (v) => fmt.pct(v)], ["themes", "Themes", (v) => fmt.n(v, 1)], ["topics", "Topics", (v) => fmt.n(v, 1)],
  ["positive_pct", "Positive", (v) => fmt.pct(v)], ["negative_pct", "Negative", (v) => fmt.pct(v)], ["mean_tone", "Mean tone", (v) => fmt.n(v, 2)],
];

function ToneBar({ f }: { f: any }) {
  if (f.positive_pct === null || f.positive_pct === undefined) return null;
  const parts: [string, number][] = [["positive", f.positive_pct], ["neutral", f.neutral_pct ?? 0], ["negative", f.negative_pct ?? 0]];
  return (
    <div className="mt-1 flex h-1.5 w-28 overflow-hidden rounded-full bg-raised" title={parts.map(([k, v]) => `${k} ${fmt.pct(v)}`).join(" · ")}>
      {parts.map(([k, v]) => <div key={k} style={{ width: `${v}%`, background: sentimentColor(k) }} />)}
    </div>
  );
}

function saveCsv(o: any, name: string) {
  const fields = ["records", "passages", "words", "codes", "coded_pct", "themes", "clusters", "topics", "positive_pct", "neutral_pct", "negative_pct", "mean_tone", "silhouette", "coherence", "modularity"];
  const q = (v: any) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const lines = [["file", "analysed", ...fields, "top_codes", "themes_found"].join(",")];
  for (const f of o.per_file) lines.push([f.filename, f.run_id ? "yes" : "no", ...fields.map((k) => f[k]), (f.top_codes ?? []).join("; "), (f.theme_labels ?? []).join("; ")].map(q).join(","));
  lines.push(["AVERAGE of analysed files", o.analysed, ...fields.map((k) => o.average[k])].map(q).join(","));
  lines.push(["TOTAL", o.analysed, ...fields.map((k) => o.total[k])].map(q).join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function ProjectOverview() {
  const { project, datasets, allRuns, refreshData, toast, openRun } = useApp();
  const nav = useNavigate();
  const [queuing, setQueuing] = useState(false);
  // refetch whenever a file is added or an analysis changes state
  const key = `${datasets.length}:${allRuns.map((r) => r.id + r.status).join()}`;
  const [o, err, loading] = useFetch<any>(project ? `/api/projects/${project.id}/overview` : null, [key]);
  if (!project || !datasets.length) return null;
  if (err) return <ErrorBox error={err} />;
  if (!o) return loading ? <Loading /> : null;
  const pending = unanalysed(datasets, allRuns);
  const running = allRuns.filter((r) => r.dataset_ids.length === 1 && ["queued", "running"].includes(r.status)).length;
  const avg = o.average;
  const open = (runId: string) => { openRun(runId); nav("/dashboard"); };
  const queue = async () => {
    setQueuing(true);
    try { const r = await analyseEach(project.id, pending); await refreshData(); toast(`${r.length} ${r.length === 1 ? "analysis" : "analyses"} queued; they run one after another`, "success"); }
    catch (e: any) { toast(e.message, "error"); }
    finally { setQueuing(false); }
  };

  return (
    <section>
      <SectionTitle id="overview" title={`Project overview · ${project.name}`} desc="Every file with its own analysis, and the average of those analyses: each analysed file counts once." />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Files" value={fmt.n(o.files)} detail={`${fmt.n(o.analysed)} analysed${running ? ` · ${running} running` : ""}`} />
        <Stat label="Records" value={fmt.n(o.total.records)} detail="in the analysed files" />
        <Stat label="Passages" value={fmt.n(o.total.passages)} detail="in the analysed files" />
        <Stat label="Words" value={fmt.n(o.total.words)} detail="in the analysed files" />
      </div>
      {pending.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl border border-warn/30 bg-warn/5 px-4 py-3 text-sm">
          <span className="text-ink2">{pending.length} {pending.length === 1 ? "file has" : "files have"} no analysis yet, so {pending.length === 1 ? "it is" : "they are"} not in the average.</span>
          <Button small kind="primary" disabled={queuing || running > 0} onClick={queue}><Icon path={ICONS.play} size={12} />{running ? "Analysing…" : `Analyse ${pending.length === 1 ? "it" : `each of the ${pending.length}`}`}</Button>
        </div>
      )}

      <h3 className="mb-3 mt-8 text-sm font-semibold text-ink">Average of the {fmt.n(o.analysed)} file {o.analysed === 1 ? "analysis" : "analyses"}</h3>
      {o.analysed === 0 ? <p className="text-sm text-muted">Analyse at least one file to see averages.</p> : (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Records per file" value={fmt.n(avg.records, 1)} detail={`${fmt.n(avg.passages, 1)} passages · ${fmt.n(avg.words, 0)} words`} />
          <Stat label="Codes per file" value={fmt.n(avg.codes, 1)} detail={`${fmt.pct(avg.coded_pct)} of passages coded`} />
          <Stat label="Themes per file" value={fmt.n(avg.themes, 1)} detail={`${fmt.n(avg.clusters, 1)} clusters · ${fmt.n(avg.topics, 1)} topics`} />
          <Stat label="Tone" value={fmt.pct(avg.positive_pct)} detail={`positive · ${fmt.pct(avg.neutral_pct)} neutral · ${fmt.pct(avg.negative_pct)} negative · mean ${fmt.n(avg.mean_tone, 2)}`} />
        </div>
      )}

      <Card className="mt-4" title="Files and their analyses" subtitle={o.note}
        actions={<Button small onClick={() => saveCsv(o, `${project.name} overview.csv`)}><Icon path={ICONS.export} size={12} />Overview (CSV)</Button>}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm tabular-nums">
            <thead>
              <tr className="border-b border-line text-left text-xs text-muted">
                <th className="py-2 pr-3 font-medium">File</th>
                {COLUMNS.map(([k, label]) => <th key={k} className="px-2 py-2 text-right font-medium">{label}</th>)}
                <th className="py-2 pl-3 font-medium">Top codes</th>
              </tr>
            </thead>
            <tbody>
              {o.per_file.map((f: any) => (
                <tr key={f.dataset_id} className={cx("border-b border-line/60", f.run_id && "cursor-pointer hover:bg-raised")} onClick={() => f.run_id && open(f.run_id)}
                  title={f.run_id ? "Open this file's dashboard" : undefined}>
                  <td className="max-w-[260px] py-2 pr-3">
                    <div className="truncate font-medium text-ink" title={f.filename}>{f.filename}</div>
                    {f.run_id ? <ToneBar f={f} /> : <Badge status={f.status === "failed" ? "failed" : "candidate"}>{f.status === "failed" ? "not readable" : "not analysed"}</Badge>}
                  </td>
                  {COLUMNS.map(([k, , show]) => <td key={k} className="px-2 py-2 text-right text-ink2">{f.run_id ? show(f[k]) : ""}</td>)}
                  <td className="max-w-[260px] truncate py-2 pl-3 text-xs text-ink2" title={(f.top_codes ?? []).join(", ")}>{(f.top_codes ?? []).join(", ")}</td>
                </tr>
              ))}
            </tbody>
            {o.analysed > 0 && (
              <tfoot>
                <tr className="border-t-2 border-line font-semibold text-ink">
                  <td className="py-2 pr-3">Average ({o.analysed} {o.analysed === 1 ? "file" : "files"})</td>
                  {COLUMNS.map(([k, , show]) => <td key={k} className="px-2 py-2 text-right">{show(avg[k])}</td>)}
                  <td />
                </tr>
                <tr className="text-ink2">
                  <td className="py-1.5 pr-3">Total</td>
                  {COLUMNS.map(([k]) => <td key={k} className="px-2 py-1.5 text-right">{k in o.total ? fmt.n(o.total[k]) : ""}</td>)}
                  <td />
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </Card>

      {o.recurring_codes.length > 0 && (
        <Card className="mt-4" title="Codes found in more than one file" subtitle="The same code label coming up in several files' analyses">
          <div className="flex flex-wrap gap-1.5">
            {o.recurring_codes.map((c: any) => (
              <span key={c.code} className="rounded-full bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent" title={`${c.passages} passages`}>
                {c.code} <span className="text-accent/70">· {c.files} files</span>
              </span>
            ))}
          </div>
        </Card>
      )}

      {o.files > 1 && (
        <p className="mt-4 text-xs text-muted">
          {o.combined_run
            ? <>There is also one thematic analysis of {o.combined_run.files} files together, with themes across all of them. <button className="font-medium text-accent hover:underline" onClick={() => open(o.combined_run.id)}>Open it</button>.</>
            : <>For one thematic analysis with themes across all files, use “Analyse all files together” on the Upload & analyse page.</>}
        </p>
      )}
    </section>
  );
}
