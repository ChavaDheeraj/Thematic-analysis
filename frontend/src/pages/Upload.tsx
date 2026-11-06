/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, fmt, type Dataset, type Mapping, type Run } from "../api";
import { Badge, Button, Card, Disclosure, Empty, ErrorBox, Field, Loading, PageHeader, PctStat, Stat, cx, inputBase, inputCls } from "../components/ui";
import { GnnUploadCard } from "../components/Gnn";
import { Icon, ICONS } from "../components/icons";
import { ALL_FILES, useApp, useFetch } from "../state";

// examples shown on the drop zone; any file with readable text is accepted, audio and video are not
const FORMATS = "TXT · DOCX · PDF · HTML · CSV · XLSX · JSON · SRT · VTT · RTF · ODT · PPTX · EPUB · MD · XML";
const isMedia = (f: File) => /^(audio|video)\//.test(f.type) || /\.(mp3|wav|m4a|aac|flac|ogg|oga|opus|wma|aiff?|amr|midi?|mp4|m4v|mov|avi|mkv|webm|wmv|flv|mpe?g|3gp|3g2|m2ts|mts|vob|ogv)$/i.test(f.name);
const usable = (d: Dataset | null) => d?.status === "ready" || d?.status === "analysed";
const MEDIA = "Audio and video files are not accepted. Upload a transcript of the recording instead (for example .txt, .docx, .srt or .vtt).";

export default function UploadPage() {
  const { project } = useApp();
  if (!project) {
    return <Empty title="Create a project to begin">Use the ＋ button next to the project selector. Projects keep uploads, analyses and researcher edits together on this machine.</Empty>;
  }
  return <UploadFlow />;
}

function UploadFlow() {
  const { dataset, datasets, allFiles, runs } = useApp();
  const [adding, setAdding] = useState(false);
  const [notes, setNotes] = useState<string[]>([]); // files of the last upload that were not added, or not readable
  const settings = useRunSettings(); // kept while files change, so one codebook can be applied to every analysis
  const files = allFiles ? datasets : dataset ? [dataset] : [];
  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="Upload & analyse" desc="Add one file or many. ThematicML reads them, codes them and finds their themes and tone on this machine: all your files together as one thematic analysis, or one file on its own." />
      {notes.length > 0 && !adding && files.length > 0 && (
        <div className="mb-4 space-y-2">
          {notes.map((n) => <ErrorBox key={n} error={n} />)}
          <Button small kind="ghost" onClick={() => setNotes([])}>Dismiss</Button>
        </div>
      )}
      {!files.length || adding ? (
        <>
          <DropZone onDone={(problems) => { setAdding(false); setNotes(problems); }} onCancel={files.length ? () => setAdding(false) : undefined} />
          <HowItWorks />
        </>
      ) : <ConfigureAndRun key={allFiles ? ALL_FILES : dataset!.id} ds={allFiles ? null : dataset} files={files} runs={runs} s={settings} onAdd={() => setAdding(true)} />}
    </div>
  );
}

const STEPS = [
  ["Read", "Splits the file into passages and keeps the original unchanged"],
  ["Code", "Finds key phrases with TF-IDF, plus any codebook rules you add"],
  ["Group", "Joins codes that occur together into candidate themes"],
  ["Measure", "Scores the tone of every passage and fits topic models"],
];

function HowItWorks() {
  return (
    <ol className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {STEPS.map(([t, d], i) => (
        <li key={t} className="rounded-xl border border-line bg-surface/70 p-4">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/10 text-[11px] font-bold text-accent">{i + 1}</span>
            <span className="text-sm font-semibold text-ink">{t}</span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-muted">{d}</p>
        </li>
      ))}
    </ol>
  );
}

// ---------------------------------------------------------------- 1 · file

function DropZone({ onDone, onCancel }: { onDone: (problems: string[]) => void; onCancel?: () => void }) {
  const { project, refreshData, setDatasetId, toast } = useApp();
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const input = useRef<HTMLInputElement>(null);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    const problems: string[] = [];
    const added: Dataset[] = [];
    setErrors([]);
    for (const [i, f] of files.entries()) {
      if (isMedia(f)) { problems.push(`${f.name}: ${MEDIA}`); continue; }
      setBusy(files.length > 1 ? `${f.name} (${i + 1} of ${files.length})` : f.name);
      const fd = new FormData();
      fd.append("files", f, f.name);
      try {
        let [res] = await api.post<any[]>(`/api/projects/${project!.id}/uploads`, fd);
        if (!res.id) { problems.push(`${f.name}: ${res.error ?? "upload failed"}`); continue; }
        // read the file with the suggested column mapping straight away; Advanced options can change it
        const suggested = res.mapping?.suggested;
        if (res.status !== "failed" && suggested?.text_col) {
          res = await api.post<Dataset>(`/api/datasets/${res.id}/mapping`, { duplicate_policy: "flag", metadata_cols: [], ...suggested });
        }
        if (res.status === "failed") problems.push(`${f.name}: ${res.error}`);
        added.push(res);
      } catch (e: any) {
        problems.push(`${f.name}: ${e.message}`);
      }
    }
    setBusy(null);
    setErrors(problems);
    if (!added.length) return;
    await refreshData();
    // several files are analysed together as one thematic analysis; a first, single file on its own
    const total = (await api.get<Dataset[]>(`/api/projects/${project!.id}/datasets`)).length;
    setDatasetId(total > 1 ? ALL_FILES : added[0].id);
    const ready = added.filter(usable);
    const note = problems.length ? ` · ${problems.length} ${problems.length === 1 ? "file needs" : "files need"} attention (see the message above)` : "";
    toast((added.length === 1
      ? (ready.length ? `${added[0].filename}: ${fmt.n(added[0].quality?.valid_records)} records ready to analyse` : `${added[0].filename} uploaded — check its columns under Advanced options`)
      : `${ready.length} of ${added.length} files added and ready to analyse`) + note, problems.length ? "error" : "success");
    onDone(problems);
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); if (!busy) upload(Array.from(e.dataTransfer.files)); }}
        onClick={() => !busy && input.current?.click()}
        className={cx("flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-colors",
          drag ? "border-accent bg-accent/5" : "border-line bg-surface hover:border-accent/60 hover:bg-accent/[0.02]", busy && "cursor-wait")}>
        {busy ? (
          <>
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
            <div className="mt-3 font-medium">Reading {busy}…</div>
            <div className="mt-1 text-xs text-muted">Parsing and checking the file on this machine</div>
          </>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-accent/10 text-accent"><Icon path={ICONS.upload} size={24} /></div>
            <div className="mt-4 text-[15px] font-semibold text-ink">Drop files here, or click to choose them</div>
            <div className="mt-1 text-sm text-muted">Blogs, articles, transcripts, reports or survey exports — one or many; several files are analysed together</div>
            <div className="mt-4 flex flex-wrap justify-center gap-1.5">
              {FORMATS.split(" · ").map((x) => <span key={x} className="rounded-md bg-raised px-2 py-0.5 text-[11px] font-medium text-ink2">{x}</span>)}
              <span className="px-1 py-0.5 text-[11px] text-muted">and any other text file · no audio or video</span>
            </div>
          </>
        )}
        <input ref={input} type="file" multiple className="hidden" onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      </div>
      {errors.map((e) => <ErrorBox key={e} error={e} />)}
      {onCancel && !busy && <Button small kind="ghost" onClick={onCancel}>Back to the analysis</Button>}
    </div>
  );
}

/** Switch between the analysis of all files together and one file, where the project has several. */
function ScopeSelect() {
  const { datasets, datasetId, setDatasetId } = useApp();
  if (datasets.length < 2) return null;
  return (
    <select aria-label="What to analyse" className={cx(inputBase, "w-56 py-1.5")} value={datasetId ?? ""} onChange={(e) => setDatasetId(e.target.value)}>
      <option value={ALL_FILES}>All files together ({datasets.length})</option>
      {datasets.map((d) => <option key={d.id} value={d.id}>Only {d.filename}</option>)}
    </select>
  );
}

function AllFilesSummary({ files, onAdd }: { files: Dataset[]; onAdd: () => void }) {
  const ok = files.filter(usable);
  const records = ok.reduce((n, f) => n + (f.quality?.valid_records ?? 0), 0);
  return (
    <div className="px-6 py-5">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent"><Icon path={ICONS.dashboard} size={22} /></div>
        <div className="min-w-0 flex-1">
          <div className="text-[15px] font-semibold text-ink">All files together</div>
          <div className="mt-0.5 text-xs text-muted">{ok.length} of {files.length} files · {fmt.n(records)} records · one thematic analysis of all of them</div>
        </div>
        <ScopeSelect />
        <Button kind="ghost" onClick={onAdd}><Icon path={ICONS.upload} size={14} />Add files</Button>
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {files.slice(0, 10).map((f) => (
          <span key={f.id} title={usable(f) ? f.filename : `${f.filename}: ${f.error ?? "columns not set"} — not included`}
            className={cx("max-w-[240px] truncate rounded-md px-2 py-0.5 text-[11px] font-medium", usable(f) ? "bg-raised text-ink2" : "bg-bad/10 text-bad line-through")}>{f.filename}</span>
        ))}
        {files.length > 10 && <span className="px-1 py-0.5 text-[11px] text-muted">+{files.length - 10} more</span>}
      </div>
    </div>
  );
}

function FileSummary({ ds, onReplace }: { ds: Dataset; onReplace: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-4 px-6 py-5">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent"><Icon path={ICONS.file} size={22} /></div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[15px] font-semibold text-ink" title={ds.filename}>{ds.filename}</div>
        <div className="mt-0.5 text-xs text-muted">
          {ds.format.toUpperCase()} · {fmt.bytes(ds.size_bytes)} · {fmt.n(ds.n_rows)} rows{ds.quality ? ` · ${fmt.n(ds.quality.valid_records)} valid` : ""} · uploaded {fmt.date(ds.uploaded_at)}
        </div>
      </div>
      <Badge status={ds.status} />
      <ScopeSelect />
      <Button kind="ghost" onClick={onReplace}><Icon path={ICONS.upload} size={14} />Add files</Button>
    </div>
  );
}

// ---------------------------------------------------------------- 2 · columns

function PreviewAndMap({ ds }: { ds: Dataset }) {
  const { refreshData, toast } = useApp();
  const [pv, err, loading] = useFetch<any>(`/api/datasets/${ds.id}/preview?rows=25`);
  const initial: Mapping = ds.mapping?.applied ?? ds.mapping?.suggested ?? { text_col: ds.columns?.[0] ?? null };
  const [m, setM] = useState<Mapping>({ duplicate_policy: "flag", metadata_cols: [], ...initial });
  const [busy, setBusy] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [editing, setEditing] = useState(!usable(ds));
  const cols = ds.columns ?? [];
  const q = ds.quality;

  const apply = async () => {
    setBusy(true);
    try {
      const out = await api.post<Dataset>(`/api/datasets/${ds.id}/mapping`, m);
      await refreshData();
      toast(out.status === "ready" ? `Validated: ${out.quality.valid_records} valid records` : out.error ?? "Validation failed", out.status === "ready" ? "success" : "error");
    } catch (e: any) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const sel = (key: keyof Mapping, label: string, required = false, hint?: string) => (
    <Field label={label} hint={hint}>
      <select className={inputCls} value={(m[key] as string) ?? ""} onChange={(e) => setM({ ...m, [key]: e.target.value || null })}>
        {!required && <option value="">— none —</option>}
        {cols.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    </Field>
  );

  return (
    <div className="space-y-4">
      {(ds.warnings?.length ?? 0) > 0 && <div className="rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn">! {ds.warnings!.join(" ")}</div>}
      {!editing && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded-xl border border-line bg-surface px-5 py-3 text-sm" style={{ boxShadow: "var(--shadow-sm)" }}>
          <span className="flex items-center gap-1.5 font-medium text-good"><Icon path={ICONS.check} size={14} />{ds.structured ? "Columns mapped" : "Read as passages"}</span>
          {([["Text", m.text_col], ["ID", m.id_col], ["Group", m.group_col], ["Timestamp", m.time_col]] as const).map(([k, v]) => (
            <span key={k}><span className="text-muted">{k}</span> <span className="font-medium text-ink">{v ?? "—"}</span></span>
          ))}
          <Button small kind="ghost" onClick={() => setEditing(true)}>Change mapping</Button>
        </div>
      )}
      <div className={cx("grid gap-4", editing && "xl:grid-cols-[1.6fr_1fr]")}>
        <Card title="Preview" subtitle="First 25 rows, exactly as parsed"
          actions={pv && <Button small kind="ghost" onClick={() => setShowProfile((x) => !x)}>{showProfile ? "Hide" : "Show"} column profile</Button>}>
          {loading && <Loading />}
          {err && <ErrorBox error={err} />}
          {pv && (
            <>
              <div className="max-h-[340px] overflow-auto rounded-lg border border-line">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-raised text-left text-ink2">
                    <tr>{pv.columns.map((c: string) => (
                      <th key={c} className={cx("whitespace-nowrap px-2.5 py-2 font-semibold", c === m.text_col && "text-accent")}>
                        {c}{c === m.text_col ? " · text" : c === m.id_col ? " · id" : c === m.time_col ? " · time" : c === m.group_col ? " · group" : ""}
                      </th>))}</tr>
                  </thead>
                  <tbody>
                    {pv.rows.map((r: any, i: number) => (
                      <tr key={i} className="border-t border-line/70">
                        {pv.columns.map((c: string) => <td key={c} className="max-w-[360px] truncate px-2.5 py-1.5" title={r[c] ?? ""}>{r[c] ?? <span className="text-muted">∅</span>}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {showProfile && (
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="text-left text-ink2"><tr><th className="py-1 pr-3 font-medium">Column</th><th className="pr-3 text-right font-medium">Non-empty</th><th className="pr-3 text-right font-medium">Missing</th><th className="pr-3 text-right font-medium">Unique</th><th className="text-right font-medium">Avg length</th></tr></thead>
                    <tbody className="tabular-nums">{pv.profile.map((p: any) => (
                      <tr key={p.column} className="border-t border-line"><td className="py-1 pr-3">{p.column}</td><td className="pr-3 text-right">{fmt.n(p.non_empty)}</td><td className="pr-3 text-right">{fmt.n(p.missing)}</td><td className="pr-3 text-right">{fmt.n(p.unique)}</td><td className="text-right">{fmt.n(p.avg_length, 1)}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </Card>
        {editing && <Card title="Column mapping" subtitle="Suggested from column names and content — check before validating">
          <div className="grid grid-cols-2 gap-3">
            {sel("text_col", "Text column", true)}
            {sel("id_col", "Record / respondent ID", false, "Without one, row numbers are used")}
            {sel("time_col", "Timestamp", false, "Enables temporal analysis")}
            {sel("group_col", "Group field", false, "For group comparisons")}
            {sel("label_col", "Label column", false, "Researcher labels for supervised coding")}
            <Field label="Duplicates" hint="Duplicate = same ID and same text">
              <select className={inputCls} value={m.duplicate_policy} onChange={(e) => setM({ ...m, duplicate_policy: e.target.value as "flag" | "exclude" })}>
                <option value="flag">Flag and keep</option>
                <option value="exclude">Exclude from analysis</option>
              </select>
            </Field>
            <Field label="Other languages" hint="Text features and sentiment are English-only">
              <select className={inputCls} value={m.language_policy ?? "keep"} onChange={(e) => setM({ ...m, language_policy: e.target.value as "keep" | "exclude_non_english" })}>
                <option value="keep">Keep (flag only)</option>
                <option value="exclude_non_english">Exclude non-English records</option>
              </select>
            </Field>
          </div>
          {cols.length > 1 && (
            <div className="mt-3">
              <Field label="Metadata columns to keep">
                <div className="flex flex-wrap gap-1.5">
                  {cols.filter((c) => c !== m.text_col).map((c) => {
                    const on = m.metadata_cols?.includes(c);
                    return <button key={c} onClick={() => setM({ ...m, metadata_cols: on ? m.metadata_cols!.filter((x) => x !== c) : [...(m.metadata_cols ?? []), c] })}
                      className={cx("rounded-md border px-2 py-0.5 text-xs", on ? "border-accent bg-accent/10 text-ink" : "border-line text-ink2 hover:bg-raised")}>{on ? "✓ " : ""}{c}</button>;
                  })}
                </div>
              </Field>
            </div>
          )}
          <div className="mt-4 flex items-center gap-3">
            <Button kind={usable(ds) ? "default" : "primary"} onClick={apply} disabled={busy || !m.text_col}>{busy ? "Validating…" : usable(ds) ? "Re-validate mapping" : "Validate & save mapping"}</Button>
            {usable(ds) && <span className="flex items-center gap-1 text-xs text-good"><Icon path={ICONS.check} size={12} />Validated</span>}
          </div>
        </Card>}
      </div>

      {q && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat label="Parsed records" value={fmt.n(q.total_records)} />
          <PctStat label="Valid records" p={q.percentages?.valid} />
          <PctStat label="Missing text" p={q.percentages?.missing_text} />
          <PctStat label="Duplicates" p={q.percentages?.duplicates} />
          <PctStat label="Excluded" p={q.percentages?.excluded} />
          {q.percentages?.non_english
            ? <PctStat label="Non-English" p={q.percentages.non_english} />
            : <Stat label="Invalid timestamps" value={fmt.n(q.invalid_timestamps)} detail={m.time_col ? "kept; no timestamp" : "no timestamp column"} />}
        </div>
      )}
      {(q?.warnings?.length ?? 0) > 0 && <ul className="list-disc space-y-0.5 pl-5 text-sm text-warn">{q.warnings.map((w: string) => <li key={w}>{w}</li>)}</ul>}
    </div>
  );
}

// ---------------------------------------------------------------- 3 · settings, 4 · run

function useRunSettings() {
  const [defaults] = useFetch<any>("/api/config/defaults");
  const [cfg, setCfg] = useState<any>(null);
  const [label, setLabel] = useState("");
  const [codebook, setCodebook] = useState("");
  const [synonyms, setSynonyms] = useState("");
  const [stopwords, setStopwords] = useState("");
  useEffect(() => { if (defaults && !cfg) setCfg(defaults.config); }, [defaults, cfg]);
  return { cfg, setCfg, label, setLabel, codebook, setCodebook, synonyms, setSynonyms, stopwords, setStopwords };
}

function ConfigureAndRun({ ds, files, runs, s, onAdd }: { ds: Dataset | null; files: Dataset[]; runs: Run[]; s: ReturnType<typeof useRunSettings>; onAdd: () => void }) {
  const { project, allRuns, refreshData, toast, openRun, system } = useApp();
  const { cfg, setCfg, label, setLabel, codebook, setCodebook, synonyms, setSynonyms, stopwords, setStopwords } = s;
  const [watch, setWatch] = useState<string | null>(null);
  const nav = useNavigate();
  // ds is the one file analysed on its own; null when all files are analysed together
  const included = files.filter(usable);
  const ready = ds ? usable(ds) : included.length > 0;
  const applied = included.map((f) => f.mapping?.applied).filter(Boolean) as Mapping[];
  const hasTime = applied.some((m) => m.time_col);
  const hasLabel = applied.some((m) => m.label_col);
  const hasGroup = applied.some((m) => m.group_col);
  const leiden = system?.optional?.leiden;

  // follow the run started here, or one already running for this file
  const watched = allRuns.find((r) => r.id === watch) ?? runs.find((r) => r.status === "queued" || r.status === "running") ?? null;
  const busy = watched !== null && ["queued", "running"].includes(watched.status);
  useEffect(() => {
    if (watch && watched?.id === watch && watched.status === "completed") {
      openRun(watched.id);
      nav("/dashboard");
    }
  }, [watched?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // one rule per line: "[Category >] Label: keyword, phrase, prefix*"
  const parsedCodebook = useMemo(() => codebook.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => {
    const at = l.indexOf(":");
    const head = at < 0 ? l : l.slice(0, at);
    const rest = at < 0 ? "" : l.slice(at + 1);
    const gt = head.indexOf(">");
    return {
      label: (gt < 0 ? head : head.slice(gt + 1)).trim(),
      category: gt < 0 ? null : head.slice(0, gt).trim() || null,
      keywords: rest.split(",").map((k) => k.trim()).filter(Boolean),
    };
  }).filter((r) => r.label && r.keywords.length), [codebook]);

  // a settings file is a codebook ({"codes": [...]}), an analysis config, or {"analysis": {...}}
  const loadSettings = async (file: File) => {
    try {
      const data = JSON.parse(await file.text());
      const a = data.analysis ?? data;
      const rules = a.codebook ?? data.codes;
      if (Array.isArray(rules)) {
        setCodebook(rules.map((r: any) => `${r.category ? `${r.category} > ` : ""}${r.label}: ${(r.keywords ?? []).join(", ")}`).join("\n"));
      }
      if (!data.codes) {
        const known = Object.fromEntries(Object.entries(a).filter(([k]) => k in cfg && !["codebook", "synonyms", "custom_stopwords"].includes(k)));
        setCfg({ ...cfg, ...known });
        if (a.synonyms) setSynonyms(Object.entries(a.synonyms).map(([k, v]) => `${k} => ${v}`).join("\n"));
        if (a.custom_stopwords) setStopwords(a.custom_stopwords.join(", "));
      }
      toast(`Loaded settings from ${file.name}`, "success");
    } catch (e: any) {
      toast(`Could not read ${file.name}: ${e.message}`, "error");
    }
  };

  if (!cfg) return <Loading />;
  const set = (k: string, v: unknown) => setCfg({ ...cfg, [k]: v });
  const num = (k: string, label: string, opts: { min?: number; max?: number; step?: number; hint?: string; nullable?: boolean } = {}) => (
    <Field label={label} hint={opts.hint}>
      <input type="number" className={inputCls} min={opts.min} max={opts.max} step={opts.step ?? 1} value={cfg[k] ?? ""}
        placeholder={opts.nullable ? "auto" : undefined}
        onChange={(e) => set(k, e.target.value === "" ? (opts.nullable ? null : cfg[k]) : Number(e.target.value))} />
    </Field>
  );
  const choice = (k: string, label: string, options: [string, string][], hint?: string) => (
    <Field label={label} hint={hint}>
      <select className={inputCls} value={cfg[k]} onChange={(e) => set(k, e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </Field>
  );
  const check = (k: string, label: string, disabled = false, hint?: string) => (
    <label className={cx("flex items-start gap-2 text-sm", disabled && "opacity-50")} title={hint}>
      <input type="checkbox" className="mt-0.5" disabled={disabled} checked={!!cfg[k] && !disabled} onChange={(e) => set(k, e.target.checked)} />
      <span>{label}{hint && <span className="block text-[11px] text-muted">{hint}</span>}</span>
    </label>
  );

  const start = async () => {
    const body = {
      dataset_ids: ds ? [ds.id] : included.map((f) => f.id),
      label: label || (ds ? undefined : `All ${included.length} files together`),
      config: {
        ...cfg,
        temporal: cfg.temporal && hasTime,
        supervised: cfg.supervised && hasLabel,
        codebook: parsedCodebook,
        synonyms: Object.fromEntries(synonyms.split("\n").map((l) => l.split("=>").map((x) => x.trim())).filter((p) => p.length === 2 && p[0] && p[1])),
        custom_stopwords: stopwords.split(/[,\n]/).map((s) => s.trim()).filter(Boolean),
      },
    };
    try {
      const r = await api.post<Run>(`/api/projects/${project!.id}/runs`, body);
      setWatch(r.id);
      await refreshData();
      toast(`${r.label} queued`);
    } catch (e: any) { toast(e.message, "error"); }
  };

  const previous = runs.filter((r) => r.status === "completed");
  const failed = !!ds && ds.status === "failed" && !ds.columns;
  const left = files.length - included.length;
  return (
    <>
      <section className="overflow-hidden rounded-2xl border border-line bg-surface" style={{ boxShadow: "var(--shadow-sm)" }}>
        {ds ? <FileSummary ds={ds} onReplace={onAdd} /> : <AllFilesSummary files={files} onAdd={onAdd} />}
        <div className="border-t border-line bg-page/40 px-6 py-5">
          {failed && <div className="mb-4"><ErrorBox error={`${ds!.filename}: ${ds!.error}`} /></div>}
          {ds && !failed && !ready && <div className="mb-4 rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn">! Check which column holds the text under Advanced options, then validate.</div>}
          {!ds && left > 0 && <div className="mb-4 rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn">! {left} {left === 1 ? "file is" : "files are"} not included (struck through above): {left === 1 ? "it could" : "they could"} not be read, or {left === 1 ? "its" : "their"} text column is not set. Choose {left === 1 ? "it" : "one"} in “What to analyse” to fix it.</div>}
          <div className="flex flex-wrap items-center gap-3">
            <Button large kind="primary" onClick={start} disabled={!ready || busy}><Icon path={ICONS.play} size={14} />{busy ? "Analysing…" : ds ? (previous.length ? "Analyse again" : "Analyse this file") : (previous.length ? "Analyse all files together again" : `Analyse all ${included.length} files together`)}</Button>
            {previous.length > 0 && (
              <Button kind="ghost" onClick={() => { openRun(previous[0].id); nav("/dashboard"); }}>
                Open the latest dashboard<Icon path={ICONS.arrow} size={13} />
              </Button>
            )}
            <span className="ml-auto text-xs text-muted">Runs locally · settings under Advanced options</span>
          </div>
          {watched && <ProgressMonitor run={watched} />}
        </div>
      </section>

      <Disclosure title="Advanced options" desc={ds ? "How the file was read, column mapping, codebook rules and every ML setting" : "Codebook rules and every ML setting for the analysis of all files"} defaultOpen={!!ds && !ready && !failed}>
        {!ds && <p className="mb-4 text-xs text-muted">Each file keeps the columns ThematicML found in it. To check or change one file's columns, choose it in “What to analyse”.</p>}
        {ds?.columns && (
          <>
            <h3 className="text-sm font-semibold text-ink">How the file was read</h3>
            <p className="mb-3 text-xs text-muted">{fmt.n(ds.n_rows)} rows · encoding {ds.encoding ?? "n/a"}{ds.structured ? "" : " · document split into passages"}</p>
            <PreviewAndMap key={ds.id} ds={ds} />
          </>
        )}
        <div className="mb-3 mt-8 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-ink">Analysis settings</h3>
            <p className="text-xs text-muted">The defaults suit most files. Every parameter is stored with the run, so results can be reproduced.</p>
          </div>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink2 hover:bg-raised" title="A codebook or analysis configuration (.json); fills the settings below, which stay editable" style={{ boxShadow: "var(--shadow-sm)" }}>
            <Icon path={ICONS.file} size={13} />Load settings file
            <input type="file" accept=".json,application/json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) loadSettings(f); e.target.value = ""; }} />
          </label>
        </div>
          <div>
            <div className="grid gap-x-6 gap-y-4 md:grid-cols-2 xl:grid-cols-4">
              <Field label="Run label (optional)"><input className={inputCls} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. Baseline TF-IDF + K-means" /></Field>
              {choice("segmentation", "Split text into", [["sentence", "Sentences"], ["paragraph", "Paragraphs"], ["record", "Whole records"]])}
              {choice("coding_method", "Coding method", [["tfidf", "TF-IDF phrase extraction"], ["rules", "Codebook rules"], ["both", "Both"]])}
              {choice("graph_method", "Theme detection", [["louvain", "Louvain"], ...(leiden ? [["leiden", "Leiden"] as [string, string]] : []), ["greedy", "Greedy modularity"], ["label_propagation", "Label propagation"]],
                leiden ? undefined : "Leiden needs python-igraph + leidenalg")}
            </div>
            <div className="mt-4 grid gap-x-6 gap-y-3 lg:grid-cols-[2fr_1fr]">
              <Field label="Codebook rules (optional)" hint="One per line — [Category >] Label: keyword, phrase, prefix*">
                <textarea className={cx(inputCls, "h-24 font-mono text-xs")} value={codebook} onChange={(e) => setCodebook(e.target.value)} placeholder={"Safety > Feeling unsafe: unsafe, harass*, scared\nPlanning > Budget: budget, cost*, afford*"} />
              </Field>
              <div className="space-y-2 pt-6">
                {parsedCodebook.length > 0 && <p className="text-xs text-ink2">{parsedCodebook.length} rules in {new Set(parsedCodebook.map((r) => r.category).filter(Boolean)).size} categories</p>}
                {cfg.coding_method === "both" && check("complement_codebook", "TF-IDF codes add terms the codebook doesn't already match")}
                {check("sentiment", "Lexicon sentiment (VADER, rule-based)", !system?.optional?.sentiment_vader)}
                {check("supervised", "Supervised classifier on labelled records", !hasLabel, hasLabel ? "Logistic regression, held-out evaluation grouped by respondent" : "Map a label column to enable")}
              </div>
            </div>
          </div>

          <Disclosure title="Text preprocessing & features" desc="Stopwords, n-grams, vocabulary, synonyms">
            <div className="grid gap-x-6 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
              {num("min_segment_words", "Min words per segment", { min: 1, hint: "Shorter fragments join a neighbour" })}
              {num("ngram_min", "N-gram min", { min: 1, max: 3 })}
              {num("ngram_max", "N-gram max", { min: 1, max: 4 })}
              {num("min_df", "Min document frequency", { min: 1 })}
              {num("max_df", "Max document share", { min: 0.05, max: 1, step: 0.05 })}
              {num("max_features", "Max vocabulary", { min: 50, step: 100 })}
            </div>
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1.5">
              {check("lowercase", "Lowercase")}
              {check("remove_stopwords", "Remove stopwords")}
              {check("keep_negations", "Keep negations (not, never, can't…)", !cfg.remove_stopwords)}
            </div>
            <div className="mt-3 grid gap-x-6 gap-y-3 md:grid-cols-2">
              <Field label="Extra stopwords" hint="Comma or newline separated"><input className={inputCls} value={stopwords} onChange={(e) => setStopwords(e.target.value)} /></Field>
              <Field label="Domain vocabulary / synonyms" hint="One per line: variant => canonical (e.g. ui => interface)">
                <textarea className={cx(inputCls, "h-16 font-mono text-xs")} value={synonyms} onChange={(e) => setSynonyms(e.target.value)} />
              </Field>
            </div>
          </Disclosure>

          <Disclosure title="Codes, clusters & topics" desc="How many codes, clusters and topics to look for">
            <div className="grid gap-x-6 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
              {num("n_codes", "Max candidate codes", { min: 1, max: 500 })}
              {num("min_code_support", "Min segments per code", { min: 1 })}
              {num("phrase_boost", "Multi-word phrase boost", { min: 1, max: 3, step: 0.05 })}
              {choice("clustering_method", "Clustering", [["kmeans", "K-means"], ["hierarchical", "Agglomerative (Ward)"]])}
              {num("n_clusters", "Clusters (k)", { min: 2, max: 100, nullable: true, hint: "Blank = choose by silhouette" })}
              {num("k_min", "Auto k from", { min: 2 })}
              {num("k_max", "Auto k to", { min: 2, max: 100 })}
              {choice("topic_method", "Topic model", [["nmf", "NMF (TF-IDF)"], ["lda", "LDA (counts)"]])}
              {num("n_topics", "Topics", { min: 2, max: 100 })}
            </div>
          </Disclosure>

          <Disclosure title="Graph, themes & time" desc="Co-occurrence graph, theme granularity, temporal analysis, seed">
            <div className="grid gap-x-6 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
              {choice("cooccurrence_level", "Co-occurrence unit", [["record", "Record"], ["segment", "Segment"]])}
              {num("min_cooccurrence", "Min co-occurrence", { min: 1 })}
              {num("min_npmi", "Min NPMI", { min: -1, max: 1, step: 0.05, hint: ">0 = above chance; raise to sparsify" })}
              {num("resolution", "Community resolution", { min: 0.1, max: 10, step: 0.1, hint: "Higher = more, smaller themes" })}
              {num("similarity_threshold", "Similarity edge threshold", { min: 0, max: 1, step: 0.05 })}
              {num("random_state", "Random seed", { min: 0, hint: "Same seed + config + data ⇒ same results" })}
              {cfg.temporal && hasTime && choice("temporal_freq", "Period", [["auto", "Auto"], ["D", "Day"], ["W", "Week"], ["M", "Month"], ["Q", "Quarter"]])}
              {cfg.temporal && hasTime && num("n_graph_windows", "Graph time windows", { min: 2, max: 24 })}
            </div>
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1.5">
              {check("similarity_edges", "Add TF-IDF similarity edges (separate edge type)")}
              {check("temporal", "Temporal analysis", !hasTime, hasTime ? undefined : "Requires a mapped timestamp column")}
            </div>
            {hasGroup && <p className="mt-2 text-[11px] text-muted">Group comparisons will run on the mapped group field.</p>}
          </Disclosure>

          <Disclosure title="Graph neural network" desc="Optional GraphSAGE link prediction after the analysis">
            <GnnUploadCard cfg={cfg} setCfg={setCfg} />
          </Disclosure>
      </Disclosure>
    </>
  );
}

function ProgressMonitor({ run }: { run: Run }) {
  const { system, refreshData, toast } = useApp();
  const stages: { id: string; label: string }[] = system?.stages ?? [];
  const retry = async () => {
    try { await api.post(`/api/runs/${run.id}/retry`); await refreshData(); toast("Retrying from the failed stage"); } catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <div className="mt-4 rounded-lg border border-line bg-page/50 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="font-medium">{run.label} <Badge status={run.status} /></div>
        <div className="text-sm tabular-nums text-ink2">{Math.round(run.progress * 100)}% complete</div>
      </div>
      <div className="mt-2 h-1.5 rounded-full bg-raised"><div className={cx("h-1.5 rounded-full transition-all", run.status === "failed" ? "bg-bad" : "bg-accent")} style={{ width: `${Math.max(2, run.progress * 100)}%` }} /></div>
      <ol className="mt-3 grid gap-1 text-sm sm:grid-cols-2 lg:grid-cols-5">
        {stages.map((s) => {
          const done = run.completed_stages?.includes(s.id);
          const cur = run.stage === s.id && run.status === "running";
          const failed = run.failed_stage === s.id && (run.status === "failed" || run.status === "interrupted");
          return (
            <li key={s.id} className={cx("flex items-center gap-1.5", done ? "text-ink" : failed ? "text-bad" : cur ? "text-accent" : "text-muted")}>
              <span aria-hidden>{done ? "✓" : failed ? "✕" : cur ? "◌" : "○"}</span>{s.label}
              {run.stage_timings?.[s.id] !== undefined && <span className="text-xs text-muted">{fmt.sec(run.stage_timings[s.id])}</span>}
            </li>
          );
        })}
      </ol>
      {run.error && (
        <div className="mt-3 space-y-2">
          <ErrorBox error={`Failed at “${run.failed_stage}”: ${run.error}`} />
          <p className="text-xs text-muted">Uploaded data and all completed stages are preserved. Retry resumes from the failed stage.</p>
          <Button onClick={retry}>↻ Retry from {run.failed_stage}</Button>
        </div>
      )}
      {(run.warnings?.length ?? 0) > 0 && (
        <div className="mt-3 rounded-lg bg-warn/10 px-3 py-2 text-sm text-warn"><div className="font-medium">! Warnings</div><ul className="list-disc pl-5">{run.warnings!.map((w) => <li key={w}>{w}</li>)}</ul></div>
      )}
    </div>
  );
}
