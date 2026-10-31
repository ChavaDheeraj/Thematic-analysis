/* eslint-disable @typescript-eslint/no-explicit-any */
// The seven steps of a thematic analysis. Steps 1–6 open one at a time on the Database page;
// step 7, the thematic map, sits under the code graph on Graphs & Patterns.
import { useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { api, download, fmt, qs } from "../api";
import { series } from "../colors";
import { useApp, useFetch } from "../state";
import { Icon, ICONS } from "./icons";
import { CodingStats, Codebook, ThemeStats } from "./RunDetails";
import { Badge, Button, Card, Disclosure, ErrorBox, Loading, PctText, SectionTitle, Stat, Table, Unavailable, cx, inputBase } from "./ui";

export const STEPS = [
  { id: "cleaned", label: "Cleaned data", sub: (p: any) => `${fmt.n(p.records)} records` },
  { id: "notes", label: "Notes", sub: (p: any) => `${fmt.n(p.ideas)} recurring ideas` },
  { id: "coding", label: "Initial coding", sub: (p: any) => `${fmt.n(p.statements)} statements` },
  { id: "codebook", label: "Codebook", sub: (p: any) => `${fmt.n(p.codes)} codes` },
  { id: "categories", label: "Categories", sub: (p: any) => `${fmt.n(p.categories)} groups` },
  { id: "themes", label: "Themes", sub: (p: any) => `${fmt.n(p.themes)} themes · ${fmt.n(p.final)} final` },
  { id: "map", label: "Thematic map", sub: (p: any) => `${fmt.n(p.links)} link${p.links === 1 ? "" : "s"} above chance` },
] as const;
export type Step = (typeof STEPS)[number]["id"];

/** The analysis at a glance: one slim row of numbered steps, each opening where that step lives. */
export function AnalysisPath({ path, active }: { path?: any; active?: Step }) {
  const nav = useNavigate();
  return (
    <nav aria-label="Thematic analysis steps" className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4 xl:grid-cols-7" style={{ boxShadow: "var(--shadow-sm)" }}>
      {STEPS.map((s, i) => (
        <button key={s.id} aria-current={active === s.id ? "step" : undefined}
          onClick={() => nav(s.id === "map" ? "/graphs#thematic-map" : `/database?tab=codes&step=${s.id}`)}
          className="flex min-w-0 items-center gap-2.5 bg-surface px-3.5 py-3 text-left transition-colors last:col-span-2 hover:bg-raised xl:last:col-span-1"
          style={active === s.id ? { boxShadow: "inset 0 -2px 0 rgb(var(--accent))" } : undefined}>
          <span className={cx("flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold", active === s.id ? "bg-accent text-accentfg" : "bg-raised text-ink2")}>{i + 1}</span>
          <span className="min-w-0">
            <span className="block truncate text-[13px] font-semibold text-ink">{s.label}</span>
            <span className="block truncate text-[11px] text-muted">{path ? s.sub(path) : "…"}</span>
          </span>
        </button>
      ))}
    </nav>
  );
}

export function ThematicAnalysis({ d, structure, structureError, runId, step, onCode, onTheme, reload }: {
  d: any; structure: any; structureError: string | null; runId: string; step: Step; onCode: (id: number) => void; onTheme: (id: number) => void; reload: () => void;
}) {
  const needsStructure = (render: (s: any) => ReactNode) => structure ? render(structure) : structureError ? <div className="mt-8"><ErrorBox error={structureError} /></div> : <Loading />;
  return (
    <div className="mt-6">
      <AnalysisPath path={structure?.path} active={step} />
      <div className="[&>:first-child]:mt-8">
        {step === "cleaned" && <Cleaned runId={runId} />}
        {step === "notes" && <Notes runId={runId} onCode={onCode} onTheme={onTheme} />}
        {step === "coding" && (
          <>
            <CodingStats d={d} />
            <Statements runId={runId} codes={d.coding.table} onCode={onCode} />
          </>
        )}
        {step === "codebook" && <Codebook d={d} runId={runId} onOpen={onCode} reload={reload} />}
        {step === "categories" && needsStructure((s) => <Categories s={s} onCode={onCode} reload={reload} />)}
        {step === "themes" && needsStructure((s) => <ThemeSection d={d} s={s} onTheme={onTheme} runId={runId} reload={reload} />)}
      </div>
    </div>
  );
}

function Pager({ page, size, total, onPage }: { page: number; size: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / size));
  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-sm">
      <span className="text-xs text-muted">{fmt.n(total)} rows</span>
      <div className="flex items-center gap-2">
        <Button small disabled={page <= 1} onClick={() => onPage(page - 1)}>‹ Prev</Button>
        <span className="tabular-nums">Page {page} / {pages}</span>
        <Button small disabled={page >= pages} onClick={() => onPage(page + 1)}>Next ›</Button>
      </div>
    </div>
  );
}

function useSearch() {
  const [text, setText] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  useEffect(() => { const t = setTimeout(() => { setSearch(text); setPage(1); }, 300); return () => clearTimeout(t); }, [text]);
  return { text, setText, search, page, setPage };
}

function ExportButton({ runId, format, children }: { runId: string; format: string; children: ReactNode }) {
  const { toast, refreshData } = useApp();
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true);
    try { await download(`/api/runs/${runId}/export/${format}`); refreshData(); } catch (e: any) { toast(e.message, "error"); }
    finally { setBusy(false); }
  };
  return <Button small disabled={busy} onClick={go}><Icon path={ICONS.export} size={13} />{busy ? "Preparing…" : children}</Button>;
}

// ---------------------------------------------------------------- 1 cleaned dataset

function Cleaned({ runId }: { runId: string }) {
  const nav = useNavigate();
  const { text, setText, search, page, setPage } = useSearch();
  const [c, err, loading] = useFetch<any>(`/api/runs/${runId}/cleaned${qs({ page, size: 25, search })}`);
  return (
    <>
      <SectionTitle title="Cleaned dataset" desc="The records this analysis read. Cleaning removes whole records and page furniture such as menus and sharing widgets; it never rewrites anyone's words, so every quotation stays verbatim." />
      {err && <ErrorBox error={err} />}
      {c && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="Records read" value={fmt.n(c.parsed)} detail="from the uploaded files" />
          <Stat label="Removed" value={fmt.n(c.removed_total)} detail={c.removed.length ? c.removed.map((r: any) => `${r.label} ${fmt.n(r.records)}`).join(" · ") : "nothing needed removing"} />
          <Stat label="Cleaned records" value={fmt.n(c.kept)} detail={c.flagged_duplicates ? `${fmt.n(c.flagged_duplicates)} flagged duplicates kept` : "used in the analysis"} />
          <Stat label="Passages" value={fmt.n(c.segments)} detail="split for coding" />
        </div>
      )}
      <Card className="mt-4" title="Cleaned records" subtitle="Original text, exactly as uploaded"
        actions={<>
          <input className={cx(inputBase, "w-44")} placeholder="Search text" value={text} onChange={(e) => setText(e.target.value)} />
          <ExportButton runId={runId} format="cleaned_csv">Cleaned dataset (CSV)</ExportButton>
          <Button small kind="ghost" onClick={() => nav("/database?tab=files")}>All records</Button>
        </>}>
        {loading && !c && <Loading />}
        {c && (
          <>
            <Table<any> rows={c.rows} rowKey={(r) => r.id} maxHeight={480} empty="No records match."
              cols={[
                { key: "text", label: "Text", render: (r) => <div className="max-w-2xl"><div className="line-clamp-2">{r.text}</div><div className="mt-0.5 truncate text-[11px] text-muted" title={r.source_file}>{r.source_file}</div></div> },
                { key: "word_count", label: "Words", num: true },
                { key: "passages", label: "Passages", num: true },
                { key: "is_duplicate", label: "", render: (r) => r.is_duplicate ? <Badge status="needs_revision">duplicate</Badge> : null },
              ]} />
            <Pager page={page} size={c.size} total={c.total} onPage={setPage} />
          </>
        )}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- 2 notes

function Notes({ runId, onCode, onTheme }: { runId: string; onCode: (id: number) => void; onTheme: (id: number) => void }) {
  const { toast } = useApp();
  const [ideas, iErr] = useFetch<any>(`/api/runs/${runId}/ideas`);
  const [notes, , , reload] = useFetch<any>(`/api/runs/${runId}/notes`);
  const [draft, setDraft] = useState("");
  const add = async () => {
    try { await api.post(`/api/runs/${runId}/notes`, { text: draft }); setDraft(""); reload(); toast("Note saved", "success"); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const del = async (id: number) => {
    try { await api.del(`/api/notes/${id}`); reload(); } catch (e: any) { toast(e.message, "error"); }
  };
  const written = (notes?.codes.length ?? 0) + (notes?.themes.length ?? 0);
  return (
    <>
      <SectionTitle title="Notes" desc="Understanding the data before coding it: the ideas that keep recurring, and your own notes on them." />
      <div className="grid items-start gap-4 lg:grid-cols-[1.5fr_1fr]">
        <Card title="Recurring ideas" subtitle={ideas?.note}>
          {iErr && <ErrorBox error={iErr} />}
          {!ideas && !iErr && <Loading />}
          {ideas && !ideas.available && <Unavailable reason={ideas.reason} />}
          {ideas?.available && (
            <ul className="-my-3 divide-y divide-line">
              {ideas.ideas.map((x: any) => (
                <li key={x.topic} className="py-3">
                  <div className="flex items-baseline justify-between gap-3">
                    <div className="min-w-0 text-sm font-semibold text-ink">{x.terms.slice(0, 4).join(" · ")}</div>
                    <div className="shrink-0 text-xs tabular-nums text-muted">{fmt.pct(x.share.value, 0)} of passages{ideas.n_files > 1 ? ` · ${x.files} files` : ""}</div>
                  </div>
                  {x.quote && <blockquote className="mt-1.5 border-l-2 border-line pl-3 text-sm italic leading-relaxed text-ink2">“{x.quote.text}”</blockquote>}
                  <button className="mt-1.5 text-xs font-medium text-accent hover:underline" onClick={() => setDraft((t) => `${t}${t ? "\n" : ""}${x.terms.slice(0, 3).join(", ")}: `)}>Note this idea</button>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card title="Your notes" subtitle="Kept with this analysis and included in reports">
          <textarea className={cx(inputBase, "h-28 w-full resize-y")} placeholder="What stands out? What keeps coming back?" value={draft} onChange={(e) => setDraft(e.target.value)} />
          <div className="mt-2"><Button small kind="primary" disabled={!draft.trim()} onClick={add}>Save note</Button></div>
          {notes && notes.memos.length === 0 && written === 0 && <p className="mt-3 text-sm text-muted">No notes yet.</p>}
          {notes && notes.memos.length > 0 && (
            <ul className="mt-3 space-y-2">
              {notes.memos.map((n: any) => (
                <li key={n.id} className="rounded-lg bg-raised px-3 py-2 text-sm text-ink">
                  <div className="whitespace-pre-wrap">{n.note}</div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-muted">
                    <span>{fmt.date(n.created_at)}</span>
                    <button className="hover:text-bad" onClick={() => del(n.id)}>Delete</button>
                  </div>
                </li>
              ))}
            </ul>
          )}
          {written > 0 && (
            <div className="mt-4 border-t border-line pt-3">
              <div className="text-xs font-medium text-ink2">Written on codes and themes</div>
              <ul className="mt-1.5 space-y-1.5 text-sm">
                {notes.codes.map((c: any) => <li key={`c${c.id}`}><button className="font-medium text-accent hover:underline" onClick={() => onCode(c.id)}>{c.label}</button> <span className="text-ink2">{c.notes}</span></li>)}
                {notes.themes.map((t: any) => <li key={`t${t.id}`}><button className="font-medium text-accent hover:underline" onClick={() => onTheme(t.id)}>{t.label}</button> <span className="text-ink2">{t.notes}</span></li>)}
              </ul>
            </div>
          )}
        </Card>
      </div>
      <Disclosure title="Topic model table" desc="The word lists behind the recurring ideas, with coherence scores"><TopicsTable runId={runId} /></Disclosure>
    </>
  );
}

function TopicsTable({ runId }: { runId: string }) {
  const [topics] = useFetch<any>(`/api/runs/${runId}/topics`);
  if (!topics) return <Loading />;
  if (!topics.topics.length) return <Unavailable reason={topics.metrics?.reason} />;
  return (
    <Table<any> rows={topics.topics} rowKey={(r) => r.topic} maxHeight={360}
      cols={[
        { key: "topic", label: "#", num: true, render: (r) => r.topic + 1 }, { key: "terms", label: "Top terms", render: (r) => <span className="text-xs">{r.terms.slice(0, 8).join(", ")}</span> },
        { key: "size", label: "Passages", num: true }, { key: "share", label: "Share", render: (r) => <PctText p={r.share} /> },
        { key: "coherence", label: "UMass", num: true, render: (r) => fmt.n(r.coherence, 3) },
      ]} />
  );
}

// ---------------------------------------------------------------- 3 initial coding

function Statements({ runId, codes, onCode }: { runId: string; codes: any[]; onCode: (id: number) => void }) {
  const { text, setText, search, page, setPage } = useSearch();
  const [code, setCode] = useState("");
  const [s, err] = useFetch<any>(`/api/runs/${runId}/statements${qs({ page, size: 25, search, code_id: code })}`);
  const active = codes.filter((c) => c.status === "candidate" || c.status === "approved").sort((a, b) => a.label.localeCompare(b.label));
  return (
    <Card className="mt-4" title="Meaningful statements and their codes"
      subtitle={s ? `${fmt.n(s.coded)} of ${fmt.n(s.segments)} passages hold at least one code. Each row is a verbatim statement; click a code to see all its passages.` : undefined}
      actions={<>
        <input className={cx(inputBase, "w-44")} placeholder="Search statements" value={text} onChange={(e) => setText(e.target.value)} />
        <select aria-label="Filter by code" className={cx(inputBase, "w-48")} value={code} onChange={(e) => { setCode(e.target.value); setPage(1); }}>
          <option value="">All codes</option>{active.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
        </select>
        <ExportButton runId={runId} format="assignments_csv">Initial coding (CSV)</ExportButton>
      </>}>
      {err && <ErrorBox error={err} />}
      {!s && !err && <Loading />}
      {s && (
        <>
          <Table<any> rows={s.rows} rowKey={(r) => r.id} maxHeight={560} empty="No statements match."
            cols={[
              { key: "text", label: "Statement", render: (r) => <div className="max-w-2xl"><div>{r.text}</div><div className="mt-0.5 truncate text-[11px] text-muted" title={r.source_file}>{r.source_file}</div></div> },
              { key: "codes", label: "Codes", render: (r) => (
                <div className="flex max-w-xs flex-wrap gap-1">
                  {r.codes.map((c: any) => (
                    <button key={c.id} onClick={() => onCode(c.id)} title={c.matched_text ? `matched “${c.matched_text}”${c.negated ? " · negation nearby" : ""}` : undefined}
                      className="rounded-full bg-raised px-2 py-0.5 text-[11px] font-medium text-ink2 transition-colors hover:bg-accent/15 hover:text-accent">{c.label}</button>
                  ))}
                </div>
              ) },
            ]} />
          <Pager page={page} size={s.size} total={s.total} onPage={setPage} />
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------- 5 categories

function CodeChip({ c, onClick }: { c: any; onClick: () => void }) {
  return (
    <button onClick={onClick} title={c.suggested ? "Suggested by text similarity, not yet assigned" : undefined}
      className={cx("rounded-full border px-2 py-0.5 text-[11px] font-medium text-ink2 transition-colors hover:border-accent hover:text-accent", c.suggested ? "border-dashed border-muted/60" : "border-line bg-raised")}>
      {c.label}
    </button>
  );
}

function Categories({ s, onCode, reload }: { s: any; onCode: (id: number) => void; reload: () => void }) {
  const { toast } = useApp();
  const max = Math.max(1, ...s.categories.map((c: any) => c.segments));
  const accept = async (c: any) => {
    try {
      await Promise.all(c.codes.filter((x: any) => x.suggested).map((x: any) => api.patch(`/api/codes/${x.id}`, { category: c.name })));
      toast(`Added to ${c.name}`, "success");
      reload();
    } catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <>
      <SectionTitle title="Categories" desc="Similar codes grouped together. Solid codes are assigned to the category. Dashed codes are suggestions matched by text similarity: accept them here, or open a code to choose its category." />
      {!s.categories.length && !s.uncategorised.length ? <Unavailable reason="No codes in this analysis." /> : (
        <div className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-3">
          {s.categories.map((c: any) => (
            <article key={c.name} className="rounded-xl border border-line bg-surface p-4" style={{ boxShadow: "var(--shadow-sm)" }}>
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-semibold text-ink">{c.name}</h3>
                {c.suggested && <Badge status="candidate">suggested</Badge>}
              </div>
              <div className="mt-0.5 text-xs text-muted">{fmt.n(c.segments)} passages · {fmt.pct(c.share.value, 0)} · {c.codes.length} codes</div>
              <div className="mt-2 h-1.5 rounded-full bg-raised"><div className="h-1.5 rounded-full bg-accent" style={{ width: `${Math.max(2, 100 * c.segments / max)}%` }} /></div>
              <div className="mt-3 flex flex-wrap gap-1">{c.codes.map((x: any) => <CodeChip key={x.id} c={x} onClick={() => onCode(x.id)} />)}</div>
              {c.n_suggested > 0 && <button className="mt-2.5 text-xs font-medium text-accent hover:underline" onClick={() => accept(c)}>Accept {c.n_suggested} suggestion{c.n_suggested > 1 ? "s" : ""}</button>}
            </article>
          ))}
          {s.uncategorised.length > 0 && (
            <article className="rounded-xl border border-dashed border-line p-4">
              <h3 className="text-sm font-semibold text-ink">Not yet categorised</h3>
              <div className="mt-0.5 text-xs text-muted">{s.uncategorised.length} codes too unlike any group to suggest one</div>
              <div className="mt-3 flex flex-wrap gap-1">{s.uncategorised.map((x: any) => <CodeChip key={x.id} c={{ ...x, suggested: false }} onClick={() => onCode(x.id)} />)}</div>
            </article>
          )}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------- 6 themes and sub-themes

const names = (codes: any[], n = 6) => `${codes.slice(0, n).map((c) => c.label).join(", ")}${codes.length > n ? ` +${codes.length - n}` : ""}`;

function ThemeSection({ d, s, onTheme, runId, reload }: { d: any; s: any; onTheme: (id: number) => void; runId: string; reload: () => void }) {
  return (
    <>
      <ThemeStats d={d} s={s} runId={runId} reload={reload} />
      <div className="mt-4 grid items-start gap-4 lg:grid-cols-2">
        {s.themes.map((t: any) => (
          <article key={t.id} className={cx("rounded-xl border border-line bg-surface p-5", t.status === "rejected" && "opacity-60")} style={{ boxShadow: "var(--shadow-sm)", borderTop: `3px solid ${series(t.rank)}` }}>
            <div className="flex items-start justify-between gap-3">
              <h3 className="text-[15px] font-semibold text-ink">{t.label}</h3>
              <Badge status={t.status}>{t.status === "approved" ? "final" : undefined}</Badge>
            </div>
            <div className="mt-1 text-xs text-muted">{fmt.pct(t.prevalence.segment.value, 0)} of passages · {fmt.n(t.segments)} passages · {t.codes.length} codes</div>
            {t.origin !== "machine" && t.definition && <p className="mt-2 text-sm text-ink2">{t.definition}</p>}
            {t.subthemes.length > 0 && (
              <ul className="mt-3 space-y-2">
                {t.subthemes.map((x: any) => (
                  <li key={x.id ?? x.label} className="border-l-2 pl-3" style={{ borderColor: series(t.rank) }}>
                    <div className="text-sm font-medium text-ink">
                      {x.id ? <button className="hover:underline" onClick={() => onTheme(x.id)}>{x.label}</button> : x.label}
                      <span className="ml-1.5 text-xs font-normal text-muted">{x.codes.length} codes</span>
                      {x.status === "rejected" && <span className="ml-1.5"><Badge status="rejected" /></span>}
                    </div>
                    <div className="text-xs leading-relaxed text-ink2">{names(x.codes)}</div>
                  </li>
                ))}
              </ul>
            )}
            {t.other_codes.length > 0 && <p className="mt-2 text-xs text-muted">{t.subthemes.length ? "Other codes" : "Codes"}: {names(t.other_codes, 8)}</p>}
            <button onClick={() => onTheme(t.id)} className="mt-3 flex items-center gap-1 text-xs font-semibold text-accent hover:underline">
              Open evidence and review<Icon path={ICONS.arrow} size={12} />
            </button>
          </article>
        ))}
      </div>
      {!s.themes.length && <Unavailable reason="No themes yet. Larger files, or a lower Min NPMI under Advanced options, give more themes." />}
    </>
  );
}
