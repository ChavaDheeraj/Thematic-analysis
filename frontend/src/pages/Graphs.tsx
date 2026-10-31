/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, download, fmt, qs, type Run } from "../api";
import { Chart, axis, baseOption, esc, lineStyle } from "../components/Chart";
import { CodeDrawer } from "../components/Evidence";
import { GnnSection, type PredictedLink } from "../components/Gnn";
import { Icon, ICONS } from "../components/icons";
import { ThematicMap } from "../components/ThematicMap";
import { Badge, Button, Card, Disclosure, Empty, ErrorBox, Field, Loading, PageHeader, SectionTitle, Stat, Table, Unavailable, cx, inputBase, inputCls } from "../components/ui";
import { chrome, series } from "../colors";
import { useApp, useFetch } from "../state";

const SAVED_KEY = "tml.graphConfig";

interface GParams { level: string; min_cooccurrence: number; min_npmi: number; similarity: boolean; similarity_threshold: number; method: string; resolution: number; include_isolated: boolean }

export default function GraphsPage() {
  const { project, dataset, run } = useApp();
  const nav = useNavigate();
  if (!project) return <Empty title="No project selected">Create or select a project first.</Empty>;
  const ok = run?.status === "completed";
  return (
    <div className="mx-auto max-w-7xl">
      <PageHeader title="Graphs & Patterns" desc="How the codes of one analysed file connect, how its themes relate, which groups they form, and how they change over time." />
      {ok ? <Graphs key={run!.id} run={run!} /> : (
        <Empty title={dataset ? `No completed analysis of ${dataset.filename}` : "No analysed file"}>
          Graphs are built from a completed analysis of one file. Analyse the file, or pick another file or analysis on the Database page.
          <div className="mt-4 flex justify-center gap-2">
            <Button kind="primary" onClick={() => nav("/upload")}>Analyse a file</Button>
            <Button onClick={() => nav("/database")}>Choose another file</Button>
          </div>
        </Empty>
      )}
    </div>
  );
}

/** Which file and analysis this page shows; they are chosen on the Database page. */
function Scope({ run }: { run: Run }) {
  const { datasets } = useApp();
  const nav = useNavigate();
  const names = run.dataset_ids.map((id) => datasets.find((d) => d.id === id)?.filename ?? id);
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl border border-line bg-surface px-4 py-2.5 text-sm" style={{ boxShadow: "var(--shadow-sm)" }}>
      <span className="text-accent"><Icon path={ICONS.file} size={15} /></span>
      <span className="font-medium text-ink" title={names.join(", ")}>{names.length > 1 ? `Combined analysis · ${names.length} files` : names[0]}</span>
      <span className="text-muted">·</span>
      <span className="text-ink2">{run.label}</span>
      <button className="ml-auto flex items-center gap-1 text-xs font-medium text-accent hover:underline" onClick={() => nav("/database")}>
        Change file or analysis on the Database page<Icon path={ICONS.arrow} size={12} />
      </button>
    </div>
  );
}

function Graphs({ run }: { run: Run }) {
  const { project, system } = useApp();
  const [allRuns] = useFetch<Run[]>(`/api/projects/${project!.id}/runs`, [run.id]);
  const completed = (allRuns ?? []).filter((r) => r.status === "completed");
  const saved = (() => { try { return JSON.parse(localStorage.getItem(SAVED_KEY) ?? "null"); } catch { return null; } })();
  const [p, setP] = useState<GParams>(() => ({
    level: run.config.cooccurrence_level, min_cooccurrence: run.config.min_cooccurrence, min_npmi: run.config.min_npmi ?? 0,
    similarity: run.config.similarity_edges, similarity_threshold: run.config.similarity_threshold, method: run.config.graph_method,
    resolution: run.config.resolution, include_isolated: true, ...(saved ?? {}),
  }));
  const [applied, setApplied] = useState<GParams>(p);
  const url = `/api/runs/${run.id}/graph${qs({ ...applied } as any)}`;
  const [g, err, loading, reload] = useFetch<any>(url);
  const [codeId, setCodeId] = useState<number | null>(null);
  const [predicted, setPredicted] = useState<PredictedLink[] | null>(null);

  return (
    <>
      <Scope run={run} />

      <Disclosure title="Graph settings" desc="Co-occurrence unit, edge filters and community detection — recomputed live from the current (researcher-edited) codes">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="Co-occurrence unit"><select className={inputCls} value={p.level} onChange={(e) => setP({ ...p, level: e.target.value })}><option value="record">Record</option><option value="segment">Segment</option></select></Field>
          <Field label="Min co-occurrence"><input type="number" min={1} className={inputCls} value={p.min_cooccurrence} onChange={(e) => setP({ ...p, min_cooccurrence: Number(e.target.value) })} /></Field>
          <Field label="Min NPMI" hint=">0 = above chance"><input type="number" step={0.05} min={-1} max={1} className={inputCls} value={p.min_npmi} onChange={(e) => setP({ ...p, min_npmi: Number(e.target.value) })} /></Field>
          <Field label="Similarity edges"><select className={inputCls} value={String(p.similarity)} onChange={(e) => setP({ ...p, similarity: e.target.value === "true" })}><option value="true">Include</option><option value="false">Exclude</option></select></Field>
          <Field label="Similarity threshold"><input type="number" step={0.05} min={0} max={1} className={inputCls} value={p.similarity_threshold} onChange={(e) => setP({ ...p, similarity_threshold: Number(e.target.value) })} /></Field>
          <Field label="Community algorithm"><select className={inputCls} value={p.method} onChange={(e) => setP({ ...p, method: e.target.value })}>
            <option value="louvain">Louvain</option>{system?.optional?.leiden && <option value="leiden">Leiden</option>}<option value="greedy">Greedy modularity</option><option value="label_propagation">Label propagation</option>
          </select></Field>
          <Field label="Resolution"><input type="number" step={0.1} min={0.1} max={10} className={inputCls} value={p.resolution} onChange={(e) => setP({ ...p, resolution: Number(e.target.value) })} /></Field>
          <Field label="Isolated codes"><select className={inputCls} value={String(p.include_isolated)} onChange={(e) => setP({ ...p, include_isolated: e.target.value === "true" })}><option value="true">Show</option><option value="false">Hide</option></select></Field>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Button kind="primary" onClick={() => setApplied(p)}>Apply</Button>
          <Button onClick={() => { const d = { level: run.config.cooccurrence_level, min_cooccurrence: run.config.min_cooccurrence, min_npmi: run.config.min_npmi ?? 0, similarity: run.config.similarity_edges, similarity_threshold: run.config.similarity_threshold, method: run.config.graph_method, resolution: run.config.resolution, include_isolated: true }; setP(d); setApplied(d); }}>Reset to run configuration</Button>
          <Button kind="ghost" onClick={() => { try { localStorage.setItem(SAVED_KEY, JSON.stringify(p)); } catch { /* storage unavailable */ } }}>Save configuration for reuse</Button>
        </div>
      </Disclosure>

      {loading && !g && <Loading label="Building graph…" />}
      {err && <ErrorBox error={err} />}
      {g && (
        <>
          <GraphExplorer g={g} onOpenCode={setCodeId} reloading={loading} predicted={predicted} />
          <ThematicMap runId={run.id} />
          <Clusters runId={run.id} />
        </>
      )}
      <TemporalGraph runId={run.id} />

      <h2 className="mb-1 mt-12 text-lg font-semibold tracking-tight text-ink">More analysis</h2>
      <p className="text-[13px] text-muted">Network statistics, comparisons, the optional neural model and data export.</p>
      {g && <Disclosure title="Network statistics" desc={g.note}><Network g={g} /></Disclosure>}
      <Disclosure title="Compare with another analysis" desc="This analysis (A) against another completed analysis (B) — of this file with other settings, or of another file">
        <AcrossUploads run={run} runs={completed} />
      </Disclosure>
      <Disclosure title="Model comparison" desc="Different configurations on the same file, side by side; a cleaner graph or more clusters is not evidence of a better model">
        <ModelComparison run={run} runs={completed} />
      </Disclosure>
      <Disclosure title="Graph neural network (GraphSAGE)" desc="Optional neural module: predicts unobserved code relationships; predictions are hypotheses, never observed facts">
        <GnnSection run={run} g={g} onPredicted={setPredicted} onOpenCode={setCodeId} />
      </Disclosure>
      <Disclosure title="Export graph data" desc="Nodes, edges, GraphML, cluster assignments, temporal results">
        <GraphExport runId={run.id} g={g} />
      </Disclosure>
      <CodeDrawer codeId={codeId} onClose={() => setCodeId(null)} onChanged={reload} />
    </>
  );
}

// ---------------------------------------------------------------- B
function GraphExplorer({ g, onOpenCode, reloading, predicted }: { g: any; onOpenCode: (id: number) => void; reloading: boolean; predicted: PredictedLink[] | null }) {
  const [edgeType, setEdgeType] = useState<"cooccurrence" | "similarity" | "both">("cooccurrence");
  const [minW, setMinW] = useState(1);
  const [labels, setLabels] = useState(true);
  const [layout, setLayout] = useState<"force" | "circular">("force");
  const [focus, setFocus] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [expand, setExpand] = useState(false);
  const [resetTick, setResetTick] = useState(0);

  const comms: number[] = useMemo(() => g.communities.filter((c: any) => c.size > 1).map((c: any) => c.community), [g]);
  const commColor = (c: number) => { const i = comms.indexOf(c); return series(i >= 0 && i < 8 ? i : -1); };
  const maxW = Math.max(1, ...g.edges.filter((e: any) => e.type === "cooccurrence").map((e: any) => e.weight));
  const edges = g.edges.filter((e: any) => (edgeType === "both" || e.type === edgeType) && (e.type === "similarity" || e.weight >= minW));
  const neighbours = useMemo(() => {
    if (focus === null) return null;
    const s = new Set<number>([focus]);
    for (const e of edges) { if (e.source === focus) s.add(e.target); if (e.target === focus) s.add(e.source); }
    if (expand) for (const e of edges) { if (s.has(e.source) || s.has(e.target)) { s.add(e.source); s.add(e.target); } }
    return s;
  }, [focus, edges, expand]);
  const maxF = Math.max(1, ...g.nodes.map((n: any) => n.frequency));
  const byId = new Set<number>(g.nodes.map((n: any) => n.id));
  const node = g.nodes.find((n: any) => n.id === focus);

  const opt = useMemo(() => () => {
    const c = chrome();
    return {
      ...baseOption(),
      tooltip: { ...baseOption().tooltip, formatter: (x: any) => x.dataType === "edge"
        ? (x.data.kind === "predicted" ? `<b>${esc(x.data.sl)}</b> — <b>${esc(x.data.tl)}</b><br/>PREDICTED by GraphSAGE (rank ${x.data.rank}, score ${x.data.w})<br/>Not an observed relationship`
          : `<b>${esc(x.data.sl)}</b> — <b>${esc(x.data.tl)}</b><br/>${esc(x.data.kind)}: ${x.data.kind === "cooccurrence" ? `${x.data.w} co-occurrences · NPMI ${x.data.npmi ?? "—"}` : `cosine ${x.data.w}`}`)
        : `<b>${esc(x.data.name)}</b><br/>${x.data.f} segments · ${fmt.pct(x.data.rp)} respondents<br/>community ${x.data.c} · degree ${x.data.deg}` },
      legend: { show: false },
      animationDurationUpdate: 300,
      series: [{
        type: "graph", layout, roam: true, draggable: true,
        force: { repulsion: 220, edgeLength: [40, 160], gravity: 0.08, friction: 0.2 },
        circular: { rotateLabel: true },
        label: { show: labels, position: "right", color: c.ink2, fontSize: 11 },
        labelLayout: { hideOverlap: true },
        emphasis: { focus: "adjacency", lineStyle: { width: 3 } },
        data: g.nodes.map((n: any) => ({
          id: String(n.id), name: n.label, f: n.frequency, rp: n.respondent_pct, c: n.community, deg: n.degree,
          symbolSize: 8 + 26 * Math.sqrt(n.frequency / maxF),
          itemStyle: { color: commColor(n.community), borderColor: c.surface, borderWidth: 2, opacity: neighbours && !neighbours.has(n.id) ? 0.15 : 1 },
          label: { show: labels && (!neighbours || neighbours.has(n.id)) },
        })),
        links: [...edges, ...(predicted ?? []).filter((p) => byId.has(p.source) && byId.has(p.target)).map((p) => ({ source: p.source, target: p.target, weight: p.score, type: "predicted", rank: p.rank }))].map((e: any) => ({
          source: String(e.source), target: String(e.target), w: e.weight, npmi: e.npmi, kind: e.type, rank: e.rank,
          sl: g.nodes.find((n: any) => n.id === e.source)?.label, tl: g.nodes.find((n: any) => n.id === e.target)?.label,
          lineStyle: {
            color: e.type === "predicted" ? series(6) : e.type === "similarity" ? series(1) : c.muted,
            type: e.type === "predicted" ? "dotted" : e.type === "similarity" ? "dashed" : "solid",
            width: e.type === "predicted" ? 2 : e.type === "similarity" ? 1 : 0.5 + 3 * (e.weight / maxW), curveness: e.type === "cooccurrence" ? 0 : 0.2,
            opacity: neighbours && !(neighbours.has(e.source) && neighbours.has(e.target)) ? 0.04 : e.type === "predicted" ? 0.9 : 0.45,
          },
        })),
      }],
    };
  }, [g, edges, labels, layout, neighbours, resetTick, predicted]); // eslint-disable-line react-hooks/exhaustive-deps

  const doSearch = () => {
    const hit = g.nodes.find((n: any) => n.label.toLowerCase().includes(search.toLowerCase()));
    if (hit) setFocus(hit.id);
  };
  const connected = node ? g.edges.filter((e: any) => e.source === node.id || e.target === node.id)
    .map((e: any) => ({ ...e, other: g.nodes.find((n: any) => n.id === (e.source === node.id ? e.target : e.source)) }))
    .filter((e: any) => e.other).sort((a: any, b: any) => b.weight - a.weight) : [];

  return (
    <>
      <SectionTitle id="explorer" title="Graph explorer" desc="Nodes = codes (size = segments). Edges: solid gray = observed co-occurrence (width = count); dashed orange = TF-IDF similarity. Colour = community." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <Card title="Concept relationship graph" subtitle={`${g.nodes.length} nodes · ${edges.length} of ${g.edges.length} edges shown${reloading ? " · updating…" : ""}`}
          actions={<>
            <input className={cx(inputBase, "w-40")} placeholder="Find a code" value={search} onChange={(e) => setSearch(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doSearch()} />
            <Button small onClick={doSearch}>Focus</Button>
          </>}>
          <div className="mb-3 flex flex-wrap items-end gap-3 text-sm">
            <Field label="Edge type"><select className={cx(inputBase, "w-40")} value={edgeType} onChange={(e) => setEdgeType(e.target.value as any)}>
              <option value="cooccurrence">Co-occurrence</option><option value="similarity">Similarity</option><option value="both">Both (distinct styles)</option></select></Field>
            <Field label={`Min co-occurrence weight: ${minW}`}><input type="range" min={1} max={maxW} value={minW} onChange={(e) => setMinW(Number(e.target.value))} className="w-40" /></Field>
            <Field label="Layout"><select className={cx(inputBase, "w-28")} value={layout} onChange={(e) => setLayout(e.target.value as any)}><option value="force">Force</option><option value="circular">Circular</option></select></Field>
            <label className="flex items-center gap-1.5 pb-2"><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />Labels</label>
            <label className="flex items-center gap-1.5 pb-2"><input type="checkbox" checked={expand} onChange={(e) => setExpand(e.target.checked)} disabled={focus === null} />Expand neighbours (2 hops)</label>
            <Button small kind="ghost" onClick={() => { setFocus(null); setMinW(1); setResetTick((t) => t + 1); }}>Reset view</Button>
          </div>
          {g.nodes.length ? <Chart option={opt} height={560} name="code-graph" onClick={(x) => x.dataType === "node" && setFocus(Number(x.data.id))} />
            : <Unavailable reason="No codes in this run." />}
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-ink2">
            {comms.slice(0, 8).map((c, i) => <span key={c} className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: series(i) }} />Community {c}</span>)}
            {comms.length > 8 && <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: series(-1) }} />Other communities ({comms.length - 8})</span>}
            <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: series(-1) }} />Singletons</span>
            {predicted && <span className="flex items-center gap-1 font-medium"><span className="inline-block w-5 border-t-2 border-dotted" style={{ borderColor: series(6) }} />{predicted.length} GNN-predicted links shown (not observed)</span>}
          </div>
        </Card>
        <Card title={node ? node.label : "Node details"} subtitle={node ? `Community ${node.community} · ${node.method}` : "Click a node to see its details"}>
          {!node ? <p className="text-sm text-muted">Select a node in the graph or search for a code.</p> : (
            <div className="space-y-3 text-sm">
              <dl className="grid grid-cols-[1fr_auto] gap-y-1 tabular-nums">
                <dt className="text-muted">Segments</dt><dd>{fmt.n(node.frequency)}</dd>
                <dt className="text-muted">Records</dt><dd>{fmt.n(node.records)} ({fmt.pct(node.record_pct)})</dd>
                <dt className="text-muted">Respondents</dt><dd>{fmt.n(node.respondents)} ({fmt.pct(node.respondent_pct)})</dd>
                <dt className="text-muted">Degree / weighted</dt><dd>{node.degree} / {fmt.n(node.weighted_degree)}</dd>
                <dt className="text-muted">Betweenness</dt><dd>{fmt.n(node.betweenness, 4)}</dd>
                <dt className="text-muted">PageRank</dt><dd>{fmt.n(node.pagerank, 4)}</dd>
              </dl>
              {node.themes.length > 0 && <div><div className="text-xs font-medium text-ink2">Themes</div>{node.themes.map((t: any) => <div key={t.id} className="flex items-center gap-2">{t.label} <Badge status={t.status} /></div>)}</div>}
              <div>
                <div className="mb-1 text-xs font-medium text-ink2">Connected codes</div>
                <div className="max-h-56 overflow-y-auto">
                  <table className="w-full text-xs tabular-nums">
                    <tbody>{connected.slice(0, 40).map((e: any, i: number) => (
                      <tr key={i} className="cursor-pointer border-t border-line hover:bg-raised" onClick={() => setFocus(e.other.id)}>
                        <td className="py-1">{e.other.label}</td><td className="text-muted">{e.type === "similarity" ? "sim." : "co-occ."}</td><td className="text-right">{e.weight}</td>
                      </tr>))}</tbody>
                  </table>
                </div>
              </div>
              <Button small kind="primary" onClick={() => onOpenCode(node.id)}>Supporting passages →</Button>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- C
function Clusters({ runId }: { runId: string }) {
  const { toast } = useApp();
  const [data, err, loading, reload] = useFetch<any>(`/api/runs/${runId}/clusters`);
  const [themes] = useFetch<any[]>(`/api/runs/${runId}/themes`);
  const [sel, setSel] = useState<number | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [rename, setRename] = useState("");
  useEffect(() => { if (data?.clusters?.length && (sel === null || !data.clusters.find((c: any) => c.cluster === sel))) setSel(data.clusters[0].cluster); }, [data]); // eslint-disable-line react-hooks/exhaustive-deps
  const c = data?.clusters?.find((x: any) => x.cluster === sel);
  useEffect(() => setRename(c?.label ?? ""), [c?.label]);
  const act = async (fn: () => Promise<unknown>, msg: string) => { try { await fn(); toast(msg, "success"); setPicked([]); reload(); } catch (e: any) { toast(e.message, "error"); } };
  const m = data?.metrics;
  return (
    <>
      <SectionTitle id="clusters" title="Pattern clustering" desc={m?.available ? `${m.algorithm} · ${m.n_clusters} clusters · silhouette ${fmt.n(m.silhouette_cosine, 3)} · ${m.representation}` : undefined} />
      {loading && !data && <Loading />}
      {err && <ErrorBox error={err} />}
      {data && !m?.available && <Unavailable reason={m?.reason} />}
      {data && m?.available && (
        <div className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
          <Card title="Clusters" subtitle={`Unassigned (no extracted terms): ${data.unassigned.numerator} of ${data.unassigned.denominator} segments (${fmt.pct(data.unassigned.value)})`}
            actions={picked.length >= 2 ? <Button small onClick={() => act(() => api.post(`/api/runs/${runId}/clusters/merge`, { clusters: picked }), "Clusters merged")}>Merge {picked.length} selected</Button> : undefined}>
            <Table<any> rows={data.clusters} rowKey={(r) => r.cluster} onRow={(r) => setSel(r.cluster)} maxHeight={420} initialSort={{ key: "passages", dir: -1 }}
              cols={[
                { key: "pick", label: "", width: "28px", render: (r) => <input type="checkbox" checked={picked.includes(r.cluster)} onClick={(e) => e.stopPropagation()} onChange={(e) => setPicked(e.target.checked ? [...picked, r.cluster] : picked.filter((x) => x !== r.cluster))} /> },
                { key: "label", label: "Cluster", render: (r) => <span className={cx(r.cluster === sel && "font-semibold")}>{r.label}</span> },
                { key: "passages", label: "Passages", num: true }, { key: "respondents", label: "Respondents", num: true },
                { key: "n_codes", label: "Codes", num: true }, { key: "cohesion", label: "Cohesion", num: true, render: (r) => fmt.n(r.cohesion, 3) },
              ]} />
            <p className="mt-2 text-[11px] text-muted">Cohesion = mean cosine similarity of passages to the cluster centroid (LSA space).</p>
          </Card>
          {c && (
            <Card title={c.label} subtitle={`Cluster ${c.cluster} · ${c.passages} passages · ${c.records} records · ${c.respondents} respondents`}>
              <div className="flex flex-wrap items-end gap-2">
                <Field label="Rename"><input className={cx(inputBase, "w-56")} value={rename} onChange={(e) => setRename(e.target.value)} /></Field>
                <Button small onClick={() => act(() => api.patch(`/api/runs/${runId}/clusters/${c.cluster}`, { label: rename }), "Cluster renamed")}>Save</Button>
                <Field label="Link to theme"><select className={cx(inputBase, "w-48")} value={c.theme_id ?? ""} onChange={(e) => e.target.value && act(() => api.patch(`/api/runs/${runId}/clusters/${c.cluster}`, { theme_id: Number(e.target.value) }), "Cluster linked to theme")}>
                  <option value="">— none —</option>{(themes ?? []).map((t: any) => <option key={t.id} value={t.id}>{t.label}</option>)}</select></Field>
                <Button small onClick={() => act(() => api.post(`/api/runs/${runId}/clusters/${c.cluster}/split?k=2`), "Cluster split in two")}>Split (K-means, k=2)</Button>
              </div>
              <div className="mt-3 text-xs font-medium text-ink2">Most distinctive terms</div>
              <div className="mt-1 flex flex-wrap gap-1">{(c.terms ?? []).map((t: any) => <span key={t.term} className="rounded bg-raised px-1.5 py-0.5 text-xs" title={`distinctiveness ${t.distinctiveness}`}>{t.term}</span>)}</div>
              <div className="mt-3 text-xs font-medium text-ink2">Top codes in this cluster</div>
              <div className="mt-1 text-xs text-ink">{c.top_codes.map((x: any) => `${x.label} (${x.segments})`).join(" · ") || "—"}</div>
              <div className="mt-3 text-xs font-medium text-ink2">Representative passages (closest to centroid)</div>
              <div className="mt-1 space-y-2">{c.representative_passages.map((q: any) => (
                <div key={q.segment_id} className="border-l-2 border-line pl-3 text-sm">{q.text}<div className="text-[11px] text-muted">{q.record_key} · {q.source_file} · similarity {q.similarity} · {q.verified ? "✓ verbatim" : "✕ mismatch"}</div></div>
              ))}</div>
            </Card>
          )}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------- D
function Network({ g }: { g: any }) {
  const s = g.stats;
  const top = (k: "degree" | "betweenness" | "pagerank", digits: number) => (
    <Table<any> rows={g.top[k]} rowKey={(r) => r.id} cols={[{ key: "label", label: "Code" }, { key: "value", label: k, num: true, render: (r) => fmt.n(r.value, digits) }]} />
  );
  return (
    <>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <Stat label="Nodes" value={fmt.n(s.nodes)} />
        <Stat label="Edges" value={fmt.n(s.edges)} detail="co-occurrence" />
        <Stat label="Communities" value={fmt.n(g.community_info.n_multi_node_communities)} detail={`${g.community_info.method} · modularity ${fmt.n(g.community_info.modularity, 3)}`} />
        <Stat label="Average degree" value={fmt.n(s.average_degree, 2)} />
        <Stat label="Density" value={fmt.n(s.density, 3)} />
        <Stat label="Components" value={fmt.n(s.connected_components)} detail={`largest ${s.largest_component}`} />
        <Stat label="Cross-community edges" value={fmt.n(s.cross_community_edges)} />
        <Stat label="Isolated concepts" value={fmt.n(s.isolated_nodes)} />
      </div>
      <div className="mt-3 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card title="Highest degree">{top("degree", 0)}</Card>
        <Card title="Highest betweenness">{top("betweenness", 4)}</Card>
        <Card title="Highest PageRank">{top("pagerank", 4)}</Card>
        <Card title="Communities">
          <div className="max-h-[360px] space-y-2 overflow-y-auto text-xs">
            {g.communities.filter((c: any) => c.size > 1).map((c: any) => (
              <div key={c.community}><div className="font-medium">Community {c.community} <span className="text-muted">· {c.size} codes · {c.internal_edges} internal edges</span></div>
                <div className="text-ink2">{c.members.map((m: any) => m.label).join(", ")}</div></div>
            ))}
            {g.isolated.length > 0 && <div><div className="font-medium">Isolated</div><div className="text-ink2">{g.isolated.map((m: any) => m.label).join(", ")}</div></div>}
          </div>
        </Card>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- E
function AcrossUploads({ run, runs }: { run: Run; runs: Run[] }) {
  const { datasets } = useApp();
  const others = runs.filter((r) => r.id !== run.id);
  const name = (r: Run) => `${r.label} · ${r.dataset_ids.map((id) => datasets.find((d) => d.id === id)?.filename ?? id).join(" + ")}`;
  const [b, setB] = useState<string>("");
  useEffect(() => {
    if (b && others.find((r) => r.id === b)) return;
    const diff = others.find((r) => r.dataset_ids.join() !== run.dataset_ids.join()) ?? others[0];
    setB(diff?.id ?? "");
  }, [run.id, runs.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const [cmp, err, loading] = useFetch<any>(b ? `/api/compare?a=${run.id}&b=${b}` : null);
  return (
    <>
      <Card actions={<Field label="Compare against (B)"><select className={cx(inputBase, "w-80")} value={b} onChange={(e) => setB(e.target.value)}>
        {!others.length && <option value="">No other completed analyses</option>}{others.map((r) => <option key={r.id} value={r.id}>{name(r)}</option>)}</select></Field>}
        title={`A: ${name(run)}`}>
        {!others.length && <Unavailable reason="Analyse this file with other settings, or analyse another file, to compare patterns." />}
        {loading && <Loading />}
        {err && <ErrorBox error={err} />}
        {cmp && <Comparison cmp={cmp} />}
      </Card>
    </>
  );
}

function Comparison({ cmp }: { cmp: any }) {
  const cc = cmp.community_changes;
  return (
    <div className="space-y-4">
      <div className="rounded-md bg-warn/10 px-3 py-2 text-sm text-warn">! {cmp.caveat}</div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
        <Stat label="Records A / B" value={`${fmt.n(cmp.a.n_records)} / ${fmt.n(cmp.b.n_records)}`} />
        <Stat label="Shared codes" value={fmt.n(cmp.shared_codes.length)} detail={`of ${cmp.a.n_codes} (A) / ${cmp.b.n_codes} (B)`} />
        <Stat label="Only in A" value={fmt.n(cmp.only_in_a.length)} />
        <Stat label="Only in B" value={fmt.n(cmp.only_in_b.length)} />
        <Stat label="New / disappeared links" value={`${cmp.new_relationships.length} / ${cmp.disappeared_relationships.length}`} detail="between shared codes" />
        <Stat label="Community agreement" value={fmt.n(cc.adjusted_rand_index, 3)} detail={`adjusted Rand index over ${cc.compared_codes} shared codes`} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <div className="mb-1 text-sm font-medium">Frequency changes (record-level prevalence)</div>
          <Table<any> rows={cmp.frequency_changes} rowKey={(r) => r.code} maxHeight={320}
            cols={[
              { key: "code", label: "Code" },
              { key: "a_pct", label: "A", num: true, render: (r) => <span title={`${r.a_records}/${cmp.a.n_records}`}>{fmt.pct(r.a_pct)}</span> },
              { key: "b_pct", label: "B", num: true, render: (r) => <span title={`${r.b_records}/${cmp.b.n_records}`}>{fmt.pct(r.b_pct)}</span> },
              { key: "diff_pct", label: "Δ pts", num: true, render: (r) => r.diff_pct === null ? "—" : `${r.diff_pct > 0 ? "+" : ""}${r.diff_pct}` },
              { key: "ci95", label: "95% CI", render: (r) => r.ci95 ? `${r.ci95[0]} to ${r.ci95[1]}` : "—" },
              { key: "p_value", label: "p", num: true, render: (r) => r.p_value ?? "—" },
              { key: "absolute_change", label: "Δ records", num: true },
            ]} />
        </div>
        <div className="space-y-3 text-sm">
          <ListBlock title="Codes found only in A" items={cmp.only_in_a} />
          <ListBlock title="Codes found only in B (new)" items={cmp.only_in_b} />
          <ListBlock title="New relationships in B" items={cmp.new_relationships.slice(0, 20).map((e: any) => `${e.a} — ${e.b} (${e.weight})`)} />
          <ListBlock title="Relationships not observed in B" items={cmp.disappeared_relationships.slice(0, 20).map((e: any) => `${e.a} — ${e.b} (${e.weight})`)} />
          <ListBlock title="Code pairs that split into different communities" items={(cc.pairs_split_apart ?? []).slice(0, 15).map((p: string[]) => p.join(" / "))} />
          <ListBlock title="Code pairs that joined a community" items={(cc.pairs_joined ?? []).slice(0, 15).map((p: string[]) => p.join(" / "))} />
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <div className="mb-1 text-sm font-medium">Shared themes (best code-overlap match)</div>
          <Table<any> rows={cmp.shared_themes} rowKey={(r) => `${r.a_theme}-${r.b_theme}`} empty="No theme pairs overlap."
            cols={[{ key: "a_theme", label: "Theme A", render: (r) => <>{r.a_theme} <Badge status={r.a_status} /></> }, { key: "b_theme", label: "Theme B", render: (r) => <>{r.b_theme} <Badge status={r.b_status} /></> }, { key: "code_overlap_jaccard", label: "Jaccard", num: true }]} />
        </div>
        <div>
          <div className="mb-1 text-sm font-medium">Model / configuration differences</div>
          <Table<any> rows={cmp.model_differences} rowKey={(r) => r.parameter} empty="Same configuration — differences reflect the data."
            cols={[{ key: "parameter", label: "Parameter" }, { key: "a", label: "A", render: (r) => String(r.a ?? "—") }, { key: "b", label: "B", render: (r) => String(r.b ?? "—") }]} />
        </div>
      </div>
    </div>
  );
}

function ListBlock({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <div className="text-xs font-medium text-ink2">{title} <span className="text-muted">({items.length})</span></div>
      <div className="text-xs text-ink">{items.length ? items.join(" · ") : <span className="text-muted">none</span>}</div>
    </div>
  );
}

// ---------------------------------------------------------------- F
function TemporalGraph({ runId }: { runId: string }) {
  const { dateFrom, dateTo } = useApp();
  const [t, err, loading] = useFetch<any>(`/api/runs/${runId}/temporal`);
  const [i, setI] = useState(0);
  const idx = useMemo(() => {
    if (!t?.available) return [];
    return t.periods.map((p: string, k: number) => ({ p, k })).filter(({ p }: any) => (!dateFrom || p >= dateFrom.slice(0, p.length)) && (!dateTo || p <= dateTo.slice(0, p.length))).map((x: any) => x.k);
  }, [t, dateFrom, dateTo]);
  const opt = useMemo(() => () => {
    if (!t?.available) return {};
    const per = idx.map((k: number) => t.periods[k]);
    const cps = new Set(t.volume_change_points);
    return {
      ...baseOption(),
      grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
      tooltip: { ...baseOption().tooltip, trigger: "axis" },
      legend: { ...baseOption().legend, type: "scroll" },
      xAxis: axis("category", { data: per, boundaryGap: false }),
      yAxis: axis("value", { axisLabel: { color: chrome().muted, formatter: "{value}%" } }),
      series: t.codes.slice(0, 6).map((c: any, j: number) => ({
        ...lineStyle(series(j)), name: c.label, data: idx.map((k: number) => c.share_pct[k]),
        markLine: j === 0 && cps.size ? { symbol: "none", silent: true, label: { color: chrome().muted, fontSize: 10, formatter: "change point" }, lineStyle: { color: chrome().axis, type: "solid" }, data: [...cps].filter((p) => per.includes(p)).map((p) => ({ xAxis: p })) } : undefined,
      })),
    };
  }, [t, idx]);
  if (t && !t.available) return null;
  return (
    <>
      <SectionTitle id="temporal" title="Temporal graph analysis" desc={t?.available ? `${t.dynamic_graph.window_rule}. ${t.dynamic_graph.caveat}` : undefined} />
      {loading && <Loading />}
      {err && <ErrorBox error={err} />}
      {t?.available && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Code share over time" subtitle={`${t.frequency} periods · ${t.records_without_timestamp} records without timestamp excluded${dateFrom || dateTo ? " · filtered by the global date range" : ""}`}>
            <Chart option={opt} height={280} name="temporal-share" />
            <div className="mt-2 text-xs text-ink2">Volume trend: {t.volume_trend.available === false ? "insufficient periods" : `${t.volume_trend.trend} (Mann–Kendall p=${t.volume_trend.p_value})`}
              {t.seasonal_decomposition?.available ? ` · seasonal decomposition computed (period ${t.seasonal_decomposition.period})` : ` · seasonality: ${t.seasonal_decomposition?.reason ?? "n/a"}`}</div>
          </Card>
          <Card title="Graph snapshots" subtitle="One co-occurrence graph per time window">
            <Table<any> rows={t.dynamic_graph.windows} rowKey={(r) => r.label}
              cols={[{ key: "label", label: "Window" }, { key: "records", label: "Records", num: true }, { key: "nodes", label: "Codes", num: true }, { key: "edges", label: "Edges", num: true }, { key: "communities", label: "Communities", num: true }, { key: "largest_community", label: "Largest", num: true }]} />
            {t.dynamic_graph.comparisons.length > 0 && (
              <div className="mt-3">
                <Field label="Before → after"><select className={inputCls} value={i} onChange={(e) => setI(Number(e.target.value))}>
                  {t.dynamic_graph.comparisons.map((c: any, k: number) => <option key={k} value={k}>{c.from} → {c.to}</option>)}</select></Field>
                {(() => { const c = t.dynamic_graph.comparisons[i]; return (
                  <div className="mt-2 space-y-2 text-xs">
                    <div><span className="font-medium">Newly connected ({c.n_new_edges}):</span> {c.new_edges.slice(0, 10).map((e: any) => `${e.pair.join(" — ")} (${e.weight})`).join(" · ") || "none"}</div>
                    <div><span className="font-medium">No longer connected ({c.n_disappeared_edges}):</span> {c.disappeared_edges.slice(0, 10).map((e: any) => `${e.pair.join(" — ")} (${e.weight})`).join(" · ") || "none"}</div>
                    <div><span className="font-medium">Largest edge-weight changes:</span> {c.weight_changes.slice(0, 8).map((e: any) => `${e.pair.join(" — ")} (${e.change > 0 ? "+" : ""}${e.change})`).join(" · ") || "none"}</div>
                    <div><span className="font-medium">Node frequency changes:</span> {c.node_frequency_changes.slice(0, 8).map((e: any) => `${e.code} ${e.before}→${e.after}`).join(" · ")}</div>
                  </div>); })()}
              </div>
            )}
          </Card>
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------- G
function ModelComparison({ run, runs }: { run: Run; runs: Run[] }) {
  const same = runs.filter((r) => r.dataset_ids.slice().sort().join() === run.dataset_ids.slice().sort().join());
  const metric = (r: Run, path: string[]) => path.reduce((o: any, k) => (o ? o[k] : undefined), r.metrics);
  const rows = same.map((r) => ({
    id: r.id, label: r.label, current: r.id === run.id,
    coding: `${r.config.coding_method} (${metric(r, ["coding", "n_codes"]) ?? "—"} codes)`,
    clustering: r.config.clustering_method, k: metric(r, ["clustering", "n_clusters"]), sil: metric(r, ["clustering", "silhouette_cosine"]), db: metric(r, ["clustering", "davies_bouldin"]),
    topics: r.config.topic_method, coh: metric(r, ["topics", "mean_coherence_umass"]), div: metric(r, ["topics", "topic_diversity"]),
    graph: r.config.graph_method, comm: metric(r, ["graph", "communities", "n_multi_node_communities"]), mod: metric(r, ["graph", "communities", "modularity"]),
    auc: metric(r, ["graph", "link_prediction_baseline", "roc_auc", "adamic_adar"]), runtime: r.duration_s,
  }));
  return (
    <>
      <Card>
        {rows.length < 2 ? <Unavailable reason="Analyse the same file with another configuration (for example hierarchical clustering or LDA) from the Upload page to compare models." /> : (
          <Table<any> rows={rows} rowKey={(r) => r.id}
            cols={[
              { key: "label", label: "Run", render: (r) => <span className={cx(r.current && "font-semibold")}>{r.label}{r.current ? " (A)" : ""}</span> },
              { key: "coding", label: "Coding" }, { key: "clustering", label: "Clustering" },
              { key: "k", label: "k", num: true, render: (r) => fmt.n(r.k) }, { key: "sil", label: "Silhouette ↑", num: true, render: (r) => fmt.n(r.sil, 3) },
              { key: "db", label: "Davies–Bouldin ↓", num: true, render: (r) => fmt.n(r.db, 3) },
              { key: "topics", label: "Topics" }, { key: "coh", label: "UMass", num: true, render: (r) => fmt.n(r.coh, 3) }, { key: "div", label: "Diversity", num: true, render: (r) => fmt.n(r.div, 3) },
              { key: "graph", label: "Graph" }, { key: "comm", label: "Comm.", num: true, render: (r) => fmt.n(r.comm) }, { key: "mod", label: "Modularity", num: true, render: (r) => fmt.n(r.mod, 3) },
              { key: "auc", label: "Link AUC", num: true, render: (r) => fmt.n(r.auc, 3) },
              { key: "runtime", label: "Runtime", num: true, render: (r) => fmt.sec(r.runtime) },
            ]} />
        )}
        <p className="mt-2 text-[11px] text-muted">The Adamic–Adar link-prediction AUC is the classical baseline a graph neural network has to beat.</p>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- H
function GraphExport({ runId, g }: { runId: string; g: any }) {
  const { toast } = useApp();
  const dl = (fmt_: string) => download(`/api/runs/${runId}/export/${fmt_}`).catch((e) => toast(e.message, "error"));
  const saveJson = (obj: unknown, name: string) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" }));
    a.download = name;
    a.click();
  };
  return (
    <>
      <Card>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => dl("graph_nodes_csv")}>Nodes (CSV)</Button>
          <Button onClick={() => dl("graph_edges_csv")}>Edges (CSV)</Button>
          <Button onClick={() => dl("graphml")}>Graph (GraphML)</Button>
          <Button onClick={() => dl("clusters_csv")}>Cluster assignments (CSV)</Button>
          <Button onClick={() => dl("temporal_json")}>Temporal results (JSON)</Button>
          <Button disabled={!g} onClick={() => saveJson({ params: g.params, stats: g.stats, community_info: g.community_info, top: g.top, communities: g.communities }, `graph-statistics-${runId}.json`)}>Graph statistics (JSON, current view)</Button>
        </div>
        <p className="mt-2 text-xs text-muted">GraphML / CSV use the run's configured graph parameters with current researcher edits; the statistics JSON uses the parameters applied above. Hover the graph for PNG / SVG.</p>
      </Card>
    </>
  );
}
