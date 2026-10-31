/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { fmt, type Run } from "../api";
import { Chart, axis, barStyle, baseOption, esc, lineStyle } from "../components/Chart";
import { CodeDrawer, ThemeDrawer } from "../components/Evidence";
import { Icon, ICONS } from "../components/icons";
import { AnalysisPath } from "../components/Thematic";
import { Badge, Button, Card, Empty, ErrorBox, Loading, Unavailable, cx } from "../components/ui";
import { chrome, isDark, sentimentColor, series } from "../colors";
import { useApp, useFetch } from "../state";

const TONES = ["positive", "neutral", "mixed", "negative", "uncertain"] as const;

export default function DashboardPage() {
  const { project, dataset, run, activeRuns } = useApp();
  const nav = useNavigate();
  if (!project) return <Empty title="No project selected">Create or select a project first.</Empty>;
  if (!run) {
    return (
      <div className="mx-auto max-w-3xl pt-6">
        <Empty title={dataset ? `${dataset.filename} has not been analysed yet` : "No analysed file yet"}>
          The dashboard shows the results of one analysed file. Analyse a file, or pick another one on the Database page.
          <div className="mt-5 flex justify-center gap-2">
            <Button kind="primary" onClick={() => nav("/upload")}><Icon path={ICONS.play} size={13} />{dataset ? "Analyse this file" : "Upload a file"}</Button>
            <Button onClick={() => nav("/database")}>Choose another file</Button>
          </div>
        </Empty>
      </div>
    );
  }
  if (run.status !== "completed") {
    const live = activeRuns.find((r) => r.id === run.id);
    return (
      <div className="mx-auto max-w-3xl pt-6">
        <Card title={`${run.label} is ${run.status}`} subtitle={dataset?.filename}>
          {live ? (
            <>
              <div className="h-2 rounded-full bg-raised"><div className="progress-shimmer h-2 rounded-full transition-all" style={{ width: `${Math.max(3, run.progress * 100)}%` }} /></div>
              <p className="mt-2 text-sm text-ink2">Stage: {run.stage ?? "starting"} · {Math.round(run.progress * 100)}% — the dashboard appears when it finishes.</p>
            </>
          ) : run.error ? <ErrorBox error={`Failed at ${run.failed_stage}: ${run.error}. Retry it from the Database page (Analysis history).`} /> : <p className="text-sm text-muted">Results appear when it completes.</p>}
        </Card>
      </div>
    );
  }
  return <Dashboard key={run.id} run={run} />;
}

function Dashboard({ run }: { run: Run }) {
  const [d, err, loading, reload] = useFetch<any>(`/api/runs/${run.id}/dashboard`);
  const [topics] = useFetch<any>(`/api/runs/${run.id}/topics`);
  const [structure, , , reloadS] = useFetch<any>(`/api/runs/${run.id}/thematic`);
  const [params, setParams] = useSearchParams();
  const [codeId, setCodeId] = useState<number | null>(null);
  const [themeId, setThemeId] = useState<number | null>(null);
  useEffect(() => {
    const c = params.get("code"), t = params.get("theme");
    if (c) setCodeId(Number(c));
    if (t) setThemeId(Number(t));
  }, [params]);
  const close = () => { setCodeId(null); setThemeId(null); if (params.get("code") || params.get("theme")) setParams({}); };

  // themes ordered by reach; each keeps one colour everywhere on the page, and its codes share it
  const themes = useMemo(() => (d ? d.themes.table.filter((t: any) => !t.parent_id).sort((a: any, b: any) => b.segments - a.segments) : []), [d]);
  const subthemes = useMemo(() => new Map<number, any[]>((structure?.themes ?? []).map((t: any) => [t.id, t.subthemes])), [structure]);
  const refresh = () => { reload(); reloadS(); };
  const codeTheme = useMemo(() => {
    const m = new Map<number, number>();
    themes.forEach((t: any, i: number) => t.code_ids.forEach((c: number) => m.has(c) || m.set(c, i)));
    return m;
  }, [themes]);

  if (loading && !d) return <Loading label="Building the dashboard…" />;
  if (err) return <ErrorBox error={err} />;
  if (!d) return null;
  const theme = themeId !== null ? d.themes.table.find((t: any) => t.id === themeId) ?? null : null;

  return (
    <div className="mx-auto max-w-7xl space-y-5">
      <Hero d={d} />
      <AnalysisPath path={structure?.path} />
      {(d.run.warnings?.length ?? 0) > 0 && <div className="rounded-lg bg-warn/10 px-4 py-2.5 text-sm text-warn">! {d.run.warnings.join(" · ")}</div>}

      <div className="grid gap-5 lg:grid-cols-5">
        <Card className="lg:col-span-3" title="Themes" subtitle="Share of passages that touch each candidate theme · click a bar for its evidence">
          <ThemeBars themes={themes} onOpen={setThemeId} />
        </Card>
        <Card className="lg:col-span-2" title="Overall tone" subtitle="Lexicon sentiment of every passage (VADER, rule-based)">
          <ToneDonut sent={d.statistics.sentiment} />
        </Card>
      </div>

      {themes.length > 0 && (
        <section>
          <h2 className="mb-3 mt-3 text-lg font-semibold tracking-tight text-ink">What the themes say</h2>
          <div className={cx("grid gap-4 md:grid-cols-2", themes.length === 3 && "xl:grid-cols-3")}>
            {themes.slice(0, 4).map((t: any, i: number) => <ThemeCard key={t.id} t={t} subthemes={subthemes.get(t.id) ?? []} color={series(i)} onOpen={() => setThemeId(t.id)} />)}
          </div>
          {themes.length > 4 && <p className="mt-2 text-xs text-muted">{themes.length - 4} more themes in the chart above and on the Database page.</p>}
        </section>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Most frequent codes" subtitle="Passages per code · colour = the theme it belongs to · click for passages">
          <CodeBars codes={d.coding.table} total={d.coding.denominators.segments} codeTheme={codeTheme} onOpen={setCodeId} />
        </Card>
        <Card title="Codes that appear together" subtitle={`Strongest co-occurring pairs, counted per ${d.ml.config.cooccurrence_level}`}>
          <CoHeatmap pairs={d.statistics.code_cooccurrence} codes={d.coding.table} />
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Tone by theme" subtitle="Sentiment mix of the passages in each theme">
          <ToneByTheme sent={d.statistics.sentiment} themes={themes} onOpen={setThemeId} />
        </Card>
        <Card title="Topics" subtitle={topics ? `${topics.topics.length} topics from ${String(topics.metrics?.method ?? "").toUpperCase()} · share of passages · statistical word groups, not themes` : undefined}>
          {topics ? <TopicBars topics={topics} /> : <Loading />}
        </Card>
      </div>

      {d.statistics.temporal.available && (
        <Card title="Over time" subtitle={`Share of passages per period for the most frequent codes · ${fmt.pct(d.statistics.temporal.coverage_pct)} of records timestamped`}>
          <Trend tmp={d.statistics.temporal} />
        </Card>
      )}

      <ModelFit d={d} />

      <CodeDrawer codeId={codeId} onClose={close} onChanged={refresh} />
      <ThemeDrawer theme={theme} runId={run.id} onClose={close} onChanged={refresh} codes={d.coding.table} themes={d.themes.table} />
    </div>
  );
}

// ---------------------------------------------------------------- header

function Hero({ d }: { d: any }) {
  const nav = useNavigate();
  const o = d.overview;
  const files = o.datasets.map((x: any) => x.filename);
  const sent = d.statistics.sentiment;
  const kpis: [string, ReactNode, ReactNode][] = [
    ["Passages", fmt.n(o.segments), `from ${fmt.n(o.total_records)} records`],
    ["Words", fmt.compact(o.total_words), `${fmt.n(o.mean_words_per_record, 0)} per record`],
    ["Codes", fmt.n(o.unique_codes), `${fmt.pct(d.coding.coded_segments?.value, 0)} of passages coded`],
    ["Themes", fmt.n(o.total_themes), `${o.approved_themes} approved`],
    ["Topics", d.themes.topics === null ? "—" : fmt.n(d.themes.topics), d.themes.topic_metrics?.method ? d.themes.topic_metrics.method.toUpperCase() : "not computed"],
    ["Positive tone", sent.available ? fmt.pct(sent.percentages.positive.value, 0) : "—", sent.available ? `${fmt.pct(sent.percentages.negative.value, 0)} negative` : "sentiment off"],
  ];
  return (
    <section className="overflow-hidden rounded-2xl border border-line bg-surface" style={{ boxShadow: "var(--shadow-sm)" }}>
      <div className="flex flex-wrap items-end justify-between gap-4 px-6 pb-5 pt-6" style={{ background: "linear-gradient(120deg, rgb(var(--accent) / 0.10), transparent 55%)" }}>
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-widest text-accent">Dashboard</div>
          <h1 className="mt-1 truncate text-[26px] font-semibold leading-tight tracking-tight text-ink" title={files.join(", ")}>
            {files.length > 1 ? `Combined analysis · ${files.length} files` : files[0]}
          </h1>
          <p className="mt-1 text-sm text-ink2">{d.run.label} · analysed {fmt.date(d.run.finished_at)} · {fmt.sec(o.processing_s)} on this machine</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => nav("/database")}>Details, files & export</Button>
          <Button kind="primary" onClick={() => nav("/upload")}><Icon path={ICONS.upload} size={14} />Analyse a file</Button>
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-px border-t border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
        {kpis.map(([label, value, sub]) => (
          <div key={label} className="bg-surface px-6 py-4">
            <dt className="text-[11px] font-semibold uppercase tracking-wider text-muted">{label}</dt>
            <dd className="mt-1 text-2xl font-bold tabular-nums tracking-tight text-ink">{value}</dd>
            <dd className="mt-0.5 truncate text-xs text-muted">{sub}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

// ---------------------------------------------------------------- charts

const hbar = (labels: string[], width = 150) => ({
  ...baseOption(),
  grid: { left: 8, right: 56, top: 4, bottom: 4, containLabel: true },
  xAxis: { ...axis("value"), show: false, max: (v: any) => v.max * 1.02 },
  yAxis: axis("category", { data: labels, inverse: true, axisLine: { show: false }, axisLabel: { color: chrome().ink2, fontSize: 12, width, overflow: "truncate" } }),
});

function ThemeBars({ themes, onOpen }: { themes: any[]; onOpen: (id: number) => void }) {
  const opt = useMemo(() => () => ({
    ...hbar(themes.map((t) => t.label), 170),
    tooltip: { ...baseOption().tooltip, formatter: (p: any) => { const t = themes[p.dataIndex]; const s = t.prevalence.segment; return `<b>${esc(t.label)}</b> · ${esc(t.status)}<br/>${s.numerator} of ${s.denominator} passages (${fmt.pct(s.value)})<br/>${t.codes.length} codes · ${t.supporting_quotations} supporting quotes`; } },
    series: [{
      ...barStyle(series(0), true), barMaxWidth: 22,
      data: themes.map((t, i) => ({ value: t.prevalence.segment.value, itemStyle: { color: series(i), borderRadius: [0, 6, 6, 0] } })),
      label: { show: true, position: "right", color: chrome().ink, fontSize: 12, fontWeight: 600, formatter: (p: any) => fmt.pct(p.value, 0) },
      showBackground: true, backgroundStyle: { color: isDark() ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.035)", borderRadius: [0, 6, 6, 0] },
    }],
  }), [themes]);
  if (!themes.length) return <Unavailable reason="No candidate themes: no community in the code graph had enough codes. Larger files, or a lower Min NPMI under Advanced options, give more themes." />;
  return <Chart option={opt} height={Math.max(150, themes.length * 44)} name="themes" onClick={(p) => onOpen(themes[p.dataIndex].id)} />;
}

function ToneDonut({ sent }: { sent: any }) {
  const data = sent.available ? TONES.filter((l) => sent.counts[l] > 0).map((l) => ({ name: l, value: sent.counts[l], itemStyle: { color: sentimentColor(l) } })) : [];
  const opt = useMemo(() => () => ({
    ...baseOption(),
    legend: { show: false },
    tooltip: { ...baseOption().tooltip, formatter: (p: any) => `<b>${esc(p.name)}</b><br/>${p.value} passages (${fmt.pct(p.percent, 1)})` },
    series: [{ type: "pie", radius: ["62%", "88%"], avoidLabelOverlap: true, label: { show: false }, padAngle: 1.5,
      itemStyle: { borderColor: chrome().surface, borderWidth: 2, borderRadius: 4 }, data }],
  }), [sent]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!sent.available) return <Unavailable reason={sent.reason} />;
  const top = [...TONES].sort((a, b) => sent.counts[b] - sent.counts[a])[0];
  return (
    <div className="flex flex-wrap items-center gap-6">
      <div className="relative w-[190px] shrink-0">
        <Chart option={opt} height={190} name="tone" />
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-2xl font-bold tabular-nums text-ink">{fmt.pct(sent.percentages[top].value, 0)}</div>
          <div className="text-xs capitalize text-muted">{top}</div>
        </div>
      </div>
      <ul className="min-w-[150px] flex-1 space-y-2 text-sm">
        {TONES.map((l) => (
          <li key={l} className={cx("flex items-center gap-2", !sent.counts[l] && "opacity-45")} title={sent.percentages[l].definition}>
            <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: sentimentColor(l) }} />
            <span className="flex-1 capitalize text-ink2">{l}</span>
            <span className="font-semibold tabular-nums text-ink">{fmt.pct(sent.percentages[l].value, 0)}</span>
            <span className="w-8 text-right text-xs tabular-nums text-muted">{sent.counts[l]}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function CodeBars({ codes, total, codeTheme, onOpen }: { codes: any[]; total: number; codeTheme: Map<number, number>; onOpen: (id: number) => void }) {
  const top = useMemo(() => codes.filter((c) => c.status === "candidate" || c.status === "approved").sort((a, b) => b.segments - a.segments).slice(0, 12), [codes]);
  const opt = useMemo(() => () => ({
    ...hbar(top.map((c) => c.label), 140),
    tooltip: { ...baseOption().tooltip, formatter: (p: any) => { const c = top[p.dataIndex]; return `<b>${esc(c.label)}</b><br/>${c.segments} of ${total} passages (${fmt.pct(c.segment_pct)})<br/>${esc(c.method)}`; } },
    series: [{
      ...barStyle(series(0), true), barMaxWidth: 16,
      data: top.map((c) => ({ value: c.segments, itemStyle: { color: series(codeTheme.has(c.id) ? codeTheme.get(c.id)! : -1), borderRadius: [0, 4, 4, 0] } })),
      label: { show: true, position: "right", color: chrome().ink2, fontSize: 11, formatter: (p: any) => `${p.value} · ${fmt.pct(top[p.dataIndex].segment_pct, 0)}` },
    }],
  }), [top, total, codeTheme]);
  if (!top.length) return <Unavailable reason="No codes were generated." />;
  return <Chart option={opt} height={Math.max(200, top.length * 28)} name="codes" onClick={(p) => onOpen(top[p.dataIndex].id)} />;
}

function CoHeatmap({ pairs, codes }: { pairs: any[]; codes: any[] }) {
  const { labels, cells, max } = useMemo(() => {
    const seg = new Map<number, number>(codes.map((c: any) => [c.id, c.segments]));
    const name = new Map<number, string>();
    pairs.forEach((p) => { name.set(p.key_a, p.code_a); name.set(p.key_b, p.code_b); });
    const keys = [...name.keys()].sort((a, b) => (seg.get(b) ?? 0) - (seg.get(a) ?? 0)).slice(0, 9);
    const at = new Map(keys.map((k, i) => [k, i]));
    const cells: [number, number, number][] = [];
    for (const p of pairs) {
      const a = at.get(p.key_a), b = at.get(p.key_b);
      if (a === undefined || b === undefined) continue;
      cells.push([a, b, p.count], [b, a, p.count]);
    }
    return { labels: keys.map((k) => name.get(k)!), cells, max: Math.max(1, ...cells.map((c) => c[2])) };
  }, [pairs, codes]);
  const opt = useMemo(() => () => {
    const c = chrome();
    return {
      ...baseOption(),
      grid: { left: 8, right: 8, top: 8, bottom: 8, containLabel: true },
      tooltip: { ...baseOption().tooltip, formatter: (p: any) => `<b>${esc(labels[p.value[0]])}</b> + <b>${esc(labels[p.value[1]])}</b><br/>together ${p.value[2]} times` },
      xAxis: { ...axis("category", { data: labels }), axisLabel: { color: c.ink2, fontSize: 11, rotate: 35, width: 90, overflow: "truncate", interval: 0 }, splitArea: { show: false } },
      yAxis: { ...axis("category", { data: labels, inverse: true }), axisLabel: { color: c.ink2, fontSize: 11, width: 110, overflow: "truncate", interval: 0 } },
      visualMap: { show: false, min: 0, max, inRange: { color: isDark() ? ["#1d2a3d", "#3987e5"] : ["#e3edfa", "#1f5fae"] } },
      series: [{ type: "heatmap", data: cells, itemStyle: { borderColor: c.surface, borderWidth: 2, borderRadius: 3 },
        label: { show: true, fontSize: 10, color: c.ink, formatter: (p: any) => (p.value[2] >= max * 0.55 ? `{w|${p.value[2]}}` : p.value[2]), rich: { w: { color: "#fff", fontSize: 10 } } },
        emphasis: { itemStyle: { borderColor: c.ink, borderWidth: 1 } } }],
    };
  }, [labels, cells, max]);
  if (labels.length < 2) return <Unavailable reason="Too few codes appear together to draw a co-occurrence map." />;
  return (
    <>
      <Chart option={opt} height={Math.max(260, labels.length * 34 + 70)} name="co-occurrence" />
      <p className="mt-1 text-[11px] text-muted">Empty cells: the pair is not among the {pairs.length} strongest pairs. The full network is on Graphs & Patterns.</p>
    </>
  );
}

function ToneByTheme({ sent, themes, onOpen }: { sent: any; themes: any[]; onOpen: (id: number) => void }) {
  const rows = useMemo(() => {
    if (!sent.available) return [];
    const order = new Map(themes.map((t, i) => [t.id, i]));
    return sent.by_theme.filter((r: any) => r.segments > 0 && order.has(r.theme_id)).sort((a: any, b: any) => (order.get(a.theme_id) ?? 99) - (order.get(b.theme_id) ?? 99));
  }, [sent, themes]);
  const opt = useMemo(() => () => ({
    ...baseOption(),
    grid: { left: 8, right: 16, top: 30, bottom: 4, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis", axisPointer: { type: "shadow" },
      formatter: (p: any) => { const r = rows[p[0].dataIndex]; return `<b>${esc(r.label)}</b> · ${r.segments} passages<br/>` + TONES.filter((l) => r[l]).map((l) => `${l}: ${r[l]} (${fmt.pct(100 * r[l] / r.segments, 0)})`).join("<br/>"); } },
    legend: { ...baseOption().legend, data: TONES.filter((l) => rows.some((r: any) => r[l])) },
    xAxis: { ...axis("value", { max: 100 }), axisLabel: { color: chrome().muted, formatter: "{value}%" } },
    yAxis: axis("category", { data: rows.map((r: any) => r.label), inverse: true, axisLine: { show: false }, axisLabel: { color: chrome().ink2, fontSize: 12, width: 140, overflow: "truncate" } }),
    series: TONES.map((l) => ({
      name: l, type: "bar", stack: "tone", barMaxWidth: 22, itemStyle: { color: sentimentColor(l), borderColor: chrome().surface, borderWidth: 1 },
      data: rows.map((r: any) => (r.segments ? +(100 * r[l] / r.segments).toFixed(1) : 0)),
    })),
  }), [rows]);
  if (!sent.available) return <Unavailable reason={sent.reason} />;
  if (!rows.length) return <Unavailable reason="No themes to compare." />;
  return <Chart option={opt} height={Math.max(170, rows.length * 44 + 40)} name="tone-by-theme" onClick={(p) => onOpen(rows[p.dataIndex].theme_id)} />;
}

function TopicBars({ topics }: { topics: any }) {
  const rows = useMemo(() => [...(topics.topics ?? [])].sort((a: any, b: any) => b.size - a.size), [topics]);
  const opt = useMemo(() => () => ({
    ...hbar(rows.map((t: any) => t.terms.slice(0, 3).join(", ")), 170),
    tooltip: { ...baseOption().tooltip, formatter: (p: any) => { const t = rows[p.dataIndex]; return `<b>Topic ${t.topic + 1}</b> · ${t.share.numerator} passages (${fmt.pct(t.share.value)})<br/>${esc(t.terms.join(", "))}<br/>UMass coherence ${fmt.n(t.coherence, 2)}`; } },
    series: [{ ...barStyle(series(6), true), barMaxWidth: 16, data: rows.map((t: any) => t.share.value),
      label: { show: true, position: "right", color: chrome().ink2, fontSize: 11, formatter: (p: any) => fmt.pct(p.value, 0) } }],
  }), [rows]);
  if (!topics.metrics?.available || !rows.length) return <Unavailable reason={topics.metrics?.reason ?? "No topics were computed."} />;
  return <Chart option={opt} height={Math.max(180, rows.length * 30)} name="topics" />;
}

function Trend({ tmp }: { tmp: any }) {
  const opt = useMemo(() => () => ({
    ...baseOption(),
    grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" },
    legend: { ...baseOption().legend, type: "scroll" },
    xAxis: axis("category", { data: tmp.periods ?? [], boundaryGap: false }),
    yAxis: axis("value", { axisLabel: { color: chrome().muted, formatter: "{value}%" } }),
    series: (tmp.codes ?? []).slice(0, 6).map((c: any, i: number) => ({ ...lineStyle(series(i)), name: c.label, data: c.share_pct })),
  }), [tmp]);
  return <Chart option={opt} height={260} name="over-time" />;
}

// ---------------------------------------------------------------- theme cards

function ThemeCard({ t, subthemes, color, onOpen }: { t: any; subthemes: any[]; color: string; onOpen: () => void }) {
  const [ev] = useFetch<any>(`/api/themes/${t.id}/evidence`);
  const quote = ev?.representative?.[0];
  const chips = subthemes.length ? subthemes : t.codes;
  return (
    <article className="flex flex-col rounded-xl border border-line bg-surface p-5" style={{ boxShadow: "var(--shadow-sm)", borderTop: `3px solid ${color}` }}>
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-[15px] font-semibold text-ink">{t.label}</h3>
        <Badge status={t.status} />
      </div>
      <div className="mt-1 text-xs text-muted">{fmt.pct(t.prevalence.segment.value, 0)} of passages · {t.segments} passages · {t.codes.length} codes</div>
      {t.origin !== "machine" && t.definition && <p className="mt-2 text-sm text-ink2">{t.definition}</p>}
      <div className="mt-3 flex flex-wrap gap-1.5">
        {chips.slice(0, 6).map((c: any) => <span key={c.id ?? c.label} className="rounded-full px-2 py-0.5 text-[11px] font-medium" style={{ background: `${color}1f`, color }}>{c.label}</span>)}
        {chips.length > 6 && <span className="px-1 py-0.5 text-[11px] text-muted">+{chips.length - 6}</span>}
      </div>
      <blockquote className="mt-4 line-clamp-4 flex-1 border-l-2 pl-3 text-sm italic leading-relaxed text-ink2" style={{ borderColor: color }}>
        {quote ? `“${quote.text}”` : ev ? <span className="not-italic text-muted">No representative passage.</span> : <span className="not-italic text-muted">Loading a representative passage…</span>}
      </blockquote>
      <button onClick={onOpen} className="mt-4 flex items-center gap-1 self-start text-xs font-semibold text-accent hover:underline">
        Open evidence and review<Icon path={ICONS.arrow} size={12} />
      </button>
    </article>
  );
}

// ---------------------------------------------------------------- model fit

function ModelFit({ d }: { d: any }) {
  const cl = d.ml.clustering, tp = d.ml.topics, g = d.ml.graph;
  const auc = g.link_prediction_baseline?.available ? g.link_prediction_baseline.roc_auc?.adamic_adar : null;
  const metrics: { label: string; value: number | null | undefined; digits: number; range?: [number, number]; note: string }[] = [
    { label: "Cluster separation", value: cl.available ? cl.silhouette_cosine : null, digits: 2, range: [-1, 1], note: `Silhouette of ${cl.n_clusters ?? "—"} K-means clusters; higher = clearer groups` },
    { label: "Topic coherence", value: tp.available ? tp.mean_coherence_umass : null, digits: 2, note: "UMass; closer to 0 = topic words co-occur more" },
    { label: "Theme modularity", value: g.communities?.modularity, digits: 2, range: [0, 1], note: `${g.communities?.n_multi_node_communities ?? "—"} code communities; higher = denser themes` },
    { label: "Link prediction", value: auc, digits: 2, range: [0.5, 1], note: "Adamic–Adar ROC AUC on held-out edges; 0.5 = chance" },
  ];
  const ks = cl.k_candidates ?? [];
  const kOpt = useMemo(() => () => ({
    ...baseOption(),
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" },
    xAxis: axis("category", { data: ks.map((k: any) => `k=${k.k}`) }),
    yAxis: axis("value", { name: "silhouette", nameTextStyle: { color: chrome().muted, fontSize: 11 } }),
    series: [{ ...lineStyle(series(0)), showSymbol: true, data: ks.map((k: any) => ({ value: k.silhouette, itemStyle: k.k === cl.n_clusters ? { color: series(1), borderColor: series(1) } : undefined, symbolSize: k.k === cl.n_clusters ? 11 : 7 })) }],
  }), [ks]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Card title="How well the models fit" subtitle={d.ml.accuracy_note}>
      <div className={cx("grid gap-6", ks.length > 1 && "lg:grid-cols-[1.2fr_1fr]")}>
        <div className="grid gap-3 sm:grid-cols-2">
          {metrics.map((m) => (
            <div key={m.label} className="rounded-lg border border-line p-4">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">{m.label}</div>
              <div className="mt-1 text-2xl font-bold tabular-nums text-ink">{m.value === null || m.value === undefined ? "—" : m.value.toFixed(m.digits)}</div>
              {m.range && m.value !== null && m.value !== undefined && (
                <div className="mt-2 h-1.5 rounded-full bg-raised">
                  <div className="h-1.5 rounded-full bg-accent" style={{ width: `${Math.min(100, Math.max(2, 100 * (m.value - m.range[0]) / (m.range[1] - m.range[0])))}%` }} />
                </div>
              )}
              <div className="mt-2 text-[11px] leading-snug text-muted">{m.note}</div>
            </div>
          ))}
        </div>
        {ks.length > 1 && (
          <div>
            <div className="mb-1 text-xs font-medium text-ink2">Choosing the number of clusters — the highlighted k had the best silhouette</div>
            <Chart option={kOpt} height={230} name="k-selection" />
          </div>
        )}
      </div>
    </Card>
  );
}
