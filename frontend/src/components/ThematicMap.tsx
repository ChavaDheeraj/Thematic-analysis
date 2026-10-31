/* eslint-disable @typescript-eslint/no-explicit-any */
// Step 7: themes, their sub-themes and how the themes relate. Node positions are computed in pixels
// for the current width, so boxes and labels cannot overlap and the picture stays the same on every load.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { fmt } from "../api";
import { chrome, series, tint } from "../colors";
import { useFetch } from "../state";
import { Chart, baseOption, esc } from "./Chart";
import { ThemeDrawer } from "./Evidence";
import { Icon, ICONS } from "./icons";
import { Badge, Button, Card, Disclosure, ErrorBox, Loading, SectionTitle, Table, Unavailable } from "./ui";

const BOX_H = 62, PILL_H = 26, PILL_GAP = 6, BAND_GAP = 26;

/** A rounded box as a symbol path: ECharts' own roundRect stretches its corners on wide shapes. */
const rounded = (w: number, h: number, r: number) => `path://M${r} 0H${w - r}Q${w} 0 ${w} ${r}V${h - r}Q${w} ${h} ${w - r} ${h}H${r}Q0 ${h} 0 ${h - r}V${r}Q0 0 ${r} 0Z`;

/** Break a label into at most `lines` lines of `width` characters. */
function wrap(text: string, width: number, lines: number) {
  const out: string[] = [];
  let cur = "";
  for (const word of text.split(" ")) {
    if (cur && (cur + " " + word).length > width) { out.push(cur); cur = word; }
    else cur = cur ? `${cur} ${word}` : word;
  }
  if (cur) out.push(cur);
  if (out.length <= lines) return out.join("\n");
  const kept = out.slice(0, lines);
  kept[lines - 1] = `${kept[lines - 1].slice(0, width - 1)}…`;
  return kept.join("\n");
}

const shown = (t: any) => t.subthemes.filter((s: any) => s.status !== "rejected");

/** Two columns of themes with their sub-themes on the outer side; each theme gets its own band, so nothing collides. */
function layout(themes: any[], width: number, showSubs: boolean) {
  const boxW = Math.min(196, Math.round(width * (showSubs ? 0.2 : 0.42)));
  const pillW = showSubs ? Math.min(216, Math.round(width * 0.21)) : 0;
  const height = (t: any) => Math.max(BOX_H, showSubs ? shown(t).length * (PILL_H + PILL_GAP) - PILL_GAP : 0) + BAND_GAP;
  const cols: any[][] = [[], []], used = [0, 0];
  for (const t of themes) {
    const side = used[0] <= used[1] ? 0 : 1;
    cols[side].push(t);
    used[side] += height(t);
  }
  const chartH = Math.max(320, Math.max(...used));
  const place = new Map<number, { x: number; y: number; side: number; boxW: number }>();
  const pills: any[] = [];
  cols.forEach((col, side) => {
    const boxX = side === 0 ? pillW + (showSubs ? 44 : 8) + boxW / 2 : width - pillW - (showSubs ? 44 : 8) - boxW / 2;
    const pillX = side === 0 ? pillW / 2 + 2 : width - pillW / 2 - 2;
    let y = (chartH - used[side]) / 2;
    for (const t of col) {
      const h = height(t);
      const cy = y + (h - BAND_GAP / 2) / 2;
      place.set(t.id, { x: boxX, y: cy, side, boxW });
      if (showSubs) {
        const span = shown(t).length * (PILL_H + PILL_GAP) - PILL_GAP;
        shown(t).forEach((s: any, i: number) => pills.push({ s, t, i, x: pillX, y: cy - span / 2 + PILL_H / 2 + i * (PILL_H + PILL_GAP) }));
      }
      y += h;
    }
  });
  return { place, pills, chartH, boxW, pillW };
}

export function ThematicMap({ runId }: { runId: string }) {
  const [s, err, loading, reload] = useFetch<any>(`/api/runs/${runId}/thematic`);
  const [sel, setSel] = useState<number | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [codes] = useFetch<any[]>(drawer ? `/api/runs/${runId}/codes` : null);
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const location = useLocation();

  useLayoutEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, [s]);
  useEffect(() => {
    if (s && location.hash === "#thematic-map") document.getElementById("thematic-map")?.scrollIntoView();
  }, [s, location.hash]);

  const themes = useMemo(() => (s?.themes ?? []).filter((t: any) => t.status !== "rejected"), [s]);
  const links = useMemo(() => (s?.links.rows ?? []).filter((r: any) => r.relation !== "chance"), [s]);
  const showSubs = width >= 860;
  const geo = useMemo(() => (width ? layout(themes, width, showSubs) : null), [themes, width, showSubs]);
  const maxLift = Math.max(1.1, ...links.filter((l: any) => l.relation === "together").map((l: any) => l.lift));

  const opt = useMemo(() => () => {
    if (!geo) return {};
    const c = chrome();
    const dim = (id: number) => sel !== null && id !== sel && !links.some((l: any) => (l.a === sel && l.b === id) || (l.b === sel && l.a === id));
    const nodes: any[] = [
      // corner anchors pin the view to the chart box, so the coordinates below are pixels
      { id: "tl", x: 0, y: 0, symbolSize: 0, label: { show: false }, tooltip: { show: false }, silent: true },
      { id: "br", x: width, y: geo.chartH, symbolSize: 0, label: { show: false }, tooltip: { show: false }, silent: true },
    ];
    const edges: any[] = [];
    for (const t of themes) {
      const p = geo.place.get(t.id)!;
      const color = series(t.rank);
      const final = t.status === "approved";
      nodes.push({
        id: `t${t.id}`, x: p.x, y: p.y, theme: t, symbol: rounded(p.boxW, BOX_H, 12), symbolSize: [p.boxW, BOX_H],
        itemStyle: { color: tint(color, c.surface, final ? 0.3 : 0.1), borderColor: color, borderWidth: t.id === sel ? 3.5 : 2, borderType: final ? "solid" : "dashed", opacity: dim(t.id) ? 0.4 : 1 },
        label: { show: true, position: "inside", color: c.ink, fontSize: 12, lineHeight: 15, formatter: () => `{t|${wrap(t.label.replace(/[{}|]/g, " "), Math.floor(p.boxW / 7), 2).replace(/\n/g, "}\n{t|")}}\n{s|${fmt.pct(t.prevalence.segment.value, 0)} of passages}`,
          rich: { t: { fontWeight: 600, fontSize: 12, color: c.ink, lineHeight: 15, align: "center" }, s: { fontSize: 10.5, color: c.ink2, lineHeight: 15, align: "center" } } },
      });
    }
    for (const { s: sub, t, i, x, y } of geo.pills) {
      const id = `s${t.id}-${i}`;
      const color = series(t.rank);
      nodes.push({
        id, x, y, sub, parent: t, symbol: rounded(geo.pillW, PILL_H, 8), symbolSize: [geo.pillW, PILL_H],
        itemStyle: { color: c.surface, borderColor: color, borderWidth: 1, opacity: dim(t.id) ? 0.3 : 1 },
        label: { show: true, position: "inside", color: c.ink2, fontSize: 11, width: geo.pillW - 16, overflow: "truncate", formatter: () => `${sub.label} (${sub.codes.length})` },
      });
      edges.push({ source: `t${t.id}`, target: id, lineStyle: { color, width: 1, opacity: dim(t.id) ? 0.15 : 0.5, curveness: 0 }, silent: true, tooltip: { show: false } });
    }
    for (const l of links) {
      const a = geo.place.get(l.a), b = geo.place.get(l.b);
      if (!a || !b) continue;
      const together = l.relation === "together";
      const on = sel === null || l.a === sel || l.b === sel;
      edges.push({
        source: `t${l.a}`, target: `t${l.b}`, link: l,
        lineStyle: {
          color: together ? c.ink2 : c.muted, type: together ? "solid" : "dashed", opacity: on ? 0.9 : 0.1,
          width: together ? 2 + 5 * Math.min(1, (l.lift - 1) / (maxLift - 1)) : 1.25,
          curveness: a.side === b.side ? 0.5 : 0.08,
        },
      });
    }
    return {
      ...baseOption(),
      tooltip: { ...baseOption().tooltip, formatter: (x: any) => {
        const d = x.data;
        if (d.link) {
          const l = d.link;
          return `<b>${esc(l.a_label)}</b><br/><b>${esc(l.b_label)}</b><br/>${l.relation === "together" ? "Together more often than chance" : "Together less often than chance"}<br/>`
            + `${fmt.n(l.shared)} shared ${s.links.unit}s · ${fmt.n(l.expected, 0)} expected · ${l.lift}× chance`;
        }
        if (d.theme) return `<b>${esc(d.theme.label)}</b><br/>${d.theme.status === "approved" ? "Final theme" : "Candidate theme"} · ${fmt.n(d.theme.segments)} passages · ${d.theme.codes.length} codes`;
        if (d.sub) return `<b>${esc(d.sub.label)}</b><br/>Sub-theme of ${esc(d.parent.label)}<br/>${fmt.n(d.sub.segments)} passages · ${esc(d.sub.codes.slice(0, 6).map((k: any) => k.label).join(", "))}${d.sub.codes.length > 6 ? "…" : ""}`;
        return "";
      } },
      animation: false,
      series: [{ type: "graph", layout: "none", roam: false, left: 0, right: 0, top: 0, bottom: 0, data: nodes, links: edges, emphasis: { disabled: true } }],
    };
  }, [geo, sel, links, themes, width, maxLift, s]); // eslint-disable-line react-hooks/exhaustive-deps

  const theme = themes.find((t: any) => t.id === sel) ?? null;
  const count = (r: string) => (s?.links.rows ?? []).filter((x: any) => x.relation === r).length;
  const hint = showSubs ? "Click a theme for its detail" : "Click a theme for its sub-themes and detail";
  return (
    <div id="thematic-map" className="scroll-mt-4">
      <SectionTitle title="Thematic map" desc="How the themes relate. Boxes are themes and the labels beside them are sub-themes. The code graph above shows the same analysis one level down." />
      {loading && !s && <Loading label="Drawing the thematic map…" />}
      {err && <ErrorBox error={err} />}
      {s && !themes.length && <Unavailable reason="This analysis has no themes to map." />}
      {s && themes.length > 0 && (
        <>
          <Card title="Themes, sub-themes and relationships" subtitle={themes.length < 2 ? "One theme, so no relationships to show" : `${count("together")} of ${s.links.rows.length} theme pairs meet more often than chance · ${count("apart")} less often · ${hint}`}>
            <div ref={box}>
              {geo && <Chart option={opt} height={geo.chartH} name="thematic-map" onClick={(x) => { if (x.data?.theme) setSel(x.data.theme.id === sel ? null : x.data.theme.id); else if (x.data?.parent) setSel(x.data.parent.id); else if (x.data?.link) setSel(x.data.link.a); }} />}
            </div>
            <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink2">
              <span className="flex items-center gap-1.5"><span className="inline-block w-6 border-t-[3px]" style={{ borderColor: chrome().ink2 }} />More often than chance</span>
              <span className="flex items-center gap-1.5"><span className="inline-block w-6 border-t border-dashed" style={{ borderColor: chrome().muted }} />Less often than chance</span>
              <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded border-2 border-dashed border-muted" />Candidate theme</span>
              <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-5 rounded border-2 border-muted bg-muted/25" />Final theme</span>
            </div>
            {themes.length > 1 && <p className="mt-2 text-[11px] leading-relaxed text-muted">{s.links.rule}</p>}
          </Card>
          {theme && (
            <Card className="mt-4" title={theme.label} subtitle={`${fmt.pct(theme.prevalence.segment.value, 0)} of passages · ${theme.codes.length} codes`}
              actions={<>
                <Button small kind="primary" onClick={() => setDrawer(true)}>Open evidence and review<Icon path={ICONS.arrow} size={12} /></Button>
                <Button small kind="ghost" onClick={() => setSel(null)}>Close</Button>
              </>}>
              <ThemeDetail t={theme} rows={s.links.rows} unit={s.links.unit} onPick={setSel} />
            </Card>
          )}
          {s.links.rows.length > 0 && <Disclosure title="All theme pairs" desc="Every pair of themes against what chance would give, including those with no line on the map"><Pairs rows={s.links.rows} unit={s.links.unit} /></Disclosure>}
        </>
      )}
      <ThemeDrawer theme={drawer ? theme : null} runId={runId} onClose={() => setDrawer(false)} onChanged={reload} codes={codes ?? []} themes={s?.themes} />
    </div>
  );
}

const RELATION: Record<string, string> = { together: "more than chance", apart: "less than chance", chance: "about chance" };

function Pairs({ rows, unit }: { rows: any[]; unit: string }) {
  return (
    <Table<any> rows={rows} rowKey={(r) => `${r.a}-${r.b}`} initialSort={{ key: "lift", dir: -1 }}
      cols={[
        { key: "a_label", label: "Theme" }, { key: "b_label", label: "Theme" },
        { key: "relation", label: "Meets", render: (r) => <span className="whitespace-nowrap">{RELATION[r.relation]}</span> },
        { key: "lift", label: "× chance", num: true }, { key: "shared", label: `Shared ${unit}s`, num: true, render: (r) => fmt.n(r.shared) },
        { key: "expected", label: "Expected", num: true, render: (r) => fmt.n(r.expected, 0) },
      ]} />
  );
}

function ThemeDetail({ t, rows, unit, onPick }: { t: any; rows: any[]; unit: string; onPick: (id: number) => void }) {
  const mine = rows.filter((r) => r.a === t.id || r.b === t.id);
  const other = (r: any) => (r.a === t.id ? { id: r.b, label: r.b_label } : { id: r.a, label: r.a_label });
  const together = mine.filter((r) => r.relation === "together");
  const apart = mine.filter((r) => r.relation === "apart");
  return (
    <div className="grid gap-5 text-sm md:grid-cols-2">
      <div className="space-y-3">
        <Badge status={t.status}>{t.status === "approved" ? "final" : undefined}</Badge>
        {t.origin !== "machine" && t.definition && <p className="text-ink2">{t.definition}</p>}
        {shown(t).length > 0 && (
          <div>
            <div className="text-xs font-medium text-ink2">Sub-themes</div>
            <ul className="mt-1 space-y-0.5 text-ink">{shown(t).map((x: any) => <li key={x.id ?? x.label}>{x.label} <span className="text-xs text-muted">{x.codes.length} codes</span></li>)}</ul>
          </div>
        )}
      </div>
      <div className="space-y-3">
        <div>
          <div className="text-xs font-medium text-ink2">Meets more often than chance</div>
          {!together.length && <p className="mt-1 text-muted">No other theme.</p>}
          {together.map((r) => (
            <div key={other(r).id} className="mt-1.5">
              <button className="text-left font-medium text-accent hover:underline" onClick={() => onPick(other(r).id)}>{other(r).label}</button>
              <div className="text-xs text-muted">{r.lift}× chance · {fmt.n(r.shared)} shared {unit}s{r.via ? ` · strongest pair ${r.via.join(" + ")}` : ""}</div>
              {r.quote && <blockquote className="mt-1 border-l-2 border-line pl-2.5 text-xs italic leading-relaxed text-ink2">“{r.quote.text}”</blockquote>}
            </div>
          ))}
        </div>
        {apart.length > 0 && (
          <div>
            <div className="text-xs font-medium text-ink2">Rarely together</div>
            {apart.map((r) => <div key={other(r).id} className="mt-1"><button className="text-left text-accent hover:underline" onClick={() => onPick(other(r).id)}>{other(r).label}</button> <span className="text-xs text-muted">{r.lift}× chance</span></div>)}
          </div>
        )}
      </div>
    </div>
  );
}
