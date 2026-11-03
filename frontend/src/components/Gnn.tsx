/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useState } from "react";
import { api, download, fmt, type Run } from "../api";
import { chrome, series } from "../colors";
import { useApp, useFetch } from "../state";
import { Chart, axis, baseOption, esc, lineStyle } from "./Chart";
import { Badge, Button, Card, ErrorBox, Field, Loading, SectionTitle, Stat, Table, Unavailable, cx, inputBase, inputCls } from "./ui";

export interface PredictedLink { source: number; target: number; score: number; rank: number; adamic_adar: number | null }

const ms = (x: any, d = 3) => (x && x.mean !== null && x.mean !== undefined ? `${x.mean.toFixed(d)}${x.n > 1 ? ` ± ${x.std.toFixed(d)}` : ""}` : "—");

/** Which GraphSAGE implementation trains the models. */
const engineLabel = (st: any) => st?.backend === "pytorch"
  ? `torch ${st.torch} · PyG ${st.torch_geometric} · ${st.cuda_available ? "CUDA available" : "CPU"}`
  : "Built-in GraphSAGE (NumPy) · CPU";

function useGnnStatus() {
  const [st] = useFetch<any>("/api/gnn/status");
  return st;
}

/** Poll a run's GNN experiments while any is queued or running. */
function useExperiments(url: string | null) {
  const [list, err, loading, reload] = useFetch<any[]>(url);
  const active = (list ?? []).some((e) => e.status === "queued" || e.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(reload, 1500);
    return () => clearInterval(t);
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps
  return [list, err, loading, reload] as const;
}

// ---------------------------------------------------------------- Upload

export function GnnUploadCard({ cfg, setCfg }: { cfg: any; setCfg: (c: any) => void }) {
  const st = useGnnStatus();
  const on = !!cfg.gnn;
  const g = cfg.gnn ?? st?.defaults ?? {};
  const set = (k: string, v: unknown) => setCfg({ ...cfg, gnn: { ...g, [k]: v } });
  return (
    <Card title="Graph neural network (optional)" subtitle="GraphSAGE link prediction on the code co-occurrence graph, trained after the pipeline finishes. Neural; not part of the classical pipeline.">
      {!st ? <Loading /> : !st.available ? <Unavailable reason={st.reason} /> : (
        <>
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" className="mt-0.5" checked={on} onChange={(e) => setCfg({ ...cfg, gnn: e.target.checked ? { ...st.defaults } : null })} />
            <span>Train GraphSAGE link prediction after the analysis
              <span className="block text-[11px] text-muted">No labels required. Needs ≥8 codes and ≥20 co-occurrence edges, at most 50% of code pairs linked. Node classification needs researcher-assigned code categories, so start it from Graphs & Patterns after reviewing codes.</span>
            </span>
          </label>
          {on && (
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Field label="Co-occurrence unit" hint="Blank = the run's setting">
                <select className={inputCls} value={g.cooccurrence_level ?? ""} onChange={(e) => set("cooccurrence_level", e.target.value || null)}>
                  <option value="">Run setting</option><option value="segment">Segment (sentence)</option><option value="record">Record</option></select>
              </Field>
              <Field label="Min NPMI" hint="Blank = run setting; raise to sparsify"><input type="number" step={0.05} className={inputCls} value={g.min_npmi ?? ""} onChange={(e) => set("min_npmi", e.target.value === "" ? null : Number(e.target.value))} /></Field>
              <Field label="Repeats (seeds)" hint="Mean ± SD over new splits"><input type="number" min={1} max={20} className={inputCls} value={g.repeats} onChange={(e) => set("repeats", Number(e.target.value))} /></Field>
              <Field label="Max epochs"><input type="number" min={1} className={inputCls} value={g.epochs} onChange={(e) => set("epochs", Number(e.target.value))} /></Field>
              <Field label="Decoder"><select className={inputCls} value={g.decoder} onChange={(e) => set("decoder", e.target.value)}><option value="dot">Dot product</option><option value="mlp">MLP</option></select></Field>
              <Field label="Device"><select className={inputCls} value={g.device} onChange={(e) => set("device", e.target.value)}>{st.devices.map((d: string) => <option key={d}>{d}</option>)}</select></Field>
            </div>
          )}
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------- Dashboard

export function GnnDashboardCard({ runId }: { runId: string }) {
  const st = useGnnStatus();
  const [list] = useExperiments(`/api/runs/${runId}/gnn`);
  const done = (list ?? []).filter((e) => e.status === "completed" && e.task === "link_prediction");
  const [latest] = useFetch<any>(done.length ? `/api/gnn/${done[0].id}` : null);
  return (
    <Card title="Graph neural network (GraphSAGE)" subtitle={st?.available ? engineLabel(st) : "Optional neural module"}>
      {!st ? <Loading /> : !st.available ? <Unavailable reason={st.reason} /> : !list?.length ? (
        <p className="text-sm text-muted">No GNN experiments for this run. Start one in Graphs & Patterns → Graph neural network.</p>
      ) : (
        <>
          <div className="mb-2 text-xs text-ink2">{list.length} experiment(s): {["completed", "running", "queued", "failed", "interrupted"].map((s) => { const n = list.filter((e) => e.status === s).length; return n ? `${n} ${s}` : null; }).filter(Boolean).join(" · ")}</div>
          {latest ? <MetricsTable exp={latest} compact /> : done.length === 0 ? <p className="text-sm text-muted">No completed link-prediction experiment yet.</p> : <Loading />}
        </>
      )}
    </Card>
  );
}

function MetricsTable({ exp, compact }: { exp: any; compact?: boolean }) {
  const m = exp.metrics;
  if (!m) return null;
  if (m.task === "node_classification") {
    const rows = [
      ...["train", "validation", "test"].map((s) => ({ name: `GraphSAGE · ${s}`, kind: s === "test" ? "held-out" : s === "validation" ? "model selection" : "training", ...m.gnn[s] })),
      ...Object.entries(exp.baselines ?? {}).map(([k, v]: any) => ({ name: k.replace(/_/g, " "), kind: "baseline · test", ...v })),
    ];
    return (
      <>
        <Table<any> rows={rows} rowKey={(r) => r.name}
          cols={[{ key: "name", label: "Model" }, { key: "kind", label: "Split" }, { key: "acc", label: "Accuracy", num: true, render: (r) => ms(r.accuracy) },
            { key: "mf1", label: "Macro F1", num: true, render: (r) => ms(r.macro_f1) }, { key: "wf1", label: "Weighted F1", num: true, render: (r) => ms(r.weighted_f1) }]} />
        {m.verdict && <p className="mt-2 text-sm"><span className="font-medium">Verdict:</span> {m.verdict}</p>}
      </>
    );
  }
  const rows = [
    ...(compact ? ["test"] : ["train", "validation", "test"]).map((s) => ({ name: "GraphSAGE", kind: s === "test" ? "test (held-out)" : s === "validation" ? "validation (checkpoint selection)" : "train (supervision edges)", gnn: true, ...m.gnn[s] })),
    ...Object.entries(exp.baselines ?? {}).map(([k, v]: any) => ({ name: k.replace(/_/g, " "), kind: "test (held-out)", best: k === m.best_baseline, ...v })),
  ];
  return (
    <>
      <Table<any> rows={rows} rowKey={(r) => `${r.name}-${r.kind}`}
        cols={[
          { key: "name", label: "Model", render: (r) => <span className={cx(r.gnn && "font-medium", r.best && "font-medium")}>{r.name}{r.best ? " (best baseline)" : ""}</span> },
          { key: "kind", label: "Split", render: (r) => <span className="text-xs text-ink2">{r.kind}</span> },
          { key: "auc", label: "ROC-AUC", num: true, render: (r) => ms(r.roc_auc) },
          { key: "ap", label: "Avg precision", num: true, render: (r) => ms(r.average_precision) },
          { key: "pk", label: "Precision@k", num: true, render: (r) => ms(r.precision_at_k) },
        ]} />
      <p className="mt-2 text-sm"><span className="font-medium">Verdict:</span> {m.verdict}</p>
      <p className="mt-1 text-[11px] text-muted">{m.repeats} repeat(s), mean ± SD over new edge splits. k = number of held-out positive edges ({m.split?.test_pos}). {m.note}</p>
    </>
  );
}

// ---------------------------------------------------------------- Graphs & Patterns

export function GnnSection({ run, g, onPredicted, onOpenCode }: { run: Run; g: any; onPredicted: (p: PredictedLink[] | null) => void; onOpenCode: (id: number) => void }) {
  const { toast } = useApp();
  const st = useGnnStatus();
  const [list, err, , reload] = useExperiments(`/api/runs/${run.id}/gnn`);
  const [sel, setSel] = useState<string | null>(null);
  useEffect(() => {
    if (list?.length && (!sel || !list.find((e) => e.id === sel))) setSel((list.find((e) => e.status === "completed") ?? list[0]).id);
  }, [list]); // eslint-disable-line react-hooks/exhaustive-deps
  const selRow = list?.find((e) => e.id === sel);
  const [exp] = useFetch<any>(sel ? `/api/gnn/${sel}` : null, [selRow?.status]);

  const [form, setForm] = useState<any>(null);
  const [check, setCheck] = useState<any>(null);
  useEffect(() => { if (st?.defaults && !form) setForm({ ...st.defaults, cooccurrence_level: g?.params?.level ?? null, min_cooccurrence: g?.params?.min_cooccurrence ?? null, min_npmi: g?.params?.min_npmi ?? null }); }, [st, g]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (k: string, v: unknown) => { setForm({ ...form, [k]: v }); setCheck(null); };
  const validate = async () => { try { setCheck(await api.post(`/api/runs/${run.id}/gnn/validate`, form)); } catch (e: any) { toast(e.message, "error"); } };
  const start = async () => {
    try { const e = await api.post(`/api/runs/${run.id}/gnn`, form); toast("GNN experiment queued"); setSel(e.id); reload(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const del = async (id: string) => {
    if (!confirm("Delete this experiment, its checkpoint and predictions?")) return;
    try { await api.del(`/api/gnn/${id}`); if (sel === id) setSel(null); reload(); } catch (e: any) { toast(e.message, "error"); }
  };

  return (
    <>
      {!st ? <Loading /> : !st.available ? <Unavailable reason={st.reason} /> : (
        <div className="space-y-4">
          <div className="grid gap-4 xl:grid-cols-[1fr_1.4fr]">
            <Card title="New experiment" subtitle={engineLabel(st)}>
              {!form ? <Loading /> : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="Task"><select className={inputCls} value={form.task} onChange={(e) => set("task", e.target.value)}>
                      <option value="link_prediction">Link prediction (no labels)</option><option value="node_classification">Node classification (needs code categories)</option></select></Field>
                    <Field label="Node features" hint="Never corpus TF-IDF (would leak co-occurrence)"><select className={inputCls} value={form.feature_set} onChange={(e) => set("feature_set", e.target.value)}>
                      <option value="label+frequency">Label n-grams + frequency + method</option><option value="label">Label n-grams</option><option value="identity">One-hot identity (transductive)</option></select></Field>
                    <Field label="Co-occurrence unit"><select className={inputCls} value={form.cooccurrence_level ?? ""} onChange={(e) => set("cooccurrence_level", e.target.value || null)}>
                      <option value="">Run setting</option><option value="segment">Segment</option><option value="record">Record</option></select></Field>
                    <Field label="Min co-occurrence"><input type="number" min={1} className={inputCls} value={form.min_cooccurrence ?? ""} onChange={(e) => set("min_cooccurrence", e.target.value === "" ? null : Number(e.target.value))} /></Field>
                    <Field label="Min NPMI"><input type="number" step={0.05} className={inputCls} value={form.min_npmi ?? ""} onChange={(e) => set("min_npmi", e.target.value === "" ? null : Number(e.target.value))} /></Field>
                    <Field label="Decoder"><select className={inputCls} value={form.decoder} onChange={(e) => set("decoder", e.target.value)}><option value="dot">Dot product</option><option value="mlp">MLP</option></select></Field>
                    <Field label="Hidden / output dim"><div className="flex gap-1"><input type="number" className={inputCls} value={form.hidden_dim} onChange={(e) => set("hidden_dim", Number(e.target.value))} /><input type="number" className={inputCls} value={form.out_dim} onChange={(e) => set("out_dim", Number(e.target.value))} /></div></Field>
                    <Field label="Layers / dropout"><div className="flex gap-1"><input type="number" min={1} max={6} className={inputCls} value={form.num_layers} onChange={(e) => set("num_layers", Number(e.target.value))} /><input type="number" step={0.05} className={inputCls} value={form.dropout} onChange={(e) => set("dropout", Number(e.target.value))} /></div></Field>
                    <Field label="Learning rate / weight decay"><div className="flex gap-1"><input type="number" step={0.001} className={inputCls} value={form.learning_rate} onChange={(e) => set("learning_rate", Number(e.target.value))} /><input type="number" step={0.0001} className={inputCls} value={form.weight_decay} onChange={(e) => set("weight_decay", Number(e.target.value))} /></div></Field>
                    <Field label="Epochs / patience"><div className="flex gap-1"><input type="number" className={inputCls} value={form.epochs} onChange={(e) => set("epochs", Number(e.target.value))} /><input type="number" className={inputCls} value={form.patience} onChange={(e) => set("patience", Number(e.target.value))} /></div></Field>
                    <Field label="Repeats / seed"><div className="flex gap-1"><input type="number" min={1} max={20} className={inputCls} value={form.repeats} onChange={(e) => set("repeats", Number(e.target.value))} /><input type="number" className={inputCls} value={form.seed} onChange={(e) => set("seed", Number(e.target.value))} /></div></Field>
                    <Field label="Device"><select className={inputCls} value={form.device} onChange={(e) => set("device", e.target.value)}>{st.devices.map((d: string) => <option key={d}>{d}</option>)}</select></Field>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button onClick={validate}>Check inputs</Button>
                    <Button kind="primary" onClick={start} disabled={check?.ok === false}>▶ Train</Button>
                  </div>
                  {check && (
                    <div className={cx("mt-3 rounded-md px-3 py-2 text-sm", check.ok ? "bg-good/10 text-good" : "bg-bad/10 text-bad")}>
                      {check.ok ? "✓ Ready" : "✕ Cannot run"} · {check.nodes} codes · {check.edges} co-occurrence edges{check.density !== undefined ? ` · density ${(100 * check.density).toFixed(0)}%` : ""}
                      {check.labels_required ? ` · categories: ${Object.entries(check.categories).map(([k, v]) => `${k} (${v})`).join(", ") || "none"}` : " · no labels required"}
                      {check.reasons.map((r: string) => <div key={r} className="mt-1">{r}</div>)}
                    </div>
                  )}
                </>
              )}
            </Card>
            <Card title="Experiments" subtitle="Click to inspect. Every experiment is kept with its configuration, graph version and checkpoint.">
              {err && <ErrorBox error={err} />}
              <Table<any> rows={list ?? []} rowKey={(r) => r.id} onRow={(r) => setSel(r.id)} maxHeight={420} empty="No experiments yet."
                cols={[
                  { key: "id", label: "Experiment", render: (r) => <span className={cx("font-mono text-xs", r.id === sel && "font-semibold text-accent")}>{r.id}</span> },
                  { key: "task", label: "Task", render: (r) => r.task.replace("_", " ") },
                  { key: "status", label: "Status", render: (r) => <span><Badge status={r.status} />{r.status === "running" ? ` ${Math.round(100 * (r.progress ?? 0))}%` : ""}</span> },
                  { key: "graph", label: "Graph", render: (r) => r.graph_info ? <span className="text-xs">{r.graph_info.nodes} × {r.graph_info.edges} · {r.graph_info.params?.level}</span> : "—" },
                  { key: "test", label: "Test score", num: true, render: (r) => r.metrics?.gnn ? (r.task === "link_prediction" ? `AUC ${ms(r.metrics.gnn.test.roc_auc)}` : `F1 ${ms(r.metrics.gnn.test.macro_f1)}`) : "—" },
                  { key: "created_at", label: "Created", render: (r) => fmt.date(r.created_at) },
                  { key: "x", label: "", render: (r) => !["queued", "running"].includes(r.status) ? <Button small kind="ghost" onClick={() => del(r.id)}>Delete</Button> : null },
                ]} />
            </Card>
          </div>
          {exp && <ExperimentDetail exp={exp} g={g} onPredicted={onPredicted} onOpenCode={onOpenCode} />}
        </div>
      )}
    </>
  );
}

function ExperimentDetail({ exp, g, onPredicted, onOpenCode }: { exp: any; g: any; onPredicted: (p: PredictedLink[] | null) => void; onOpenCode: (id: number) => void }) {
  const { toast } = useApp();
  const done = exp.status === "completed";
  const [preds] = useFetch<any[]>(done ? `/api/gnn/${exp.id}/predictions?limit=200` : null);
  const [emb] = useFetch<any>(done ? `/api/gnn/${exp.id}/embeddings` : null);
  const [shown, setShown] = useState(false);
  const [rescored, setRescored] = useState<any>(null);
  useEffect(() => { setShown(false); setRescored(null); onPredicted(null); }, [exp.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const linkPreds = (rescored?.predictions ?? preds ?? []).filter((p: any) => p.kind === undefined || p.kind === "link");
  const toggle = () => {
    const next = !shown;
    setShown(next);
    onPredicted(next ? linkPreds.map((p: any) => ({ source: p.source_code_id, target: p.target_code_id, score: p.score, rank: p.rank, adamic_adar: p.adamic_adar })) : null);
  };
  const isLink = exp.task === "link_prediction";
  const hist = exp.history ?? [];
  const lossOpt = useMemo(() => () => ({
    ...baseOption(), grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" }, legend: { ...baseOption().legend, data: ["train loss", "validation loss"] },
    xAxis: axis("category", { data: hist.map((h: any) => h.epoch), name: "epoch", nameTextStyle: { color: chrome().muted, fontSize: 11 } }),
    yAxis: axis("value", { scale: true }),
    series: [{ ...lineStyle(series(0)), name: "train loss", data: hist.map((h: any) => h.train_loss) },
      { ...lineStyle(series(1)), name: "validation loss", data: hist.map((h: any) => h.val_loss) }],
  }), [hist]);
  const valKey = isLink ? "val_auc" : "val_macro_f1";
  const valOpt = useMemo(() => () => ({
    ...baseOption(), grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    tooltip: { ...baseOption().tooltip, trigger: "axis" },
    xAxis: axis("category", { data: hist.map((h: any) => h.epoch) }),
    yAxis: axis("value", { scale: true }),
    series: [{ ...lineStyle(series(0)), name: isLink ? "validation AUC" : "validation macro F1", data: hist.map((h: any) => h[valKey]),
      markLine: exp.metrics?.best_epoch ? { symbol: "none", silent: true, lineStyle: { color: chrome().axis, type: "solid" }, label: { color: chrome().muted, fontSize: 10, formatter: "selected checkpoint" }, data: [{ xAxis: String(exp.metrics.best_epoch[0]) }] } : undefined }],
  }), [hist]); // eslint-disable-line react-hooks/exhaustive-deps
  const comm = useMemo(() => Object.fromEntries((g?.nodes ?? []).map((n: any) => [n.id, n.community])), [g]);
  const comms: number[] = useMemo(() => (g?.communities ?? []).filter((c: any) => c.size > 1).map((c: any) => c.community), [g]);
  const embOpt = useMemo(() => () => {
    const pts = emb?.points ?? [];
    const groups = [...comms.slice(0, 8), -999];
    return {
      ...baseOption(), grid: { left: 8, right: 16, top: 30, bottom: 8, containLabel: true },
      tooltip: { ...baseOption().tooltip, formatter: (p: any) => `<b>${esc(p.data[2])}</b><br/>${esc(p.data[3])} segments · community ${esc(p.data[4])}` },
      legend: { ...baseOption().legend, type: "scroll" },
      xAxis: axis("value", { scale: true, axisLabel: { show: false } }), yAxis: axis("value", { scale: true, axisLabel: { show: false } }),
      series: groups.map((cg, i) => ({
        type: "scatter", name: cg === -999 ? "Other / singleton" : `Community ${cg}`, symbolSize: 10,
        itemStyle: { color: series(cg === -999 ? -1 : i), borderColor: chrome().surface, borderWidth: 2 },
        data: pts.filter((p: any) => (cg === -999 ? !comms.slice(0, 8).includes(comm[p.id]) : comm[p.id] === cg)).map((p: any) => [p.x, p.y, p.label, p.frequency, comm[p.id] ?? "—"]),
      })),
    };
  }, [emb, comm, comms]);
  const rescore = async () => {
    try { const r = await api.post(`/api/gnn/${exp.id}/rescore?top_k=50`); setRescored(r); toast(r.graph_changed ? "Re-scored the current (edited) graph with the saved checkpoint" : "Re-scored with the saved checkpoint; graph unchanged", "success"); }
    catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <Card title={`Experiment ${exp.id}`} subtitle={`${exp.task.replace("_", " ")} · ${exp.status}${exp.duration_s ? ` · ${fmt.sec(exp.duration_s)}` : ""}${exp.device ? ` · ${exp.device}` : ""} · graph version ${exp.graph_version ?? "—"}`}
      actions={done ? <>
        <Button small onClick={() => download(`/api/gnn/${exp.id}/export`)}>Predictions (CSV)</Button>
        <Button small onClick={() => download(`/api/gnn/${exp.id}/export?what=experiment`)}>Experiment (JSON)</Button>
      </> : undefined}>
      {exp.error && <ErrorBox error={exp.error} />}
      {exp.status === "running" && <div className="text-sm text-ink2">Training… {Math.round(100 * (exp.progress ?? 0))}%</div>}
      {done && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Graph" value={`${exp.graph_info.nodes} × ${exp.graph_info.edges}`} detail={`codes × observed co-occurrence edges (${exp.graph_info.params.level}, min ${exp.graph_info.params.min_cooccurrence}, NPMI > ${exp.graph_info.params.min_npmi})`} />
            {isLink ? <Stat label="Held-out split (repeat 1)" value={`${exp.metrics.split.test_pos} + ${exp.metrics.split.test_neg}`} detail={`test positive + negative pairs · val ${exp.metrics.split.val_pos}+${exp.metrics.split.val_neg} · train ${exp.metrics.split.train_supervision}`} />
              : <Stat label="Labelled codes" value={fmt.n(exp.metrics.split.labelled)} detail={`train ${exp.metrics.split.train} · val ${exp.metrics.split.val} · test ${exp.metrics.split.test}`} />}
            <Stat label="Selected epoch" value={exp.metrics.best_epoch.join(", ")} detail={`epochs run ${(exp.metrics.epochs_run ?? []).join(", ") || "—"} (early stopping)`} />
            <Stat label="Training time" value={fmt.sec(exp.metrics.training_runtime_s)} detail={`${exp.metrics.repeats} repeat(s) · ${exp.versions?.torch ? `torch ${exp.versions.torch}` : ""}`} />
          </div>
          <MetricsTable exp={exp} />
          <div className="grid gap-4 xl:grid-cols-2">
            <div><div className="mb-1 text-xs font-medium text-ink2">Training monitor (repeat 1): loss</div><Chart option={lossOpt} height={220} name={`gnn-loss-${exp.id}`} /></div>
            <div><div className="mb-1 text-xs font-medium text-ink2">{isLink ? "Validation ROC-AUC" : "Validation macro F1"} per epoch</div><Chart option={valOpt} height={220} name={`gnn-val-${exp.id}`} /></div>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <div>
              <div className="mb-1 text-xs font-medium text-ink2">Learned code representations</div>
              <div className="mb-1 text-[11px] text-muted">{emb?.method}. Colour = classical graph community (Graphs & Patterns, section B), for comparison.</div>
              {emb?.points?.length ? <Chart option={embOpt} height={320} name={`gnn-embeddings-${exp.id}`} /> : <Loading />}
            </div>
            {isLink ? (
              <div>
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-ink2">Predicted relationships (not observed)</span>
                  <Badge status="needs_revision">predictions — hypotheses only</Badge>
                  <span className="ml-auto flex gap-1">
                    <Button small onClick={toggle}>{shown ? "Hide on graph" : "Show on graph"}</Button>
                    <Button small kind="ghost" onClick={rescore} title="Load the saved checkpoint and score the current graph">Re-score current graph</Button>
                  </span>
                </div>
                {rescored && <p className="mb-1 text-[11px] text-muted">Re-scored with checkpoint (trained on graph {rescored.trained_on_graph_version}; current {rescored.graph_version}{rescored.graph_changed ? ", changed since training" : ", unchanged"}).</p>}
                <Table<any> rows={linkPreds} rowKey={(r) => `${r.source_code_id}-${r.target_code_id}`} maxHeight={300}
                  cols={[
                    { key: "rank", label: "#", num: true },
                    { key: "pair", label: "Code pair (no observed co-occurrence edge)", render: (r) => <span><button className="underline decoration-dotted" onClick={() => onOpenCode(r.source_code_id)}>{r.source_label}</button> — <button className="underline decoration-dotted" onClick={() => onOpenCode(r.target_code_id)}>{r.target_label}</button></span> },
                    { key: "score", label: "GNN score", num: true, render: (r) => r.score.toFixed(3) },
                    { key: "adamic_adar", label: "Adamic–Adar", num: true, render: (r) => r.adamic_adar ?? "—" },
                  ]} />
                <p className="mt-1 text-[11px] text-muted">Pairs that do not meet the co-occurrence threshold in the data, ranked by the selected checkpoint (repeat 1) using all observed edges. Click a code to read its source passages.</p>
              </div>
            ) : (
              <div>
                <div className="mb-1 text-xs font-medium text-ink2">Test confusion matrix (repeat 1)</div>
                <ConfusionMatrix cm={exp.metrics.test_detail_first_repeat.confusion_matrix} />
              </div>
            )}
          </div>
          <details className="text-xs text-ink2">
            <summary className="cursor-pointer">Configuration, features and provenance</summary>
            <dl className="mt-2 grid grid-cols-1 gap-y-1 sm:grid-cols-[12rem_minmax(0,1fr)]">
              <dt className="text-muted">Node features</dt><dd>{exp.graph_info.features}</dd>
              <dt className="text-muted">Edges dropped</dt><dd>{Object.entries(exp.graph_info.dropped_edges).map(([k, v]) => `${k}: ${v}`).join(" · ")}</dd>
              <dt className="text-muted">Hyperparameters</dt><dd className="break-all font-mono">{Object.entries(exp.config).map(([k, v]) => `${k}=${v}`).join(" · ")}</dd>
              <dt className="text-muted">Software</dt><dd>{Object.entries(exp.versions ?? {}).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>
              <dt className="text-muted">Checkpoint</dt><dd className="break-all font-mono">{exp.checkpoint_path}</dd>
            </dl>
          </details>
        </div>
      )}
    </Card>
  );
}

function ConfusionMatrix({ cm }: { cm: { labels: string[]; matrix: number[][] } }) {
  return (
    <table className="text-xs tabular-nums">
      <thead><tr><th className="pr-2 text-left font-medium text-muted">true ↓ / predicted →</th>{cm.labels.map((l) => <th key={l} className="px-2 text-right font-medium text-ink2">{l}</th>)}</tr></thead>
      <tbody>{cm.matrix.map((r, i) => <tr key={i} className="border-t border-line"><th className="pr-2 text-left font-medium text-ink2">{cm.labels[i]}</th>{r.map((v, j) => <td key={j} className={cx("px-2 text-right", i === j && "font-semibold")}>{v}</td>)}</tr>)}</tbody>
    </table>
  );
}

// ---------------------------------------------------------------- Database

export function GnnDatabaseSection({ projectId }: { projectId: string }) {
  const { toast } = useApp();
  const [list, err, , reload] = useExperiments(`/api/projects/${projectId}/gnn`);
  const [status, setStatus] = useState("all");
  const [task, setTask] = useState("all");
  const rows = (list ?? []).filter((e) => (status === "all" || e.status === status) && (task === "all" || e.task === task));
  const del = async (id: string) => {
    if (!confirm("Delete this experiment, its checkpoint and predictions?")) return;
    try { await api.del(`/api/gnn/${id}`); reload(); } catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <>
      <SectionTitle id="gnn-experiments" title="GNN experiments" desc="Configurations, training history, metrics, predictions and checkpoints for every GraphSAGE experiment in this project" />
      <Card actions={<>
        <select className={cx(inputBase, "w-36")} value={task} onChange={(e) => setTask(e.target.value)}><option value="all">All tasks</option><option value="link_prediction">Link prediction</option><option value="node_classification">Node classification</option></select>
        <select className={cx(inputBase, "w-32")} value={status} onChange={(e) => setStatus(e.target.value)}><option value="all">All statuses</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="running">Running</option><option value="interrupted">Interrupted</option></select>
      </>}>
        {err && <ErrorBox error={err} />}
        <Table<any> rows={rows} rowKey={(r) => r.id} maxHeight={400} empty="No GNN experiments." initialSort={{ key: "created_at", dir: -1 }}
          cols={[
            { key: "id", label: "Experiment", render: (r) => <span className="font-mono text-xs">{r.id}</span> },
            { key: "run_id", label: "Run", render: (r) => <span className="font-mono text-xs">{r.run_id}</span> },
            { key: "task", label: "Task", render: (r) => r.task.replace("_", " ") },
            { key: "status", label: "Status", render: (r) => <div><Badge status={r.status} />{r.error && <div className="max-w-xs text-[11px] text-bad">{r.error}</div>}</div> },
            { key: "graph_version", label: "Graph version", render: (r) => <span className="font-mono text-xs">{r.graph_version ?? "—"}</span> },
            { key: "graph", label: "Nodes × edges", render: (r) => r.graph_info ? `${r.graph_info.nodes} × ${r.graph_info.edges}` : "—" },
            { key: "cfg", label: "Model", render: (r) => <span className="text-xs text-ink2">GraphSAGE {r.config?.num_layers}L · {r.config?.hidden_dim}/{r.config?.out_dim} · {r.config?.decoder} · lr {r.config?.learning_rate} · seed {r.config?.seed} ×{r.config?.repeats}</span> },
            { key: "test", label: "Test (mean ± SD)", render: (r) => r.metrics?.gnn ? (r.task === "link_prediction" ? `AUC ${ms(r.metrics.gnn.test.roc_auc)}` : `macro F1 ${ms(r.metrics.gnn.test.macro_f1)}`) : "—" },
            { key: "base", label: "Best baseline", render: (r) => r.metrics?.best_baseline ? <span className="text-xs">{r.metrics.best_baseline.replace(/_/g, " ")}</span> : "—" },
            { key: "device", label: "Device", render: (r) => r.device ?? "—" },
            { key: "duration_s", label: "Duration", num: true, render: (r) => fmt.sec(r.duration_s) },
            { key: "created_at", label: "Created", render: (r) => fmt.date(r.created_at) },
            {
              key: "x", label: "Actions", render: (r) => (
                <div className="flex gap-1">
                  {r.status === "completed" && <Button small kind="ghost" onClick={() => download(`/api/gnn/${r.id}/export`).catch((e) => toast(e.message, "error"))}>CSV</Button>}
                  <Button small kind="ghost" onClick={() => download(`/api/gnn/${r.id}/export?what=experiment`).catch((e) => toast(e.message, "error"))}>JSON</Button>
                  {!["queued", "running"].includes(r.status) && <Button small kind="danger" onClick={() => del(r.id)}>Delete</Button>}
                </div>
              ),
            },
          ]} />
      </Card>
    </>
  );
}
