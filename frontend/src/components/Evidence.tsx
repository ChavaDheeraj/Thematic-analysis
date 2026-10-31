/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { api, fmt } from "../api";
import { useApp, useFetch } from "../state";
import { Badge, Button, ErrorBox, Field, Loading, Modal, PctText, Quote, cx, inputCls } from "./ui";

function Passage({ q, children }: { q: any; children?: React.ReactNode }) {
  const off = q.match_start !== undefined && q.match_start !== null ? q.match_start - q.char_start : null;
  const end = q.match_end !== undefined && q.match_end !== null ? q.match_end - q.char_start : null;
  return (
    <div className="rounded-md border border-line bg-surface p-3">
      <Quote text={q.text} start={off} end={end} before={q.context_before?.slice(-120)} after={q.context_after?.slice(0, 120)} />
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted">
        <span>Record <span className="font-mono text-ink2">{q.record_key}</span></span>
        <span>{q.source_file}</span>
        <span>chars {q.char_start}–{q.char_end}</span>
        {q.timestamp && <span>{fmt.day(q.timestamp)}</span>}
        {q.group && <span>group {q.group}</span>}
        {q.sentiment_label && <span>sentiment {q.sentiment_label} ({q.sentiment?.toFixed(2)})</span>}
        {q.method && <span>method {q.method}{q.score !== null && q.score !== undefined ? ` · score ${q.score.toFixed(3)}` : ""}</span>}
        {q.negated && <Badge status="needs_revision">negation nearby</Badge>}
        <span className={q.verified ? "text-good" : "text-bad"}>{q.verified ? "✓ verbatim match to source" : "✕ does not match source"}</span>
        {children}
      </div>
    </div>
  );
}

export function CodeDrawer({ codeId, onClose, onChanged }: { codeId: number | null; onClose: () => void; onChanged: () => void }) {
  const { toast } = useApp();
  const [ev, err, loading, reload] = useFetch<any>(codeId ? `/api/codes/${codeId}/evidence?limit=300` : null);
  const [picked, setPicked] = useState<number[]>([]);
  const [splitLabel, setSplitLabel] = useState("");
  const [edit, setEdit] = useState<any>(null);
  useEffect(() => { setPicked([]); setSplitLabel(""); }, [codeId]);
  useEffect(() => { if (ev) setEdit({ label: ev.code.label, description: ev.code.description ?? "", category: ev.code.category ?? "", notes: ev.code.notes ?? "" }); }, [ev]);

  const save = async (extra: Record<string, unknown> = {}) => {
    try { await api.patch(`/api/codes/${codeId}`, { ...edit, ...extra }); toast("Code saved", "success"); reload(); onChanged(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const split = async () => {
    try {
      await api.post(`/api/codes/${codeId}/split`, { assignment_ids: picked, new_label: splitLabel });
      toast(`${picked.length} passages moved to “${splitLabel}”`, "success");
      setPicked([]); setSplitLabel(""); reload(); onChanged();
    } catch (e: any) { toast(e.message, "error"); }
  };
  return (
    <Modal open={codeId !== null} onClose={onClose} title={ev ? `Code · ${ev.code.label}` : "Code"} wide>
      {loading && <Loading />}
      {err && <ErrorBox error={err} />}
      {ev && edit && (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
            <div className="space-y-2">
              <Field label="Label"><input className={inputCls} value={edit.label} onChange={(e) => setEdit({ ...edit, label: e.target.value })} /></Field>
              <Field label="Definition / description"><textarea className={cx(inputCls, "h-20")} value={edit.description} onChange={(e) => setEdit({ ...edit, description: e.target.value })} /></Field>
            </div>
            <div className="space-y-2">
              <Field label="Category" hint="Group codes into categories"><input className={inputCls} value={edit.category} onChange={(e) => setEdit({ ...edit, category: e.target.value })} /></Field>
              <Field label="Researcher notes"><textarea className={cx(inputCls, "h-20")} value={edit.notes} onChange={(e) => setEdit({ ...edit, notes: e.target.value })} /></Field>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge status={ev.code.status} />
            <span className="text-muted">method {ev.code.method}{ev.code.feature ? ` · feature “${ev.code.feature}”` : ""} · {ev.code.origin === "machine" ? "machine-generated candidate" : "researcher-edited"}</span>
            <div className="ml-auto flex gap-2">
              <Button small onClick={() => save()}>Save edits</Button>
              <Button small kind="primary" onClick={() => save({ status: "approved" })}>✓ Approve</Button>
              <Button small kind="danger" onClick={() => save({ status: "rejected" })}>✕ Reject</Button>
              {ev.code.status !== "candidate" && <Button small kind="ghost" onClick={() => save({ status: "candidate" })}>Reset to candidate</Button>}
            </div>
          </div>
          <div className="rounded-md bg-raised p-3">
            <div className="text-sm font-medium">Split code</div>
            <div className="mt-1 text-xs text-muted">Tick passages below, then move them into a new code. {picked.length} selected.</div>
            <div className="mt-2 flex gap-2">
              <input className={cx(inputCls, "max-w-xs")} placeholder="New code label" value={splitLabel} onChange={(e) => setSplitLabel(e.target.value)} />
              <Button small disabled={!picked.length || !splitLabel.trim()} onClick={split}>Move selected</Button>
            </div>
          </div>
          <div className="text-sm text-ink2">Showing {ev.items.length} of {fmt.n(ev.total)} supporting passages (exact source text; the matched phrase is highlighted).</div>
          <div className="space-y-2">
            {ev.items.map((q: any) => (
              <label key={q.assignment_id} className="flex gap-2">
                <input type="checkbox" className="mt-3" checked={picked.includes(q.assignment_id)}
                  onChange={(e) => setPicked(e.target.checked ? [...picked, q.assignment_id] : picked.filter((x) => x !== q.assignment_id))} />
                <div className="min-w-0 flex-1"><Passage q={q} /></div>
              </label>
            ))}
          </div>
        </div>
      )}
    </Modal>
  );
}

export function ThemeDrawer({ theme, runId, onClose, onChanged, codes, themes }: { theme: any | null; runId: string; onClose: () => void; onChanged: () => void; codes: any[]; themes?: any[] }) {
  const { toast } = useApp();
  const [ev, err, loading, reload] = useFetch<any>(theme ? `/api/themes/${theme.id}/evidence` : null);
  const [edit, setEdit] = useState<any>(null);
  useEffect(() => {
    if (theme) setEdit({ label: theme.label, definition: theme.definition ?? "", notes: theme.notes ?? "", code_ids: theme.code_ids, parent_id: theme.parent_id ?? 0 });
  }, [theme]);
  const save = async (extra: Record<string, unknown> = {}) => {
    try { await api.patch(`/api/themes/${theme.id}`, { ...edit, ...extra }); toast("Theme saved", "success"); onChanged(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const verify = async (segment_id: number, verified: boolean) => {
    try { await api.post(`/api/runs/${runId}/annotations`, { segment_id, verified, theme_id: theme.id }); reload(); onChanged(); }
    catch (e: any) { toast(e.message, "error"); }
  };
  const active = codes.filter((c) => c.status === "candidate" || c.status === "approved");
  const parents = (themes ?? []).filter((t) => t.id !== theme?.id && !t.parent_id);
  return (
    <Modal open={theme !== null} onClose={onClose} title={theme ? `Theme · ${theme.label}` : "Theme"} wide>
      {theme && edit && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge status={theme.status} />
            <span className="text-muted">{theme.origin === "machine" ? "Machine-generated (template label)" : "Researcher-edited"} · source {theme.source}</span>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Field label="Label"><input className={inputCls} value={edit.label} onChange={(e) => setEdit({ ...edit, label: e.target.value })} /></Field>
              <Field label="Definition" hint="Write the researcher-authored definition here."><textarea className={cx(inputCls, "h-28")} value={edit.definition} onChange={(e) => setEdit({ ...edit, definition: e.target.value })} /></Field>
              <Field label="Notes"><textarea className={cx(inputCls, "h-16")} value={edit.notes} onChange={(e) => setEdit({ ...edit, notes: e.target.value })} /></Field>
              {parents.length > 0 && (
                <Field label="Position">
                  <select className={inputCls} value={edit.parent_id} onChange={(e) => setEdit({ ...edit, parent_id: Number(e.target.value) })}>
                    <option value={0}>Main theme</option>{parents.map((t) => <option key={t.id} value={t.id}>Sub-theme of {t.label}</option>)}
                  </select>
                </Field>
              )}
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-ink2">Associated codes ({edit.code_ids.length})</div>
              <div className="max-h-64 space-y-0.5 overflow-y-auto rounded-md border border-line p-2 text-sm">
                {active.map((c) => (
                  <label key={c.id} className="flex items-center gap-2">
                    <input type="checkbox" checked={edit.code_ids.includes(c.id)} onChange={(e) => setEdit({ ...edit, code_ids: e.target.checked ? [...edit.code_ids, c.id] : edit.code_ids.filter((x: number) => x !== c.id) })} />
                    <span className="truncate">{c.label}</span><span className="ml-auto text-xs tabular-nums text-muted">{c.segments}</span>
                  </label>
                ))}
              </div>
              <dl className="mt-3 grid grid-cols-[9rem_1fr] gap-y-1 text-xs">
                <dt className="text-muted">Respondent coverage</dt><dd><PctText p={theme.prevalence.respondent} /></dd>
                <dt className="text-muted">Record coverage</dt><dd><PctText p={theme.prevalence.record} /></dd>
                <dt className="text-muted">Supporting passages</dt><dd>{fmt.n(theme.segments)}</dd>
                <dt className="text-muted">Majority polarity</dt><dd>{theme.majority_polarity ?? "—"}</dd>
                <dt className="text-muted">Closest topic</dt><dd>{theme.associated_topic ? `Topic ${theme.associated_topic.topic} (${theme.associated_topic.share_pct}% of passages)` : "—"}</dd>
              </dl>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button small onClick={() => save()}>Save edits</Button>
            <Button small kind="primary" onClick={() => save({ status: "approved" })}>✓ Approve theme</Button>
            <Button small onClick={() => save({ status: "needs_revision" })}>! Needs revision</Button>
            <Button small kind="danger" onClick={() => save({ status: "rejected" })}>✕ Reject</Button>
          </div>
          {loading && <Loading />}
          {err && <ErrorBox error={err} />}
          {ev && (
            <>
              <div>
                <h4 className="mb-2 font-semibold">Representative quotations <span className="text-xs font-normal text-muted">ranked by theme codes matched, then similarity to the theme centroid</span></h4>
                {ev.representative.length === 0 && <p className="text-sm text-muted">None stored (researcher-created theme).</p>}
                <div className="space-y-2">{ev.representative.map((q: any) => (
                  <Passage key={q.segment_id} q={q}>
                    <span>{q.codes_matched} codes matched</span>
                    <VerifyButtons state={q.researcher_verified} onSet={(v) => verify(q.segment_id, v)} />
                  </Passage>
                ))}</div>
              </div>
              <div>
                <h4 className="mb-1 font-semibold">Contradictory evidence (candidates)</h4>
                <p className="mb-2 text-xs text-muted">{ev.contradictory_rule}</p>
                {ev.contradictory.length === 0 && <p className="text-sm text-muted">No polarity-contrasting passages found{theme.majority_polarity ? "" : " (sentiment not available)"}.</p>}
                <div className="space-y-2">{ev.contradictory.map((q: any) => (
                  <Passage key={q.segment_id} q={q}><VerifyButtons state={q.researcher_verified} onSet={(v) => verify(q.segment_id, v)} /></Passage>
                ))}</div>
              </div>
            </>
          )}
        </div>
      )}
    </Modal>
  );
}

function VerifyButtons({ state, onSet }: { state: boolean | null; onSet: (v: boolean) => void }) {
  return (
    <span className="ml-auto flex gap-1">
      <button className={cx("rounded border px-1.5 py-0.5", state === true ? "border-good bg-good/10 text-good" : "border-line text-ink2")} onClick={() => onSet(true)}>✓ Supports</button>
      <button className={cx("rounded border px-1.5 py-0.5", state === false ? "border-bad bg-bad/10 text-bad" : "border-line text-ink2")} onClick={() => onSet(false)}>✕ Does not</button>
    </span>
  );
}
