import { useEffect, useState, type ReactNode } from "react";
import { fmt, type Pct } from "../api";
import { Icon, ICONS } from "./icons";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function Card({ title, subtitle, actions, children, className, id }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; id?: string;
}) {
  return (
    <section id={id} className={cx("rounded-xl border border-line bg-surface", className)} style={{ boxShadow: "var(--shadow-sm)" }}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-line px-5 py-3.5">
          <div className="min-w-0">
            {title && <h3 className="text-[15px] font-semibold text-ink">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

/** Page title row: heading and one-line description on the left, page actions on the right. */
export function PageHeader({ title, desc, actions }: { title: string; desc?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-[26px] font-semibold leading-tight tracking-tight text-ink">{title}</h1>
        {desc && <p className="mt-1 max-w-2xl text-sm text-ink2">{desc}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function SectionTitle({ id, title, desc }: { id?: string; title: ReactNode; desc?: ReactNode }) {
  return (
    <div id={id} className="mb-4 mt-12 scroll-mt-16">
      <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
      {desc && <p className="mt-1 max-w-4xl text-[13px] leading-relaxed text-muted">{desc}</p>}
    </div>
  );
}

/** Stat tile: label · value · optional detail line. */
export function Stat({ label, value, detail, hint }: { label: string; value: ReactNode; detail?: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4 transition-shadow hover:shadow-sm" title={hint} style={{ boxShadow: "var(--shadow-sm)" }}>
      <div className="text-xs font-medium text-muted uppercase tracking-wide">{label}</div>
      <div className="mt-2 text-2xl font-bold tabular-nums text-ink">{value}</div>
      {detail && <div className="mt-1.5 text-xs text-muted">{detail}</div>}
    </div>
  );
}

/** A percentage with its numerator / denominator always visible. */
export function PctStat({ label, p, unavailable }: { label: string; p?: Pct | null; unavailable?: string }) {
  if (!p || p.value === null || p.value === undefined) {
    return <Stat label={label} value={<span className="text-muted">—</span>} detail={unavailable ?? "Not available"} />;
  }
  return (
    <Stat
      label={label}
      value={fmt.pct(p.value, 2)}
      detail={<span>{fmt.n(p.numerator)} / {fmt.n(p.denominator)}{p.ci95 ? ` · 95% CI ${p.ci95[0]}–${p.ci95[1]}%` : ""}</span>}
      hint={p.definition}
    />
  );
}

export function PctText({ p }: { p?: Pct | null }) {
  if (!p || p.value === null) return <span className="text-muted">—</span>;
  return <span title={p.definition}>{fmt.pct(p.value, 1)} <span className="text-muted">({fmt.n(p.numerator)}/{fmt.n(p.denominator)})</span></span>;
}

const BADGE: Record<string, string> = {
  completed: "bg-good/12 text-good border-good/20",
  approved: "bg-good/12 text-good border-good/20",
  ready: "bg-accent/12 text-accent border-accent/20",
  analysed: "bg-good/12 text-good border-good/20",
  running: "bg-accent/12 text-accent border-accent/20",
  queued: "bg-raised text-ink2 border-line",
  uploaded: "bg-raised text-ink2 border-line",
  candidate: "bg-raised text-ink2 border-line",
  failed: "bg-bad/12 text-bad border-bad/20",
  rejected: "bg-bad/12 text-bad border-bad/20",
  interrupted: "bg-warn/12 text-warn border-warn/20",
  merged: "bg-raised text-muted border-line",
  needs_revision: "bg-warn/12 text-warn border-warn/20",
};
const ICON: Record<string, string> = {
  completed: "✓", approved: "✓", analysed: "✓",
  failed: "✕", rejected: "✕",
  interrupted: "!", needs_revision: "!",
  running: "◌",
};

export function Badge({ status, children }: { status: string; children?: ReactNode }) {
  return (
    <span className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold", BADGE[status] ?? "bg-raised text-ink2 border-line")}>
      {ICON[status] && <span aria-hidden className="text-[10px]">{ICON[status]}</span>}
      {children ?? status.replace("_", " ")}
    </span>
  );
}

export function Button({ children, onClick, kind = "default", disabled, title, type = "button", small, large }: {
  children: ReactNode; onClick?: () => void; kind?: "default" | "primary" | "danger" | "ghost"; disabled?: boolean; title?: string;
  type?: "button" | "submit"; small?: boolean; large?: boolean;
}) {
  const k = {
    default: "border border-line bg-surface text-ink hover:bg-raised",
    primary: "border-transparent bg-accent text-accentfg hover:opacity-90",
    danger: "border border-bad/30 text-bad hover:bg-bad/8",
    ghost: "border-transparent text-ink2 hover:bg-raised hover:text-ink",
  }[kind];
  const shadow = kind === "primary"
    ? "0 1px 3px 0 rgb(var(--accent) / 0.35)"
    : kind !== "ghost"
    ? "var(--shadow-sm)"
    : "none";
  return (
    <button type={type} title={title} disabled={disabled} onClick={onClick}
      style={{ boxShadow: shadow }}
      className={cx(
        "inline-flex items-center gap-1.5 rounded-lg font-medium transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-40",
        "active:scale-[0.97]",
        small ? "px-2.5 py-1.5 text-xs" : large ? "px-5 py-2.5 text-[15px]" : "px-3.5 py-2 text-sm", k
      )}>
      {children}
    </button>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-semibold text-ink2">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-[11px] text-muted leading-relaxed">{hint}</span>}
    </label>
  );
}

/** Input styling without a width, for controls the caller sizes: `cx(inputBase, "w-40")`. */
export const inputBase = "rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/15 disabled:opacity-50 disabled:cursor-not-allowed transition-all";
export const inputCls = `w-full ${inputBase}`;

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line px-8 py-16 text-center bg-surface/60">
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-accent/10 text-accent"><Icon path={ICONS.file} size={20} /></div>
      <div className="font-semibold text-ink text-[15px]">{title}</div>
      {children && <div className="mx-auto mt-2 max-w-md text-sm text-muted leading-relaxed">{children}</div>}
    </div>
  );
}

/** A collapsed panel for secondary detail: title and one-line description, content on demand. */
export function Disclosure({ title, desc, children, defaultOpen, id }: { title: ReactNode; desc?: ReactNode; children: ReactNode; defaultOpen?: boolean; id?: string }) {
  return (
    <details id={id} open={defaultOpen} className="group mt-3 scroll-mt-16 rounded-xl border border-line bg-surface" style={{ boxShadow: "var(--shadow-sm)" }}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl px-5 py-3.5 hover:bg-raised/50 [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="text-sm font-semibold text-ink">{title}</span>
          {desc && <span className="mt-0.5 block text-xs text-muted">{desc}</span>}
        </span>
        <span className="shrink-0 text-muted transition-transform group-open:rotate-180"><Icon path={ICONS.chevron} size={16} /></span>
      </summary>
      <div className="border-t border-line px-5 pb-5 [&>:first-child]:mt-4">{children}</div>
    </details>
  );
}

export function Unavailable({ reason }: { reason?: string }) {
  return (
    <div className="rounded-lg bg-raised/60 border border-line/50 px-4 py-3 text-sm text-ink2">
      <span className="font-semibold">Unavailable.</span> {reason ?? "This analysis was not computed for the selected run."}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-sm text-muted">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="rounded-xl border border-bad/25 bg-bad/6 px-4 py-3 text-sm text-bad flex items-start gap-2.5">
      <span className="mt-px shrink-0 font-bold">✕</span>
      <span>{error}</span>
    </div>
  );
}

export function Modal({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-[2px]" onClick={onClose}>
      <div
        className={cx("fade-in h-full overflow-y-auto border-l border-line bg-page", wide ? "w-full max-w-5xl" : "w-full max-w-2xl")}
        style={{ boxShadow: "var(--shadow-xl)" }}
        onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-line bg-page/95 backdrop-blur-sm px-6 py-4">
          <h2 className="truncate text-base font-semibold text-ink">{title}</h2>
          <button onClick={onClose} title="Close (Esc)"
            className="rounded-lg p-1.5 text-muted hover:bg-raised hover:text-ink transition-colors">
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void }) {
  return (
    <div className="flex flex-wrap gap-0.5 border-b border-line">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={cx(
            "-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-all duration-150",
            value === t.id
              ? "border-accent text-ink"
              : "border-transparent text-muted hover:text-ink2 hover:border-line"
          )}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/** Simple sortable table. Columns render with tabular figures for numeric alignment. */
export interface Col<T> { key: string; label: string; render?: (r: T) => ReactNode; num?: boolean; sort?: (r: T) => number | string; width?: string }

export function Table<T>({ rows, cols, rowKey, onRow, maxHeight, initialSort, empty = "No rows." }: {
  rows: T[]; cols: Col<T>[]; rowKey: (r: T) => string | number; onRow?: (r: T) => void; maxHeight?: number;
  initialSort?: { key: string; dir: 1 | -1 }; empty?: string;
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const col = sort && cols.find((c) => c.key === sort.key);
  const sorted = col
    ? [...rows].sort((a, b) => {
        const f = col.sort ?? ((r: T) => (r as Record<string, unknown>)[col.key] as number | string);
        const x = f(a), y = f(b);
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (x < y ? -1 : 1) * sort!.dir;
      })
    : rows;
  return (
    <div className="overflow-auto rounded-xl border border-line" style={maxHeight ? { maxHeight } : undefined}>
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 z-[1] bg-raised/90 backdrop-blur-sm text-left text-xs text-ink2">
          <tr>
            {cols.map((c) => (
              <th key={c.key} style={{ width: c.width }}
                className={cx("cursor-pointer select-none whitespace-nowrap px-3.5 py-2.5 font-semibold tracking-wide border-b border-line", c.num && "text-right")}
                onClick={() => setSort((s) => ({ key: c.key, dir: s?.key === c.key ? ((-s.dir) as 1 | -1) : (c.num ? -1 : 1) }))}>
                {c.label}{sort?.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={cols.length} className="px-3 py-10 text-center text-muted">{empty}</td></tr>
          )}
          {sorted.map((r) => (
            <tr key={rowKey(r)} onClick={onRow ? () => onRow(r) : undefined}
              className={cx("border-t border-line/60", onRow && "cursor-pointer hover:bg-accent/4 transition-colors")}>
              {cols.map((c) => (
                <td key={c.key} className={cx("px-3.5 py-2 align-top", c.num && "text-right tabular-nums")}>
                  {c.render ? c.render(r) : String((r as Record<string, unknown>)[c.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Highlights [start, end) inside text without altering it. */
export function Quote({ text, start, end, before, after }: { text: string; start?: number | null; end?: number | null; before?: string; after?: string }) {
  let body: ReactNode = text;
  if (start !== null && start !== undefined && end !== null && end !== undefined && end > start && end <= text.length) {
    body = <>{text.slice(0, start)}<mark className="rounded bg-accent/20 px-0.5 py-px text-ink">{text.slice(start, end)}</mark>{text.slice(end)}</>;
  }
  return (
    <blockquote className="whitespace-pre-wrap border-l-[3px] border-accent/30 pl-4 text-sm leading-relaxed text-ink/90">
      {before && <span className="text-muted">…{before}</span>}
      <span className="text-ink">{body}</span>
      {after && <span className="text-muted">{after}…</span>}
    </blockquote>
  );
}
