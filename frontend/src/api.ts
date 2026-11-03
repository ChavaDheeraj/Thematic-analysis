/* eslint-disable @typescript-eslint/no-explicit-any */
import { engineFetch } from "./engine/client";

/** The browser edition answers /api requests inside the page (engine/); the server edition sends them to the server. */
export const IN_BROWSER = import.meta.env.VITE_ENGINE === "browser";
const send: (url: string, init?: RequestInit) => Promise<Response> = IN_BROWSER ? engineFetch : (url, init) => fetch(url, init);
// in the browser edition a failed request is the engine's own error, never a missing server
const unreachable = (e: unknown) => new ApiError(0, IN_BROWSER ? String((e as Error)?.message ?? e) : NO_SERVER);

export type Pct = { numerator: number; denominator: number; value: number | null; definition: string; level?: string; ci95?: [number, number] };

export interface Project { id: string; name: string; description: string; created_at: string; n_datasets: number; n_runs: number }
export interface Dataset {
  id: string; project_id: string; filename: string; format: string; size_bytes: number; uploaded_at: string;
  status: "uploaded" | "ready" | "analysed" | "failed"; columns: string[] | null; n_rows: number | null; error: string | null;
  mapping: { suggested?: Mapping; applied?: Mapping } | null; quality: any; warnings: string[] | null; n_runs?: number;
  encoding?: string | null; structured?: number;
}
export interface Mapping {
  text_col: string | null; id_col?: string | null; time_col?: string | null; group_col?: string | null; label_col?: string | null;
  metadata_cols?: string[]; duplicate_policy?: "flag" | "exclude"; language_policy?: "keep" | "exclude_non_english";
}
export interface Run {
  id: string; project_id: string; dataset_ids: string[]; label: string; config: any; status: "queued" | "running" | "completed" | "failed" | "interrupted";
  stage: string | null; progress: number; completed_stages: string[]; error: string | null; failed_stage: string | null;
  created_at: string; started_at: string | null; finished_at: string | null; duration_s: number | null; peak_memory_mb: number | null;
  metrics: any; stage_timings: Record<string, number> | null; warnings: string[] | null; package_version: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

/** The page was served by something other than ThematicML's server: a website-only host (Vercel), Live Server, a saved file, or `npm run dev` with the server stopped. */
export const NO_SERVER = "Not connected to the ThematicML server, so nothing can be saved here. On your computer run .\\start.ps1 and open http://127.0.0.1:8000.";
// ThematicML answers every request, errors included, in JSON; anything else came from another server
const fromThematicML = (res: Response) => (res.headers.get("content-type") ?? "").includes("application/json");

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  let res: Response;
  try { res = await send(url, init); } catch (e) { throw unreachable(e); }
  if (!fromThematicML(res)) throw new ApiError(res.status, NO_SERVER);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : j.detail.map((d: any) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
    } catch { /* not JSON */ }
    throw new ApiError(res.status, msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T = any>(url: string) => request<T>("GET", url),
  post: <T = any>(url: string, body?: unknown) => request<T>("POST", url, body ?? {}),
  patch: <T = any>(url: string, body: unknown) => request<T>("PATCH", url, body),
  put: <T = any>(url: string, body: unknown) => request<T>("PUT", url, body),
  del: <T = any>(url: string) => request<T>("DELETE", url),
};

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

/** Trigger a browser download for a GET endpoint that returns a file. */
export async function download(url: string): Promise<void> {
  let res: Response;
  try { res = await send(url); } catch (e) { throw unreachable(e); }
  if (!res.ok) {
    if (!fromThematicML(res)) throw new ApiError(res.status, NO_SERVER);
    let msg = res.statusText;
    try { msg = (await res.json()).detail ?? msg; } catch { /* ignore */ }
    throw new ApiError(res.status, msg);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") ?? "";
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  const name = m ? decodeURIComponent(m[1]) : url.split("/").pop() ?? "download";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export const fmt = {
  n: (v: number | null | undefined, digits = 0) =>
    v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 }),
  pct: (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? "—" : `${v.toFixed(digits)}%`),
  bytes: (b: number | null | undefined) => {
    if (b === null || b === undefined) return "—";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0, x = b;
    while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
    return `${x.toFixed(i ? 1 : 0)} ${u[i]}`;
  },
  date: (s: string | null | undefined) => (s ? new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z").toLocaleString() : "—"),
  day: (s: string | null | undefined) => (s ? s.slice(0, 10) : "—"),
  sec: (s: number | null | undefined) => (s === null || s === undefined ? "—" : s < 60 ? `${s.toFixed(1)} s` : `${(s / 60).toFixed(1)} min`),
  compact: (v: number | null | undefined) =>
    v === null || v === undefined ? "—" : Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v),
};
