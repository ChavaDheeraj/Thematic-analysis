// The page side of the browser edition: starts the engine worker and answers fetch-style /api requests
// through it. One tab at a time owns the engine, because two copies would overwrite each other's saves.

export interface EngineStatus {
  phase: "idle" | "loading" | "ready" | "failed" | "blocked";
  message: string;
  progress: number;
}

type Pending = { resolve: (r: Response) => void; reject: (e: Error) => void };

let status: EngineStatus = { phase: "idle", message: "", progress: 0 };
const listeners = new Set<(s: EngineStatus) => void>();
const pending = new Map<number, Pending>();
let seq = 0;
let started: Promise<Worker> | null = null;

function setStatus(s: EngineStatus) {
  status = s;
  listeners.forEach((l) => l(s));
}

export function engineStatus(): EngineStatus { return status; }

export function onEngineStatus(listener: (s: EngineStatus) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function spawn(): Worker {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev: MessageEvent) => {
    const m = ev.data;
    if (m.type === "status") return setStatus(m.status);
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.type === "error") return p.reject(new Error(m.message));
    const empty = m.status === 204 || m.status === 304;
    p.resolve(new Response(empty ? null : m.body, { status: m.status, headers: m.headers }));
  };
  worker.onerror = (e) => setStatus({ phase: "failed", message: e.message || "The analysis engine stopped.", progress: 0 });
  return worker;
}

function start(): Promise<Worker> {
  if (started) return started;
  setStatus({ phase: "loading", message: "Starting the analysis engine", progress: 0.01 });
  // ask the browser not to clear this site's storage when space runs low
  navigator.storage?.persist?.().catch(() => undefined);
  const locks = (navigator as any).locks; // eslint-disable-line @typescript-eslint/no-explicit-any
  started = new Promise<Worker>((resolve) => {
    if (!locks?.request) return resolve(spawn());
    locks.request("thematicml-engine", { ifAvailable: true }, (lock: unknown) => {
      if (!lock) {
        setStatus({ phase: "blocked", message: "ThematicML is already open in another tab or window of this browser.", progress: 0 });
        return undefined;
      }
      resolve(spawn());
      return new Promise(() => undefined); // hold the lock while this tab is open
    });
  });
  return started;
}

/** `fetch` for /api paths, answered by the in-browser engine. */
export async function engineFetch(input: string, init?: RequestInit): Promise<Response> {
  const worker = await start();
  const req = new Request(new URL(input, location.href), init);
  const url = new URL(req.url);
  const body = req.method === "GET" || req.method === "HEAD" ? null : await req.arrayBuffer();
  const id = ++seq;
  return new Promise<Response>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "request", id, method: req.method, url: url.pathname + url.search, headers: [...req.headers.entries()], body },
      body ? [body] : []);
  });
}

/** Start loading the engine without waiting for a request. */
export function warmUp(): void { start(); }
