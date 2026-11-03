// The browser edition's engine: ThematicML's Python code running in Pyodide inside this worker.
// The page sends every /api request here (engine/client.ts) instead of to a server; projects and uploads
// are kept in this browser's IndexedDB, mounted at /data.
/* eslint-disable @typescript-eslint/no-explicit-any */
import type { EngineStatus } from "./client";

interface Manifest { pyodide: string; packages: string[]; later: string[]; wheels: string[]; python: string }

const ctx = self as any;
const t0 = performance.now();
const report = (s: EngineStatus) => {
  console.info(`[engine ${((performance.now() - t0) / 1000).toFixed(1)}s] ${s.message}`); // start-up timings, for diagnosing slow devices
  ctx.postMessage({ type: "status", status: s });
};

let py: any;
let handle: (method: string, url: string, headers: string[][], body: Uint8Array) => Promise<any>;

// IndexedDB saves run one at a time; a save asked for while one runs starts once it finishes
let saving: Promise<void> = Promise.resolve();
let queued: Promise<void> | null = null;
function persist(): Promise<void> {
  if (queued) return queued;
  queued = saving.then(() => {
    queued = null;
    return new Promise<void>((resolve) => py.FS.syncfs(false, (err: unknown) => {
      if (err) console.error("Saving to browser storage failed", err);
      resolve();
    }));
  });
  saving = queued;
  return queued;
}

async function boot(): Promise<void> {
  report({ phase: "loading", message: "Starting the analysis engine", progress: 0.02 });
  const manifest: Manifest = await (await fetch("/engine/manifest.json", { cache: "no-cache" })).json();
  const base = `https://cdn.jsdelivr.net/pyodide/v${manifest.pyodide}/full/`;
  const { loadPyodide } = await import(/* @vite-ignore */ `${base}pyodide.mjs`);
  py = await loadPyodide({ indexURL: base });
  report({ phase: "loading", message: "Loading the scientific libraries (the first visit downloads them once)", progress: 0.15 });
  let loaded = 0;
  await py.loadPackage(manifest.packages, {
    messageCallback: (m: string) => {
      if (m.startsWith("Loaded")) report({ phase: "loading", message: "Loading the scientific libraries (the first visit downloads them once)", progress: Math.min(0.55, 0.15 + (++loaded) * 0.012) });
    },
    errorCallback: (m: string) => console.warn(m),
  });
  report({ phase: "loading", message: "Adding the document readers", progress: 0.58 });
  const site = py.runPython("import site; site.getsitepackages()[0]");
  for (const w of manifest.wheels) py.unpackArchive(await (await fetch(`/engine/wheels/${w}`)).arrayBuffer(), "zip", { extractDir: site });
  const src = await (await fetch(`/engine/${manifest.python}`)).json();
  for (const [path, text] of Object.entries(src.files as Record<string, string>)) {
    const file = `/app/python/${path}`;
    py.FS.mkdirTree(file.slice(0, file.lastIndexOf("/")));
    py.FS.writeFile(file, text);
  }
  report({ phase: "loading", message: "Opening your projects", progress: 0.62 });
  py.FS.mkdirTree("/data");
  py.FS.mount(py.FS.filesystems.IDBFS, {}, "/data");
  await new Promise<void>((resolve, reject) => py.FS.syncfs(true, (err: unknown) => (err ? reject(err) : resolve())));
  report({ phase: "loading", message: "Preparing the analysis methods", progress: 0.66 });
  py.runPython("import sys; sys.path.insert(0, '/app/python')");
  const browser = py.pyimport("thematicml_api.browser");
  browser.start("/data", persist);
  handle = browser.handle;
  report({ phase: "ready", message: "Ready", progress: 1 });
  // packages only analyses need load after the app is usable; analyses wait for them
  browser.load_later(py.loadPackage(manifest.later ?? []).catch((e: unknown) => console.warn("Background packages failed to load", e)));
}

const ready = boot().catch((e: any) => {
  console.error(e);
  report({ phase: "failed", message: String(e?.message ?? e).split("\n")[0] || "The analysis engine could not start.", progress: 0 });
  throw e;
});

ctx.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  if (m?.type !== "request") return;
  try {
    await ready;
    const result = await handle(m.method, m.url, m.headers, new Uint8Array(m.body ?? new ArrayBuffer(0)));
    const [status, headers, body] = result.toJs();
    result.destroy();
    const bytes: Uint8Array = body instanceof Uint8Array ? body : new Uint8Array(body);
    ctx.postMessage({ type: "response", id: m.id, status, headers, body: bytes.buffer }, [bytes.buffer]);
    persist();
  } catch (e: any) {
    ctx.postMessage({ type: "error", id: m.id, message: String(e?.message ?? e).split("\n")[0] });
  }
};
