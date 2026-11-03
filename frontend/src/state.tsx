import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { NO_SERVER, api, type Dataset, type Project, type Run } from "./api";

function load(key: string): string | null {
  try { return localStorage.getItem(`tml.${key}`); } catch { return null; }
}
function save(key: string, value: string | null) {
  try { if (value === null) localStorage.removeItem(`tml.${key}`); else localStorage.setItem(`tml.${key}`, value); } catch { /* storage unavailable */ }
}

/** Analyses of exactly this one file (runs are listed newest first). */
export function fileRuns(runs: Run[], datasetId: string | null): Run[] {
  return runs.filter((r) => r.dataset_ids.length === 1 && r.dataset_ids[0] === datasetId);
}
const latestCompleted = (runs: Run[]) => runs.find((r) => r.status === "completed") ?? null;
/** The analysis to open for a file: its own latest one, else the latest that included it (a combined analysis). */
const defaultRun = (runs: Run[], datasetId: string | null) =>
  latestCompleted(fileRuns(runs, datasetId)) ?? latestCompleted(runs.filter((r) => datasetId !== null && r.dataset_ids.includes(datasetId)));

export interface Toast { id: number; kind: "info" | "error" | "success"; text: string }

interface AppState {
  system: any;
  /** The page cannot reach ThematicML's server, so nothing can be loaded or saved. */
  serverDown: boolean;
  projects: Project[];
  project: Project | null;
  setProjectId: (id: string | null) => void;
  datasets: Dataset[];
  /** The one file being analysed; every page works on a single file at a time. */
  datasetId: string | null;
  dataset: Dataset | null;
  setDatasetId: (id: string | null) => void;
  /** Analyses of the selected file only. */
  runs: Run[];
  /** Every analysis in the project, including ones that combined several files before single-file mode. */
  allRuns: Run[];
  run: Run | null;
  runId: string | null;
  setRunId: (id: string | null) => void;
  /** Show this analysis; a single-file analysis also selects its file. */
  openRun: (id: string) => void;
  dateFrom: string;
  dateTo: string;
  setDateRange: (from: string, to: string) => void;
  refreshProjects: () => Promise<void>;
  refreshData: () => Promise<void>;
  activeRuns: Run[];
  toast: (text: string, kind?: Toast["kind"]) => void;
  toasts: Toast[];
  theme: string;
  setTheme: (t: string) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<any>(null);
  const [serverDown, setServerDown] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectIdRaw] = useState<string | null>(load("project"));
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetIdRaw] = useState<string | null>(load("dataset") === "all" ? null : load("dataset"));
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunIdRaw] = useState<string | null>(load("run"));
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [theme, setThemeRaw] = useState<string>(load("theme") ?? "system");
  const toastId = useRef(0);

  const toast = useCallback((text: string, kind: Toast["kind"] = "info") => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, kind, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 8000 : 4000);
  }, []);

  const setTheme = (t: string) => {
    setThemeRaw(t);
    save("theme", t);
  };
  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  const setProjectId = (id: string | null) => {
    setProjectIdRaw(id);
    save("project", id);
    setDatasetId(null);
  };
  const setDatasetId = (id: string | null) => {
    setDatasetIdRaw(id);
    save("dataset", id);
    setRunId(defaultRun(runs, id)?.id ?? null);
  };
  const setRunId = (id: string | null) => {
    setRunIdRaw(id);
    save("run", id);
  };
  const openRun = (id: string) => {
    const r = runs.find((x) => x.id === id);
    if (r?.dataset_ids.length === 1) {
      setDatasetIdRaw(r.dataset_ids[0]);
      save("dataset", r.dataset_ids[0]);
    }
    setRunId(id);
  };

  const refreshProjects = useCallback(async () => {
    const ps = await api.get<Project[]>("/api/projects");
    setProjects(ps);
    if (ps.length && !ps.find((p) => p.id === projectId)) {
      setProjectIdRaw(ps[0].id);
      save("project", ps[0].id);
    }
    if (!ps.length) setProjectIdRaw(null);
  }, [projectId]);

  const refreshData = useCallback(async () => {
    if (!projectId) {
      setDatasets([]);
      setRuns([]);
      return;
    }
    const [ds, rs] = await Promise.all([
      api.get<Dataset[]>(`/api/projects/${projectId}/datasets`),
      api.get<Run[]>(`/api/projects/${projectId}/runs`),
    ]);
    setDatasets(ds);
    setRuns(rs);
  }, [projectId]);

  useEffect(() => {
    api.get("/api/system").then(setSystem).catch((e) => { setSystem(null); setServerDown(e.message === NO_SERVER); });
    // the banner explains a missing server; a toast would only repeat it
    refreshProjects().catch((e) => e.message !== NO_SERVER && toast(String(e.message ?? e), "error"));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    refreshData().catch((e) => e.message !== NO_SERVER && toast(String(e.message ?? e), "error"));
  }, [refreshData, toast]);

  // keep the selected file valid, defaulting to the newest upload
  useEffect(() => {
    if (!datasets.length || (datasetId && datasets.find((d) => d.id === datasetId))) return;
    setDatasetIdRaw(datasets[0].id);
    save("dataset", datasets[0].id);
  }, [datasets, datasetId]);

  // keep the selected run valid, defaulting to the latest completed analysis of the selected file
  const scopedRuns = useMemo(() => fileRuns(runs, datasetId), [runs, datasetId]);
  useEffect(() => {
    if (!runs.length) return;
    if (runId && runs.find((r) => r.id === runId)) return;
    setRunId(defaultRun(runs, datasetId)?.id ?? null);
  }, [runs, datasetId, runId]);

  // poll while any run is queued or running
  const activeRuns = runs.filter((r) => r.status === "queued" || r.status === "running");
  const prevActive = useRef<string[]>([]);
  useEffect(() => {
    if (!activeRuns.length) return;
    const t = setInterval(() => refreshData().catch(() => undefined), 1200);
    return () => clearInterval(t);
  }, [activeRuns.length, refreshData]);
  useEffect(() => {
    const now = activeRuns.map((r) => r.id);
    for (const id of prevActive.current) {
      if (!now.includes(id)) {
        const r = runs.find((x) => x.id === id);
        if (r?.status === "completed") toast(`${r.label} completed`, "success");
        if (r?.status === "failed") toast(`${r.label} failed at ${r.failed_stage}: ${r.error}`, "error");
      }
    }
    prevActive.current = now;
  }, [runs]); // eslint-disable-line react-hooks/exhaustive-deps

  const project = projects.find((p) => p.id === projectId) ?? null;
  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const run = runs.find((r) => r.id === runId) ?? null;

  const value: AppState = {
    system, serverDown, projects, project, setProjectId, datasets, datasetId, dataset, setDatasetId, runs: scopedRuns, allRuns: runs, run, runId, setRunId, openRun,
    dateFrom, dateTo, setDateRange: (f, t) => { setDateFrom(f); setDateTo(t); },
    refreshProjects, refreshData, activeRuns, toast, toasts, theme, setTheme,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside provider");
  return v;
}

/** Fetch JSON for a key; refetches when the key changes. Returns [data, error, loading, reload]. */
export function useFetch<T>(url: string | null, deps: unknown[] = []): [T | null, string | null, boolean, () => void] {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!url) {
      setData(null);
      return;
    }
    let alive = true;
    setLoading(true);
    setErr(null);
    api.get<T>(url)
      .then((d) => alive && setData(d))
      .catch((e) => alive && (setErr(e.message ?? String(e)), setData(null)))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [url, tick, ...deps]); // eslint-disable-line react-hooks/exhaustive-deps
  return [data, err, loading, () => setTick((t) => t + 1)];
}
