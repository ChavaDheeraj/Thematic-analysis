import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { NO_SERVER, api, type Dataset, type Project, type Run } from "./api";

function load(key: string): string | null {
  try { return localStorage.getItem(`tml.${key}`); } catch { return null; }
}
function save(key: string, value: string | null) {
  try { if (value === null) localStorage.removeItem(`tml.${key}`); else localStorage.setItem(`tml.${key}`, value); } catch { /* storage unavailable */ }
}

/** The scope that analyses every file of the project together, in place of one file. */
export const ALL_FILES = "__all";

/** Analyses in a scope (newest first): of exactly this one file, or, for ALL_FILES, of several files together. */
export function fileRuns(runs: Run[], datasetId: string | null): Run[] {
  if (datasetId === ALL_FILES) return runs.filter((r) => r.dataset_ids.length > 1);
  return runs.filter((r) => r.dataset_ids.length === 1 && r.dataset_ids[0] === datasetId);
}
const latestCompleted = (runs: Run[]) => runs.find((r) => r.status === "completed") ?? null;
/** The analysis to open for a file: its own latest one, else the latest that included it (a combined analysis). */
const defaultRun = (runs: Run[], datasetId: string | null) =>
  latestCompleted(fileRuns(runs, datasetId)) ??
  (datasetId === ALL_FILES ? null : latestCompleted(runs.filter((r) => datasetId !== null && r.dataset_ids.includes(datasetId))));

export interface Toast { id: number; kind: "info" | "error" | "success"; text: string }

interface AppState {
  system: any;
  /** The page cannot reach ThematicML's server, so nothing can be loaded or saved. */
  serverDown: boolean;
  projects: Project[];
  project: Project | null;
  setProjectId: (id: string | null) => void;
  datasets: Dataset[];
  /** What every page shows: ALL_FILES (one analysis of the project's files together) or one file's id. */
  datasetId: string | null;
  /** The selected file; null when all files are analysed together. */
  dataset: Dataset | null;
  /** True when the pages show the analysis of all the project's files together. */
  allFiles: boolean;
  setDatasetId: (id: string | null) => void;
  /** Analyses in the selected scope: of the selected file, or of all files together. */
  runs: Run[];
  /** Every analysis in the project, including ones that combined several files before single-file mode. */
  allRuns: Run[];
  run: Run | null;
  runId: string | null;
  setRunId: (id: string | null) => void;
  /** Show this analysis, and select its file, or all files for an analysis of several. */
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
    if (r) {
      const scope = r.dataset_ids.length === 1 ? r.dataset_ids[0] : ALL_FILES;
      setDatasetIdRaw(scope);
      save("dataset", scope);
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

  // keep the selection valid. A project with several files opens on the analysis of all of them together;
  // a single file can still be chosen in the Database page's file selector.
  const openedProject = useRef<string | null>(null);
  useEffect(() => {
    if (!datasets.length || !projectId) return;
    const opening = openedProject.current !== projectId;
    openedProject.current = projectId;
    const valid = datasetId === ALL_FILES ? datasets.length > 1 : !!datasets.find((d) => d.id === datasetId);
    if (valid && !(opening && datasets.length > 1 && datasetId !== ALL_FILES)) return;
    const next = datasets.length > 1 ? ALL_FILES : datasets[0].id;
    setDatasetIdRaw(next);
    save("dataset", next);
    setRunId(defaultRun(runs, next)?.id ?? null);
  }, [datasets, datasetId, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

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
    system, serverDown, projects, project, setProjectId, datasets, datasetId, dataset, allFiles: datasetId === ALL_FILES, setDatasetId, runs: scopedRuns, allRuns: runs, run, runId, setRunId, openRun,
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
