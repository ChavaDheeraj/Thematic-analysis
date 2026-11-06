import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { IN_BROWSER, api, download } from "./api";
import { engineStatus, onEngineStatus, type EngineStatus } from "./engine/client";
import { Icon, ICONS } from "./components/icons";
import { Badge, Button, Field, Modal, cx, inputBase, inputCls } from "./components/ui";
import DashboardPage from "./pages/Dashboard";
import DatabasePage from "./pages/Database";
import GraphsPage from "./pages/Graphs";
import UploadPage from "./pages/Upload";
import { useApp } from "./state";

const NAV = [
  { to: "/upload", label: "Upload & analyse", sub: "One file or many together", icon: ICONS.upload },
  { to: "/dashboard", label: "Dashboard", sub: "Visual results", icon: ICONS.dashboard },
  { to: "/graphs", label: "Graphs & Patterns", sub: "Relationships", icon: ICONS.graph },
  { to: "/database", label: "Database", sub: "Details, files & export", icon: ICONS.database },
];

export default function App() {
  const { toasts, serverDown } = useApp();
  const [newOpen, setNewOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const nav = useNavigate();
  return (
    <div className="flex h-full min-h-0">
      <Sidebar onNewProject={() => setNewOpen(true)} onSettings={() => setSettingsOpen(true)} />
      <div className="flex min-w-0 flex-1 flex-col">
        <MobileHeader onNewProject={() => setNewOpen(true)} onSettings={() => setSettingsOpen(true)} />
        <main className="min-h-0 flex-1 overflow-y-auto px-5 py-6 md:px-8">
          {serverDown && <ServerDown />}
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/graphs" element={<GraphsPage />} />
            <Route path="/database" element={<DatabasePage />} />
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Routes>
        </main>
      </div>
      {IN_BROWSER && <EngineGate />}
      <NewProject open={newOpen} onClose={() => setNewOpen(false)} onCreated={() => nav("/upload")} />
      <Settings open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      {/* Toast notifications */}
      <div className="fixed bottom-5 right-5 z-[60] flex w-96 max-w-[calc(100vw-2.5rem)] flex-col gap-2.5">
        {toasts.map((t) => (
          <div key={t.id} role="status"
            className={cx(
              "fade-in flex items-start gap-3 rounded-xl border px-4 py-3 text-sm shadow-lg",
              t.kind === "error"
                ? "border-bad/30 bg-surface text-ink"
                : t.kind === "success"
                ? "border-good/30 bg-surface text-ink"
                : "border-line bg-surface text-ink"
            )}>
            <span className={cx("mt-px shrink-0 rounded-full p-0.5",
              t.kind === "error" ? "text-bad" : t.kind === "success" ? "text-good" : "text-muted")}>
              <Icon path={t.kind === "error" ? ICONS.alert : t.kind === "success" ? ICONS.check : ICONS.info} size={14} />
            </span>
            <span className="flex-1 leading-relaxed">{t.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** The browser edition's start-up screen: covers the app until the engine in this tab is ready. */
function EngineGate() {
  const [s, setS] = useState<EngineStatus>(engineStatus());
  useEffect(() => onEngineStatus(setS), []);
  if (s.phase === "ready") return null;
  const problem = s.phase === "failed" || s.phase === "blocked";
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-page/95 px-4 backdrop-blur-sm" role={problem ? "alert" : "status"} aria-live="polite">
      <div className="w-full max-w-md rounded-2xl border border-line bg-surface p-7" style={{ boxShadow: "var(--shadow-xl)" }}>
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-xs font-bold text-white">TM</div>
          <div>
            <div className="text-[15px] font-bold tracking-tight text-ink">ThematicML</div>
            <div className="text-[11px] text-muted">Runs entirely in this browser</div>
          </div>
        </div>
        {s.phase === "blocked" ? (
          <>
            <h1 className="mt-6 text-lg font-semibold text-ink">Already open in another tab</h1>
            <p className="mt-2 text-sm leading-relaxed text-ink2">{s.message} Only one tab can use your projects at a time, so that nothing is overwritten. Close the other tab, then reload this one.</p>
            <div className="mt-5"><Button kind="primary" onClick={() => location.reload()}>Reload</Button></div>
          </>
        ) : s.phase === "failed" ? (
          <>
            <h1 className="mt-6 text-lg font-semibold text-ink">The analysis engine could not start</h1>
            <p className="mt-2 text-sm leading-relaxed text-ink2">It needs a recent Chrome, Edge, Firefox or Safari, and an internet connection to download its libraries. Your projects are kept in this browser and are not affected.</p>
            <p className="mt-2 break-words rounded-md bg-raised px-3 py-2 font-mono text-[11px] text-muted">{s.message}</p>
            <div className="mt-5"><Button kind="primary" onClick={() => location.reload()}>Try again</Button></div>
          </>
        ) : (
          <>
            <h1 className="mt-6 text-lg font-semibold text-ink">Setting up ThematicML</h1>
            <p className="mt-1 text-sm text-ink2">{s.message || "Starting the analysis engine"}…</p>
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-line">
              <div className="progress-shimmer h-1.5 rounded-full transition-all duration-500" style={{ width: `${Math.max(4, s.progress * 100)}%` }} />
            </div>
            <p className="mt-4 text-xs leading-relaxed text-muted">
              Analyses run on this device and your files never leave it. The first visit downloads about 60 MB of analysis libraries;
              later visits reuse them. Your projects are saved in this browser.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function ServerDown() {
  return (
    <div role="alert" className="mx-auto mb-6 flex max-w-7xl items-start gap-3 rounded-xl border border-bad/30 bg-bad/6 px-5 py-4 text-sm">
      <span className="mt-0.5 shrink-0 text-bad"><Icon path={ICONS.alert} size={16} /></span>
      <div>
        <div className="font-semibold text-ink">Not connected to the ThematicML server</div>
        <p className="mt-1 leading-relaxed text-ink2">
          Projects, uploads and analyses run on ThematicML's own Python server, and there is none behind this page, so nothing can be created or loaded here.
          That is what happens on a website-only host such as Vercel, with VS Code Live Server, or with a page opened from a file.
          On your computer, run <code className="rounded bg-raised px-1 font-mono">.\start.ps1</code> in the project folder,
          then open <a className="font-medium text-accent hover:underline" href="http://127.0.0.1:8000">http://127.0.0.1:8000</a>.
        </p>
      </div>
    </div>
  );
}

function ProjectPicker({ onNewProject, className }: { onNewProject: () => void; className?: string }) {
  const { projects, project, setProjectId } = useApp();
  return (
    <div className={cx("flex gap-1.5", className)}>
      <select className={cx(inputBase, "min-w-0 flex-1 py-1.5")} value={project?.id ?? ""} onChange={(e) => setProjectId(e.target.value || null)} aria-label="Project">
        {!projects.length && <option value="">No projects</option>}
        {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
      </select>
      <button onClick={onNewProject} title="New project" aria-label="New project"
        className="flex h-[34px] w-[34px] shrink-0 items-center justify-center rounded-lg border border-line bg-surface text-ink2 transition-colors hover:bg-raised hover:text-ink"
        style={{ boxShadow: "var(--shadow-sm)" }}>
        <Icon path={ICONS.plus} size={14} />
      </button>
    </div>
  );
}

function Sidebar({ onNewProject, onSettings }: { onNewProject: () => void; onSettings: () => void }) {
  const { activeRuns, system } = useApp();
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-line bg-surface md:flex">
      <div className="px-5 pb-5 pt-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-xs font-bold text-white" style={{ boxShadow: "0 2px 8px rgb(var(--accent) / 0.4)" }}>
            TM
          </div>
          <div>
            <div className="text-[15px] font-bold tracking-tight text-ink">ThematicML</div>
            <div className="mt-0.5 text-[11px] leading-none text-muted">Research workspace</div>
          </div>
        </div>
        <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-good/10 px-2.5 py-1 text-[10px] font-semibold text-good" title={IN_BROWSER ? "All processing runs in this browser, on this device. No LLM or remote AI service is used." : "All processing runs on this machine. No LLM or remote AI service is used."}>
          <span className="h-1.5 w-1.5 rounded-full bg-good" />
          Local · No LLM
        </span>
      </div>

      <div className="px-4 pb-4">
        <div className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-widest text-muted">Project</div>
        <ProjectPicker onNewProject={onNewProject} />
      </div>

      <nav className="flex flex-col gap-0.5 border-t border-line px-3 pt-3">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to}
            className={({ isActive }) => cx(
              "group flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors",
              isActive ? "bg-accent/10 text-accent" : "text-ink2 hover:bg-raised hover:text-ink"
            )}>
            {({ isActive }) => (
              <>
                <span className={cx("shrink-0 transition-colors", isActive ? "text-accent" : "text-muted group-hover:text-ink2")}>
                  <Icon path={n.icon} size={16} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium">{n.label}</div>
                  <div className="mt-0.5 text-[11px] leading-none text-muted">{n.sub}</div>
                </div>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto border-t border-line px-4 py-4">
        {activeRuns.length > 0 && (
          <div className="mb-3 space-y-2">
            <div className="text-[10px] font-semibold uppercase tracking-widest text-muted">Running</div>
            {activeRuns.map((r) => (
              <div key={r.id} className="rounded-lg bg-raised p-2.5">
                <div className="mb-1.5 flex justify-between text-xs">
                  <span className="truncate pr-2 font-medium">{r.label}</span>
                  <span className="font-semibold tabular-nums text-accent">{Math.round(r.progress * 100)}%</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-line">
                  <div className="progress-shimmer h-1 rounded-full transition-all duration-300" style={{ width: `${Math.max(3, r.progress * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
        <button onClick={onSettings}
          className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium text-ink2 transition-colors hover:bg-raised hover:text-ink">
          <Icon path={ICONS.settings} size={15} />Settings
        </button>
        <div className="mt-2 px-3 text-[11px] text-muted">v{system?.version ?? "—"} · {IN_BROWSER ? "runs in this browser" : "runs on this machine"}</div>
      </div>
    </aside>
  );
}

function MobileHeader({ onNewProject, onSettings }: { onNewProject: () => void; onSettings: () => void }) {
  return (
    <header className="border-b border-line bg-surface px-4 py-3 md:hidden">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent text-[10px] font-bold text-white">TM</div>
        <ProjectPicker onNewProject={onNewProject} className="min-w-0 flex-1" />
        <button onClick={onSettings} title="Settings" aria-label="Settings" className="rounded-lg p-2 text-ink2 hover:bg-raised"><Icon path={ICONS.settings} size={16} /></button>
      </div>
      <nav className="mt-2 flex gap-1 overflow-x-auto">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} className={({ isActive }) => cx("flex items-center gap-1.5 whitespace-nowrap rounded-lg px-2.5 py-1.5 text-xs font-medium", isActive ? "bg-accent/10 text-accent" : "text-ink2 hover:bg-raised")}>
            <Icon path={n.icon} size={13} />{n.label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}

function NewProject({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const { refreshProjects, setProjectId, toast } = useApp();
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const submit = async () => {
    try {
      const p = await api.post("/api/projects", { name, description: desc });
      await refreshProjects();
      setProjectId(p.id);
      setName(""); setDesc("");
      onClose();
      onCreated();
      toast(`Project "${p.name}" created`, "success");
    } catch (e: any) { toast(e.message, "error"); } // eslint-disable-line @typescript-eslint/no-explicit-any
  };
  return (
    <Modal open={open} onClose={onClose} title="New research project">
      <div className="space-y-4">
        <Field label="Project name"><input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} autoFocus placeholder="e.g. Solo Female Travel Study" /></Field>
        <Field label="Research questions / description" hint="Included in exported reports.">
          <textarea className={cx(inputCls, "h-28 resize-none")} value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Describe the research goal, dataset source, or key questions..." />
        </Field>
        <div className="flex gap-2 pt-1">
          <Button kind="primary" disabled={!name.trim()} onClick={submit}>Create project</Button>
          <Button kind="ghost" onClick={onClose}>Cancel</Button>
        </div>
      </div>
    </Modal>
  );
}

function Settings({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { system, project, refreshProjects, setProjectId, toast, theme, setTheme } = useApp();
  const [settings, setSettings] = useState<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [pname, setPname] = useState("");
  const [pdesc, setPdesc] = useState("");
  useEffect(() => {
    if (!open) return;
    api.get("/api/settings").then(setSettings).catch((e) => toast(e.message, "error"));
    setPname(project?.name ?? "");
    setPdesc(project?.description ?? "");
  }, [open, project, toast]);
  const save = async () => {
    try {
      setSettings(await api.put("/api/settings", { max_upload_mb: Number(settings.max_upload_mb), max_hierarchical_segments: Number(settings.max_hierarchical_segments) }));
      toast("Settings saved", "success");
    } catch (e: any) { toast(e.message, "error"); } // eslint-disable-line @typescript-eslint/no-explicit-any
  };
  const saveProject = async () => {
    try { await api.patch(`/api/projects/${project!.id}`, { name: pname, description: pdesc }); await refreshProjects(); toast("Project updated", "success"); }
    catch (e: any) { toast(e.message, "error"); } // eslint-disable-line @typescript-eslint/no-explicit-any
  };
  const del = async () => {
    if (!project || !confirm(`Permanently delete project "${project.name}", all uploads, analyses and exports? This cannot be undone.`)) return;
    try { await api.del(`/api/projects/${project.id}`); setProjectId(null); await refreshProjects(); onClose(); toast("Project deleted"); }
    catch (e: any) { toast(e.message, "error"); } // eslint-disable-line @typescript-eslint/no-explicit-any
  };
  const opt = system?.optional ?? {};
  return (
    <Modal open={open} onClose={onClose} title="Settings">
      <div className="space-y-6 text-sm">
        <section>
          <h3 className="mb-3 text-[13px] font-semibold text-ink2 uppercase tracking-wide">Appearance</h3>
          <div className="flex gap-2">
            {["system", "light", "dark"].map((t) => (
              <button key={t}
                onClick={() => setTheme(t)}
                className={cx(
                  "rounded-lg border px-3 py-1.5 text-xs font-medium capitalize transition-all",
                  theme === t
                    ? "border-accent bg-accent/10 text-accent"
                    : "border-line bg-surface text-ink2 hover:bg-raised"
                )}>
                {t}
              </button>
            ))}
          </div>
        </section>
        <div className="border-t border-line" />
        <section>
          <h3 className="mb-3 text-[13px] font-semibold text-ink2 uppercase tracking-wide">Local storage</h3>
          {system?.edition === "browser" ? (
            <p className="text-xs leading-relaxed text-ink2">
              Your projects, uploads and analyses are saved in this browser on this device. Other devices and browsers have their own, separate projects.
              Clearing this site's data deletes them, so keep a copy with <span className="font-medium">Download backup (ZIP)</span> below.
            </p>
          ) : (<>
          <dl className="grid grid-cols-[8rem_1fr] gap-y-2 text-xs">
            <dt className="text-muted pt-px">Workspace</dt><dd className="break-all font-mono bg-raised rounded px-2 py-1">{system?.workspace}</dd>
            <dt className="text-muted pt-px">Database</dt><dd className="break-all font-mono bg-raised rounded px-2 py-1">{system?.database}</dd>
          </dl>
          <p className="mt-2 text-xs text-muted">Set the <code className="rounded bg-raised px-1 py-px font-mono">THEMATICML_HOME</code> environment variable before starting the server to use a different folder.</p>
          </>)}
        </section>
        {settings && (
          <>
            <div className="border-t border-line" />
            <section>
              <h3 className="mb-3 text-[13px] font-semibold text-ink2 uppercase tracking-wide">Resource limits</h3>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Max upload size (MB)"><input type="number" min={1} className={inputCls} value={settings.max_upload_mb} onChange={(e) => setSettings({ ...settings, max_upload_mb: e.target.value })} /></Field>
                <Field label="Max segments for hierarchical clustering" hint="Agglomerative clustering is O(n²) in memory.">
                  <input type="number" min={100} className={inputCls} value={settings.max_hierarchical_segments} onChange={(e) => setSettings({ ...settings, max_hierarchical_segments: e.target.value })} />
                </Field>
              </div>
              <div className="mt-3"><Button kind="primary" small onClick={save}>Save limits</Button></div>
            </section>
          </>
        )}
        <div className="border-t border-line" />
        <section>
          <h3 className="mb-3 text-[13px] font-semibold text-ink2 uppercase tracking-wide">Installed modules</h3>
          <ul className="space-y-2 text-xs">
            {[
              { key: "leiden", label: "Leiden community detection (python-igraph, leidenalg)", val: opt.leiden },
              { key: "vader", label: "Lexicon sentiment (VADER, rule-based)", val: opt.sentiment_vader },
              { key: "statsmodels", label: "Seasonal decomposition & ARIMA (statsmodels)", val: opt.statsmodels },
              { key: "pdf", label: "PDF reports (fpdf2)", val: opt.pdf_reports },
              { key: "gnn", label: `Graph neural networks (GraphSAGE) — ${opt.gnn?.backend === "pytorch" ? "PyTorch Geometric" : "built in (NumPy, CPU)"}. ${opt.gnn?.reason ?? ""}`, val: opt.gnn?.available },
            ].map(({ key, label, val }) => (
              <li key={key} className="flex items-start gap-2">
                <Badge status={val ? "completed" : "candidate"}>{val ? "installed" : "not installed"}</Badge>
                <span className="text-muted pt-px">{label}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-muted">ThematicML never calls an LLM or remote AI service. There are no remote integrations to configure.</p>
        </section>
        {project && (
          <>
            <div className="border-t border-line" />
            <section>
              <h3 className="mb-3 text-[13px] font-semibold text-ink2 uppercase tracking-wide">Project</h3>
              <div className="space-y-3">
                <Field label="Name"><input className={inputCls} value={pname} onChange={(e) => setPname(e.target.value)} /></Field>
                <Field label="Research questions / description">
                  <textarea className={cx(inputCls, "h-20 resize-none")} value={pdesc} onChange={(e) => setPdesc(e.target.value)} />
                </Field>
                <div className="flex flex-wrap gap-2 pt-1">
                  <Button small kind="primary" onClick={saveProject}>Save project</Button>
                  <Button small onClick={() => download(`/api/projects/${project.id}/backup`).catch((e) => toast(e.message, "error"))}>Download backup (ZIP)</Button>
                  <Button small kind="danger" onClick={del}>Delete project</Button>
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </Modal>
  );
}
