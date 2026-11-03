// Pack the browser edition's engine into public/engine: ThematicML's Python code, and the pure-Python
// wheels it needs that Pyodide does not ship. Runs before `vite build --mode browser` (npm run build:browser).
// The browser loads Pyodide and its scientific packages from the jsDelivr CDN and everything here from the site.
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");
const out = join(here, "..", "public", "engine");

export const PYODIDE = "314.0.7";
// Pyodide packages loaded at start-up (their own dependencies come with them)
const PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "joblib", "pydantic", "lxml",
  "beautifulsoup4", "fastapi", "pillow", "fonttools", "python-dateutil", "pytz", "threadpoolctl"];
// loaded in the background once the app is usable; analyses wait for them (statsmodels: seasonal trends only)
const LATER = ["statsmodels"];
// pure-Python wheels from PyPI, pinned to the versions the server edition is tested with
const WHEELS = {
  "openpyxl": "3.1.5", "et-xmlfile": "2.0.0", "python-docx": "1.2.0", "pypdf": "6.19.0", "fpdf2": "2.8.9",
  "defusedxml": "0.7.1", "vaderSentiment": "3.3.2", "python-multipart": "0.0.22",
};
const SOURCES = [["src", "thematicml"], ["backend", "thematicml_api"]];

async function* pyFiles(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory() && e.name !== "__pycache__") yield* pyFiles(p);
    else if (e.isFile() && e.name.endsWith(".py")) yield p;
  }
}

async function wheel(name, version) {
  const meta = await (await fetch(`https://pypi.org/pypi/${name}/${version}/json`)).json();
  const w = meta.urls.find((u) => u.packagetype === "bdist_wheel" && u.filename.endsWith("-none-any.whl"));
  if (!w) throw new Error(`${name} ${version} has no pure-Python wheel on PyPI`);
  const dest = join(out, "wheels", w.filename);
  if (existsSync(dest) && createHash("sha256").update(await readFile(dest)).digest("hex") === w.digests.sha256) return w.filename;
  const bytes = Buffer.from(await (await fetch(w.url)).arrayBuffer());
  if (createHash("sha256").update(bytes).digest("hex") !== w.digests.sha256) throw new Error(`checksum mismatch for ${w.filename}`);
  await writeFile(dest, bytes);
  return w.filename;
}

const files = {};
for (const [base, pkg] of SOURCES) {
  for await (const p of pyFiles(join(repo, base, pkg))) files[relative(join(repo, base), p).split(sep).join("/")] = await readFile(p, "utf8");
}
const python = JSON.stringify({ files });
const tag = createHash("sha256").update(python).digest("hex").slice(0, 12);

await mkdir(join(out, "wheels"), { recursive: true });
const wheels = [];
for (const [name, version] of Object.entries(WHEELS)) wheels.push(await wheel(name, version));
for (const f of await readdir(join(out, "wheels"))) if (!wheels.includes(f)) await rm(join(out, "wheels", f));
for (const f of await readdir(out)) if (/^python-.*\.json$/.test(f)) await rm(join(out, f));
// the file name carries a hash of the code, so a new version is never served from a stale cache
await writeFile(join(out, `python-${tag}.json`), python);
await writeFile(join(out, "manifest.json"), JSON.stringify({ pyodide: PYODIDE, packages: PACKAGES, later: LATER, wheels, python: `python-${tag}.json` }, null, 1));
console.log(`engine: ${Object.keys(files).length} Python files (python-${tag}.json), ${wheels.length} wheels, Pyodide ${PYODIDE}`);
// open download sockets otherwise keep Node alive at exit (and trip a libuv assertion on Windows)
process.exit(0);
