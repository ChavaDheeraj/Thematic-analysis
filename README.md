# ThematicML

A local-first machine learning platform for qualitative research. ThematicML applies classical NLP, clustering,
topic models, graph algorithms and time-series statistics to find codes, patterns and candidate themes in
qualitative text, and keeps every result traceable to the exact source passage.

- **No LLMs or generative AI.** Nothing is sent to a remote AI service. Labels come from extracted terms and templates.
- **Local.** Data, models and results live in a SQLite database and a workspace folder on your machine.
- **Researcher-controlled.** Codes and themes start as *machine-generated candidates*; you review, edit, merge, split and approve them.
- **Reproducible.** Every run stores its full configuration, random seed, package version and stage timings.

## Install

Requires Python 3.10+ (tested on 3.14) and, to rebuild the web UI, Node.js 18+.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[all]"           # core + sentiment, temporal, Leiden, PDF reports, web API, tests
```

`pip install -e .` installs only the core package. Optional extras:

| Extra | Adds | Without it |
|---|---|---|
| `sentiment` | VADER lexicon sentiment (rule-based) | sentiment shows as unavailable |
| `temporal` | statsmodels: seasonal decomposition, ARIMA | those two steps are skipped; trends and change points still run |
| `leiden` | Leiden community detection | Louvain, greedy modularity and label propagation remain |
| `reports` | PDF export | other export formats remain |
| `api` | FastAPI web backend | Python API only |
| `gnn` | PyTorch + PyTorch Geometric: GraphSAGE link prediction / node classification (neural) | GNN sections show *not installed*; everything else works |

No GPU is needed. For the GNN on CPU (recommended; ~200 MB download), install PyTorch's CPU build first:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

Verified here: Python 3.14.2, torch 2.14.1+cpu, torch_geometric 2.8.1 on Windows 11. CUDA is used only if you choose `device: cuda` or `auto` and a CUDA build of torch is installed.

## Run the web application

```powershell
cd frontend; npm install; npm run build; cd ..    # once, or after changing the UI
.\start.ps1                                         # or: python -m uvicorn thematicml_api.main:app --port 8000
```

Open <http://127.0.0.1:8000>, the address the start script prints. Opening `index.html` with VS Code Live Server, from the file system or with `npm run dev` while the server is stopped shows the pages but cannot create projects or load anything: every `/api` request then fails with 404, and the app says it is not connected. To host the website on its own (Vercel, Netlify, GitHub Pages), build the browser edition instead, which runs the analysis in the page; see *Use from other devices* below. Stop the server with **Ctrl+C** in its terminal. Analyses still running when it stops are marked *interrupted* on the next start and can be retried from the failed stage.

By default the server listens on `127.0.0.1` only and has **no login**: anyone who can reach the port can read and change every project. To let other devices in, use `-Network` or hosting with a password (see *Use from other devices*).

### Verified setup

Checked on Windows 11 with Python 3.14.2, Node 24, pandas 3.0, scikit-learn 1.9, networkx 3.7, torch 2.14.1+cpu and torch_geometric 2.8.1, in a project-local `.venv` (the global Python is untouched). The pages were exercised in Microsoft Edge (light, dark, 1440 px and 390 px).

### Supported files

| Format | Becomes |
|---|---|
| CSV, XLSX (first sheet), JSON (list of objects, or an object holding one list), JSONL | one record per row; you map text, ID, timestamp, group and label columns |
| TXT, DOCX (paragraphs; tables are skipped with a warning) | one record per paragraph, with paragraph number |
| PDF (text layer only, no OCR) | one record per reconstructed paragraph, with section heading and page: lines are re-joined across line and page breaks (hyphenation repaired), running headers/footers, page numbers, table rows, captions, contents pages and back matter (acknowledgements, funding, author notes) are dropped, and the text stops at the reference list; every removal is counted in the upload warnings |
| HTML (saved web page) | one record per article paragraph plus one per reader comment; navigation, sidebars, share widgets, comment forms and duplicated contents lists are dropped; articles split across sibling containers are joined; respondent IDs are pseudonymous (`<page>:author`, `<page>:comment-<n>`) and commenter names are not stored. **Transcripts** (podcast or interview pages with ≥ 5 `Name:` turns) become one record per turn with respondent `<page>:speaker-<n>`; names and timestamps are not stored |

Legacy `.xls` and TSV are not supported. Corrupted or empty files are kept as *failed* uploads with the parser's reason. The default upload limit is 200 MB (Settings).

**Other languages.** Text features, stopwords and sentiment are English-only. Every record is screened with an offline stopword-profile detector (en, pt, es, fr, de, it, nl, id); records clearly in another language are reported in the quality summary and, with *Other languages → Exclude* at mapping time, set aside with the reason `non_english` (English abstracts of foreign-language papers are kept). **Repeated uploads.** When a run includes several datasets, a passage (≥ 8 words) that the same respondent already contributed through another upload — e.g. the same page saved twice — is analysed once; the count is shown as a warning (turn off with `dedupe_across_datasets`).

### Storage, backup and restore

Everything lives in the workspace folder: `~/.thematicml` by default, or the folder named by `THEMATICML_HOME`. It holds `thematicml.sqlite3` (all projects, records, codes, themes, review edits, GNN experiments) and `projects/<id>/` (original uploads exactly as uploaded, run artifacts, GNN checkpoints, exports).

- **Back up:** stop the server, then copy the whole workspace folder. That copy is a complete, restorable backup.
- **Restore:** copy the folder back (or anywhere) and point `THEMATICML_HOME` at it. Stored paths are relative to the workspace, so a restored or moved workspace works in its new location (verified, including original-file downloads and GNN checkpoints).
- **Per-project ZIP** (Export → *Project data & backup*): a portable archive of the project's database rows as JSON plus its files, for archiving or sharing. There is no one-click import of this ZIP yet; use the folder copy to restore.
- **Deleting** a dataset removes its records, the analyses that used it and their files. Deleting a project removes everything under it. Neither can be undone.

For UI development, run the backend on port 8000 and `npm run dev` in `frontend/` (Vite proxies `/api`).

The app has four pages:

1. **Upload & analyse** — choose one file and press *Analyse this file*. The reading check, column mapping and all ML settings are folded under *Advanced options*.
2. **Dashboard** — the seven analysis steps as one row, then charts: themes, tone, what the themes say, codes, co-occurrence, tone by theme, topics, trends and model fit.
3. **Graphs & Patterns** — the interactive code graph, the thematic map, pattern clustering and temporal analysis. Network statistics, comparisons, the GNN and graph exports are folded under *More analysis*.
4. **Database** — file, analysis, date, search and export controls, with tabs for files and records, the thematic analysis steps, quality and statistics, model metrics, and history and storage.

### Use from other devices

There are two editions of the same app.

**Browser edition (the website, for example on Vercel).** The whole app, analysis included, runs inside the page: ThematicML's Python code runs in [Pyodide](https://pyodide.org) (Python compiled to WebAssembly) in a background worker, and the page answers its own `/api` requests. Nothing needs a server, so the site works on any device with a recent Chrome, Edge, Firefox or Safari, and files never leave the device.

- **Build:** `cd frontend; npm run build:browser` packs the Python code and eight pure-Python wheels into `public/engine/` (`scripts/pack-python.mjs`) and builds with `VITE_ENGINE=browser` into `frontend/dist-browser` (`dist` stays the server edition that `start.ps1` serves). `vercel.json` (repository root, or `frontend/` as the root directory) runs this on every push. To try it locally: `npx vite preview --mode browser --outDir dist-browser`.
- **Start-up:** the first visit downloads Pyodide and its scientific libraries (about 60 MB, from the jsDelivr CDN); later visits load them from the browser cache. Starting the engine still takes tens of seconds on each visit or reload, longer on phones; moving between pages does not restart it.
- **Storage:** projects live in this browser's IndexedDB, saved after every change. Each device and browser has its own projects, and clearing the site's data deletes them: use *Download backup (ZIP)* in Settings to keep a copy. One tab at a time can open them.
- **Speed and limits:** analyses take about 3–4 times as long as on a computer (the 1,201-row sample CSV: 17 s instead of 4.4 s). Leiden community detection is not available (Louvain is used). Very large uploads may exceed a phone browser's memory.
- **GNN:** the built-in NumPy GraphSAGE (see below) trains in the page.

**Server edition (`start.ps1`, Docker).** Analyses run on one computer or server and every device sees the same projects. Once other devices can reach it, a password protects everything: set `THEMATICML_PASSWORD` and everyone signs in as `researcher` (or `THEMATICML_USER`). The browser asks once per session.

- **Same Wi-Fi or office network:** `.\start.ps1 -Network` asks for a password and prints the addresses to open on a phone or laptop, such as `http://192.168.1.20:8000`. This computer must stay on, and Windows may ask to allow Python on private networks. The password travels unencrypted on the local network, so use it on networks you trust.
- **Anywhere, over the internet:** host the whole app on a server. `render.yaml` sets it up on [Render](https://render.com): *New → Blueprint*, pick this repository, enter a password. It builds the Dockerfile, keeps the workspace on a 1 GB disk and serves over HTTPS. The disk needs a paid instance. *Starter* (512 MB) was enough in a test with the same limit: analysing the 1,201-row sample CSV peaked at 260 MB and a 7 MB thesis PDF at 290 MB; choose *standard* if very large uploads fail. Any host that runs a Docker image with a persistent disk works the same way.

## Use from Python

```python
from thematicml import Project, AnalysisConfig

cfg = AnalysisConfig(
    text_col="response_text", id_col="respondent_id", time_col="created_at", group_col="region",
    coding_method="tfidf", clustering_method="kmeans", topic_method="nmf", graph_method="leiden", temporal=True,
)
project = Project("customer-research")
project.load("sample_data/customer_feedback_synthetic.csv")
result = project.analyze(cfg)

result.summary()
result.codes.to_csv("codes.csv", index=False)
result.themes.to_csv("themes.csv", index=False)
result.metrics.to_json("metrics.json")
result.export("pdf")            # also: docx, html, xlsx, codes_csv, graphml, ...
```

Projects opened from Python and from the web app share the same workspace. See `example_config.json` for every parameter.

## The pipeline

| Stage | Method |
|---|---|
| Preprocessing | Validation (missing text, duplicates, unparseable timestamps), sentence/paragraph segmentation with character offsets into the unchanged original |
| Features | TF-IDF with n-grams, configurable stopwords that keep negations, domain synonyms |
| Initial coding | Ranked TF-IDF phrases with redundancy pruning; keyword/wildcard codebook rules with optional categories; with both, TF-IDF codes fill in terms the codebook does not already match; optional logistic-regression classifier on researcher labels |
| Clustering | TF-IDF → truncated SVD → K-means or Ward agglomerative; k chosen by silhouette when not fixed |
| Topics | NMF (TF-IDF) or LDA (counts) over content terms (negation-only terms, kept for coding, are left out); UMass coherence and topic diversity |
| Statistics | Prevalence at segment, record and respondent level with explicit numerator/denominator and Wilson CIs; co-occurrence; chi-square/Fisher group tests with Benjamini–Hochberg correction; VADER sentiment |
| Graph | Code co-occurrence graph (NPMI-filtered) plus TF-IDF similarity edges; Louvain/Leiden/greedy/label-propagation communities; degree, betweenness, PageRank; Adamic–Adar link-prediction baseline |
| Themes | Graph communities become candidate themes, labelled by template; representative quotations by code overlap and centroid similarity; polarity-contrasting passages as candidate contradictory evidence |
| Temporal | Per-period shares, rolling mean, exponential smoothing, Mann–Kendall + Sen's slope, change points, seasonal decomposition/ARIMA when data allows, graph snapshots per time window |

Each stage is checkpointed, so a failed or interrupted run resumes from the stage that failed. Runs left unfinished when the server stops are marked *interrupted* on the next start.

### The seven steps of a thematic analysis

The Dashboard shows the steps as one numbered row; each opens where that step lives. Steps 1–6 are under **Database → Thematic analysis**, one at a time. Step 7 sits under the code graph on **Graphs & Patterns**.

| Step | What you get | Export |
|---|---|---|
| 1. Cleaned dataset | How the uploaded records became the analysed ones, and the records themselves. Cleaning removes whole records and page furniture; words are never rewritten, so quotations stay verbatim | `cleaned_csv` |
| 2. Notes | Recurring ideas drafted from the topic model, each with a passage, beside your own notes; notes written on codes and themes are gathered here too | reports |
| 3. Initial coding | Every meaningful statement (a passage with at least one code) with its codes; search it or filter by code | `assignments_csv` |
| 4. Codebook | Code name, definition, category and a verbatim quotation (the code's most typical readable passage); review actions live here | `codebook_csv` |
| 5. Categories | Similar codes grouped. Codes without a category get a dashed suggestion: the nearest category by mean TF-IDF vector when cosine ≥ 0.30, otherwise a group of leftover codes. Suggestions are not stored until accepted; codes too unlike anything stay uncategorised | `categories_csv` |
| 6. Final themes and sub-themes | A theme is a candidate until a researcher approves it; then it is final. Candidate sub-themes group a theme's codes by category (at least two codes); a theme can also be set as a sub-theme of another (one level deep) | `themes_csv` |
| 7. Thematic map | Themes, sub-themes and how the themes relate. Solid lines join themes that share more passages than their sizes would give by chance, dashed lines those that share fewer. Save it as PNG or SVG | `theme_links_csv` |

Theme pairs are tested in the run's co-occurrence unit with a two-sided hypergeometric test, Benjamini–Hochberg adjusted across the pairs, and must also differ from chance by at least 5%. Most pairs usually sit near chance, and a map with few lines is a real result. Passages of one document are not independent, so read the lines as a guide rather than a statistical claim. On the `t.data` corpus, nearest-category suggestions agreed with the draft codebook for 53% of its 68 codes when each was held out (about 11% by chance), which is why they are labelled suggestions.

### What it does not claim

- Clusters, topics and candidate themes are statistical groupings, not interpretations. They need researcher review. The same goes for recurring ideas, suggested categories, candidate sub-themes and the links on the thematic map.
- No single "accuracy" is reported for the unsupervised pipeline. Classification metrics appear only with labelled data and a respondent-grouped held-out split, next to a majority-class baseline.
- Centrality describes position in the code graph, not importance.
- A code missing from a later dataset may just be absent from that sample. Comparisons show dataset sizes, confidence intervals and this caveat.
- Lexicon sentiment misses sarcasm, domain language and context.

### Graph neural networks (optional)

ThematicML trains a **GraphSAGE** model on the code co-occurrence graph of a completed run. Start an experiment from **Graphs & Patterns → More analysis → Graph neural network**, tick it under *Advanced options* on **Upload** (it then runs after the pipeline), or from Python with `result.train_gnn(GNNConfig(...))`. Results appear on Graphs & Patterns (metrics, training curves, embeddings, predicted links, optional overlay on the code graph) and under **Database → Model metrics** (every experiment, with CSV/JSON export).

- **Link prediction** (no labels needed): learns from observed co-occurrence edges and ranks unobserved code pairs. Predictions are labelled as hypotheses everywhere; they are never shown as observed relationships.
- **Node classification**: only with researcher-assigned code *categories* (≥2 categories with ≥3 codes each, ≥10 labelled codes; codebook rules carry their category). It is refused, with the reason, when labels are missing — labels are never generated. Test macro-F1 is reported next to three baselines on the same split: majority class, logistic regression on the node features alone (no graph), and a graph-only neighbour vote (each code takes the category with the most co-occurrence weight among its training-labelled neighbours).

Leakage rules: the input graph contains observed co-occurrence edges only (similarity edges are excluded because they are computed from the same segments); node features are hashed character n-grams of the code label, log frequency and coding method — never corpus TF-IDF; validation and test edges are removed from message passing (test scoring uses train + validation edges); negatives never include known edges; the checkpoint is chosen on validation AUC and the test split is scored once per repeat. Each experiment repeats over several seeds (new split each time) and reports mean ± SD next to Adamic–Adar, Jaccard, common-neighbour and preferential-attachment baselines scored on the **same** test edges. The verdict is paired over repeats: the GNN is called higher or lower than the best baseline only when the mean per-repeat difference ± 2 standard errors excludes zero, and it states in how many repeats the GNN was higher (node classification uses macro-F1 the same way). Graphs where more than 50% of code pairs are linked are refused (raise minimum NPMI / co-occurrence or use segment-level co-occurrence).

Two implementations train the same model. With the `gnn` extra installed, **PyTorch Geometric** is used; without it, and in the browser edition, a built-in **NumPy GraphSAGE** (`thematicml/gnn/numpy_backend.py`) with the same layers, initialisation, Adam optimiser, early stopping, splits and baselines, and hand-written gradients checked against finite differences. Random draws differ, so the same seed gives similar rather than identical numbers: on the tuned t.data run (5 repeats) test ROC-AUC was 0.792 ± 0.025 with NumPy and 0.799 ± 0.030 with PyTorch, both below Adamic–Adar (0.859). `THEMATICML_GNN_BACKEND=numpy` or `pytorch` picks one. Checkpoints are saved under the run folder as `model.pt` (PyTorch, loaded with `torch.load(weights_only=True)`) or `model.npz` (NumPy arrays only, loaded without pickle), and only from inside the workspace; a PyTorch checkpoint cannot be re-scored where PyTorch is not installed. GraphSAGE's mean aggregation ignores edge weights. Remaining nondeterminism: CPU results are seeded; CUDA kernels may differ slightly between runs.

### Tuning and reproducible builds (optional tools)

- `tools/tune.py spec.json --out results.jsonl` runs configuration variants in memory (nothing is written to the workspace) and reports label-free quality measures: topic NPMI coherence × diversity and seed stability, cluster silhouette and seed agreement (adjusted Rand), graph density, modularity, community seed agreement and link-prediction baselines, plus the code, topic and theme lists for reading. Measures guide the choice; reading the codes and quotations decides it.
- `tools/build_project.py --project NAME --config configs/<file>.json --files ...` imports files (skipping byte-identical copies and files already in the project), applies the mapping, runs the analysis and queues the GNN experiments listed in the config, so a stored run can be rebuilt from one file.

## Example: the `t.data/` corpus

The corpus is not in this repository: it is saved third-party articles, papers and reader comments that are not ours to republish. The configurations in `configs/` rebuild the analysis from your own copy of the sources.

**Before** (`Solo female travel (t.data)` project): nine saved web pages, 593 records from 137 respondents, TF-IDF codes only. The codes were mostly generic words (*Think, Way, Things, Great, Shannon*), topics had negative coherence and the two candidate themes were *People & Time* and *Safety & Experience*.

**After** (`Solo female travel (t.data, tuned)` project, rebuilt with `python tools/build_project.py --config configs/solo-female-travel.json`): 38 files — the nine pages, 24 further web pages (podcast and interview transcripts among them) and five papers or theses (one in Portuguese) — giving 2,266 records from 189 respondents after dropping exact re-saves and setting aside 120 paragraphs in another language (Portuguese). The configuration combines a draft codebook of 68 keyword codes in 9 categories (`configs/solo-female-travel.codebook.json`; drafted by an AI assistant from the papers in `t.data` and **meant for researcher review**) with 30 complementary TF-IDF codes, 270 domain stopwords (conversation fillers, generic verbs, academic boilerplate), NMF with 10 topics, K-means with k = 6 and a segment-level code graph (co-occurrence ≥ 3, NPMI > 0.05, Leiden). Settings were chosen with `tools/tune.py` over 53 variants in three rounds and by reading the codes, topics and themes.

Same 2,266-record corpus, label-free measures, 5 seeds:

| Measure | Previous configuration | Tuned |
|---|---|---|
| Topic quality (NPMI × diversity) | 0.121 (one topic is *al / et / et al*) | 0.182 |
| Code graph density / modularity | 0.727 / 0.210 | 0.129 / 0.347 |
| Candidate themes | 2 | 5 |
| Cluster silhouette / seed agreement (ARI) | 0.083 / 0.58 | 0.101 / 0.58 |
| Topic seed agreement (Jaccard) / community seed agreement (ARI) | 1.00 / 1.00 | 0.85 / 0.79 |

The five candidate themes group risk perception and its sociocultural context; on-the-ground safety (accommodation, transport, night, intuition, help from locals, police and staff); motivations and growth (freedom, confidence, comfort zone, self-discovery); online communities, tours, family and itinerary sharing; and practical protection against theft and illness. Communities shift between seeds and with small data changes (removing four table fragments moved the harassment and violence codes from one theme to another), so treat them as candidates. Clusters remain weak at every k tried (silhouette < 0.2).

Reading 272 random held-out codebook matches, about 89% were on topic before the last pruning round (single rater: the same assistant that drafted the codebook).

GraphSAGE on the tuned run (98 codes, 611 edges, 10 repeats; hyperparameters chosen by random search on validation scores only):

| Task | GraphSAGE | Best baseline | Other baselines |
|---|---|---|---|
| Link prediction, test ROC-AUC | 0.818 ± 0.026 | Adamic–Adar 0.860 ± 0.023 | common neighbours 0.852, Jaccard 0.832, preferential attachment 0.810 |
| Node classification (9 codebook categories), test macro-F1 | 0.272 ± 0.138 | neighbour vote (graph only) 0.333 ± 0.141 | features only 0.194, majority class 0.037 |

GraphSAGE scored below Adamic–Adar in all 10 paired repeats (mean −0.042). For categories, the graph carries signal (neighbour vote ≫ majority class), but GraphSAGE adds nothing over a simple vote; with about ten test codes per repeat the estimates are wide. Validation-selected settings did not transfer: the best validation macro-F1 was 0.47, test 0.27. The untuned default gave 0.786 (link prediction) and 0.227 (node classification) on separate confirmation seeds.

## Security and privacy

- **No network use.** The Python package and backend make no outbound requests and no LLM or remote AI service is called; the browser only talks to the local server. Network access is needed only to install packages.
- **Logs.** The start scripts and Docker image run uvicorn with `--no-access-log`, because request URLs can contain search terms and record filters. A failed analysis writes its Python traceback to `projects/<id>/runs/<run>/error.log` inside the workspace; tracebacks can quote short fragments of data.
- **Uploads** are parsed only (HTML is never executed or fetched); file names are reduced to their base name. Parser error messages do not include local paths.
- **Checkpoints** are loaded with `torch.load(weights_only=True)` and only from inside the workspace.
- **Display.** Researcher-entered labels are HTML-escaped in chart tooltips.
- **Sign-in.** Without `THEMATICML_PASSWORD` the app has no login, which is only safe on `127.0.0.1`. With it, every request except the health check needs HTTP Basic sign-in; `start.ps1 -Network` and the Docker image refuse to start without one. Use HTTPS (as Render does) beyond a trusted local network.
- **Not addressed:** one shared password, no per-user accounts or permissions, no limit on sign-in attempts; the SQLite file and uploads are not encrypted at rest (use disk encryption for sensitive data); malicious archive files (for example XLSX/DOCX zip bombs) are limited only by the upload size cap.

## Known limitations

- Clusters on short, heterogeneous text can be weak and unstable. On the `t.data` corpus silhouette stayed below 0.2 for every k tried, automatic k always ran to the top of its range, and K-means runs with different seeds agreed only moderately (adjusted Rand index about 0.58 at k = 6); on the synthetic sample 0.37 and about 0.51. Treat clusters as prompts for reading, not findings.
- Graph communities can also vary with the seed on sparse graphs (Leiden on the tuned `t.data` graph: adjusted Rand index about 0.79 between seeds; synthetic sample: identical).
- PDF tables made only of words are recognised by their capitalised cells and missing sentence ends; a table with mostly lowercase cells (for example a Y/N feature matrix) can still come through as one long paragraph.
- Change-point detection finds sustained level shifts; it does not flag one-period spikes (the synthetic September pricing spike is the highest month but is not flagged). Mann–Kendall tests monotonic trends only.
- The supervised classifier is evaluated on one respondent-grouped hold-out split **and** 5-fold respondent-grouped cross-validation, with the TF-IDF vectorizer re-fitted on training records only. On the synthetic sample: hold-out accuracy 0.713, cross-validation 0.675 ± 0.042, majority baseline 0.217. Synthetic template text makes this easy; it says nothing about real data.
- Topic and theme labels are templates built from top terms; they need a researcher-written definition.

## Tests

```powershell
python -m pytest            # GNN tests are skipped automatically if torch_geometric is not installed
pip install playwright      # optional: browser smoke check using the installed Edge or Chrome
python tools/ui_smoke.py http://127.0.0.1:8000 screenshots/
```

The suite covers data import for every format (including HTML article/comment extraction), validation, percentages, coding, clustering, topics, graph construction and metrics, temporal statistics, evidence verification, persistence across restarts, moving/restoring a workspace, deletion, cross-run ID validation, checkpoint resume, leakage-free classifier evaluation, GNN splits/leakage/checkpoints, exports and the full HTTP workflow.

## Sample data

`sample_data/customer_feedback_synthetic.csv` is **synthetic**, generated by `sample_data/generate_sample.py` (deterministic). It has 1,201 template-assembled responses over 2024 with planted patterns: performance complaints rise after June, support complaints fall, and pricing spikes in September. It also includes empty responses and one duplicate to exercise validation. It is not real research data.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: thematicml` when starting the server | Run `pip install -e .` in the project's virtual environment, or use `start.ps1`, which uses `.venv` |
| Port 8000 already in use | `.\start.ps1 -Port 8010` (or `PORT=8010 ./start.sh`) |
| Blank page at `/` | The UI is not built: `cd frontend; npm install; npm run build` |
| PyTorch install fails with "connection reset" | Retry with `pip install --retries 10 --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu` |
| GNN says the graph is too dense | Use segment-level co-occurrence or raise minimum NPMI / co-occurrence (Graphs & Patterns → H) |
| An HTML page yields almost nothing | It is an index/home page, or its article loads with JavaScript and was not saved; save the article page itself |
| Docker: `failed to connect to the docker API` | Start Docker Desktop first |

## Docker

```bash
docker build -t thematicml .
docker run -p 8000:8000 -e THEMATICML_PASSWORD=choose-one -v thematicml-data:/data thematicml
```

The image will not start without `THEMATICML_PASSWORD`. It listens on `$PORT` when a hosting service sets one.

The image contains the classical pipeline with the sentiment, temporal, Leiden and PDF extras but not the GNN (PyTorch is large); add `gnn` to the extras in the Dockerfile to include it. The image was built and run here (Docker 29.1) with a 512 MB memory limit.
