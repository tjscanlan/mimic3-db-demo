# Architecture Decision Record

A log of the architecturally significant decisions made building this project, in roughly chronological order. Each entry is Accepted unless noted otherwise — this is a single-maintainer portfolio project, so nothing here has gone through a review process; "Accepted" means "implemented and current."

## Index

1. [Three parallel model paths, one shared evaluation harness](#adr-001)
2. [Structured path: XGBoost with stratified k-fold CV](#adr-002)
3. [Text path: DistilBERT + LoRA](#adr-003)
4. [Every prediction logged uniformly as JSONL](#adr-004)
5. [NOTEEVENTS is permanently synthetic](#adr-005)
6. [Synthetic notes carry an explicit marker and a deliberate, mild label correlation](#adr-006)
7. [Process isolation for the xgboost/torch OpenMP segfault](#adr-007)
8. [RAG v2: k-NN retrieval-as-classifier, index-as-artifact](#adr-008)
9. [Gradio demo shells out to a subprocess for text inference](#adr-009)
10. [Raw data, synthetic data, and trained artifacts are never committed](#adr-010)
11. [CLAUDE.md and docs/initial_startup.md are local-only](#adr-011)
12. [CI: GitHub Actions reruns all three paths on every PR](#adr-012)
13. [CI gates on fixed thresholds, not historical/main-branch comparison](#adr-013)
14. [rag_knn's CI gate is non-blocking, not hidden or blocking](#adr-014)
15. [CI runs the SLM path with production defaults](#adr-015)
16. [CI renders a real markdown table via tabulate](#adr-016)
17. [torch is pinned to CPU-only wheels on Linux](#adr-017)
18. [CI restricts the PhysioNet download to tables the code actually reads](#adr-018)
19. [The regression gate script never imports xgboost or torch](#adr-019)
20. [CI verifies notes are synthetic as a defense-in-depth check](#adr-020)
21. [Cost/latency dashboard: illustrative $ estimate, benchmark + live-accumulation data, a separate log stream](#adr-021)

---

<a id="adr-001"></a>
## ADR-001: Three parallel model paths, one shared evaluation harness

**Status:** Accepted

**Context:** The project's purpose is a head-to-head portfolio comparison of modeling approaches on the same prediction task (30-day readmission), not just building one working model.

**Decision:** Build three independent prediction paths against the same 129-admission cohort — structured features → XGBoost, discharge-note text → fine-tuned DistilBERT+LoRA, and note embeddings → k-NN retrieval (added later, see ADR-008) — and route every path's predictions through one shared logging/metrics harness (`src/eval/`) rather than separate evaluation code per path.

**Consequences:** Any two paths are directly comparable (same cohort, same CV scheme, same logged schema). The cost is coordination overhead — a schema or convention change in `src/eval/logger.py` affects all three paths at once, and each path's training script independently implements the same stratified-k-fold CV loop rather than sharing one CV harness (each script does have its own reasons to differ: XGBoost fits per fold, the SLM fine-tunes per fold, k-NN has no fitting step at all — see ADR-008).

---

<a id="adr-002"></a>
## ADR-002: Structured path uses XGBoost with stratified k-fold CV

**Status:** Accepted

**Context:** The original project plan (`docs/initial_startup.md`) left the exact GBT library open ("XGBoost/LightGBM"). The cohort is small (129 rows, ~8.5% positive), so cross-validation matters more than raw training speed, and class imbalance makes stratification necessary to keep every fold's positive count non-trivial.

**Decision:** XGBoost, evaluated via `StratifiedKFold` (`src/models/train_baseline.py`), default `n_splits=5`, `seed=42`.

**Consequences:** XGBoost's Homebrew/libomp build on macOS is what collides with torch's bundled OpenMP (see ADR-007) — a direct downstream consequence of this choice. LightGBM was not implemented or benchmarked against it; no comparison data exists to revisit this decision against.

---

<a id="adr-003"></a>
## ADR-003: Text path uses DistilBERT + LoRA

**Status:** Accepted

**Context:** The original plan left the SLM choice open (ClinicalBERT vs. DistilBERT vs. a small generative model). The cohort is tiny and CPU-only training is the norm for local dev on this project (see ADR-007, ADR-017), which favors a smaller base model and a parameter-efficient fine-tuning method over full fine-tuning of a larger clinical-domain model.

**Decision:** `distilbert-base-uncased` (~66M params) fine-tuned via LoRA (`src/models/train_slm.py`, `peft`), not ClinicalBERT and not a generative model.

**Consequences:** Faster iteration and CPU-feasible fine-tuning (25 fold-fits complete in low minutes — see ADR-015), at the cost of no clinical-domain pretraining that ClinicalBERT would have provided. Since the text path only ever trains on synthetic notes (ADR-005), any accuracy delta this choice would produce on real clinical language is untested and moot for this project as scoped.

---

<a id="adr-004"></a>
## ADR-004: Every prediction logged uniformly as JSONL

**Status:** Accepted

**Context:** Comparing paths head-to-head (ADR-001) requires a common prediction record, independent of which model produced it.

**Decision:** `src/eval/logger.py::log_predictions()` writes one `eval_logs/{model}_{run_id}.jsonl` file per run, one line per prediction, fixed schema: `model, run_id, timestamp, fold, hadm_id, prediction, ground_truth, confidence, latency_ms`. `prediction` is a naive 0.5-threshold convenience field only; `src/eval/metrics.py` and `check_regression.py` both work from `confidence` directly, since 0.5 is a poor operating point at this dataset's ~8.5% positive rate.

**Consequences:** `eval_logs/*.jsonl` is small enough to commit directly to git (unlike raw data or model artifacts, see ADR-010) — it doubles as a lightweight experiment-tracking log, and is exactly what CI reads after each run (ADR-012). `run_id` (a UTC timestamp string) doubling as both a uniqueness key and a sort key is what makes the "latest run per model" pattern in `metrics.py` and `check_regression.py` work without a separate index.

---

<a id="adr-005"></a>
## ADR-005: NOTEEVENTS is permanently synthetic

**Status:** Accepted

**Context:** The MIMIC-III demo release ships `NOTEEVENTS.csv` with 0 data rows — PhysioNet strips discharge-summary text from the demo; real note text exists only in the full credentialed MIMIC-III database, which requires CITI certification. The maintainer has decided not to pursue that certification.

**Decision:** The text and RAG paths run on synthetic notes fabricated from the real structured data (`data/generate_mock_noteevents.py`), permanently — not as a placeholder pending future credentialed access.

**Consequences:** The text and RAG paths' numbers are pipeline sanity checks, never real findings about text-based readmission prediction, and must never be reported as clinical results (reinforced by ADR-006 and ADR-020). This also means DistilBERT+LoRA (ADR-003) will never be evaluated against real clinical language in this project.

---

<a id="adr-006"></a>
## ADR-006: Synthetic notes carry an explicit marker and a deliberate, mild label correlation

**Status:** Accepted

**Context:** Given ADR-005, a classifier trained on fabricated text needs *something* learnable to prove the fine-tuning pipeline runs end-to-end, while the fabrication must never be mistaken for real clinical data.

**Decision:** Every generated note is prefixed with `SYNTHETIC_MARKER` (`"[SYNTHETIC NOTE -- generated for pipeline development, NOT real clinical text]"`), and phrasing is drawn from a high-risk/low-risk phrase pool with `P(high-risk phrase | label=1) = 0.75` vs. `0.25` — mild, not perfect separation, so the classification task has signal without being trivial.

**Consequences:** This is circular by construction: the model is learning a correlation the generation script itself created, so any reported accuracy is a pipeline sanity check, not evidence about real readmission prediction (ADR-005). `src/nlp/preprocess.py` strips the leading marker before tokenization (zero discriminative signal, wastes tokens). `data/verify_synthetic_notes.py` checks the marker's presence in CI (ADR-020).

---

<a id="adr-007"></a>
## ADR-007: Process isolation for the xgboost/torch OpenMP segfault

**Status:** Accepted

**Context:** xgboost's Homebrew libomp and torch's bundled OpenMP conflict and segfault when both are imported into the same Python process (confirmed via macOS crash reports during `src/models/train_slm.py` development).

**Decision:** Never import both dependency trees into one process. `app.py` never imports `torch`/`transformers`/`peft`/`sentence_transformers`; text-model inference is served by shelling out to `uv run python -m src.nlp.predict` as a fresh subprocess per request (ADR-009). `src/nlp/dataset.py` (torch `Dataset` wrapper) is split out of `src/nlp/preprocess.py` so the latter stays torch-free and importable from `app.py`. `src/rag/build_index.py` (embeds notes with sentence-transformers) is a standalone offline CLI nothing else imports. `src/rag/knn.py` is pure numpy/pandas at inference time, so unlike the text path it's safe to import directly into `app.py`'s process.

**Consequences:** This constraint now shapes every new module boundary in the codebase — any new code touching both dependency trees must keep them in separate processes. It's also why `check_regression.py` (ADR-019) and `data/verify_synthetic_notes.py` are written to need neither `xgboost` nor `torch`/`transformers` at all.

---

<a id="adr-008"></a>
## ADR-008: RAG v2: k-NN retrieval-as-classifier, index-as-artifact

**Status:** Accepted

**Context:** The original plan listed an "optional v2: RAG layer (retrieve similar historical notes)" without specifying the retrieval mechanism or how it would feed a prediction.

**Decision:** k-NN over note embeddings, used directly as a classifier (`src/rag/knn.py`, `src/models/train_rag.py`) — not a generative RAG pipeline. Embeddings are built once (`src/rag/build_index.py`, `all-MiniLM-L6-v2`) and persisted to `models/rag_knn/`; that persisted index *is* the deployable artifact. `train_rag.py`'s docstring makes this explicit: there's no `make_model()`/`train_final_model()`/`save_final_model()` because k-NN has no fitting step beyond the embedding index itself.

**Consequences:** Simpler artifact lifecycle than the other two paths (no separate "train" vs. "save for deployment" step — building the index *is* training). Also means `rag_knn`'s quality is entirely a function of embedding quality and `k`, with no model capacity to compensate — plausibly related to why it's currently the weakest-performing path (ADR-014).

---

<a id="adr-009"></a>
## ADR-009: Gradio demo shells out to a subprocess for text inference

**Status:** Accepted

**Context:** Direct consequence of ADR-007 — `app.py` needs to serve XGBoost (in-process) and text-model (torch) predictions from one running demo process without segfaulting.

**Decision:** `app.py` shells out to `uv run python -m src.nlp.predict --hadm-id ... --json-out /tmp/out.json` as a fresh subprocess per text-model request; `src/nlp/predict.py` writes its result as JSON to a file rather than stdout, specifically to be robust to any stray library stdout noise from the subprocess.

**Consequences:** Per-request subprocess spawn + model reload is real latency overhead the demo accepts in exchange for never risking a segfault. The RAG path avoids this entirely (ADR-007's last point) since `knn.py` is pure numpy and safe in-process.

---

<a id="adr-010"></a>
## ADR-010: Raw data, synthetic data, and trained artifacts are never committed

**Status:** Accepted

**Context:** Raw MIMIC data is PHI-adjacent even in demo form; synthetic notes and trained models are regenerable build output, not source.

**Decision:** `.gitignore` excludes `data/raw/*`, `data/mock/*`, and `models/*` (each keeps only a `.gitkeep`). Only download/generation scripts live in `data/`; `eval_logs/*.jsonl` is the deliberate exception (ADR-004) since it's small, useful history, and contains no raw patient data.

**Consequences:** A fresh checkout (including CI) starts with none of this and must regenerate it every time — this is exactly why the CI workflow (ADR-012) has explicit download/generate/build-index steps rather than assuming any of it pre-exists.

---

<a id="adr-011"></a>
## ADR-011: CLAUDE.md and docs/initial_startup.md are local-only

**Status:** Accepted

**Context:** `docs/initial_startup.md` is the maintainer's personal planning notes; `CLAUDE.md` is AI-agent guidance that's useful locally but not part of the project's public-facing artifact.

**Decision:** Both are gitignored — present on disk, absent from git history.

**Consequences:** This ADR file, by contrast, is deliberately committed (`docs/ADR.md`, not gitignored) since it documents decisions for any reader of the repository, not just local tooling.

---

<a id="adr-012"></a>
## ADR-012: CI — GitHub Actions reruns all three paths on every PR

**Status:** Accepted

**Context:** No CI existed; the project wanted a concrete MLOps/EvalOps artifact demonstrating automated regression detection, not just a one-off local eval harness.

**Decision:** `.github/workflows/eval-regression.yml` triggers on PRs and pushes to `main` (path-filtered to skip doc-only changes), and re-runs the full pipeline from scratch each time: download data → generate synthetic notes → verify synthetic (ADR-020) → train/evaluate all three paths → check regression (ADR-013). Fastest/cheapest path (XGBoost) runs before the slowest (SLM fine-tune, ADR-015) to fail fast.

**Consequences:** Every PR pays the full pipeline cost (data download, index build, fine-tune) rather than a cached/incremental check — acceptable at this cohort size (low minutes total) but wouldn't scale to a larger dataset without rethinking caching or a smoke-test tier.

---

<a id="adr-013"></a>
## ADR-013: CI gates on fixed thresholds, not historical/main-branch comparison

**Status:** Accepted

**Context:** Two designs were considered: compare each PR's metrics against a fixed floor committed in repo config, or against the previous run's numbers (mirroring `src/eval/metrics.py::drift_analysis`, which already compares a model's two most recent logged runs).

**Decision:** Fixed thresholds, committed in `src/eval/thresholds.json`, checked by `src/eval/check_regression.py`.

**Consequences:** Simpler and fully deterministic — no dependency on CI run history, no risk of a slowly-drifting baseline masking gradual regression. The trade-off is that thresholds are static and must be manually revisited (e.g., when `rag_knn` is eventually fixed, ADR-014) rather than auto-adjusting to a new normal.

---

<a id="adr-014"></a>
## ADR-014: rag_knn's CI gate is non-blocking, not hidden or blocking

**Status:** Accepted

**Context:** At the time this gate was built, `rag_knn`'s logged performance (roc_auc=0.399) was already below random (0.5) — a real, pre-existing weakness in the retrieval path, not a CI artifact. Three options existed: set its threshold quietly low enough to always pass (hiding the issue), set it at a normal bar alongside the other two paths (blocking all PRs on an unrelated known issue), or something in between.

**Decision:** `rag_knn`'s threshold floor sits just under its own current (bad) baseline, and `"blocking": false` in `thresholds.json` means it reports FAIL/WARN in the summary table but never flips CI red. The threshold entry carries an explicit `_note` explaining why.

**Consequences:** The gate ships without conflating "add CI" with "fix the RAG path." The honest cost: `rag_knn` can regress further without blocking a PR, so this relies on the warning actually being read, not just the exit code. `thresholds.json`'s note says to flip it to `blocking: true` once the retrieval path is fixed and re-baselined.

---

<a id="adr-015"></a>
## ADR-015: CI runs the SLM path with production defaults

**Status:** Accepted

**Context:** `train_slm.py`'s default `--epochs 5 --n-splits 5` (25 fold-fits, CPU-only) is the slowest step in the workflow. A reduced-epoch/fewer-fold "CI smoke test" config was considered for speed.

**Decision:** CI runs the same defaults as local training, just with `--seed 42` passed explicitly for clarity.

**Consequences:** CI numbers are directly comparable to local runs — no separate "CI-only" thresholds or asterisked caveat needed. The cost is a slower CI job; acceptable given the tiny cohort keeps even 25 fold-fits to low minutes.

---

<a id="adr-016"></a>
## ADR-016: CI renders a real markdown table via tabulate

**Status:** Accepted

**Context:** `check_regression.py`'s summary table can render as plain `to_string()` text (no new dependency) or a real markdown table via `pandas.DataFrame.to_markdown()` (requires the `tabulate` package) in the GitHub Actions job summary.

**Decision:** Added `tabulate` as a dependency for a properly rendered table in the PR's checks tab.

**Consequences:** One extra dependency in exchange for a CI check that reads as polished when shown to a reviewer — a deliberate trade given this project's portfolio purpose, made explicitly over the repo's general preference for a minimal dependency set.

---

<a id="adr-017"></a>
## ADR-017: torch is pinned to CPU-only wheels on Linux

**Status:** Accepted

**Context:** `torch>=2.13.0`'s default Linux dependency resolution pulls the full CUDA toolkit stack (`cuda-toolkit`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton`, etc. — 17 extra packages) even on a GPU-less GitHub Actions runner. Invisible on the maintainer's macOS dev machine, since macOS wheels have no CUDA variant.

**Decision:** `pyproject.toml` adds a `[tool.uv.sources]`/`[[tool.uv.index]]` override routing `torch` to `https://download.pytorch.org/whl/cpu` specifically when `sys_platform == 'linux'`, `explicit = true` so it can't shadow PyPI resolution for anything else.

**Consequences:** Regenerating `uv.lock` after editing `pyproject.toml` correctly dropped all 17 CUDA/NVIDIA packages from the Linux resolution. This is now load-bearing tribal knowledge: if `uv.lock` is ever regenerated and CUDA packages reappear, the override broke (see `CLAUDE.md`'s architectural-constraints section).

---

<a id="adr-018"></a>
## ADR-018: CI restricts the PhysioNet download to tables the code actually reads

**Status:** Accepted

**Context:** `data/download_mimic_demo.py`'s default file list is all 24 demo tables (~102MB, 77.7MB of which is the unused `CHARTEVENTS.csv`), but only 6 are read anywhere in the codebase (`ADMISSIONS`, `PATIENTS`, `DIAGNOSES_ICD`, `LABEVENTS`, `ICUSTAYS`, `D_ICD_DIAGNOSES`), plus `NOTEEVENTS.csv` which the script's own `validate_downloads()` unconditionally requires to exist (even though its content is never read downstream — see ADR-005).

**Decision:** The CI workflow passes an explicit `--files` list of just those 7 tables.

**Consequences:** Cuts the CI download to roughly 12MB. Local/manual runs of `download_mimic_demo.py` still default to all 24 tables (unchanged) — this restriction is CI-specific, not a change to the script's default behavior.

---

<a id="adr-019"></a>
## ADR-019: The regression gate script never imports xgboost or torch

**Status:** Accepted

**Context:** Direct application of ADR-007's constraint to new code: `check_regression.py` runs as the final CI step, after all three training subprocesses have already exited.

**Decision:** `check_regression.py` only reads the logged JSONL (via `src.eval.logger.load_all_logs`) and computes `roc_auc`/`pr_auc` with `scikit-learn` — it never touches the model libraries themselves, only their already-logged output.

**Consequences:** No risk of this script ever becoming the thing that finally puts xgboost and torch in the same process. It also means the script needs no knowledge of how a model was trained, only that it logged in the shared schema (ADR-004).

---

<a id="adr-020"></a>
## ADR-020: CI verifies notes are synthetic as a defense-in-depth check

**Status:** Accepted

**Context:** ADR-005 and ADR-006 establish that this project must never process real discharge-summary text. Nothing in the current pipeline would pull real text — no PhysioNet credentials are configured anywhere — but that invariant isn't verified anywhere; it just happens to be true today.

**Decision:** `data/verify_synthetic_notes.py`, run as a CI step right after note generation and before any training: asserts the real download's `NOTEEVENTS.csv` still has 0 data rows (i.e., no credentials somehow got configured and pulled real content), and that every row of the synthetic `NOTEEVENTS.csv` starts with `SYNTHETIC_MARKER` (ADR-006). Exits non-zero and fails the build if either check fails.

**Consequences:** This asserts the outcome that actually matters (no real text in play) rather than a proxy for it (no credentials configured) — durable even if something upstream changes (e.g., `PHYSIONET_USER`/`PHYSIONET_PASS` ever got added as repo secrets for an unrelated reason). Adds one fast, cheap CI step in exchange for that guarantee.

---

<a id="adr-021"></a>
## ADR-021: Cost/latency dashboard: illustrative $ estimate, benchmark + live-accumulation data, a separate log stream

**Status:** Accepted

**Context:** ADR-001–ADR-004 established a shared harness for comparing model *accuracy*, but nothing measured the cost of actually serving a prediction — a real gap given ADR-009's subprocess-per-request pattern for the text path is a meaningfully different cost profile than the in-process XGBoost and k-NN paths. This project has no billed inference API to pull real dollar costs from, and a demo dashboard opened cold with zero data makes a weak first impression.

**Decision:** Three sub-decisions, made together:
- **Cost model:** `src/eval/cost_logger.py::estimate_cost_usd()` converts measured latency through a hand-picked, clearly-labeled `REFERENCE_INSTANCE_HOURLY_USD` constant (illustrative, ~a small general-purpose cloud CPU instance) rather than a relative-units-only or latency-only framing — a concrete, relatable number, at the cost of it looking more precise than it is if the labeling is ever stripped out. The UI and module docstring both say explicitly that this isn't real billing data.
- **Data source:** `src/eval/run_cost_benchmark.py` runs a one-time sample (`--n-samples`, default 20) through all three paths and logs every prediction's latency/cost, so the dashboard has a real distribution to show immediately; every live click in the demo's Patient Explorer tab (`app.py::run_admission`) appends more samples to the same per-model log afterward via `src/serving.py`'s latency-measuring predict functions.
- **Log separation:** cost/latency samples go to a new `cost_logs/{model}.jsonl` stream (`src/eval/cost_logger.py`), not `eval_logs/` — different purpose (production-cost dashboard vs. CV-accuracy regression gate), different shape (continuously-appended per-model file vs. one-file-per-CV-run), and keeping them apart means a cost sample can never affect `check_regression.py`'s "latest run per model" logic (ADR-013).

Confirmed empirically running the benchmark script: structured XGBoost (~1.2ms) and RAG k-NN (~0.3ms) are sub-millisecond in-process calls, while the text path's subprocess-per-request pattern costs ~2,800–4,200ms per prediction — a real, roughly four-orders-of-magnitude latency gap, exactly the "worth it in production" signal this dashboard exists to surface (and a direct, measured consequence of ADR-007/ADR-009's process-isolation design).

**Consequences:** `src/serving.py` was extracted as a new shared module (model loading + single-admission prediction with latency timing) so `app.py` and `run_cost_benchmark.py` don't duplicate that logic — both are now thin callers over it. `cost_logs/` is gitignored like `data/raw`, `data/mock`, and `models/` (regenerable, and — unlike `eval_logs/` — meant to grow indefinitely with local usage rather than serve as shared committed history), so a fresh checkout's dashboard is empty until the benchmark script runs. The aggregation that joins `cost_logs/` latency/cost with `eval_logs/` accuracy into the dashboard's comparison table and chart (`src/eval/cost_dashboard.py::build_dashboard_data`) is where the remaining real judgment calls live (which percentile to show, how to scale cost to something legible, chart encoding) and was left as a human-authored contribution rather than fully scaffolded.
