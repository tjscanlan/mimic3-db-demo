# mimic3-db-demo

A healthcare ML portfolio project: predict 30-day hospital readmission on the [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) (public, 100-patient subset from PhysioNet), comparing three parallel approaches on the same patients — structured feature engineering feeding an XGBoost model, clinical text feeding a fine-tuned DistilBERT+LoRA classifier, and k-NN retrieval over note embeddings — converging on a shared evaluation harness. A Gradio demo ties all three together.

This is a research/education project, not a clinical tool.

**Note on text data:** the MIMIC-III demo ships `NOTEEVENTS.csv` empty — PhysioNet strips discharge-summary text from the demo release; real notes only exist in the full credentialed database, which requires CITI certification this project doesn't pursue. So the text and RAG paths here run on *synthetic* discharge summaries (`data/generate_mock_noteevents.py`), generated from the real structured data with a deliberate artificial correlation baked in — a permanent design choice, not a placeholder awaiting real data. They're pipeline sanity checks, not real findings about text-based readmission prediction — see that script's docstring for details. Likewise, the cohort is only 129 admissions / 11 positives, so treat any subgroup/calibration metric as illustrative rather than statistically defensible.

## Setup

```bash
uv sync
```

## Getting the data

1. Create a free [PhysioNet](https://physionet.org/) account if you don't already have one, then visit the [demo dataset page](https://physionet.org/content/mimiciii-demo/1.4/) and accept its Data Use Agreement. No CITI training is required for the demo (only for the full credentialed MIMIC-III database).
2. (Optional) The demo is open access and downloads unauthenticated by default. If you ever need HTTP basic auth — e.g. reusing this script against the full credentialed database — set `PHYSIONET_USER`/`PHYSIONET_PASS` env vars, or add a `physionet.org` entry to `~/.netrc`.
3. Run the download script:
   ```bash
   uv run python data/download_mimic_demo.py
   ```
   It's safe to re-run — already-downloaded files are skipped. Raw CSVs land in `data/raw/`, which is gitignored; **raw MIMIC data is never committed**.
4. Generate the synthetic discharge notes the text/RAG paths need (also gitignored):
   ```bash
   uv run python data/generate_mock_noteevents.py
   ```
5. (Optional) Confirm no real note text is in play — checks the real download stayed empty and every synthetic note carries its marker:
   ```bash
   uv run python data/verify_synthetic_notes.py
   ```

## Exploring the data

```bash
uv run jupyter lab notebooks/01_eda.ipynb
```

## Training the models

Each script runs stratified k-fold CV and logs every fold's predictions to `eval_logs/*.jsonl`. Pass `--save-model-dir` / `--save-adapter-dir` (or, for the RAG index, just run `build_index`) to also produce the artifact the Gradio demo loads for live inference.

```bash
# Structured path: XGBoost on ADMISSIONS/PATIENTS/DIAGNOSES_ICD features
uv run python -m src.models.train_baseline --save-model-dir models/structured_xgboost

# Text path: DistilBERT + LoRA fine-tuned on synthetic note text
uv run python -m src.models.train_slm --save-adapter-dir models/text_distilbert_lora

# RAG path: builds the note-embedding index, then evaluates k-NN retrieval-as-classifier over it.
# The index itself (not a trained model file) is the deployable artifact.
uv run python -m src.rag.build_index
uv run python -m src.models.train_rag
```

Each script accepts `--help` for its full option list (data dir, number of folds, seed, etc.).

## Evaluating results

```bash
uv run jupyter lab notebooks/02_baseline_model.ipynb   # structured path
uv run jupyter lab notebooks/03_slm_baseline.ipynb     # text path
uv run jupyter lab notebooks/04_eval_harness.ipynb     # subgroup / calibration / drift, across eval_logs/
```

## Running the demo

```bash
uv run python app.py
```

The demo checks for the three trained artifacts above on startup and prints the exact commands to generate any that are missing. It never imports `torch`/`transformers` directly — see `app.py`'s docstring for why (an XGBoost/PyTorch OpenMP conflict segfaults when both are loaded in one process) — so text-model predictions are served via a subprocess call to `src/nlp/predict.py`.

The demo's **Cost / Latency Dashboard** tab compares all three paths on more than accuracy — latency and an illustrative estimated cost per prediction, so the comparison includes what a model would actually cost to run in production. It's empty on a fresh checkout; seed it once with a small benchmark sweep (grows further with every prediction you make in the Patient Explorer tab):

```bash
uv run python -m src.eval.run_cost_benchmark
```

## CI

Every PR reruns all three training paths (`.github/workflows/eval-regression.yml`) and fails the build if a model's ROC-AUC or PR-AUC on the current run drops below the fixed floor in `src/eval/thresholds.json`. `rag_knn` is currently a non-blocking (warn-only) check — see that file for why.

Run the same check locally against whatever's already in `eval_logs/`:

```bash
uv run python -m src.eval.check_regression
```
