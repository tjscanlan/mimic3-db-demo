# mimic3-db-demo

A healthcare ML portfolio project: predict 30-day hospital readmission on the [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) (public, 100-patient subset from PhysioNet), comparing two parallel approaches on the same patients — structured feature engineering feeding a classic gradient-boosted model, and clinical text feeding a fine-tuned small language model — converging on a shared evaluation harness.

This is a research/education project, not a clinical tool.

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

**Note:** the demo release ships `NOTEEVENTS.csv` empty (header only) — PhysioNet strips discharge-summary text from the demo; it's only available in the full credentialed MIMIC-III database. The download script and EDA notebook both flag this explicitly.

## Exploring the data

```bash
uv run jupyter lab notebooks/01_eda.ipynb
```
