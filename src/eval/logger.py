"""Prediction logging for the shared eval harness (CLAUDE.md: every
prediction logged as JSONL -- model, input id, prediction, ground truth,
latency, confidence).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EVAL_LOGS_DIR = Path("eval_logs")
REQUIRED_COLUMNS = ["fold", "hadm_id", "y_true", "y_prob", "latency_ms"]
LOG_COLUMNS = [
    "model", "run_id", "timestamp", "fold", "hadm_id",
    "prediction", "ground_truth", "confidence", "latency_ms",
]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log_predictions(
    predictions: pd.DataFrame,
    model: str,
    run_id: str | None = None,
    eval_logs_dir: Path = EVAL_LOGS_DIR,
) -> Path:
    """predictions: run_cv(..., return_predictions=True)'s 3rd return value
    (columns: fold, hadm_id, y_true, y_prob, latency_ms). Writes one JSONL
    file, one line per row; returns the path written.

    `prediction` is a naive 0.5-threshold convenience field only -- at this
    dataset's ~8.5% positive rate it is a poor operating point and is NOT
    used by src/eval/metrics.py, which works from `confidence` directly.
    """
    missing = set(REQUIRED_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")

    run_id = run_id or new_run_id()
    timestamp = datetime.now(timezone.utc).isoformat()
    eval_logs_dir = Path(eval_logs_dir)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_logs_dir / f"{model}_{run_id}.jsonl"

    with out_path.open("w") as f:
        for row in predictions.itertuples(index=False):
            f.write(
                json.dumps(
                    {
                        "model": model,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "fold": int(row.fold),
                        "hadm_id": int(row.hadm_id),
                        "prediction": int(row.y_prob >= 0.5),
                        "ground_truth": int(row.y_true),
                        "confidence": float(row.y_prob),
                        "latency_ms": float(row.latency_ms),
                    }
                )
                + "\n"
            )
    return out_path


def load_all_logs(eval_logs_dir: Path = EVAL_LOGS_DIR) -> pd.DataFrame:
    """Concatenate every eval_logs/*.jsonl into one DataFrame. Returns an
    empty DataFrame with LOG_COLUMNS (not an error) if no logs exist yet --
    the fresh-checkout / never-run-yet state is expected.
    """
    eval_logs_dir = Path(eval_logs_dir)
    files = sorted(eval_logs_dir.glob("*.jsonl"))
    if not files:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
