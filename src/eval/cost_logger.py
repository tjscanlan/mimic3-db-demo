"""Cost/latency logging for app.py's Cost / Latency Dashboard tab (see
docs/ADR.md, ADR-021).

Deliberately separate from eval_logs/ -- that stream is per-fold CV
accuracy data consumed by check_regression.py's "latest run per model"
logic (metrics.py, check_regression.py); this one is single-inference
latency/cost samples that accumulate continuously (one file per model,
appended to, not one file per run). Keeping them apart means a stray cost
sample can never affect the regression gate.

REFERENCE_INSTANCE_HOURLY_USD is an illustrative, hand-picked constant
(~a small general-purpose cloud CPU instance, on-demand, rounded) -- this
demo has no billed inference API behind it, so "cost" here is latency
converted through an assumed compute rate, not a real invoice. Never
present it as more precise than that.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COST_LOGS_DIR = Path("cost_logs")
REFERENCE_INSTANCE_HOURLY_USD = 0.10

COLUMNS = ["model", "timestamp", "hadm_id", "source", "latency_ms", "estimated_cost_usd"]


def estimate_cost_usd(latency_ms: float) -> float:
    return (latency_ms / 1000 / 3600) * REFERENCE_INSTANCE_HOURLY_USD


def log_cost_sample(
    model: str,
    hadm_id: int,
    latency_ms: float,
    source: str,
    cost_logs_dir: Path = COST_LOGS_DIR,
) -> None:
    """source: 'live' (a Patient Explorer click) or 'benchmark'
    (run_cost_benchmark.py). Appends one line to cost_logs/{model}.jsonl.
    """
    cost_logs_dir = Path(cost_logs_dir)
    cost_logs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hadm_id": int(hadm_id),
        "source": source,
        "latency_ms": float(latency_ms),
        "estimated_cost_usd": estimate_cost_usd(latency_ms),
    }
    with (cost_logs_dir / f"{model}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def load_cost_logs(cost_logs_dir: Path = COST_LOGS_DIR) -> pd.DataFrame:
    """Concatenate every cost_logs/*.jsonl into one DataFrame. Returns an
    empty DataFrame with COLUMNS (not an error) if nothing has been logged
    yet -- the fresh-checkout / benchmark-not-run-yet state is expected.
    """
    cost_logs_dir = Path(cost_logs_dir)
    files = sorted(cost_logs_dir.glob("*.jsonl"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
