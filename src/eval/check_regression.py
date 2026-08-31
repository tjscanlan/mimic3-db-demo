"""Fixed-threshold regression gate over eval_logs/*.jsonl, for CI.

Deliberately does NOT import xgboost or torch/transformers -- it only needs
the logged predictions (ground_truth, confidence columns), not the models
themselves, so it can run as a lightweight final step after all three
training subprocesses without adding a fourth process boundary (see
app.py/src/models/train_slm.py/src/rag/build_index.py for why xgboost and
torch can never share a process).

Usage:
    uv run python -m src.eval.check_regression
Exit code 0 = every blocking model is at or above its threshold.
Exit code 1 = at least one blocking model is below threshold, or a
              blocking model configured in thresholds.json has no logged
              run at all (fail-closed, not silently skipped).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.logger import load_all_logs

DEFAULT_THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"


def latest_run_metrics(logs: pd.DataFrame, model: str) -> dict | None:
    """This model's most recent logged run's roc_auc/pr_auc, or None if
    nothing has been logged for it yet.
    """
    sub = logs[logs["model"] == model]
    if sub.empty:
        return None
    latest_run_id = sub["run_id"].max()
    sub = sub[sub["run_id"] == latest_run_id]
    return {
        "run_id": latest_run_id,
        "n": len(sub),
        "roc_auc": roc_auc_score(sub["ground_truth"], sub["confidence"]),
        "pr_auc": average_precision_score(sub["ground_truth"], sub["confidence"]),
    }


def evaluate_model(model: str, result: dict | None, spec: dict) -> tuple[dict, bool]:
    """Compare one model's latest-run metrics against its threshold spec.

    Returns (row, is_blocking_failure):
      - row: a dict for the report table (model, n, roc_auc, roc_auc_min,
        pr_auc, pr_auc_min, status).
      - is_blocking_failure: True iff this model's outcome should flip the
        script's exit code to 1.
    """
    blocking = spec.get("blocking", True)

    if result is None:
        row = {
            "model": model,
            "n": "-",
            "roc_auc": "-",
            "roc_auc_min": spec["roc_auc_min"],
            "pr_auc": "-",
            "pr_auc_min": spec["pr_auc_min"],
            "status": "FAIL (no logged run)" if blocking else "WARN (no logged run)",
        }
        return row, blocking

    passed = (
        result["roc_auc"] >= spec["roc_auc_min"]
        and result["pr_auc"] >= spec["pr_auc_min"]
    )
    if passed:
        status = "PASS"
    elif blocking:
        status = "FAIL"
    else:
        status = "WARN (non-blocking)"

    row = {
        "model": model,
        "n": result["n"],
        "roc_auc": round(result["roc_auc"], 3),
        "roc_auc_min": spec["roc_auc_min"],
        "pr_auc": round(result["pr_auc"], 3),
        "pr_auc_min": spec["pr_auc_min"],
        "status": status,
    }
    return row, (blocking and not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_PATH, type=Path)
    parser.add_argument("--eval-logs-dir", default=Path("eval_logs"), type=Path)
    args = parser.parse_args()

    config = json.loads(args.thresholds.read_text())
    logs = load_all_logs(args.eval_logs_dir)

    rows = []
    any_blocking_failure = False
    for model, spec in config["models"].items():
        result = latest_run_metrics(logs, model)
        row, is_blocking_failure = evaluate_model(model, result, spec)
        rows.append(row)
        any_blocking_failure = any_blocking_failure or is_blocking_failure

    report = pd.DataFrame(rows)
    table = report.to_markdown(index=False)
    print(table)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Eval regression gate\n\n")
            f.write(table + "\n")

    if any_blocking_failure:
        print("\nFAIL: one or more blocking models regressed below their threshold.")
        return 1
    print("\nPASS: all blocking models at or above threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
