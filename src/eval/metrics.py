"""Subgroup, calibration, and cross-run drift metrics on top of eval_logs/.

SAMPLE SIZE WARNING: the MIMIC-III demo cohort is 129 admissions / 11
positives (8.5%) total, split further by 5-fold CV and by subgroup. Every
number this module produces below the whole-cohort level has single- or
low-double-digit n. Treat all subgroup/calibration/drift output as
illustrative of the harness working, not as a statistically defensible
finding.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.features.build_dataset import build_feature_table

# Concrete axes, chosen empirically against the real 129-row cohort (not
# hardcoded blind): gender_male (70/59 split), age >=65 (92/37, a standard
# readmission-risk cutoff), admission_type_EMERGENCY (119/10 -- kept for
# clinical relevance despite the severe imbalance, flagged above).
SUBGROUP_AXES = {
    "gender": lambda df: df["gender_male"].map({1: "male", 0: "female"}),
    "age_65_plus": lambda df: (df["age_years"] >= 65).map({True: "65_plus", False: "under_65"}),
    "admission_type_emergency": lambda df: df["admission_type_EMERGENCY"].map(
        {1: "emergency", 0: "non_emergency"}
    ),
}


def _safe_auc(fn, y_true: pd.Series, y_score: pd.Series) -> float:
    """NaN (not a crash) when a subgroup slice has only one class present."""
    return fn(y_true, y_score) if y_true.nunique() > 1 else float("nan")


def subgroup_breakdown(logs: pd.DataFrame, data_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Per model x subgroup-axis x subgroup-value: n, accuracy, roc_auc, pr_auc.

    Uses only each model's most recent logged run (same convention as
    calibration_summary) -- pooling across runs would double-count rows
    once a model has 2+ runs logged, since every run covers the same full
    cohort.

    Joins logs back to build_feature_table() via hadm_id -- works for both
    models uniformly since hadm_id is the shared cohort key.
    """
    latest_run_id = logs.groupby("model")["run_id"].transform("max")
    logs = logs[logs["run_id"] == latest_run_id]

    features = build_feature_table(data_dir)
    merged = logs.merge(features, on="hadm_id", how="left")

    rows = []
    for axis_name, axis_fn in SUBGROUP_AXES.items():
        merged[axis_name] = axis_fn(merged)
        for (model, value), g in merged.groupby(["model", axis_name]):
            rows.append(
                {
                    "model": model,
                    "axis": axis_name,
                    "value": value,
                    "n": len(g),
                    "accuracy": accuracy_score(g["ground_truth"], g["prediction"]),
                    "roc_auc": _safe_auc(roc_auc_score, g["ground_truth"], g["confidence"]),
                    "pr_auc": _safe_auc(average_precision_score, g["ground_truth"], g["confidence"]),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(
    logs: pd.DataFrame, model: str, run_id: str | None = None, n_bins: int = 5
) -> dict:
    """Reliability-diagram data + Brier score for one model.

    Defaults to that model's most recent run (run_id=None).
    """
    sub = logs[logs["model"] == model]
    if run_id is None:
        run_id = sorted(sub["run_id"].unique())[-1]
    sub = sub[sub["run_id"] == run_id]

    prob_true, prob_pred = calibration_curve(sub["ground_truth"], sub["confidence"], n_bins=n_bins)
    return {
        "model": model,
        "run_id": run_id,
        "n": len(sub),
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": brier_score_loss(sub["ground_truth"], sub["confidence"]),
    }


def drift_analysis(logs: pd.DataFrame, model: str) -> dict:
    """Compare the most recent run of `model` against its immediately
    preceding run. Drift here means run-over-run on this static dataset
    (CV-fold-reshuffle/seed variation), not production/temporal drift --
    MIMIC-III's per-patient date de-identification shift rules out
    admission-date as a shared time axis.

    Returns {"status": "insufficient_history", "n_runs": int, "message": str}
    if fewer than 2 runs of `model` are logged -- expected on a fresh
    checkout or after only one run, not an error.
    """
    sub = logs[logs["model"] == model]
    run_ids = sorted(sub["run_id"].unique())
    if len(run_ids) < 2:
        return {
            "status": "insufficient_history",
            "n_runs": len(run_ids),
            "message": f"Only {len(run_ids)} run(s) logged for '{model}'; need >=2 to compute drift.",
        }

    prev_id, latest_id = run_ids[-2], run_ids[-1]
    prev, latest = sub[sub["run_id"] == prev_id], sub[sub["run_id"] == latest_id]
    ks_stat, ks_pvalue = ks_2samp(prev["confidence"], latest["confidence"])

    return {
        "status": "ok",
        "model": model,
        "previous_run_id": prev_id,
        "latest_run_id": latest_id,
        "previous_n": len(prev),
        "latest_n": len(latest),
        "mean_confidence_previous": prev["confidence"].mean(),
        "mean_confidence_latest": latest["confidence"].mean(),
        "mean_confidence_shift": latest["confidence"].mean() - prev["confidence"].mean(),
        "roc_auc_previous": roc_auc_score(prev["ground_truth"], prev["confidence"]),
        "roc_auc_latest": roc_auc_score(latest["ground_truth"], latest["confidence"]),
        "pr_auc_previous": average_precision_score(prev["ground_truth"], prev["confidence"]),
        "pr_auc_latest": average_precision_score(latest["ground_truth"], latest["confidence"]),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
    }
