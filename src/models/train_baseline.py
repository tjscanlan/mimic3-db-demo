"""Baseline XGBoost model for 30-day readmission, evaluated via stratified k-fold CV.

Usage:
    uv run python -m src.models.train_baseline
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.build_dataset import NON_FEATURE_COLS, build_feature_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def make_model(seed: int, scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
    )


def run_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, list[xgb.XGBClassifier]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    models = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_pos_train = y_train.sum()
        scale_pos_weight = (len(y_train) - n_pos_train) / n_pos_train

        model = make_model(seed, scale_pos_weight)
        model.fit(X_train, y_train)
        models.append(model)

        y_score = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "fold": fold,
                "n_test": len(y_test),
                "n_pos_test": int(y_test.sum()),
                "roc_auc": roc_auc_score(y_test, y_score),
                "pr_auc": average_precision_score(y_test, y_score),
            }
        )

    return pd.DataFrame(rows), models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--json-out", default=None, type=Path,
        help="Optional path to write the per-fold report as JSON (for cross-process callers).",
    )
    args = parser.parse_args()

    df = build_feature_table(args.data_dir)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["readmit_30d"]

    log.info(
        "cohort: %d admissions, %d features, %d positives (%.1f%%)",
        len(df), len(feature_cols), y.sum(), 100 * y.mean(),
    )

    report, _ = run_cv(X, y, n_splits=args.n_splits, seed=args.seed)
    log.info("\n%s", report.to_string(index=False))
    log.info(
        "\nroc_auc  mean=%.3f std=%.3f\npr_auc   mean=%.3f std=%.3f",
        report["roc_auc"].mean(), report["roc_auc"].std(),
        report["pr_auc"].mean(), report["pr_auc"].std(),
    )
    if args.json_out:
        report.to_json(args.json_out, orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
