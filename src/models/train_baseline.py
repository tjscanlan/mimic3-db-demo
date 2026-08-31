"""Baseline XGBoost model for 30-day readmission, evaluated via stratified k-fold CV.

Usage:
    uv run python -m src.models.train_baseline
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.logger import log_predictions
from src.features.build_dataset import NON_FEATURE_COLS, build_feature_table

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EVAL_MODEL_ID = "structured_xgboost"


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
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    hadm_ids: pd.Series | None = None,
    return_predictions: bool = False,
) -> (
    tuple[pd.DataFrame, list[xgb.XGBClassifier]]
    | tuple[pd.DataFrame, list[xgb.XGBClassifier], pd.DataFrame]
):
    if return_predictions and hadm_ids is None:
        raise ValueError("hadm_ids is required when return_predictions=True")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    pred_rows = []
    models = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        n_pos_train = y_train.sum()
        scale_pos_weight = (len(y_train) - n_pos_train) / n_pos_train

        model = make_model(seed, scale_pos_weight)
        model.fit(X_train, y_train)
        models.append(model)

        # Timed as one batched predict_proba() call -- no true single-example
        # latency is available; wall time is divided evenly across this
        # fold's test rows as an approximation (documented, not hidden).
        t0 = time.perf_counter()
        y_score = model.predict_proba(X_test)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000 / len(X_test)

        rows.append(
            {
                "fold": fold,
                "n_test": len(y_test),
                "n_pos_test": int(y_test.sum()),
                "roc_auc": roc_auc_score(y_test, y_score),
                "pr_auc": average_precision_score(y_test, y_score),
            }
        )
        if return_predictions:
            for hid, yt, yp in zip(hadm_ids.iloc[test_idx].to_numpy(), y_test.to_numpy(), y_score):
                pred_rows.append(
                    {"fold": fold, "hadm_id": int(hid), "y_true": int(yt), "y_prob": float(yp), "latency_ms": latency_ms}
                )

    if return_predictions:
        return pd.DataFrame(rows), models, pd.DataFrame(pred_rows)
    return pd.DataFrame(rows), models


def train_final_model(
    data_dir: Path = Path("data/raw"), seed: int = 42
) -> tuple[xgb.XGBClassifier, list[str]]:
    """Fit one model on the FULL cohort (no CV split) for live inference/serving."""
    df = build_feature_table(data_dir)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["readmit_30d"]

    n_pos = y.sum()
    scale_pos_weight = (len(y) - n_pos) / n_pos
    model = make_model(seed, scale_pos_weight)
    model.fit(X, y)
    return model, feature_cols


def save_final_model(model: xgb.XGBClassifier, feature_cols: list[str], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.json"
    model.save_model(model_path)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--json-out", default=None, type=Path,
        help="Optional path to write the per-fold report as JSON (for cross-process callers).",
    )
    parser.add_argument(
        "--save-model-dir", default=None, type=Path,
        help=f"Optional: also fit a final model on the FULL cohort (no CV split) and save it "
             f"here for live inference, e.g. models/{EVAL_MODEL_ID}",
    )
    args = parser.parse_args()

    df = build_feature_table(args.data_dir)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X, y = df[feature_cols], df["readmit_30d"]

    log.info(
        "cohort: %d admissions, %d features, %d positives (%.1f%%)",
        len(df), len(feature_cols), y.sum(), 100 * y.mean(),
    )

    report, _, predictions = run_cv(
        X, y, n_splits=args.n_splits, seed=args.seed, hadm_ids=df["hadm_id"], return_predictions=True
    )
    log.info("\n%s", report.to_string(index=False))
    log.info(
        "\nroc_auc  mean=%.3f std=%.3f\npr_auc   mean=%.3f std=%.3f",
        report["roc_auc"].mean(), report["roc_auc"].std(),
        report["pr_auc"].mean(), report["pr_auc"].std(),
    )
    if args.json_out:
        report.to_json(args.json_out, orient="records")

    log_path = log_predictions(predictions, model=EVAL_MODEL_ID)
    log.info("logged %d predictions to %s", len(predictions), log_path)

    if args.save_model_dir:
        final_model, final_feature_cols = train_final_model(args.data_dir, seed=args.seed)
        saved_path = save_final_model(final_model, final_feature_cols, args.save_model_dir)
        log.info("saved final model to %s", saved_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
