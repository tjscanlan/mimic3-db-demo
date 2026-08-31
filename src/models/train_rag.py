"""k-NN retrieval-as-classifier over precomputed note embeddings, evaluated
via the same stratified k-fold CV as train_baseline.py/train_slm.py.

No make_model()/train_final_model()/save_final_model() here -- k-NN has no
fitting step beyond the embedding index itself; models/rag_knn/ (built by
`uv run python -m src.rag.build_index`) IS the deployable artifact. This is
a deliberate design choice, not a missing feature.

Usage:
    uv run python -m src.models.train_rag
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.eval.logger import log_predictions
from src.features.build_dataset import build_feature_table
from src.rag.knn import RAG_INDEX_DIR, knn_predict, load_index

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EVAL_MODEL_ID = "rag_knn"


def run_cv(
    embeddings: np.ndarray,
    hadm_ids: np.ndarray,
    labels: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    k: int = 5,
    return_predictions: bool = False,
) -> tuple[pd.DataFrame, None] | tuple[pd.DataFrame, None, pd.DataFrame]:
    y = labels.loc[hadm_ids].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(hadm_ids, y), start=1):
        train_pool = hadm_ids[train_idx]
        test_hadm_ids = hadm_ids[test_idx]
        test_labels = labels.loc[test_hadm_ids].to_numpy()

        t0 = time.perf_counter()
        probs = []
        for hid in test_hadm_ids:
            prob, _ = knn_predict(int(hid), embeddings, hadm_ids, labels, k=k, pool_hadm_ids=train_pool)
            probs.append(prob)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(test_hadm_ids)
        probs = np.array(probs)

        rows.append(
            {
                "fold": fold,
                "n_test": len(test_labels),
                "n_pos_test": int(test_labels.sum()),
                "roc_auc": roc_auc_score(test_labels, probs),
                "pr_auc": average_precision_score(test_labels, probs),
            }
        )
        if return_predictions:
            for hid, yt, yp in zip(test_hadm_ids, test_labels, probs):
                pred_rows.append(
                    {
                        "fold": fold,
                        "hadm_id": int(hid),
                        "y_true": int(yt),
                        "y_prob": float(yp),
                        "latency_ms": latency_ms,
                    }
                )

    if return_predictions:
        return pd.DataFrame(rows), None, pd.DataFrame(pred_rows)
    return pd.DataFrame(rows), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--index-dir", default=RAG_INDEX_DIR, type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument(
        "--json-out", default=None, type=Path,
        help="Optional path to write the per-fold report as JSON (for cross-process callers).",
    )
    args = parser.parse_args()

    if not (args.index_dir / "meta.json").exists():
        log.error(
            "no RAG index at %s; run: uv run python -m src.rag.build_index --out-dir %s",
            args.index_dir, args.index_dir,
        )
        return 1

    index = load_index(args.index_dir)
    labels = build_feature_table(args.data_dir).set_index("hadm_id")["readmit_30d"]

    n_pos = labels.loc[index.hadm_ids].sum()
    log.info(
        "cohort: %d admissions, %d positives (%.1f%%)",
        len(index.hadm_ids), n_pos, 100 * n_pos / len(index.hadm_ids),
    )

    report, _, predictions = run_cv(
        index.embeddings, index.hadm_ids, labels, n_splits=args.n_splits, seed=args.seed,
        k=args.k, return_predictions=True,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
