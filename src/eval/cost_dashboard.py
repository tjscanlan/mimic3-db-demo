"""Joins cost_logs/ (latency/cost) with eval_logs/ (accuracy) into the
comparison app.py's Cost / Latency Dashboard tab renders -- the "who's
worth it in production" view (see docs/ADR.md, ADR-021).
"""

import matplotlib.figure
import pandas as pd

from src.eval.check_regression import latest_run_metrics
from src.eval.cost_logger import load_cost_logs
from src.eval.logger import load_all_logs

MODEL_DISPLAY_NAMES = {
    "structured_xgboost": "Structured (XGBoost)",
    "text_distilbert_lora": "Text (DistilBERT + LoRA)",
    "rag_knn": "Retrieval k-NN",
}


def build_dashboard_data() -> tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """Returns (summary_df, chart_fig).

    summary_df: one row per model in MODEL_DISPLAY_NAMES, with at least
    n_samples, a measure of typical latency_ms, an estimated cost figure
    (see src.eval.cost_logger.estimate_cost_usd -- note per-prediction
    cost is tiny, e.g. ~1e-8 to 1e-4 USD, so scale it to something legible
    like cost per 1,000 predictions rather than showing raw per-prediction
    dollars), and roc_auc/pr_auc from that model's latest logged CV run
    (src.eval.check_regression.latest_run_metrics).

    chart_fig: a matplotlib figure visualizing latency vs. accuracy across
    the three models, so it's immediately visible whether a slower/costlier
    model is actually buying more accuracy.

    Must not raise if cost_logs/ is empty for a model (benchmark not run
    yet) or eval_logs/ has no run for a model -- render something sensible
    (e.g. "-" / NaN) rather than crashing the dashboard tab.
    """
    # TODO(human): implement.
    raise NotImplementedError
