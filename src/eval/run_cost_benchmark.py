"""One-time benchmark seeding cost_logs/ with a real latency distribution
per model, so app.py's Cost / Latency Dashboard isn't empty on first launch
(see docs/ADR.md, ADR-021). Every live demo click in the Patient Explorer
tab adds more samples to the same log afterward.

Iterates a sample of admissions through all three model paths and logs
each prediction's latency/cost via src.eval.cost_logger. The text-model
path is the slow one here (subprocess spawn + model load per admission,
see src/serving.py::predict_text) -- --n-samples defaults small enough to
keep this a quick one-time setup step, not a CI-grade sweep.

Usage:
    uv run python -m src.eval.run_cost_benchmark
"""

import argparse
import logging
import random
import sys

from src.eval.cost_logger import log_cost_sample
from src.serving import RAG_K, load_all_models, predict_rag, predict_structured, predict_text

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    models = load_all_models()
    hadm_ids = models.features_df.index.tolist()
    sample = random.Random(args.seed).sample(hadm_ids, min(args.n_samples, len(hadm_ids)))

    for i, hadm_id in enumerate(sample, 1):
        log.info("[%d/%d] hadm_id=%s", i, len(sample), hadm_id)

        _prob, latency_ms = predict_structured(models, hadm_id)
        log_cost_sample("structured_xgboost", hadm_id, latency_ms, source="benchmark")

        _prob, _neighbors, latency_ms = predict_rag(models, hadm_id, k=RAG_K)
        log_cost_sample("rag_knn", hadm_id, latency_ms, source="benchmark")

        _prob, latency_ms = predict_text(hadm_id)
        log_cost_sample("text_distilbert_lora", hadm_id, latency_ms, source="benchmark")

    log.info("done -- %d sample(s) per model logged to cost_logs/", len(sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
