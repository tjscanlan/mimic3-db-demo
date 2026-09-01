"""Shared single-admission inference across all three model paths, with
wall-clock latency measurement for the cost/latency dashboard (see
docs/ADR.md, ADR-021). Used by both app.py (live demo clicks) and
src/eval/run_cost_benchmark.py (the one-time seed benchmark) so the two
don't duplicate model-loading/prediction logic.

Structured and RAG predictions run in-process here -- safe, no torch import
(see ADR-007/ADR-019 in docs/ADR.md). Text predictions shell out to
`uv run python -m src.nlp.predict` as a subprocess, exactly like app.py
always has; this module only adds latency measurement around that existing
pattern, it doesn't change the isolation itself.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.features.build_dataset import build_feature_table
from src.rag.knn import RagIndex, knn_predict, load_index

MODELS_DIR = Path("models")
STRUCTURED_MODEL_ID = "structured_xgboost"
STRUCTURED_MODEL_DIR = MODELS_DIR / STRUCTURED_MODEL_ID
# Mirrors src/models/train_slm.py's EVAL_MODEL_ID by convention only -- that
# module cannot be imported here, it pulls in torch at module load.
TEXT_MODEL_ID = "text_distilbert_lora"
TEXT_ADAPTER_DIR = MODELS_DIR / TEXT_MODEL_ID
RAG_MODEL_ID = "rag_knn"
RAG_INDEX_DIR = MODELS_DIR / RAG_MODEL_ID
RAG_K = 5


@dataclass
class Models:
    xgb_model: xgb.XGBClassifier
    feature_cols: list[str]
    features_df: pd.DataFrame
    rag_index: RagIndex
    rag_labels: pd.Series


def load_all_models(
    structured_model_dir: Path = STRUCTURED_MODEL_DIR,
    rag_index_dir: Path = RAG_INDEX_DIR,
) -> Models:
    features_df = build_feature_table().set_index("hadm_id", drop=False)
    meta = json.loads((structured_model_dir / "meta.json").read_text())

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(structured_model_dir / "model.json")

    rag_index = load_index(rag_index_dir)

    return Models(
        xgb_model=xgb_model,
        feature_cols=meta["feature_cols"],
        features_df=features_df,
        rag_index=rag_index,
        rag_labels=features_df["readmit_30d"],
    )


def predict_structured(models: Models, hadm_id: int) -> tuple[float, float]:
    """Returns (probability, latency_ms)."""
    X_row = models.features_df.loc[[hadm_id], models.feature_cols]
    start = time.perf_counter()
    prob = float(models.xgb_model.predict_proba(X_row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    return prob, latency_ms


def predict_rag(models: Models, hadm_id: int, k: int = RAG_K) -> tuple[float, pd.DataFrame, float]:
    """Returns (probability, neighbors_df, latency_ms)."""
    start = time.perf_counter()
    prob, neighbors_df = knn_predict(
        hadm_id, models.rag_index.embeddings, models.rag_index.hadm_ids, models.rag_labels, k=k
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return prob, neighbors_df, latency_ms


def predict_text(hadm_id: int, adapter_dir: Path = TEXT_ADAPTER_DIR, timeout: int = 60) -> tuple[float, float]:
    """Returns (probability, latency_ms). Latency includes the full
    subprocess spawn + model-load cost, not just model compute -- that is
    the real production cost of this serving pattern (ADR-009), and is
    deliberately not hidden from the dashboard.
    """
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "pred.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "src.nlp.predict",
                "--hadm-id", str(hadm_id),
                "--adapter-dir", str(adapter_dir),
                "--json-out", str(out_path),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        payload = json.loads(out_path.read_text()) if out_path.exists() else {"error": result.stderr.strip()}
    latency_ms = (time.perf_counter() - start) * 1000

    if "error" in payload:
        raise RuntimeError(f"Text-model prediction failed: {payload['error']}")
    return payload["confidence"], latency_ms
