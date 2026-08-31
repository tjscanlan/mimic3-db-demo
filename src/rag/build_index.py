"""Builds the k-NN retrieval index: embeds all notes with sentence-transformers
and persists to models/rag_knn/.

The ONE file in this feature that imports sentence-transformers/torch -- see
app.py's torch/xgboost same-process segfault note (src/models/train_slm.py
history). Unlike src/nlp/predict.py, nothing else in this codebase imports
this module, so there's no same-process conflict to manage even at import
time -- it's a standalone offline CLI, structurally like
data/download_mimic_demo.py.

Usage:
    uv run python -m src.rag.build_index
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.nlp.preprocess import load_notes_with_labels

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_INDEX_DIR = Path("models/rag_knn")


def build_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(texts, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32
    )


def save_index(embeddings: np.ndarray, hadm_ids: list[int], model_name: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "hadm_ids.npy", np.array(hadm_ids, dtype=np.int64))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "dim": int(embeddings.shape[1]),
                "n_notes": int(embeddings.shape[0]),
                "built_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path("data/raw"), type=Path)
    parser.add_argument("--notes-path", default=Path("data/mock/NOTEEVENTS.csv"), type=Path)
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--out-dir", default=RAG_INDEX_DIR, type=Path)
    args = parser.parse_args()

    # load_notes_with_labels() returns readmit_30d too, but it is deliberately
    # NOT persisted below -- labels are always freshly joined from
    # compute_label()/build_feature_table() at use time (single source of truth).
    df = load_notes_with_labels(args.data_dir, args.notes_path)
    log.info("embedding %d notes with %s", len(df), args.model_name)
    embeddings = build_embeddings(df["text"].tolist(), args.model_name)
    saved_dir = save_index(embeddings, df["hadm_id"].tolist(), args.model_name, args.out_dir)
    log.info("saved index (%d x %d) to %s", *embeddings.shape, saved_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
