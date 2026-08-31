"""Pure numpy/pandas k-NN retrieval-as-classifier over precomputed note
embeddings.

NEVER import torch/sentence_transformers here -- safe to import directly
into app.py's process; unlike the text-model path this is pure numpy math
at inference time, no subprocess isolation needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAG_INDEX_DIR = Path("models/rag_knn")


@dataclass
class RagIndex:
    embeddings: np.ndarray  # (n, dim), L2-normalized
    hadm_ids: np.ndarray  # (n,) int64, same row order as embeddings
    meta: dict


def load_index(index_dir: Path = RAG_INDEX_DIR) -> RagIndex:
    index_dir = Path(index_dir)
    return RagIndex(
        embeddings=np.load(index_dir / "embeddings.npy"),
        hadm_ids=np.load(index_dir / "hadm_ids.npy"),
        meta=json.loads((index_dir / "meta.json").read_text()),
    )


def knn_predict(
    query_hadm_id: int,
    embeddings: np.ndarray,
    hadm_ids: np.ndarray,
    labels: pd.Series,
    k: int = 5,
    pool_hadm_ids: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Similarity-weighted k-NN readmission probability + evidence.

    labels: indexed by hadm_id -> readmit_30d.
    pool_hadm_ids: restrict the neighbor pool to these admissions (used by
    CV to prevent test-fold members from serving as each other's
    neighbors); default is the whole corpus minus the query itself.

    Returns (probability, neighbors_df) where neighbors_df has columns
    hadm_id, similarity, readmit_30d, sorted by similarity descending.

    Cosine similarity can be negative; negative weights are clipped to 0
    before weighting (a negative "vote" is nonsensical). If the clipped
    weight sum is ~0 (degenerate case), falls back to an unweighted mean
    of the k neighbor labels instead of dividing by zero.
    """
    idx = np.where(hadm_ids == query_hadm_id)[0]
    if len(idx) == 0:
        raise ValueError(f"hadm_id {query_hadm_id} not found in index")
    query_vec = embeddings[idx[0]]

    pool_mask = hadm_ids != query_hadm_id
    if pool_hadm_ids is not None:
        pool_mask &= np.isin(hadm_ids, pool_hadm_ids)
    pool_embeddings, pool_hadm = embeddings[pool_mask], hadm_ids[pool_mask]

    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)  # defensive re-normalize
    sims = pool_embeddings @ q

    order = np.argsort(-sims)[:k]
    top_hadm, top_sims = pool_hadm[order], sims[order]
    top_labels = labels.loc[top_hadm].to_numpy()

    weights = np.clip(top_sims, 0, None)
    weight_sum = weights.sum()
    prob = (
        float(top_labels.mean())
        if weight_sum < 1e-9
        else float((weights * top_labels).sum() / weight_sum)
    )

    neighbors_df = pd.DataFrame(
        {
            "hadm_id": top_hadm.astype(int),
            "similarity": top_sims,
            "readmit_30d": top_labels.astype(int),
        }
    )
    return prob, neighbors_df
