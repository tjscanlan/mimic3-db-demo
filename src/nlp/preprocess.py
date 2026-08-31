"""Note preprocessing for the fine-tuned SLM readmission classifier.

*** Operates on SYNTHETIC note text only (data/mock/NOTEEVENTS.csv). ***
See data/generate_mock_noteevents.py: notes carry a deliberate, artificial
correlation with the label baked in for pipeline-development purposes.
"""

import re
from pathlib import Path

import pandas as pd

from src.features.build_dataset import compute_label, load_raw_tables

# Strips a leading "[...]" marker (the synthetic-data tag prepended by
# data/generate_mock_noteevents.py) before tokenization -- it's a generation
# artifact with zero discriminative signal, and would waste tokens/context.
_LEADING_BRACKET_MARKER = re.compile(r"^\[[^\]]*\]\s*")


def clean_text(text: str) -> str:
    return _LEADING_BRACKET_MARKER.sub("", text).strip()


def load_notes_with_labels(
    data_dir: Path = Path("data/raw"),
    notes_path: Path = Path("data/mock/NOTEEVENTS.csv"),
) -> pd.DataFrame:
    """One row per hadm_id: hadm_id, subject_id, text (cleaned), readmit_30d."""
    data_dir = Path(data_dir)
    notes = pd.read_csv(notes_path, parse_dates=["charttime"])
    notes = notes.sort_values("charttime").drop_duplicates("hadm_id", keep="first")

    tables = load_raw_tables(data_dir)
    label = compute_label(tables["admissions"])

    df = notes[["hadm_id", "subject_id", "text"]].merge(label, on="hadm_id", how="inner")
    df["text"] = df["text"].map(clean_text)
    return df.reset_index(drop=True)
