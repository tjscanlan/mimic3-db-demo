"""Torch Dataset wrapper for tokenized note text.

Split out of preprocess.py so that module can stay torch-free -- app.py
needs load_notes_with_labels()/clean_text() without pulling torch into its
process (xgboost + torch in one process segfaults, see train_slm.py).
"""

import torch
from torch.utils.data import Dataset


class NoteDataset(Dataset):
    """Eagerly tokenizes text; returns a Trainer-compatible dict per item."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
