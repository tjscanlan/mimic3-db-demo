"""Fine-tuned DistilBERT + LoRA classifier for 30-day readmission from note
text, evaluated via the same stratified k-fold CV as train_baseline.py.

*** SYNTHETIC note text (data/mock/NOTEEVENTS.csv) only -- see
data/generate_mock_noteevents.py. These numbers are a pipeline sanity check,
NOT a real finding about text-based readmission prediction. ***

Usage:
    uv run python -m src.models.train_slm
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.nlp.preprocess import NoteDataset, load_notes_with_labels

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Picks up HF_TOKEN from .env (if present) into the process environment --
# huggingface_hub reads HF_TOKEN automatically from there for higher rate
# limits/faster downloads (and it's required for any gated model). No other
# wiring needed; must run before any from_pretrained() call.
load_dotenv()

MODEL_NAME = "distilbert-base-uncased"


def make_model(seed: int):
    """Fresh base model + LoRA adapters for one fold; base weights frozen."""
    torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention proj names
        # pre_classifier/classifier are randomly-initialized task heads, not
        # covered by LoRA adapters -- without unfreezing them here the model
        # has no trainable path to the classification decision at all.
        modules_to_save=["pre_classifier", "classifier"],
        bias="none",
    )
    return get_peft_model(base, lora_config)


class WeightedLossTrainer(Trainer):
    """Per-fold class-weighted CrossEntropyLoss, computed from that fold's
    training split only (never test) -- the text-model analogue of
    train_baseline.py's per-fold scale_pos_weight.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def run_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 5,
    max_length: int = 128,
    lr: float = 2e-4,
    batch_size: int = 8,
) -> tuple[pd.DataFrame, list]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts, labels = df["text"].tolist(), df["readmit_30d"].tolist()
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows, models = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y), start=1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        train_ds = NoteDataset(train_texts, train_labels, tokenizer, max_length)
        test_ds = NoteDataset(test_texts, test_labels, tokenizer, max_length)

        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        class_weights = torch.tensor(
            [len(train_labels) / (2 * n_neg), len(train_labels) / (2 * n_pos)],
            dtype=torch.float,
        )

        model = make_model(seed)
        with tempfile.TemporaryDirectory() as out_dir:
            training_args = TrainingArguments(
                output_dir=out_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                logging_strategy="no",
                save_strategy="no",
                report_to=[],
                seed=seed,
            )
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                class_weights=class_weights,
            )
            trainer.train()
            preds = trainer.predict(test_ds)

        models.append(model)
        probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
        rows.append(
            {
                "fold": fold,
                "n_test": len(test_labels),
                "n_pos_test": int(sum(test_labels)),
                "roc_auc": roc_auc_score(test_labels, probs),
                "pr_auc": average_precision_score(test_labels, probs),
            }
        )
        log.info("fold %d: roc_auc=%.3f pr_auc=%.3f", fold, rows[-1]["roc_auc"], rows[-1]["pr_auc"])

    return pd.DataFrame(rows), models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--notes-path", default="data/mock/NOTEEVENTS.csv", type=Path)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()

    log.warning(
        "*** SYNTHETIC note text -- these metrics are a pipeline sanity check, "
        "NOT a real finding. See data/generate_mock_noteevents.py. ***"
    )

    df = load_notes_with_labels(args.data_dir, args.notes_path)
    log.info(
        "cohort: %d admissions with notes, %d positives (%.1f%%)",
        len(df), df["readmit_30d"].sum(), 100 * df["readmit_30d"].mean(),
    )

    report, _ = run_cv(
        df,
        n_splits=args.n_splits,
        seed=args.seed,
        epochs=args.epochs,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    log.info("\n%s", report.to_string(index=False))
    log.info(
        "\nroc_auc  mean=%.3f std=%.3f\npr_auc   mean=%.3f std=%.3f",
        report["roc_auc"].mean(), report["roc_auc"].std(),
        report["pr_auc"].mean(), report["pr_auc"].std(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
