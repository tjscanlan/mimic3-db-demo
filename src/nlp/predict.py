"""Standalone CLI for live text-model inference via a saved LoRA adapter.

Always run this as a fresh subprocess (see app.py) -- importing
torch/transformers/peft into the same process as xgboost segfaults
(confirmed: xgboost's Homebrew libomp + torch's bundled OpenMP conflict).

Usage:
    uv run python -m src.nlp.predict --hadm-id 100375 --json-out /tmp/out.json
    uv run python -m src.nlp.predict --text "some note text" --json-out /tmp/out.json
"""

import argparse
import json
import sys
from pathlib import Path


def predict_confidence(text: str, adapter_dir: Path, max_length: int = 128) -> float:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.models.train_slm import MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0, 1].item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hadm-id", type=int)
    group.add_argument("--text", type=str)
    parser.add_argument("--adapter-dir", default=Path("models/text_distilbert_lora"), type=Path)
    parser.add_argument("--data-dir", default=Path("data/raw"), type=Path)
    parser.add_argument("--notes-path", default=Path("data/mock/NOTEEVENTS.csv"), type=Path)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument(
        "--json-out", default=None, type=Path,
        help="Write {'confidence': float} JSON here instead of stdout "
             "(robust to any stray library stdout noise).",
    )
    args = parser.parse_args()

    if not (args.adapter_dir / "adapter_config.json").exists():
        result = {
            "error": f"no adapter at {args.adapter_dir}; run: "
                     f"uv run python -m src.models.train_slm --save-adapter-dir {args.adapter_dir}"
        }
    elif args.hadm_id is not None:
        from src.nlp.preprocess import load_notes_with_labels

        df = load_notes_with_labels(args.data_dir, args.notes_path)
        row = df[df["hadm_id"] == args.hadm_id]
        if row.empty:
            result = {"error": f"hadm_id {args.hadm_id} not found in {args.notes_path}"}
        else:
            confidence = predict_confidence(row.iloc[0]["text"], args.adapter_dir, args.max_length)
            result = {"confidence": confidence}
    else:
        from src.nlp.preprocess import clean_text

        confidence = predict_confidence(clean_text(args.text), args.adapter_dir, args.max_length)
        result = {"confidence": confidence}

    payload = json.dumps(result)
    if args.json_out:
        args.json_out.write_text(payload)
    else:
        print(payload)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
