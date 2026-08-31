"""Gradio demo: 30-day readmission prediction, structured vs. text model.

Deliberately never imports torch/transformers/peft in this process --
xgboost (Homebrew libomp) + torch (bundled OpenMP) in one process segfaults
(confirmed via macOS crash reports, see src/models/train_slm.py history).
Text-model predictions are served by shelling out to
`uv run python -m src.nlp.predict` as a fresh subprocess per request.

Usage:
    uv run python app.py
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import xgboost as xgb

from src.features.build_dataset import build_feature_table
from src.features.patient_summary import build_patient_summary_table
from src.nlp.preprocess import load_notes_with_labels

MODELS_DIR = Path("models")
STRUCTURED_MODEL_ID = "structured_xgboost"
STRUCTURED_MODEL_DIR = MODELS_DIR / STRUCTURED_MODEL_ID
# Mirrors src/models/train_slm.py's EVAL_MODEL_ID by convention only -- that
# module cannot be imported here, it pulls in torch at module load.
TEXT_MODEL_ID = "text_distilbert_lora"
TEXT_ADAPTER_DIR = MODELS_DIR / TEXT_MODEL_ID


def check_prerequisites() -> None:
    missing = []
    if not Path("data/raw/ADMISSIONS.csv").exists():
        missing.append("uv run python data/download_mimic_demo.py")
    if not Path("data/mock/NOTEEVENTS.csv").exists():
        missing.append("uv run python data/generate_mock_noteevents.py")
    if not (STRUCTURED_MODEL_DIR / "model.json").exists():
        missing.append(f"uv run python -m src.models.train_baseline --save-model-dir {STRUCTURED_MODEL_DIR}")
    if not (TEXT_ADAPTER_DIR / "adapter_config.json").exists():
        missing.append(f"uv run python -m src.models.train_slm --save-adapter-dir {TEXT_ADAPTER_DIR}")
    if missing:
        print("Missing prerequisite(s). Run, in order:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)


check_prerequisites()

features_df = build_feature_table().set_index("hadm_id", drop=False)
summary_df = build_patient_summary_table().set_index("hadm_id")
notes_df = load_notes_with_labels().set_index("hadm_id")

meta = json.loads((STRUCTURED_MODEL_DIR / "meta.json").read_text())
feature_cols = meta["feature_cols"]

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(STRUCTURED_MODEL_DIR / "model.json")

HADM_IDS = sorted(features_df.index.tolist())


def predict_text_confidence(hadm_id: int, timeout: int = 60) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "pred.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "src.nlp.predict",
                "--hadm-id", str(hadm_id),
                "--adapter-dir", str(TEXT_ADAPTER_DIR),
                "--json-out", str(out_path),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        payload = json.loads(out_path.read_text()) if out_path.exists() else {"error": result.stderr.strip()}
    if "error" in payload:
        raise gr.Error(f"Text-model prediction failed: {payload['error']}")
    return payload["confidence"]


def run_admission(hadm_id: int):
    summary = summary_df.loc[hadm_id]
    note_text = notes_df.loc[hadm_id, "text"]
    ground_truth = int(features_df.loc[hadm_id, "readmit_30d"])

    marital_status = summary.marital_status if pd.notna(summary.marital_status) else "Unknown"
    summary_md = (
        f"**Age:** {summary.age_years:.0f}  **Gender:** {summary.gender}  "
        f"**Admission type:** {summary.admission_type}  **Insurance:** {summary.insurance}  "
        f"**Marital status:** {marital_status}  **LOS:** {summary.los_days:.1f} days\n\n"
        f"**Primary diagnosis:** {summary.primary_diagnosis}\n\n"
        f"**Other diagnoses:** {summary.other_diagnoses}"
    )
    ground_truth_md = f"**Ground truth -- readmitted within 30 days:** {'YES' if ground_truth else 'NO'}"

    X_row = features_df.loc[[hadm_id], feature_cols]
    structured_prob = float(xgb_model.predict_proba(X_row)[0, 1])
    structured_label = {"Readmit <=30d": structured_prob, "No readmit": 1 - structured_prob}

    yield (
        summary_md,
        note_text,
        ground_truth_md,
        structured_label,
        gr.update(value=None, label="Text model (running -- subprocess cold start, a few seconds)..."),
    )

    text_prob = predict_text_confidence(hadm_id)
    text_label = {"Readmit <=30d": text_prob, "No readmit": 1 - text_prob}
    yield (summary_md, note_text, ground_truth_md, structured_label, text_label)


with gr.Blocks(title="MIMIC-III Readmission Demo") as demo:
    gr.Markdown(
        "## 30-Day Readmission Prediction Demo\n"
        "**Research/education portfolio project only -- not a clinical tool.** "
        "Predictions are from models trained on the small (129-admission) MIMIC-III "
        "Clinical Database *Demo* and should never inform real clinical decisions."
    )
    with gr.Row():
        dropdown = gr.Dropdown(choices=HADM_IDS, label="Select admission (hadm_id)")
        random_btn = gr.Button("Random admission")

    with gr.Row():
        summary_out = gr.Markdown(label="Patient / admission summary")
        with gr.Column():
            gr.Markdown(
                "**Note:** this text is SYNTHETIC, generated for pipeline "
                "development (see `data/generate_mock_noteevents.py`) -- it is "
                "NOT a real clinical note and carries a deliberate artificial "
                "correlation with the label. Treat the text model's result as a "
                "pipeline demonstration, not a real text-based finding."
            )
            note_out = gr.Textbox(label="Discharge note (synthetic)", lines=8, interactive=False)

    ground_truth_out = gr.Markdown()
    with gr.Row():
        structured_out = gr.Label(label="Structured model (XGBoost)")
        text_out = gr.Label(label="Text model (DistilBERT + LoRA)")

    outputs = [summary_out, note_out, ground_truth_out, structured_out, text_out]
    dropdown.change(run_admission, inputs=dropdown, outputs=outputs)
    random_btn.click(lambda: random.choice(HADM_IDS), outputs=dropdown).then(
        run_admission, inputs=dropdown, outputs=outputs
    )
    demo.load(lambda: random.choice(HADM_IDS), outputs=dropdown).then(
        run_admission, inputs=dropdown, outputs=outputs
    )

if __name__ == "__main__":
    demo.launch()
