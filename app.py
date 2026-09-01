"""Gradio demo: 30-day readmission prediction, structured vs. text model.

Deliberately never imports torch/transformers/peft in this process --
xgboost (Homebrew libomp) + torch (bundled OpenMP) in one process segfaults
(confirmed via macOS crash reports, see src/models/train_slm.py history).
Text-model predictions are served by shelling out to
`uv run python -m src.nlp.predict` as a fresh subprocess per request.
Model loading and single-admission prediction (with latency measurement)
live in src/serving.py, shared with src/eval/run_cost_benchmark.py.

Usage:
    uv run python app.py
"""

import random
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

from src.eval.cost_dashboard import build_dashboard_data
from src.eval.cost_logger import log_cost_sample
from src.features.patient_summary import build_patient_summary_table
from src.nlp.preprocess import load_notes_with_labels
from src.serving import (
    RAG_INDEX_DIR,
    RAG_K,
    STRUCTURED_MODEL_DIR,
    TEXT_ADAPTER_DIR,
    load_all_models,
    predict_rag,
    predict_structured,
    predict_text,
)


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
    if not (RAG_INDEX_DIR / "meta.json").exists():
        missing.append("uv run python -m src.rag.build_index")
    if missing:
        print("Missing prerequisite(s). Run, in order:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)


check_prerequisites()

models = load_all_models()
features_df = models.features_df
summary_df = build_patient_summary_table().set_index("hadm_id")
notes_df = load_notes_with_labels().set_index("hadm_id")

HADM_IDS = sorted(features_df.index.tolist())


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

    structured_prob, structured_latency_ms = predict_structured(models, hadm_id)
    log_cost_sample("structured_xgboost", hadm_id, structured_latency_ms, source="live")
    structured_label = {"Readmit <=30d": structured_prob, "No readmit": 1 - structured_prob}

    rag_prob, neighbors_df, rag_latency_ms = predict_rag(models, hadm_id, k=RAG_K)
    log_cost_sample("rag_knn", hadm_id, rag_latency_ms, source="live")
    rag_label = {"Readmit <=30d": rag_prob, "No readmit": 1 - rag_prob}
    neighbors_df = neighbors_df.rename(columns={"readmit_30d": "readmitted_30d"})
    neighbors_df["similarity"] = neighbors_df["similarity"].round(3)

    yield (
        summary_md,
        note_text,
        ground_truth_md,
        structured_label,
        gr.update(value=None, label="Text model (running -- subprocess cold start, a few seconds)..."),
        rag_label,
        neighbors_df,
    )

    try:
        text_prob, text_latency_ms = predict_text(hadm_id, TEXT_ADAPTER_DIR)
    except RuntimeError as e:
        raise gr.Error(str(e))
    log_cost_sample("text_distilbert_lora", hadm_id, text_latency_ms, source="live")
    text_label = {"Readmit <=30d": text_prob, "No readmit": 1 - text_prob}
    yield (summary_md, note_text, ground_truth_md, structured_label, text_label, rag_label, neighbors_df)


def refresh_dashboard():
    return build_dashboard_data()


with gr.Blocks(title="MIMIC-III Readmission Demo") as demo:
    gr.Markdown(
        "## 30-Day Readmission Prediction Demo\n"
        "**Research/education portfolio project only -- not a clinical tool.** "
        "Predictions are from models trained on the small (129-admission) MIMIC-III "
        "Clinical Database *Demo*, now compared against a k-NN retrieval classifier "
        "over note embeddings, and should never inform real clinical decisions."
    )

    with gr.Tab("Patient Explorer"):
        with gr.Row():
            dropdown = gr.Dropdown(choices=HADM_IDS, label="Select admission (hadm_id)")
            random_btn = gr.Button("Random admission")

        model_toggle = gr.CheckboxGroup(
            choices=["Structured (XGBoost)", "Text (DistilBERT + LoRA)", "Retrieval k-NN"],
            value=["Structured (XGBoost)", "Text (DistilBERT + LoRA)", "Retrieval k-NN"],
            label="Show/hide model predictions",
        )

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
            with gr.Column(visible=True) as structured_col:
                structured_out = gr.Label(label="Structured model (XGBoost)")
            with gr.Column(visible=True) as text_col:
                text_out = gr.Label(label="Text model (DistilBERT + LoRA)")
            with gr.Column(visible=True) as rag_col:
                rag_out = gr.Label(label="Retrieval k-NN model")
                gr.Markdown(
                    "**Retrieved evidence** -- top-k most similar SYNTHETIC notes "
                    "(excluding this admission). Notes were generated with a "
                    "deliberate label-correlated phrasing pattern (see "
                    "`data/generate_mock_noteevents.py`), so retrieval trivially "
                    "finds notes sharing that injected phrasing -- expected "
                    "pipeline behavior, not a discovered clinical insight."
                )
                rag_evidence_out = gr.Dataframe(
                    label=f"Top-{RAG_K} nearest neighbor notes",
                    headers=["hadm_id", "similarity", "readmitted_30d"],
                )

        outputs = [summary_out, note_out, ground_truth_out, structured_out, text_out, rag_out, rag_evidence_out]
        dropdown.change(run_admission, inputs=dropdown, outputs=outputs)
        random_btn.click(lambda: random.choice(HADM_IDS), outputs=dropdown).then(
            run_admission, inputs=dropdown, outputs=outputs
        )
        demo.load(lambda: random.choice(HADM_IDS), outputs=dropdown).then(
            run_admission, inputs=dropdown, outputs=outputs
        )

        def toggle_columns(selected: list[str]):
            return (
                gr.update(visible="Structured (XGBoost)" in selected),
                gr.update(visible="Text (DistilBERT + LoRA)" in selected),
                gr.update(visible="Retrieval k-NN" in selected),
            )

        model_toggle.change(toggle_columns, inputs=model_toggle, outputs=[structured_col, text_col, rag_col])

    with gr.Tab("Cost / Latency Dashboard"):
        gr.Markdown(
            "**Illustrative only** -- this demo has no billed inference API. "
            "\"Cost\" is latency converted through an assumed reference compute "
            "rate (`src/eval/cost_logger.py`), not real billing data. Seeded by "
            "a one-time benchmark (`uv run python -m src.eval.run_cost_benchmark`) "
            "and grows with every prediction made in the Patient Explorer tab."
        )
        dashboard_refresh_btn = gr.Button("Refresh")
        dashboard_table = gr.Dataframe(label="Model comparison")
        dashboard_plot = gr.Plot(label="Latency vs. accuracy")

        dashboard_refresh_btn.click(refresh_dashboard, outputs=[dashboard_table, dashboard_plot])
        demo.load(refresh_dashboard, outputs=[dashboard_table, dashboard_plot])

if __name__ == "__main__":
    demo.launch()
