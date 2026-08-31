"""Display-ready (not model-ready) admission summary for the Gradio demo.

Complements build_dataset.py's build_feature_table(), whose output is
one-hot-encoded / binary-flag / not human readable.
"""

from pathlib import Path

import pandas as pd

from src.features.build_dataset import compute_label, load_raw_tables


def build_patient_summary_table(data_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """One row per hadm_id: hadm_id, subject_id, gender, age_years,
    admission_type, insurance, marital_status, los_days,
    primary_diagnosis, other_diagnoses, readmit_30d.
    """
    data_dir = Path(data_dir)
    tables = load_raw_tables(data_dir)
    admissions, patients, diagnoses = tables["admissions"], tables["patients"], tables["diagnoses"]

    dx_titles = pd.read_csv(data_dir / "D_ICD_DIAGNOSES.csv", dtype={"icd9_code": str})
    dx_with_titles = diagnoses.merge(dx_titles[["icd9_code", "short_title"]], on="icd9_code", how="left")
    dx_with_titles["short_title"] = dx_with_titles["short_title"].fillna(
        dx_with_titles["icd9_code"].map(lambda c: f"ICD9 {c}")
    )
    dx_by_hadm = (
        dx_with_titles.sort_values(["hadm_id", "seq_num"])
        .groupby("hadm_id")["short_title"]
        .apply(list)
    )

    label = compute_label(admissions)
    df = (
        admissions[
            ["hadm_id", "subject_id", "admittime", "dischtime", "admission_type", "insurance", "marital_status"]
        ]
        .merge(patients[["subject_id", "gender", "dob"]], on="subject_id", how="left")
        .merge(label, on="hadm_id", how="left")
    )
    df["age_years"] = ((df["admittime"] - df["dob"]).dt.days / 365.25).clip(upper=90)
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    df["primary_diagnosis"] = df["hadm_id"].map(lambda h: (dx_by_hadm.get(h) or ["Unknown"])[0])
    df["other_diagnoses"] = df["hadm_id"].map(
        lambda h: ", ".join((dx_by_hadm.get(h) or [])[1:4]) or "None recorded"
    )

    return df[
        [
            "hadm_id", "subject_id", "gender", "age_years", "admission_type",
            "insurance", "marital_status", "los_days", "primary_diagnosis",
            "other_diagnoses", "readmit_30d",
        ]
    ]
