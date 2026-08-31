"""Build the structured feature table for 30-day readmission prediction.

One row per hadm_id (129 rows for the MIMIC-III demo).
"""

from pathlib import Path

import pandas as pd

TOP_ICD9_CODES = ["4019", "42731", "5849", "4280", "25000", "51881"]

NON_FEATURE_COLS = ["hadm_id", "subject_id", "readmit_30d"]


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "admissions": pd.read_csv(
            data_dir / "ADMISSIONS.csv", parse_dates=["admittime", "dischtime", "deathtime"]
        ),
        "patients": pd.read_csv(data_dir / "PATIENTS.csv", parse_dates=["dob"]),
        "diagnoses": pd.read_csv(data_dir / "DIAGNOSES_ICD.csv", dtype={"icd9_code": str}),
        "labevents": pd.read_csv(data_dir / "LABEVENTS.csv"),
        "icustays": pd.read_csv(data_dir / "ICUSTAYS.csv"),
    }


def compute_label(admissions: pd.DataFrame) -> pd.DataFrame:
    adm = admissions.sort_values(["subject_id", "admittime"]).copy()
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    gap_days = (adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() / 86400
    adm["readmit_30d"] = (gap_days <= 30).fillna(False).astype(int)
    return adm[["hadm_id", "readmit_30d"]]


def compute_demographics(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    merged = admissions[["hadm_id", "subject_id", "admittime"]].merge(
        patients[["subject_id", "gender", "dob"]], on="subject_id", how="left"
    )
    age_years = (merged["admittime"] - merged["dob"]).dt.days / 365.25
    merged["age_years"] = age_years.clip(upper=90)
    merged["gender_male"] = (merged["gender"] == "M").astype(int)
    return merged[["hadm_id", "gender_male", "age_years"]]


def bucket_rare_categories(s: pd.Series, min_count: int) -> pd.Series:
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), "Other")


def compute_admission_categoricals(admissions: pd.DataFrame) -> pd.DataFrame:
    df = admissions[["hadm_id", "admission_type", "insurance", "marital_status"]].copy()
    df["insurance"] = bucket_rare_categories(df["insurance"], min_count=5)
    df["marital_status"] = bucket_rare_categories(
        df["marital_status"].fillna("Unknown"), min_count=10
    )
    dummies = pd.get_dummies(
        df[["admission_type", "insurance", "marital_status"]], drop_first=True, dtype=int
    )
    return pd.concat([df[["hadm_id"]], dummies], axis=1)


def compute_los(admissions: pd.DataFrame) -> pd.DataFrame:
    df = admissions[["hadm_id", "admittime", "dischtime"]].copy()
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    return df[["hadm_id", "los_days"]]


def compute_diagnosis_features(
    diagnoses: pd.DataFrame, top_codes: list[str] = TOP_ICD9_CODES
) -> pd.DataFrame:
    n_diagnoses = diagnoses.groupby("hadm_id").size().rename("n_diagnoses")
    flags = (
        diagnoses[diagnoses["icd9_code"].isin(top_codes)]
        .assign(present=1)
        .pivot_table(index="hadm_id", columns="icd9_code", values="present", fill_value=0)
        .reindex(columns=top_codes, fill_value=0)
        .add_prefix("dx_")
    )
    return pd.concat([n_diagnoses, flags], axis=1).reset_index()


def compute_lab_features(labevents: pd.DataFrame) -> pd.DataFrame:
    labevents = labevents.dropna(subset=["hadm_id"])
    n_labs = labevents.groupby("hadm_id").size().rename("n_labs")
    pct_abnormal_labs = (
        labevents.assign(abnormal_flag=labevents["flag"] == "abnormal")
        .groupby("hadm_id")["abnormal_flag"]
        .mean()
        .rename("pct_abnormal_labs")
    )
    return pd.concat([n_labs, pct_abnormal_labs], axis=1).reset_index()


def compute_icu_features(icustays: pd.DataFrame, admission_spine: pd.DataFrame) -> pd.DataFrame:
    icu_agg = icustays.groupby("hadm_id")["los"].sum().rename("icu_los_total_days")
    df = admission_spine[["hadm_id"]].merge(icu_agg, on="hadm_id", how="left")
    df["icu_los_total_days"] = df["icu_los_total_days"].fillna(0)
    df["had_icu_stay"] = (df["icu_los_total_days"] > 0).astype(int)
    return df[["hadm_id", "had_icu_stay", "icu_los_total_days"]]


def build_feature_table(data_dir: Path = Path("data/raw")) -> pd.DataFrame:
    data_dir = Path(data_dir)
    tables = load_raw_tables(data_dir)
    admissions = tables["admissions"]

    spine = admissions[["hadm_id", "subject_id"]].copy()
    label = compute_label(admissions)
    demographics = compute_demographics(admissions, tables["patients"])
    categoricals = compute_admission_categoricals(admissions)
    los = compute_los(admissions)
    diagnoses = compute_diagnosis_features(tables["diagnoses"])
    labs = compute_lab_features(tables["labevents"])
    icu = compute_icu_features(tables["icustays"], spine)

    df = spine
    for part in [label, demographics, categoricals, los, diagnoses, labs, icu]:
        df = df.merge(part, on="hadm_id", how="left")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df
