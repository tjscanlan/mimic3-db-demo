"""Generate SYNTHETIC discharge-summary text to stand in for NOTEEVENTS.

The MIMIC-III demo ships NOTEEVENTS empty (see notebooks/01_eda.ipynb) — real
note text only exists in the full credentialed MIMIC-III database. This
script fabricates plausible-looking but entirely synthetic discharge
summaries from the real structured data (diagnoses, demographics, LOS) so
the NLP pipeline (step 3) can be built and tested before/without that access.

THIS IS NOT REAL CLINICAL DATA. Every generated note is prefixed with an
explicit synthetic-data marker, and the output is never committed to git
(see .gitignore) or treated as ground truth for anything beyond pipeline
development.

The generated text carries a deliberate, mild, artificial correlation with
the readmit_30d label (readmitted admissions lean toward "high risk" stock
phrasing) so a classifier trained on it has something learnable — this is
circular by construction and must never be reported as a real finding. It
exists solely to prove the fine-tuning pipeline runs end-to-end.

Usage:
    uv run python data/generate_mock_noteevents.py
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.build_dataset import compute_label, load_raw_tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SYNTHETIC_MARKER = (
    "[SYNTHETIC NOTE -- generated for pipeline development, NOT real clinical text]"
)

HIGH_RISK_PHRASES = [
    "Multiple comorbidities remain incompletely controlled at discharge; "
    "close outpatient follow-up was stressed given elevated risk of early return.",
    "Discharge planning was complicated by unresolved clinical issues; "
    "the care team flagged this patient for early follow-up given readmission risk.",
    "Patient and family were counseled extensively on warning signs, "
    "given a complex hospital course and residual instability.",
]
LOW_RISK_PHRASES = [
    "Patient tolerated the hospital course well and was discharged in stable condition.",
    "No acute issues were identified at discharge; routine follow-up was arranged.",
    "Patient was discharged home in good condition with standard follow-up instructions.",
]

# P(high-risk phrasing) given the label -- mild, not perfect separation, so a
# classifier trained on this has signal to find without the task being trivial.
P_HIGH_RISK_GIVEN_LABEL = {1: 0.75, 0: 0.25}


def build_note_text(row: pd.Series, diagnosis_titles: list[str], rng: random.Random) -> str:
    gender_word = "male" if row["gender"] == "M" else "female"
    age = int(min(row["age_years"], 90))
    primary_dx = diagnosis_titles[0] if diagnosis_titles else "an unspecified condition"
    other_dx = diagnosis_titles[1:4]

    parts = [
        SYNTHETIC_MARKER,
        f"Patient is a {age}-year-old {gender_word} admitted with {primary_dx}.",
    ]
    if other_dx:
        parts.append("Additional diagnoses noted during this admission: " + ", ".join(other_dx) + ".")

    parts.append(f"Length of stay was approximately {row['los_days']:.0f} day(s).")
    if row["had_icu_stay"]:
        parts.append("Course included an ICU stay.")

    risk_pool = (
        HIGH_RISK_PHRASES
        if rng.random() < P_HIGH_RISK_GIVEN_LABEL[row["readmit_30d"]]
        else LOW_RISK_PHRASES
    )
    parts.append(rng.choice(risk_pool))
    parts.append("Follow-up care was arranged as clinically indicated.")
    return " ".join(parts)


def generate_mock_noteevents(data_dir: Path, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    tables = load_raw_tables(data_dir)
    admissions, patients, diagnoses = tables["admissions"], tables["patients"], tables["diagnoses"]

    dx_titles = pd.read_csv(data_dir / "D_ICD_DIAGNOSES.csv", dtype={"icd9_code": str})
    dx_with_titles = diagnoses.merge(dx_titles[["icd9_code", "short_title"]], on="icd9_code", how="left")
    dx_by_hadm = (
        dx_with_titles.sort_values(["hadm_id", "seq_num"])
        .groupby("hadm_id")["short_title"]
        .apply(lambda s: [t for t in s if pd.notna(t)])
    )

    label = compute_label(admissions)
    base = admissions[["hadm_id", "subject_id", "admittime", "dischtime"]].merge(
        patients[["subject_id", "gender", "dob"]], on="subject_id", how="left"
    ).merge(label, on="hadm_id", how="left")
    base["age_years"] = (base["admittime"] - base["dob"]).dt.days / 365.25
    base["los_days"] = (base["dischtime"] - base["admittime"]).dt.total_seconds() / 86400
    base["had_icu_stay"] = base["hadm_id"].isin(tables["icustays"]["hadm_id"])

    rows = []
    for i, row in base.iterrows():
        titles = dx_by_hadm.get(row["hadm_id"], [])
        text = build_note_text(row, titles, rng)
        rows.append(
            {
                "row_id": i + 1,
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "chartdate": row["dischtime"],
                "charttime": row["dischtime"],
                "storetime": row["dischtime"],
                "category": "Discharge summary (SYNTHETIC)",
                "description": "Report",
                "cgid": "",
                "iserror": "",
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path("data/raw"), type=Path)
    parser.add_argument("--out-path", default=Path("data/mock/NOTEEVENTS.csv"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    df = generate_mock_noteevents(args.data_dir, seed=args.seed)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)
    log.info("wrote %d synthetic notes to %s", len(df), args.out_path)
    log.info("this is SYNTHETIC data -- never treat it as real clinical text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
