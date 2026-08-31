"""Fail loudly if the text/RAG pipelines would ever run on non-synthetic
note text.

This project does not pursue MIMIC-III full-credentialed access (no CITI
certification) -- the text and RAG paths are permanently built on synthetic
notes (see generate_mock_noteevents.py), not a placeholder awaiting real
data. This script is a safety net for that decision, not a TODO: it checks
(1) the real demo download's NOTEEVENTS.csv stayed empty -- i.e. no
credentials somehow got configured and pulled real discharge-summary text
into this environment -- and (2) every row of the synthetic NOTEEVENTS.csv
that the pipelines actually read carries the explicit synthetic-data marker.

Usage:
    uv run python data/verify_synthetic_notes.py
"""

import sys
from pathlib import Path

import pandas as pd

from generate_mock_noteevents import SYNTHETIC_MARKER


def main() -> int:
    raw_path = Path("data/raw/NOTEEVENTS.csv")
    mock_path = Path("data/mock/NOTEEVENTS.csv")
    ok = True

    if not raw_path.exists():
        print(f"missing: {raw_path} (run data/download_mimic_demo.py first)", file=sys.stderr)
        ok = False
    else:
        raw_rows = len(pd.read_csv(raw_path))
        if raw_rows > 0:
            print(
                f"UNEXPECTED: {raw_path} has {raw_rows} data row(s) -- this project "
                "never processes real discharge-summary text. Check that "
                "PHYSIONET_USER/PHYSIONET_PASS and ~/.netrc aren't configured.",
                file=sys.stderr,
            )
            ok = False

    if not mock_path.exists():
        print(f"missing: {mock_path} (run data/generate_mock_noteevents.py first)", file=sys.stderr)
        ok = False
    else:
        mock = pd.read_csv(mock_path)
        if mock.empty:
            print(f"{mock_path} has 0 rows -- synthetic note generation produced nothing", file=sys.stderr)
            ok = False
        else:
            unmarked = ~mock["text"].str.startswith(SYNTHETIC_MARKER)
            if unmarked.any():
                print(
                    f"{unmarked.sum()} row(s) in {mock_path} are missing the synthetic-data marker",
                    file=sys.stderr,
                )
                ok = False
            else:
                print(f"OK: {len(mock)} synthetic notes verified in {mock_path}, all marked non-clinical")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
