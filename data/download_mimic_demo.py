"""Download the MIMIC-III Clinical Database Demo from PhysioNet into data/raw/.

The demo (https://physionet.org/content/mimiciii-demo/1.4/) is open access —
no credentialing required, just DUA acceptance on the PhysioNet site. This
script supports optional HTTP basic auth anyway (env vars or ~/.netrc), since
the same pattern is needed for the full credentialed MIMIC-III database.

Usage:
    uv run python data/download_mimic_demo.py
"""

import argparse
import logging
import netrc
import os
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://physionet.org/files/mimiciii-demo/1.4/"

REQUIRED_FILES = [
    "ADMISSIONS.csv",
    "PATIENTS.csv",
    "DIAGNOSES_ICD.csv",
    "LABEVENTS.csv",
    "NOTEEVENTS.csv",
]

DEFAULT_FILES = REQUIRED_FILES + [
    "CALLOUT.csv",
    "CAREGIVERS.csv",
    "CHARTEVENTS.csv",
    "CPTEVENTS.csv",
    "D_CPT.csv",
    "D_ICD_DIAGNOSES.csv",
    "D_ICD_PROCEDURES.csv",
    "D_ITEMS.csv",
    "D_LABITEMS.csv",
    "DATETIMEEVENTS.csv",
    "DRGCODES.csv",
    "ICUSTAYS.csv",
    "INPUTEVENTS_CV.csv",
    "INPUTEVENTS_MV.csv",
    "MICROBIOLOGYEVENTS.csv",
    "OUTPUTEVENTS.csv",
    "PRESCRIPTIONS.csv",
    "PROCEDUREEVENTS_MV.csv",
    "PROCEDURES_ICD.csv",
    "SERVICES.csv",
    "TRANSFERS.csv",
]

# Rows expected to be zero for specific files in the demo release — a
# mismatch here is expected behavior, not a download failure.
KNOWN_EMPTY_FILES = {
    "NOTEEVENTS.csv": (
        "NOTEEVENTS.csv has 0 data rows — this is expected for the MIMIC-III "
        "demo. PhysioNet strips discharge-summary text from the demo release; "
        "it's only present in the full credentialed MIMIC-III database."
    ),
}


def get_auth() -> requests.auth.HTTPBasicAuth | None:
    if os.environ.get("PHYSIONET_USER") and not os.environ.get("PHYSIONET_PASS"):
        log.warning(
            "PHYSIONET_USER is set but PHYSIONET_PASS is not; ignoring both and proceeding unauthenticated"
        )
        return None
    if os.environ.get("PHYSIONET_PASS") and not os.environ.get("PHYSIONET_USER"):
        log.warning(
            "PHYSIONET_PASS is set but PHYSIONET_USER is not; ignoring both and proceeding unauthenticated"
        )
        return None
    if os.environ.get("PHYSIONET_USER") and os.environ.get("PHYSIONET_PASS"):
        return requests.auth.HTTPBasicAuth(
            os.environ["PHYSIONET_USER"], os.environ["PHYSIONET_PASS"]
        )
    try:
        auth = netrc.netrc().authenticators("physionet.org")
        if auth:
            return requests.auth.HTTPBasicAuth(auth[0], auth[2])
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    log.info("no PhysioNet credentials configured; proceeding unauthenticated")
    return None


def download_file(session: requests.Session, base_url: str, filename: str, out_dir: Path) -> bool:
    """Download one file into out_dir, skipping if already present. Returns True on success."""
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip  %s (already present)", filename)
        return True

    url = base_url.rstrip("/") + "/" + filename
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        part.rename(dest)
        log.info("done  %s", filename)
        return True
    except requests.RequestException as e:
        log.error("fail  %s (%s)", filename, e)
        part.unlink(missing_ok=True)
        return False


def validate_downloads(out_dir: Path, required_files: list[str]) -> bool:
    """Check required files exist and log row counts. Returns True if all required files are usable."""
    ok = True
    for filename in required_files:
        path = out_dir / filename
        if not path.exists():
            log.error("missing required file: %s", filename)
            ok = False
            continue

        row_count = len(pd.read_csv(path))
        if filename in KNOWN_EMPTY_FILES:
            if row_count == 0:
                log.warning(KNOWN_EMPTY_FILES[filename])
            else:
                log.info("%-20s %d rows (note: expected 0 in prior demo releases)", filename, row_count)
        else:
            log.info("%-20s %d rows", filename, row_count)

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-dir", default=Path(__file__).parent / "raw", type=Path)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    auth = get_auth()
    session = requests.Session()
    if auth:
        session.auth = auth

    all_ok = True
    for filename in args.files:
        if not download_file(session, args.base_url, filename, args.out_dir):
            all_ok = False

    if not validate_downloads(args.out_dir, REQUIRED_FILES):
        all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
