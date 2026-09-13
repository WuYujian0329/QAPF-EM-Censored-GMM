"""Validate the public archive layout without third-party dependencies."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    ".gitignore",
    "CITATION.cff",
    "README.md",
    "requirements.txt",
    "run_demo.py",
    "run_censored_demo.py",
    "data/thesis_experiment_datasets.xlsx",
    "docs/right_censoring.md",
    "src/qapf_em/censored.py",
    "tests/test_censored.py",
)


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Archive is missing required files: {', '.join(missing)}")
    workbook = ROOT / "data" / "thesis_experiment_datasets.xlsx"
    if not zipfile.is_zipfile(workbook):
        raise ValueError("The packaged workbook is not a valid .xlsx archive")
    print("Archive validation passed: release files, censored module, and dataset workbook are present.")


if __name__ == "__main__":
    main()
