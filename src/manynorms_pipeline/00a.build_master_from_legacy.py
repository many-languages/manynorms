# src/manynorms_pipeline/00.build_master_from_legacy.py

from pathlib import Path
import subprocess
import os
import pandas as pd

# find the repo root 
def get_repo_root() -> Path:
    """
    Try to find the git repo root. Fallback to current working directory.
    Works in scripts, notebooks, and GitHub Actions.
    """
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
        ).decode().strip()
        return Path(root)
    except Exception:
        return Path(os.getcwd()).resolve()

# fix up dois
def canonicalize_doi(s: pd.Series) -> pd.Series:
    """
    Normalize DOIs:
      - treat None/NaN safely
      - strip URL prefixes like 'https://doi.org/'
      - lowercase
      - turn empty strings into NaN
    """
    s = s.astype("string").str.strip()

    # remove common URL prefixes
    s = s.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)

    # lowercase and replace empty with <NA>
    s = s.str.lower()
    s = s.replace({"": pd.NA})

    return s

# build the new source of truth file 
def build_master_from_legacy() -> Path:
    repo_root = get_repo_root()

    legacy_path = repo_root / "data" / "raw" / "legacy" / "both_lab_table.csv"
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Legacy file not found at {legacy_path}. "
            "Make sure you put both_lab_table.csv there."
        )

    print(f"Reading legacy file from {legacy_path}")
    df = pd.read_csv(legacy_path)

    # Basic sanity check: ensure expected columns exist
    required_cols = ["DOI", "TITLE", "JOURNAL", "YEAR", "code"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Legacy file is missing required columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    # Build the master table
    master = pd.DataFrame()

    master["doi"] = canonicalize_doi(df["DOI"])
    master["title"] = df["TITLE"].astype("string").str.strip()
    master["journal"] = df["JOURNAL"].astype("string").str.strip()
    master["year"] = df["YEAR"]

    # Map legacy Yes/No codes to normalized decisions
    code = df["code"].astype("string").str.strip().str.lower()
    decision_map = {"yes": "yes", "no": "no"}
    master["decision"] = code.map(decision_map)

    # Everything from this file is considered "decided" by the legacy process
    master["source"] = "legacy_lab_table"
    master["decision_source"] = "legacy_lab_table"

    # Optional: timestamp as ISO date
    master["last_updated"] = pd.Timestamp.today().date().isoformat()

    # Optional: deduplicate on doi+title
    before = len(master)
    master = master.drop_duplicates(subset=["doi", "title"], keep="first")
    after = len(master)

    print(f"Rows in legacy file: {before}")
    print(f"Rows after dedup on (doi, title): {after}")

    # Where to save
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "articles_master.csv"

    master.to_csv(out_path, index=False)
    print(f"Wrote master table to {out_path}")

    # Some quick stats
    print("\nDecision counts:")
    print(master["decision"].value_counts(dropna=False))

    return out_path

# run the thing 
if __name__ == "__main__":
    build_master_from_legacy()