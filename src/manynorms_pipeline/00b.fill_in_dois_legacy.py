

"""00b.fill_in_dois_legacy

One-time maintenance script to try to fill in missing DOIs for legacy
articles in `data/processed/articles_master.csv`.

This is **not** meant to be part of the recurring pipeline. Run it locally
as needed, inspect the log, and commit the updated master table if the
results look good.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
from typing import Optional

import pandas as pd
import requests
from urllib.parse import urlencode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Return the git repo root if possible, otherwise the current directory."""

    try:
        root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
        return Path(root)
    except Exception:
        return Path(os.getcwd()).resolve()


def canonicalize_doi(doi: str | None) -> Optional[str]:
    """Normalize a DOI string.

    - Treat None/NaN safely
    - Strip common URL prefixes
    - Lowercase
    - Return None for empty strings
    """

    if doi is None:
        return None

    s = str(doi).strip()
    if not s:
        return None

    # strip URL prefixes
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)

    s = s.lower().strip()
    return s or None


def normalize_title(title: str | None) -> str:
    """Normalize titles for comparison: lowercase, strip punctuation & collapse spaces."""

    if not isinstance(title, str):
        return ""

    t = title.lower().strip()
    # remove punctuation
    t = re.sub(r"[^\w\s]", "", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def get_year_from_crossref_item(item: dict) -> Optional[int]:
    """Try to extract a publication year from a Crossref work item."""

    def year_from_date_parts(obj: dict | None) -> Optional[int]:
        if not obj:
            return None
        parts = obj.get("date-parts") or []
        if not parts or not parts[0]:
            return None
        y = parts[0][0]
        try:
            return int(y)
        except Exception:
            return None

    msg = item or {}
    for key in ("published-print", "published-online", "created", "issued"):
        y = year_from_date_parts(msg.get(key))
        if y is not None:
            return y
    return None


def query_crossref_for_doi(title: str, year: Optional[int] = None) -> Optional[str]:
    """Query Crossref for a single DOI given a title (and optional year).

    We are deliberately conservative:
    - Fetch the top few results
    - Normalize titles and require an exact match (ignoring case/punctuation)
    - If year is available, require it to match within +/- 1 year
    - If we find exactly one matching DOI, return it; otherwise return None.
    """

    if not title:
        return None

    params: dict[str, str] = {
        "query.bibliographic": title,
        "rows": "5",
    }
    if year is not None:
        # narrow the search window, but don't rely exclusively on it
        params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"

    url = "https://api.crossref.org/works?" + urlencode(params)

    try:
        resp = requests.get(url, timeout=15)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    items = data.get("message", {}).get("items", []) or []
    if not items:
        return None

    target_title_norm = normalize_title(title)

    matches: list[str] = []

    for item in items:
        cr_titles = item.get("title") or []
        if not cr_titles:
            continue
        cr_title = cr_titles[0]
        cr_title_norm = normalize_title(cr_title)

        if not cr_title_norm:
            continue

        if cr_title_norm != target_title_norm:
            continue

        # title matches; optionally check year window
        cr_year = get_year_from_crossref_item(item)
        if year is not None and cr_year is not None:
            if abs(cr_year - year) > 1:
                # too far apart; skip
                continue

        doi = canonicalize_doi(item.get("DOI"))
        if doi:
            matches.append(doi)

    # Only accept if we have a single clear match
    if len(matches) == 1:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def fill_missing_dois_in_legacy() -> Path:
    """Fill missing DOIs for legacy rows in the master table.

    - Reads `data/processed/articles_master.csv`
    - Restricts to rows with source == "legacy_lab_table" and missing DOI
    - Uses Crossref to find DOIs, but only fills when there is a single
      high-confidence match
    - Writes the updated master table in-place and also writes a small log
      of updated rows.
    """

    repo_root = get_repo_root()
    master_path = repo_root / "data" / "processed" / "articles_master.csv"

    if not master_path.exists():
        raise FileNotFoundError(f"Master table not found at {master_path}")

    print(f"Reading master table from {master_path}")
    df = pd.read_csv(master_path)

    # Ensure expected columns exist
    for col in ("title", "doi", "source"):
        if col not in df.columns:
            raise ValueError(f"Master table is missing required column: {col!r}")

    # Work on a copy
    df["doi"] = df["doi"].astype("string")
    df["title"] = df["title"].astype("string")

    mask_missing = df["doi"].isna() | (df["doi"].str.strip() == "")
    mask_legacy = df["source"] == "legacy_lab_table"
    todo_mask = mask_missing & mask_legacy

    to_fix = df[todo_mask].copy()
    print(f"Found {len(to_fix)} legacy rows with missing DOIs")

    if to_fix.empty:
        print("Nothing to do; no legacy rows with missing DOIs.")
        return master_path

    updated_indices: list[int] = []

    for idx, row in to_fix.iterrows():
        title = row.get("title", "")
        year_val = row.get("year", None)

        try:
            year = int(year_val) if pd.notna(year_val) else None
        except Exception:
            year = None

        print("\n----------------------------------------")
        print(f"Row index: {idx}")
        print(f"Title: {title!r}")
        print(f"Year:  {year}")

        doi = query_crossref_for_doi(title=title, year=year)

        if doi:
            print(f" → Found DOI: {doi}")
            df.at[idx, "doi"] = doi
            updated_indices.append(idx)
        else:
            print(" → No confident DOI match found; leaving as missing")

    # Write updated master
    df.to_csv(master_path, index=False)
    print("\n========================================")
    print(f"Finished. Updated {len(updated_indices)} rows with new DOIs.")
    print(f"Wrote updated master table to {master_path}")

    # Also write a log of updated rows, if any
    if updated_indices:
        log_path = master_path.with_name("articles_master_doi_updates_legacy.csv")
        df.loc[updated_indices].to_csv(log_path, index=False)
        print(f"Wrote DOI update log to {log_path}")

    return master_path


if __name__ == "__main__":
    fill_missing_dois_in_legacy()