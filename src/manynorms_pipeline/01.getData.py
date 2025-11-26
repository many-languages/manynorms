# libraries
import math, requests, pandas as pd, re
import textwrap
import numpy as np
from pathlib import Path

# base url for open alex search
base_url = (
    "https://api.openalex.org/works?"
    "filter=title_and_abstract.search:lexical+database+OR+lexical+norms+OR+linguistic+database+OR+linguistic+norms,"
    "publication_year:2018-2025,"
    "type:types/article|types/dataset|types/preprint|types/supplementary-materials|types/report|types/book-chapter"
    "&sort=relevance_score:desc"
    "&per_page=200"     # bump page size to reduce calls
)

# get abstract into tokens for api call
def decode_abstract(inv):
    if not isinstance(inv, dict) or not inv: return None
    pos2tok = {p:t for t,ps in inv.items() for p in ps}
    txt = " ".join(pos2tok.get(i,"") for i in range(max(pos2tok)+1))
    txt = re.sub(r"\s+([,.!?;:])", r"\1", txt)
    return re.sub(r"\s{2,}", " ", txt).strip() or None

# probe for total
probe = requests.get(base_url + "&page=1", timeout=30)
probe.raise_for_status()
meta = probe.json()["meta"]
total, per_page = meta["count"], meta["per_page"]
pages = math.ceil(total / per_page)
print(f"total={total}, per_page={per_page}, pages={pages}")

# get the data from open alex
rows = []
for p in range(1, pages+1):
    r = requests.get(base_url + f"&page={p}", timeout=60)
    r.raise_for_status()
    for w in r.json().get("results", []):
        rows.append({
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "authors": "; ".join(
                name
                for name in (
                    a.get("author", {}).get("display_name")
                    for a in w.get("authorships", [])
                )
                if name
            ),
            "abstract": decode_abstract(w.get("abstract_inverted_index")),
            # OpenAlex doesn’t store author-entered keywords; concepts are the closest proxy
            "keywords": [c["display_name"] for c in w.get("concepts", [])],
            "openalex_id": w.get("id"),
            "is_oa": (w.get("open_access") or {}).get("is_oa"),
            "cited_by": w.get("cited_by_count", 0),
        })
    print(f"page {p}/{pages}… collected {len(rows)}", end="\r")

# put into data frame
df = pd.DataFrame(rows)

# put together the data we need for the next step 
def summarize_abstracts(df: pd.DataFrame, n_show: int = 5):
    # treat empty strings/whitespace as missing
    has_abs = df["abstract"].astype("string").str.strip().ne("").fillna(False)

    total = len(df)
    with_abs = int(has_abs.sum())
    without_abs = total - with_abs
    pct = (with_abs / total * 100) if total else 0.0

    print(f"Total rows: {total}")
    print(f"With abstract: {with_abs} ({pct:.1f}%)")
    print(f"Missing abstract: {without_abs}")

    if without_abs:
        # show a few examples that are missing
        missing = df.loc[~has_abs, ["title", "year", "doi", "venue", "openalex_id"]].head(n_show)
        print("\nExamples missing abstracts:")
        for _, r in missing.iterrows():
            print("•", r["year"], "|", (r["title"] or "")[:120].rstrip(), "|", r["venue"] or "", "| DOI:", r["doi"] or "—")

    return has_abs

# Run the summary on your df - set show to 0 for this pipeline
has_abs_mask = summarize_abstracts(df, n_show=0)

# try to get missing abstracts
# ---- polite headers (some APIs appreciate a contact) ----
CONTACT_EMAIL = "ebuchanan@harrisburgu.edu"  # set yours
HEADERS = {"Accept": "application/json", "User-Agent": f"LAB-abstract-enricher ({CONTACT_EMAIL})"}

def _clean_doi(doi: str) -> str:
    if not doi: return ""
    doi = doi.strip()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)

def safe_request(url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"Other error: {str(e)}"

def fetch_crossref_abstract(doi):
    doi = _clean_doi(doi)
    url = f"https://api.crossref.org/works/{doi}"
    js, err = safe_request(url, headers=HEADERS)
    if err: return None, "ERROR", err
    abs_ = (js.get("message") or {}).get("abstract")
    if abs_:
        # strip tags
        abs_ = re.sub(r"<[^>]+>", "", abs_)
        return abs_.strip(), "Crossref", None
    return None, "MISSING", None

def fetch_europepmc_abstract(doi):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": f"DOI:{doi}", "format": "json", "pageSize": 1}
    js, err = safe_request(url, params=params, headers=HEADERS)
    if err: return None, "ERROR", err
    res = js.get("resultList", {}).get("result", [])
    if res and res[0].get("abstractText"):
        return res[0]["abstractText"], "EuropePMC", None
    return None, "MISSING", None

def get_abstract_by_doi(doi):
    doi = _clean_doi(doi)
    if not doi: return None, "MISSING", None
    # try Crossref then Europe PMC
    for fetcher in (fetch_crossref_abstract, fetch_europepmc_abstract):
        abs_, status, err = fetcher(doi)
        if status == "ERROR":  # API error
            return None, status, err
        if status != "MISSING":  # success
            return abs_, status, None
    return None, "MISSING", None

def enrich_missing_abstracts(df, doi_col="doi", abs_col="abstract", sleep=0.3):
    """
    For rows where df[abs_col] is empty, try to fetch an abstract by DOI.
    Prints status for each attempt; only writes into df[abs_col] on success.
    Expects get_abstract_by_doi() -> (abstract, status, err).
    """
    import time
    import pandas as pd

    if abs_col not in df.columns:
        df[abs_col] = None

    mask_missing = df[abs_col].isna() | (df[abs_col].astype(str).str.strip() == "")
    idxs = df.index[mask_missing].tolist()

    for i in idxs:
        doi = str(df.at[i, doi_col] or "").strip()
        if not doi:
            print(f"[Row {i}] No DOI, skipping.")
            continue

        abs_, status, err = get_abstract_by_doi(doi)

        if status == "ERROR":
            print(f"[Row {i}] DOI {doi}: API error -> {err}")
        elif status == "MISSING":
            print(f"[Row {i}] DOI {doi}: No abstract found.")
        else:
            print(f"[Row {i}] DOI {doi}: Abstract found via {status}.")
            df.at[i, abs_col] = abs_

        time.sleep(sleep)  # be polite to APIs

    return df

# apply function to current df
df = enrich_missing_abstracts(df)

# write out final data to a consistent location in the repo
project_root = Path(__file__).resolve().parents[2]
out_dir = project_root / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "new_data_to_classify.csv"

df.to_csv(out_path, index=False)