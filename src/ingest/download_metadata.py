"""Phase 1, step 1: download the arXiv metadata snapshot and filter to ML papers.

Pulls the Cornell arXiv metadata dataset from Kaggle, streams the (~4GB) JSON-lines
snapshot, keeps only papers in our ML categories, and writes a clean, normalized
JSON-lines subset to `data/arxiv_ml.jsonl`.

Prerequisites:
    pip install kaggle           (in the project venv)
    ~/.kaggle/kaggle.json        (Kaggle API token, chmod 600)

Usage:
    python -m src.ingest.download_metadata               # full filtered subset
    python -m src.ingest.download_metadata --limit 50000 # first N matches (slice)
    python -m src.ingest.download_metadata --min-year 2018
    python -m src.ingest.download_metadata --skip-download  # reuse existing raw file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import settings

KAGGLE_DATASET = "Cornell-University/arxiv"
RAW_DIR = Path("data/raw")
RAW_SNAPSHOT = RAW_DIR / "arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = Path("data/arxiv_ml.jsonl")


def download_snapshot() -> None:
    """Download + unzip the Kaggle snapshot into data/raw (idempotent)."""
    if RAW_SNAPSHOT.exists():
        print(f"Raw snapshot already present: {RAW_SNAPSHOT} (skipping download)")
        return

    # Import lazily so --skip-download works without the kaggle package installed.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as exc:
        # KaggleApi authenticates on import; a missing token raises here.
        raise SystemExit(
            "Kaggle auth failed. Place your token at ~/.kaggle/kaggle.json "
            "(Kaggle -> Settings -> Create New Token).\n"
            f"Underlying error: {exc}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} to {RAW_DIR} (this is ~4GB, may take a while)...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
    if not RAW_SNAPSHOT.exists():
        raise SystemExit(
            f"Download finished but {RAW_SNAPSHOT} not found. "
            f"Contents of {RAW_DIR}: {[p.name for p in RAW_DIR.iterdir()]}"
        )
    print("Download complete.")


def _paper_year(record: dict) -> int | None:
    """Best-effort publication year from update_date or the earliest version."""
    date = record.get("update_date") or ""
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    versions = record.get("versions") or []
    if versions:
        created = versions[0].get("created", "")
        # Format like "Mon, 2 Apr 2007 19:18:42 GMT" -> grab the year token.
        for token in created.split():
            if token.isdigit() and len(token) == 4:
                return int(token)
    return None


def filter_snapshot(limit: int | None, min_year: int | None) -> tuple[int, int]:
    """Stream the raw snapshot, keep ML papers, write normalized JSONL.

    Returns (kept, scanned).
    """
    categories = settings.ml_categories
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    scanned = 0
    with RAW_SNAPSHOT.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,} | kept {kept:,}")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = (rec.get("categories") or "").split()
            if not any(c in categories for c in cats):
                continue

            year = _paper_year(rec)
            if min_year is not None and (year is None or year < min_year):
                continue

            out = {
                "arxiv_id": rec.get("id", ""),
                "title": " ".join((rec.get("title") or "").split()),
                "abstract": " ".join((rec.get("abstract") or "").split()),
                "categories": cats,
                "authors": rec.get("authors", ""),
                "year": year,
                "update_date": rec.get("update_date", ""),
            }
            if not out["arxiv_id"] or not out["abstract"]:
                continue

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1
            if limit is not None and kept >= limit:
                print(f"  reached --limit {limit:,}, stopping.")
                break

    return kept, scanned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="keep at most N matching papers (for the validation slice)")
    parser.add_argument("--min-year", type=int, default=None,
                        help="drop papers older than this year")
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse an already-downloaded raw snapshot")
    args = parser.parse_args()

    if not args.skip_download:
        download_snapshot()
    elif not RAW_SNAPSHOT.exists():
        raise SystemExit(f"--skip-download set but {RAW_SNAPSHOT} not found.")

    print(f"Filtering to categories: {sorted(settings.ml_categories)}")
    kept, scanned = filter_snapshot(args.limit, args.min_year)
    print(f"\nDone. Kept {kept:,} of {scanned:,} scanned -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
