"""Ingest Craigslist used-car listings (Kaggle: austinreese/craigslist-carstrucks-data).

Manual download: Kaggle requires login, so the zip is downloaded by hand into
data/raw/kaggle/source/archive.zip. This script reads vehicles.csv straight
from the zip, keeps the columns we need, and saves raw Parquet to
data/raw/kaggle/listings.parquet.

No cleaning happens here (fake prices, free-text models, duplicates are all
handled in dbt staging). This is a one-time snapshot, not part of the weekly job.

Run from the repo root:
    python ingest/kaggle_listings.py
"""

import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data" / "raw" / "kaggle" / "source" / "archive.zip"
CSV_NAME = "vehicles.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "kaggle" / "listings.parquet"

# The source has 41 columns; most of its size is free text (descriptions,
# URLs, image lists). We keep only model inputs and fields needed for cleaning.
# Address fields and lat/long are dropped deliberately (seller privacy).
COLUMNS = [
    # identity / join key
    "id",                    # Craigslist posting id
    "VIN",                   # for dedup in dbt; never shown in the app
    "year",
    "manufacturer",
    "model",                 # free text, e.g. "explorer st"
    # target
    "price",
    # features
    "odometer",
    "condition",
    "cylinders",             # text, e.g. "6 cylinders"
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",                  # body type, e.g. "SUV"
    "paint_color",
    "state",
    # data-quality signals
    "seller_type",
    "odometer_broken",
    "odometer_rolled_over",
    # time
    "posting_date",
    "updated_date",
]


def load(zip_path: Path) -> pd.DataFrame:
    """Read vehicles.csv from the zip into a DataFrame with only COLUMNS."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path.relative_to(REPO_ROOT)} not found. Download it from "
            "https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data "
            "and place the zip there (don't unzip)."
        )
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(CSV_NAME) as f:
            # usecols: skip the heavy text columns at read time, and fail
            # loudly if any expected column is missing.
            # price as float64: some listings have joke prices (e.g. 1e24)
            # that overflow int64. float64 keeps them (dbt filters them) and
            # still stores every realistic price exactly (up to 2**53).
            df = pd.read_csv(
                f, usecols=COLUMNS, dtype={"price": "float64"}, low_memory=False
            )

    df["ingested_at"] = datetime.now(timezone.utc)
    return df[COLUMNS + ["ingested_at"]]  # stable column order


def save(df: pd.DataFrame, path: Path) -> None:
    """Write the DataFrame to Parquet, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def main() -> None:
    df = load(SOURCE_ZIP)
    save(df, OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(
        f"Saved {len(df):,} rows x {df.shape[1]} cols ({size_mb:.0f} MB) "
        f"to {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
